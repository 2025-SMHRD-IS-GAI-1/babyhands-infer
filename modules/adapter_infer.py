# modules/adapter_infer.py
import os
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# ===== 설정 =====
SEQ_LEN = 10
FEAT_DIM = 55
ADAPTER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'adapter')
ADAPTER_TFLITE = os.path.join(ADAPTER_DIR, 'adapter_5cls_10x55.tflite')  # 파일명 확인
ADAPTER_LABELS_TXT = os.path.join(ADAPTER_DIR, 'adapter_labels.txt')

# 기본 임계/연속
ADAPTER_CONF_THRESH = 0.80   # 기본값(보조 장치)
CONSEC_REQUIRE = 3           # 기본 연속 요구 프레임 수
HARD_THRESHOLD = 0.90        # 스위칭 확신 임계
MARGIN_OVER_BASE = 0.10      # base 대비 마진(선택: base_conf 제공 시)

# ★ 자모 세트
CONSONANTS = {'ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'}
ADAPTER_TARGETS = {'ㅠ','ㅘ','ㅙ','ㅝ','ㅞ'}  # 어댑터 대상 모음

# ★ 라벨별 강화 조건(난이도 높은 모음은 더 빡세게)
LABEL_THRESH = {
    'default': 0.92,
    'ㅠ': 0.93,
    'ㅘ': 0.95,
    'ㅙ': 0.97,
    'ㅝ': 0.95,
    'ㅞ': 0.97,
}
LABEL_CONSEC = {
    'default': 3,
    'ㅙ': 4,
    'ㅞ': 4,
}

# ===== MediaPipe =====
RIGHT_HAND_IDS = list(range(21))
PAIR_TIP_PIP = [(4,2),(8,6),(12,10),(16,14),(20,18)]
EXTRA_PAIRS   = [(5,9),(9,13),(13,17),(1,5),(2,5),(0,5),(0,9),(0,17)]

def _safe_norm(a, b):
    return float(np.linalg.norm(a - b))

def extract_55(result, w, h):
    """오른손 우선, 없으면 왼손 사용. 손 없으면 0벡터."""
    hand_lms = result.right_hand_landmarks if result.right_hand_landmarks else result.left_hand_landmarks
    if hand_lms is None:
        return np.zeros(FEAT_DIM, dtype=np.float32)

    pts = []
    for i in RIGHT_HAND_IDS:
        x = hand_lms.landmark[i].x * w
        y = hand_lms.landmark[i].y * h
        pts.append([x, y])
    pts = np.array(pts, dtype=np.float32)

    # 기준 정규화: 손목 중심, 손목-중지 MCP 거리
    wrist = pts[0].copy()
    pts_centered = pts - wrist
    ref = _safe_norm(pts[0], pts[9]) or 1.0
    pts_norm = pts_centered / ref

    # 42(=21*2) + 5 + 8 = 55
    feat_xy = pts_norm.flatten()
    d_tip   = [_safe_norm(pts_norm[a], pts_norm[b]) for a,b in PAIR_TIP_PIP]
    d_extra = [_safe_norm(pts_norm[a], pts_norm[b]) for a,b in EXTRA_PAIRS]
    feat = np.concatenate([feat_xy, np.array(d_tip + d_extra, dtype=np.float32)], axis=0)

    if feat.shape[0] != FEAT_DIM:
        if feat.shape[0] < FEAT_DIM:
            feat = np.pad(feat, (0, FEAT_DIM - feat.shape[0]), constant_values=0.0)
        else:
            feat = feat[:FEAT_DIM]
    return feat.astype(np.float32)

def load_labels(txt_path):
    if not os.path.exists(txt_path):
        return None
    with open(txt_path, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    return names

class AdapterEngine:
    """(10,55) 시퀀스 누적 → 어댑터 TFLite 추론 → 스위칭 판단"""
    def __init__(self, tflite_path=ADAPTER_TFLITE, labels_txt=ADAPTER_LABELS_TXT, conf_thresh=ADAPTER_CONF_THRESH):
        import tensorflow as tf
        self.labels = load_labels(labels_txt)

        # TFLite 로드
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_detail  = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]

        # 입력 shape 로그
        in_shape = self.input_detail['shape']  # [1,10,55] 기대
        self.expected_seq = int(in_shape[1])
        self.expected_dim = int(in_shape[2])
        print(f"[Adapter] loaded: {tflite_path}")
        print(f"[Adapter] input shape = (batch,{self.expected_seq},{self.expected_dim})")
        if self.expected_seq != SEQ_LEN or self.expected_dim != FEAT_DIM:
            raise RuntimeError(
                f"Adapter expects (10,{self.expected_dim}) but extractor makes (10,{FEAT_DIM}). "
                f"→ 잘못된 TFLite 파일 경로/이름 확인."
            )

        self.seq = deque(maxlen=SEQ_LEN)
        self.conf_thresh = conf_thresh
        self.last_preds = deque(maxlen=max(CONSEC_REQUIRE, max(LABEL_CONSEC.values(), default=3)))

        # MediaPipe 준비
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=1,
            refine_face_landmarks=False,
            smooth_landmarks=True,
            enable_segmentation=False
        )

    def close(self):
        if hasattr(self, 'holistic'):
            self.holistic.close()

    def update_with_frame(self, bgr_frame):
        """BGR frame 1장 입력 → (10,55) 누적. 준비되면 adapter pred 반환."""
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self.holistic.process(rgb)

        feat55 = extract_55(result, w, h)
        self.seq.append(feat55)

        if len(self.seq) < SEQ_LEN:
            return None

        x = np.array(self.seq, dtype=np.float32)[None, ...]  # (1,10,55)
        self.interpreter.set_tensor(self.input_detail['index'], x)
        self.interpreter.invoke()
        prob = self.interpreter.get_tensor(self.output_detail['index'])[0]  # (C,)

        top_idx = int(np.argmax(prob))
        top_p   = float(prob[top_idx])
        name = self.labels[top_idx] if self.labels and top_idx < len(self.labels) else str(top_idx)

        # 연속성 버퍼 업데이트
        self.last_preds.append(name)

        return {'index': top_idx, 'name': name, 'conf': top_p, 'probs': prob.tolist()}

    def _is_consecutive(self, name: str, need: int) -> bool:
        """최근 need개가 모두 같은 name인지"""
        if len(self.last_preds) < need:
            return False
        return all(t == name for t in list(self.last_preds)[-need:])

    def should_switch(self, adapter_pred, base_name, base_conf: float | None = None) -> bool:
        """
        보수 스위칭 규칙:
          - base가 자음이면 스위칭 금지 (자음 보호)
          - 어댑터 타깃(ㅠ/ㅘ/ㅙ/ㅝ/ㅞ)만 대상
          - 라벨별 연속 프레임/임계 적용
          - (선택) base_conf 제공 시 마진 적용
        """
        if adapter_pred is None:
            return False

        y = adapter_pred['name']
        p = adapter_pred['conf']

        # 자음에서 들어온 경우는 절대 스위칭 금지 (ㅅ→ㅠ 오인치 차단)
        if base_name in CONSONANTS:
            return False

        # 어댑터 타깃만 허용
        if y not in ADAPTER_TARGETS:
            return False

        # 라벨별 요구치
        need_consec = LABEL_CONSEC.get(y, LABEL_CONSEC.get('default', CONSEC_REQUIRE))
        need_p = max(HARD_THRESHOLD, LABEL_THRESH.get(y, LABEL_THRESH.get('default', 0.92)))

        # 연속성 + 확신
        if not self._is_consecutive(y, need_consec):
            return False
        if p < need_p:
            return False

        # 베이스 대비 마진(옵션)
        if base_conf is not None and p < (float(base_conf) + MARGIN_OVER_BASE):
            return False

        return True
