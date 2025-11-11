# capture_adapter55.py
# (10,55) 시퀀스 캡쳐: 2.25초 동안 10프레임 (프레임당 0.225s)
# 키:
#   1 / NumPad1 / z  -> ㅠ
#   2 / NumPad2 / x  -> ㅅ
#   3 / NumPad3 / c  -> ㅘ
#   4 / NumPad4 / v  -> ㅙ
#   5 / NumPad5 / b  -> ㅝ
#   6 / NumPad6 / n  -> ㅞ
#   s  -> 캡쳐 시작
#   q  -> 종료

import os
import cv2
import time
import json
import numpy as np
import mediapipe as mp

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(ROOT_DIR, 'data', 'adapter55')
os.makedirs(SAVE_DIR, exist_ok=True)

SEQ_LEN = 10
FEAT_DIM = 55
FRAME_INTERVAL = 0.225
CAM_INDEX = 0

LABEL_MAP = {
    '1': 'ㅠ',
    '2': 'ㅅ',
    '3': 'ㅘ',
    '4': 'ㅙ',
    '5': 'ㅝ',
    '6': 'ㅞ',
}

# VK_NUMPAD codes
NUMPAD = {
    0x61: '1',  # NumPad1
    0x62: '2',  # NumPad2
    0x63: '3',  # NumPad3
    0x64: '4',  # NumPad4
    0x65: '5',  # NumPad5
    0x66: '6',  # NumPad6
}

RIGHT_HAND_IDS = list(range(21))
PAIR_TIP_PIP = [(4,2),(8,6),(12,10),(16,14),(20,18)]
EXTRA_PAIRS   = [(5,9),(9,13),(13,17),(1,5),(2,5),(0,5),(0,9),(0,17)]

def _safe_norm(a, b):
    return np.linalg.norm(a - b)

def extract_55_from_holistic(holistic_result, image_w, image_h):
    hand_lms = holistic_result.right_hand_landmarks
    if hand_lms is None:
        hand_lms = holistic_result.left_hand_landmarks
    if hand_lms is None:
        return np.zeros(FEAT_DIM, dtype=np.float32)

    rhl = hand_lms.landmark
    pts = []
    for i in RIGHT_HAND_IDS:
        x = rhl[i].x * image_w
        y = rhl[i].y * image_h
        pts.append([x, y])
    pts = np.array(pts, dtype=np.float32)

    wrist = pts[0].copy()
    pts_centered = pts - wrist
    ref = _safe_norm(pts[0], pts[9])
    if ref < 1e-6:
        ref = 1.0
    pts_norm = pts_centered / ref

    feat_xy = pts_norm.flatten()
    d_tip_pip = [_safe_norm(pts_norm[a], pts_norm[b]) for a,b in PAIR_TIP_PIP]
    d_extra   = [_safe_norm(pts_norm[a], pts_norm[b]) for a,b in EXTRA_PAIRS]

    feat = np.concatenate([feat_xy, np.array(d_tip_pip + d_extra, dtype=np.float32)], axis=0)
    if feat.shape[0] != FEAT_DIM:
        feat = np.pad(feat, (0, FEAT_DIM - feat.shape[0]), constant_values=0.0)
    return feat.astype(np.float32)

def main():
    for key in LABEL_MAP.keys():
        os.makedirs(os.path.join(SAVE_DIR, key), exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print('웹캠을 열 수 없습니다.')
        return

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_styles   = mp.solutions.drawing_styles

    current_label = None
    seq_buffer = []

    meta_path = os.path.join(SAVE_DIR, 'capture_meta.jsonl')
    meta_f = open(meta_path, 'a', encoding='utf-8')

    hints = ' | '.join([f'{k}={v}' for k,v in LABEL_MAP.items()])

    with mp_holistic.Holistic(
        model_complexity=1,
        refine_face_landmarks=False,
        smooth_landmarks=True,
        enable_segmentation=False
    ) as holistic:
        print('준비 완료')
        print(f'{hints}  |  s=캡쳐  q=종료')
        print('예비키: z/x/c/v/b/n  |  NumPad1~6 지원')
        print('현재 라벨: None')

        saved_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print('프레임을 읽을 수 없습니다.')
                break

            h, w = frame.shape[:2]
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            vis = frame.copy()
            hand_lms = result.right_hand_landmarks if result.right_hand_landmarks else result.left_hand_landmarks
            if hand_lms:
                mp_drawing.draw_landmarks(
                    vis, hand_lms, mp.solutions.hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            info1 = f'Label: {current_label if current_label else "None"} ({hints})'
            info2 = f's=CAPTURE {SEQ_LEN} | saved={saved_count}'
            cv2.putText(vis, info1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(vis, info2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            if len(seq_buffer) > 0:
                cv2.putText(vis, f'Capturing {len(seq_buffer)}/{SEQ_LEN}', (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2, cv2.LINE_AA)

            cv2.imshow('adapter55 capture', vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('종료')
                break

            # 덱스: 숫자키/넘패드/예비키 매핑
            if key in (ord('1'), ord('z')) or key in NUMPAD and key == 0x61:
                current_label = '1'
                print('라벨=1 (ㅠ)')
            elif key in (ord('2'), ord('x')) or key in NUMPAD and key == 0x62:
                current_label = '2'
                print('라벨=2 (ㅅ)')
            elif key in (ord('3'), ord('c')) or key in NUMPAD and key == 0x63:
                current_label = '3'
                print('라벨=3 (ㅘ)')
            elif key in (ord('4'), ord('v')) or key in NUMPAD and key == 0x64:
                current_label = '4'
                print('라벨=4 (ㅙ)')
            elif key in (ord('5'), ord('b')) or key in NUMPAD and key == 0x65:
                current_label = '5'
                print('라벨=5 (ㅝ)')
            elif key in (ord('6'), ord('n')) or key in NUMPAD and key == 0x66:
                current_label = '6'
                print('라벨=6 (ㅞ)')

            elif key == ord('s'):
                if current_label is None:
                    print(f'라벨 먼저 선택: {hints}')
                    continue

                seq_buffer = []
                need = SEQ_LEN
                print(f'{need} 프레임을 {FRAME_INTERVAL:.3f}s 간격으로 캡쳐합니다...')

                while len(seq_buffer) < need:
                    ok2, frm = cap.read()
                    if not ok2:
                        print('프레임 읽기 실패')
                        break

                    frm = cv2.flip(frm, 1)
                    rh, rw = frm.shape[:2]
                    rgb2 = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                    res2 = holistic.process(rgb2)

                    feat = extract_55_from_holistic(res2, rw, rh)
                    seq_buffer.append(feat)

                    show = frm.copy()
                    hand_lms2 = res2.right_hand_landmarks if res2.right_hand_landmarks else res2.left_hand_landmarks
                    if hand_lms2:
                        mp_drawing.draw_landmarks(
                            show, hand_lms2, mp.solutions.hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )
                    cv2.putText(show, f'Capturing {len(seq_buffer)}/{need}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2, cv2.LINE_AA)
                    cv2.imshow('adapter55 capture', show)

                    time.sleep(FRAME_INTERVAL)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        seq_buffer = []
                        print('캡쳐 중단')
                        break

                if len(seq_buffer) == need:
                    seq_arr = np.stack(seq_buffer, axis=0).astype(np.float32)  # (10,55)
                    t = int(time.time() * 1000)
                    lbl_dir = os.path.join(SAVE_DIR, current_label)
                    os.makedirs(lbl_dir, exist_ok=True)

                    fname = f'{t}_len{SEQ_LEN}x{FEAT_DIM}.npy'
                    fpath = os.path.join(lbl_dir, fname)
                    np.save(fpath, seq_arr)

                    meta = {
                        'ts_ms': t,
                        'label_key': current_label,
                        'label_str': LABEL_MAP.get(current_label, ''),
                        'shape': list(seq_arr.shape),
                        'path': fpath.replace(ROOT_DIR + os.sep, ''),
                        'note': 'adapter55 capture (mirror on, auto-hand, wrist-centered, distance-normalized)'
                    }
                    meta_f.write(json.dumps(meta, ensure_ascii=False) + '\n')
                    meta_f.flush()

                    saved_count += 1
                    print(f'SAVED: {fpath}  shape={seq_arr.shape}')

    meta_f.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
