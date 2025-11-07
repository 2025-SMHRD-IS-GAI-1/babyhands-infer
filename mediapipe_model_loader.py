"""
MediaPipe + LSTM 모델 로더
Sign_Language_Translation 프로젝트 참고
원본 프로젝트의 구조를 그대로 사용
"""
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from pathlib import Path
from typing import List, Optional
from modules.holistic_module import HolisticDetector
from modules.utils import Vector_Normalization

class MediaPipeModelLoader:
    """
    MediaPipe Holistic + LSTM 모델 로더
    31개 한글 자음/모음 인식
    """
    def __init__(self, model_path: str = None, classes_path: str = None):
        """
        모델 로더 초기화
        
        Args:
            model_path: TensorFlow Lite 모델 파일 경로 (.tflite)
            classes_path: 클래스 이름 파일 경로 (31개 자음/모음)
        """
        base_dir = Path(__file__).parent
        
        # 기본 모델 경로
        self.model_path = model_path or str(base_dir / "model.tflite")
        self.classes_path = classes_path or str(base_dir / "korean_jamo.txt")
        
        # MediaPipe Holistic 초기화 (원본 프로젝트와 동일)
        self.detector = HolisticDetector(min_detection_confidence=0.3)
        
        # LSTM 모델
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        # 31개 한글 자음/모음 클래스 (원본 프로젝트 순서)
        self.classes = self._load_classes()
        
        # 시퀀스 버퍼 (LSTM 입력용)
        self.sequence_buffer = []
        self.sequence_length = 10  # 원본 프로젝트는 10
        
        self._loaded = False
        
        # 모델 로드 시도
        try:
            self.load_model()
        except Exception as e:
            print(f" 모델 로드 실패 - MediaPipe는 사용 가능하지만 LSTM 모델 없음: {e}")
            print(" MediaPipe만으로도 hand landmarks 추출은 가능합니다.")
            self._loaded = False
    
    def _load_classes(self) -> List[str]:
        """31개 한글 자음/모음 클래스 로드 (원본 프로젝트 순서)"""
        # 원본 프로젝트의 순서대로
        default_classes = [
            'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
            'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
            'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ'
        ]
        
        # 파일에서 로드 시도
        if os.path.exists(self.classes_path):
            try:
                with open(self.classes_path, 'r', encoding='utf-8') as f:
                    classes = [line.strip() for line in f.readlines() if line.strip()]
                if len(classes) == 31:
                    return classes
            except Exception as e:
                print(f"클래스 파일 로드 실패, 기본 목록 사용: {e}")
        
        return default_classes
    
    def load_model(self):
        """TensorFlow Lite 모델 로드"""
        if self._loaded:
            return True
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        # TensorFlow Lite 인터프리터 로드
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        # 입력/출력 상세 정보
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self._loaded = True
        print(f" LSTM 모델 로드 완료: {self.model_path}")
        print(f"   클래스 수: {len(self.classes)}")
        return True
    
    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인"""
        return self._loaded
    
    def get_detector(self):
        """MediaPipe Holistic Detector 객체 반환"""
        return self.detector
    
    def get_classes(self) -> List[str]:
        """클래스 목록 반환"""
        return self.classes
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        MediaPipe Holistic을 사용하여 오른손 랜드마크 추출 및 벡터/각도 계산
        원본 프로젝트의 webcam_test_model_tflite.py와 동일한 방식
        
        Args:
            frame: BGR 이미지 프레임
            
        Returns:
            feature 벡터 (None if 손이 감지되지 않음)
        """
        # Holistic 처리
        self.detector.findHolistic(frame, draw=False)
        right_hand_lmList, right_hand_landmarks = self.detector.findRighthandLandmark(frame)
        
        if right_hand_landmarks is None:
            return None
        
        # 오른손 랜드마크를 joint 배열로 변환 (21개 포인트, x, y 좌표만)
        joint = np.zeros((21, 2))
        for j, lm in enumerate(right_hand_landmarks.landmark):
            joint[j] = [lm.x, lm.y]
        
        # 벡터 정규화 (원본 프로젝트와 동일)
        vector, angle_label = Vector_Normalization(joint)
        
        # feature 벡터 생성 (원본 프로젝트와 동일)
        d = np.concatenate([vector.flatten(), angle_label.flatten()])
        
        return d
    
    def predict(self, feature: np.ndarray) -> tuple:
        """
        LSTM 모델을 사용하여 예측
        
        Args:
            feature: feature 벡터 (벡터 + 각도)
            
        Returns:
            (prediction, confidence) 튜플
        """
        if not self._loaded or self.interpreter is None:
            return "", 0.0
        
        # 시퀀스 버퍼에 추가
        self.sequence_buffer.append(feature)
        
        # 시퀀스 길이 유지
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)
        
        # 충분한 프레임이 모이지 않았으면 예측 안 함
        if len(self.sequence_buffer) < self.sequence_length:
            return "", 0.0
        
        # 시퀀스 데이터 준비 (원본 프로젝트와 동일)
        input_data = np.expand_dims(
            np.array(self.sequence_buffer[-self.sequence_length:], dtype=np.float32), 
            axis=0
        )
        
        # 모델 입력
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # 예측 결과
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = output_data[0]  # (31,)
        
        # 가장 높은 신뢰도의 클래스
        class_id = int(np.argmax(predictions))
        confidence = float(predictions[class_id])
        
        if class_id < len(self.classes):
            return self.classes[class_id], confidence
        
        return "", 0.0
    
    def reset_sequence(self):
        """시퀀스 버퍼 초기화"""
        self.sequence_buffer = []
