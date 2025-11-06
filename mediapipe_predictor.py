"""
MediaPipe + LSTM 기반 수어 예측기
Sign_Language_Translation 프로젝트 참고
"""
import cv2
import numpy as np
from mediapipe_model_loader import MediaPipeModelLoader

class SignLanguagePredictor:
    def __init__(self, model_loader: MediaPipeModelLoader):
        """
        수어 예측기 초기화
        
        Args:
            model_loader: MediaPipeModelLoader 인스턴스
        """
        self.model_loader = model_loader
        self.detector = model_loader.get_detector()
        self.classes = model_loader.get_classes()
        
        # 자음/모음 전체 목록 (실제 학습된 순서)
        # 원본 프로젝트: 14개 자음 + 17개 모음 = 31개
        self.consonants = [
            'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
        ]
        self.vowels = [
            'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
            'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ'
        ]
        
        # 자음/모음 분류
        self.consonant_classes = []
        self.vowel_classes = []
        self._categorize_classes()
        
        print(f"✅ MediaPipe + LSTM 모델 초기화 완료")
        print(f"   - 전체 클래스 수: {len(self.classes)}개 (31개 자음/모음)")
        print(f"   - 자음: {len(self.consonant_classes)}개")
        print(f"   - 모음: {len(self.vowel_classes)}개")
        print(f"   - LSTM 모델 로드: {'✅' if model_loader.is_loaded() else '❌ (MediaPipe만 사용)'}")
    
    def _categorize_classes(self):
        """클래스를 자음/모음으로 분류"""
        for char in self.classes:
            if char in self.consonants:
                self.consonant_classes.append({
                    'korean': char,
                    'learned': True
                })
            elif char in self.vowels:
                self.vowel_classes.append({
                    'korean': char,
                    'learned': True
                })
    
    def predict(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> tuple:
        """
        프레임에서 수어 예측
        
        Args:
            frame: 입력 이미지 프레임 (BGR)
            confidence_threshold: 신뢰도 임계값
            
        Returns:
            (prediction, confidence) 튜플
            prediction: 예측된 한글 자음/모음
            confidence: 신뢰도 (0.0 ~ 1.0)
        """
        # Hand landmarks 추출
        landmarks = self.model_loader.extract_landmarks(frame)
        
        if landmarks is None:
            return "", 0.0
        
        # LSTM 모델로 예측
        if self.model_loader.is_loaded():
            prediction, confidence = self.model_loader.predict(landmarks)
            
            if confidence >= confidence_threshold:
                return prediction, confidence
        
        return "", 0.0
    
    def get_consonant_list(self) -> list:
        """학습된 자음 클래스 리스트 반환"""
        return self.consonant_classes
    
    def get_vowel_list(self) -> list:
        """학습된 모음 클래스 리스트 반환"""
        return self.vowel_classes
    
    def get_all_consonants(self) -> list:
        """전체 자음 목록 반환 (학습 여부 포함)"""
        learned_koreans = {item['korean'] for item in self.consonant_classes}
        return [
            {'korean': cons, 'learned': cons in learned_koreans}
            for cons in self.consonants
        ]
    
    def get_all_vowels(self) -> list:
        """전체 모음 목록 반환 (학습 여부 포함)"""
        learned_koreans = {item['korean'] for item in self.vowel_classes}
        return [
            {'korean': vow, 'learned': vow in learned_koreans}
            for vow in self.vowels
        ]
    
    def get_all_mappings(self) -> dict:
        """전체 클래스 매핑 (자음/모음 -> 인덱스)"""
        return {char: idx for idx, char in enumerate(self.classes)}
    
    def get_learned_only(self) -> dict:
        """학습된 자음/모음만 반환"""
        return {
            'consonants': self.consonant_classes,
            'vowels': self.vowel_classes
        }

