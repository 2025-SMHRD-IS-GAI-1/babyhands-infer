"""
원본 Sign_Language_Translation 프로젝트의 holistic_module.py
MediaPipe Holistic 사용
"""
import cv2
import mediapipe as mp

class HolisticDetector():
    def __init__(self,
               static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               refine_face_landmarks=False,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.refine_face_landmarks = refine_face_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHolistic = mp.solutions.holistic
        self.holistics = self.mpHolistic.Holistic(
            self.static_image_mode,
            self.model_complexity,
            self.smooth_landmarks,
            self.enable_segmentation,
            self.smooth_segmentation,
            self.refine_face_landmarks,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHolistic(self, img, draw=False):
        """MediaPipe Holistic으로 손 랜드마크 추출"""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.holistics.process(imgRGB)
        
        if draw and self.results.right_hand_landmarks:
            annotated_image = img.copy()
            self.mpDraw.draw_landmarks(
                annotated_image, 
                self.results.right_hand_landmarks, 
                self.mpHolistic.HAND_CONNECTIONS
            )
            return annotated_image
        
        return img

    def findRighthandLandmark(self, img, draw=False):
        """오른손 랜드마크 리스트 반환"""
        self.right_hand_lmList = []
        
        if self.results.right_hand_landmarks:
            for id, lm in enumerate(self.results.right_hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x*w), int(lm.y*h), lm.z
                self.right_hand_lmList.append([id, cx, cy, cz])
        
        return self.right_hand_lmList, self.results.right_hand_landmarks

