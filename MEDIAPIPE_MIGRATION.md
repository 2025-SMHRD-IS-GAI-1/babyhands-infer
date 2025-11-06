# MediaPipe + LSTM 모델 마이그레이션 가이드

## 개요

YOLO 기반 모델에서 **MediaPipe + LSTM** 기반 모델로 전환했습니다.
[Sign_Language_Translation 프로젝트](https://github.com/JunYong-Choi/Sign_Language_Translation)를 참고하여 구현했습니다.

## 주요 변경사항

### 1. 모델 구조 변경
- **기존**: YOLO v3 (이미지 기반 객체 감지)
- **신규**: MediaPipe Hand Landmarks + LSTM (시퀀스 기반 분류)

### 2. 인식 가능한 클래스
- **전체 31개 한글 자음/모음** 인식
  - 자음: 19개 (ㄱ, ㄲ, ㄴ, ㄷ, ...)
  - 모음: 21개 (ㅏ, ㅐ, ㅑ, ...)

### 3. 작동 방식
1. MediaPipe로 손 랜드마크(21개 포인트) 추출
2. 30프레임 시퀀스 데이터 생성
3. LSTM 모델로 예측

## 파일 구조

```
python/
├── mediapipe_model_loader.py    # MediaPipe + LSTM 모델 로더
├── mediapipe_predictor.py       # 예측기
├── main.py                      # FastAPI 서버 (수정됨)
├── korean_jamo.txt              # 31개 자음/모음 목록
└── model.tflite                 # LSTM 모델 파일 (학습 필요)
```

## 모델 파일 준비

### 방법 1: 기존 모델 사용 (없는 경우)
1. [Sign_Language_Translation](https://github.com/JunYong-Choi/Sign_Language_Translation) 저장소에서 모델 다운로드
2. `models/` 폴더에서 `.tflite` 파일 찾기
3. `python/model.tflite`로 복사

### 방법 2: 새로 학습
Sign_Language_Translation 프로젝트의 학습 스크립트를 사용하여 모델을 학습할 수 있습니다.

## 설치

```bash
pip install -r requirements.txt
```

새로 추가된 패키지:
- `mediapipe==0.10.9`
- `tensorflow==2.15.0`
- `scikit-learn==1.3.2`

## 사용 방법

### 1. 모델 없이 테스트 (MediaPipe만)
```bash
python main.py
```
- MediaPipe는 작동하며 hand landmarks를 추출할 수 있습니다
- LSTM 모델이 없어도 서버는 시작됩니다
- 예측은 하지 않지만 landmarks 추출은 가능

### 2. 모델과 함께 사용
`model.tflite` 파일을 `python/` 폴더에 넣으면 완전한 예측이 가능합니다.

## API 엔드포인트

모든 기존 API는 동일하게 작동합니다:

- `GET /api/consonants` - 학습된 자음 리스트
- `GET /api/vowels` - 학습된 모음 리스트
- `GET /api/all-consonants` - 전체 자음 (31개 중 학습된 것)
- `GET /api/all-vowels` - 전체 모음 (31개 중 학습된 것)
- `GET /api/learned-only` - 학습된 것만
- `WS /ws` - WebSocket 실시간 예측

## 모델 학습 방법

1. 데이터 수집
   - 31개 자음/모음 각각에 대한 영상 촬영
   - 여러 사람의 데이터 수집 권장

2. 데이터 전처리
   - MediaPipe로 hand landmarks 추출
   - 시퀀스 데이터 생성 (30프레임)

3. 모델 학습
   - LSTM 모델 구성
   - TensorFlow/Keras로 학습
   - TensorFlow Lite로 변환

자세한 내용은 [Sign_Language_Translation 프로젝트](https://github.com/JunYong-Choi/Sign_Language_Translation)를 참고하세요.

## 장점

1. **전체 자음/모음 지원**: 31개 모두 인식 가능
2. **실시간 처리**: MediaPipe는 빠른 hand tracking
3. **시퀀스 기반**: 손 움직임의 시간적 정보 활용
4. **경량화**: TensorFlow Lite 모델은 작고 빠름

## 참고 자료

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [Sign_Language_Translation 프로젝트](https://github.com/JunYong-Choi/Sign_Language_Translation)
- [TensorFlow Lite](https://www.tensorflow.org/lite)

