from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import json
import asyncio
# MediaPipe + LSTM 모델 사용
from mediapipe_model_loader import MediaPipeModelLoader
from mediapipe_predictor import SignLanguagePredictor

app = FastAPI()

# CORS 설정 (JSP 서버와 통신하기 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MediaPipe + LSTM 모델 로더 초기화
model_loader = MediaPipeModelLoader()
predictor = SignLanguagePredictor(model_loader)

@app.get("/")
async def root():
    return {"message": "수어 예측 서버가 실행 중입니다."}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model_loader.is_loaded()}

@app.get("/api/consonants")
async def get_consonants():
    """
    학습된 자음 클래스 리스트 반환
    """
    consonants = predictor.get_consonant_list()
    return {
        "success": True,
        "count": len(consonants),
        "consonants": consonants
    }

@app.get("/api/vowels")
async def get_vowels():
    """
    학습된 모음 클래스 리스트 반환
    """
    vowels = predictor.get_vowel_list()
    return {
        "success": True,
        "count": len(vowels),
        "vowels": vowels
    }

@app.get("/api/classes")
async def get_all_classes():
    """
    전체 클래스 매핑 정보 반환
    """
    mappings = predictor.get_all_mappings()
    consonants = predictor.get_consonant_list()
    vowels = predictor.get_vowel_list()
    
    return {
        "success": True,
        "total_classes": len(mappings),
        "consonants": consonants,
        "vowels": vowels,
        "all_mappings": mappings
    }

@app.get("/api/all-consonants")
async def get_all_consonants():
    """
    전체 자음 목록 반환 (학습 여부 포함)
    학습된 자음: learned=True
    학습되지 않은 자음: learned=False
    """
    all_consonants = predictor.get_all_consonants()
    learned_count = sum(1 for c in all_consonants if c['learned'])
    
    return {
        "success": True,
        "total": len(all_consonants),
        "learned": learned_count,
        "not_learned": len(all_consonants) - learned_count,
        "consonants": all_consonants
    }

@app.get("/api/all-vowels")
async def get_all_vowels():
    """
    전체 모음 목록 반환 (학습 여부 포함)
    학습된 모음: learned=True
    학습되지 않은 모음: learned=False
    """
    all_vowels = predictor.get_all_vowels()
    learned_count = sum(1 for v in all_vowels if v['learned'])
    
    return {
        "success": True,
        "total": len(all_vowels),
        "learned": learned_count,
        "not_learned": len(all_vowels) - learned_count,
        "vowels": all_vowels
    }

@app.get("/api/learned-only")
async def get_learned_only():
    """
    학습된 자음/모음만 반환 (학습되지 않은 것은 제외)
    """
    learned = predictor.get_learned_only()
    
    return {
        "success": True,
        "consonant_count": len(learned['consonants']),
        "vowel_count": len(learned['vowels']),
        **learned
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("클라이언트 연결됨")
    
    # 모델이 로드되지 않았을 경우 경고
    if not model_loader.is_loaded():
        await websocket.send_json({
            "prediction": "",
            "accuracy": 0,
            "error": "모델이 로드되지 않았습니다. 서버 로그를 확인하세요.",
            "warning": True
        })
    
    try:
        while True:
            # 클라이언트로부터 이미지 데이터 수신
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                if message.get("type") == "image":
                    # Base64 인코딩된 이미지 데이터
                    image_data = message.get("data")
                    
                    if image_data:
                        # Base64 디코딩
                        try:
                            # "data:image/jpeg;base64," 부분 제거
                            if "," in image_data:
                                image_bytes = base64.b64decode(image_data.split(",")[1])
                            else:
                                image_bytes = base64.b64decode(image_data)
                            
                            nparr = np.frombuffer(image_bytes, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if frame is not None and frame.size > 0:
                                # 예측 수행
                                prediction, accuracy = predictor.predict(frame)
                                
                                # 자음/모음으로 변환
                                korean_char = convert_to_korean(prediction)
                                
                                # 응답 전송
                                response = {
                                    "prediction": korean_char,
                                    "accuracy": int(accuracy * 100)
                                }
                                
                                await websocket.send_json(response)
                            else:
                                # 이미지 디코딩 실패
                                await websocket.send_json({
                                    "prediction": "",
                                    "accuracy": 0,
                                    "error": "이미지 디코딩 실패"
                                })
                        except Exception as decode_error:
                            print(f"이미지 디코딩 오류: {decode_error}")
                            await websocket.send_json({
                                "prediction": "",
                                "accuracy": 0,
                                "error": f"이미지 처리 오류: {str(decode_error)}"
                            })
                    else:
                        await websocket.send_json({
                            "prediction": "",
                            "accuracy": 0,
                            "error": "이미지 데이터가 없습니다"
                        })
                            
            except json.JSONDecodeError as json_error:
                # JSON 파싱 오류
                print(f"JSON 파싱 오류: {json_error}")
                await websocket.send_json({
                    "prediction": "",
                    "accuracy": 0,
                    "error": "잘못된 JSON 형식"
                })
            except Exception as e:
                print(f"예측 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_json({
                    "prediction": "",
                    "accuracy": 0,
                    "error": str(e)
                })
                
    except WebSocketDisconnect:
        print("클라이언트 연결 해제됨")
    except Exception as e:
        print(f"WebSocket 오류: {e}")
        import traceback
        traceback.print_exc()

def convert_to_korean(prediction: str) -> str:
    # 모델이 이미 한글 자모(ㄱ~ㅎ, ㅏ~ㅟ)를 반환하므로 변환 불필요
    return prediction

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010, reload=False, factory=False)