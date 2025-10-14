# LLaVA-1.6-Vicuna-7B Inference System

이 프로젝트는 LLaVA-1.6-Vicuna-7B 모델을 사용한 효율적인 이미지-텍스트 inference 시스템입니다.

## 프로젝트 구조

```
├── config.py              # 설정 파일
├── model_loader.py         # 모델 로딩 관리
├── image_processor.py      # 이미지 전처리
├── inference_engine.py     # 메인 inference 엔진
├── main.py                # 명령줄 인터페이스
├── example_usage.py       # 사용 예제
├── requirements.txt       # 의존성 패키지
└── README.md             # 이 파일
```

## 설치

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. CUDA 환경 확인 (GPU 사용 시):
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 사용법

### 1. 단일 이미지 inference

```bash
python main.py single --image path/to/image.jpg --question "이 이미지에서 무엇을 볼 수 있나요?"
```

URL 이미지 사용:
```bash
python main.py single --image "https://example.com/image.jpg" --question "이 이미지를 설명해주세요."
```

시스템 프롬프트 추가:
```bash
python main.py single --image image.jpg --question "질문" --system-prompt "당신은 도움이 되는 AI 어시스턴트입니다."
```

결과를 JSON 파일로 저장:
```bash
python main.py single --image image.jpg --question "질문" --output result.json
```

특정 GPU 사용:
```bash
python main.py single --gpu 0 --image image.jpg --question "질문"
```

### 2. 배치 inference

배치 데이터 파일 (JSON 형식):
```json
[
  {
    "image": "path/to/image1.jpg",
    "question": "첫 번째 질문"
  },
  {
    "image": "https://example.com/image2.jpg",
    "question": "두 번째 질문"
  }
]
```

배치 실행:
```bash
python main.py batch --batch-file batch_data.json --output results.json
```

### 3. 대화형 모드

```bash
python main.py interactive
```

대화형 모드 명령어:
- `/load <image_path>` - 이미지 로드
- `/quit` - 종료
- `/help` - 도움말

### 4. 모델 정보 확인

```bash
python main.py info
```

## Python API 사용

### 기본 사용법

```python
from inference_engine import LLaVAInferenceEngine

# 엔진 초기화
engine = LLaVAInferenceEngine()

# 모델 로드
engine.load_model()

# inference 수행
result = engine.generate_response(
    image="path/to/image.jpg",
    question="이 이미지에서 무엇을 볼 수 있나요?"
)

print(result["response"])

# 모델 언로드
engine.unload_model()
```

### 커스텀 설정

```python
from inference_engine import LLaVAInferenceEngine
from config import Config

class CustomConfig(Config):
    GENERATION_CONFIG = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
    }

engine = LLaVAInferenceEngine(CustomConfig())
```

### 배치 처리

```python
batch_data = [
    {"image": "image1.jpg", "question": "질문 1"},
    {"image": "image2.jpg", "question": "질문 2"}
]

results = engine.batch_generate_responses(batch_data)
```

## GPU 설정

### 방법 1: 명령줄에서 설정
```bash
# 0번 GPU 사용
python main.py single --gpu 0 --image image.jpg --question "질문"

# 여러 GPU 사용 (0, 1번)
python main.py single --gpu "0,1" --image image.jpg --question "질문"
```

### 방법 2: 환경변수로 설정
```bash
# 터미널에서 실행 전 설정
export CUDA_VISIBLE_DEVICES=0
python main.py single --image image.jpg --question "질문"

# 또는 한 줄로
CUDA_VISIBLE_DEVICES=0 python main.py single --image image.jpg --question "질문"
```

### 방법 3: 코드에서 설정
```python
from config import Config
from inference_engine import LLaVAInferenceEngine

# GPU 0번 설정
Config.set_cuda_device("0")

# 엔진 초기화 및 사용
engine = LLaVAInferenceEngine()
engine.load_model()
```

### 방법 4: Config 클래스 상속
```python
class GPU0Config(Config):
    CUDA_VISIBLE_DEVICES = "0"
    DEVICE = "cuda:0"

engine = LLaVAInferenceEngine(GPU0Config())
```

### GPU 정보 확인
```bash
python main.py info --gpu 0
```

또는 코드에서:
```python
from config import Config
device_info = Config.get_device_info()
print(device_info)
```

## 설정 옵션

`config.py`에서 다음 설정을 변경할 수 있습니다:

- `MODEL_NAME`: 사용할 모델 이름
- `DEVICE`: 사용할 디바이스 ("cuda", "cpu", "auto", "cuda:0")
- `CUDA_VISIBLE_DEVICES`: 사용할 GPU ID(s) ("0", "0,1" 등)
- `TORCH_DTYPE`: 모델 데이터 타입 ("float16", "bfloat16", "float32")
- `GENERATION_CONFIG`: 텍스트 생성 파라미터
- `IMAGE_SIZE`: 이미지 크기 (기본값: 336x336)

## 예제 실행

```bash
python example_usage.py
```

이 스크립트는 다음 예제들을 실행합니다:
1. 단일 이미지 inference
2. 배치 inference
3. 커스텀 설정 사용
4. 에러 핸들링

## 시스템 요구사항

- Python 3.8+
- CUDA 11.0+ (GPU 사용 시)
- 최소 8GB RAM
- GPU 사용 시 최소 8GB VRAM 권장

## 주요 기능

- **모듈화된 구조**: 각 기능이 별도 파일로 분리되어 유지보수 용이
- **유연한 이미지 입력**: 로컬 파일, URL, numpy 배열 지원
- **배치 처리**: 여러 이미지를 효율적으로 처리
- **대화형 모드**: 실시간 이미지 분석 및 질의응답
- **커스터마이징**: 설정 파일을 통한 쉬운 파라미터 조정
- **에러 핸들링**: robust한 에러 처리 및 로깅
- **메모리 관리**: 모델 로딩/언로딩을 통한 메모리 효율성

## 문제 해결

### CUDA 메모리 부족
```python
# config.py에서 다음 설정 변경
TORCH_DTYPE = "float16"  # 또는 "bfloat16"
```

### 모델 다운로드 실패
```python
# config.py에서 로컬 모델 경로 설정
MODEL_PATH = "/path/to/local/model"
```

### 이미지 로딩 에러
- 이미지 파일 경로 확인
- 인터넷 연결 확인 (URL 이미지의 경우)
- 지원되는 이미지 형식: JPG, PNG, BMP, TIFF

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
