# Neural River Operation Python Library (in progress)

## 프로젝트 소개 (Project Introduction)

neuralriverops는 인공지능(LSTM)을 이용하여 댐방류시 댐하류 하천의 기상상황을 고려하여 하천의 수위를 예측합니다.
이 프로그램에는 MLOps(MLFlow)와 연계되어 효율적인 모델과 데이터의 관리가 가능합니다.

## 주요 기능 (Key Features)

- **수위 예측 (Water Level Prediction)**: LSTM 모델을 사용하여 하천 수위를 예측
- **MLOps 파이프라인 (MLOps Pipeline)**: MLFlow를 이용하여 다중 수위관측지점의 모델 학습, 평가, 배포를 위한 자동화된 파이프라인

## 시스템 구조 (System Architecture)

이 저장소는 다음과 같은 주요 구성 요소로 이루어져 있습니다:

- **mlops/**: 모델 학습, 예측, 평가를 위한 MLOps 파이프라인
  - `basin_config.py`: 유역 구성 관리
  - `retrain.py`: 모델 재학습 모듈
  - `retrain_eval.py`: 재학습 모델의 Plotting
  - `model.py`: 모델 레지스트리 및 관리
  - `forecast.py`: 수위 예측 모듈
  - `forecast_eval.py`: 예측측 모델의 Plotting


- **data/**: 유역 구성 및 데이터 저장
  - `keum_river_basin/`: 금강 유역 데이터
  - `basins_config.json`: 유역 구성 정보

## 시작하기 (Getting Started)

### 필수 조건 (Prerequisites)

- Python 3.8+
- MLflow
- PyTorch
- NeuralHydrology (동일 repo의 ZIP파일을 설치)

### 설치 (Installation)

```bash
# 저장소 복제
git clone https://github.com/realtime-dam-operation-using-AI/neuralriverops.git
cd neuralriverops

# 필요한 패키지 설치
# (설치 명령어는 프로젝트에 맞게 수정 필요)
```

## 사용 방법 (Usage)

Makefile과 동일 레벨의 폴더에서

```bash
# MLFlow 서버실행 (브라우저에서 http://localhost:5000/ 를 실행하면 MLFlow가 실행됨됨)
make start-floodai

# 재학습 실행 예시 (용담댐하류, normal LSTM 적용시)
make retrain basin=1

# 재학습 실행 예시 (용담댐하류, autoregressive LSTM 적용시)
make retrain basin=1 AR=1

# 예측 실행 예시 (용담댐하류, normal LSTM 적용시)
make forecast basin=1

# 예측 실행 예시 (용담댐하류, autoregressive LSTM 적용시)
make forecast basin=1 AR=1

# MLFlow 서버종료료
make stop-floodai

