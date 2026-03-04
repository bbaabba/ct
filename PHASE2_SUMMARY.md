# Phase 2: AI 모델 개발 완료 ✅

## 📊 구현된 컴포넌트

### 1. 데이터 수집 시스템

#### [data/collectors/historical_collector.py](data/collectors/historical_collector.py)

**주요 기능:**
- ✅ 바이낸스에서 역사적 OHLCV 데이터 수집
- ✅ CSV 파일로 저장 및 로드
- ✅ 다중 심볼 동시 수집
- ✅ 기존 데이터 업데이트 (증분 수집)
- ✅ Rate Limit 자동 관리

**사용 예제:**
```python
from data.collectors import HistoricalDataCollector
from data.binance_client import BinanceClient

# 클라이언트 생성
client = BinanceClient(testnet=True)
collector = HistoricalDataCollector(client)

# 최근 30일 데이터 수집
df = collector.get_recent_data(
    symbol='BTCUSDT',
    interval='1h',
    days=30
)

print(f"수집된 데이터: {len(df)}개")
print(df.head())
```

**주요 메서드:**
- `collect_klines()` - 특정 기간 데이터 수집
- `get_recent_data()` - 최근 N일 데이터 수집
- `collect_multiple_symbols()` - 여러 심볼 동시 수집
- `update_existing_data()` - 기존 CSV 업데이트

---

### 2. 특징 엔지니어링

#### [data/processors/feature_engineering.py](data/processors/feature_engineering.py)

**50개 이상 기술 지표 자동 계산:**

| 카테고리 | 지표 |
|---------|------|
| **이동평균** | SMA(7,25,50,99), EMA(7,12,26,50), WMA(14) |
| **모멘텀** | RSI(7,14,21), MACD, Stochastic, CCI, Williams %R, ROC, MOM |
| **변동성** | Bollinger Bands, ATR, NATR, True Range |
| **거래량** | OBV, AD, ADOSC, Volume SMA, Volume Ratio |
| **추세** | ADX, +DI, -DI, AROON, SAR |
| **가격 변화** | Returns, Log Returns, Price Changes, High/Low Ratio |
| **추가 특징** | Volatility, Price vs MA, MA Cross, 시간 특징 |

**핵심 파이프라인:**
```python
from data.processors import FeatureEngineering

# 전체 ML 데이터 준비 (원스톱)
ml_data = FeatureEngineering.prepare_ml_data(
    df=df,                      # 원본 OHLCV 데이터
    target_column='close',      # 예측 대상
    lookback=60,                # LSTM 입력 시퀀스 길이
    forecast_horizon=1,         # 예측 시점
    normalize=True,             # 정규화 여부
    train_size=0.7,             # 학습 데이터 비율
    val_size=0.15               # 검증 데이터 비율
)

# 반환 데이터:
# - X_train, y_train
# - X_val, y_val
# - X_test, y_test
# - scaler (정규화 객체)
# - feature_columns (특징 컬럼 리스트)
```

**주요 메서드:**
- `calculate_all_indicators()` - 모든 기술 지표 계산
- `create_sequences()` - LSTM용 시퀀스 생성
- `normalize_data()` - MinMax / Standard 정규화
- `prepare_ml_data()` - 전체 파이프라인 (원스톱)

---

### 3. LSTM 모델 아키텍처

#### [models/rnn/lstm_models.py](models/rnn/lstm_models.py)

**5가지 LSTM 모델 구현:**

#### 1) LSTMPricePredictor (가격 예측)
```python
from models.rnn import LSTMPricePredictor

model = LSTMPricePredictor(
    input_size=50,      # 특징 개수
    hidden_size=128,    # LSTM 은닉 유닛
    num_layers=2,       # LSTM 레이어 수
    dropout=0.2         # 드롭아웃 비율
)

# 입력: (batch_size, sequence_length, input_size)
# 출력: (batch_size, 1) - 예측 가격
```

#### 2) LSTMDirectionClassifier (방향 분류)
```python
from models.rnn import LSTMDirectionClassifier

model = LSTMDirectionClassifier(
    input_size=50,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    num_classes=3  # 상승/횡보/하락
)

# 출력: (batch_size, 3) - 클래스 로짓
```

#### 3) MultiTimeframeLSTM (다중 타임프레임)
```python
from models.rnn import MultiTimeframeLSTM

model = MultiTimeframeLSTM(
    input_size=50,
    hidden_size=128,
    dropout=0.2
)

# 입력: x_1h, x_4h, x_1d (각각 다른 타임프레임)
# 출력: (batch_size, 1) - 통합 예측
```

#### 4) BidirectionalLSTM (양방향)
- 과거와 미래 정보를 모두 활용
- 더 나은 문맥 이해

#### 5) AttentionLSTM (어텐션 메커니즘)
- 중요한 시점에 집중
- 어텐션 가중치 시각화 가능

---

### 4. 학습 파이프라인

#### [models/training/trainer.py](models/training/trainer.py)

**ModelTrainer 클래스:**
```python
from models.training import ModelTrainer

trainer = ModelTrainer(
    model=model,
    device='cuda'  # 또는 'cpu'
)

# 학습
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    lr=0.001,
    early_stopping_patience=10,
    save_path='models/saved/best_model.pth'
)

# 평가
metrics = trainer.evaluate(test_loader)
# -> {'loss': ..., 'mae': ..., 'rmse': ..., 'mape': ..., 'r2_score': ...}

# 예측
predictions = trainer.predict(X_test)

# 학습 이력 시각화
trainer.plot_training_history('training_history.png')
```

**주요 기능:**
- ✅ 자동 Early Stopping
- ✅ Learning Rate Scheduler (ReduceLROnPlateau)
- ✅ Gradient Clipping (폭발 방지)
- ✅ 모델 자동 저장 (최적 성능 시)
- ✅ 다양한 평가 메트릭 (MAE, RMSE, MAPE, R²)
- ✅ 학습 이력 시각화

---

### 5. 학습 데이터셋

#### [models/training/dataset.py](models/training/dataset.py)

**PyTorch Dataset 클래스:**
```python
from models.training import CryptoDataset
from torch.utils.data import DataLoader

# 데이터셋 생성
train_dataset = CryptoDataset(X_train, y_train)

# 데이터로더 생성
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
```

---

## 🚀 전체 학습 파이프라인 예제

### [train_model_example.py](train_model_example.py)

**실행 방법:**
```bash
python train_model_example.py
```

**파이프라인 단계:**
1. ✅ 바이낸스에서 BTCUSDT 1시간 봉 데이터 수집 (최근 180일)
2. ✅ 50개 이상 기술 지표 자동 계산
3. ✅ 데이터 정규화 및 시퀀스 생성
4. ✅ Train/Val/Test 분할 (70/15/15)
5. ✅ LSTM 모델 생성 및 학습
6. ✅ 최적 모델 자동 저장
7. ✅ 테스트 세트 평가
8. ✅ 학습 이력 그래프 저장

**예상 출력:**
```
======================================================================
LSTM 가격 예측 모델 학습
======================================================================

[ 1/6 ] 데이터 수집 시작...
✅ 데이터 수집 완료: 4320개

[ 2/6 ] 특징 엔지니어링...
기술 지표 계산 시작 (데이터: 4320개)
✅ 기술 지표 계산 완료: 85개 컬럼
시퀀스 생성 완료: X (4259, 60, 50), y (4259,)
데이터 분할 완료:
  Train: 2981 (70%)
  Val: 639 (15%)
  Test: 639 (15%)

[ 3/6 ] 데이터로더 생성...
✅ 데이터로더 생성 완료

[ 4/6 ] 모델 생성...
LSTMPricePredictor 초기화: input=50, hidden=128, layers=2
✅ 모델 생성 완료

[ 5/6 ] 모델 학습 시작...
Epoch [1/50] - Train Loss: 0.002345, Val Loss: 0.001876, LR: 0.001000
Epoch [2/50] - Train Loss: 0.001654, Val Loss: 0.001432, LR: 0.001000
...
✅ 최적 모델 저장 (Val Loss: 0.001234)
✅ 학습 완료! 최적 Val Loss: 0.001234

[ 6/6 ] 모델 평가...
테스트 평가 결과:
  Loss: 0.001289
  MAE: 0.035421
  RMSE: 0.045678
  MAPE: 0.12%
  R²: 0.9876

======================================================================
학습 결과 요약
======================================================================
데이터:
  심볼: BTCUSDT
  간격: 1h
  기간: 최근 180일
  Train: 2981, Val: 639, Test: 639

모델:
  타입: LSTM
  입력 크기: 50
  Hidden Size: 128
  Layers: 2

성능:
  MAE: 0.035421
  RMSE: 0.045678
  MAPE: 0.12%
  R²: 0.9876

모델 저장 위치: models/saved/lstm_btc_1h_best.pth
======================================================================
```

---

## 📈 성능 메트릭 설명

| 메트릭 | 설명 | 좋은 값 |
|--------|------|---------|
| **MAE** | Mean Absolute Error (평균 절대 오차) | 낮을수록 좋음 |
| **RMSE** | Root Mean Squared Error (평균 제곱근 오차) | 낮을수록 좋음 |
| **MAPE** | Mean Absolute Percentage Error (%) | 낮을수록 좋음 (<5% 우수) |
| **R²** | 결정 계수 (설명력) | 높을수록 좋음 (1.0에 가까울수록) |

---

## 🎯 다음 단계: Phase 3

Phase 3에서는 학습된 모델을 활용한 **거래 전략 시스템**을 구축합니다:

1. **기본 전략 인터페이스** (BaseStrategy)
2. **기술적 분석 전략**
   - Moving Average Crossover
   - RSI Mean Reversion
   - Bollinger Bands Strategy
3. **ML 기반 전략**
   - LSTM 예측 기반 거래
   - 신뢰도 기반 포지션 사이징
4. **리스크 관리**
   - Kelly Criterion
   - Stop-Loss / Take-Profit
   - 포지션 사이징

---

## 📝 주요 파일 정리

| 파일 | 설명 | 상태 |
|------|------|------|
| [data/collectors/historical_collector.py](data/collectors/historical_collector.py) | 데이터 수집기 | ✅ |
| [data/processors/feature_engineering.py](data/processors/feature_engineering.py) | 특징 엔지니어링 | ✅ |
| [models/rnn/lstm_models.py](models/rnn/lstm_models.py) | LSTM 모델 | ✅ |
| [models/training/dataset.py](models/training/dataset.py) | PyTorch Dataset | ✅ |
| [models/training/trainer.py](models/training/trainer.py) | 학습 트레이너 | ✅ |
| [train_model_example.py](train_model_example.py) | 학습 예제 | ✅ |

---

## 💡 사용 팁

### 1. 데이터 수집
```python
# 더 많은 데이터 수집 (1년)
df = collector.get_recent_data('BTCUSDT', '1h', days=365)

# 여러 심볼 동시 수집
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
results = collector.collect_multiple_symbols(symbols, '1h', '2023-01-01')
```

### 2. 하이퍼파라미터 튜닝
```python
# 더 깊은 모델
model = LSTMPricePredictor(
    input_size=50,
    hidden_size=256,  # 더 큰 hidden size
    num_layers=3,     # 더 많은 레이어
    dropout=0.3       # 더 강한 정규화
)

# 더 긴 lookback
ml_data = FeatureEngineering.prepare_ml_data(
    df=df,
    lookback=120,  # 60 → 120
    ...
)
```

### 3. 앙상블 예측
```python
# 여러 모델 학습 후 평균
predictions = []
for model_path in ['model1.pth', 'model2.pth', 'model3.pth']:
    trainer.load_model(model_path)
    pred = trainer.predict(X_test)
    predictions.append(pred)

# 앙상블 예측
ensemble_pred = np.mean(predictions, axis=0)
```

---

## 🎉 Phase 2 완료!

모든 AI 모델 개발 컴포넌트가 성공적으로 구현되었습니다!
