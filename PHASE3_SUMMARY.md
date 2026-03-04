# Phase 3: 거래 전략 시스템 완료 ✅

## 📊 구현된 컴포넌트

### 1. 기본 전략 인터페이스

#### [strategies/base.py](strategies/base.py)

**핵심 클래스:**
- `BaseStrategy` - 모든 전략의 기본 클래스
- `Signal` - 거래 신호 Enum (BUY, SELL, HOLD)
- `TradeSignal` - 거래 신호 상세 정보

**주요 메서드:**
```python
from strategies.base import BaseStrategy, Signal, TradeSignal

class MyStrategy(BaseStrategy):
    def __init__(self, params=None):
        super().__init__(name='MyStrategy', params=params)

    def calculate_indicators(self, data):
        # 필요한 기술 지표 계산
        return data

    def generate_signal(self, data):
        # 거래 신호 생성
        return TradeSignal(...)
```

**주요 기능:**
- ✅ 데이터 유효성 검사
- ✅ 신뢰도 기반 포지션 크기 계산
- ✅ 전략 학습/초기화 인터페이스
- ✅ 메타데이터 관리

---

### 2. 기술적 분석 전략

#### 2-1. 이동평균 교차 전략

##### [strategies/technical/ma_crossover.py](strategies/technical/ma_crossover.py)

**전략 개요:**
- 골든크로스 (단기 MA > 장기 MA) → 매수
- 데드크로스 (단기 MA < 장기 MA) → 매도

**사용 예제:**
```python
from strategies.technical import MACrossoverStrategy

strategy = MACrossoverStrategy(params={
    'fast_period': 7,      # 단기 이동평균
    'slow_period': 25,     # 장기 이동평균
    'ma_type': 'EMA',      # SMA, EMA, WMA
    'min_confidence': 0.6  # 최소 신뢰도
})

# 신호 생성
signal = strategy.generate_signal(df)

print(f"신호: {signal.signal.name}")
print(f"신뢰도: {signal.confidence:.2%}")
print(f"이유: {signal.reason}")
```

**신뢰도 계산:**
- MA 차이의 강도 (정규화)
- 거래량 증가 여부
- 종합 점수: (MA 차이 강도 + 거래량 강도) / 2

---

#### 2-2. RSI 평균회귀 전략

##### [strategies/technical/rsi_strategy.py](strategies/technical/rsi_strategy.py)

**전략 개요:**
- RSI < 30 (과매도) → 매수
- RSI > 70 (과매수) → 매도

**사용 예제:**
```python
from strategies.technical import RSIStrategy

strategy = RSIStrategy(params={
    'rsi_period': 14,         # RSI 기간
    'oversold_level': 30,     # 과매도 기준
    'overbought_level': 70,   # 과매수 기준
    'min_confidence': 0.6
})

# 신호 생성
signal = strategy.generate_signal(df)
```

**신뢰도 계산:**
- 과매도/과매수 강도
- RSI 전환 방향 (상승/하락 전환 시 가중치)
- 거래량 증가 여부

---

#### 2-3. 볼린저 밴드 전략

##### [strategies/technical/bollinger_bands.py](strategies/technical/bollinger_bands.py)

**전략 개요:**
- 하단 밴드 터치 → 매수 (반등 기대)
- 상단 밴드 터치 → 매도 (조정 기대)

**사용 예제:**
```python
from strategies.technical import BollingerBandsStrategy

strategy = BollingerBandsStrategy(params={
    'period': 20,              # 볼린저 밴드 기간
    'std_dev': 2.0,            # 표준편차 배수
    'touch_threshold': 0.02,   # 밴드 터치 기준 (2%)
    'min_confidence': 0.6
})

# 신호 생성
signal = strategy.generate_signal(df)
```

**신뢰도 계산:**
- 밴드 내 가격 위치 (0 ~ 1)
- RSI 과매도/과매수 추가 가중치
- 거래량 증가 여부

---

### 3. ML 기반 전략

#### [strategies/ml_based/lstm_strategy.py](strategies/ml_based/lstm_strategy.py)

**전략 개요:**
- 학습된 LSTM 모델로 다음 가격 예측
- 예측 방향과 크기에 따라 거래 신호 생성

**사용 예제:**
```python
from strategies.ml_based import LSTMStrategy

strategy = LSTMStrategy(params={
    'model_path': 'models/saved/lstm_btc_1h_best.pth',
    'lookback': 60,           # LSTM 입력 시퀀스
    'threshold': 0.005,       # 가격 변화 기준 (0.5%)
    'min_confidence': 0.7
})

# 모델 로드 (자동)
signal = strategy.generate_signal(df)

print(f"현재 가격: {signal.price:.2f}")
print(f"예측 가격: {signal.metadata['predicted_price']:.2f}")
print(f"변화율: {signal.metadata['price_change_pct']:.2%}")
```

**주요 기능:**
- ✅ 자동 모델 로드
- ✅ 50개 이상 기술 지표 자동 계산
- ✅ 데이터 정규화 및 시퀀스 생성
- ✅ 예측 가격 기반 신호 생성

**신뢰도 계산:**
- 예측 변화율의 크기 (변화율이 클수록 높음)
- threshold의 5배 기준으로 정규화

---

### 4. 리스크 관리 시스템

#### [execution/risk_manager.py](execution/risk_manager.py)

**핵심 클래스:**
- `RiskManager` - 리스크 관리자
- `Position` - 포지션 정보
- `RiskLimits` - 리스크 제한 설정

**사용 예제:**
```python
from execution import RiskManager, RiskLimits

# 리스크 제한 설정
limits = RiskLimits(
    max_position_size_pct=0.2,    # 최대 포지션 20%
    max_daily_loss_pct=0.05,      # 일일 손실 제한 5%
    max_total_exposure_pct=0.8,   # 총 노출 80%
    stop_loss_pct=0.02,           # 스톱로스 2%
    take_profit_pct=0.04,         # 이익실현 4%
    kelly_fraction=0.25           # Kelly Criterion 비율
)

# 리스크 관리자 생성
risk_manager = RiskManager(
    initial_balance=10000.0,
    limits=limits
)

# 포지션 크기 계산
position_info = risk_manager.calculate_position_size(
    symbol='BTCUSDT',
    price=50000.0,
    confidence=0.8,
    win_rate=0.55,     # 예상 승률
    avg_win=0.02,      # 예상 평균 이익
    avg_loss=0.01      # 예상 평균 손실
)

print(f"포지션 크기: {position_info['position_size_usdt']:.2f} USDT")
print(f"수량: {position_info['quantity']:.6f}")
print(f"Stop-Loss: {position_info['stop_loss']:.2f}")
print(f"Take-Profit: {position_info['take_profit']:.2f}")
```

**Kelly Criterion:**
```
Kelly% = W - [(1-W) / R]

W: 승률
R: 평균 이익 / 평균 손실 비율

실제 적용: Kelly% × kelly_fraction × confidence
```

**포지션 관리:**
```python
# 포지션 추가
risk_manager.add_position(
    symbol='BTCUSDT',
    side='BUY',
    entry_price=50000.0,
    quantity=0.04,
    stop_loss=49000.0,
    take_profit=52000.0
)

# Stop-Loss / Take-Profit 확인
action = risk_manager.check_stop_loss_take_profit(
    symbol='BTCUSDT',
    current_price=49000.0
)

if action == 'STOP_LOSS':
    # 손절 실행
    pnl = risk_manager.close_position('BTCUSDT', 49000.0)
elif action == 'TAKE_PROFIT':
    # 익절 실행
    pnl = risk_manager.close_position('BTCUSDT', 52000.0)

# 현재 상태 확인
status = risk_manager.get_status()
print(f"현재 잔고: {status['current_balance']:.2f} USDT")
print(f"총 손익: {status['total_pnl']:.2f} USDT ({status['total_pnl_pct']:.2%})")
print(f"일일 손익: {status['daily_pnl']:.2f} USDT")
print(f"포지션 수: {status['num_positions']}")
```

**주요 기능:**
- ✅ Kelly Criterion 포지션 크기 계산
- ✅ 일일 손실 제한 (자동 거래 중단)
- ✅ 총 노출 관리
- ✅ Stop-Loss / Take-Profit 자동 확인
- ✅ 포지션 추적 및 손익 계산

---

## 🚀 전체 테스트 파이프라인

### [test_phase3.py](test_phase3.py)

**실행 방법:**
```bash
python test_phase3.py
```

**테스트 항목:**
1. ✅ MA Crossover 전략 테스트
2. ✅ RSI 전략 테스트
3. ✅ Bollinger Bands 전략 테스트
4. ✅ Risk Manager 테스트

**예상 출력:**
```
======================================================================
Phase 3: 거래 전략 시스템 테스트
======================================================================

======================================================================
[ 1/4 ] MA Crossover 전략 테스트
======================================================================
✅ 데이터 수집 완료: 168개
신호: BUY
신뢰도: 75.23%
가격: 50234.56
이유: 골든크로스 발생 (Fast MA: 50345.12 > Slow MA: 50123.45)

======================================================================
[ 2/4 ] RSI 전략 테스트
======================================================================
✅ 데이터 수집 완료: 168개
신호: HOLD
신뢰도: 0.00%
가격: 50234.56
이유: 중립 구간

======================================================================
[ 3/4 ] Bollinger Bands 전략 테스트
======================================================================
✅ 데이터 수집 완료: 168개
신호: SELL
신뢰도: 68.45%
가격: 50234.56
이유: 볼린저 상단 밴드 터치 (BB%: 98.23%)

======================================================================
[ 4/4 ] Risk Manager 테스트
======================================================================
포지션 정보:
  크기: 400.00 USDT
  수량: 0.008000
  Stop-Loss: 49000.00
  Take-Profit: 52000.00
  Kelly %: 4.00%

리스크 관리자 상태:
  현재 잔고: 10000.00 USDT
  총 손익: 0.00 USDT (0.00%)
  포지션 수: 1
  총 노출: 400.00 USDT (4.00%)

가격 49000.00에서 조치: STOP_LOSS
포지션 청산 손익: -8.00 USDT
최종 잔고: 9992.00 USDT

======================================================================
✅ Phase 3 테스트 완료!
======================================================================
```

---

## 📈 전략 비교표

| 전략 | 타입 | 장점 | 단점 | 적합한 시장 |
|------|------|------|------|------------|
| **MA Crossover** | 추세 추종 | 단순, 명확한 신호 | 횡보장 시 손실 | 강한 추세장 |
| **RSI** | 평균회귀 | 과매도/과매수 포착 | 강한 추세 시 조기 진입 | 횡보장 |
| **Bollinger Bands** | 평균회귀 | 변동성 고려 | 브레이크아웃 시 손실 | 횡보장 |
| **LSTM** | 머신러닝 | 복잡한 패턴 인식 | 데이터 의존적 | 모든 시장 |

---

## 💡 전략 조합 예제

### 1. 다중 전략 앙상블

```python
from strategies.technical import MACrossoverStrategy, RSIStrategy, BollingerBandsStrategy

# 전략 생성
strategies = [
    MACrossoverStrategy(),
    RSIStrategy(),
    BollingerBandsStrategy()
]

# 각 전략의 신호 수집
signals = []
for strategy in strategies:
    signal = strategy.generate_signal(df)
    signals.append(signal)

# 다수결 투표 (예: 2개 이상 BUY면 BUY)
buy_count = sum(1 for s in signals if s.signal == Signal.BUY)
sell_count = sum(1 for s in signals if s.signal == Signal.SELL)

if buy_count >= 2:
    final_signal = Signal.BUY
    avg_confidence = sum(s.confidence for s in signals if s.signal == Signal.BUY) / buy_count
elif sell_count >= 2:
    final_signal = Signal.SELL
    avg_confidence = sum(s.confidence for s in signals if s.signal == Signal.SELL) / sell_count
else:
    final_signal = Signal.HOLD
    avg_confidence = 0.0
```

### 2. 전략 가중 평균

```python
# 각 전략에 가중치 부여
weights = {
    'MA_Crossover': 0.3,
    'RSI': 0.3,
    'LSTM': 0.4
}

# 신호 스코어 계산
total_score = 0.0
total_confidence = 0.0

for strategy, weight in zip(strategies, weights.values()):
    signal = strategy.generate_signal(df)

    score = 0
    if signal.signal == Signal.BUY:
        score = 1
    elif signal.signal == Signal.SELL:
        score = -1

    total_score += score * weight * signal.confidence

# 최종 판단
if total_score > 0.3:
    final_signal = Signal.BUY
elif total_score < -0.3:
    final_signal = Signal.SELL
else:
    final_signal = Signal.HOLD
```

---

## 🎯 다음 단계: Phase 4

Phase 4에서는 **전략 스위칭 시스템**을 구축합니다:

1. **시장 상태 감지**
   - Regime Detection (추세/횡보/변동성)
   - 시장 구조 분석

2. **전략 성과 추적**
   - 실시간 성과 모니터링
   - 샤프 비율, 승률, 최대 낙폭 계산

3. **동적 전략 선택**
   - 시장 상태에 따른 전략 자동 선택
   - 성과 기반 가중치 조정

4. **메타 학습**
   - 전략 조합 최적화
   - 강화학습 기반 전략 할당

---

## 📝 주요 파일 정리

| 파일 | 설명 | 상태 |
|------|------|------|
| [strategies/base.py](strategies/base.py) | 전략 기본 인터페이스 | ✅ |
| [strategies/technical/ma_crossover.py](strategies/technical/ma_crossover.py) | MA 교차 전략 | ✅ |
| [strategies/technical/rsi_strategy.py](strategies/technical/rsi_strategy.py) | RSI 전략 | ✅ |
| [strategies/technical/bollinger_bands.py](strategies/technical/bollinger_bands.py) | 볼린저 밴드 전략 | ✅ |
| [strategies/ml_based/lstm_strategy.py](strategies/ml_based/lstm_strategy.py) | LSTM 전략 | ✅ |
| [execution/risk_manager.py](execution/risk_manager.py) | 리스크 관리자 | ✅ |
| [test_phase3.py](test_phase3.py) | Phase 3 테스트 | ✅ |

---

## 💡 사용 팁

### 1. 전략 파라미터 최적화

```python
# Grid Search 예제
from itertools import product

fast_periods = [5, 7, 10]
slow_periods = [20, 25, 30]

best_params = None
best_sharpe = -float('inf')

for fast, slow in product(fast_periods, slow_periods):
    if fast >= slow:
        continue

    strategy = MACrossoverStrategy(params={
        'fast_period': fast,
        'slow_period': slow
    })

    # 백테스트 실행 (Phase 4에서 구현 예정)
    # sharpe_ratio = backtest(strategy, data)

    # if sharpe_ratio > best_sharpe:
    #     best_sharpe = sharpe_ratio
    #     best_params = {'fast_period': fast, 'slow_period': slow}
```

### 2. 실시간 신호 모니터링

```python
import time

strategy = RSIStrategy()

while True:
    # 최신 데이터 가져오기
    df = collector.get_recent_data('BTCUSDT', '1h', days=7)

    # 신호 생성
    signal = strategy.generate_signal(df)

    if signal.signal != Signal.HOLD and signal.confidence > 0.7:
        print(f"⚠️ 알림: {signal.signal.name} 신호 발생!")
        print(f"   신뢰도: {signal.confidence:.2%}")
        print(f"   이유: {signal.reason}")

    # 1시간마다 확인
    time.sleep(3600)
```

### 3. 리스크 관리 통합

```python
# 전략 + 리스크 관리 통합 예제
strategy = LSTMStrategy()
risk_manager = RiskManager(initial_balance=10000.0)

# 신호 생성
signal = strategy.generate_signal(df)

if signal.signal == Signal.BUY and signal.confidence > 0.7:
    # 포지션 크기 계산
    position = risk_manager.calculate_position_size(
        symbol='BTCUSDT',
        price=signal.price,
        confidence=signal.confidence
    )

    if position['position_size_usdt'] > 0:
        # 주문 실행 (Phase 5에서 구현)
        print(f"매수 주문: {position['quantity']:.6f} @ {signal.price:.2f}")
        print(f"Stop-Loss: {position['stop_loss']:.2f}")
        print(f"Take-Profit: {position['take_profit']:.2f}")
```

---

## 🎉 Phase 3 완료!

모든 거래 전략 시스템이 성공적으로 구현되었습니다!

**구현 완료:**
- ✅ BaseStrategy 인터페이스
- ✅ 기술적 분석 전략 3종 (MA, RSI, BB)
- ✅ ML 기반 전략 (LSTM)
- ✅ 리스크 관리 시스템 (Kelly Criterion)
- ✅ 테스트 스크립트

**다음 단계:**
- Phase 4: 전략 스위칭 시스템 (시장 상태 감지 및 동적 전략 선택)
