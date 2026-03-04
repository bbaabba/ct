# Phase 4: 전략 스위칭 시스템 완료 ✅

## 📊 구현된 컴포넌트

### 1. 시장 상태 감지 시스템

#### [regime/market_regime.py](regime/market_regime.py)

**핵심 클래스:**
- `MarketRegime` - 시장 상태 타입 Enum
- `RegimeDetector` - 시장 상태 감지기
- `RegimeInfo` - 시장 상태 정보

**시장 상태 타입:**
```python
class MarketRegime(Enum):
    TRENDING_UP = "trending_up"          # 상승 추세
    TRENDING_DOWN = "trending_down"      # 하락 추세
    RANGING = "ranging"                  # 횡보
    HIGH_VOLATILITY = "high_volatility"  # 고변동성
    LOW_VOLATILITY = "low_volatility"    # 저변동성
```

**사용 예제:**
```python
from regime import RegimeDetector

# 감지기 생성
detector = RegimeDetector(params={
    'adx_trending_threshold': 25,      # ADX 추세 기준
    'adx_strong_trend_threshold': 40,  # 강한 추세 기준
    'volatility_high_threshold': 0.03, # 고변동성 기준 (3%)
    'volatility_low_threshold': 0.01,  # 저변동성 기준 (1%)
    'lookback_period': 50              # 분석 기간
})

# 시장 상태 감지
regime_info = detector.detect_regime(df)

print(f"시장 상태: {regime_info.regime.value}")
print(f"신뢰도: {regime_info.confidence:.2%}")
print(f"메트릭: {regime_info.metrics}")

# 적합한 전략 추천
suitable_strategies = detector.get_suitable_strategies(regime_info.regime)
print(f"추천 전략: {suitable_strategies}")
```

**감지 지표:**
- ADX (Average Directional Index) - 추세 강도
- +DI / -DI - 추세 방향
- ATR (Average True Range) - 변동성
- Bollinger Bands Width - 변동성
- Linear Regression Slope - 추세 방향
- 가격 vs 이동평균 - 추세 강도

**감지 로직:**
```
1. 변동성 판단
   - ATR% > 3% → HIGH_VOLATILITY
   - ATR% < 1% → LOW_VOLATILITY
   - BB Width > 10% → HIGH_VOLATILITY
   - BB Width < 5% → LOW_VOLATILITY

2. 추세 판단
   - ADX > 40 → 강한 추세
     - +DI > -DI → TRENDING_UP
     - +DI < -DI → TRENDING_DOWN
   - ADX > 25 → 중간 추세
   - ADX < 25 → RANGING

3. 가격 vs 이동평균
   - Price > SMA20 & SMA50 → TRENDING_UP
   - Price < SMA20 & SMA50 → TRENDING_DOWN
   - Otherwise → RANGING

최종 점수가 가장 높은 상태 선택
```

**시장 상태별 추천 전략:**

| 시장 상태 | 추천 전략 | 이유 |
|-----------|-----------|------|
| TRENDING_UP | MA_Crossover, LSTM | 추세 추종 전략 유리 |
| TRENDING_DOWN | MA_Crossover, LSTM | 추세 추종 전략 유리 |
| RANGING | RSI, BollingerBands | 평균회귀 전략 유리 |
| HIGH_VOLATILITY | BollingerBands, RSI | 변동성 기반 전략 유리 |
| LOW_VOLATILITY | MA_Crossover, LSTM | 안정적 추세 추종 |

---

### 2. 전략 성과 추적 시스템

#### [strategy_selector/performance_tracker.py](strategy_selector/performance_tracker.py)

**핵심 클래스:**
- `PerformanceTracker` - 성과 추적기
- `StrategyMetrics` - 전략 성과 메트릭
- `Trade` - 거래 기록

**사용 예제:**
```python
from strategy_selector import PerformanceTracker

# 추적기 생성
tracker = PerformanceTracker()

# 전략 추가
tracker.add_strategy('MA_Crossover')
tracker.add_strategy('RSI')

# 거래 기록
tracker.record_trade(
    strategy_name='MA_Crossover',
    timestamp=datetime.now(),
    side='BUY',
    price=50000.0,
    quantity=0.01,
    pnl=None  # 진입 시
)

# 청산 시 (PnL 포함)
tracker.record_trade(
    strategy_name='MA_Crossover',
    timestamp=datetime.now(),
    side='SELL',
    price=51000.0,
    quantity=0.01,
    pnl=10.0  # 수익
)

# 성과 계산
tracker.calculate_metrics('MA_Crossover', initial_capital=10000)

# 메트릭 조회
metrics = tracker.get_metrics('MA_Crossover')
print(f"승률: {metrics.win_rate:.2%}")
print(f"총 손익: {metrics.total_pnl:.2f}")
print(f"샤프 비율: {metrics.sharpe_ratio:.2f}")
print(f"최대 낙폭: {metrics.max_drawdown:.2%}")
print(f"Profit Factor: {metrics.profit_factor:.2f}")

# 전략 비교
comparison = tracker.compare_strategies(initial_capital=10000)
for rank, (name, sharpe, return_pct) in enumerate(comparison, 1):
    print(f"{rank}. {name}: 샤프 {sharpe:.2f}, 수익률 {return_pct:.2%}")
```

**성과 메트릭:**

| 메트릭 | 설명 | 계산 방식 |
|--------|------|-----------|
| **승률** | 수익 거래 비율 | 승리 거래 / 총 거래 |
| **평균 이익** | 수익 거래 평균 | sum(수익) / 수익 거래 수 |
| **평균 손실** | 손실 거래 평균 | sum(손실) / 손실 거래 수 |
| **Profit Factor** | 총 이익/손실 비율 | 총 이익 / 총 손실 |
| **샤프 비율** | 위험 대비 수익률 | (평균 수익률 - 무위험 수익률) / 수익률 표준편차 × √252 |
| **최대 낙폭** | 최대 손실 폭 | (Peak - Trough) / Initial Capital |
| **총 수익률** | 전체 수익률 | 총 손익 / 초기 자본 |

**샤프 비율 계산:**
```python
# 일일 수익률 기준
returns_array = np.array(daily_returns)
avg_return = np.mean(returns_array)
std_return = np.std(returns_array)

# 연율화 (252 거래일 가정)
sharpe_ratio = (avg_return / std_return) * np.sqrt(252)
```

---

### 3. 동적 전략 선택 시스템

#### [strategy_selector/strategy_selector.py](strategy_selector/strategy_selector.py)

**핵심 클래스:**
- `StrategySelector` - 동적 전략 선택기

**사용 예제:**
```python
from strategy_selector import StrategySelector
from strategies.technical import MACrossoverStrategy, RSIStrategy, BollingerBandsStrategy

# 전략 딕셔너리
strategies = {
    'MA_Crossover': MACrossoverStrategy(),
    'RSI': RSIStrategy(),
    'BollingerBands': BollingerBandsStrategy()
}

# 선택기 생성
selector = StrategySelector(
    strategies=strategies,
    params={
        'regime_weight': 0.5,        # 시장 상태 가중치
        'performance_weight': 0.5,   # 성과 가중치
        'min_trades_for_perf': 10,   # 성과 평가 최소 거래 수
        'rebalance_interval': 24     # 리밸런싱 간격 (시간)
    }
)

# 최적 전략 선택
best_name, best_strategy = selector.select_best_strategy(df)
print(f"선택된 전략: {best_name}")

# 신호 생성
signal = best_strategy.generate_signal(df)
print(f"신호: {signal.signal.name}")
print(f"신뢰도: {signal.confidence:.2%}")
```

**앙상블 신호 생성:**
```python
# 모든 전략의 가중 평균 신호
ensemble_signal = selector.generate_ensemble_signal(df)

print(f"앙상블 신호: {ensemble_signal.signal.name}")
print(f"신뢰도: {ensemble_signal.confidence:.2%}")
print(f"가중 스코어: {ensemble_signal.metadata['weighted_score']:.2f}")

# 각 전략 상세
for name, details in ensemble_signal.metadata['strategy_signals'].items():
    print(f"{name}: {details['signal']} (가중치: {details['weight']:.2%})")
```

**가중치 계산 로직:**

**1) 시장 상태 기반 가중치:**
```python
# 시장 상태 감지
regime_info = detector.detect_regime(data)

# 적합한 전략 목록
suitable_strategies = ['MA_Crossover', 'LSTM']  # TRENDING_UP 예시

# 적합한 전략에만 가중치 부여
base_weight = 1.0 / len(suitable_strategies)

regime_weights = {
    'MA_Crossover': base_weight * regime_info.confidence,
    'RSI': 0.0,
    'BollingerBands': 0.0,
    'LSTM': base_weight * regime_info.confidence
}
```

**2) 성과 기반 가중치:**
```python
# 각 전략의 샤프 비율 수집
sharpe_ratios = {
    'MA_Crossover': 1.5,
    'RSI': 0.8,
    'BollingerBands': 1.2,
    'LSTM': 2.0
}

total_sharpe = sum(sharpe_ratios.values())  # 5.5

# 샤프 비율 기반 가중치
performance_weights = {
    'MA_Crossover': 1.5 / 5.5 = 0.273,
    'RSI': 0.8 / 5.5 = 0.145,
    'BollingerBands': 1.2 / 5.5 = 0.218,
    'LSTM': 2.0 / 5.5 = 0.364
}
```

**3) 통합 가중치:**
```python
# 파라미터
regime_weight = 0.5
performance_weight = 0.5

# 통합 계산
combined_weights = {}
for strategy in strategies:
    combined = (
        regime_weights[strategy] * regime_weight +
        performance_weights[strategy] * performance_weight
    )
    combined_weights[strategy] = combined

# 정규화
total = sum(combined_weights.values())
combined_weights = {k: v / total for k, v in combined_weights.items()}
```

**앙상블 신호 계산:**
```python
# 각 전략 신호 수집
signals = {
    'MA_Crossover': TradeSignal(BUY, confidence=0.8),
    'RSI': TradeSignal(HOLD, confidence=0.5),
    'BollingerBands': TradeSignal(SELL, confidence=0.7),
    'LSTM': TradeSignal(BUY, confidence=0.9)
}

# 가중 평균 스코어 계산
weighted_score = 0.0
for strategy, signal in signals.items():
    score = 0.0
    if signal.signal == BUY:
        score = 1.0
    elif signal.signal == SELL:
        score = -1.0

    weighted_score += score * signal.confidence * combined_weights[strategy]

# 최종 신호 결정
if weighted_score > 0.2:
    final_signal = BUY
elif weighted_score < -0.2:
    final_signal = SELL
else:
    final_signal = HOLD
```

**리밸런싱:**
```python
# 24시간마다 자동 리밸런싱
if selector.should_rebalance():
    selector.rebalance(df)
    print(f"리밸런싱 완료: {selector.current_weights}")
```

---

## 🚀 전체 테스트 파이프라인

### [test_phase4.py](test_phase4.py)

**실행 방법:**
```bash
python test_phase4.py
```

**테스트 항목:**
1. ✅ 시장 상태 감지 테스트
2. ✅ 전략 성과 추적 테스트
3. ✅ 동적 전략 선택 테스트

**예상 출력:**
```
======================================================================
Phase 4: 전략 스위칭 시스템 테스트
======================================================================

======================================================================
[ 1/3 ] 시장 상태 감지 테스트
======================================================================
✅ 데이터 수집 완료: 336개
시장 상태: trending_up
신뢰도: 65.23%
메트릭:
  adx: 32.45
  plus_di: 28.34
  minus_di: 18.67
  atr_pct: 0.0234
  bb_width: 0.0876
  lr_slope: 125.34
  price_vs_sma20: 0.0245
  price_vs_sma50: 0.0387
추천 전략: ['MA_Crossover', 'LSTM']

======================================================================
[ 2/3 ] 전략 성과 추적 테스트
======================================================================
======================================================================
전략 성과 요약
======================================================================

[MA_Crossover]
  총 거래: 20
  승률: 60.00%
  총 손익: 234.56
  수익률: 2.35%
  샤프 비율: 1.45
  최대 낙폭: 1.23%
  Profit Factor: 2.34

[RSI]
  총 거래: 20
  승률: 55.00%
  총 손익: 178.92
  수익률: 1.79%
  샤프 비율: 1.12
  최대 낙폭: 1.56%
  Profit Factor: 1.89

전략 순위:
  1. MA_Crossover: 샤프 1.45, 수익률 2.35%
  2. RSI: 샤프 1.12, 수익률 1.79%

======================================================================
[ 3/3 ] 동적 전략 선택 테스트
======================================================================
✅ 데이터 수집 완료: 336개
시장 상태: trending_up (신뢰도: 65.23%)
전략 가중치:
  MA_Crossover: 45.67%
  RSI: 12.34%
  BollingerBands: 8.99%
선택된 전략: MA_Crossover
신호: BUY
신뢰도: 78.34%
이유: 골든크로스 발생

앙상블 신호 생성:
신호: BUY
신뢰도: 72.45%
이유: 앙상블 신호 (스코어: 0.58)
가중 스코어: 0.58
각 전략 신호:
  MA_Crossover: BUY (신뢰도: 78.34%, 가중치: 45.67%)
  RSI: HOLD (신뢰도: 0.00%, 가중치: 12.34%)
  BollingerBands: SELL (신뢰도: 68.23%, 가중치: 8.99%)

======================================================================
✅ Phase 4 테스트 완료!
======================================================================
```

---

## 📈 전략 스위칭 시나리오

### 시나리오 1: 추세장 → 횡보장 전환

**초기 상태 (추세장):**
```
시장 상태: TRENDING_UP
추천 전략: MA_Crossover (60%), LSTM (40%)
활성 전략: MA_Crossover
신호: BUY
```

**시장 변화 후 (횡보장):**
```
시장 상태: RANGING
추천 전략: RSI (55%), BollingerBands (45%)
활성 전략: RSI (자동 전환)
신호: HOLD
```

### 시나리오 2: 성과 기반 전환

**초기 상태:**
```
MA_Crossover: 샤프 1.2, 가중치 40%
RSI: 샤프 0.8, 가중치 30%
LSTM: 샤프 1.5, 가중치 30%
```

**성과 변화 후:**
```
MA_Crossover: 샤프 0.5 (급락), 가중치 15%
RSI: 샤프 1.8 (급상승), 가중치 45%
LSTM: 샤프 1.5 (유지), 가중치 40%

→ RSI로 자동 전환
```

### 시나리오 3: 앙상블 모드

```python
# 모든 전략을 가중 평균으로 사용
ensemble_signal = selector.generate_ensemble_signal(df)

# 예시 결과:
# MA_Crossover (45%): BUY (0.8)
# RSI (30%): HOLD (0.0)
# BollingerBands (25%): SELL (0.7)
#
# 가중 스코어 = 0.45 * 1.0 * 0.8 + 0.30 * 0.0 * 0.0 + 0.25 * (-1.0) * 0.7
#             = 0.36 + 0.0 - 0.175
#             = 0.185
#
# 0.185 < 0.2 → HOLD
```

---

## 💡 사용 팁

### 1. 시장 상태별 전략 최적화

```python
# 시장 상태에 따라 전략 파라미터 조정
detector = RegimeDetector()
regime_info = detector.detect_regime(df)

if regime_info.regime == MarketRegime.HIGH_VOLATILITY:
    # 고변동성 시 더 보수적인 파라미터
    strategy = BollingerBandsStrategy(params={
        'std_dev': 2.5,  # 더 넓은 밴드
        'touch_threshold': 0.03  # 더 엄격한 기준
    })
elif regime_info.regime == MarketRegime.TRENDING_UP:
    # 추세장 시 공격적인 파라미터
    strategy = MACrossoverStrategy(params={
        'fast_period': 5,   # 더 빠른 진입
        'slow_period': 20
    })
```

### 2. 실시간 모니터링

```python
import time

selector = StrategySelector(strategies)

while True:
    # 최신 데이터 가져오기
    df = collector.get_recent_data('BTCUSDT', '1h', days=14)

    # 리밸런싱 확인
    if selector.should_rebalance():
        selector.rebalance(df)
        logger.info("전략 리밸런싱 완료")

    # 앙상블 신호 생성
    signal = selector.generate_ensemble_signal(df)

    if signal.signal != Signal.HOLD:
        logger.warning(f"신호 발생: {signal.signal.name}")
        # 실제 거래 실행 (Phase 5에서 구현)

    time.sleep(3600)  # 1시간마다
```

### 3. 백테스팅 통합

```python
# 과거 데이터로 전략 성과 평가
df_historical = collector.collect_klines(
    symbol='BTCUSDT',
    interval='1h',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# 슬라이딩 윈도우로 백테스트
for i in range(len(df_historical) - 168):
    window = df_historical.iloc[i:i+168]

    # 신호 생성
    signal = selector.generate_ensemble_signal(window)

    # 거래 실행 (모의)
    if signal.signal == Signal.BUY:
        # 매수 로직
        pass
    elif signal.signal == Signal.SELL:
        # 매도 로직
        pass

    # 성과 기록
    tracker.record_trade(...)

# 최종 성과 평가
tracker.calculate_metrics('Ensemble')
```

---

## 🎯 다음 단계: Phase 5

Phase 5에서는 **실전 배포 준비**를 진행합니다:

1. **백테스팅 엔진**
   - 과거 데이터 기반 전략 검증
   - 성과 시각화
   - 리스크 분석

2. **페이퍼 트레이딩**
   - 실시간 모의 거래
   - 주문 실행 로직
   - 포지션 관리

3. **모니터링 시스템**
   - 실시간 성과 대시보드
   - 알림 시스템 (Slack/Telegram)
   - 로깅 및 감사

4. **실전 거래**
   - 실제 주문 실행
   - 리스크 제한 적용
   - 비상 정지 메커니즘

---

## 📝 주요 파일 정리

| 파일 | 설명 | 상태 |
|------|------|------|
| [regime/market_regime.py](regime/market_regime.py) | 시장 상태 감지기 | ✅ |
| [strategy_selector/performance_tracker.py](strategy_selector/performance_tracker.py) | 성과 추적기 | ✅ |
| [strategy_selector/strategy_selector.py](strategy_selector/strategy_selector.py) | 동적 전략 선택기 | ✅ |
| [test_phase4.py](test_phase4.py) | Phase 4 테스트 | ✅ |

---

## 🎉 Phase 4 완료!

모든 전략 스위칭 시스템이 성공적으로 구현되었습니다!

**구현 완료:**
- ✅ 시장 상태 감지 시스템 (5개 상태)
- ✅ 전략 성과 추적 시스템 (7개 메트릭)
- ✅ 동적 전략 선택기 (가중 평균 앙상블)
- ✅ 자동 리밸런싱
- ✅ 테스트 스크립트

**다음 단계:**
- Phase 5: 실전 배포 준비 (백테스팅, 페이퍼 트레이딩, 모니터링)
