# 백테스팅 사용 가이드

## 📚 목차
1. [백테스팅이란?](#백테스팅이란)
2. [기본 백테스트 실행](#기본-백테스트-실행)
3. [개선된 백테스트 실행](#개선된-백테스트-실행)
4. [파라미터 설정](#파라미터-설정)
5. [결과 해석](#결과-해석)
6. [최적화 방법](#최적화-방법)

---

## 백테스팅이란?

백테스팅(Backtesting)은 **과거 데이터를 이용하여 거래 전략의 성과를 시뮬레이션**하는 과정입니다.

### 목적
- ✅ 전략의 수익성 검증
- ✅ 리스크 평가 (최대 낙폭, 변동성)
- ✅ 전략 간 비교
- ✅ 파라미터 최적화
- ✅ 실전 배포 전 검증

### 주의사항
⚠️ **백테스트 결과가 좋다고 해서 실전에서도 반드시 수익이 나는 것은 아닙니다!**

- 과거 성과 ≠ 미래 성과
- 오버피팅(Overfitting) 주의
- 수수료, 슬리피지 고려 필수
- 충분히 긴 기간과 다양한 시장 조건 테스트 필요

---

## 기본 백테스트 실행

### 1. 단순 백테스트 (최근 30일)

```bash
python test_backtest.py
```

**실행 내용:**
- 심볼: BTCUSDT
- 기간: 최근 30일
- 초기 자본: $10,000
- 수수료: 0.1%
- 슬리피지: 0.05%

**결과 파일:**
- `backtesting/results/equity_curve.csv` - Equity curve 데이터

**예상 실행 시간:** 약 30초 ~ 1분

---

## 개선된 백테스트 실행

### 2. 메뉴 방식 백테스트

```bash
python test_backtest_optimized.py
```

**메뉴 옵션:**

#### 옵션 1: 장기 백테스트 (6개월)
```
선택: 1
```
- 기간: 최근 6개월
- 월별 성과 분석 포함
- Equity curve 저장: `equity_curve_6months.csv`

#### 옵션 2: 다중 심볼 비교
```
선택: 2
```
- 심볼: BTCUSDT, ETHUSDT, BNBUSDT
- 기간: 최근 3개월
- 심볼별 성과 비교표 출력

#### 옵션 3: 파라미터 최적화
```
선택: 3
```
- 신뢰도 임계값 그리드 서치: [0.4, 0.5, 0.6, 0.7]
- 최적 파라미터 자동 추천
- Sharpe ratio 및 수익률 기준

#### 옵션 4: 모두 실행
```
선택: 4
```
- 위의 1, 2, 3을 모두 순차 실행
- 종합 분석 리포트 생성

---

## 파라미터 설정

### 백테스트 엔진 파라미터

```python
from backtesting.backtest_engine import BacktestEngine

engine = BacktestEngine(
    symbol='BTCUSDT',          # 거래 심볼
    interval='1h',              # 시간 간격 (1m, 5m, 15m, 1h, 4h, 1d)
    initial_capital=10000.0,    # 초기 자본 (USDT)
    commission=0.001,           # 거래 수수료 (0.1%)
    slippage=0.0005             # 슬리피지 (0.05%)
)
```

### 백테스트 실행 파라미터

```python
result = engine.run(
    start_date='2024-01-01',    # 시작일 (YYYY-MM-DD)
    end_date='2024-12-31',      # 종료일 (YYYY-MM-DD)
    lookback_days=14            # 지표 계산을 위한 lookback 기간
)
```

### 전략 파라미터 (strategies/ 내부 수정)

현재 기본값 (보수적):
```python
# MA Crossover
fast_period = 7
slow_period = 25

# RSI
oversold = 30
overbought = 70

# Bollinger Bands
period = 20
std_dev = 2.0

# 신호 신뢰도 임계값
min_confidence = 0.6  # 60% 이상만 거래
```

**개선 제안 (더 적극적):**
```python
# MA Crossover
fast_period = 5
slow_period = 20

# RSI
oversold = 35
overbought = 65

# 신호 신뢰도 임계값
min_confidence = 0.5  # 50% 이상 거래
```

---

## 결과 해석

### 백테스트 결과 주요 지표

#### 1. 수익률 관련
```
총 수익: $500.00 (5.00%)
```
- **절대 수익**: 달러 금액
- **수익률**: 초기 자본 대비 백분율
- **평가**: 양수면 수익, 음수면 손실

#### 2. 리스크 관련
```
최대 낙폭: $200.00 (2.00%)
```
- **Max Drawdown**: 고점 대비 최대 하락폭
- **평가**: 낮을수록 안정적 (보통 10% 이하 선호)

#### 3. 위험 조정 수익
```
Sharpe Ratio: 1.50
Sortino Ratio: 2.00
```
- **Sharpe Ratio**: 변동성 대비 수익 (1.0 이상 양호, 2.0 이상 우수)
- **Sortino Ratio**: 하방 변동성만 고려 (Sharpe보다 높음)

#### 4. 거래 통계
```
총 거래: 50
승리 거래: 30
손실 거래: 20
승률: 60.00%
```
- **승률**: 50% 이상 선호
- **거래 수**: 너무 많으면 과매매, 너무 적으면 기회 부족

#### 5. 수익 구조
```
Profit Factor: 1.80
```
- **Profit Factor**: 총 이익 / 총 손실
- **평가**: 1.5 이상 양호, 2.0 이상 우수

### 좋은 백테스트 결과 예시

```
✅ 우수한 결과:
- 총 수익률: +20% (6개월)
- Max Drawdown: -5%
- Sharpe Ratio: 2.0
- 승률: 55%
- Profit Factor: 2.5
- 총 거래: 30~50 (적절한 빈도)
```

### 나쁜 백테스트 결과 예시

```
❌ 부족한 결과:
- 총 수익률: -10%
- Max Drawdown: -25%
- Sharpe Ratio: 0.3
- 승률: 35%
- Profit Factor: 0.7
- 총 거래: 200+ (과매매) 또는 0~3 (기회 부족)
```

---

## 최적화 방법

### 1. 파라미터 튜닝 순서

#### Step 1: 현재 상태 파악
```bash
python test_backtest.py
```
- 기본 설정으로 먼저 실행
- 결과 확인 (수익률, Sharpe, 거래 수)

#### Step 2: 신호 신뢰도 조정
```python
# strategies/base.py 또는 각 전략 파일에서 수정
min_confidence = 0.5  # 0.6 → 0.5로 낮춤
```
- 거래 기회 증가 목적
- 너무 낮추면 품질 저하

#### Step 3: 전략 파라미터 조정
```python
# MA Crossover (strategies/technical/ma_crossover.py)
'fast_period': 5,   # 7 → 5
'slow_period': 20,  # 25 → 20

# RSI (strategies/technical/rsi_strategy.py)
'oversold_level': 35,   # 30 → 35
'overbought_level': 65, # 70 → 65
```

#### Step 4: 재테스트
```bash
python test_backtest.py
```
- 변경 후 성과 비교
- 개선되면 채택, 아니면 롤백

### 2. 그리드 서치 자동화

```bash
python test_backtest_optimized.py
# 선택: 3 (파라미터 최적화)
```

자동으로 여러 파라미터 조합 테스트하여 최적값 찾기.

### 3. 다양한 시장 조건 테스트

#### 상승장 테스트
```python
start_date = '2024-10-01'  # 비트코인 상승 구간
end_date = '2024-12-31'
```

#### 하락장 테스트
```python
start_date = '2024-06-01'  # 비트코인 하락 구간
end_date = '2024-08-31'
```

#### 횡보장 테스트
```python
start_date = '2024-03-01'  # 비트코인 횡보 구간
end_date = '2024-05-31'
```

**목표**: 모든 시장 조건에서 안정적인 성과

### 4. 여러 심볼 테스트

```bash
python test_backtest_optimized.py
# 선택: 2 (다중 심볼 비교)
```

- BTCUSDT, ETHUSDT, BNBUSDT 동시 테스트
- 심볼별 최적 파라미터 발견

---

## 백테스팅 워크플로우 (권장)

### 단계별 가이드

#### Phase 1: 기본 검증 (1~2일)
```bash
# 1. 기본 백테스트 (30일)
python test_backtest.py

# 2. 결과 확인
# - 거래가 발생했는가?
# - 수익이 났는가?
# - Sharpe ratio가 1.0 이상인가?
```

**통과 조건**: 거래 3회 이상, 수익률 양수, Sharpe > 0.5

---

#### Phase 2: 파라미터 조정 (3~5일)
```bash
# 3. 파라미터 최적화
python test_backtest_optimized.py
# 선택: 3

# 4. 최적 파라미터 적용
# - strategies/ 파일 수정
# - 재테스트
```

**통과 조건**: Sharpe > 1.0, 승률 > 50%, Profit Factor > 1.5

---

#### Phase 3: 장기 검증 (5~7일)
```bash
# 5. 6개월 백테스트
python test_backtest_optimized.py
# 선택: 1

# 6. 월별 성과 확인
# - 모든 월에서 수익인가?
# - Max Drawdown이 10% 이하인가?
```

**통과 조건**: 6개월 누적 수익 양수, Max DD < 10%

---

#### Phase 4: 다각도 검증 (7~10일)
```bash
# 7. 여러 심볼 테스트
python test_backtest_optimized.py
# 선택: 2

# 8. 다양한 기간 테스트
# - 상승장, 하락장, 횡보장 각각 테스트
```

**통과 조건**: 최소 2개 심볼에서 양호한 성과

---

#### Phase 5: Paper Trading (10~30일)
```bash
# 9. 실시간 Paper Trading
python -c "from trading_bot import TradingBot; bot = TradingBot(paper_trading=True); bot.run_continuous(interval_minutes=60, max_cycles=None)"

# 10. 일주일 ~ 한 달 실시간 검증
# - 백테스트와 유사한 성과가 나오는가?
```

**통과 조건**: Paper trading 성과 ≈ 백테스트 성과

---

#### Phase 6: 실전 배포 (매우 신중)
- 소액으로 시작 (초기 자본의 10~20%)
- 일주일 단위로 성과 모니터링
- 문제 없으면 점진적으로 자본 증가

---

## 자주 묻는 질문 (FAQ)

### Q1: 백테스트 결과가 "거래 없음"으로 나옵니다.

**A:** 현재 전략이 매우 보수적으로 설정되어 있기 때문입니다.

**해결 방법:**
1. 신뢰도 임계값 낮추기 (0.6 → 0.5)
2. MA 기간 짧게 조정 (fast: 5, slow: 20)
3. 더 긴 백테스트 기간 사용 (3~6개월)

---

### Q2: Sharpe Ratio가 음수로 나옵니다.

**A:** 전략이 손실을 내고 있거나 변동성이 매우 큽니다.

**해결 방법:**
1. 전략 파라미터 재조정
2. Stop-Loss 강화 (2% → 1.5%)
3. 포지션 크기 축소 (20% → 15%)

---

### Q3: 백테스트는 좋은데 Paper Trading에서 손실이 납니다.

**A:** 오버피팅(Overfitting) 가능성이 높습니다.

**해결 방법:**
1. 더 긴 백테스트 기간 사용
2. 다양한 시장 조건에서 테스트
3. 파라미터를 덜 최적화 (일반화)
4. Walk-forward 분석 수행

---

### Q4: 어떤 심볼이 가장 좋나요?

**A:** 변동성과 유동성을 고려해야 합니다.

**추천:**
- **BTCUSDT**: 가장 안정적, 유동성 최고
- **ETHUSDT**: BTC 다음으로 안정적
- **BNBUSDT**: 변동성 높음, 수익 기회 많지만 리스크도 높음

---

### Q5: 백테스트에 얼마나 시간을 써야 하나요?

**A:** 최소 1~2주, 이상적으로는 1개월

**이유:**
- 충분한 데이터 수집
- 다양한 시장 조건 경험
- 파라미터 최적화
- Paper trading 검증

---

## 다음 단계

백테스트가 만족스럽다면:

1. ✅ **Paper Trading** (가상 거래)
   ```bash
   python -c "from trading_bot import TradingBot; bot = TradingBot(paper_trading=True); bot.run_continuous()"
   ```

2. ✅ **실전 배포** (소액부터)
   - 초기 자본의 10~20%로 시작
   - 일주일 단위 모니터링
   - 점진적 자본 증가

3. ✅ **지속적 모니터링**
   - 일일 성과 체크
   - 월별 리밸런싱
   - 시장 환경 변화 대응

---

## 관련 문서

- [BACKTEST_SUMMARY.md](BACKTEST_SUMMARY.md) - 백테스트 결과 요약
- [PHASE3_SUMMARY.md](PHASE3_SUMMARY.md) - 거래 전략 설명
- [PHASE4_SUMMARY.md](PHASE4_SUMMARY.md) - 전략 스위칭 설명
- [README.md](README.md) - 전체 시스템 개요

---

**마지막 업데이트**: 2026년 1월 7일
