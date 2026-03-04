# 백테스트 거래 수 0 문제 해결 가이드

## 🔍 문제 요약

백테스트 결과 거래 수가 0으로 나타나는 이유:

1. **전략이 너무 보수적**: 교차, 극단값에서만 신호 생성
2. **시장 조건 불일치**: 30일간 강한 상승 추세로 교차/극단값 발생 안 함
3. **신뢰도 필터**: 모든 신호의 신뢰도가 0.0 → 0.6 임계값 통과 못함

---

## ⚡ Quick Fix (즉시 적용)

### 방법 1: 신뢰도 임계값 낮추기 (가장 빠름)

**파일**: `trading_bot/bot.py`

**수정 위치**: Line 286, 299

```python
# 현재 (보수적)
if ensemble_signal.signal == Signal.BUY and ensemble_signal.confidence > 0.6:

# 변경 (적극적)
if ensemble_signal.signal == Signal.BUY and ensemble_signal.confidence > 0.4:
```

**동일하게 Line 299도 수정:**
```python
# 현재
if ensemble_signal.confidence > 0.6:

# 변경
if ensemble_signal.confidence > 0.4:
```

**영향**: 더 많은 거래 기회 발생 (단, 품질은 다소 낮아질 수 있음)

---

### 방법 2: MA Crossover 전략 개선 (권장)

**파일**: `strategies/technical/ma_crossover.py`

**수정 위치**: Line 129-135

**현재 코드:**
```python
# 추세 지속 (홀딩)
else:
    if current['fast_ma'] > current['slow_ma']:
        reason = "상승 추세 지속 중"
    else:
        reason = "하락 추세 지속 중"
    confidence = 0.0  # ← 문제!
```

**개선 코드:**
```python
# 추세 지속 시에도 신호 생성
else:
    if current['fast_ma'] > current['slow_ma']:
        # 상승 추세 지속 → 매수 신호
        signal = Signal.BUY

        # MA 간격에 비례한 신뢰도 (최대 50%)
        ma_spread = (current['fast_ma'] - current['slow_ma']) / current['slow_ma']
        trend_strength = min(ma_spread * 10, 0.5)

        # 거래량 고려
        volume_boost = min((current['volume_ratio'] - 1.0) * 0.2, 0.1)

        confidence = min(trend_strength + volume_boost, 0.6)
        reason = f"상승 추세 지속 중 (강도: {confidence:.2%})"
    else:
        # 하락 추세 지속 → 매도 신호
        signal = Signal.SELL

        ma_spread = (current['slow_ma'] - current['fast_ma']) / current['fast_ma']
        trend_strength = min(ma_spread * 10, 0.5)

        volume_boost = min((current['volume_ratio'] - 1.0) * 0.2, 0.1)

        confidence = min(trend_strength + volume_boost, 0.6)
        reason = f"하락 추세 지속 중 (강도: {confidence:.2%})"
```

**영향**: 추세가 지속될 때도 방향성에 따라 신호 생성 (신뢰도 20~60%)

---

### 방법 3: RSI 전략 개선

**파일**: `strategies/technical/rsi_strategy.py`

**수정 위치**: Line 96-98 (중립 구간 처리 부분 이후)

**추가 코드:**
```python
# 기존 코드 유지 (Line 96-132)
# ...

# 중립 구간에서도 방향성 신호 생성 (추가)
else:
    # RSI가 50 이상이면 상승 압력
    if current['rsi'] >= 50:
        signal = Signal.BUY
        # 50에서 70 사이: 신뢰도 0~0.4
        rsi_strength = (current['rsi'] - 50) / 20  # 0~1
        confidence = rsi_strength * 0.4  # 최대 40%
        reason = f"RSI 중립-강세 ({current['rsi']:.2f})"

    # RSI가 50 미만이면 하락 압력
    else:
        signal = Signal.SELL
        # 30에서 50 사이: 신뢰도 0~0.4
        rsi_strength = (50 - current['rsi']) / 20
        confidence = rsi_strength * 0.4
        reason = f"RSI 중립-약세 ({current['rsi']:.2f})"

    # RSI 변화 방향 고려
    if current['rsi_change'] > 0:
        confidence *= 1.2  # 상승 중이면 boost
    elif current['rsi_change'] < 0:
        confidence *= 1.2

    confidence = min(confidence, 0.5)
```

**영향**: RSI 중립 구간에서도 방향성 신호 생성 (신뢰도 10~50%)

---

## 🚀 적용 순서

### Step 1: Quick Fix (5분)
```bash
# 방법 1만 적용: bot.py 수정
# Line 286, 299: 0.6 → 0.4

# 재테스트
python test_backtest.py
```

**예상 결과**: 거래 발생 (단, 품질은 낮을 수 있음)

---

### Step 2: 전략 개선 (15분)
```bash
# 방법 2, 3 적용
# ma_crossover.py, rsi_strategy.py 수정

# 재테스트
python test_backtest.py
```

**예상 결과**: 더 나은 품질의 거래 발생

---

### Step 3: 검증 (30분)
```bash
# 장기 백테스트로 검증
python test_backtest_optimized.py
# 선택: 1 (6개월 백테스트)
```

**예상 결과**: 다양한 시장 조건에서 안정적 성과

---

## 📊 예상 개선 효과

### Before (현재)
```
총 거래: 0
신호 발생: HOLD only
신뢰도: 0.0
```

### After (Quick Fix - 방법 1)
```
총 거래: 5~10 (예상)
신호 발생: BUY/SELL/HOLD
신뢰도: 0.4~0.6
```

### After (전략 개선 - 방법 2+3)
```
총 거래: 10~20 (예상)
신호 발생: BUY/SELL/HOLD (더 균형잡힘)
신뢰도: 0.3~0.7 (범위 확대)
승률: 50~60% (예상)
```

---

## ⚠️ 주의사항

### 과최적화 방지
- 너무 많은 거래가 발생하면 오히려 역효과
- 목표: 하루 1~2회 정도의 적절한 거래 빈도

### 백테스트 재검증
- 수정 후 반드시 재테스트
- 여러 기간, 여러 심볼로 검증

### Paper Trading
- 백테스트가 좋아도 반드시 Paper Trading으로 실시간 검증
- 최소 1주일 이상 모니터링

---

## 🎯 최종 권장 사항

**단계별 적용:**

1. **즉시**: Quick Fix (방법 1) 적용 → 테스트
2. **당일**: 전략 개선 (방법 2, 3) 적용 → 재테스트
3. **익일**: 장기 백테스트 (3~6개월) 검증
4. **주간**: Paper Trading 시작
5. **월간**: 실전 배포 (소액)

**목표 지표:**
- 총 거래: 10~30 (월간)
- 승률: > 50%
- Sharpe Ratio: > 1.0
- Max Drawdown: < 10%

---

## 📝 체크리스트

- [ ] bot.py 신뢰도 임계값 수정 (0.6 → 0.4)
- [ ] ma_crossover.py 추세 지속 신호 추가
- [ ] rsi_strategy.py 중립 구간 신호 추가
- [ ] 기본 백테스트 재실행
- [ ] 결과 확인 (거래 발생 여부)
- [ ] 장기 백테스트 (6개월) 실행
- [ ] 다중 심볼 비교
- [ ] Paper Trading 시작

---

**작성일**: 2026년 1월 7일
**상태**: 문제 분석 완료, 해결책 제시
