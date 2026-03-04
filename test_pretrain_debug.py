"""사전학습 흐름 디버그 테스트 — 실제 백테스트와 동일한 경로를 재현"""
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── 1. feature_builder 테스트 ──
print("=" * 60)
print("1. TFFeatureBuilder 테스트")
print("=" * 60)

from ai.feature_builder import TFFeatureBuilder
fb = TFFeatureBuilder()

# 가짜 1m OHLCV (300행)
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(300) * 0.1)
df_fake = pd.DataFrame({
    'open': prices,
    'high': prices + np.abs(np.random.randn(300) * 0.05),
    'low': prices - np.abs(np.random.randn(300) * 0.05),
    'close': prices + np.random.randn(300) * 0.02,
    'volume': np.random.uniform(100, 1000, 300),
}, index=pd.date_range('2025-01-01', periods=300, freq='1min'))

for tf in ['1m', '3m', '5m', '15m', '10s']:
    try:
        features = fb.prepare_features(df_fake.copy(), tf)
        nan_count = np.isnan(features).sum()
        inf_count = np.isinf(features).sum()
        print(f"  {tf}: shape={features.shape}, nan={nan_count}, inf={inf_count}, "
              f"min={features.min():.2f}, max={features.max():.2f} ✅")
    except Exception as e:
        print(f"  {tf}: FAILED — {e} ❌")

# ── 2. MarketAnalyzer v2 생성 테스트 ──
print()
print("=" * 60)
print("2. MarketAnalyzer v2 (1m) 생성 + add_training_sample")
print("=" * 60)

from ai.market_analyzer import MarketAnalyzer
ma = MarketAnalyzer(
    sequence_length=60,
    model_path=None,
    timeframe='1m',
    feature_builder=fb
)
print(f"  _use_v2_features={ma._use_v2_features}, input_size={ma.input_size}")
print(f"  feature_names={ma.feature_names}")

# add_training_sample 테스트
context_size = 300
seq_len = 60
samples_added = 0
samples_failed = 0

for i in range(seq_len + 5, 200):
    try:
        start_idx = max(0, i - context_size)
        current_df = df_fake.iloc[start_idx:i].copy()
        entry_price = df_fake.iloc[i - 1]['close']
        future_price = df_fake.iloc[min(i + 2, len(df_fake) - 1)]['close']
        log_return = math.log(future_price / entry_price)
        actual_direction = 'UP' if log_return > 0 else 'DOWN'
        net_pct = (future_price / entry_price - 1) * 100
        net_log_return = log_return - 0.001

        ma.add_training_sample(
            df=current_df,
            actual_direction=actual_direction,
            actual_price_change=net_pct,
            sample_weight=1.0,
            future_log_return=net_log_return
        )
        samples_added += 1
    except Exception as e:
        samples_failed += 1
        if samples_failed <= 3:
            print(f"  샘플 추가 예외 (i={i}): {e}")

print(f"  samples_added={samples_added}, samples_failed={samples_failed}")
print(f"  training_buffer 실제 크기={len(ma.training_buffer)}")

# ── 3. batch_train 테스트 ──
print()
print("=" * 60)
print("3. MarketAnalyzer batch_train")
print("=" * 60)

if len(ma.training_buffer) >= ma.min_samples_for_training:
    result = ma.batch_train(epochs=2)
    print(f"  결과: {result}")
else:
    print(f"  ❌ 샘플 부족: {len(ma.training_buffer)} < {ma.min_samples_for_training}")

# ── 4. TickAnalyzer v2 생성 + 학습 테스트 ──
print()
print("=" * 60)
print("4. TickAnalyzer v2 (10s) 생성 + add_training_sample + batch_train")
print("=" * 60)

from ai.market_analyzer import TickAnalyzer
ta = TickAnalyzer(
    sequence_length=30,
    hidden_size=64,
    model_path=None,
    feature_builder=fb
)
print(f"  _use_v2_features={ta._use_v2_features}, input_size={ta.input_size}")

# 합성 10s 데이터
micro_prices = 100 + np.cumsum(np.random.randn(600) * 0.01)
df_10s = pd.DataFrame({
    'open': micro_prices,
    'high': micro_prices + np.abs(np.random.randn(600) * 0.005),
    'low': micro_prices - np.abs(np.random.randn(600) * 0.005),
    'close': micro_prices + np.random.randn(600) * 0.002,
    'volume': np.random.uniform(10, 100, 600),
}, index=pd.date_range('2025-01-01', periods=600, freq='10s'))

tick_seq_len = 30
tick_context = max(tick_seq_len * 4, 300)
tick_samples = 0
tick_failed = 0

for i in range(tick_seq_len + 5, len(df_10s) - 3):
    try:
        start_idx = max(0, i - tick_context)
        current_df = df_10s.iloc[start_idx:i].copy()
        entry_price = df_10s.iloc[i - 1]['close']
        future_price = df_10s.iloc[i + 2]['close']
        log_return = math.log(future_price / entry_price)
        actual_direction = 'UP' if log_return > 0 else 'DOWN'
        net_pct = (future_price / entry_price - 1) * 100

        ta.add_training_sample(
            df=current_df,
            actual_direction=actual_direction,
            actual_price_change=net_pct,
            sample_weight=1.0
        )
        tick_samples += 1
    except Exception as e:
        tick_failed += 1
        if tick_failed <= 3:
            print(f"  틱 샘플 추가 예외 (i={i}): {e}")

print(f"  tick_samples={tick_samples}, tick_failed={tick_failed}")
print(f"  training_buffer 실제 크기={len(ta.training_buffer)}")

if len(ta.training_buffer) >= ta.min_samples_for_training:
    result = ta.batch_train(epochs=2)
    print(f"  결과: {result}")
else:
    print(f"  ❌ 샘플 부족: {len(ta.training_buffer)} < {ta.min_samples_for_training}")

# ── 5. IntegratedAIAnalyzer + MarketAnalyzer 교체 테스트 ──
print()
print("=" * 60)
print("5. IntegratedAIAnalyzer 생성 후 MarketAnalyzer v2 교체")
print("=" * 60)

from ai.market_analyzer import IntegratedAIAnalyzer
try:
    ia = IntegratedAIAnalyzer(
        model_weight=0.4,
        technical_weight=0.6,
        model_path=None,
        enabled_indicators=['trend_ma', 'rsi', 'macd', 'bollinger']
    )
    print(f"  IntegratedAI 내부 MA: _use_v2={ia.market_analyzer._use_v2_features}, "
          f"input_size={ia.market_analyzer.input_size}")

    # v2로 교체
    new_ma = MarketAnalyzer(
        sequence_length=60,
        model_path=None,
        timeframe='1m',
        feature_builder=fb
    )
    new_ma.set_timeframe('1m')
    ia.market_analyzer = new_ma
    print(f"  교체 후 MA: _use_v2={ia.market_analyzer._use_v2_features}, "
          f"input_size={ia.market_analyzer.input_size}")
    print("  ✅ 교체 성공")
except Exception as e:
    print(f"  ❌ 실패: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("테스트 완료")
print("=" * 60)
