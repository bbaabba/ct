"""run_backtest() 사전학습 흐름 전체 재현 테스트 (v2 precompute 패턴)

실제 백테스트와 동일한 객체 생성 → 동일한 사전학습 루프 실행
각 단계마다 성공/실패/원인 출력
"""
import sys
import os
import math
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── 가짜 1m OHLCV 생성 ──
np.random.seed(42)
N = 10080  # 7일 * 24h * 60min
prices = 97000 + np.cumsum(np.random.randn(N) * 5)
df_1m = pd.DataFrame({
    'open': prices,
    'high': prices + np.abs(np.random.randn(N) * 3),
    'low': prices - np.abs(np.random.randn(N) * 3),
    'close': prices + np.random.randn(N) * 1,
    'volume': np.random.uniform(100, 5000, N),
}, index=pd.date_range('2025-02-14', periods=N, freq='1min'))

print(f"가짜 1m 데이터: {len(df_1m)}행")
print()

trading_fee_rate = 0.0004

# ════════════════════════════════════════════════════════════
# STEP 1: 객체 생성
# ════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: 객체 생성")
print("=" * 70)

from ai.feature_builder import TFFeatureBuilder
from ai.market_analyzer import MarketAnalyzer, TickAnalyzer, MultiTimeframeAnalyzer, TrainingSample

try:
    fb = TFFeatureBuilder()
    print(f"  [OK] TFFeatureBuilder")
except Exception as e:
    print(f"  [FAIL] TFFeatureBuilder: {e}")
    sys.exit(1)

# 1a. MultiTimeframeAnalyzer
mtf_intervals = ['1m', '3m', '5m', '15m']
try:
    mtf_analyzer = MultiTimeframeAnalyzer(
        symbol='BTCUSDT',
        intervals=mtf_intervals,
        model_dir='models',
        trend_filter=True,
        min_agreement=0.5
    )
    print(f"  [OK] MTF analyzers: {list(mtf_analyzer.analyzers.keys())}")
    for iv, ana in mtf_analyzer.analyzers.items():
        print(f"       {iv}: v2={ana._use_v2_features}, input={ana.input_size}, seq={ana.sequence_length}")
except Exception as e:
    print(f"  [FAIL] MTF: {e}")
    traceback.print_exc()
    mtf_analyzer = None

# 1b. v2 MarketAnalyzer
try:
    market_analyzer = MarketAnalyzer(
        sequence_length=60, model_path=None,
        timeframe='1m', feature_builder=fb
    )
    market_analyzer.set_timeframe('1m')
    print(f"  [OK] MarketAnalyzer v2: input={market_analyzer.input_size}, v2={market_analyzer._use_v2_features}")
except Exception as e:
    print(f"  [FAIL] MarketAnalyzer: {e}")
    traceback.print_exc()
    market_analyzer = None

# 1c. TickAnalyzer
try:
    tick_analyzer = TickAnalyzer(
        sequence_length=30, hidden_size=64,
        model_path=None, feature_builder=fb
    )
    print(f"  [OK] TickAnalyzer v2: input={tick_analyzer.input_size}, v2={tick_analyzer._use_v2_features}")
except Exception as e:
    print(f"  [FAIL] TickAnalyzer: {e}")
    traceback.print_exc()
    tick_analyzer = None

# ════════════════════════════════════════════════════════════
# STEP 2: 1m 사전학습 (precompute 패턴)
# ════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("STEP 2: 1m 모델 사전학습 (precompute)")
print("=" * 70)

if market_analyzer:
    look_ahead = 3
    seq_len = market_analyzer.sequence_length

    t0 = time.time()
    print(f"  피처 사전 계산 중... ({len(df_1m)}행)")
    try:
        all_features = market_analyzer.prepare_features(df_1m)
        print(f"  피처 shape: {all_features.shape}")
    except Exception as e:
        print(f"  [FAIL] prepare_features: {e}")
        traceback.print_exc()
        all_features = None

    if all_features is not None:
        close_arr = df_1m['close'].values
        n_rows = len(all_features)
        samples_added = 0

        for i in range(seq_len, n_rows - look_ahead):
            sequence = all_features[i - seq_len:i].copy()
            entry_price = close_arr[i - 1]
            future_price = close_arr[i + look_ahead - 1]
            log_return = math.log(future_price / entry_price)

            direction = 'UP' if log_return > 0 else 'DOWN'
            gross_pct = (future_price / entry_price - 1) * 100
            net_pct = gross_pct - trading_fee_rate * 2 * 100
            net_log_return = log_return - 2 * trading_fee_rate

            sample = TrainingSample(
                features=sequence,
                actual_direction=direction,
                actual_price_change=net_pct,
                timestamp=df_1m.index[i],
                sample_weight=1.0,
                future_log_return=net_log_return
            )
            market_analyzer.training_buffer.append(sample)
            samples_added += 1

        elapsed = time.time() - t0
        print(f"  샘플: {samples_added}개, 버퍼={len(market_analyzer.training_buffer)}, 소요={elapsed:.1f}초")

        if samples_added >= market_analyzer.min_samples_for_training:
            try:
                result = market_analyzer.batch_train(epochs=3)
                status = 'OK' if result.get('status') == 'completed' else 'FAIL'
                print(f"  [{status}] batch_train: loss={result.get('avg_loss', '?')}, "
                      f"acc={result.get('direction_accuracy', 0):.1%}")
            except Exception as e:
                print(f"  [FAIL] batch_train: {e}")
                traceback.print_exc()

# ════════════════════════════════════════════════════════════
# STEP 3: MTF 사전학습 (precompute 패턴)
# ════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("STEP 3: MTF 사전학습 (precompute)")
print("=" * 70)

if mtf_analyzer:
    resample_map = {'1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min'}
    look_ahead_map = {'1m': 3, '3m': 3, '5m': 3, '15m': 3}

    for interval in mtf_intervals:
        rule = resample_map.get(interval)
        if not rule:
            print(f"  [{interval}] [FAIL] resample_map에 키 없음!")
            continue

        try:
            resampled = df_1m.resample(rule).agg({
                'open': 'first', 'high': 'max',
                'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()
        except Exception as e:
            print(f"  [{interval}] [FAIL] 리샘플링: {e}")
            continue

        if len(resampled) < 50:
            print(f"  [{interval}] [SKIP] 데이터 부족 ({len(resampled)}/50)")
            continue

        analyzer = mtf_analyzer.get_interval_analyzer(interval)
        if not analyzer:
            print(f"  [{interval}] [FAIL] analyzer 없음!")
            continue

        seq_len = analyzer.sequence_length
        look_ahead = look_ahead_map.get(interval, 2)

        t0 = time.time()
        print(f"  [{interval}] 피처 사전 계산 중... ({len(resampled)}행)")
        try:
            all_features = analyzer.prepare_features(resampled)
        except Exception as e:
            print(f"  [{interval}] [FAIL] prepare_features: {e}")
            traceback.print_exc()
            continue

        close_arr = resampled['close'].values
        n_rows = len(all_features)
        samples_added = 0

        for i in range(seq_len, n_rows - look_ahead):
            sequence = all_features[i - seq_len:i].copy()
            entry_price = close_arr[i - 1]
            future_price = close_arr[i + look_ahead - 1]
            log_return = math.log(future_price / entry_price)

            direction = 'UP' if log_return > 0 else 'DOWN'
            gross_pct = (future_price / entry_price - 1) * 100
            net_pct = gross_pct - trading_fee_rate * 2 * 100
            net_log_return = log_return - 2 * trading_fee_rate

            sample = TrainingSample(
                features=sequence,
                actual_direction=direction,
                actual_price_change=net_pct,
                timestamp=resampled.index[i],
                sample_weight=1.0,
                future_log_return=net_log_return
            )
            analyzer.training_buffer.append(sample)
            samples_added += 1

        elapsed = time.time() - t0
        print(f"  [{interval}] 샘플={samples_added}, 버퍼={len(analyzer.training_buffer)}, 소요={elapsed:.1f}초")

        if samples_added >= analyzer.min_samples_for_training:
            try:
                result = analyzer.batch_train(epochs=3)
                status = 'OK' if result.get('status') == 'completed' else 'FAIL'
                print(f"  [{interval}] [{status}] loss={result.get('avg_loss', '?')}, "
                      f"acc={result.get('direction_accuracy', 0):.1%}")
            except Exception as e:
                print(f"  [{interval}] [FAIL] batch_train: {e}")
                traceback.print_exc()

# ════════════════════════════════════════════════════════════
# STEP 4: 10s 틱 사전학습 (precompute 패턴)
# ════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("STEP 4: 10s 틱 모델 사전학습 (precompute)")
print("=" * 70)

if tick_analyzer:
    # 합성 10초봉 (처음 120개 1분봉만 사용 — 빠른 테스트)
    df_subset = df_1m.iloc[:120]
    micro_candles = []
    ticks_per_candle = 6

    for i in range(len(df_subset)):
        row = df_subset.iloc[i]
        o, h, l, c, v = row['open'], row['high'], row['low'], row['close'], row['volume']
        base_ts = int(df_subset.index[i].timestamp() * 1000)

        key_prices = [o, l, h, c] if c >= o else [o, h, l, c]
        x_key = np.linspace(0, 1, len(key_prices))
        x_interp = np.linspace(0, 1, ticks_per_candle + 1)
        interp_prices = np.interp(x_interp, x_key, key_prices)

        noise_scale = (h - l) * 0.02 if h > l else 0.0001
        noise = np.random.normal(0, noise_scale, len(interp_prices))
        noise[0] = 0; noise[-1] = 0
        interp_prices = interp_prices + noise
        interp_prices = np.clip(interp_prices, l * 0.999, h * 1.001)
        sub_volume = max(v / ticks_per_candle, 1.0)

        for j in range(ticks_per_candle):
            ts = base_ts + j * 10000
            sub_open = interp_prices[j]
            sub_close = interp_prices[j + 1]
            sub_high = max(sub_open, sub_close) * (1 + abs(np.random.normal(0, 0.0001)))
            sub_low = min(sub_open, sub_close) * (1 - abs(np.random.normal(0, 0.0001)))
            micro_candles.append({
                'timestamp': ts,
                'open': sub_open, 'high': sub_high,
                'low': sub_low, 'close': sub_close,
                'volume': sub_volume + np.random.uniform(-sub_volume * 0.3, sub_volume * 0.3),
            })

    df_10s = pd.DataFrame(micro_candles)
    df_10s['timestamp'] = pd.to_datetime(df_10s['timestamp'], unit='ms')
    df_10s.set_index('timestamp', inplace=True)
    df_10s['volume'] = df_10s['volume'].clip(lower=1.0)

    print(f"  합성 10초봉: {len(df_10s)}개")

    seq_len = tick_analyzer.sequence_length
    tick_look_ahead = 3

    t0 = time.time()
    print(f"  피처 사전 계산 중...")
    try:
        all_features = tick_analyzer.prepare_features(df_10s)
        print(f"  피처 shape: {all_features.shape}")
    except Exception as e:
        print(f"  [FAIL] prepare_features: {e}")
        traceback.print_exc()
        all_features = None

    if all_features is not None:
        close_arr = df_10s['close'].values
        n_rows = len(all_features)
        samples_added = 0

        for i in range(seq_len, n_rows - tick_look_ahead):
            sequence = all_features[i - seq_len:i].copy()
            entry_price = close_arr[i - 1]
            future_price = close_arr[i + tick_look_ahead - 1]
            log_return = math.log(future_price / entry_price)

            direction = 'UP' if log_return > 0 else 'DOWN'
            gross_pct = (future_price / entry_price - 1) * 100
            net_pct = gross_pct - trading_fee_rate * 2 * 100

            tick_analyzer.training_buffer.append({
                'features': sequence,
                'direction': direction,
                'price_change': net_pct,
                'timing_label': 0.5,
                'sample_weight': 1.0,
                'timestamp': df_10s.index[i].timestamp() if hasattr(df_10s.index[i], 'timestamp') else 0,
            })
            samples_added += 1

        elapsed = time.time() - t0
        print(f"  샘플={samples_added}, 버퍼={len(tick_analyzer.training_buffer)}, 소요={elapsed:.1f}초")

        if samples_added >= tick_analyzer.min_samples_for_training:
            try:
                result = tick_analyzer.batch_train(epochs=3)
                status = 'OK' if result.get('status') == 'completed' else 'FAIL'
                print(f"  [{status}] loss={result.get('avg_loss', '?')}, "
                      f"acc={result.get('direction_accuracy', 0):.1%}")
            except Exception as e:
                print(f"  [FAIL] batch_train: {e}")
                traceback.print_exc()

# ════════════════════════════════════════════════════════════
# STEP 5: 모델 저장
# ════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("STEP 5: 모델 저장")
print("=" * 70)

os.makedirs("models", exist_ok=True)

if market_analyzer:
    try:
        path = "models/transformer_BTCUSDT_1m.pt"
        market_analyzer.save_model(path)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"  [OK] 1m: {size:,} bytes")
    except Exception as e:
        print(f"  [FAIL] 1m: {e}")

if mtf_analyzer:
    for interval, analyzer in mtf_analyzer.analyzers.items():
        try:
            path = f"models/transformer_BTCUSDT_{interval}_base.pt"
            analyzer.save_model(path)
            size = os.path.getsize(path) if os.path.exists(path) else 0
            print(f"  [OK] MTF {interval}: {size:,} bytes")
        except Exception as e:
            print(f"  [FAIL] MTF {interval}: {e}")

if tick_analyzer:
    try:
        path = "models/tick_transformer_BTCUSDT_10s.pt"
        tick_analyzer.save_model(path)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"  [OK] 10s: {size:,} bytes")
    except Exception as e:
        print(f"  [FAIL] 10s: {e}")

print()
print("=" * 70)
print("테스트 완료")
print("=" * 70)
