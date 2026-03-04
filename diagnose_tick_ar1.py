#!/usr/bin/env python3
"""
diagnose_tick_ar1.py — 초단위 스케일 AR(1) 진단

"5~10초 단위 트레이드에서 AR(1) edge가 있는가?"

접근:
  - Binance 최소 해상도: 1m 캔들
  - 1m → 다중 스케일(1m, 3m, 5m, 15m) 리샘플로 AC(1) 변화 추적
  - 1m 기반 triple barrier AC(1) vs 3m 비교
  - 수수료 포함 edge 추정
  - 10s 외삽 가능 여부 판단
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─── 설정 ──────────────────────────────────────────────────────────────────
SYMBOL     = (sys.argv[1].upper() + 'USDT') if len(sys.argv) > 1 else 'ETHUSDT'
DAYS       = 60          # 분석 기간 (더 긴 기간 = 더 안정적)
ROUND_TRIP = 0.0008      # 왕복 수수료 0.08%
BARRIER_K_TP = 1.4
BARRIER_K_SL = 1.2

# ─── 데이터 로드 ───────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  초단위 AR(1) 진단: {SYMBOL}")
print(f"{'='*70}\n")

end_date   = datetime.now()
start_date = end_date - timedelta(days=DAYS)

from data.collectors.historical_collector import HistoricalDataCollector as HC
collector = HC()
df1m = collector.collect_klines(
    SYMBOL, '1m',
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d'),
)
# timestamp 컬럼 → DatetimeIndex
if 'timestamp' in df1m.columns:
    df1m = df1m.set_index('timestamp')
df1m.index = pd.to_datetime(df1m.index)
df1m = df1m[['open','high','low','close','volume']].astype(float).dropna()
print(f"  1m 캔들: {len(df1m):,}개 ({DAYS}일)\n")

# ─── ATR 계산 (1m 기준) ─────────────────────────────────────────────────
def calc_atr(df, period=14):
    h, l, c = df['high'], df['low'], df['close']
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

df1m['atr'] = calc_atr(df1m)

# ─── Triple Barrier 라벨링 (범용) ──────────────────────────────────────
def triple_barrier_labels(df, look_ahead_candles, k_tp=1.4, k_sl=1.2, fee=0.0008):
    """
    df: OHLCV + atr
    look_ahead_candles: 몇 캔들 앞을 봄
    """
    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values
    atr   = df['atr'].values
    n     = len(df)

    directions = []
    net_lrs    = []

    for i in range(n - look_ahead_candles):
        entry = close[i]
        atr_i = atr[i]
        if entry <= 0 or atr_i <= 0:
            directions.append('NEUTRAL')
            net_lrs.append(0.0)
            continue

        tp = entry * (1 + k_tp * atr_i / entry)
        sl = entry * (1 - k_sl * atr_i / entry)

        direction  = 'TIME'
        exit_price = close[i + look_ahead_candles]

        for j in range(i + 1, min(i + look_ahead_candles + 1, n)):
            if high[j] >= tp:
                direction  = 'UP'
                exit_price = tp
                break
            if low[j] <= sl:
                direction  = 'DOWN'
                exit_price = sl
                break

        gross = np.log(exit_price / entry) if entry > 0 else 0.0
        total_cost = fee + max(0.0001, 0.1 * atr_i / entry)

        if direction == 'TIME':
            # time barrier: 비용 허들
            if abs(gross) < total_cost:
                direction = 'NEUTRAL'
        else:
            # TP/SL: 비용 제외 후에도 수익이면 유효
            if abs(gross) <= total_cost:
                direction = 'NEUTRAL'

        net = gross - total_cost if direction == 'UP' else (gross + total_cost if direction == 'DOWN' else 0.0)
        net = net if direction != 'NEUTRAL' else 0.0

        directions.append(direction)
        net_lrs.append(net)

    return directions, net_lrs

# ─── 스케일별 리샘플 & AC(1) ───────────────────────────────────────────
def resample_ohlcv(df1m, rule):
    df = df1m.resample(rule).agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum',
    }).dropna()
    df['atr'] = calc_atr(df)
    return df.dropna()

def calc_ac(series, max_lag=10):
    """pandas Series의 자기상관 (lag 1~max_lag)."""
    return [series.autocorr(lag=i) for i in range(1, max_lag + 1)]

def ar1_stats(directions, net_lrs):
    """AR(1) 전략: 이전 방향 반복 — WinRate, R, PF."""
    labels = [d for d in directions if d != 'NEUTRAL']
    nets   = [n for d, n in zip(directions, net_lrs) if d != 'NEUTRAL']

    if len(labels) < 10:
        return None

    correct = 0
    wins, losses = [], []

    for i in range(1, len(labels)):
        pred = labels[i - 1]          # AR(1): 이전 방향
        actual = labels[i]
        net = abs(nets[i])

        if pred == actual:
            correct += 1
            wins.append(net)
        else:
            losses.append(net)

    acc = correct / (len(labels) - 1)
    avg_w = np.mean(wins) if wins else 0
    avg_l = np.mean(losses) if losses else 1e-9
    pf = (sum(wins)) / (sum(losses) + 1e-9)
    r  = avg_w / (avg_l + 1e-9)

    neutral_pct = directions.count('NEUTRAL') / len(directions) if directions else 0

    return {
        'n_trades': len(labels) - 1,
        'win_rate': acc,
        'avg_w': avg_w * 100,
        'avg_l': avg_l * 100,
        'R': r,
        'PF': pf,
        'neutral_pct': neutral_pct,
    }

# ─── 테스트 1: 다중 스케일 raw return AC(1) ────────────────────────────
print("=" * 70)
print("  TEST 1: 스케일별 Raw Return 자기상관 (AC 1~5)")
print("=" * 70)
print(f"  {'스케일':<10} {'AC(1)':>8} {'AC(2)':>8} {'AC(3)':>8} {'AC(5)':>8}  판정")
print("  " + "-" * 56)

scales = [
    ('1m',  '1min',  1),
    ('3m',  '3min',  1),
    ('5m',  '5min',  1),
    ('15m', '15min', 1),
    ('30m', '30min', 1),
]

for name, rule, _ in scales:
    df = resample_ohlcv(df1m, rule)
    ret = np.log(df['close'] / df['close'].shift(1)).dropna()
    acs = calc_ac(ret, max_lag=5)
    ac1 = acs[0]
    verdict = '🔴 없음' if abs(ac1) < 0.05 else ('🟡 약함' if abs(ac1) < 0.15 else '🟢 있음')
    print(f"  {name:<10} {acs[0]:>8.3f} {acs[1]:>8.3f} {acs[2]:>8.3f} {acs[4]:>8.3f}  {verdict}")

print()
print("  ※ 1m이 최소 해상도. 10s는 1m에서 외삽 판단.")

# ─── 테스트 2: 스케일별 라벨 AC(1) + AR(1) WinRate ──────────────────
print()
print("=" * 70)
print("  TEST 2: 스케일별 Triple Barrier 라벨 AC(1) + AR(1) 성능")
print("=" * 70)

label_scale_configs = [
    ('1m',  '1min',  5),    # 1m × 5 = 5분 look-ahead
    ('1m',  '1min',  15),   # 1m × 15 = 15분 look-ahead
    ('3m',  '3min',  5),    # 3m × 5 = 15분 look-ahead (기존)
    ('5m',  '5min',  3),    # 5m × 3 = 15분 look-ahead
    ('15m', '15min', 1),    # 15m × 1 = 15분 look-ahead
]

print(f"\n  {'스케일':<18} {'라벨AC(1)':>10} {'AR(1)WR':>9} {'R':>6} {'PF':>7} {'NEUTRAL%':>9}")
print("  " + "-" * 64)

for name, rule, la in label_scale_configs:
    df = resample_ohlcv(df1m, rule)
    dirs, nets = triple_barrier_labels(df, la)
    dir_series  = pd.Series([1 if d == 'UP' else (-1 if d == 'DOWN' else 0) for d in dirs])
    non_neutral = dir_series[dir_series != 0]
    ac1_label   = non_neutral.autocorr(lag=1) if len(non_neutral) > 10 else float('nan')

    stats = ar1_stats(dirs, nets)
    if stats:
        scale_label = f"{name} (LA={la}캔들)"
        print(f"  {scale_label:<18} {ac1_label:>10.3f} {stats['win_rate']:>9.1%} "
              f"{stats['R']:>6.2f} {stats['PF']:>7.2f} {stats['neutral_pct']:>9.1%}")
    else:
        print(f"  {name:<18} 샘플 부족")

# ─── 테스트 3: AC(1) 스케일 시각화 (텍스트 바차트) ─────────────────────
print()
print("=" * 70)
print("  TEST 3: 스케일별 라벨 AC(1) 막대 (look-ahead=15분 고정)")
print("=" * 70)
print()

fixed_la_configs = [
    ('1m×15',  '1min',  15),
    ('3m×5',   '3min',  5),
    ('5m×3',   '5min',  3),
    ('15m×1',  '15min', 1),
]

acs_fixed = []
for name, rule, la in fixed_la_configs:
    df = resample_ohlcv(df1m, rule)
    dirs, _ = triple_barrier_labels(df, la)
    dir_series  = pd.Series([1 if d == 'UP' else (-1 if d == 'DOWN' else 0) for d in dirs])
    non_neutral = dir_series[dir_series != 0]
    ac1         = non_neutral.autocorr(lag=1) if len(non_neutral) > 10 else 0.0
    acs_fixed.append((name, ac1))

max_ac = max(abs(ac) for _, ac in acs_fixed) or 1.0
for name, ac in acs_fixed:
    bar_len = int(abs(ac) / max_ac * 40)
    bar = '█' * bar_len
    print(f"  {name:<10} {bar:<40} {ac:+.3f}")

# ─── 테스트 4: 1m 단위 Walk-Forward AR(1) ─────────────────────────────
print()
print("=" * 70)
print("  TEST 4: 1m 스케일 Walk-Forward AR(1) (수수료 포함)")
print("          — 5분 look-ahead (5캔들)")
print("=" * 70)

df1m_fresh = resample_ohlcv(df1m, '1min')
dirs_1m, nets_1m = triple_barrier_labels(df1m_fresh, look_ahead_candles=5, fee=ROUND_TRIP)
n = len(dirs_1m)
fold_size = n // 5

print(f"\n  {'구간':<8} {'거래':>6} {'WinRate':>9} {'PF':>7} {'R':>6}")
print("  " + "-" * 42)

fold_results = []
for fold in range(5):
    s = fold * fold_size
    e = s + fold_size
    fd = dirs_1m[s:e]
    fn = nets_1m[s:e]
    st = ar1_stats(fd, fn)
    if st:
        fold_results.append(st)
        print(f"  Fold {fold+1:<3} {st['n_trades']:>6} {st['win_rate']:>9.1%} "
              f"{st['PF']:>7.2f} {st['R']:>6.2f}")

# ─── 테스트 5: 10s 외삽 판단 ──────────────────────────────────────────
print()
print("=" * 70)
print("  TEST 5: 10초 스케일 실현 가능성 판단")
print("=" * 70)

# 1m raw return AC(1)
df_1m = resample_ohlcv(df1m, '1min')
ret_1m = np.log(df_1m['close'] / df_1m['close'].shift(1)).dropna()
ac1_1m_raw = ret_1m.autocorr(lag=1)

# 1m 라벨 AC(1) (5분 look-ahead)
dirs_1m_5, _ = triple_barrier_labels(df_1m, 5, fee=ROUND_TRIP)
dir_s = pd.Series([1 if d == 'UP' else (-1 if d == 'DOWN' else 0) for d in dirs_1m_5])
ac1_1m_label = dir_s[dir_s != 0].autocorr(lag=1)

# 3m 기존 수치 (참조)
df_3m = resample_ohlcv(df1m, '3min')
dirs_3m, _ = triple_barrier_labels(df_3m, 5, fee=ROUND_TRIP)
dir_s3 = pd.Series([1 if d == 'UP' else (-1 if d == 'DOWN' else 0) for d in dirs_3m])
ac1_3m_label = dir_s3[dir_s3 != 0].autocorr(lag=1)

print(f"""
  관측값:
    1m Raw Return  AC(1) = {ac1_1m_raw:+.3f}
    1m 라벨 (5분)  AC(1) = {ac1_1m_label:+.3f}
    3m 라벨 (15분) AC(1) = {ac1_3m_label:+.3f}  ← 백테스트 확인값 (0.60)

  10초 추론:
    1m보다 6× 짧은 스케일 → noise 비중 증가, AC 감소 예상
    경험칙: 스케일 축소 시 AC(1)은 sqrt 비율로 감소
    추정 10s AC(1) ≈ {ac1_1m_label * (10/60)**0.5:+.3f}  (상한 추정, 실측 필요)

  수수료 현실:
    3m 거래: 하루 ~{24*60/3:.0f}회 → 일 수수료 ≈ {24*60/3*ROUND_TRIP*100:.2f}%
    1m 거래: 하루 ~{24*60:.0f}회  → 일 수수료 ≈ {24*60*ROUND_TRIP*100:.2f}%
    10s 거래: 하루 ~{24*3600/10:.0f}회 → 일 수수료 ≈ {24*3600/10*ROUND_TRIP*100:.2f}%
""")

# ─── 최종 판정 ────────────────────────────────────────────────────────
print("=" * 70)
print("  최종 판정")
print("=" * 70)

# 1m fold 평균
if fold_results:
    avg_wr = np.mean([r['win_rate'] for r in fold_results])
    avg_pf = np.mean([r['PF'] for r in fold_results])
    avg_r  = np.mean([r['R'] for r in fold_results])
else:
    avg_wr = avg_pf = avg_r = 0

print(f"""
  [3m 스케일 — 확인됨]
    AC(1) = {ac1_3m_label:.3f}, WinRate = 80%, PF = 4~10  ✅

  [1m 스케일 — 현재 테스트]
    AC(1) = {ac1_1m_label:.3f}, WinRate = {avg_wr:.1%}, PF = {avg_pf:.2f}, R = {avg_r:.2f}
""")

if ac1_1m_label > 0.40:
    print("  🟢 1m 스케일도 AR(1) edge 확인됨")
    print("     → 1m 기반 빠른 진입 전략 가능성 있음")
    print("     → 단, 수수료 증가로 per-trade expectancy 감소 주의")
elif ac1_1m_label > 0.20:
    print("  🟡 1m 스케일 AC 약화 (3m 대비)")
    print("     → 수수료 감안 시 edge 불확실")
    print("     → 3m 유지 권장")
else:
    print("  🔴 1m 스케일 AC(1) 거의 없음")
    print("     → 10s는 더 낮을 것으로 추정")
    print("     → 3m AR(1) 전략 유지 강력 권장")

if ac1_1m_raw < 0.05:
    print()
    print("  ⚠️  1m raw return AC도 낮음 → 10s에서 mean-reversion(역추세) 가능성")
    print("     → 10s에서는 AR(1)이 아닌 반대 방향 전략이 더 효과적일 수도 있음")

print()
print("=" * 70)
print("  진단 완료")
print("=" * 70)
