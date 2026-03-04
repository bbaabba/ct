"""피처 신호 종합 진단 스크립트

사용법:
    python diagnose_features.py eth
    python diagnose_features.py btc

진단 내용:
    1) 단순 피처 신호 테스트 (last-step 33개 → Logistic)
    2) 라벨 자기상관 분석 (lag-1~10)
    3) 피처 중요도 (RandomForest top-10)
    4) 스케일별 분리 테스트 (Macro/Mid/Micro 각각)
    5) 시퀀스 통계 피처 테스트 (mean/std/last/trend)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from config.config import Config
from data.collectors.historical_collector import HistoricalDataCollector as HistoricalCollector
from ai.feature_builder import TFFeatureBuilder
from ai.sample_buffer import triple_barrier_label_precompute, CANDLE_LOOK_AHEAD
from ai.market_analyzer import DIR_MAP

SYMBOL_MAP = {'btc': 'BTCUSDT', 'eth': 'ETHUSDT'}

# 시퀀스 길이 (unified_analyzer 기본값)
MACRO_SEQ = 60
MID_SEQ = 40
MICRO_SEQ = 30


def collect_data(symbol: str):
    """SFT와 동일한 데이터 수집 + 리샘플."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    print(f"\n데이터 수집: {symbol} ({start_date} ~ {end_date})")
    collector = HistoricalCollector()
    df = collector.collect_klines(
        symbol=symbol, interval='1m',
        start_date=start_date, end_date=end_date, save_to_csv=True
    )
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    keep = ['open', 'high', 'low', 'close', 'volume', 'trades', 'taker_buy_base']
    df = df[[c for c in keep if c in df.columns]].copy()

    agg = {'open': 'first', 'high': 'max', 'low': 'min',
           'close': 'last', 'volume': 'sum'}
    if 'trades' in df.columns:
        agg['trades'] = 'sum'
    if 'taker_buy_base' in df.columns:
        agg['taker_buy_base'] = 'sum'

    df_15m = df.resample('15min').agg(agg).dropna()
    df_3m = df.resample('3min').agg(agg).dropna()
    df_1m = df

    print(f"  15m: {len(df_15m)}캔들, 3m: {len(df_3m)}캔들, 1m: {len(df_1m)}캔들")
    return df_15m, df_3m, df_1m


def build_samples(df_15m, df_3m, df_1m):
    """SFT와 동일한 피처 추출 + Triple Barrier 라벨링."""
    fb = TFFeatureBuilder()
    macro_feat = fb.prepare_features(df_15m, '15m')
    mid_feat = fb.prepare_features(df_3m, '3m')
    micro_feat = fb.prepare_features(df_1m, '1m')

    look_ahead = CANDLE_LOOK_AHEAD['3m']
    close_3m = df_3m['close'].values
    high_3m = df_3m['high'].values
    low_3m = df_3m['low'].values
    atr_series = TFFeatureBuilder.compute_raw_atr(df_3m)

    macro_list, mid_list, micro_list, labels = [], [], [], []
    macro_step = max(1, len(macro_feat) // len(mid_feat))
    micro_step = max(1, len(micro_feat) // len(mid_feat))

    n_neutral = 0
    for i in range(max(MID_SEQ, 10), len(mid_feat) - look_ahead):
        if i + look_ahead >= len(close_3m):
            break

        entry_price = close_3m[i]
        atr_val = float(atr_series.iloc[i]) if i < len(atr_series) else 0.0

        direction, net_lr, barrier_type, timing = triple_barrier_label_precompute(
            entry_price=entry_price, atr=atr_val,
            high_arr=high_3m[i + 1:i + 1 + look_ahead],
            low_arr=low_3m[i + 1:i + 1 + look_ahead],
            close_arr=close_3m[i + 1:i + 1 + look_ahead],
        )

        if direction == 'NEUTRAL':
            n_neutral += 1
            continue

        macro_idx = min(int(i * macro_step), len(macro_feat) - 1)
        micro_idx = min(int(i * micro_step), len(micro_feat) - 1)

        if (macro_idx < MACRO_SEQ or i < MID_SEQ or micro_idx < MICRO_SEQ):
            continue

        macro_list.append(macro_feat[macro_idx - MACRO_SEQ:macro_idx])
        mid_list.append(mid_feat[i - MID_SEQ:i])
        micro_list.append(micro_feat[micro_idx - MICRO_SEQ:micro_idx])
        labels.append(DIR_MAP[direction])

    macro_arr = np.array(macro_list)   # (n, 60, 11)
    mid_arr = np.array(mid_list)       # (n, 40, 11)
    micro_arr = np.array(micro_list)   # (n, 30, 11)
    y = np.array(labels)

    n_total = len(y) + n_neutral
    print(f"\n라벨링 결과:")
    print(f"  전체: {n_total}개 (NEUTRAL 제외: {n_neutral}개, {n_neutral/max(n_total,1):.1%})")
    print(f"  학습 샘플: {len(y)}개")
    print(f"  UP: {(y==1).sum()} ({(y==1).mean():.1%}), DOWN: {(y==0).sum()} ({(y==0).mean():.1%})")

    return macro_arr, mid_arr, micro_arr, y


def run_diagnosis(macro_arr, mid_arr, micro_arr, y):
    """종합 진단 실행."""
    n = len(y)
    n_val = max(1, int(n * 0.2))
    n_gap = max(1, int(n * 0.1))
    n_train = n - n_val - n_gap

    ti = np.arange(n_train)
    vi = np.arange(n_train + n_gap, n)
    y_train, y_val = y[ti], y[vi]

    print(f"\n데이터 분할: train={n_train}, gap={n_gap}, val={n_val}")
    print(f"  Train: UP={( y_train==1).sum()}, DOWN={(y_train==0).sum()}")
    print(f"  Val:   UP={(y_val==1).sum()}, DOWN={(y_val==0).sum()}")

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        print("ERROR: 단일 클래스 — 진단 불가")
        return

    FEATURE_NAMES = [
        'M_return', 'M_atr_ratio', 'M_bb_pos', 'M_vol_delta',
        'M_vol_change', 'M_mom_slope', 'M_htf_trend', 'M_vol_cycle',
        'M_regime', 'M_hl_struct', 'M_htf_bias',
        'm_return', 'm_atr_ratio', 'm_bb_pos', 'm_vol_delta',
        'm_vol_change', 'm_mom_slope', 'm_di_diff', 'm_taker_buy',
        'm_trade_int', 'm_vwap_dev', 'm_htf_bias',
        'u_return', 'u_atr_ratio', 'u_bb_pos', 'u_vol_delta',
        'u_vol_change', 'u_mom_slope', 'u_di_diff', 'u_taker_buy',
        'u_trade_int', 'u_vwap_dev', 'u_htf_bias',
    ]

    # ═══════════════════════════════════════════════════
    # TEST 1: Last-step 단순 피처 (33개)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  [1/5] 단순 피처 신호 테스트 (last-step 33개)")
    print("=" * 70)

    X_last = np.concatenate([
        macro_arr[:, -1, :],
        mid_arr[:, -1, :],
        micro_arr[:, -1, :],
    ], axis=1)

    X_tr, X_va = X_last[ti], X_last[vi]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    lr.fit(X_tr_s, y_train)
    lr_train = lr.score(X_tr_s, y_train)
    lr_val = lr.score(X_va_s, y_val)
    lr_proba = lr.predict_proba(X_va_s)[:, 1]
    lr_auc = roc_auc_score(y_val, lr_proba)

    print(f"  Logistic: train={lr_train:.1%}, val={lr_val:.1%}, AUC={lr_auc:.3f}")
    print(f"  Prob dist: mean={np.mean(lr_proba):.3f}, std={np.std(lr_proba):.3f}")

    if lr_val < 0.52:
        print("  → ⚠️ 신호 없음 (val < 52%)")
    elif lr_val >= 0.55:
        print("  → ✅ 선형 신호 존재 (val ≥ 55%)")
    else:
        print("  → 🟡 약한 신호 (52-55%)")

    # ═══════════════════════════════════════════════════
    # TEST 2: 스케일별 분리 테스트
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  [2/5] 스케일별 분리 테스트")
    print("=" * 70)

    for name, arr in [('Macro(15m)', macro_arr), ('Mid(3m)', mid_arr), ('Micro(1m)', micro_arr)]:
        X_sc = arr[:, -1, :]
        X_sc_tr = StandardScaler().fit_transform(X_sc[ti])
        X_sc_va = StandardScaler().fit_transform(X_sc[vi])  # note: should use same scaler
        sc = StandardScaler()
        X_sc_tr = sc.fit_transform(X_sc[ti])
        X_sc_va = sc.transform(X_sc[vi])
        lr_sc = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
        lr_sc.fit(X_sc_tr, y_train)
        sc_val = lr_sc.score(X_sc_va, y_val)
        try:
            sc_auc = roc_auc_score(y_val, lr_sc.predict_proba(X_sc_va)[:, 1])
        except Exception:
            sc_auc = 0.0
        print(f"  {name:12s}: val={sc_val:.1%}, AUC={sc_auc:.3f}")

    # ═══════════════════════════════════════════════════
    # TEST 3: 시퀀스 통계 피처 (mean/std/last/trend 추출)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  [3/5] 시퀀스 통계 피처 (mean+std+last+trend = 44개/스케일)")
    print("=" * 70)

    def extract_seq_stats(arr):
        """(n, seq, 11) → (n, 44): mean, std, last, trend per feature."""
        n_samples = arr.shape[0]
        n_feat = arr.shape[2]
        stats = np.zeros((n_samples, n_feat * 4))
        for f in range(n_feat):
            col = arr[:, :, f]  # (n, seq)
            stats[:, f * 4 + 0] = col.mean(axis=1)
            stats[:, f * 4 + 1] = col.std(axis=1)
            stats[:, f * 4 + 2] = col[:, -1]  # last value
            # trend: last_5_mean - first_5_mean
            stats[:, f * 4 + 3] = col[:, -5:].mean(axis=1) - col[:, :5].mean(axis=1)
        return stats

    X_seq = np.concatenate([
        extract_seq_stats(macro_arr),
        extract_seq_stats(mid_arr),
        extract_seq_stats(micro_arr),
    ], axis=1)

    sc_seq = StandardScaler()
    X_seq_tr = sc_seq.fit_transform(X_seq[ti])
    X_seq_va = sc_seq.transform(X_seq[vi])

    lr_seq = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    lr_seq.fit(X_seq_tr, y_train)
    seq_train = lr_seq.score(X_seq_tr, y_train)
    seq_val = lr_seq.score(X_seq_va, y_val)
    seq_proba = lr_seq.predict_proba(X_seq_va)[:, 1]
    seq_auc = roc_auc_score(y_val, seq_proba)

    print(f"  Logistic(seq_stats): train={seq_train:.1%}, val={seq_val:.1%}, AUC={seq_auc:.3f}")

    if seq_val > lr_val + 0.02:
        print(f"  → 시퀀스 패턴에 추가 신호 있음 (+{seq_val - lr_val:.1%})")
    else:
        print(f"  → 시퀀스 통계 추가 효과 미미 (last-step과 유사)")

    # ═══════════════════════════════════════════════════
    # TEST 4: 라벨 자기상관
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  [4/5] 라벨 자기상관 분석")
    print("=" * 70)

    y_f = y.astype(float)
    y_centered = y_f - y_f.mean()
    var = np.sum(y_centered ** 2)

    threshold = 2.0 / np.sqrt(n)

    print(f"  유의 기준: |ac| > {threshold:.4f} (2/√n, n={n})")
    print()

    significant = []
    for lag in [1, 2, 3, 5, 10, 20, 50]:
        if lag < n and var > 0:
            ac = np.sum(y_centered[:-lag] * y_centered[lag:]) / var
            marker = " ★" if abs(ac) > threshold else ""
            print(f"  lag-{lag:2d}: {ac:+.4f}{marker}")
            if abs(ac) > threshold:
                significant.append((lag, ac))

    print()
    if not significant:
        print("  → ⚠️ 모든 lag에서 자기상관 ≈ 0 — 라벨이 사실상 random")
    else:
        print(f"  → 유의한 자기상관 발견: {len(significant)}개 lag")

    # 방향전환 빈도 (runs test)
    up_ratio = y_f.mean()
    changes = np.sum(np.diff(y_f) != 0)
    expected = 2 * up_ratio * (1 - up_ratio) * (n - 1)
    print(f"\n  라벨 분포: UP={up_ratio:.1%}, DOWN={1-up_ratio:.1%}")
    print(f"  방향전환: {changes}회 (random 기대={expected:.0f}, 비율={changes/expected:.2f})")

    if changes / expected > 1.05:
        print("  → 전환 과다 — 연속 패턴 약함 (mean-reversion?)")
    elif changes / expected < 0.95:
        print("  → 전환 부족 — 연속 패턴 있음 (momentum?)")
    else:
        print("  → random에 근접")

    # ═══════════════════════════════════════════════════
    # TEST 5: RandomForest 피처 중요도
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  [5/5] RandomForest 피처 중요도 (last-step 33개)")
    print("=" * 70)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_tr_s, y_train)
    rf_train = rf.score(X_tr_s, y_train)
    rf_val = rf.score(X_va_s, y_val)
    rf_proba = rf.predict_proba(X_va_s)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_proba)

    print(f"  RF: train={rf_train:.1%}, val={rf_val:.1%}, AUC={rf_auc:.3f}")
    print()

    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1]
    print(f"  {'Rank':>4s}  {'Feature':15s}  {'Importance':>10s}  {'Cum%':>6s}")
    print(f"  {'─'*4}  {'─'*15}  {'─'*10}  {'─'*6}")
    cum = 0
    for rank, idx in enumerate(top_idx, 1):
        fname = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feat_{idx}"
        imp = importances[idx]
        cum += imp
        print(f"  {rank:4d}  {fname:15s}  {imp:10.4f}  {cum:5.1%}")
        if rank >= 15:
            break

    # ═══════════════════════════════════════════════════
    # TEST 6: AR(1) Baseline — "이전 라벨 반복" 전략
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  [6/8] AR(1) Baseline (이전 라벨 = 다음 예측)")
    print("=" * 70)

    # 전체 데이터에서 AR(1) 정확도
    ar1_correct = np.sum(y[1:] == y[:-1])
    ar1_total = len(y) - 1
    ar1_acc = ar1_correct / ar1_total
    print(f"  전체 AR(1) acc: {ar1_acc:.1%} ({ar1_correct}/{ar1_total})")

    # Val 구간에서만
    y_va = y[vi]
    if len(y_va) > 1:
        ar1_va_correct = np.sum(y_va[1:] == y_va[:-1])
        ar1_va_acc = ar1_va_correct / (len(y_va) - 1)
        print(f"  Val AR(1) acc:  {ar1_va_acc:.1%}")
    else:
        ar1_va_acc = 0.5

    print(f"\n  해석: AR(1)={ar1_acc:.1%}면 단순 '이전 방향 반복'이")
    print(f"        모든 딥러닝 모델을 이기는 중")

    # ═══════════════════════════════════════════════════
    # TEST 7: Rolling Autocorrelation — 레짐별 안정성
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  [7/8] Rolling Autocorrelation (레짐별 안정성)")
    print("=" * 70)

    window = 500  # 500샘플 윈도우
    n_windows = max(1, (n - window) // (window // 2))  # 50% overlap
    rolling_ac1 = []
    rolling_ar1_acc = []

    print(f"  Window={window}, Stride={window//2}, N_windows={n_windows}")
    print()
    print(f"  {'Window':>8s}  {'Samples':>8s}  {'AC(1)':>8s}  {'AR(1)':>8s}  {'UP%':>6s}  {'판정'}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*10}")

    for w in range(n_windows):
        start = w * (window // 2)
        end = min(start + window, n)
        if end - start < 100:
            break
        y_w = y[start:end].astype(float)
        y_w_c = y_w - y_w.mean()
        var_w = np.sum(y_w_c ** 2)

        if var_w > 0:
            ac1_w = np.sum(y_w_c[:-1] * y_w_c[1:]) / var_w
        else:
            ac1_w = 0.0

        ar1_w = np.sum(y[start+1:end] == y[start:end-1]) / (end - start - 1)
        up_pct = y_w.mean()

        rolling_ac1.append(ac1_w)
        rolling_ar1_acc.append(ar1_w)

        # 판정
        if ac1_w > 0.4:
            verdict = "강한 momentum"
        elif ac1_w > 0.1:
            verdict = "약한 momentum"
        elif ac1_w < -0.1:
            verdict = "mean-reversion"
        else:
            verdict = "random"

        print(f"  {start:>8d}  {end-start:>8d}  {ac1_w:>+8.3f}  {ar1_w:>7.1%}  {up_pct:>5.1%}  {verdict}")

    rolling_ac1 = np.array(rolling_ac1)
    rolling_ar1_acc = np.array(rolling_ar1_acc)

    print(f"\n  AC(1) 통계: mean={rolling_ac1.mean():+.3f}, "
          f"std={rolling_ac1.std():.3f}, "
          f"min={rolling_ac1.min():+.3f}, max={rolling_ac1.max():+.3f}")
    print(f"  AR(1) 통계: mean={rolling_ar1_acc.mean():.1%}, "
          f"std={rolling_ar1_acc.std():.1%}, "
          f"min={rolling_ar1_acc.min():.1%}, max={rolling_ar1_acc.max():.1%}")

    ac1_stable = rolling_ac1.std() < 0.15
    ac1_always_positive = rolling_ac1.min() > 0.05
    if ac1_stable and ac1_always_positive:
        print("  → ✅ 자기상관 안정적 — 모든 윈도우에서 양수 유지")
    elif ac1_always_positive:
        print("  → 🟡 항상 양수이나 변동 큼 — 레짐 의존적")
    else:
        print("  → ⚠️ 자기상관 불안정 — 일부 구간에서 소실/반전")

    # ═══════════════════════════════════════════════════
    # TEST 8: Walk-Forward Lag Feature 테스트
    #   피처 + lag 조합이 미래에도 작동하는지 검증
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  [8/8] Walk-Forward Lag Feature 테스트")
    print("=" * 70)

    # Lag 피처 생성: y[i-1], y[i-2], y[i-3], running_mean(5), running_mean(10)
    def make_lag_features(y_arr):
        """라벨 시퀀스 → lag 피처 매트릭스."""
        n_s = len(y_arr)
        lag_feat = np.zeros((n_s, 5))
        for i in range(n_s):
            lag_feat[i, 0] = y_arr[i - 1] if i >= 1 else 0.5
            lag_feat[i, 1] = y_arr[i - 2] if i >= 2 else 0.5
            lag_feat[i, 2] = y_arr[i - 3] if i >= 3 else 0.5
            lag_feat[i, 3] = y_arr[max(0, i-5):i].mean() if i >= 1 else 0.5
            lag_feat[i, 4] = y_arr[max(0, i-10):i].mean() if i >= 1 else 0.5
        return lag_feat

    # Walk-forward: 5-fold time-series split
    wf_size = n // 6  # 각 fold 크기
    wf_results = []

    print(f"  Fold 크기: ~{wf_size}샘플, 5-fold time-series CV")
    print()
    print(f"  {'Fold':>4s}  {'Train':>10s}  {'Val':>10s}  "
          f"{'LR(feat)':>9s}  {'LR(lag)':>8s}  {'LR(feat+lag)':>12s}  {'AR(1)':>6s}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  "
          f"{'─'*9}  {'─'*8}  {'─'*12}  {'─'*6}")

    for fold in range(5):
        tr_end = (fold + 1) * wf_size
        va_start = tr_end
        va_end = min(tr_end + wf_size, n)
        if va_end - va_start < 50:
            break

        # 피처 only
        X_f_tr = X_last[:tr_end]
        X_f_va = X_last[va_start:va_end]
        y_f_tr = y[:tr_end]
        y_f_va = y[va_start:va_end]

        sc_wf = StandardScaler()
        X_f_tr_s = sc_wf.fit_transform(X_f_tr)
        X_f_va_s = sc_wf.transform(X_f_va)

        lr_f = LogisticRegression(max_iter=500, solver='lbfgs', C=1.0)
        lr_f.fit(X_f_tr_s, y_f_tr)
        feat_val = lr_f.score(X_f_va_s, y_f_va)

        # Lag only
        lag_tr = make_lag_features(y[:tr_end])
        lag_va = make_lag_features(y[:va_end])[va_start:va_end]
        lr_l = LogisticRegression(max_iter=500, solver='lbfgs', C=1.0)
        lr_l.fit(lag_tr, y_f_tr)
        lag_val = lr_l.score(lag_va, y_f_va)

        # 피처 + lag 결합
        X_combo_tr = np.concatenate([X_f_tr_s, lag_tr], axis=1)
        X_combo_va = np.concatenate([X_f_va_s, lag_va], axis=1)
        lr_c = LogisticRegression(max_iter=500, solver='lbfgs', C=1.0)
        lr_c.fit(X_combo_tr, y_f_tr)
        combo_val = lr_c.score(X_combo_va, y_f_va)

        # AR(1)
        ar1_wf = np.sum(y_f_va[1:] == y_f_va[:-1]) / (len(y_f_va) - 1)

        wf_results.append({
            'feat': feat_val, 'lag': lag_val,
            'combo': combo_val, 'ar1': ar1_wf,
        })

        print(f"  {fold+1:4d}  {f'0-{tr_end}':>10s}  {f'{va_start}-{va_end}':>10s}  "
              f"{feat_val:8.1%}  {lag_val:7.1%}  {combo_val:11.1%}  {ar1_wf:5.1%}")

    # 평균
    if wf_results:
        avg_feat = np.mean([r['feat'] for r in wf_results])
        avg_lag = np.mean([r['lag'] for r in wf_results])
        avg_combo = np.mean([r['combo'] for r in wf_results])
        avg_ar1 = np.mean([r['ar1'] for r in wf_results])

        print(f"\n  {'평균':>4s}  {'':>10s}  {'':>10s}  "
              f"{avg_feat:8.1%}  {avg_lag:7.1%}  {avg_combo:11.1%}  {avg_ar1:5.1%}")

        print(f"\n  분석:")
        if avg_lag > avg_feat + 0.02:
            print(f"    Lag 피처({avg_lag:.1%}) > 기존 피처({avg_feat:.1%})")
            print(f"    → 자기상관이 기존 피처보다 강력한 예측 변수")
        if avg_combo > avg_lag + 0.01:
            print(f"    Combo({avg_combo:.1%}) > Lag만({avg_lag:.1%})")
            print(f"    → 기존 피처가 lag에 추가 정보 제공")
        if avg_ar1 > avg_combo:
            print(f"    AR(1)({avg_ar1:.1%}) ≥ Combo({avg_combo:.1%})")
            print(f"    → 단순 반복이 최강 — 복잡한 모델 불필요")

        # 안정성 체크
        lag_vals = [r['lag'] for r in wf_results]
        lag_std = np.std(lag_vals)
        if lag_std > 0.05:
            print(f"\n  ⚠️ Lag 피처 성능 변동 큼 (std={lag_std:.1%})")
            print(f"     → 자기상관이 레짐에 따라 불안정")
        else:
            print(f"\n  ✅ Lag 피처 성능 안정적 (std={lag_std:.1%})")

    # ═══════════════════════════════════════════════════
    # 종합 판정
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  종합 판정")
    print("=" * 70)

    results = {
        'lr_last_val': lr_val, 'lr_last_auc': lr_auc,
        'lr_seq_val': seq_val, 'lr_seq_auc': seq_auc,
        'rf_val': rf_val, 'rf_auc': rf_auc,
        'ar1_acc': ar1_acc,
        'has_autocorr': len(significant) > 0,
    }

    print(f"\n  {'테스트':25s}  {'Val Acc':>8s}  {'AUC':>6s}")
    print(f"  {'─'*25}  {'─'*8}  {'─'*6}")
    print(f"  {'AR(1) 이전라벨반복':25s}  {ar1_acc:7.1%}  {'  -':>6s}")
    if wf_results:
        print(f"  {'WF Lag피처 (평균)':25s}  {avg_lag:7.1%}  {'  -':>6s}")
        print(f"  {'WF Feat+Lag (평균)':25s}  {avg_combo:7.1%}  {'  -':>6s}")
    print(f"  {'Logistic (last-step)':25s}  {lr_val:7.1%}  {lr_auc:.3f}")
    print(f"  {'Logistic (seq-stats)':25s}  {seq_val:7.1%}  {seq_auc:.3f}")
    print(f"  {'RandomForest (last-step)':25s}  {rf_val:7.1%}  {rf_auc:.3f}")
    print(f"  {'Random baseline':25s}  {'50.0%':>8s}  {'0.500':>6s}")

    best_val = max(lr_val, seq_val, rf_val)
    best_auc = max(lr_auc, seq_auc, rf_auc)

    print(f"\n  핵심 비교:")
    print(f"    AR(1)={ar1_acc:.1%} vs 최고 피처 모델={best_val:.1%}")
    if ar1_acc > best_val + 0.05:
        print(f"    → 🔴 AR(1)이 모든 피처 모델을 압도 — 피처 의미 없음")
        print(f"    → Markov/AR 기반 접근이 가장 효율적")
    elif ar1_acc > best_val:
        print(f"    → 🟡 AR(1) > 피처 — lag 정보가 핵심 변수")

    if wf_results and avg_combo > avg_ar1 + 0.01:
        print(f"    → 🟢 Feat+Lag({avg_combo:.1%}) > AR(1)({avg_ar1:.1%})")
        print(f"       피처가 lag에 추가 가치 제공 — 결합 모델 유효")

    # 자기상관 안정성 경고
    if not ac1_stable:
        print(f"\n  ⚠️ 경고: 자기상관이 레짐에 따라 불안정")
        print(f"     AC(1) std={rolling_ac1.std():.3f}, range=[{rolling_ac1.min():+.3f}, {rolling_ac1.max():+.3f}]")
        print(f"     → walk-forward에서 성능 변동 예상")

    return results


def main():
    if len(sys.argv) < 2:
        print("사용법: python diagnose_features.py [eth|btc]")
        sys.exit(1)

    sym_key = sys.argv[1].lower()
    symbol = SYMBOL_MAP.get(sym_key, sym_key.upper())

    print(f"\n{'='*70}")
    print(f"  🔬 피처 신호 종합 진단: {symbol}")
    print(f"{'='*70}")

    df_15m, df_3m, df_1m = collect_data(symbol)
    macro_arr, mid_arr, micro_arr, y = build_samples(df_15m, df_3m, df_1m)
    results = run_diagnosis(macro_arr, mid_arr, micro_arr, y)

    print(f"\n{'='*70}")
    print(f"  진단 완료")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
