"""AR(1) 전략 백테스트 — 수수료 포함 실전 성과 검증

사용법:
    python backtest_ar1.py eth

전략: "이전 라벨 방향을 반복"
핵심 질문: 80% accuracy에서 실제 R은 얼마인가?
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from data.collectors.historical_collector import HistoricalDataCollector as HistoricalCollector
from ai.feature_builder import TFFeatureBuilder
from ai.sample_buffer import (
    triple_barrier_label_precompute, CANDLE_LOOK_AHEAD,
    ROUND_TRIP_FEE, BASE_SLIPPAGE, SLIPPAGE_VOL_SCALE, SLIPPAGE_MAX,
    compute_trading_costs,
)
from ai.market_analyzer import DIR_MAP

SYMBOL_MAP = {'btc': 'BTCUSDT', 'eth': 'ETHUSDT'}


def collect_and_label(symbol: str):
    """데이터 수집 + Triple Barrier 라벨링 (비용 포함 net_lr 반환)."""
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

    df_3m = df.resample('3min').agg(agg).dropna()
    print(f"  3m 캔들: {len(df_3m)}개")

    look_ahead = CANDLE_LOOK_AHEAD['3m']
    close_3m = df_3m['close'].values
    high_3m = df_3m['high'].values
    low_3m = df_3m['low'].values
    atr_series = TFFeatureBuilder.compute_raw_atr(df_3m)

    # 모든 캔들에 대해 라벨링 (NEUTRAL 포함)
    trades = []  # (timestamp_idx, direction, net_lr, barrier_type, gross_lr, entry_price, atr)

    for i in range(10, len(close_3m) - look_ahead):
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

        # gross log return (비용 차감 전)
        exit_idx = min(i + look_ahead, len(close_3m) - 1)
        gross_lr = np.log(close_3m[exit_idx] / entry_price) if entry_price > 0 else 0.0

        _, total_cost = compute_trading_costs(atr_val, entry_price)

        trades.append({
            'idx': i,
            'direction': direction,
            'net_lr': net_lr,
            'barrier_type': barrier_type,
            'gross_lr': gross_lr,
            'entry_price': entry_price,
            'atr': atr_val,
            'total_cost': total_cost,
            'timing': timing,
            'timestamp': df_3m.index[i] if hasattr(df_3m.index, '__getitem__') else i,
        })

    print(f"  전체 라벨: {len(trades)}개")
    dir_counts = {}
    for t in trades:
        dir_counts[t['direction']] = dir_counts.get(t['direction'], 0) + 1
    print(f"  분포: {dir_counts}")

    return trades, df_3m


def backtest_ar1(trades, skip_neutral=True):
    """AR(1) 전략 백테스트.

    전략: 이전 non-NEUTRAL 라벨 방향으로 진입.
    수익 계산: 방향 맞으면 +|net_lr|, 틀리면 -|net_lr| (비용 이미 포함)

    실제로는 더 정교해야 하지만, 핵심 질문에 답하기 위해:
    - 맞으면: 해당 barrier의 net return 획득
    - 틀리면: 반대 barrier 크기만큼 손실 (근사)
    """
    print("\n" + "=" * 70)
    print("  AR(1) 전략 백테스트")
    print("=" * 70)

    # 전략 시뮬레이션
    equity = [1.0]  # 시작 자본 = 1.0 (비율)
    pnl_list = []  # 개별 거래 PnL
    wins = []
    losses = []

    prev_dir = None  # 이전 non-NEUTRAL 방향
    n_trades = 0
    n_correct = 0
    n_skipped_neutral = 0

    for t in trades:
        actual_dir = t['direction']

        # NEUTRAL은 거래하지 않음 (AR(1) 관점에서도 스킵)
        if actual_dir == 'NEUTRAL':
            n_skipped_neutral += 1
            continue

        if prev_dir is None:
            # 첫 번째 non-NEUTRAL — 진입 불가, 방향만 기록
            prev_dir = actual_dir
            continue

        # AR(1) 예측: 이전 방향
        predicted = prev_dir
        is_correct = (predicted == actual_dir)

        # PnL 계산
        # net_lr은 "올바른 방향으로 진입했을 때의 비용 차감 수익률"
        # 맞으면: +net_lr
        # 틀리면: -net_lr (대칭 근사, TP≠SL이므로 정확하진 않지만 보수적)
        net_lr = abs(t['net_lr'])
        total_cost = t['total_cost']

        if is_correct:
            trade_pnl = net_lr
            wins.append(trade_pnl)
            n_correct += 1
        else:
            trade_pnl = -net_lr
            losses.append(trade_pnl)

        pnl_list.append(trade_pnl)
        new_equity = equity[-1] * (1 + trade_pnl)
        equity.append(new_equity)
        n_trades += 1

        # 방향 업데이트
        prev_dir = actual_dir

    equity = np.array(equity)

    # ── 결과 계산 ──
    print(f"\n  기본 통계:")
    print(f"    총 거래: {n_trades}건 (NEUTRAL 스킵: {n_skipped_neutral}건)")
    print(f"    정확도:  {n_correct/max(n_trades,1):.1%} ({n_correct}/{n_trades})")

    if wins:
        avg_win = np.mean(wins)
        med_win = np.median(wins)
    else:
        avg_win = med_win = 0

    if losses:
        avg_loss = abs(np.mean(losses))
        med_loss = abs(np.median(losses))
    else:
        avg_loss = med_loss = 0

    # R ratio
    r_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

    print(f"\n  수익 구조:")
    print(f"    평균 Win:   {avg_win:.4%} ({len(wins)}건)")
    print(f"    평균 Loss:  {avg_loss:.4%} ({len(losses)}건)")
    print(f"    중앙값 Win:  {med_win:.4%}")
    print(f"    중앙값 Loss: {med_loss:.4%}")
    print(f"    R (avg W/L): {r_ratio:.2f}")

    # Profit Factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    print(f"\n  Profit Factor:")
    print(f"    Gross Profit: {gross_profit:.4%}")
    print(f"    Gross Loss:   {gross_loss:.4%}")
    print(f"    PF = {pf:.2f}")

    # 총 수익률
    total_return = equity[-1] / equity[0] - 1
    print(f"\n  총 수익률: {total_return:.2%}")

    # MDD
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    mdd = drawdown.min()
    mdd_idx = np.argmin(drawdown)
    print(f"  MDD: {mdd:.2%} (거래 #{mdd_idx})")

    # Calmar-like ratio
    if abs(mdd) > 0:
        calmar = total_return / abs(mdd)
        print(f"  Return/MDD: {calmar:.2f}")

    # 연환산 (90일 기준)
    if total_return > -1:
        ann_return = (1 + total_return) ** (365 / 90) - 1
        print(f"  연환산 수익률: {ann_return:.1%} (90일 → 365일)")

    # Expectancy
    expectancy = np.mean(pnl_list) if pnl_list else 0
    print(f"\n  기대값/거래: {expectancy:.4%}")

    # ── Walk-Forward 분할 ──
    print("\n" + "-" * 70)
    print("  Walk-Forward 분석 (4구간)")
    print("-" * 70)

    quarter = n_trades // 4
    print(f"\n  {'구간':>6s}  {'거래':>5s}  {'WinRate':>8s}  {'PF':>6s}  {'Return':>8s}  {'MDD':>8s}  {'AvgW':>8s}  {'AvgL':>8s}")
    print(f"  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for q in range(4):
        s = q * quarter
        e = (q + 1) * quarter if q < 3 else n_trades
        q_pnl = pnl_list[s:e]
        if not q_pnl:
            continue

        q_wins = [p for p in q_pnl if p > 0]
        q_losses = [p for p in q_pnl if p <= 0]
        q_wr = len(q_wins) / len(q_pnl)
        q_gp = sum(q_wins) if q_wins else 0
        q_gl = abs(sum(q_losses)) if q_losses else 0
        q_pf = q_gp / q_gl if q_gl > 0 else float('inf')

        q_equity = np.cumprod([1.0] + [1 + p for p in q_pnl])
        q_return = q_equity[-1] - 1
        q_peak = np.maximum.accumulate(q_equity)
        q_dd = ((q_equity - q_peak) / q_peak).min()

        q_aw = np.mean(q_wins) if q_wins else 0
        q_al = abs(np.mean(q_losses)) if q_losses else 0

        label = f"Q{q+1}"
        print(f"  {label:>6s}  {len(q_pnl):>5d}  {q_wr:>7.1%}  {q_pf:>5.2f}  {q_return:>+7.2%}  {q_dd:>7.2%}  {q_aw:>7.4%}  {q_al:>7.4%}")

    # ── 핵심 비교: "거래 안 하기" vs AR(1) ──
    print("\n" + "-" * 70)
    print("  핵심 판정")
    print("-" * 70)

    print(f"\n  Win Rate: {n_correct/max(n_trades,1):.1%}")
    print(f"  R ratio:  {r_ratio:.2f}")
    print(f"  PF:       {pf:.2f}")
    print(f"  MDD:      {mdd:.2%}")
    print(f"  Total:    {total_return:.2%}")

    if pf > 1.5 and abs(mdd) < 0.15:
        print(f"\n  ✅ AR(1) 전략 실전 가능성 있음")
        print(f"     PF > 1.5 + MDD < 15% — 추가 필터링으로 개선 여지")
    elif pf > 1.0:
        print(f"\n  🟡 AR(1) 양수 기대값이나 리스크 대비 수익 부족")
        print(f"     PF = {pf:.2f} — 거래 비용/슬리피지 민감")
    else:
        print(f"\n  🔴 AR(1) 전략 실전 불가")
        print(f"     PF < 1.0 — 수수료 차감 후 손실")
        print(f"     80% accuracy에도 R이 너무 낮음")

    # ── 비용 민감도 분석 ──
    print("\n" + "-" * 70)
    print("  비용 민감도 분석")
    print("-" * 70)

    for fee_mult, label in [(0, "수수료 0"), (0.5, "수수료 50%"), (1.0, "현재"), (1.5, "수수료 150%"), (2.0, "수수료 2x")]:
        adj_pnl = []
        for i, t_info in enumerate([(w, True) for w in wins] + [(l, False) for l in losses]):
            val, is_win = t_info
            # 대략적 비용 조정 (net_lr에서 비용 역산)
            adj_pnl.append(val if is_win else val)  # 이미 net_lr 기반

        # 간단 근사: 추가 비용만큼 각 거래에서 차감
        extra_cost = ROUND_TRIP_FEE * (fee_mult - 1.0)
        adj_pnl_list = [p - extra_cost for p in pnl_list]
        adj_wins = [p for p in adj_pnl_list if p > 0]
        adj_losses = [p for p in adj_pnl_list if p <= 0]
        adj_gp = sum(adj_wins) if adj_wins else 0
        adj_gl = abs(sum(adj_losses)) if adj_losses else 0
        adj_pf = adj_gp / adj_gl if adj_gl > 0 else 0
        adj_total = np.prod([1 + p for p in adj_pnl_list]) - 1

        print(f"  {label:>10s}: PF={adj_pf:.2f}, Total={adj_total:+.2%}")

    return {
        'n_trades': n_trades,
        'accuracy': n_correct / max(n_trades, 1),
        'pf': pf,
        'mdd': mdd,
        'total_return': total_return,
        'r_ratio': r_ratio,
        'expectancy': expectancy,
    }


def main():
    if len(sys.argv) < 2:
        print("사용법: python backtest_ar1.py [eth|btc]")
        sys.exit(1)

    sym_key = sys.argv[1].lower()
    symbol = SYMBOL_MAP.get(sym_key, sym_key.upper())

    print(f"\n{'='*70}")
    print(f"  AR(1) 전략 백테스트: {symbol}")
    print(f"  전략: 이전 라벨 방향 반복 (비용 포함)")
    print(f"{'='*70}")

    trades, df_3m = collect_and_label(symbol)
    results = backtest_ar1(trades)

    print(f"\n{'='*70}")
    print(f"  백테스트 완료")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
