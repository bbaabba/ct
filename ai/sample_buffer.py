"""링 버퍼 + Triple Barrier 지연 라벨링 — TF별 학습 샘플 수집 파이프라인"""
import time
import bisect
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from utils.logger import setup_logger

logger = setup_logger(__name__)


# ── TF별 캔들 Look-ahead ──────────────────────────────────────
CANDLE_LOOK_AHEAD = {
    '1m':  15,   # 15캔들 × 1min  = 15분
    '3m':  5,    #  5캔들 × 3min  = 15분
    '5m':  5,    #  5캔들 × 5min  = 25분
    '15m': 5,    #  5캔들 × 15min = 75분
}
LOOK_AHEAD_SECONDS = {
    '1m':  CANDLE_LOOK_AHEAD['1m']  * 60,     # 900s  = 15분
    '3m':  CANDLE_LOOK_AHEAD['3m']  * 180,    # 900s  = 15분
    '5m':  CANDLE_LOOK_AHEAD['5m']  * 300,    # 1500s = 25분
    '15m': CANDLE_LOOK_AHEAD['15m'] * 900,    # 4500s = 75분
}

# ── Triple Barrier 파라미터 ───────────────────────────────────
BARRIER_K_TP = 1.4    # α — TP = max(floor, α × ATR/price)  (1.4:1.2 = 완만한 비대칭)
BARRIER_K_SL = 1.2    # β — SL = max(floor, β × ATR/price)

# 최소 배리어 플로어 (최소 수익성 허들 #1)
# - 배리어 자체가 거래 비용 미만이면 안 됨 → 구조적 보호
# - 직접 비용 차감(compute_net_label)이 추가 보호층(허들 #2)으로 동작
# - TP/SL 비율을 BARRIER_K 비율과 동일하게 유지 (floor 지배 시에도 비대칭 보존)
BARRIER_MIN_SL_PCT = 0.0012   # 0.12% — SL 최소 배리어
BARRIER_MIN_TP_PCT = BARRIER_MIN_SL_PCT * (BARRIER_K_TP / BARRIER_K_SL)  # ≈0.14% — K비율 연동

# MAE 품질 필터 임계값
# UP label: TP 히트 전 SL 방향 MAE / SL 거리 > 임계값 → NEUTRAL (버텨서 겨우 TP)
# DOWN label: SL 히트 전 TP 방향 MAE / TP 거리 > 임계값 → NEUTRAL (우연히 SL)
MAE_QUALITY_THRESHOLD = 0.80

# TIME barrier NEUTRAL 허들 (최소 수익성 허들 #3)
# - compute_net_label에서 max(FEE_DEADZONE, ATR허들, total_cost) 중 최엄격 적용
FEE_DEADZONE = 0.0008  # 0.08%

# ── NEUTRAL ATR Gate ────────────────────────────────────
# End-of-window |return| < k × ATR/price → NEUTRAL override
# 배리어 히트 후 복귀한 mean-reversion 케이스를 NEUTRAL로 재분류
NEUTRAL_ATR_K = 0.5

# ── 실질 거래 비용 (직접 차감) ─────────────────────────────────
# Net Return = ln(exit/entry) − (round_trip_fee + effective_slippage)
# |net return| ≤ 0 → NEUTRAL (수수료도 못 벌면 거래 가치 없음)
ROUND_TRIP_FEE = 0.0008          # 0.08%  (Binance futures taker 0.04% × 2)

# 가변 슬리피지 모델 — 변동성 비례 페널티
# effective_slippage = min(SLIPPAGE_MAX, BASE + SLIPPAGE_VOL_SCALE × ATR/price)
# 장점: ATR이 직접 스케일링 → vol_ratio 기준점 불필요, 직관적
BASE_SLIPPAGE = 0.0002           # 0.02%  — 최소 슬리피지 (유동성 좋은 시장)
SLIPPAGE_VOL_SCALE = 0.10        # ATR/price에 대한 슬리피지 비례 계수
SLIPPAGE_MAX = 0.0015            # 0.15%  — 극단 변동성 캡


def compute_trading_costs(atr: float, entry_price: float) -> Tuple[float, float]:
    """실질 거래 비용 — 왕복 수수료 + 가변 슬리피지.

    슬리피지 = min(cap, BASE + scale × ATR/price)
    - BTC 1m ATR/price ≈ 0.1% → slip ≈ 0.02+0.01 = 0.03%
    - BTC 15m ATR/price ≈ 0.5% → slip ≈ 0.02+0.05 = 0.07%
    - 급등락 ATR/price ≈ 1.5% → slip = 0.15% (cap)

    Returns:
        (effective_slippage, total_cost)
    """
    atr_pct = atr / entry_price if entry_price > 0 and atr > 0 else 0.0
    effective_slippage = BASE_SLIPPAGE + SLIPPAGE_VOL_SCALE * atr_pct
    effective_slippage = min(effective_slippage, SLIPPAGE_MAX)
    total_cost = ROUND_TRIP_FEE + effective_slippage
    return effective_slippage, total_cost


def compute_net_label(
    gross_log_return: float,
    total_cost: float,
    barrier_type: str,
    atr_pct: float = 0.0,
) -> Tuple[str, float]:
    """비용 차감 후 실질 방향 라벨 결정.

    TP/SL 히트: |gross| > total_cost → 유효 방향, else NEUTRAL
    TIME: 간접 허들(FEE_DEADZONE, NEUTRAL_ATR_K)과 직접 비용 결합

    net return 부호 규칙:
        UP   → gross − cost  (양수 축소)
        DOWN → gross + cost  (음수 축소 = short 수익 감소 반영)
        NEUTRAL → gross 원본 (참고용)
    """
    abs_gross = abs(gross_log_return)

    if barrier_type == 'TIME':
        # TIME: 간접 허들 + 직접 비용 중 가장 엄격한 값 적용
        neutral_thr = max(FEE_DEADZONE, NEUTRAL_ATR_K * atr_pct, total_cost)
        if abs_gross < neutral_thr:
            return 'NEUTRAL', gross_log_return
    else:
        # TP/SL 히트: 비용 초과 시에만 유효 방향
        if abs_gross <= total_cost:
            return 'NEUTRAL', gross_log_return

    # 허들 통과 → 비용 차감
    if gross_log_return > 0:
        return 'UP', gross_log_return - total_cost
    else:
        return 'DOWN', gross_log_return + total_cost


def compute_barriers(entry_price: float, atr: float,
                     k_tp: float = BARRIER_K_TP, k_sl: float = BARRIER_K_SL,
                     ) -> Tuple[float, float]:
    """TP/SL 배리어 계산 — ATR 기반 + 최소 플로어.

    Upper = price × (1 + max(0.15%, α × ATR/price))
    Lower = price × (1 - max(0.1%,  β × ATR/price))
    """
    if entry_price <= 0:
        return entry_price, entry_price

    atr_pct = atr / entry_price if atr > 0 else 0.0
    tp_pct = max(BARRIER_MIN_TP_PCT, k_tp * atr_pct)
    sl_pct = max(BARRIER_MIN_SL_PCT, k_sl * atr_pct)

    tp_barrier = entry_price * (1.0 + tp_pct)
    sl_barrier = entry_price * (1.0 - sl_pct)
    return tp_barrier, sl_barrier


@dataclass
class SnapshotEntry:
    """피처 스냅샷 — 라벨링 전 상태"""
    timestamp: float          # time.time() 기준
    price: float              # 스냅샷 시점 가격
    features: np.ndarray      # (seq_len, n_features)
    future_log_return: float = 0.0   # 라벨링 후 채워짐
    volatility_z: float = 0.0        # 변동성 Z-score (live soft weight용)
    atr: float = 0.0                 # raw ATR — Triple Barrier 배리어 스케일링용


@dataclass
class LabeledSample:
    """라벨링 완료된 학습 샘플"""
    features: np.ndarray
    direction: str            # 'UP', 'DOWN', or 'NEUTRAL'
    price_change: float       # 절대 가격 변화
    log_return: float         # 로그 수익률
    timestamp: float
    sample_weight: float = 1.0
    timing_score: float = 0.0  # 배리어 도달 속도 (0=TIME, 1=즉시)


# ── Triple Barrier 라벨링 (Precompute용 — OHLCV 접근) ──────────

def triple_barrier_label_precompute(
    entry_price: float,
    atr: float,
    high_arr: np.ndarray,   # candles [i .. i+N-1]의 high
    low_arr: np.ndarray,    # candles [i .. i+N-1]의 low
    close_arr: np.ndarray,  # candles [i .. i+N-1]의 close
    k_tp: float = BARRIER_K_TP,
    k_sl: float = BARRIER_K_SL,
) -> Tuple[str, float, str, float]:
    """Triple Barrier 라벨링 (OHLCV 기반) — 실질 수익률 + 타이밍 스코어.

    Phase 1: ATR 기반 TP/SL 배리어로 gross 방향 탐지
    Phase 2: 거래 비용(수수료 + 가변 슬리피지) 차감 → net return
             |net| ≤ 0 → NEUTRAL (비용 미달 거래 제거)
    Phase 3: 배리어 도달 속도 → timing_score (0~1)

    Returns:
        (direction, net_log_return, barrier_type, timing_score)
        timing_score: 1.0=즉시 도달, 0.0=TIME barrier(미도달)
    """
    if entry_price <= 0:
        return ('NEUTRAL', 0.0, 'TIME', 0.0)

    tp_barrier, sl_barrier = compute_barriers(entry_price, atr, k_tp, k_sl)

    # Phase 1: Gross 방향 탐지
    exit_price = entry_price
    barrier_type = 'TIME'
    hit_candle_idx = -1  # 배리어 도달 캔들 인덱스 (-1 = 미도달)
    n_candles = len(high_arr)

    for j in range(n_candles):
        hit_tp = high_arr[j] >= tp_barrier
        hit_sl = low_arr[j] <= sl_barrier

        if hit_tp and hit_sl:
            # 동일 캔들 TP/SL 동시 도달 → 실제 순서 판별 불가, 샘플 폐기
            return ('NEUTRAL', 0.0, 'CONFLICT', 0.0)
        elif hit_tp:
            exit_price, barrier_type, hit_candle_idx = tp_barrier, 'TP', j
            break
        elif hit_sl:
            exit_price, barrier_type, hit_candle_idx = sl_barrier, 'SL', j
            break
    else:
        # Time barrier: N캔들 내 미도달 → 마지막 close 기준
        exit_price = close_arr[-1] if len(close_arr) > 0 else entry_price

    gross_lr = np.log(exit_price / entry_price) if entry_price > 0 else 0.0

    # Phase 2: 비용 차감 → net return 기준 방향 결정
    _, total_cost = compute_trading_costs(atr, entry_price)
    atr_pct = atr / entry_price if entry_price > 0 and atr > 0 else 0.0
    direction, net_lr = compute_net_label(gross_lr, total_cost, barrier_type, atr_pct)

    # Phase 3: 타이밍 스코어 — 배리어 도달 속도
    # 빨리 도달할수록 높은 점수: 1.0 - (hit_idx / N)
    # TIME barrier (미도달) → 0.0
    if hit_candle_idx >= 0 and n_candles > 0:
        timing_score = 1.0 - (hit_candle_idx / n_candles)
    else:
        timing_score = 0.0

    # Phase 4: MAE 품질 필터 — "버텨서 겨우 TP/SL" 샘플 제거
    # UP: TP 히트 전 SL 방향으로 얼마나 근접했는가 (MAE / SL_distance)
    # DOWN: SL 히트 전 TP 방향으로 얼마나 근접했는가 (MAE / TP_distance)
    # → 임계값 초과 = 운으로 살아난 저품질 샘플 → NEUTRAL로 폐기
    if direction != 'NEUTRAL' and hit_candle_idx >= 0:
        _n = hit_candle_idx + 1  # 배리어 히트 캔들 포함
        if direction == 'UP':
            _mae = float(entry_price - np.min(low_arr[:_n]))
            _sl_dist = float(entry_price - sl_barrier)
            _mae_ratio = _mae / _sl_dist if _sl_dist > 1e-10 else 0.0
        else:  # DOWN
            _mae = float(np.max(high_arr[:_n]) - entry_price)
            _tp_dist = float(tp_barrier - entry_price)
            _mae_ratio = _mae / _tp_dist if _tp_dist > 1e-10 else 0.0

        if _mae_ratio > MAE_QUALITY_THRESHOLD:
            direction = 'NEUTRAL'
            net_lr = 0.0
            barrier_type = 'MAE_REJECT'

    return (direction, net_lr, barrier_type, timing_score)


class PriceLog:
    """
    공유 가격 기록 — 이진탐색으로 특정 시점의 가격 조회

    (ts, close, high, low) 튜플 저장 — 10초 구간 내 wick 포착.
    사전학습 triple_barrier_label_precompute와 동일한 해상도 제공.

    maxlen=50000 → ~5.5시간 분량 (@10초 간격)
    """

    def __init__(self, maxlen: int = 50000):
        self._timestamps: deque = deque(maxlen=maxlen)
        self._prices: deque = deque(maxlen=maxlen)
        self._highs: deque = deque(maxlen=maxlen)
        self._lows: deque = deque(maxlen=maxlen)

    def append(self, timestamp: float, price: float,
               high: Optional[float] = None, low: Optional[float] = None):
        self._timestamps.append(timestamp)
        self._prices.append(price)
        self._highs.append(high if high is not None else price)
        self._lows.append(low if low is not None else price)

    def lookup_price(self, target_time: float) -> Optional[float]:
        """target_time에 가장 가까운 가격을 이진탐색으로 반환"""
        n = len(self._timestamps)
        if n == 0:
            return None

        # deque → list 변환 없이 첫/끝 체크
        if target_time <= self._timestamps[0]:
            return self._prices[0]
        if target_time >= self._timestamps[-1]:
            return None  # 아직 미래 데이터 없음

        # 이진탐색 (list 변환 최소화)
        ts_list = list(self._timestamps)
        idx = bisect.bisect_left(ts_list, target_time)

        if idx >= n:
            return self._prices[-1]
        if idx == 0:
            return self._prices[0]

        # 가장 가까운 쪽 선택
        before = ts_list[idx - 1]
        after = ts_list[idx]
        if (target_time - before) <= (after - target_time):
            return self._prices[idx - 1]
        else:
            return self._prices[idx]

    def lookup_range(self, start_time: float, end_time: float
                     ) -> List[Tuple[float, float, float, float]]:
        """[start_time, end_time] 구간 내 모든 (timestamp, close, high, low) 반환.

        Triple Barrier 라벨링에서 look-ahead 윈도우 내 high/low로 배리어 히트 감지.
        사전학습 triple_barrier_label_precompute와 동일한 방식.
        PriceLog는 ~10초 간격이므로 1m 15캔들=15분 윈도우에 ~90개 데이터포인트.
        """
        n = len(self._timestamps)
        if n == 0:
            return []

        ts_list = list(self._timestamps)
        pr_list = list(self._prices)
        hi_list = list(self._highs)
        lo_list = list(self._lows)

        lo_idx = bisect.bisect_left(ts_list, start_time)
        hi_idx = bisect.bisect_right(ts_list, end_time)

        return [(ts_list[i], pr_list[i], hi_list[i], lo_list[i])
                for i in range(lo_idx, hi_idx)]

    def __len__(self):
        return len(self._timestamps)


class SampleRingBuffer:
    """
    TF별 링 버퍼 — 스냅샷 저장 → Triple Barrier 지연 라벨링

    - 필터링 없이 모든 스냅샷 저장
    - look_ahead 시간 경과 후 PriceLog에서 미래 가격 조회하여 라벨링
    - ATR 기반 TP/SL 배리어 (최소 플로어 보장) 로 경로 의존적 라벨 생성
    """

    def __init__(self, timeframe: str, maxlen: int = 2000):
        self.timeframe = timeframe
        self.look_ahead = LOOK_AHEAD_SECONDS.get(timeframe, 180)
        self._pending: deque = deque(maxlen=maxlen)
        self._labeled_count: int = 0
        self._total_added: int = 0

    def add_snapshot(self, entry: SnapshotEntry):
        """필터링 없이 스냅샷 저장"""
        self._pending.append(entry)
        self._total_added += 1

    def label_matured_entries(
        self,
        price_log: PriceLog,
        current_time: float,
    ) -> List[LabeledSample]:
        """Triple Barrier 라벨링 — PriceLog high/low 기반 (사전학습과 동일).

        PriceLog의 (ts, close, high, low) 데이터로 배리어 히트를 정밀 감지.
        사전학습 triple_barrier_label_precompute와 동일한 로직:
        - high >= TP → UP, low <= SL → DOWN
        - 같은 구간에서 양쪽 히트 시 SL 우선 (보수적)
        - End-of-window NEUTRAL override 없음 (사전학습과 일치)

        Returns:
            라벨링된 샘플 리스트 (비어있을 수 있음)
        """
        labeled: List[LabeledSample] = []

        while self._pending:
            entry = self._pending[0]
            elapsed = current_time - entry.timestamp

            if elapsed < self.look_ahead:
                break  # 아직 미성숙

            # pop
            self._pending.popleft()

            # look-ahead 윈도우 내 모든 (ts, close, high, low) 조회
            target_end = entry.timestamp + self.look_ahead
            price_range = price_log.lookup_range(entry.timestamp + 1, target_end)

            if not price_range:
                continue  # 가격 데이터 없음

            atr = entry.atr
            direction = 'NEUTRAL'
            exit_price = entry.price
            barrier_type = 'TIME'
            hit_elapsed = -1.0  # 배리어 도달까지 경과 시간 (-1 = 미도달)

            if entry.price > 0:
                tp_barrier, sl_barrier = compute_barriers(
                    entry.price, atr, BARRIER_K_TP, BARRIER_K_SL
                )

                # High/Low 기반 배리어 히트 감지 (사전학습과 동일)
                for ts, px, hi, lo in price_range:
                    hit_tp = hi >= tp_barrier
                    hit_sl = lo <= sl_barrier

                    if hit_tp and hit_sl:
                        # 동일 구간 TP/SL 동시 도달 → 순서 판별 불가, 샘플 폐기
                        direction = 'CONFLICT'
                        break
                    elif hit_tp:
                        direction, exit_price, barrier_type = 'UP', tp_barrier, 'TP'
                        hit_elapsed = ts - entry.timestamp
                        break
                    elif hit_sl:
                        direction, exit_price, barrier_type = 'DOWN', sl_barrier, 'SL'
                        hit_elapsed = ts - entry.timestamp
                        break
                else:
                    # 배리어 미도달 → 마지막 close 사용 (사전학습과 동일)
                    exit_price = price_range[-1][1]
            else:
                # 가격 0 이하 → fallback
                future_price = price_log.lookup_price(target_end)
                if future_price is None:
                    continue
                exit_price = future_price

            # TP/SL 동시 도달 → 샘플 폐기 (구조적 방향 편향 방지)
            if direction == 'CONFLICT':
                continue

            gross_lr = np.log(exit_price / entry.price) if entry.price > 0 else 0.0

            # 실질 비용 차감 → net return 기준 방향 결정
            # (사전학습 triple_barrier_label_precompute와 동일 로직)
            _, total_cost = compute_trading_costs(entry.atr, entry.price)
            _atr_pct = entry.atr / entry.price if entry.atr > 0 and entry.price > 0 else 0.0
            direction, net_lr = compute_net_label(
                gross_lr, total_cost, barrier_type, _atr_pct
            )

            # 타이밍 스코어: 배리어 도달 속도 (0=TIME, 1=즉시)
            if hit_elapsed >= 0 and self.look_ahead > 0:
                timing_score = max(0.0, 1.0 - (hit_elapsed / self.look_ahead))
            else:
                timing_score = 0.0

            price_change = exit_price - entry.price

            sample = LabeledSample(
                features=entry.features,
                direction=direction,
                price_change=price_change,
                log_return=net_lr,
                timestamp=entry.timestamp,
                sample_weight=entry.volatility_z,  # raw z-score, sigmoid은 호출부에서 적용
                timing_score=timing_score,
            )
            labeled.append(sample)
            self._labeled_count += 1

        return labeled

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def stats(self) -> dict:
        return {
            'timeframe': self.timeframe,
            'look_ahead_sec': self.look_ahead,
            'total_added': self._total_added,
            'pending': len(self._pending),
            'labeled': self._labeled_count,
        }
