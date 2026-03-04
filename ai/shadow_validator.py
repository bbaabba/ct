"""Shadow Validation — 모델 콜드스타트 보호 시스템.

새 모델이나 리셋된 LoRA 전문가를 즉시 실전 투입하지 않고,
가상 예측을 누적하여 승률/수익비가 검증된 후에만 LIVE 전환.

상태 전이:
    SHADOW ──(승격)──> LIVE ──(강등)──> DEMOTED ──(재승격)──> LIVE
         ↑                                          |
         └────────────(force_shadow)─────────────────┘

사용:
    sv = ShadowValidator()
    sv.record_prediction('LONG', 0.75, 2500.0)
    sv.update_prices(2510.0)
    if sv.is_live:
        # 실거래 허용
"""
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

from utils.logger import setup_logger
from ai.sample_buffer import compute_trading_costs

logger = setup_logger(__name__)


class ShadowState(Enum):
    SHADOW = "shadow"       # 가상 예측만 기록, 거래 차단
    LIVE = "live"           # 모델이 실거래 제어
    DEMOTED = "demoted"     # 성능 저하로 shadow 복귀


@dataclass
class VirtualTrade:
    """가상 예측 기록."""
    timestamp: float
    direction: str           # 'LONG' or 'SHORT'
    confidence: float
    entry_price: float
    atr: float = 0.0                 # ATR — 가변 슬리피지 계산용
    resolved: bool = False
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None


@dataclass
class ShadowConfig:
    """승격/강등 임계값 설정."""
    # 승격 기준 (SHADOW/DEMOTED → LIVE)
    # 트레이딩에서 승률 < 50%도 PF > 1이면 수익 가능 (큰 수익/작은 손실)
    # 가상거래는 15캔들 고정 해소 → 실제 SL/TP 트레이딩보다 불리하므로 기준 완화
    min_samples: int = 30
    min_accuracy: float = 0.38
    min_profit_factor: float = 0.85

    # 강등 기준 (LIVE → DEMOTED)
    demotion_window: int = 50
    demotion_accuracy_floor: float = 0.30
    demotion_profit_factor_floor: float = 0.65

    # DEMOTED 상태 연성 게이트: 신호를 완전 차단하지 않고 스케일링
    # 0.0 = 완전 차단 (기존), 1.0 = 차단 없음
    demoted_signal_scale: float = 0.4

    # 승격 후 보호 기간: 승격 직후 N건은 강등 평가 유예
    # (승격 직후 old 데이터로 즉시 강등되는 현상 방지)
    promotion_grace_trades: int = 30

    # 해소 설정 — Triple Barrier 1m look-ahead(15분)에 맞춤
    resolution_candles: int = 15       # 15캔들 후 해소
    resolution_interval_s: float = 60  # 1분봉 간격(초)
    min_move_threshold: float = 0.0001 # 최소 가격 변동


class ShadowValidator:
    """모델 콜드스타트 보호 — 가상 예측 기반 승격/강등."""

    def __init__(self, config: Optional[ShadowConfig] = None):
        self.config = config or ShadowConfig()
        self.state = ShadowState.SHADOW
        self._virtual_trades: deque = deque(maxlen=2000)
        self._pending_trades: List[VirtualTrade] = []
        self._state_change_time: float = time.time()
        self._promotion_count: int = 0
        self._demotion_count: int = 0
        self._demotion_trade_offset: int = 0  # 강등 시점의 resolved 수 (재승격 평가 기준점)
        self._promotion_trade_offset: int = 0  # 승격 시점의 resolved 수 (grace period 기준)

    @property
    def is_live(self) -> bool:
        """실거래 허용 상태 여부."""
        return self.state == ShadowState.LIVE

    def record_prediction(
        self, direction: str, confidence: float, entry_price: float,
        atr: float = 0.0, sim_time: Optional[float] = None,
    ):
        """가상 예측 기록 (shadow/live 모두에서 호출).

        Args:
            sim_time: 백테스트 시 캔들 타임스탬프(epoch seconds). None이면 time.time().
        """
        if direction not in ('LONG', 'SHORT'):
            return
        vt = VirtualTrade(
            timestamp=sim_time if sim_time is not None else time.time(),
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            atr=atr,
        )
        self._pending_trades.append(vt)

    def update_prices(self, current_price: float, sim_time: Optional[float] = None):
        """pending trades를 현재 가격으로 해소 + 상태 전이 평가.

        Args:
            sim_time: 백테스트 시 캔들 타임스탬프(epoch seconds). None이면 time.time().
        """
        if current_price <= 0:
            return
        now = sim_time if sim_time is not None else time.time()
        resolution_time = self.config.resolution_candles * self.config.resolution_interval_s

        still_pending = []
        for vt in self._pending_trades:
            age = now - vt.timestamp
            if age >= resolution_time:
                vt.exit_price = current_price
                move = (current_price - vt.entry_price) / vt.entry_price
                gross_pnl = move if vt.direction == 'LONG' else -move
                # 실질 비용 차감 (수수료 + 가변 슬리피지)
                _, total_cost = compute_trading_costs(vt.atr, vt.entry_price)
                vt.pnl_pct = gross_pnl - total_cost
                vt.resolved = True
                self._virtual_trades.append(vt)
            else:
                still_pending.append(vt)
        self._pending_trades = still_pending

        self._evaluate_state_transition()

    def _evaluate_state_transition(self):
        """승격/강등 조건 평가."""
        resolved = [vt for vt in self._virtual_trades if vt.resolved]

        if self.state == ShadowState.SHADOW:
            if len(resolved) >= self.config.min_samples:
                stats = self._compute_stats(resolved[-self.config.min_samples:])
                if (stats['accuracy'] >= self.config.min_accuracy and
                        stats['profit_factor'] >= self.config.min_profit_factor):
                    self._promote()

        elif self.state == ShadowState.DEMOTED:
            # 강등 이후 새로 쌓인 데이터만 평가 (이전 데이터 오염 방지)
            new_resolved = resolved[self._demotion_trade_offset:]
            if len(new_resolved) >= self.config.min_samples:
                stats = self._compute_stats(new_resolved[-self.config.min_samples:])
                if (stats['accuracy'] >= self.config.min_accuracy and
                        stats['profit_factor'] >= self.config.min_profit_factor):
                    self._promote()

        elif self.state == ShadowState.LIVE:
            # 승격 후 보호 기간: grace trades만큼은 강등 평가 유예
            trades_since_promotion = len(resolved) - self._promotion_trade_offset
            if trades_since_promotion < self.config.promotion_grace_trades:
                return
            if len(resolved) >= self.config.demotion_window:
                # 승격 이후 데이터만으로 강등 평가
                post_promotion = resolved[self._promotion_trade_offset:]
                eval_window = post_promotion[-self.config.demotion_window:]
                stats = self._compute_stats(eval_window)
                if (stats['accuracy'] < self.config.demotion_accuracy_floor or
                        stats['profit_factor'] < self.config.demotion_profit_factor_floor):
                    self._demote()

    def _compute_stats(self, trades: list) -> dict:
        """가상 거래 통계 계산."""
        if not trades:
            return {'accuracy': 0.0, 'profit_factor': 0.0, 'count': 0}
        threshold = self.config.min_move_threshold
        wins = sum(1 for t in trades if t.pnl_pct and t.pnl_pct > threshold)
        losses = sum(1 for t in trades if t.pnl_pct and t.pnl_pct < -threshold)
        total = wins + losses
        accuracy = wins / total if total > 0 else 0.0
        sum_wins = sum(t.pnl_pct for t in trades if t.pnl_pct and t.pnl_pct > 0)
        sum_losses = abs(sum(t.pnl_pct for t in trades if t.pnl_pct and t.pnl_pct < 0))
        profit_factor = sum_wins / sum_losses if sum_losses > 0 else float('inf')
        return {'accuracy': accuracy, 'profit_factor': profit_factor, 'count': total}

    def _promote(self):
        """SHADOW/DEMOTED → LIVE 승격."""
        self.state = ShadowState.LIVE
        self._state_change_time = time.time()
        self._promotion_count += 1
        self._promotion_trade_offset = len([vt for vt in self._virtual_trades if vt.resolved])
        logger.info(
            f"  🟢 ShadowValidator → LIVE 승격 (#{self._promotion_count}), "
            f"grace={self.config.promotion_grace_trades}건 보호"
        )

    def _demote(self):
        """LIVE → DEMOTED 강등.

        히스토리 유지 — 재승격은 강등 이후 새 데이터 기준으로 평가.
        (clear하면 0건 → acc=0% → 재승격 불가 루프)
        """
        self.state = ShadowState.DEMOTED
        self._state_change_time = time.time()
        self._demotion_count += 1
        self._demotion_trade_offset = len([vt for vt in self._virtual_trades if vt.resolved])
        logger.warning(f"  🔴 ShadowValidator → DEMOTED 강등 (#{self._demotion_count})")

    def force_shadow(self, reason: str = ""):
        """강제 shadow 복귀 (모델 리셋/핫스왑 시)."""
        self.state = ShadowState.SHADOW
        self._virtual_trades.clear()
        self._pending_trades.clear()
        self._state_change_time = time.time()
        self._demotion_trade_offset = 0
        self._promotion_trade_offset = 0
        logger.info(f"  ⏳ ShadowValidator → SHADOW (강제): {reason}")

    def get_status(self) -> dict:
        """현재 상태 및 통계 조회.

        DEMOTED 상태: 강등 이후 새 데이터 기준 통계 표시.
        SHADOW/LIVE: 최근 resolved 기준.
        """
        resolved = [vt for vt in self._virtual_trades if vt.resolved]

        # 상태별 평가 대상 결정
        if self.state == ShadowState.DEMOTED:
            eval_trades = resolved[self._demotion_trade_offset:]
            eval_count = len(eval_trades)
        else:
            eval_trades = resolved
            eval_count = len(resolved)

        stats = self._compute_stats(eval_trades[-self.config.min_samples:]) if eval_trades else {}
        return {
            'state': self.state.value,
            'resolved_count': eval_count,       # DEMOTED: 강등 이후 건수
            'total_resolved': len(resolved),    # 전체 누적 건수
            'pending_count': len(self._pending_trades),
            'recent_accuracy': round(stats.get('accuracy', 0), 4),
            'recent_profit_factor': round(stats.get('profit_factor', 0), 4),
            'promotions': self._promotion_count,
            'demotions': self._demotion_count,
        }

    # ─── Serialization ─────────────────────────────────────

    def to_dict(self) -> dict:
        """상태 직렬화 (모델 저장 시 포함)."""
        return {
            'state': self.state.value,
            'virtual_trades': [
                {
                    'timestamp': vt.timestamp,
                    'direction': vt.direction,
                    'confidence': vt.confidence,
                    'entry_price': vt.entry_price,
                    'atr': vt.atr,
                    'resolved': vt.resolved,
                    'exit_price': vt.exit_price,
                    'pnl_pct': vt.pnl_pct,
                }
                for vt in self._virtual_trades
            ],
            'pending_trades': [
                {
                    'timestamp': vt.timestamp,
                    'direction': vt.direction,
                    'confidence': vt.confidence,
                    'entry_price': vt.entry_price,
                    'atr': vt.atr,
                }
                for vt in self._pending_trades
            ],
            'promotion_count': self._promotion_count,
            'demotion_count': self._demotion_count,
            'demotion_trade_offset': self._demotion_trade_offset,
            'promotion_trade_offset': self._promotion_trade_offset,
        }

    def from_dict(self, data: dict):
        """직렬화된 상태로부터 복원."""
        import time as _time

        try:
            self.state = ShadowState(data.get('state', 'shadow'))
        except ValueError:
            self.state = ShadowState.SHADOW

        self._virtual_trades.clear()
        for vt_d in data.get('virtual_trades', []):
            vt = VirtualTrade(
                timestamp=vt_d['timestamp'],
                direction=vt_d['direction'],
                confidence=vt_d['confidence'],
                entry_price=vt_d['entry_price'],
                atr=vt_d.get('atr', 0.0),
                resolved=vt_d.get('resolved', False),
                exit_price=vt_d.get('exit_price'),
                pnl_pct=vt_d.get('pnl_pct'),
            )
            self._virtual_trades.append(vt)

        self._pending_trades = []
        for vt_d in data.get('pending_trades', []):
            vt = VirtualTrade(
                timestamp=vt_d['timestamp'],
                direction=vt_d['direction'],
                confidence=vt_d['confidence'],
                entry_price=vt_d['entry_price'],
                atr=vt_d.get('atr', 0.0),
            )
            self._pending_trades.append(vt)

        self._promotion_count = data.get('promotion_count', 0)
        self._demotion_count = data.get('demotion_count', 0)
        self._demotion_trade_offset = data.get('demotion_trade_offset', 0)
        self._promotion_trade_offset = data.get('promotion_trade_offset', 0)
        self._state_change_time = _time.time()
