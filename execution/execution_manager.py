"""스마트 주문 실행 엔진 (ExecutionManager)

cascade_conf / fee_ev_ratio 기반 3단계 체결 모드:
  1. AGGRESSIVE_TAKER  — 시장가 즉시 체결 (높은 확신)
  2. PASSIVE_MAKER     — Post-Only 지정가 + OBI 미세조정 (중간 확신)
  3. FALLBACK_MARKET   — 기본 시장가 (낮은 확신, 기존 동작)

지정가(Passive) 미체결 시 Chase & Timeout 로직으로 추격/취소.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

class ExecutionMode(Enum):
    AGGRESSIVE_TAKER = "aggressive_taker"
    PASSIVE_MAKER = "passive_maker"
    FALLBACK_MARKET = "fallback_market"


@dataclass
class ExecutionContext:
    """체결 판단에 필요한 스냅샷 데이터"""
    symbol: str
    side: str                    # 'LONG' | 'SHORT'
    quantity: float              # 이미 precision 처리된 수량
    cascade_conf: float          # [0, 1]
    fee_ev_ratio: float          # 로깅용 유지 (모드 선택에서 제외)
    obi: float                   # [-1, 1] OBI
    best_bid: float
    best_ask: float
    book_ticker_age: float       # seconds since last WS update
    timing_confidence: float = 0.0  # [0, 1] timing_head 출력 (체결 모드 선택 기준)


@dataclass
class ExecutionResult:
    """체결 결과"""
    success: bool
    mode: ExecutionMode
    filled_price: float          # 가중평균 체결가
    filled_quantity: float
    unfilled_quantity: float     # 미체결 잔량
    order_ids: List[str] = field(default_factory=list)
    was_chased: bool = False     # 추격 체결 여부
    fee_saved_bps: float = 0.0   # maker vs taker 절약 추정 (bps)
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# ExecutionManager
# ═══════════════════════════════════════════════════════════════

class ExecutionManager:
    """스마트 주문 실행 관리자

    Args:
        client: BinanceFuturesClient 인스턴스 (live 전용, paper일 때 None 가능)
        symbol: 거래 심볼 (예: 'BTCUSDT')
        paper_trading: 페이퍼 트레이딩 모드 여부
        cascade_conf_getter: 현재 시점 cascade_conf를 반환하는 콜백 (chase 중 재확인용)
        candle_low_high_getter: 현재 캔들 low/high 반환 콜백 (paper passive 시뮬용)
                                → (low: float, high: float)
    """

    # ── 임계값 상수 ──
    AGGRESSIVE_MIN_CONF = 0.65
    AGGRESSIVE_MIN_TIMING = 0.70
    PASSIVE_MIN_CONF = 0.50
    OBI_THRESHOLD = 0.2
    OBI_STRONG = 0.4
    CHASE_TIMEOUT_SEC = 10.0
    CHASE_POLL_SEC = 1.0
    CHASE_MIN_CONF = 0.5
    MAX_BOOK_AGE_SEC = 5.0

    # 수수료 기준 (bps)
    TAKER_FEE_BPS = 4.0   # 0.04%
    MAKER_FEE_BPS = 2.0   # 0.02%

    def __init__(
        self,
        client,
        symbol: str,
        paper_trading: bool = True,
        cascade_conf_getter: Optional[Callable[[], float]] = None,
        candle_low_high_getter: Optional[Callable[[], tuple]] = None,
    ):
        self.client = client
        self.symbol = symbol
        self.paper_trading = paper_trading
        self._cascade_conf_getter = cascade_conf_getter
        self._candle_low_high_getter = candle_low_high_getter

        # 심볼 규격 캐시 (초기화 시 1회 로드)
        self._symbol_rules: Dict[str, dict] = {}
        self._load_symbol_rules(symbol)

        # 통계
        self._stats = {
            'aggressive_count': 0,
            'passive_count': 0,
            'fallback_count': 0,
            'chase_count': 0,
            'gtx_reject_count': 0,
            'min_notional_skip_count': 0,
        }

    # ── 초기화: 거래소 규격 캐싱 ──

    def _load_symbol_rules(self, symbol: str):
        """exchangeInfo에서 tick_size, min_notional, step_size 추출 → 캐싱

        봇 시작 시 1회만 호출. API Rate Limit 방지.
        """
        tick_size = 0.10 if 'BTC' in symbol else 0.01
        min_notional = 5.0
        step_size = 0.001

        if self.client:
            try:
                info = self.client.get_symbol_info(symbol)
                if info:
                    for f in info.get('filters', []):
                        ft = f.get('filterType', '')
                        if ft == 'PRICE_FILTER':
                            tick_size = float(f['tickSize'])
                        elif ft == 'MIN_NOTIONAL':
                            min_notional = float(f.get('notional', 5.0))
                        elif ft == 'LOT_SIZE':
                            step_size = float(f['stepSize'])
                    logger.info(
                        f"[EXEC] {symbol} 규격 로드: tick={tick_size}, "
                        f"min_notional={min_notional}, step={step_size}"
                    )
            except Exception as e:
                logger.warning(f"[EXEC] {symbol} 규격 로드 실패, 폴백 사용: {e}")

        self._symbol_rules[symbol] = {
            'tick_size': tick_size,
            'min_notional': min_notional,
            'step_size': step_size,
        }

    def _get_rules(self, symbol: str) -> dict:
        """캐싱된 심볼 규격 반환"""
        if symbol not in self._symbol_rules:
            self._load_symbol_rules(symbol)
        return self._symbol_rules[symbol]

    # ═══════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════

    def execute(self, ctx: ExecutionContext) -> ExecutionResult:
        """메인 진입점: 모드 선택 → 실행 → 결과 반환"""
        mode = self._select_mode(ctx)

        logger.info(f"[EXEC] 모드: {mode.value} | conf={ctx.cascade_conf:.2f} "
                     f"timing={ctx.timing_confidence:.2f} obi={ctx.obi:+.2f} "
                     f"book_age={ctx.book_ticker_age:.1f}s")

        if self.paper_trading:
            result = self._execute_paper(mode, ctx)
        else:
            result = self._execute_live(mode, ctx)

        # 통계 업데이트
        if result.success:
            if mode == ExecutionMode.AGGRESSIVE_TAKER:
                self._stats['aggressive_count'] += 1
            elif mode == ExecutionMode.PASSIVE_MAKER:
                self._stats['passive_count'] += 1
            else:
                self._stats['fallback_count'] += 1
            if result.was_chased:
                self._stats['chase_count'] += 1

        return result

    def get_stats(self) -> dict:
        """체결 통계 반환"""
        return dict(self._stats)

    # ═══════════════════════════════════════════════════════════
    # Mode Selection
    # ═══════════════════════════════════════════════════════════

    def _select_mode(self, ctx: ExecutionContext) -> ExecutionMode:
        """cascade_conf + timing_confidence 기반 체결 모드 결정

        - AGGRESSIVE: conf ≥ 0.65 AND timing ≥ 0.70  → 빠른 진입 (taker)
        - PASSIVE:    conf ≥ 0.50 AND timing < 0.70   → 좋은 가격 (maker)
        - FALLBACK:   나머지
        """
        book_fresh = ctx.book_ticker_age < self.MAX_BOOK_AGE_SEC

        # AGGRESSIVE: 높은 확신 + 빠른 타이밍 → 즉시 taker
        if (ctx.cascade_conf >= self.AGGRESSIVE_MIN_CONF
                and ctx.timing_confidence >= self.AGGRESSIVE_MIN_TIMING):
            return ExecutionMode.AGGRESSIVE_TAKER

        # PASSIVE: 중간 확신 + 느린 타이밍 → maker로 좋은 가격
        if (book_fresh
                and ctx.cascade_conf >= self.PASSIVE_MIN_CONF
                and ctx.timing_confidence < self.AGGRESSIVE_MIN_TIMING):
            return ExecutionMode.PASSIVE_MAKER

        return ExecutionMode.FALLBACK_MARKET

    # ═══════════════════════════════════════════════════════════
    # OBI Limit Price Calculation
    # ═══════════════════════════════════════════════════════════

    def _calc_obi_offset_ticks(self, side: str, obi: float) -> int:
        """OBI 기반 틱 오프셋 계산

        LONG:  매도 압력(OBI < -0.2) → 더 싸게 매수 (음수 오프셋)
        SHORT: 매수 압력(OBI > +0.2) → 더 비싸게 매도 (양수 오프셋)
        """
        if side == 'LONG':
            if obi < -self.OBI_STRONG:
                return -2
            elif obi < -self.OBI_THRESHOLD:
                return -1
        elif side == 'SHORT':
            if obi > self.OBI_STRONG:
                return 2
            elif obi > self.OBI_THRESHOLD:
                return 1
        return 0

    def _calc_limit_price(self, side: str, ctx: ExecutionContext) -> float:
        """OBI 미세조정 반영된 지정가 산출"""
        rules = self._get_rules(ctx.symbol)
        tick_size = rules['tick_size']
        offset = self._calc_obi_offset_ticks(side, ctx.obi)

        if side == 'LONG':
            price = ctx.best_bid + (offset * tick_size)
        else:  # SHORT
            price = ctx.best_ask + (offset * tick_size)

        # 가격 정밀도 맞추기
        if self.client:
            price = self.client.round_price(ctx.symbol, price)

        return price

    # ═══════════════════════════════════════════════════════════
    # Live Execution
    # ═══════════════════════════════════════════════════════════

    def _execute_live(self, mode: ExecutionMode, ctx: ExecutionContext) -> ExecutionResult:
        """실제 거래 실행 분기"""
        if mode == ExecutionMode.AGGRESSIVE_TAKER:
            return self._live_market_order(ctx)
        elif mode == ExecutionMode.PASSIVE_MAKER:
            return self._live_passive_order(ctx)
        else:
            return self._live_market_order(ctx)

    def _live_market_order(self, ctx: ExecutionContext) -> ExecutionResult:
        """시장가 즉시 체결 (AGGRESSIVE / FALLBACK 공용)"""
        from api.futures_client import BinanceAPIError

        order_side = 'BUY' if ctx.side == 'LONG' else 'SELL'
        try:
            result = self.client.place_market_order(
                symbol=ctx.symbol,
                side=order_side,
                quantity=ctx.quantity,
            )
            filled_price = float(result.get('avgPrice', 0))
            filled_qty = float(result.get('executedQty', 0))
            order_id = str(result.get('orderId', ''))

            if filled_qty <= 0:
                filled_price = (ctx.best_bid + ctx.best_ask) / 2
                filled_qty = ctx.quantity

            return ExecutionResult(
                success=True,
                mode=ExecutionMode.AGGRESSIVE_TAKER if ctx.cascade_conf >= self.AGGRESSIVE_MIN_CONF else ExecutionMode.FALLBACK_MARKET,
                filled_price=filled_price,
                filled_quantity=filled_qty,
                unfilled_quantity=max(0, ctx.quantity - filled_qty),
                order_ids=[order_id],
            )
        except BinanceAPIError as e:
            logger.error(f"[EXEC] 시장가 주문 실패: [{e.code}] {e.message}")
            return ExecutionResult(
                success=False,
                mode=ExecutionMode.FALLBACK_MARKET,
                filled_price=0.0,
                filled_quantity=0.0,
                unfilled_quantity=ctx.quantity,
                error=f"[{e.code}] {e.message}",
            )
        except Exception as e:
            logger.error(f"[EXEC] 시장가 주문 예외: {e}")
            return ExecutionResult(
                success=False,
                mode=ExecutionMode.FALLBACK_MARKET,
                filled_price=0.0,
                filled_quantity=0.0,
                unfilled_quantity=ctx.quantity,
                error=str(e),
            )

    def _live_passive_order(self, ctx: ExecutionContext) -> ExecutionResult:
        """Post-Only 지정가 + Chase & Timeout

        GTX 거부 시 즉시 FALLBACK_MARKET으로 전환.
        """
        from api.futures_client import BinanceAPIError

        order_side = 'BUY' if ctx.side == 'LONG' else 'SELL'
        limit_price = self._calc_limit_price(ctx.side, ctx)

        logger.info(f"[EXEC] Passive: {order_side} {ctx.quantity} @ {limit_price:.2f} "
                     f"(bid={ctx.best_bid:.2f}, ask={ctx.best_ask:.2f}, obi={ctx.obi:+.2f})")

        # ── Post-Only 주문 발송 ──
        try:
            result = self.client.place_order(
                symbol=ctx.symbol,
                side=order_side,
                order_type='LIMIT',
                quantity=ctx.quantity,
                price=limit_price,
                time_in_force='GTX',  # Post-Only
            )
        except BinanceAPIError as e:
            # GTX 거부: 스프레드 교차 → 즉시 시장가 전환
            if e.code in (-5022, -4131, -1015):
                logger.warning(f"[EXEC] GTX 거부 ({e.code}): {e.message} → 시장가 전환")
                self._stats['gtx_reject_count'] += 1
                return self._live_market_order(ctx)
            # 기타 에러
            logger.error(f"[EXEC] Passive 주문 실패: [{e.code}] {e.message}")
            return ExecutionResult(
                success=False,
                mode=ExecutionMode.PASSIVE_MAKER,
                filled_price=0.0,
                filled_quantity=0.0,
                unfilled_quantity=ctx.quantity,
                error=f"[{e.code}] {e.message}",
            )
        except Exception as e:
            logger.error(f"[EXEC] Passive 주문 예외: {e}")
            return ExecutionResult(
                success=False,
                mode=ExecutionMode.PASSIVE_MAKER,
                filled_price=0.0,
                filled_quantity=0.0,
                unfilled_quantity=ctx.quantity,
                error=str(e),
            )

        order_id = result.get('orderId')
        if not order_id:
            return ExecutionResult(
                success=False,
                mode=ExecutionMode.PASSIVE_MAKER,
                filled_price=0.0,
                filled_quantity=0.0,
                unfilled_quantity=ctx.quantity,
                error="orderId 없음",
            )

        # ── Chase & Timeout 루프 ──
        return self._chase_loop_live(ctx, order_id, limit_price)

    def _chase_loop_live(
        self, ctx: ExecutionContext, order_id, limit_price: float
    ) -> ExecutionResult:
        """10초 폴링 → 미체결 시 추격 또는 취소

        Args:
            ctx: 원본 실행 컨텍스트
            order_id: 지정가 주문 ID
            limit_price: 제출한 지정가
        """
        from api.futures_client import BinanceAPIError

        start = time.time()
        filled_qty = 0.0
        avg_price = 0.0
        all_order_ids = [str(order_id)]

        while (time.time() - start) < self.CHASE_TIMEOUT_SEC:
            time.sleep(self.CHASE_POLL_SEC)

            try:
                status_resp = self.client.get_order(ctx.symbol, order_id)
            except Exception as e:
                logger.warning(f"[EXEC] 주문 상태 조회 실패: {e}")
                continue

            status = status_resp.get('status', '')
            exec_qty = float(status_resp.get('executedQty', 0))
            exec_price = float(status_resp.get('avgPrice', 0))

            if status == 'FILLED':
                logger.info(f"[EXEC] Passive 완전 체결: {exec_qty} @ {exec_price:.2f}")
                return ExecutionResult(
                    success=True,
                    mode=ExecutionMode.PASSIVE_MAKER,
                    filled_price=exec_price,
                    filled_quantity=exec_qty,
                    unfilled_quantity=0.0,
                    order_ids=all_order_ids,
                    fee_saved_bps=self.TAKER_FEE_BPS - self.MAKER_FEE_BPS,
                )

            if status == 'PARTIALLY_FILLED':
                filled_qty = exec_qty
                avg_price = exec_price
                logger.debug(f"[EXEC] 부분 체결: {filled_qty}/{ctx.quantity}")

            if status in ('CANCELED', 'EXPIRED', 'REJECTED'):
                logger.info(f"[EXEC] 주문 종료 ({status}): 체결 {exec_qty}")
                filled_qty = exec_qty
                avg_price = exec_price if exec_qty > 0 else 0.0
                break

        # ── 타임아웃: 미체결 잔량 처리 ──
        unfilled_qty = ctx.quantity - filled_qty

        # 아직 열린 주문이면 취소
        if unfilled_qty > 0:
            try:
                self.client.cancel_order(ctx.symbol, order_id)
                logger.info(f"[EXEC] 타임아웃 → 주문 취소 (잔량: {unfilled_qty:.4f})")
            except Exception as e:
                logger.warning(f"[EXEC] 주문 취소 실패: {e}")

        # 잔량이 없으면 성공
        if unfilled_qty <= 0:
            return ExecutionResult(
                success=True,
                mode=ExecutionMode.PASSIVE_MAKER,
                filled_price=avg_price,
                filled_quantity=filled_qty,
                unfilled_quantity=0.0,
                order_ids=all_order_ids,
                fee_saved_bps=self.TAKER_FEE_BPS - self.MAKER_FEE_BPS,
            )

        # ── Chase 판단 ──
        rules = self._get_rules(ctx.symbol)
        current_price = (ctx.best_bid + ctx.best_ask) / 2
        remaining_value = unfilled_qty * current_price

        # Min Notional 검증
        if remaining_value < rules['min_notional']:
            logger.info(f"[EXEC] 추격 포기: 잔량 {remaining_value:.2f} USDT < "
                         f"최소 {rules['min_notional']} USDT")
            self._stats['min_notional_skip_count'] += 1

            if filled_qty > 0:
                return ExecutionResult(
                    success=True,
                    mode=ExecutionMode.PASSIVE_MAKER,
                    filled_price=avg_price,
                    filled_quantity=filled_qty,
                    unfilled_quantity=unfilled_qty,
                    order_ids=all_order_ids,
                    fee_saved_bps=self.TAKER_FEE_BPS - self.MAKER_FEE_BPS,
                )
            return ExecutionResult(
                success=False,
                mode=ExecutionMode.PASSIVE_MAKER,
                filled_price=0.0,
                filled_quantity=0.0,
                unfilled_quantity=ctx.quantity,
                order_ids=all_order_ids,
                error="미체결 + 잔량 MIN_NOTIONAL 미달",
            )

        # cascade_conf 재확인
        current_conf = self._cascade_conf_getter() if self._cascade_conf_getter else ctx.cascade_conf

        if current_conf >= self.CHASE_MIN_CONF:
            # ── 추격: 남은 수량 시장가 ──
            logger.info(f"[EXEC] 추격 시작: conf={current_conf:.2f} ≥ {self.CHASE_MIN_CONF} "
                         f"→ 시장가 {unfilled_qty:.4f}")

            chase_ctx = ExecutionContext(
                symbol=ctx.symbol,
                side=ctx.side,
                quantity=self.client.round_quantity(ctx.symbol, unfilled_qty),
                cascade_conf=current_conf,
                fee_ev_ratio=ctx.fee_ev_ratio,
                obi=ctx.obi,
                best_bid=ctx.best_bid,
                best_ask=ctx.best_ask,
                book_ticker_age=ctx.book_ticker_age,
            )
            chase_result = self._live_market_order(chase_ctx)

            if chase_result.success:
                # 가중평균 체결가 계산
                total_qty = filled_qty + chase_result.filled_quantity
                if total_qty > 0:
                    combined_price = (
                        (avg_price * filled_qty + chase_result.filled_price * chase_result.filled_quantity)
                        / total_qty
                    )
                else:
                    combined_price = chase_result.filled_price

                all_order_ids.extend(chase_result.order_ids)
                self._stats['chase_count'] += 1

                return ExecutionResult(
                    success=True,
                    mode=ExecutionMode.PASSIVE_MAKER,
                    filled_price=combined_price,
                    filled_quantity=total_qty,
                    unfilled_quantity=max(0, ctx.quantity - total_qty),
                    order_ids=all_order_ids,
                    was_chased=True,
                    # 부분은 maker, 추격분은 taker → 가중 절약
                    fee_saved_bps=(self.TAKER_FEE_BPS - self.MAKER_FEE_BPS) * (filled_qty / total_qty) if total_qty > 0 else 0.0,
                )
            else:
                # 추격도 실패
                if filled_qty > 0:
                    return ExecutionResult(
                        success=True,
                        mode=ExecutionMode.PASSIVE_MAKER,
                        filled_price=avg_price,
                        filled_quantity=filled_qty,
                        unfilled_quantity=unfilled_qty,
                        order_ids=all_order_ids,
                        was_chased=True,
                        error=f"추격 실패: {chase_result.error}",
                    )
                return ExecutionResult(
                    success=False,
                    mode=ExecutionMode.PASSIVE_MAKER,
                    filled_price=0.0,
                    filled_quantity=0.0,
                    unfilled_quantity=ctx.quantity,
                    order_ids=all_order_ids,
                    error=f"체결+추격 모두 실패: {chase_result.error}",
                )
        else:
            # ── 신호 약화 → 추격 포기 ──
            logger.info(f"[EXEC] 추격 포기: conf={current_conf:.2f} < {self.CHASE_MIN_CONF}")

            if filled_qty > 0:
                return ExecutionResult(
                    success=True,
                    mode=ExecutionMode.PASSIVE_MAKER,
                    filled_price=avg_price,
                    filled_quantity=filled_qty,
                    unfilled_quantity=unfilled_qty,
                    order_ids=all_order_ids,
                    fee_saved_bps=self.TAKER_FEE_BPS - self.MAKER_FEE_BPS,
                )
            return ExecutionResult(
                success=False,
                mode=ExecutionMode.PASSIVE_MAKER,
                filled_price=0.0,
                filled_quantity=0.0,
                unfilled_quantity=ctx.quantity,
                order_ids=all_order_ids,
                error="미체결 + 신호 약화 → 취소",
            )

    # ═══════════════════════════════════════════════════════════
    # Paper Trading Execution
    # ═══════════════════════════════════════════════════════════

    def _execute_paper(self, mode: ExecutionMode, ctx: ExecutionContext) -> ExecutionResult:
        """페이퍼 트레이딩 실행 분기"""
        if mode == ExecutionMode.AGGRESSIVE_TAKER:
            return self._paper_market(ctx)
        elif mode == ExecutionMode.PASSIVE_MAKER:
            return self._paper_passive(ctx)
        else:
            return self._paper_market(ctx)

    def _paper_market(self, ctx: ExecutionContext) -> ExecutionResult:
        """페이퍼: 시장가 시뮬레이션

        LONG → ask 가격에 체결 (taker)
        SHORT → bid 가격에 체결 (taker)
        """
        if ctx.side == 'LONG':
            fill_price = ctx.best_ask if ctx.best_ask > 0 else ctx.best_bid
        else:
            fill_price = ctx.best_bid if ctx.best_bid > 0 else ctx.best_ask

        # 호가 데이터 없으면 mid price
        if fill_price <= 0:
            fill_price = (ctx.best_bid + ctx.best_ask) / 2

        return ExecutionResult(
            success=True,
            mode=ExecutionMode.AGGRESSIVE_TAKER if ctx.cascade_conf >= self.AGGRESSIVE_MIN_CONF else ExecutionMode.FALLBACK_MARKET,
            filled_price=fill_price,
            filled_quantity=ctx.quantity,
            unfilled_quantity=0.0,
            order_ids=['paper_market'],
        )

    def _paper_passive(self, ctx: ExecutionContext) -> ExecutionResult:
        """페이퍼: Passive Maker 시뮬레이션

        지정가를 산출 후, 현재 캔들의 low/high로 체결 가능성 판단.
        - LONG: 캔들 low ≤ limit_price → 체결
        - SHORT: 캔들 high ≥ limit_price → 체결
        미체결 시 cascade_conf 재확인 → chase 또는 취소.
        """
        limit_price = self._calc_limit_price(ctx.side, ctx)

        logger.info(f"[EXEC] Paper Passive: {ctx.side} @ {limit_price:.2f} "
                     f"(bid={ctx.best_bid:.2f}, ask={ctx.best_ask:.2f}, obi={ctx.obi:+.2f})")

        # 캔들 low/high로 체결 판단
        filled = False
        if self._candle_low_high_getter:
            try:
                candle_low, candle_high = self._candle_low_high_getter()
                if ctx.side == 'LONG' and candle_low <= limit_price:
                    filled = True
                elif ctx.side == 'SHORT' and candle_high >= limit_price:
                    filled = True
            except Exception:
                pass

        if not filled:
            # 간단한 휴리스틱: 지정가가 best_bid/ask에 매우 가까우면 체결로 간주
            # (페이퍼에서는 보수적으로 처리)
            if ctx.side == 'LONG' and limit_price >= ctx.best_bid:
                filled = True
            elif ctx.side == 'SHORT' and limit_price <= ctx.best_ask:
                filled = True

        if filled:
            return ExecutionResult(
                success=True,
                mode=ExecutionMode.PASSIVE_MAKER,
                filled_price=limit_price,
                filled_quantity=ctx.quantity,
                unfilled_quantity=0.0,
                order_ids=['paper_passive'],
                fee_saved_bps=self.TAKER_FEE_BPS - self.MAKER_FEE_BPS,
            )

        # ── 미체결: chase 판단 ──
        current_conf = self._cascade_conf_getter() if self._cascade_conf_getter else ctx.cascade_conf

        if current_conf >= self.CHASE_MIN_CONF:
            logger.info(f"[EXEC] Paper 추격: conf={current_conf:.2f} → 시장가 전환")
            chase_result = self._paper_market(ctx)
            chase_result.was_chased = True
            chase_result.mode = ExecutionMode.PASSIVE_MAKER
            return chase_result
        else:
            logger.info(f"[EXEC] Paper 추격 포기: conf={current_conf:.2f}")
            return ExecutionResult(
                success=False,
                mode=ExecutionMode.PASSIVE_MAKER,
                filled_price=0.0,
                filled_quantity=0.0,
                unfilled_quantity=ctx.quantity,
                order_ids=['paper_passive_timeout'],
                error="Paper 미체결 + 신호 약화",
            )
