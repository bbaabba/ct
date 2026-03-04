"""텔레그램 실시간 알림 모듈 — cascade_conf / fee_ratio 모니터링"""
import asyncio
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional

import urllib.request
import urllib.parse
import json

from utils.logger import setup_logger

logger = setup_logger(__name__)


class TelegramNotifier:
    """
    텔레그램 봇 API를 통한 실시간 거래 알림

    알림 종류:
    - 매 사이클 요약 (cascade_conf, fee_ratio, 방향)
    - 진입/청산 이벤트
    - Fee Hard Block 발동
    - 에러/경고
    """

    def __init__(self, bot_token: str, chat_id: str, throttle_seconds: float = 30.0):
        """
        Args:
            bot_token: 텔레그램 봇 토큰
            chat_id: 수신 채팅 ID
            throttle_seconds: 일반 사이클 요약 최소 전송 간격 (초)
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.throttle_seconds = throttle_seconds
        self._enabled = bool(bot_token and chat_id)
        self._last_summary_time = 0.0
        self._last_fee_block_time = 0.0
        self._send_queue: list = []
        self._lock = threading.Lock()

        if self._enabled:
            logger.info(f"TelegramNotifier 활성화 (chat_id={chat_id[:6]}...)")
        else:
            logger.info("TelegramNotifier 비활성 (토큰/chat_id 미설정)")

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── 전송 API ──────────────────────────────────────────────

    def _send_message(self, text: str, parse_mode: str = "HTML"):
        """동기 HTTP로 텔레그램 메시지 전송 (별도 스레드 권장)"""
        if not self._enabled:
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status != 200:
                    logger.warning(f"텔레그램 전송 실패: HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"텔레그램 전송 오류: {e}")

    def _send_async(self, text: str):
        """비동기 전송 (메인 루프 차단 방지)"""
        t = threading.Thread(target=self._send_message, args=(text,), daemon=True)
        t.start()

    # ── 사이클 요약 ──────────────────────────────────────────

    def notify_cycle(self, ctx: Dict[str, Any], price: float):
        """매 사이클 핵심 지표 전송 (throttle 적용)"""
        if not self._enabled:
            return
        now = time.time()
        if now - self._last_summary_time < self.throttle_seconds:
            return
        self._last_summary_time = now

        cascade_dir = ctx.get('cascade_dir') or '-'
        cascade_conf = ctx.get('cascade_conf') or 0
        cascade_fire = ctx.get('cascade_fire', False)
        fee_ev = ctx.get('fee_ev_ratio') or 0
        final = ctx.get('final_score') or 0
        regime = ctx.get('regime', '-')
        action = ctx.get('action', 'HOLD')
        side = ctx.get('side', '-')

        fire_icon = "🔥" if cascade_fire else "⏳"
        dir_icon = {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "⚪"}.get(cascade_dir, "⚪")

        msg = (
            f"<b>📊 Cycle</b>  {datetime.now().strftime('%H:%M:%S')}\n"
            f"💰 <code>{price:,.2f}</code> USDT\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"{dir_icon} Cascade: <b>{cascade_dir}</b> conf=<code>{cascade_conf:.3f}</code> {fire_icon}\n"
            f"💵 Fee EV: <code>{fee_ev:.1f}x</code>\n"
            f"📈 Final: <code>{final:.3f}</code>\n"
            f"🌐 Regime: <code>{regime}</code>\n"
            f"▶️ {action} / {side}"
        )
        self._send_async(msg)

    # ── 이벤트 알림 (즉시 전송, throttle 없음) ───────────────

    def notify_entry(self, side: str, price: float, size_pct: float,
                     cascade_conf: float, fee_ev: float,
                     sl: Optional[float] = None, tp: Optional[float] = None):
        """진입 알림"""
        if not self._enabled:
            return
        icon = "🟢" if side == "LONG" else "🔴"
        msg = (
            f"{icon} <b>진입 {side}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 Price: <code>{price:,.2f}</code>\n"
            f"📐 Size: <code>{size_pct:.1%}</code>\n"
            f"🔗 Cascade: <code>{cascade_conf:.3f}</code>\n"
            f"💵 Fee EV: <code>{fee_ev:.1f}x</code>\n"
        )
        if sl:
            msg += f"⛔ SL: <code>{sl:,.2f}</code>\n"
        if tp:
            msg += f"🎯 TP: <code>{tp:,.2f}</code>"
        self._send_async(msg)

    def notify_exit(self, side: str, entry_price: float, exit_price: float,
                    pnl: float, pnl_pct: float, reason: str):
        """청산 알림"""
        if not self._enabled:
            return
        icon = "✅" if pnl >= 0 else "❌"
        msg = (
            f"{icon} <b>청산 {side}</b> — {reason}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📥 Entry: <code>{entry_price:,.2f}</code>\n"
            f"📤 Exit:  <code>{exit_price:,.2f}</code>\n"
            f"💰 PnL: <code>{pnl:+,.2f}</code> USDT (<code>{pnl_pct:+.2%}</code>)"
        )
        self._send_async(msg)

    def notify_fee_block(self, expected_move: float, cost: float, ratio: float):
        """Fee Hard Block 발동 알림 (60초 throttle)"""
        if not self._enabled:
            return
        now = time.time()
        if now - self._last_fee_block_time < 60.0:
            return
        self._last_fee_block_time = now

        msg = (
            f"🚫 <b>Fee Hard Block</b>\n"
            f"예상변동 <code>{expected_move:.4%}</code> / "
            f"비용 <code>{cost:.4%}</code> = <code>{ratio:.1f}x</code> &lt; 1.5x"
        )
        self._send_async(msg)

    def notify_error(self, error_msg: str):
        """에러 알림 (60초 throttle)"""
        if not self._enabled:
            return
        msg = f"🚨 <b>Error</b>\n<code>{error_msg[:500]}</code>"
        self._send_async(msg)
