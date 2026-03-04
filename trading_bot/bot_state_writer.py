"""봇 상태를 JSON 파일로 기록하는 모듈 (대시보드용)"""
import json
import os
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd

from utils.logger import setup_logger

logger = setup_logger(__name__)


class BotStateWriter:
    """
    봇 상태를 JSON 파일로 원자적 기록

    대시보드가 별도 프로세스에서 이 파일들을 읽어 실시간 상태를 표시합니다.
    원자적 쓰기(tempfile + os.replace)로 읽기/쓰기 경합을 방지합니다.
    """

    def __init__(self, state_dir: str = "bot_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)

        self.status_path = self.state_dir / "status.json"
        self.trade_history_path = self.state_dir / "trade_history.json"
        self.candles_path = self.state_dir / "candles.json"
        self.missed_opps_path = self.state_dir / "missed_opportunities.json"

        # 쓰기 주기 제어
        self._last_status_write = 0.0
        self._last_candle_write = 0.0
        self._last_missed_write = 0.0
        self.status_interval = 2.0   # 상태 기록 주기 (초)
        self.candle_interval = 5.0   # 캔들 기록 주기 (초)
        self.missed_interval = 10.0  # 놓친 기회 기록 주기 (초)

        logger.info(f"BotStateWriter 초기화: {self.state_dir}")

    def _atomic_write(self, filepath: Path, data: Any):
        """원자적 파일 쓰기 (tempfile → os.replace)"""
        tmp_fd = None
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.state_dir), suffix=".tmp"
            )
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                tmp_fd = None  # fdopen이 소유권 가져감
                json.dump(data, f, ensure_ascii=False, default=str)
            os.replace(tmp_path, str(filepath))
            tmp_path = None  # replace 성공 시 정리 불필요
        except Exception as e:
            logger.warning(f"상태 파일 쓰기 실패 ({filepath.name}): {e}")
        finally:
            if tmp_fd is not None:
                os.close(tmp_fd)
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def write_status(self, status: Dict[str, Any]):
        """봇 상태 스냅샷 기록"""
        status['_written_at'] = datetime.now().isoformat()
        self._atomic_write(self.status_path, status)

    def write_trade_history(self, trades: List[Dict[str, Any]]):
        """거래 내역 기록 (최근 500건)"""
        recent_trades = trades[-500:] if len(trades) > 500 else trades
        self._atomic_write(self.trade_history_path, {
            'trades': recent_trades,
            'total_count': len(trades),
            '_written_at': datetime.now().isoformat()
        })

    def write_candles(self, candles_df: Optional[pd.DataFrame]):
        """캔들 데이터 기록 (최근 200개)"""
        if candles_df is None or candles_df.empty:
            return

        df = candles_df.tail(200).copy()

        # timestamp 인덱스를 컬럼으로 변환
        if 'timestamp' not in df.columns:
            df = df.reset_index()

        records = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')

        self._atomic_write(self.candles_path, {
            'candles': records,
            'count': len(records),
            '_written_at': datetime.now().isoformat()
        })

    def write_missed_opportunities(self, missed_opps: List[Dict[str, Any]]):
        """놓친 기회 내역 기록 (최근 50건)"""
        if not missed_opps:
            return
        self._atomic_write(self.missed_opps_path, {
            'missed_opportunities': missed_opps,
            'total_count': len(missed_opps),
            '_written_at': datetime.now().isoformat()
        })

    def periodic_write(self, bot) -> None:
        """
        주기적으로 상태를 기록 (거래 루프에서 호출)

        Args:
            bot: FuturesTradingBot 인스턴스
        """
        now = time.time()

        # 상태 기록 (매 2초)
        if now - self._last_status_write >= self.status_interval:
            self.write_status(bot.get_status())
            self._last_status_write = now

        # 캔들 기록 (매 5초)
        if now - self._last_candle_write >= self.candle_interval:
            if bot.data_manager:
                self.write_candles(bot.data_manager.get_candles_df())
            self._last_candle_write = now

        # 놓친 기회 기록 (매 10초)
        if now - self._last_missed_write >= self.missed_interval:
            if hasattr(bot, '_missed_opportunities') and bot._missed_opportunities:
                self.write_missed_opportunities(bot._missed_opportunities)
            self._last_missed_write = now

    def cleanup(self):
        """상태 파일 정리"""
        for path in [self.status_path, self.trade_history_path, self.candles_path, self.missed_opps_path]:
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass
