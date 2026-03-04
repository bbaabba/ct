"""봇 상태 JSON 파일 읽기 모듈 (대시보드용)"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd


class BotStateReader:
    """
    BotStateWriter가 기록한 JSON 파일에서 봇 상태를 읽습니다.

    파일이 없거나 손상된 경우 안전하게 None/빈값을 반환합니다.
    """

    def __init__(self, state_dir: str = "bot_state"):
        self.state_dir = Path(state_dir)

    def _read_json(self, filepath: Path) -> Optional[Dict]:
        """JSON 파일 안전 읽기"""
        try:
            if not filepath.exists():
                return None
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, OSError):
            return None

    def read_status(self) -> Optional[Dict[str, Any]]:
        """봇 상태 읽기"""
        return self._read_json(self.state_dir / "status.json")

    def read_trade_history(self) -> List[Dict[str, Any]]:
        """거래 내역 읽기 (없으면 빈 리스트)"""
        data = self._read_json(self.state_dir / "trade_history.json")
        if data and 'trades' in data:
            return data['trades']
        return []

    def read_candles(self) -> Optional[pd.DataFrame]:
        """캔들 데이터를 DataFrame으로 읽기"""
        data = self._read_json(self.state_dir / "candles.json")
        if data and 'candles' in data and len(data['candles']) > 0:
            df = pd.DataFrame(data['candles'])
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return None

    def read_missed_opportunities(self) -> List[Dict[str, Any]]:
        """놓친 기회 내역 읽기 (없으면 빈 리스트)"""
        data = self._read_json(self.state_dir / "missed_opportunities.json")
        if data and 'missed_opportunities' in data:
            return data['missed_opportunities']
        return []

    def get_data_age_seconds(self) -> Optional[float]:
        """상태 데이터의 경과 시간(초) 반환"""
        status = self.read_status()
        if status and '_written_at' in status:
            try:
                written = datetime.fromisoformat(status['_written_at'])
                return (datetime.now() - written).total_seconds()
            except (ValueError, TypeError):
                return None
        return None

    def is_bot_alive(self, max_age_seconds: float = 30.0) -> bool:
        """봇이 살아있는지 확인 (상태 데이터 신선도 기반)"""
        age = self.get_data_age_seconds()
        if age is None:
            return False
        return age < max_age_seconds
