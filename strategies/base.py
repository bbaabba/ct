"""거래 전략 기본 인터페이스"""
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)


class Signal(Enum):
    """거래 신호"""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradeSignal:
    """거래 신호 상세 정보"""
    signal: Signal
    confidence: float  # 0.0 ~ 1.0
    price: float
    timestamp: pd.Timestamp
    reason: str
    metadata: Optional[Dict[str, Any]] = None


class BaseStrategy(ABC):
    """
    거래 전략 기본 클래스

    모든 전략은 이 클래스를 상속받아 구현해야 합니다.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            name: 전략 이름
            params: 전략 파라미터 (각 전략마다 다름)
        """
        self.name = name
        self.params = params or {}
        self.is_fitted = False

        logger.info(f"{self.name} 전략 초기화 (파라미터: {self.params})")

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> TradeSignal:
        """
        거래 신호 생성

        Args:
            data: OHLCV 데이터프레임 (최신 데이터 포함)

        Returns:
            TradeSignal 객체
        """
        pass

    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        전략에 필요한 기술 지표 계산

        Args:
            data: 원본 OHLCV 데이터프레임

        Returns:
            지표가 추가된 데이터프레임
        """
        pass

    def fit(self, data: pd.DataFrame) -> 'BaseStrategy':
        """
        전략 학습/초기화 (필요한 경우)

        Args:
            data: 학습용 데이터

        Returns:
            self (메서드 체이닝용)
        """
        logger.info(f"{self.name} 전략 학습 시작 (데이터: {len(data)}개)")
        self.is_fitted = True
        return self

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        입력 데이터 유효성 검사

        Args:
            data: 검사할 데이터프레임

        Returns:
            유효 여부
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        if data is None or len(data) == 0:
            logger.error(f"{self.name}: 빈 데이터프레임")
            return False

        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            logger.error(f"{self.name}: 필수 컬럼 누락: {missing_cols}")
            return False

        return True

    def get_position_size(self, confidence: float, account_balance: float) -> float:
        """
        신뢰도 기반 포지션 크기 계산

        Args:
            confidence: 신호 신뢰도 (0.0 ~ 1.0)
            account_balance: 계좌 잔고

        Returns:
            포지션 크기 (USDT)
        """
        # 기본: 신뢰도에 비례하여 포지션 크기 결정
        max_position_pct = self.params.get('max_position_pct', 0.2)
        min_confidence = self.params.get('min_confidence', 0.6)

        if confidence < min_confidence:
            return 0.0

        # 신뢰도에 따라 선형적으로 포지션 크기 증가
        position_pct = max_position_pct * confidence
        position_size = account_balance * position_pct

        return position_size

    def __repr__(self) -> str:
        return f"{self.name}Strategy(params={self.params})"
