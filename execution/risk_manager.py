"""리스크 관리 시스템"""
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class RiskLimits:
    """리스크 제한 설정"""
    max_position_size_pct: float = 0.2  # 최대 포지션 크기 (계좌의 20%)
    max_daily_loss_pct: float = 0.05  # 최대 일일 손실 (5%)
    max_total_exposure_pct: float = 0.8  # 최대 총 노출 (80%)
    stop_loss_pct: float = 0.01  # 스톱로스 (1%)
    take_profit_pct: float = 0.02  # 이익실현 (2%)
    kelly_fraction: float = 0.25  # Kelly Criterion 적용 비율 (보수적)


class RiskManager:
    """
    리스크 관리자

    주요 기능:
    - 포지션 크기 계산 (Kelly Criterion)
    - Stop-Loss / Take-Profit 설정
    - 일일 손실 제한 확인
    - 총 노출 관리
    """

    def __init__(self, initial_balance: float, limits: Optional[RiskLimits] = None):
        """
        Args:
            initial_balance: 초기 계좌 잔고 (USDT)
            limits: 리스크 제한 설정
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.limits = limits or RiskLimits()

        self.positions: Dict[str, Position] = {}
        self.daily_pnl: float = 0.0
        self.last_reset_date = datetime.now().date()

        logger.info(f"RiskManager 초기화: 잔고 {initial_balance:.2f} USDT")
        logger.info(f"리스크 설정: {self.limits}")

    def reset_daily_stats(self):
        """일일 통계 초기화"""
        today = datetime.now().date()

        if today > self.last_reset_date:
            logger.info(f"일일 통계 초기화 (일일 손익: {self.daily_pnl:.2f} USDT)")
            self.daily_pnl = 0.0
            self.last_reset_date = today

    def calculate_kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        confidence: float = 1.0
    ) -> float:
        """
        Kelly Criterion을 사용한 포지션 크기 계산

        Kelly% = W - [(1-W) / R]
        - W: 승률
        - R: 평균 이익/평균 손실 비율

        Args:
            win_rate: 승률 (0.0 ~ 1.0)
            avg_win: 평균 이익 (비율)
            avg_loss: 평균 손실 (비율)
            confidence: 신호 신뢰도 (0.0 ~ 1.0)

        Returns:
            포지션 비율 (0.0 ~ 1.0)
        """
        if avg_loss == 0:
            logger.warning("평균 손실이 0입니다. Kelly 계산 불가")
            return 0.0

        # Win/Loss Ratio
        win_loss_ratio = avg_win / avg_loss

        # Kelly Percentage
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)

        # 음수면 0
        if kelly_pct < 0:
            kelly_pct = 0.0

        # 보수적 적용 (일반적으로 Full Kelly의 25~50% 사용)
        kelly_pct = kelly_pct * self.limits.kelly_fraction

        # 신뢰도 적용
        kelly_pct = kelly_pct * confidence

        # 최대 포지션 크기 제한
        kelly_pct = min(kelly_pct, self.limits.max_position_size_pct)

        return kelly_pct

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        confidence: float,
        win_rate: float = 0.55,
        avg_win: float = 0.02,
        avg_loss: float = 0.01
    ) -> Dict[str, float]:
        """
        포지션 크기 계산

        Args:
            symbol: 거래 심볼
            price: 현재 가격
            confidence: 신호 신뢰도
            win_rate: 예상 승률
            avg_win: 예상 평균 이익
            avg_loss: 예상 평균 손실

        Returns:
            {
                'position_size_usdt': USDT 금액,
                'quantity': 수량,
                'stop_loss': 스톱로스 가격,
                'take_profit': 이익실현 가격
            }
        """
        # 일일 통계 초기화
        self.reset_daily_stats()

        # 일일 손실 제한 확인
        if self.daily_pnl < 0:
            daily_loss_pct = abs(self.daily_pnl) / self.initial_balance

            if daily_loss_pct >= self.limits.max_daily_loss_pct:
                logger.warning(f"일일 손실 제한 도달 ({daily_loss_pct:.2%} >= {self.limits.max_daily_loss_pct:.2%})")
                return {
                    'position_size_usdt': 0.0,
                    'quantity': 0.0,
                    'stop_loss': 0.0,
                    'take_profit': 0.0
                }

        # Kelly Criterion 포지션 크기
        kelly_pct = self.calculate_kelly_size(win_rate, avg_win, avg_loss, confidence)

        # USDT 포지션 크기
        position_size_usdt = self.current_balance * kelly_pct

        # 총 노출 확인
        current_exposure = sum(pos.entry_price * pos.quantity for pos in self.positions.values())
        max_exposure = self.current_balance * self.limits.max_total_exposure_pct

        if current_exposure + position_size_usdt > max_exposure:
            logger.warning(f"총 노출 제한 초과 (현재: {current_exposure:.2f}, 추가: {position_size_usdt:.2f}, 한도: {max_exposure:.2f})")
            position_size_usdt = max(0, max_exposure - current_exposure)

        # 수량 계산
        quantity = position_size_usdt / price

        # Stop-Loss 및 Take-Profit 설정
        stop_loss = price * (1 - self.limits.stop_loss_pct)
        take_profit = price * (1 + self.limits.take_profit_pct)

        result = {
            'position_size_usdt': position_size_usdt,
            'quantity': quantity,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'kelly_pct': kelly_pct
        }

        logger.info(f"포지션 크기 계산 ({symbol}): {result}")

        return result

    def add_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        """포지션 추가"""
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        self.positions[symbol] = position

        logger.info(f"포지션 추가: {symbol} {side} {quantity:.4f} @ {entry_price:.2f}")

    def close_position(
        self,
        symbol: str,
        exit_price: float
    ) -> Optional[float]:
        """
        포지션 청산

        Args:
            symbol: 거래 심볼
            exit_price: 청산 가격

        Returns:
            손익 (USDT)
        """
        if symbol not in self.positions:
            logger.warning(f"포지션을 찾을 수 없습니다: {symbol}")
            return None

        position = self.positions[symbol]

        # 손익 계산
        if position.side == 'BUY':
            pnl = (exit_price - position.entry_price) * position.quantity
        else:  # SELL
            pnl = (position.entry_price - exit_price) * position.quantity

        # 잔고 및 일일 손익 업데이트
        self.current_balance += pnl
        self.daily_pnl += pnl

        logger.info(f"포지션 청산: {symbol} @ {exit_price:.2f}, 손익: {pnl:.2f} USDT")
        logger.info(f"현재 잔고: {self.current_balance:.2f} USDT, 일일 손익: {self.daily_pnl:.2f} USDT")

        # 포지션 제거
        del self.positions[symbol]

        return pnl

    def check_stop_loss_take_profit(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[str]:
        """
        Stop-Loss / Take-Profit 확인

        Args:
            symbol: 거래 심볼
            current_price: 현재 가격

        Returns:
            'STOP_LOSS', 'TAKE_PROFIT', 또는 None
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        # Stop-Loss 확인
        if position.stop_loss and current_price <= position.stop_loss:
            logger.warning(f"Stop-Loss 도달: {symbol} @ {current_price:.2f} (SL: {position.stop_loss:.2f})")
            return 'STOP_LOSS'

        # Take-Profit 확인
        if position.take_profit and current_price >= position.take_profit:
            logger.info(f"Take-Profit 도달: {symbol} @ {current_price:.2f} (TP: {position.take_profit:.2f})")
            return 'TAKE_PROFIT'

        return None

    def get_total_exposure(self) -> float:
        """총 노출 계산 (USDT)"""
        return sum(pos.entry_price * pos.quantity for pos in self.positions.values())

    def get_total_pnl(self, current_prices: Dict[str, float]) -> float:
        """총 미실현 손익 계산"""
        total_pnl = 0.0

        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]

            if position.side == 'BUY':
                pnl = (current_price - position.entry_price) * position.quantity
            else:  # SELL
                pnl = (position.entry_price - current_price) * position.quantity

            total_pnl += pnl

        return total_pnl

    def get_status(self) -> Dict:
        """현재 상태 반환"""
        return {
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'total_pnl': self.current_balance - self.initial_balance,
            'total_pnl_pct': (self.current_balance - self.initial_balance) / self.initial_balance,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.initial_balance,
            'num_positions': len(self.positions),
            'total_exposure': self.get_total_exposure(),
            'exposure_pct': self.get_total_exposure() / self.current_balance if self.current_balance > 0 else 0,
        }
