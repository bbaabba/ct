"""전략 성과 추적 시스템"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Trade:
    """거래 기록"""
    timestamp: datetime
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    pnl: Optional[float] = None


@dataclass
class StrategyMetrics:
    """전략 성과 메트릭"""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_return: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)


class PerformanceTracker:
    """
    전략 성과 추적기

    각 전략의 거래 기록과 성과 지표를 추적합니다.
    """

    def __init__(self):
        self.strategies: Dict[str, StrategyMetrics] = {}
        logger.info("PerformanceTracker 초기화")

    def add_strategy(self, strategy_name: str):
        """전략 추가"""
        if strategy_name not in self.strategies:
            self.strategies[strategy_name] = StrategyMetrics(strategy_name=strategy_name)
            logger.info(f"전략 추가: {strategy_name}")

    def record_trade(
        self,
        strategy_name: str,
        timestamp: datetime,
        side: str,
        price: float,
        quantity: float,
        pnl: Optional[float] = None
    ):
        """
        거래 기록

        Args:
            strategy_name: 전략 이름
            timestamp: 거래 시각
            side: 매수/매도
            price: 거래 가격
            quantity: 수량
            pnl: 손익 (청산 시)
        """
        if strategy_name not in self.strategies:
            self.add_strategy(strategy_name)

        trade = Trade(
            timestamp=timestamp,
            side=side,
            price=price,
            quantity=quantity,
            pnl=pnl
        )

        metrics = self.strategies[strategy_name]
        metrics.trades.append(trade)
        metrics.total_trades += 1

        # PnL이 있으면 승/패 기록
        if pnl is not None:
            if pnl > 0:
                metrics.winning_trades += 1
            elif pnl < 0:
                metrics.losing_trades += 1

            metrics.total_pnl += pnl

        logger.debug(f"{strategy_name}: 거래 기록 ({side} {quantity:.4f} @ {price:.2f}, PnL: {pnl})")

    def calculate_metrics(self, strategy_name: str, initial_capital: float = 10000.0):
        """
        성과 메트릭 계산

        Args:
            strategy_name: 전략 이름
            initial_capital: 초기 자본
        """
        if strategy_name not in self.strategies:
            logger.warning(f"전략을 찾을 수 없습니다: {strategy_name}")
            return

        metrics = self.strategies[strategy_name]

        if metrics.total_trades == 0:
            logger.warning(f"{strategy_name}: 거래 기록 없음")
            return

        # 1. 승률
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades

        # 2. 평균 이익/손실
        wins = [t.pnl for t in metrics.trades if t.pnl is not None and t.pnl > 0]
        losses = [t.pnl for t in metrics.trades if t.pnl is not None and t.pnl < 0]

        if wins:
            metrics.avg_win = np.mean(wins)
        if losses:
            metrics.avg_loss = abs(np.mean(losses))

        # 3. Profit Factor (총 이익 / 총 손실)
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0

        if total_losses > 0:
            metrics.profit_factor = total_wins / total_losses

        # 4. 총 수익률
        metrics.total_return = metrics.total_pnl / initial_capital

        # 5. 샤프 비율 (일일 수익률 기준)
        if len(metrics.daily_returns) > 1:
            returns_array = np.array(metrics.daily_returns)
            avg_return = np.mean(returns_array)
            std_return = np.std(returns_array)

            if std_return > 0:
                # 연율화 (252 거래일 가정)
                metrics.sharpe_ratio = (avg_return / std_return) * np.sqrt(252)

        # 6. 최대 낙폭 (Maximum Drawdown)
        if metrics.trades:
            cumulative_pnl = 0.0
            peak = 0.0
            max_dd = 0.0

            for trade in metrics.trades:
                if trade.pnl is not None:
                    cumulative_pnl += trade.pnl
                    peak = max(peak, cumulative_pnl)
                    drawdown = (peak - cumulative_pnl) / initial_capital if initial_capital > 0 else 0
                    max_dd = max(max_dd, drawdown)

            metrics.max_drawdown = max_dd

        logger.info(f"{strategy_name} 성과 메트릭 계산 완료:")
        logger.info(f"  총 거래: {metrics.total_trades}, 승률: {metrics.win_rate:.2%}")
        logger.info(f"  총 손익: {metrics.total_pnl:.2f}, 수익률: {metrics.total_return:.2%}")
        logger.info(f"  샤프 비율: {metrics.sharpe_ratio:.2f}, 최대 낙폭: {metrics.max_drawdown:.2%}")

    def get_metrics(self, strategy_name: str) -> Optional[StrategyMetrics]:
        """전략 메트릭 조회"""
        return self.strategies.get(strategy_name)

    def get_all_metrics(self) -> Dict[str, StrategyMetrics]:
        """모든 전략 메트릭 조회"""
        return self.strategies

    def compare_strategies(self, initial_capital: float = 10000.0) -> List[tuple]:
        """
        전략 비교 (성과 순위)

        Returns:
            [(전략명, 샤프비율, 수익률), ...]
        """
        results = []

        for name, metrics in self.strategies.items():
            self.calculate_metrics(name, initial_capital)
            results.append((name, metrics.sharpe_ratio, metrics.total_return))

        # 샤프 비율 기준 내림차순 정렬
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def reset_strategy(self, strategy_name: str):
        """전략 통계 초기화"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name] = StrategyMetrics(strategy_name=strategy_name)
            logger.info(f"{strategy_name} 통계 초기화")

    def get_summary(self) -> str:
        """요약 통계"""
        summary = "=" * 70 + "\n"
        summary += "전략 성과 요약\n"
        summary += "=" * 70 + "\n\n"

        for name, metrics in self.strategies.items():
            summary += f"[{name}]\n"
            summary += f"  총 거래: {metrics.total_trades}\n"
            summary += f"  승률: {metrics.win_rate:.2%}\n"
            summary += f"  총 손익: {metrics.total_pnl:.2f}\n"
            summary += f"  수익률: {metrics.total_return:.2%}\n"
            summary += f"  샤프 비율: {metrics.sharpe_ratio:.2f}\n"
            summary += f"  최대 낙폭: {metrics.max_drawdown:.2%}\n"
            summary += f"  Profit Factor: {metrics.profit_factor:.2f}\n"
            summary += "\n"

        return summary
