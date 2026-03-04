"""동적 전략 선택 시스템"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from strategies.base import BaseStrategy, Signal, TradeSignal
from regime.market_regime import MarketRegime, RegimeDetector, RegimeInfo
from strategy_selector.performance_tracker import PerformanceTracker, StrategyMetrics
from utils.logger import setup_logger

logger = setup_logger(__name__)


class StrategySelector:
    """
    동적 전략 선택기

    시장 상태와 전략 성과를 기반으로 최적의 전략을 선택합니다.
    """

    def __init__(
        self,
        strategies: Dict[str, BaseStrategy],
        regime_detector: Optional[RegimeDetector] = None,
        performance_tracker: Optional[PerformanceTracker] = None,
        params: Optional[Dict] = None
    ):
        """
        Args:
            strategies: {전략명: 전략 객체} 딕셔너리
            regime_detector: 시장 상태 감지기
            performance_tracker: 성과 추적기
            params: 선택기 파라미터
        """
        self.strategies = strategies
        self.regime_detector = regime_detector or RegimeDetector()
        self.performance_tracker = performance_tracker or PerformanceTracker()

        default_params = {
            'regime_weight': 0.5,        # 시장 상태 가중치
            'performance_weight': 0.5,   # 성과 가중치
            'min_trades_for_perf': 10,   # 성과 평가 최소 거래 수
            'lookback_trades': 50,       # 성과 평가 거래 수
            'rebalance_interval': 24     # 리밸런싱 간격 (시간)
        }

        if params:
            default_params.update(params)

        self.params = default_params

        # 전략 추적기에 등록
        for strategy_name in strategies.keys():
            self.performance_tracker.add_strategy(strategy_name)

        self.last_rebalance_time = datetime.now()
        self.current_weights: Dict[str, float] = {}

        logger.info(f"StrategySelector 초기화: {len(strategies)}개 전략")
        logger.info(f"  파라미터: {self.params}")

    def detect_market_regime(self, data: pd.DataFrame) -> RegimeInfo:
        """시장 상태 감지"""
        return self.regime_detector.detect_regime(data)

    def calculate_regime_weights(self, regime_info: RegimeInfo) -> Dict[str, float]:
        """
        시장 상태 기반 전략 가중치 계산

        Args:
            regime_info: 시장 상태 정보

        Returns:
            {전략명: 가중치} 딕셔너리
        """
        weights = {name: 0.0 for name in self.strategies.keys()}

        # 시장 상태에 적합한 전략 목록
        suitable_strategies = self.regime_detector.get_suitable_strategies(regime_info.regime)

        if not suitable_strategies:
            # 모든 전략에 동일 가중치
            equal_weight = 1.0 / len(self.strategies)
            return {name: equal_weight for name in self.strategies.keys()}

        # 적합한 전략에만 가중치 부여
        base_weight = 1.0 / len(suitable_strategies)

        for strategy_name in suitable_strategies:
            if strategy_name in weights:
                weights[strategy_name] = base_weight

        # 신뢰도 반영
        for strategy_name in weights:
            weights[strategy_name] *= regime_info.confidence

        logger.debug(f"시장 상태 기반 가중치: {weights}")

        return weights

    def calculate_performance_weights(self) -> Dict[str, float]:
        """
        성과 기반 전략 가중치 계산

        Returns:
            {전략명: 가중치} 딕셔너리
        """
        weights = {name: 0.0 for name in self.strategies.keys()}

        # 각 전략의 샤프 비율 수집
        sharpe_ratios = {}

        for strategy_name in self.strategies.keys():
            metrics = self.performance_tracker.get_metrics(strategy_name)

            if metrics is None or metrics.total_trades < self.params['min_trades_for_perf']:
                # 거래 수가 부족하면 평균 가중치
                sharpe_ratios[strategy_name] = 0.0
            else:
                sharpe_ratios[strategy_name] = max(metrics.sharpe_ratio, 0.0)

        # 샤프 비율이 모두 0이면 동일 가중치
        total_sharpe = sum(sharpe_ratios.values())

        if total_sharpe == 0:
            equal_weight = 1.0 / len(self.strategies)
            return {name: equal_weight for name in self.strategies.keys()}

        # 샤프 비율 기반 가중치 계산
        for strategy_name, sharpe in sharpe_ratios.items():
            weights[strategy_name] = sharpe / total_sharpe

        logger.debug(f"성과 기반 가중치: {weights}")

        return weights

    def calculate_combined_weights(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        시장 상태 + 성과 기반 통합 가중치 계산

        Args:
            data: OHLCV 데이터

        Returns:
            {전략명: 가중치} 딕셔너리
        """
        # 1. 시장 상태 감지
        regime_info = self.detect_market_regime(data)

        # 2. 시장 상태 기반 가중치
        regime_weights = self.calculate_regime_weights(regime_info)

        # 3. 성과 기반 가중치
        performance_weights = self.calculate_performance_weights()

        # 4. 통합 가중치
        combined_weights = {}

        for strategy_name in self.strategies.keys():
            regime_w = regime_weights.get(strategy_name, 0.0)
            perf_w = performance_weights.get(strategy_name, 0.0)

            combined = (
                regime_w * self.params['regime_weight'] +
                perf_w * self.params['performance_weight']
            )

            combined_weights[strategy_name] = combined

        # 정규화
        total_weight = sum(combined_weights.values())

        if total_weight > 0:
            combined_weights = {
                name: weight / total_weight
                for name, weight in combined_weights.items()
            }

        logger.info(f"통합 가중치: {combined_weights}")

        return combined_weights

    def select_best_strategy(self, data: pd.DataFrame) -> Tuple[str, BaseStrategy]:
        """
        최적 전략 선택

        Args:
            data: OHLCV 데이터

        Returns:
            (전략명, 전략 객체)
        """
        weights = self.calculate_combined_weights(data)

        # 가장 높은 가중치의 전략 선택
        best_strategy_name = max(weights, key=weights.get)
        best_strategy = self.strategies[best_strategy_name]

        logger.info(f"선택된 전략: {best_strategy_name} (가중치: {weights[best_strategy_name]:.2%})")

        return best_strategy_name, best_strategy

    def generate_ensemble_signal(self, data: pd.DataFrame) -> TradeSignal:
        """
        앙상블 신호 생성 (신뢰도 가중 다수결 + 최강 신호)

        Args:
            data: OHLCV 데이터

        Returns:
            통합 TradeSignal
        """
        # 가중치 계산
        weights = self.calculate_combined_weights(data)

        # 각 전략의 신호 수집
        signals = {}
        for strategy_name, strategy in self.strategies.items():
            try:
                signal = strategy.generate_signal(data)
                signals[strategy_name] = signal
            except Exception as e:
                logger.error(f"{strategy_name} 신호 생성 실패: {e}")
                continue

        if not signals:
            return TradeSignal(
                signal=Signal.HOLD,
                confidence=0.0,
                price=data['close'].iloc[-1],
                timestamp=pd.Timestamp.now(),
                reason="모든 전략 신호 생성 실패"
            )

        # 방법 1: 신뢰도 가중 투표 방식
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0

        for strategy_name, signal in signals.items():
            weight = weights.get(strategy_name, 0.0)

            if weight == 0.0:
                continue

            # 신뢰도 * 가중치로 투표
            vote_strength = signal.confidence * weight

            if signal.signal == Signal.BUY:
                buy_score += vote_strength
            elif signal.signal == Signal.SELL:
                sell_score += vote_strength
            else:  # HOLD
                hold_score += vote_strength

        # 방법 2: 최강 신호 찾기
        strongest_signal = None
        strongest_strength = 0.0

        for strategy_name, signal in signals.items():
            weight = weights.get(strategy_name, 0.0)
            strength = signal.confidence * weight

            if strength > strongest_strength and signal.signal != Signal.HOLD:
                strongest_strength = strength
                strongest_signal = signal

        # 최종 신호 결정 로직
        total_score = buy_score + sell_score + hold_score

        # 1) 한 방향이 절대적 우위 (60% 이상)
        if total_score > 0:
            buy_ratio = buy_score / total_score
            sell_ratio = sell_score / total_score

            if buy_ratio > 0.6:
                final_signal = Signal.BUY
                final_confidence = buy_score
                reason = f"다수결 BUY ({buy_ratio:.0%} 지지)"
            elif sell_ratio > 0.6:
                final_signal = Signal.SELL
                final_confidence = sell_score
                reason = f"다수결 SELL ({sell_ratio:.0%} 지지)"
            # 2) 과반수 (50% 이상)
            elif buy_ratio > sell_ratio and buy_ratio > 0.5:
                final_signal = Signal.BUY
                final_confidence = buy_score
                reason = f"과반수 BUY ({buy_ratio:.0%})"
            elif sell_ratio > buy_ratio and sell_ratio > 0.5:
                final_signal = Signal.SELL
                final_confidence = sell_score
                reason = f"과반수 SELL ({sell_ratio:.0%})"
            # 3) 박빙이면 최강 신호 선택
            elif strongest_signal and strongest_strength > 0.3:
                final_signal = strongest_signal.signal
                final_confidence = strongest_strength
                reason = f"최강 신호 {final_signal.name} (신뢰도: {strongest_strength:.2%})"
            # 4) 모두 약하면 HOLD
            else:
                final_signal = Signal.HOLD
                final_confidence = max(buy_score, sell_score, hold_score)
                reason = f"신호 불명확 (BUY:{buy_ratio:.0%} SELL:{sell_ratio:.0%})"
        else:
            final_signal = Signal.HOLD
            final_confidence = 0.0
            reason = "신호 없음"

        # 신호 상세 정보
        signal_details = {
            name: {
                'signal': signal.signal.name,
                'confidence': signal.confidence,
                'weight': weights.get(name, 0.0),
                'vote_strength': signal.confidence * weights.get(name, 0.0)
            }
            for name, signal in signals.items()
        }

        ensemble_signal = TradeSignal(
            signal=final_signal,
            confidence=final_confidence,
            price=data['close'].iloc[-1],
            timestamp=pd.Timestamp.now(),
            reason=reason,
            metadata={
                'buy_score': buy_score,
                'sell_score': sell_score,
                'hold_score': hold_score,
                'weighted_score': final_confidence,  # 가중 점수 추가
                'strategy_signals': signal_details,
                'weights': weights
            }
        )

        logger.info(f"앙상블 신호: {final_signal.name}, 신뢰도: {final_confidence:.2%}")

        return ensemble_signal

    def should_rebalance(self) -> bool:
        """리밸런싱 필요 여부 확인"""
        hours_since_rebalance = (datetime.now() - self.last_rebalance_time).total_seconds() / 3600

        return hours_since_rebalance >= self.params['rebalance_interval']

    def rebalance(self, data: pd.DataFrame):
        """전략 가중치 리밸런싱"""
        if not self.should_rebalance():
            return

        logger.info("전략 가중치 리밸런싱 시작")

        # 새로운 가중치 계산
        self.current_weights = self.calculate_combined_weights(data)

        self.last_rebalance_time = datetime.now()

        logger.info(f"리밸런싱 완료: {self.current_weights}")

    def get_status(self) -> Dict:
        """현재 상태 반환"""
        return {
            'strategies': list(self.strategies.keys()),
            'current_weights': self.current_weights,
            'last_rebalance': self.last_rebalance_time,
            'params': self.params
        }
