"""Phase 4: 전략 스위칭 시스템 테스트"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.binance_client import BinanceClient
from data.collectors.historical_collector import HistoricalDataCollector
from strategies.technical.ma_crossover import MACrossoverStrategy
from strategies.technical.rsi_strategy import RSIStrategy
from strategies.technical.bollinger_bands import BollingerBandsStrategy
from regime.market_regime import RegimeDetector, MarketRegime
from strategy_selector.performance_tracker import PerformanceTracker
from strategy_selector.strategy_selector import StrategySelector
from datetime import datetime

from utils.logger import setup_logger

logger = setup_logger(__name__)


def test_regime_detection():
    """시장 상태 감지 테스트"""
    logger.info("=" * 70)
    logger.info("[ 1/3 ] 시장 상태 감지 테스트")
    logger.info("=" * 70)

    # 데이터 수집
    client = BinanceClient(testnet=True)
    collector = HistoricalDataCollector(client)

    df = collector.get_recent_data('BTCUSDT', '1h', days=14)
    logger.info(f"✅ 데이터 수집 완료: {len(df)}개")

    # 시장 상태 감지
    detector = RegimeDetector()
    regime_info = detector.detect_regime(df)

    logger.info(f"시장 상태: {regime_info.regime.value}")
    logger.info(f"신뢰도: {regime_info.confidence:.2%}")
    logger.info(f"메트릭:")
    for key, value in regime_info.metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    # 적합한 전략 추천
    suitable_strategies = detector.get_suitable_strategies(regime_info.regime)
    logger.info(f"추천 전략: {suitable_strategies}")
    logger.info("")


def test_performance_tracker():
    """전략 성과 추적 테스트"""
    logger.info("=" * 70)
    logger.info("[ 2/3 ] 전략 성과 추적 테스트")
    logger.info("=" * 70)

    tracker = PerformanceTracker()

    # 전략 추가
    tracker.add_strategy('MA_Crossover')
    tracker.add_strategy('RSI')

    # 모의 거래 기록
    import random
    from datetime import timedelta

    base_time = datetime.now()

    for i in range(20):
        # MA Crossover 거래
        pnl_ma = random.uniform(-50, 100)
        tracker.record_trade(
            strategy_name='MA_Crossover',
            timestamp=base_time + timedelta(hours=i),
            side='BUY' if i % 2 == 0 else 'SELL',
            price=50000 + random.uniform(-1000, 1000),
            quantity=0.01,
            pnl=pnl_ma if i % 2 == 1 else None
        )

        # RSI 거래
        pnl_rsi = random.uniform(-30, 80)
        tracker.record_trade(
            strategy_name='RSI',
            timestamp=base_time + timedelta(hours=i),
            side='BUY' if i % 2 == 0 else 'SELL',
            price=50000 + random.uniform(-1000, 1000),
            quantity=0.01,
            pnl=pnl_rsi if i % 2 == 1 else None
        )

    # 성과 계산
    tracker.calculate_metrics('MA_Crossover', initial_capital=10000)
    tracker.calculate_metrics('RSI', initial_capital=10000)

    # 요약 출력
    logger.info(tracker.get_summary())

    # 전략 비교
    comparison = tracker.compare_strategies(initial_capital=10000)
    logger.info("전략 순위:")
    for rank, (name, sharpe, return_pct) in enumerate(comparison, 1):
        logger.info(f"  {rank}. {name}: 샤프 {sharpe:.2f}, 수익률 {return_pct:.2%}")
    logger.info("")


def test_strategy_selector():
    """동적 전략 선택 테스트"""
    logger.info("=" * 70)
    logger.info("[ 3/3 ] 동적 전략 선택 테스트")
    logger.info("=" * 70)

    # 데이터 수집
    client = BinanceClient(testnet=True)
    collector = HistoricalDataCollector(client)

    df = collector.get_recent_data('BTCUSDT', '1h', days=14)
    logger.info(f"✅ 데이터 수집 완료: {len(df)}개")

    # 전략 생성
    strategies = {
        'MA_Crossover': MACrossoverStrategy(),
        'RSI': RSIStrategy(),
        'BollingerBands': BollingerBandsStrategy()
    }

    # 전략 선택기 생성
    selector = StrategySelector(
        strategies=strategies,
        params={
            'regime_weight': 0.6,
            'performance_weight': 0.4,
            'min_trades_for_perf': 5
        }
    )

    # 시장 상태 감지
    regime_info = selector.detect_market_regime(df)
    logger.info(f"시장 상태: {regime_info.regime.value} (신뢰도: {regime_info.confidence:.2%})")

    # 가중치 계산
    weights = selector.calculate_combined_weights(df)
    logger.info(f"전략 가중치:")
    for name, weight in weights.items():
        logger.info(f"  {name}: {weight:.2%}")

    # 최적 전략 선택
    best_name, best_strategy = selector.select_best_strategy(df)
    logger.info(f"선택된 전략: {best_name}")

    # 선택된 전략으로 신호 생성
    signal = best_strategy.generate_signal(df)
    logger.info(f"신호: {signal.signal.name}")
    logger.info(f"신뢰도: {signal.confidence:.2%}")
    logger.info(f"이유: {signal.reason}")
    logger.info("")

    # 앙상블 신호 생성
    logger.info("앙상블 신호 생성:")
    ensemble_signal = selector.generate_ensemble_signal(df)
    logger.info(f"신호: {ensemble_signal.signal.name}")
    logger.info(f"신뢰도: {ensemble_signal.confidence:.2%}")
    logger.info(f"이유: {ensemble_signal.reason}")
    logger.info(f"가중 스코어: {ensemble_signal.metadata['weighted_score']:.2f}")

    # 각 전략 신호 상세
    logger.info("각 전략 신호:")
    for name, details in ensemble_signal.metadata['strategy_signals'].items():
        logger.info(f"  {name}: {details['signal']} (신뢰도: {details['confidence']:.2%}, 가중치: {details['weight']:.2%})")
    logger.info("")


def main():
    """메인 테스트 함수"""
    logger.info("=" * 70)
    logger.info("Phase 4: 전략 스위칭 시스템 테스트")
    logger.info("=" * 70)
    logger.info("")

    try:
        # 1. 시장 상태 감지 테스트
        test_regime_detection()

        # 2. 성과 추적 테스트
        test_performance_tracker()

        # 3. 동적 전략 선택 테스트
        test_strategy_selector()

        logger.info("=" * 70)
        logger.info("✅ Phase 4 테스트 완료!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"테스트 실패: {e}", exc_info=True)
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
