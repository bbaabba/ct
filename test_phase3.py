"""Phase 3: 거래 전략 시스템 테스트"""
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
from execution.risk_manager import RiskManager, RiskLimits

from utils.logger import setup_logger

logger = setup_logger(__name__)


def test_ma_crossover_strategy():
    """이동평균 교차 전략 테스트"""
    logger.info("=" * 70)
    logger.info("[ 1/4 ] MA Crossover 전략 테스트")
    logger.info("=" * 70)

    # 데이터 수집
    client = BinanceClient(testnet=True)
    collector = HistoricalDataCollector(client)

    df = collector.get_recent_data('BTCUSDT', '1h', days=7)
    logger.info(f"✅ 데이터 수집 완료: {len(df)}개")

    # 전략 생성
    strategy = MACrossoverStrategy(params={
        'fast_period': 7,
        'slow_period': 25,
        'ma_type': 'EMA'
    })

    # 신호 생성
    signal = strategy.generate_signal(df)

    logger.info(f"신호: {signal.signal.name}")
    logger.info(f"신뢰도: {signal.confidence:.2%}")
    logger.info(f"가격: {signal.price:.2f}")
    logger.info(f"이유: {signal.reason}")
    logger.info(f"메타데이터: {signal.metadata}")
    logger.info("")


def test_rsi_strategy():
    """RSI 전략 테스트"""
    logger.info("=" * 70)
    logger.info("[ 2/4 ] RSI 전략 테스트")
    logger.info("=" * 70)

    # 데이터 수집
    client = BinanceClient(testnet=True)
    collector = HistoricalDataCollector(client)

    df = collector.get_recent_data('BTCUSDT', '1h', days=7)
    logger.info(f"✅ 데이터 수집 완료: {len(df)}개")

    # 전략 생성
    strategy = RSIStrategy(params={
        'rsi_period': 14,
        'oversold_level': 30,
        'overbought_level': 70
    })

    # 신호 생성
    signal = strategy.generate_signal(df)

    logger.info(f"신호: {signal.signal.name}")
    logger.info(f"신뢰도: {signal.confidence:.2%}")
    logger.info(f"가격: {signal.price:.2f}")
    logger.info(f"이유: {signal.reason}")
    logger.info(f"메타데이터: {signal.metadata}")
    logger.info("")


def test_bollinger_bands_strategy():
    """볼린저 밴드 전략 테스트"""
    logger.info("=" * 70)
    logger.info("[ 3/4 ] Bollinger Bands 전략 테스트")
    logger.info("=" * 70)

    # 데이터 수집
    client = BinanceClient(testnet=True)
    collector = HistoricalDataCollector(client)

    df = collector.get_recent_data('BTCUSDT', '1h', days=7)
    logger.info(f"✅ 데이터 수집 완료: {len(df)}개")

    # 전략 생성
    strategy = BollingerBandsStrategy(params={
        'period': 20,
        'std_dev': 2.0,
        'touch_threshold': 0.02
    })

    # 신호 생성
    signal = strategy.generate_signal(df)

    logger.info(f"신호: {signal.signal.name}")
    logger.info(f"신뢰도: {signal.confidence:.2%}")
    logger.info(f"가격: {signal.price:.2f}")
    logger.info(f"이유: {signal.reason}")
    logger.info(f"메타데이터: {signal.metadata}")
    logger.info("")


def test_risk_manager():
    """리스크 관리자 테스트"""
    logger.info("=" * 70)
    logger.info("[ 4/4 ] Risk Manager 테스트")
    logger.info("=" * 70)

    # 리스크 관리자 생성
    limits = RiskLimits(
        max_position_size_pct=0.2,
        max_daily_loss_pct=0.05,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )

    risk_manager = RiskManager(initial_balance=10000.0, limits=limits)

    # 포지션 크기 계산
    symbol = 'BTCUSDT'
    price = 50000.0
    confidence = 0.8

    position_info = risk_manager.calculate_position_size(
        symbol=symbol,
        price=price,
        confidence=confidence,
        win_rate=0.55,
        avg_win=0.02,
        avg_loss=0.01
    )

    logger.info(f"포지션 정보:")
    logger.info(f"  크기: {position_info['position_size_usdt']:.2f} USDT")
    logger.info(f"  수량: {position_info['quantity']:.6f}")
    logger.info(f"  Stop-Loss: {position_info['stop_loss']:.2f}")
    logger.info(f"  Take-Profit: {position_info['take_profit']:.2f}")
    logger.info(f"  Kelly %: {position_info['kelly_pct']:.2%}")
    logger.info("")

    # 포지션 추가
    risk_manager.add_position(
        symbol=symbol,
        side='BUY',
        entry_price=price,
        quantity=position_info['quantity'],
        stop_loss=position_info['stop_loss'],
        take_profit=position_info['take_profit']
    )

    # 상태 확인
    status = risk_manager.get_status()
    logger.info(f"리스크 관리자 상태:")
    logger.info(f"  현재 잔고: {status['current_balance']:.2f} USDT")
    logger.info(f"  총 손익: {status['total_pnl']:.2f} USDT ({status['total_pnl_pct']:.2%})")
    logger.info(f"  포지션 수: {status['num_positions']}")
    logger.info(f"  총 노출: {status['total_exposure']:.2f} USDT ({status['exposure_pct']:.2%})")
    logger.info("")

    # Stop-Loss 확인
    current_price = 49000.0  # 2% 하락
    action = risk_manager.check_stop_loss_take_profit(symbol, current_price)
    logger.info(f"가격 {current_price:.2f}에서 조치: {action}")

    # 포지션 청산
    pnl = risk_manager.close_position(symbol, current_price)
    logger.info(f"포지션 청산 손익: {pnl:.2f} USDT")

    # 최종 상태
    final_status = risk_manager.get_status()
    logger.info(f"최종 잔고: {final_status['current_balance']:.2f} USDT")
    logger.info("")


def main():
    """메인 테스트 함수"""
    logger.info("=" * 70)
    logger.info("Phase 3: 거래 전략 시스템 테스트")
    logger.info("=" * 70)
    logger.info("")

    try:
        # 1. MA Crossover 전략 테스트
        test_ma_crossover_strategy()

        # 2. RSI 전략 테스트
        test_rsi_strategy()

        # 3. Bollinger Bands 전략 테스트
        test_bollinger_bands_strategy()

        # 4. Risk Manager 테스트
        test_risk_manager()

        logger.info("=" * 70)
        logger.info("✅ Phase 3 테스트 완료!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"테스트 실패: {e}", exc_info=True)
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
