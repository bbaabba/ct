"""전체 시스템 통합 테스트"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from trading_bot import TradingBot
from utils.logger import setup_logger

logger = setup_logger(__name__)


def test_single_cycle():
    """단일 사이클 테스트"""
    logger.info("=" * 70)
    logger.info("전체 시스템 통합 테스트 - 단일 사이클")
    logger.info("=" * 70)
    logger.info("")

    # 봇 생성
    bot = TradingBot(
        symbol='BTCUSDT',
        interval='1h',
        initial_balance=10000.0,
        testnet=True,
        paper_trading=True
    )

    logger.info("봇 초기화 완료")
    logger.info("")

    # 1회 사이클 실행
    logger.info("1회 거래 사이클 실행 시작...")
    logger.info("")

    success = bot.execute_trading_cycle()

    if success:
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ 단일 사이클 테스트 성공!")
        logger.info("=" * 70)
        logger.info("")

        # 상태 리포트
        report = bot.get_status_report()
        logger.info("최종 상태 리포트:")
        logger.info(f"  사이클 수: {report['bot']['cycle_count']}")
        logger.info(f"  현재 잔고: {report['risk']['current_balance']:.2f} USDT")
        logger.info(f"  총 손익: {report['risk']['total_pnl']:.2f} USDT ({report['risk']['total_pnl_pct']:.2%})")
        logger.info(f"  포지션: {report['bot']['current_position']}")
        logger.info("")

        return True
    else:
        logger.error("")
        logger.error("=" * 70)
        logger.error("❌ 단일 사이클 테스트 실패")
        logger.error("=" * 70)
        logger.error("")
        return False


def test_multiple_cycles():
    """다중 사이클 테스트 (3회)"""
    logger.info("=" * 70)
    logger.info("전체 시스템 통합 테스트 - 다중 사이클 (3회)")
    logger.info("=" * 70)
    logger.info("")

    # 봇 생성
    bot = TradingBot(
        symbol='BTCUSDT',
        interval='1h',
        initial_balance=10000.0,
        testnet=True,
        paper_trading=True
    )

    logger.info("봇 초기화 완료")
    logger.info("")

    # 3회 사이클 실행 (간격 0분 - 테스트용)
    logger.info("3회 거래 사이클 연속 실행 시작...")
    logger.info("")

    success_count = 0

    for i in range(3):
        logger.info(f"\n>>> 사이클 {i+1}/3 실행 중...\n")

        success = bot.execute_trading_cycle()

        if success:
            success_count += 1
            logger.info(f"\n>>> 사이클 {i+1}/3 완료 ✅\n")
        else:
            logger.error(f"\n>>> 사이클 {i+1}/3 실패 ❌\n")

        # 짧은 대기 (실제로는 interval_minutes 적용)
        if i < 2:
            import time
            logger.info("다음 사이클까지 5초 대기...")
            time.sleep(5)

    logger.info("")
    logger.info("=" * 70)
    if success_count == 3:
        logger.info("✅ 다중 사이클 테스트 성공! (3/3)")
    else:
        logger.warning(f"⚠️ 다중 사이클 테스트 부분 성공 ({success_count}/3)")
    logger.info("=" * 70)
    logger.info("")

    # 최종 리포트
    report = bot.get_status_report()
    logger.info("최종 상태 리포트:")
    logger.info(f"  총 사이클: {report['bot']['cycle_count']}")
    logger.info(f"  현재 잔고: {report['risk']['current_balance']:.2f} USDT")
    logger.info(f"  총 손익: {report['risk']['total_pnl']:.2f} USDT ({report['risk']['total_pnl_pct']:.2%})")
    logger.info(f"  일일 손익: {report['risk']['daily_pnl']:.2f} USDT ({report['risk']['daily_pnl_pct']:.2%})")
    logger.info(f"  활성 포지션: {report['risk']['num_positions']}")
    logger.info("")

    # 전략별 성과
    logger.info("전략별 성과:")
    for strategy_name, data in report['strategies'].items():
        metrics = data['metrics']
        if metrics and metrics.total_trades > 0:
            logger.info(f"  [{strategy_name}]")
            logger.info(f"    총 거래: {metrics.total_trades}")
            logger.info(f"    승률: {metrics.win_rate:.2%}")
            logger.info(f"    총 손익: {metrics.total_pnl:.2f} USDT")
        else:
            logger.info(f"  [{strategy_name}] - 거래 없음")
    logger.info("")

    return success_count == 3


def test_strategy_switching():
    """전략 스위칭 테스트"""
    logger.info("=" * 70)
    logger.info("전략 스위칭 기능 테스트")
    logger.info("=" * 70)
    logger.info("")

    bot = TradingBot(
        symbol='BTCUSDT',
        interval='1h',
        initial_balance=10000.0,
        testnet=True,
        paper_trading=True
    )

    # 데이터 수집
    df = bot.get_latest_data(days=14)

    if df is None:
        logger.error("데이터 수집 실패")
        return False

    # 시장 상태 감지
    logger.info("[ 1/3 ] 시장 상태 감지...")
    regime_info = bot.regime_detector.detect_regime(df)
    logger.info(f"✅ 현재 시장 상태: {regime_info.regime.value}")
    logger.info(f"   신뢰도: {regime_info.confidence:.2%}")
    logger.info("")

    # 추천 전략
    suitable = bot.regime_detector.get_suitable_strategies(regime_info.regime)
    logger.info(f"   추천 전략: {suitable}")
    logger.info("")

    # 전략 가중치 계산
    logger.info("[ 2/3 ] 전략 가중치 계산...")
    weights = bot.strategy_selector.calculate_combined_weights(df)
    logger.info("✅ 전략 가중치:")
    for name, weight in weights.items():
        logger.info(f"   {name}: {weight:.2%}")
    logger.info("")

    # 최적 전략 선택
    logger.info("[ 3/3 ] 최적 전략 선택...")
    best_name, best_strategy = bot.strategy_selector.select_best_strategy(df)
    logger.info(f"✅ 선택된 전략: {best_name}")
    logger.info("")

    # 앙상블 신호
    logger.info("앙상블 신호 생성...")
    signal = bot.strategy_selector.generate_ensemble_signal(df)
    logger.info(f"✅ 앙상블 신호: {signal.signal.name}")
    logger.info(f"   신뢰도: {signal.confidence:.2%}")
    logger.info(f"   가중 스코어: {signal.metadata.get('weighted_score', 0):.2f}")
    logger.info("")

    logger.info("=" * 70)
    logger.info("✅ 전략 스위칭 기능 정상 작동")
    logger.info("=" * 70)
    logger.info("")

    return True


def main():
    """메인 테스트 함수"""
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " " * 15 + "전체 시스템 통합 테스트" + " " * 31 + "║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info("")

    results = []

    try:
        # 테스트 1: 단일 사이클
        logger.info("\n" + "▶" * 35)
        logger.info("테스트 1: 단일 사이클 실행")
        logger.info("▶" * 35 + "\n")
        result1 = test_single_cycle()
        results.append(("단일 사이클", result1))

        # 테스트 2: 다중 사이클
        logger.info("\n" + "▶" * 35)
        logger.info("테스트 2: 다중 사이클 실행 (3회)")
        logger.info("▶" * 35 + "\n")
        result2 = test_multiple_cycles()
        results.append(("다중 사이클 (3회)", result2))

        # 테스트 3: 전략 스위칭
        logger.info("\n" + "▶" * 35)
        logger.info("테스트 3: 전략 스위칭 기능")
        logger.info("▶" * 35 + "\n")
        result3 = test_strategy_switching()
        results.append(("전략 스위칭", result3))

        # 최종 요약
        logger.info("\n")
        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║" + " " * 22 + "테스트 결과 요약" + " " * 30 + "║")
        logger.info("╚" + "═" * 68 + "╝")
        logger.info("")

        for test_name, result in results:
            status = "✅ 성공" if result else "❌ 실패"
            logger.info(f"  {test_name:30s} : {status}")

        logger.info("")

        all_passed = all(result for _, result in results)

        if all_passed:
            logger.info("╔" + "═" * 68 + "╗")
            logger.info("║" + " " * 15 + "🎉 모든 테스트 통과! 🎉" + " " * 28 + "║")
            logger.info("╚" + "═" * 68 + "╝")
            logger.info("")
            logger.info("시스템이 정상적으로 작동합니다.")
            logger.info("실전 배포를 진행할 수 있습니다.")
        else:
            logger.warning("╔" + "═" * 68 + "╗")
            logger.warning("║" + " " * 15 + "⚠️  일부 테스트 실패  ⚠️" + " " * 25 + "║")
            logger.warning("╚" + "═" * 68 + "╝")
            logger.warning("")
            logger.warning("시스템 점검이 필요합니다.")

        return all_passed

    except Exception as e:
        logger.error(f"치명적 오류 발생: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
