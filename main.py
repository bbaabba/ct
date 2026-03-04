"""바이낸스 AI 자동거래 시스템 - 메인 실행 파일

통합 모델 (Multi-Scale TCN + Global MoE) 전용.
"""
import sys
import time
import signal
import subprocess
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

# 전역 봇 인스턴스 (시그널 핸들러용)
_bot_instance = None


def signal_handler(signum, frame):
    """시그널 핸들러 (Ctrl+C 등)"""
    global _bot_instance
    print("\n\n⚠️  종료 신호 수신...")
    if _bot_instance:
        _bot_instance.stop()
    sys.exit(0)


def print_banner():
    """배너 출력"""
    print()
    print("=" * 74)
    print("║" + " " * 72 + "║")
    print("║     🤖 AI 선물 자동거래 시스템 (Futures Auto Trading Bot)              ║")
    print("║" + " " * 72 + "║")
    print("║     • Multi-Scale TCN + Global MoE 통합 모델                           ║")
    print("║     • Shadow Mode 보호 + MoE 가변 포지셔닝                             ║")
    print("║     • 실시간 WebSocket 데이터 스트림                                   ║")
    print("║     • 자동 Stop-Loss / Take-Profit                                     ║")
    print("║     • 리스크 관리 (드로다운, 연속손실 제한)                            ║")
    print("║" + " " * 72 + "║")
    print("=" * 74)
    print()


def print_config_status():
    """설정 상태 출력"""
    print("━" * 70)
    print("📋 시스템 설정")
    print("━" * 70)

    # API 키 상태
    if Config.BINANCE_API_KEY and Config.BINANCE_SECRET_KEY:
        print(f"  ✅ API 키: 설정됨")
    else:
        print(f"  ⚠️  API 키: 미설정 (페이퍼 트레이딩만 가능)")

    print(f"  📌 테스트넷: {'예' if Config.IS_TESTNET else '아니오 (실거래)'}")
    print(f"  💰 초기 자본: {Config.INITIAL_CAPITAL:,.0f} USDT")
    print(f"  📊 거래 심볼: {', '.join(Config.TRADING_SYMBOLS)}")
    print(f"  🔧 최대 레버리지: {Config.MAX_LEVERAGE}x")
    print("━" * 70)
    print()


def _start_dashboard():
    """Streamlit 대시보드를 백그라운드로 자동 실행"""
    try:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run",
                "streamlit_dashboard/app.py",
                "--server.port", "8501",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ],
            cwd=str(Path(__file__).parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info("Streamlit 대시보드 시작 (http://localhost:8501)")
        return proc
    except Exception as e:
        logger.warning(f"대시보드 실행 실패: {e}")
        return None


def run_futures_bot(
    symbol: str = 'BTCUSDT',
    interval: str = '1m',
    leverage: int = 3,
    paper_trading: bool = True,
    enabled_indicators: list = None,
):
    """AI 선물 자동 거래 봇 실행 (통합 모델 전용)"""
    global _bot_instance

    from trading_bot.futures_bot import FuturesTradingBot

    enabled_indicators = enabled_indicators or ['trend_ma', 'rsi', 'macd', 'bollinger']

    print()
    print("🚀 AI 선물 자동 거래 봇 시작")
    print("=" * 70)
    print(f"  심볼: {symbol}")
    print(f"  간격: {interval}")
    print(f"  기본 레버리지: {leverage}x")
    print(f"  페이퍼 트레이딩: {'예' if paper_trading else '아니오 (실거래)'}")
    print(f"  🧠 통합 모델: Multi-Scale TCN + MoE (15m/3m/10s)")
    print(f"     Shadow Mode + MoE 가변 포지셔닝 + Freshness Gate")
    print(f"  📈 활성 지표: {len(enabled_indicators)}개 ({', '.join(enabled_indicators)})")
    print("=" * 70)
    print()

    if not paper_trading:
        print("⚠️  경고: 실제 거래 모드입니다!")
        confirm = input("계속하시겠습니까? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("취소되었습니다.")
            return

    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 봇 생성 (통합 모델 고정)
    bot = FuturesTradingBot(
        symbol=symbol,
        interval=interval,
        initial_balance=Config.INITIAL_CAPITAL,
        testnet=Config.IS_TESTNET,
        paper_trading=paper_trading,
        min_leverage=1,
        max_leverage=int(Config.MAX_LEVERAGE),
        default_leverage=leverage,
        max_position_pct=Config.MAX_POSITION_SIZE,
        max_daily_loss_pct=Config.MAX_DAILY_LOSS,
        use_ai_model=True,
        enabled_indicators=enabled_indicators,
    )

    _bot_instance = bot

    # 봇 시작
    bot.start()

    # 대시보드 자동 실행
    dashboard_proc = _start_dashboard()

    print()
    print("💡 Ctrl+C를 누르면 봇이 안전하게 종료됩니다.")
    if dashboard_proc:
        print("📊 대시보드: http://localhost:8501")
    print()

    # 메인 루프 (봇은 별도 스레드에서 실행됨)
    try:
        while bot.state.value in ['RUNNING', 'PAUSED', 'IDLE']:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    # 봇 중지
    bot.stop()

    # 대시보드 종료
    if dashboard_proc and dashboard_proc.poll() is None:
        dashboard_proc.terminate()
        try:
            dashboard_proc.wait(timeout=5)
        except:
            dashboard_proc.kill()


def get_trading_settings() -> dict:
    """거래 설정 입력 (통합 모델 전용)"""
    print()
    print("━" * 70)
    print("⚙️  거래 설정")
    print("━" * 70)

    # 심볼 선택
    print()
    print("📊 거래 심볼:")
    print("  [1] BTCUSDT (비트코인)")
    print("  [2] ETHUSDT (이더리움)")
    print("  [3] BNBUSDT (바이낸스코인)")
    print("  [4] 직접 입력")

    symbol_choice = input("선택 (기본: 2): ").strip() or '2'

    symbol_map = {
        '1': 'BTCUSDT',
        '2': 'ETHUSDT',
        '3': 'BNBUSDT'
    }

    if symbol_choice in symbol_map:
        symbol = symbol_map[symbol_choice]
    elif symbol_choice == '4':
        symbol = input("심볼 입력 (예: BTCUSDT): ").strip().upper()
    else:
        symbol = 'BTCUSDT'

    # 간격 선택
    print()
    print("⏱️  캔들 간격:")
    print("  [1] 1분 (빠른 거래)")
    print("  [2] 5분 (권장)")
    print("  [3] 15분 (안정적)")
    print("  [4] 1시간 (장기)")

    interval_choice = input("선택 (기본: 1): ").strip() or '1'

    interval_map = {
        '1': '1m',
        '2': '5m',
        '3': '15m',
        '4': '1h'
    }
    interval = interval_map.get(interval_choice, '1m')

    # 레버리지 선택
    print()
    print("🔧 기본 레버리지 (AI가 시장 상황에 따라 자동 조정):")
    print("  [1] 1x (안전)")
    print("  [2] 3x (권장)")
    print("  [3] 5x (보통)")
    print("  [4] 10x (고위험)")

    leverage_choice = input("선택 (기본: 3): ").strip() or '3'

    leverage_map = {
        '1': 1,
        '2': 3,
        '3': 5,
        '4': 10
    }
    leverage = leverage_map.get(leverage_choice, 5)

    # 통합 모델 안내
    print()
    print("  🧠 통합 모델: Multi-Scale TCN + MoE (15m/3m/10s 자동 통합)")
    print("     Shadow Mode 보호 + MoE 가변 포지셔닝 + Freshness Gate 활성")

    # 기술적 지표 선택
    print()
    print("📈 기술적 지표 선택:")
    print("  [1] 핵심 4개 (권장) - MA트렌드, RSI, MACD, 볼린저")
    print("  [2] 확장 6개 - 핵심 + 지지/저항, 스토캐스틱")
    print("  [3] 전체 10개 - 모든 지표 사용")
    print("  [4] 직접 선택")
    print()
    print("  ℹ️  지표가 적을수록 신호가 명확하고, 많을수록 정교하지만 충돌 가능")

    indicator_choice = input("선택 (기본: 1): ").strip() or '1'

    # 지표 정의 (이름: 설명)
    all_indicators = {
        'trend_ma': 'MA 트렌드 (이동평균선)',
        'rsi': 'RSI (과매수/과매도)',
        'macd': 'MACD (추세/모멘텀)',
        'bollinger': '볼린저밴드 (변동성)',
        'support_resistance': '지지/저항선',
        'stochastic': '스토캐스틱 (%K/%D)',
        'volume': '거래량 분석',
        'momentum': '모멘텀',
        'adx': 'ADX (추세 강도)',
        'obv': 'OBV (거래량 흐름)',
    }

    # 프리셋 정의
    indicator_presets = {
        '1': ['trend_ma', 'rsi', 'macd', 'bollinger'],  # 핵심 4개
        '2': ['trend_ma', 'rsi', 'macd', 'bollinger', 'support_resistance', 'stochastic'],  # 확장 6개
        '3': list(all_indicators.keys()),  # 전체 10개
    }

    if indicator_choice in indicator_presets:
        enabled_indicators = indicator_presets[indicator_choice]
    elif indicator_choice == '4':
        print()
        print("📋 사용할 지표를 선택하세요 (쉼표로 구분, 최소 2개):")
        for i, (key, desc) in enumerate(all_indicators.items(), 1):
            print(f"  [{i}] {key}: {desc}")
        print()
        print("  예: 1,2,3,4 또는 trend_ma,rsi,macd,bollinger")

        custom_input = input("선택: ").strip()

        # 숫자 입력 처리
        if custom_input and custom_input[0].isdigit():
            indices = [int(x.strip()) for x in custom_input.split(',') if x.strip().isdigit()]
            keys = list(all_indicators.keys())
            enabled_indicators = [keys[i-1] for i in indices if 1 <= i <= len(keys)]
        else:
            # 이름 직접 입력
            enabled_indicators = [x.strip() for x in custom_input.split(',')]
            enabled_indicators = [x for x in enabled_indicators if x in all_indicators]

        # 최소 2개 보장
        if len(enabled_indicators) < 2:
            print("   ⚠️  최소 2개 필요 → 기본 4개로 설정")
            enabled_indicators = indicator_presets['1']
    else:
        enabled_indicators = indicator_presets['1']

    print(f"   ✅ 활성 지표 ({len(enabled_indicators)}개): {', '.join(enabled_indicators)}")

    print()
    print(f"✅ 설정 완료: {symbol} / {interval} / {leverage}x / 통합 모델")
    print(f"   🧠 Multi-Scale TCN + MoE (Shadow Mode)")
    print(f"   📈 기술적 지표: {len(enabled_indicators)}개 활성")

    return {
        'symbol': symbol,
        'interval': interval,
        'leverage': leverage,
        'enabled_indicators': enabled_indicators,
    }


def show_backtest_menu():
    """통합 모델 학습 & 백테스트 메뉴"""
    print()
    print("━" * 70)
    print("📊 통합 모델 학습 & 백테스트 (Multi-Scale TCN + Global MoE)")
    print("━" * 70)

    # 심볼 선택
    print()
    print("  거래 심볼:")
    print("    [1] BTCUSDT (비트코인)")
    print("    [2] ETHUSDT (이더리움)")
    print("    [3] 직접 입력")
    symbol_choice = input("  선택 (기본: 2): ").strip() or '2'

    symbol_map = {'1': 'btc', '2': 'eth'}
    if symbol_choice in symbol_map:
        symbol_arg = symbol_map[symbol_choice]
    elif symbol_choice == '3':
        symbol_arg = input("  심볼 입력 (예: SOLUSDT): ").strip()
    else:
        symbol_arg = 'eth'

    print()
    print("  모드 선택:")
    print("    [1] SSL 사전학습 — 3개 인코더 + 공유 백본 SSL")
    print("    [2] SFT 파인튜닝 — 인코더 동결 후 MoE + Heads 파인튜닝")
    print("    [3] 백테스트 — 통합 모델 검증 (라이브 유사)")
    print("    [4] 풀 파이프라인 — SSL → SFT → 백테스트 일괄")
    print()
    print("    [0] 돌아가기")
    print()

    choice = input("  선택: ").strip()

    if choice == '0':
        return

    mode_map = {
        '1': 'unified-ssl',
        '2': 'unified-sft',
        '3': 'unified-backtest',
        '4': 'unified-pipeline',
    }

    if choice in mode_map:
        mode = mode_map[choice]
        print(f"\n🔄 {mode} 실행 중 ({symbol_arg.upper()})...\n")
        subprocess.run(['python', 'test_backtest.py', mode, symbol_arg], cwd=Config.BASE_DIR)


def show_system_status():
    """시스템 상태 출력"""
    print()
    print("━" * 70)
    print("📋 시스템 상태")
    print("━" * 70)

    # API 연결 테스트
    print()
    print("🔌 API 연결 상태:")

    try:
        from api.futures_client import BinanceFuturesClient
        client = BinanceFuturesClient(testnet=Config.IS_TESTNET)

        if client.ping():
            print("  ✅ 선물 API: 연결됨")

            # 서버 시간
            server_time = client.get_server_time()
            from datetime import datetime
            server_dt = datetime.fromtimestamp(server_time / 1000)
            print(f"  📅 서버 시간: {server_dt}")

            # 마크 가격 조회
            try:
                mark = client.get_mark_price('BTCUSDT')
                print(f"  💰 BTC 마크가격: {float(mark['markPrice']):,.2f} USDT")
            except:
                pass

            # 계정 정보 (API 키 있을 경우)
            if Config.BINANCE_API_KEY:
                try:
                    balance = client.get_usdt_balance()
                    print(f"  💳 USDT 잔고: {balance['available']:.2f} USDT")
                except:
                    print("  ⚠️  계정 정보 조회 실패 (API 키 확인 필요)")
        else:
            print("  ❌ 선물 API: 연결 실패")

    except Exception as e:
        print(f"  ❌ 연결 오류: {e}")

    print()
    input("Enter를 눌러 계속...")


def main_menu():
    """메인 메뉴"""
    while True:
        print()
        print("=" * 74)
        print("║" + " " * 28 + "🎮 메인 메뉴" + " " * 28 + "║")
        print("=" * 74)
        print("║" + " " * 72 + "║")
        print("║   [1] 🤖 AI 자동 거래 시작 (페이퍼 트레이딩)                           ║")
        print("║   [2] 💰 AI 자동 거래 시작 (실제 거래) ⚠️                              ║")
        print("║   [3] 📋 시스템 상태 확인                                              ║")
        print("║   [4] 📊 통합 모델 학습 & 백테스트                                      ║")
        print("║   [0] 🚪 종료                                                          ║")
        print("║" + " " * 72 + "║")
        print("=" * 74)
        print()

        choice = input("선택하세요: ").strip()

        if choice == '1':
            # 페이퍼 트레이딩
            settings = get_trading_settings()
            run_futures_bot(
                symbol=settings['symbol'],
                interval=settings['interval'],
                leverage=settings['leverage'],
                paper_trading=True,
                enabled_indicators=settings.get('enabled_indicators', ['trend_ma', 'rsi', 'macd', 'bollinger']),
            )

        elif choice == '2':
            # 실제 거래
            if not Config.BINANCE_API_KEY or not Config.BINANCE_SECRET_KEY:
                print()
                print("❌ API 키가 설정되지 않았습니다.")
                print("   .env 파일에 BINANCE_API_KEY와 BINANCE_SECRET_KEY를 설정해주세요.")
                continue

            settings = get_trading_settings()
            run_futures_bot(
                symbol=settings['symbol'],
                interval=settings['interval'],
                leverage=settings['leverage'],
                paper_trading=False,
                enabled_indicators=settings.get('enabled_indicators', ['trend_ma', 'rsi', 'macd', 'bollinger']),
            )

        elif choice == '3':
            # 시스템 상태
            show_system_status()

        elif choice == '4':
            # 백테스트
            show_backtest_menu()

        elif choice == '0':
            print()
            print("👋 프로그램을 종료합니다.")
            break

        else:
            print("⚠️  잘못된 선택입니다. 다시 선택해주세요.")


def main():
    """메인 함수"""
    print_banner()
    print_config_status()

    # API 키 경고
    if not Config.BINANCE_API_KEY or not Config.BINANCE_SECRET_KEY:
        print("⚠️  API 키가 설정되지 않았습니다.")
        print("   페이퍼 트레이딩(모의 거래)만 사용 가능합니다.")
        print()
        print("   실제 거래를 위해서는:")
        print("   1. 바이낸스에서 API 키를 발급받으세요")
        print("   2. .env 파일에 다음을 추가하세요:")
        print("      BINANCE_API_KEY=your_api_key")
        print("      BINANCE_SECRET_KEY=your_secret_key")
        print()

    # 메인 메뉴 실행
    main_menu()


if __name__ == '__main__':
    main()
