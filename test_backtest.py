"""AI 모델 학습 & 백테스트 실행 스크립트

4-Stage Pipeline: SSL → SFT → Backtest → Walk-Forward
+ 통합 모델 (Multi-Scale TCN + Global MoE)

사용법:
    python test_backtest.py ssl eth              # SSL 사전학습
    python test_backtest.py sft eth              # SFT 파인튜닝
    python test_backtest.py backtest eth         # 백테스트 (라이브 유사)
    python test_backtest.py walkforward eth      # 워크포워드
    python test_backtest.py pipeline eth         # SSL → SFT → 백테스트
    python test_backtest.py eth                  # 레거시 (기존 IS/OOS 방식)

    # 통합 모델 (Multi-Scale TCN + Global MoE)
    python test_backtest.py unified-ssl eth      # 통합 SSL
    python test_backtest.py unified-sft eth      # 통합 SFT
    python test_backtest.py unified-backtest eth # 통합 백테스트
    python test_backtest.py unified-pipeline eth # 통합 풀 파이프라인
    python test_backtest.py unified-wf eth      # 통합 Walk-Forward (Train 9m → Test 3m)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config
from trading_bot.futures_bot import FuturesTradingBot
from utils.logger import setup_logger

logger = setup_logger(__name__)

SYMBOL_MAP = {
    'btc': 'BTCUSDT',
    'eth': 'ETHUSDT',
}

MODES = {
    'ssl', 'sft', 'backtest', 'walkforward', 'pipeline',
    'unified-ssl', 'unified-sft', 'unified-backtest', 'unified-pipeline',
    'unified-wf',
}


def _create_bot(symbol: str, unified: bool = False) -> FuturesTradingBot:
    """공통 봇 생성."""
    return FuturesTradingBot(
        symbol=symbol,
        interval='1m',
        initial_balance=Config.INITIAL_CAPITAL,
        testnet=Config.IS_TESTNET,
        paper_trading=True,
        min_leverage=1,
        max_leverage=3,
        default_leverage=2,
        max_position_pct=Config.MAX_POSITION_SIZE,
        max_daily_loss_pct=Config.MAX_DAILY_LOSS,
        use_ai_model=True,
        ai_model_weight=0.4,
        mtf_intervals=['1m', '3m', '5m', '15m'],
        enabled_indicators=['trend_ma', 'rsi', 'macd', 'bollinger'],
    )


def run_ssl(symbol: str):
    """Stage 1: SSL Pre-training (1~2년 데이터, 라벨 불필요)."""
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    print()
    print("=" * 70)
    print(f"  [Stage 1] SSL Pre-training: {symbol}")
    print(f"   기간: {start_date} ~ {end_date} (약 1년)")
    print("=" * 70)
    print()

    bot = _create_bot(symbol)
    bot.run_ssl_pretrain(start_date=start_date, end_date=end_date, epochs=10)


def run_sft(symbol: str):
    """Stage 2: SFT Fine-tuning (수개월 라벨 데이터)."""
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    print()
    print("=" * 70)
    print(f"  [Stage 2] SFT Fine-tuning: {symbol}")
    print(f"   기간: {start_date} ~ {end_date} (약 3개월)")
    print("=" * 70)
    print()

    bot = _create_bot(symbol)
    bot.run_sft(start_date=start_date, end_date=end_date)


def run_backtest(symbol: str):
    """Stage 3: Backtest (라이브 유사, trading_ready 모델 검증)."""
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    print()
    print("=" * 70)
    print(f"  [Stage 3] Backtest: {symbol}")
    print(f"   기간: {start_date} ~ {end_date} (약 1개월)")
    print("=" * 70)
    print()

    bot = _create_bot(symbol)
    bot.run_backtest_new(start_date=start_date, end_date=end_date, step=1)


def run_walkforward(symbol: str):
    """Stage 4: Walk-Forward Optimization."""
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

    print()
    print("=" * 70)
    print(f"  [Stage 4] Walk-Forward: {symbol}")
    print(f"   기간: {start_date} ~ {end_date} (약 6개월)")
    print("=" * 70)
    print()

    bot = _create_bot(symbol)
    bot.run_walkforward(
        start_date=start_date, end_date=end_date,
        train_months=3, test_months=1, step=1
    )


def run_pipeline(symbol: str):
    """Full Pipeline: SSL → SFT → Backtest."""
    from datetime import datetime, timedelta

    now = datetime.now()
    ssl_start = (now - timedelta(days=365)).strftime('%Y-%m-%d')
    ssl_end = now.strftime('%Y-%m-%d')
    sft_start = (now - timedelta(days=90)).strftime('%Y-%m-%d')
    sft_end = now.strftime('%Y-%m-%d')
    bt_start = (now - timedelta(days=30)).strftime('%Y-%m-%d')
    bt_end = now.strftime('%Y-%m-%d')

    print()
    print("=" * 70)
    print(f"  Full Pipeline: {symbol}")
    print(f"   [1] SSL: {ssl_start} ~ {ssl_end}")
    print(f"   [2] SFT: {sft_start} ~ {sft_end}")
    print(f"   [3] BT:  {bt_start} ~ {bt_end}")
    print("=" * 70)
    print()

    bot = _create_bot(symbol)
    bot.run_ssl_pretrain(start_date=ssl_start, end_date=ssl_end, epochs=10)
    bot.run_sft(start_date=sft_start, end_date=sft_end)
    bot.run_backtest_new(start_date=bt_start, end_date=bt_end, step=1)
    print("\n  Full Pipeline 완료!")


def run_legacy(symbol: str):
    """레거시: 기존 IS/OOS 방식 백테스트."""
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    print()
    print("=" * 70)
    print(f"  레거시 백테스트: {symbol}")
    print(f"   기간: {start_date} ~ {end_date}")
    print("=" * 70)
    print()

    bot = _create_bot(symbol)
    bot.run_backtest(start_date=start_date, end_date=end_date, step=1)


# ── 통합 모델 (Multi-Scale TCN + Global MoE) ──

def run_unified_ssl(symbol: str):
    """통합 모델 SSL 사전학습."""
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    print()
    print("=" * 70)
    print(f"  [통합 Stage 1] SSL Pre-training: {symbol}")
    print(f"   아키텍처: Multi-Scale TCN + Global MoE")
    print(f"   기간: {start_date} ~ {end_date} (약 1년)")
    print("=" * 70)
    print()

    bot = _create_bot(symbol, unified=True)
    bot.run_ssl_pretrain_unified(start_date=start_date, end_date=end_date, epochs=10)


def run_unified_sft(symbol: str):
    """통합 모델 SFT."""
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    print()
    print("=" * 70)
    print(f"  [통합 Stage 2] SFT Fine-tuning: {symbol}")
    print(f"   기간: {start_date} ~ {end_date} (약 3개월)")
    print("=" * 70)
    print()

    bot = _create_bot(symbol, unified=True)
    bot.run_sft_unified(start_date=start_date, end_date=end_date)


def run_unified_backtest(symbol: str):
    """통합 모델 백테스트."""
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    print()
    print("=" * 70)
    print(f"  [통합 Stage 3] Backtest: {symbol}")
    print(f"   기간: {start_date} ~ {end_date} (약 1개월)")
    print("=" * 70)
    print()

    bot = _create_bot(symbol, unified=True)
    bot.run_backtest_unified(start_date=start_date, end_date=end_date, step=1)


def run_unified_pipeline(symbol: str):
    """통합 풀 파이프라인: SSL → SFT → Backtest."""
    print()
    print("=" * 70)
    print(f"  통합 파이프라인: {symbol}")
    print(f"   아키텍처: Multi-Scale TCN + Global MoE")
    print("=" * 70)
    print()

    bot = _create_bot(symbol, unified=True)
    bot.run_unified_pipeline(
        start_date='', end_date='',  # pipeline 내부에서 날짜 계산
        ssl_days=365, sft_days=90, bt_days=30
    )


def run_unified_walkforward(symbol: str):
    """통합 Walk-Forward: Train 9개월 → Test 3개월, 겹침 0."""
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime('%Y-%m-%d')
    # 최소 train(9) + test(3) + 여유 → 24개월 데이터 사용
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    print()
    print("=" * 70)
    print(f"  통합 Walk-Forward: {symbol}")
    print(f"   기간: {start_date} ~ {end_date} (약 24개월)")
    print(f"   윈도우: Train=9개월, Test=3개월, 겹침=0")
    print("=" * 70)
    print()

    bot = _create_bot(symbol, unified=True)
    bot.run_unified_walkforward(
        start_date=start_date, end_date=end_date,
        train_months=9, test_months=3, step=1
    )


def run():
    args = sys.argv[1:]

    # 모드와 심볼 파싱
    mode = None
    symbol_arg = None

    for arg in args:
        if arg.lower() in MODES:
            mode = arg.lower()
        else:
            symbol_arg = arg.lower()

    # 기본값
    if symbol_arg is None:
        symbol_arg = 'eth'
    symbol = SYMBOL_MAP.get(symbol_arg, symbol_arg.upper())

    # 모드별 실행
    if mode == 'ssl':
        run_ssl(symbol)
    elif mode == 'sft':
        run_sft(symbol)
    elif mode == 'backtest':
        run_backtest(symbol)
    elif mode == 'walkforward':
        run_walkforward(symbol)
    elif mode == 'pipeline':
        run_pipeline(symbol)
    elif mode == 'unified-ssl':
        run_unified_ssl(symbol)
    elif mode == 'unified-sft':
        run_unified_sft(symbol)
    elif mode == 'unified-backtest':
        run_unified_backtest(symbol)
    elif mode == 'unified-pipeline':
        run_unified_pipeline(symbol)
    elif mode == 'unified-wf':
        run_unified_walkforward(symbol)
    else:
        run_legacy(symbol)


if __name__ == '__main__':
    run()
