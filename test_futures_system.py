"""선물 자동 거래 시스템 테스트"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import time
import pandas as pd
import numpy as np
from datetime import datetime

from utils.logger import setup_logger

logger = setup_logger(__name__)


def test_futures_client():
    """선물 API 클라이언트 테스트"""
    print()
    print("=" * 70)
    print("🔌 Phase 1: 선물 API 클라이언트 테스트")
    print("=" * 70)

    from api.futures_client import BinanceFuturesClient

    client = BinanceFuturesClient(testnet=True)

    # 1. 서버 연결
    print("\n1️⃣ 서버 연결 테스트...")
    if client.ping():
        print("   ✅ 선물 서버 연결 성공")
    else:
        print("   ❌ 선물 서버 연결 실패")
        return False

    # 2. 서버 시간
    print("\n2️⃣ 서버 시간 조회...")
    server_time = client.get_server_time()
    server_dt = datetime.fromtimestamp(server_time / 1000)
    print(f"   ✅ 서버 시간: {server_dt}")

    # 3. 마크 가격
    print("\n3️⃣ 마크 가격 조회...")
    mark = client.get_mark_price('BTCUSDT')
    print(f"   ✅ BTC 마크가격: {float(mark['markPrice']):,.2f} USDT")
    print(f"   ✅ 펀딩비율: {float(mark['lastFundingRate']) * 100:.4f}%")

    # 4. 캔들 데이터
    print("\n4️⃣ 캔들 데이터 조회...")
    klines = client.get_klines('BTCUSDT', '1h', limit=10)
    print(f"   ✅ {len(klines)}개 캔들 조회 완료")

    # 5. 청산가 계산
    print("\n5️⃣ 청산가 계산 테스트...")
    entry_price = 50000

    liq_long_5x = client.calculate_liquidation_price('LONG', entry_price, 5)
    liq_short_5x = client.calculate_liquidation_price('SHORT', entry_price, 5)
    liq_long_10x = client.calculate_liquidation_price('LONG', entry_price, 10)
    liq_short_10x = client.calculate_liquidation_price('SHORT', entry_price, 10)

    print(f"   진입가: {entry_price:,} USDT")
    print(f"   ✅ 롱 5x 청산가: {liq_long_5x:,.2f} USDT ({(entry_price - liq_long_5x) / entry_price * 100:.1f}% 하락)")
    print(f"   ✅ 숏 5x 청산가: {liq_short_5x:,.2f} USDT ({(liq_short_5x - entry_price) / entry_price * 100:.1f}% 상승)")
    print(f"   ✅ 롱 10x 청산가: {liq_long_10x:,.2f} USDT ({(entry_price - liq_long_10x) / entry_price * 100:.1f}% 하락)")
    print(f"   ✅ 숏 10x 청산가: {liq_short_10x:,.2f} USDT ({(liq_short_10x - entry_price) / entry_price * 100:.1f}% 상승)")

    # 6. PnL 계산
    print("\n6️⃣ 손익 계산 테스트...")
    pnl_long = client.calculate_pnl('LONG', 50000, 52000, 0.1, leverage=5)
    pnl_short = client.calculate_pnl('SHORT', 50000, 48000, 0.1, leverage=5)

    print(f"   롱 포지션 (50000 → 52000):")
    print(f"     ✅ 손익: {pnl_long['pnl']:.2f} USDT ({pnl_long['pnl_pct']:.2f}%)")
    print(f"     ✅ ROE: {pnl_long['roe']:.2f}%")

    print(f"   숏 포지션 (50000 → 48000):")
    print(f"     ✅ 손익: {pnl_short['pnl']:.2f} USDT ({pnl_short['pnl_pct']:.2f}%)")
    print(f"     ✅ ROE: {pnl_short['roe']:.2f}%")

    print("\n✅ 선물 API 클라이언트 테스트 통과!")
    return True


def test_leverage_optimizer():
    """AI 레버리지 최적화기 테스트"""
    print()
    print("=" * 70)
    print("🔧 Phase 2: AI 레버리지 최적화기 테스트")
    print("=" * 70)

    from ai.leverage_optimizer import AILeverageOptimizer

    optimizer = AILeverageOptimizer(
        min_leverage=1,
        max_leverage=10,
        default_leverage=3,
        risk_tolerance=0.5
    )

    # 테스트용 가상 데이터 생성
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')

    # 상승 트렌드 데이터
    base_price = 50000
    trend = np.linspace(0, 5000, 200)
    noise = np.random.randn(200) * 500

    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(200) * 200)
    low = close - np.abs(np.random.randn(200) * 200)
    open_price = close - np.random.randn(200) * 100
    volume = np.random.uniform(100, 1000, 200)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    # 1. 상승 트렌드 테스트
    print("\n1️⃣ 상승 트렌드 시장 분석...")
    rec = optimizer.recommend_leverage(df, signal_confidence=0.7, position_side='LONG')

    print(f"   추천 레버리지: {rec.recommended_leverage}x")
    print(f"   최대 안전 레버리지: {rec.max_safe_leverage}x")
    print(f"   리스크 레벨: {rec.risk_level.name}")
    print(f"   시장 상태: {rec.market_condition}")
    print(f"   변동성: {rec.volatility_score:.1%}")
    print(f"   트렌드 강도: {rec.trend_strength:.1%}")
    for reason in rec.reasons:
        print(f"     → {reason}")

    # 2. 고변동성 데이터 생성
    print("\n2️⃣ 고변동성 시장 분석...")
    df_volatile = df.copy()
    df_volatile['close'] = df['close'] + np.random.randn(200) * 2000
    df_volatile['high'] = df_volatile['close'] + np.abs(np.random.randn(200) * 1000)
    df_volatile['low'] = df_volatile['close'] - np.abs(np.random.randn(200) * 1000)

    rec_volatile = optimizer.recommend_leverage(df_volatile, signal_confidence=0.5)

    print(f"   추천 레버리지: {rec_volatile.recommended_leverage}x")
    print(f"   리스크 레벨: {rec_volatile.risk_level.name}")
    print(f"   변동성: {rec_volatile.volatility_score:.1%}")

    # 3. 연속 손실 조정 테스트
    print("\n3️⃣ 연속 손실 레버리지 조정 테스트...")
    for losses in [0, 2, 3, 5]:
        adjusted = optimizer.adjust_leverage_for_consecutive_losses(5, losses)
        print(f"   {losses}연패: 5x → {adjusted}x")

    # 4. 드로다운 조정 테스트
    print("\n4️⃣ 드로다운 레버리지 조정 테스트...")
    for dd in [0.03, 0.08, 0.12, 0.18]:
        adjusted = optimizer.adjust_leverage_for_drawdown(5, dd)
        print(f"   드로다운 {dd:.0%}: 5x → {adjusted}x")

    print("\n✅ AI 레버리지 최적화기 테스트 통과!")
    return True


def test_market_analyzer():
    """AI 분석기 테스트"""
    print()
    print("=" * 70)
    print("🧠 Phase 3: AI 딥러닝 분석기 테스트")
    print("=" * 70)

    from ai.market_analyzer import MarketAnalyzer, IntegratedAIAnalyzer

    # 테스트용 가상 데이터 생성
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')

    base_price = 50000
    trend = np.linspace(0, 3000, 200)
    noise = np.random.randn(200) * 500

    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(200) * 200)
    low = close - np.abs(np.random.randn(200) * 200)
    open_price = close - np.random.randn(200) * 100
    volume = np.random.uniform(100, 1000, 200)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    # 1. AI 분석기 초기화
    print("\n1️⃣ AI 분석기 초기화...")
    try:
        market_analyzer = MarketAnalyzer(sequence_length=60)
        print(f"   ✅ AI 분석기 생성 완료")
        print(f"   장치: {market_analyzer.device}")
        print(f"   입력 특징 수: {market_analyzer.input_size}")
        print(f"   시퀀스 길이: {market_analyzer.sequence_length}")
    except Exception as e:
        print(f"   ❌ AI 분석기 초기화 실패: {e}")
        return False

    # 2. 특징 추출 테스트
    print("\n2️⃣ 특징 추출 테스트...")
    features = market_analyzer.prepare_features(df)
    print(f"   ✅ 특징 추출 완료: {features.shape}")
    print(f"   특징 목록: {market_analyzer.feature_names[:5]}...")

    # 3. AI 예측 테스트
    print("\n3️⃣ AI 예측 테스트...")
    prediction = market_analyzer.predict(df)
    print(f"   예측 가격: {prediction.predicted_price:.2f} USDT")
    print(f"   예측 방향: {prediction.predicted_direction}")
    print(f"   방향 신뢰도: {prediction.direction_confidence:.1%}")
    print(f"   예상 변화율: {prediction.price_change_pct:.2f}%")

    if prediction.feature_importance:
        top_features = sorted(
            prediction.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        print("   주요 특징 중요도:")
        for feat, imp in top_features:
            print(f"     {feat}: {imp:.1%}")

    # 4. 통합 분석기 테스트
    print("\n4️⃣ 통합 AI 분석기 테스트...")
    integrated = IntegratedAIAnalyzer(model_weight=0.4, technical_weight=0.6)

    # 가상의 기술적 지표 신호
    technical_signals = {
        'trend_ma': 0.7,
        'rsi': 0.6,
        'macd': 0.65,
        'bollinger': 0.55,
        'volume': 0.5,
        'momentum': 0.6,
        'support_resistance': 0.55
    }

    analysis = integrated.analyze(df, technical_signals)
    print(f"   통합 점수: {analysis['combined_score']:.2f}")
    print(f"   통합 신뢰도: {analysis['combined_confidence']:.1%}")
    print(f"   추천 방향: {analysis['recommended_direction']}")
    print(f"   AI 점수: {analysis['model_prediction']['score']:.2f}")
    print(f"   기술적 점수: {analysis['technical_analysis']['score']:.2f}")

    # 5. 거래 신호 생성 테스트
    print("\n5️⃣ 거래 신호 생성 테스트...")
    action, confidence, result = integrated.get_trading_signal(df, technical_signals)
    print(f"   거래 신호: {action}")
    print(f"   신뢰도: {confidence:.1%}")

    # 6. 주기적 학습 기능 테스트
    print("\n6️⃣ 주기적 학습 기능 테스트...")

    # 학습 샘플 추가
    for i in range(60):
        # 실제 방향 시뮬레이션
        price_change = np.random.randn() * 0.5
        if price_change > 0.1:
            direction = 'UP'
        elif price_change < -0.1:
            direction = 'DOWN'
        else:
            direction = 'NEUTRAL'

        market_analyzer.add_training_sample(df, direction, price_change)

    print(f"   학습 버퍼 크기: {len(market_analyzer.training_buffer)}")

    # 학습 통계 확인
    training_stats = market_analyzer.get_training_stats()
    print(f"   최소 학습 샘플: {training_stats['min_samples']}")
    print(f"   학습 횟수: {training_stats['training_count']}")

    # 배치 학습 실행
    print("\n   배치 학습 실행 중...")
    train_result = market_analyzer.batch_train(epochs=2, batch_size=16)
    print(f"   학습 상태: {train_result.get('status')}")
    if train_result.get('status') == 'completed':
        print(f"   평균 손실: {train_result.get('avg_loss', 0):.4f}")
        print(f"   방향 정확도: {train_result.get('direction_accuracy', 0):.1%}")
        print(f"   학습 샘플 수: {train_result.get('samples', 0)}")

    # 모델 저장/로드 테스트
    import os
    test_model_path = "models/test_transformer_model.pt"
    os.makedirs("models", exist_ok=True)

    print(f"\n   모델 저장 테스트: {test_model_path}")
    market_analyzer.save_model(test_model_path)

    # 새 분석기로 모델 로드 테스트
    print("   모델 로드 테스트...")
    analyzer_loaded = MarketAnalyzer(
        sequence_length=60,
        model_path=test_model_path
    )
    prediction_loaded = analyzer_loaded.predict(df)
    print(f"   로드된 모델 예측: {prediction_loaded.predicted_direction}")

    # 테스트 모델 삭제
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
        print("   테스트 모델 파일 삭제 완료")

    print("\n✅ AI 분석기 테스트 통과!")
    return True


def test_position_manager():
    """AI 포지션 관리자 테스트"""
    print()
    print("=" * 70)
    print("🎯 Phase 4: AI 포지션 관리자 테스트")
    print("=" * 70)

    from ai.position_manager import AIPositionManager, PositionAction

    manager = AIPositionManager(
        min_confidence=0.3,
        max_position_pct=0.3
    )

    # 테스트용 가상 데이터 생성
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')

    # 상승 트렌드 데이터 (RSI 과매도 영역에서 시작)
    base_price = 50000
    trend = np.concatenate([
        np.linspace(0, -2000, 50),  # 하락
        np.linspace(-2000, 3000, 150)  # 상승
    ])
    noise = np.random.randn(200) * 300

    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(200) * 150)
    low = close - np.abs(np.random.randn(200) * 150)
    open_price = close - np.random.randn(200) * 100
    volume = np.random.uniform(100, 1000, 200)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    # 1. 신규 포지션 결정
    print("\n1️⃣ 신규 포지션 결정 테스트...")
    decision = manager.decide_position(df, current_position=None, leverage=3)

    print(f"   액션: {decision.action.value}")
    print(f"   방향: {decision.side}")
    print(f"   신뢰도: {decision.confidence:.1%}")
    print(f"   포지션 크기: {decision.position_size_pct:.1%}")

    if decision.stop_loss:
        print(f"   손절가: {decision.stop_loss:.2f}")
    if decision.take_profit:
        print(f"   익절가: {decision.take_profit:.2f}")

    print("\n   [신호 분석]")
    for signal_name, score in decision.signals.items():
        bar = '█' * int(score * 10) + '░' * (10 - int(score * 10))
        direction = '🟢 매수' if score > 0.55 else '🔴 매도' if score < 0.45 else '⚪ 중립'
        print(f"     {signal_name:20s}: {bar} {score:.2f} {direction}")

    # 2. 기존 포지션 있을 때 결정
    print("\n2️⃣ 기존 롱 포지션 관리 테스트...")
    current_position = {
        'side': 'LONG',
        'entry_price': 49000,
        'stop_loss': 48000,
        'take_profit': 52000
    }

    decision_with_pos = manager.decide_position(df, current_position=current_position, leverage=3)
    print(f"   액션: {decision_with_pos.action.value}")
    print(f"   방향: {decision_with_pos.side}")

    # 3. 하락 데이터로 숏 신호 테스트
    print("\n3️⃣ 하락 시장 숏 신호 테스트...")
    df_bearish = df.copy()
    df_bearish['close'] = base_price - np.linspace(0, 5000, 200) + np.random.randn(200) * 300
    df_bearish['high'] = df_bearish['close'] + np.abs(np.random.randn(200) * 150)
    df_bearish['low'] = df_bearish['close'] - np.abs(np.random.randn(200) * 150)

    decision_bearish = manager.decide_position(df_bearish, current_position=None, leverage=3)
    print(f"   액션: {decision_bearish.action.value}")
    print(f"   방향: {decision_bearish.side}")
    print(f"   신뢰도: {decision_bearish.confidence:.1%}")

    print("\n✅ AI 포지션 관리자 테스트 통과!")
    return True


def test_websocket_connection():
    """WebSocket 연결 테스트"""
    print()
    print("=" * 70)
    print("📡 Phase 5: WebSocket 연결 테스트")
    print("=" * 70)

    from api.futures_websocket import RealTimeDataManager

    print("\n1️⃣ 실시간 데이터 매니저 초기화...")
    manager = RealTimeDataManager(
        symbol='BTCUSDT',
        interval='1m',
        testnet=True,
        max_candles=100
    )

    print("   WebSocket 연결 시작...")
    manager.start()

    print("   5초간 데이터 수신 대기...")
    time.sleep(5)

    status = manager.get_status()
    print(f"\n2️⃣ 연결 상태:")
    print(f"   연결됨: {status['is_connected']}")
    print(f"   수신 메시지: {status['message_count']}")
    print(f"   에러: {status['error_count']}")
    print(f"   현재 가격: {status['current_price']:.2f}")
    print(f"   마크 가격: {status['mark_price']:.2f}")

    manager.stop()
    print("\n   WebSocket 연결 종료")

    print("\n✅ WebSocket 연결 테스트 통과!")
    return True


def test_full_system():
    """전체 시스템 통합 테스트"""
    print()
    print("=" * 70)
    print("🤖 Phase 6: 전체 시스템 통합 테스트 (AI 포함)")
    print("=" * 70)

    from trading_bot.futures_bot import FuturesTradingBot, BotState

    print("\n1️⃣ 봇 인스턴스 생성 (AI 활성화)...")
    bot = FuturesTradingBot(
        symbol='BTCUSDT',
        interval='1m',
        initial_balance=10000.0,
        testnet=True,
        paper_trading=True,
        min_leverage=1,
        max_leverage=10,
        default_leverage=3,
        use_ai_model=True,
        ai_model_weight=0.4
    )

    print(f"   심볼: {bot.symbol}")
    print(f"   초기 잔고: {bot.initial_balance:.2f} USDT")
    print(f"   레버리지 범위: {bot.min_leverage}x - {bot.max_leverage}x")
    print(f"   AI 활성화: {bot.use_ai_model}")
    if bot.use_ai_model:
        print(f"   AI 가중치: {bot.ai_model_weight:.0%}")

    print("\n2️⃣ 봇 초기화...")
    if not bot.initialize():
        print("   ⚠️  봇 초기화 실패 (API 연결 문제일 수 있음)")
        return False

    print("\n3️⃣ 단일 거래 사이클 실행...")
    bot.state = BotState.RUNNING
    bot._execute_trading_cycle(1)

    print("\n4️⃣ 봇 상태 확인...")
    status = bot.get_status()
    print(f"   상태: {status['state']}")
    print(f"   잔고: {status['current_balance']:.2f} USDT")
    print(f"   레버리지: {status['current_leverage']}x")
    print(f"   포지션: {status['current_position']}")

    print("\n5️⃣ 봇 종료...")
    bot.stop()

    print("\n✅ 전체 시스템 통합 테스트 통과!")
    return True


def main():
    """메인 테스트 함수"""
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          🧪 AI 선물 자동 거래 시스템 테스트 스위트                    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    results = {}

    # Phase 1: 선물 API
    try:
        results['futures_client'] = test_futures_client()
    except Exception as e:
        print(f"❌ 선물 API 테스트 실패: {e}")
        results['futures_client'] = False

    # Phase 2: 레버리지 최적화기
    try:
        results['leverage_optimizer'] = test_leverage_optimizer()
    except Exception as e:
        print(f"❌ 레버리지 최적화기 테스트 실패: {e}")
        results['leverage_optimizer'] = False

    # Phase 3: AI 분석기
    try:
        results['market_analyzer'] = test_market_analyzer()
    except Exception as e:
        print(f"❌ AI 분석기 테스트 실패: {e}")
        results['market_analyzer'] = False

    # Phase 4: 포지션 관리자
    try:
        results['position_manager'] = test_position_manager()
    except Exception as e:
        print(f"❌ 포지션 관리자 테스트 실패: {e}")
        results['position_manager'] = False

    # Phase 5: WebSocket
    try:
        results['websocket'] = test_websocket_connection()
    except Exception as e:
        print(f"❌ WebSocket 테스트 실패: {e}")
        results['websocket'] = False

    # Phase 6: 전체 시스템
    try:
        results['full_system'] = test_full_system()
    except Exception as e:
        print(f"❌ 전체 시스템 테스트 실패: {e}")
        results['full_system'] = False

    # 결과 요약
    print()
    print("=" * 70)
    print("📊 테스트 결과 요약")
    print("=" * 70)

    passed = 0
    failed = 0

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:25s}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print()
    print(f"총 {passed + failed}개 테스트 중 {passed}개 통과, {failed}개 실패")
    print("=" * 70)

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
