"""AI 기반 선물 자동 거래 봇"""
import time
import threading
import pandas as pd
import torch
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from api.futures_client import BinanceFuturesClient
from api.futures_websocket import RealTimeDataManager, UserDataStream
from ai.leverage_optimizer import AILeverageOptimizer, LeverageRecommendation
from ai.position_manager import AIPositionManager, PositionDecision, PositionAction
from ai.unified_analyzer import UnifiedAnalyzer, UnifiedTrainingSample
from regime.market_regime import RegimeDetector, MarketRegime, RegimeInfo
from data.collectors.historical_collector import HistoricalDataCollector
from execution.execution_manager import ExecutionManager, ExecutionContext, ExecutionResult, ExecutionMode
from trading_bot.bot_state_writer import BotStateWriter
from trading_bot.telegram_notifier import TelegramNotifier
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ── 사전학습 데이터 범위 설정 ──────────────────────────────────
PRETRAIN_CANDLE_LIMITS = {
    '15m': 2880,    # ~30일
    '5m':  2880,    # ~10일
    '3m':  2880,    # ~6일
    '1m':  4320,    # ~3일 (72시간)
}
PRETRAIN_TICK_SOURCE_CANDLES = 1440  # 1m 캔들 → 합성 10초봉 (~24시간 = 8640개)


class BotState(Enum):
    """봇 상태"""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


@dataclass
class TradingStats:
    """거래 통계"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    consecutive_losses: int = 0
    best_trade: float = 0.0
    worst_trade: float = 0.0


class FuturesTradingBot:
    """
    AI 기반 선물 자동 거래 봇

    주요 기능:
    1. AI가 자동으로 레버리지 결정
    2. AI가 자동으로 포지션 방향(Long/Short) 결정
    3. 실시간 데이터 기반 의사결정
    4. 자동 SL/TP 관리
    5. 리스크 관리 (드로다운, 연속 손실 제한)
    """

    def __init__(
        self,
        symbol: str = 'BTCUSDT',
        interval: str = '1m',
        initial_balance: float = 1000.0,
        testnet: bool = True,
        paper_trading: bool = True,
        # 레버리지 설정
        min_leverage: int = 1,
        max_leverage: int = 10,
        default_leverage: int = 3,
        # 리스크 설정
        max_position_pct: float = 0.3,
        max_daily_loss_pct: float = 0.05,
        max_drawdown_pct: float = 0.15,
        # 거래 설정
        trade_cooldown_seconds: int = 0,
        min_confidence_long: float = 0.15,  # LONG 신호 최소 신뢰도 (스코어 기반: abs(final) 범위 0.12~0.4)
        min_confidence_short: float = 0.15,  # SHORT 신호 최소 신뢰도
        # AI 모델 설정
        use_ai_model: bool = True,
        ai_model_weight: float = 0.4,
        model_path: Optional[str] = None,
        # Multi-Timeframe 설정
        use_multi_timeframe: bool = True,
        mtf_intervals: List[str] = None,
        # 거래 수수료 설정 (모델 학습용)
        trading_fee_rate: float = 0.0004,  # 0.04% (바이낸스 선물 기본 수수료)
        # 기술적 지표 설정
        enabled_indicators: List[str] = None,
    ):
        """
        Args:
            symbol: 거래 심볼
            interval: 캔들 간격
            initial_balance: 초기 자본 (페이퍼 트레이딩용)
            testnet: 테스트넷 사용 여부
            paper_trading: 페이퍼 트레이딩 여부
            min_leverage: 최소 레버리지
            max_leverage: 최대 레버리지
            default_leverage: 기본 레버리지
            max_position_pct: 최대 포지션 크기 (자본 대비)
            max_daily_loss_pct: 일일 최대 손실율
            max_drawdown_pct: 최대 허용 드로다운
            trade_cooldown_seconds: 거래 간 최소 대기 시간
            min_confidence_long: LONG 신호 최소 신뢰도
            min_confidence_short: SHORT 신호 최소 신뢰도
        """
        self.symbol = symbol.upper()
        self.interval = interval
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.testnet = testnet
        self.paper_trading = paper_trading

        # 레버리지 설정
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        self.default_leverage = default_leverage  # 사용자 선택 레버리지 보존
        self.current_leverage = default_leverage

        # 리스크 설정
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.trade_cooldown = trade_cooldown_seconds
        self.min_confidence_long = min_confidence_long
        self.min_confidence_short = min_confidence_short

        # AI 모델 설정
        self.use_ai_model = use_ai_model
        self.ai_model_weight = ai_model_weight
        self.model_path = model_path

        # Multi-Timeframe 설정
        self.use_multi_timeframe = use_multi_timeframe
        self.mtf_intervals = mtf_intervals or ['1m', '3m', '5m', '15m']

        # AI 모델 학습 설정
        self.model_training_interval_minutes = 30  # 학습 주기 (분)
        self.model_save_path = model_path or f"models/transformer_{symbol}_{interval}.pt"
        self.last_prediction_direction: Optional[str] = None
        self.last_prediction_price: Optional[float] = None


        # 거래 수수료 설정 (AI 모델 학습 시 수수료 고려)
        # 왕복 거래 시 수수료: 진입 + 청산 = 2배
        self.trading_fee_rate = trading_fee_rate

        # 기술적 지표 설정
        self.enabled_indicators = enabled_indicators or ['trend_ma', 'rsi', 'macd', 'bollinger']


        # 상태 변수
        self.state = BotState.IDLE
        self.current_position: Optional[Dict] = None
        self.last_trade_time: Optional[datetime] = None
        self.daily_pnl: float = 0.0
        self.daily_reset_date = datetime.now().date()

        # 통계
        self.stats = TradingStats()
        self.trade_history: List[Dict] = []

        # Kelly Criterion 사이징
        self._kelly_window: int = 100          # 최근 100건 기준
        self._kelly_ema_alpha: float = 0.1     # EMA 평활 계수
        self._kelly_smoothed: float = 0.10     # 초기값 10%
        self._kelly_max_cap: float = 0.15      # Hard cap 15%
        self._kelly_min_floor: float = 0.03    # 최소 3% (0 수렴 방지)

        # API 클라이언트
        self.client = BinanceFuturesClient(testnet=testnet)

        # AI 시스템
        self.leverage_optimizer = AILeverageOptimizer(
            min_leverage=min_leverage,
            max_leverage=max_leverage,
            default_leverage=default_leverage
        )
        from config.config import Config
        self.position_manager = AIPositionManager(
            min_confidence_long=min_confidence_long,
            min_confidence_short=min_confidence_short,
            max_position_pct=max_position_pct,
            enabled_indicators=self.enabled_indicators,
            volatility_threshold=Config.VOLATILITY_FILTER_THRESHOLD,
        )

        # v2 학습 파이프라인: 피처 빌더 (AI 분석기보다 먼저 생성)
        from ai.feature_builder import TFFeatureBuilder
        self.feature_builder = TFFeatureBuilder()

        # 레거시 분석기 스텁 (잔존 guard 조건 호환용 — 항상 None/False)
        self.market_analyzer = None
        self.integrated_analyzer = None
        self.mtf_analyzer = None
        self.tick_analyzer = None
        self.hybrid_analyzer = None
        self.use_tick_model = False

        # ── 통합 모델 (Multi-Scale TCN + GRU Pool) ──
        self.unified_analyzer: Optional[UnifiedAnalyzer] = None

        self._last_unified_prediction = None

        if use_ai_model:
            try:
                self.unified_analyzer = UnifiedAnalyzer(symbol=self.symbol)
                logger.info("🧠 통합 모델 활성화 (Multi-Scale TCN + GRU Pool)")
            except Exception as e:
                logger.warning(f"통합 모델 초기화 실패: {e}")
                self.use_ai_model = False

        # 실시간 데이터 관리자
        self.data_manager: Optional[RealTimeDataManager] = None
        self.user_stream: Optional[UserDataStream] = None

        # 히스토리컬 데이터 수집기 — 메인넷 선물 API 사용
        # kline은 공개 API이므로 테스트넷 관계없이 메인넷에서 수집 (풀 히스토리)
        _hist_client = BinanceFuturesClient(testnet=False)
        self.historical_collector = HistoricalDataCollector(client=_hist_client)

        # 스레드
        self._trading_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 스레드 안전 락 (RLock: 같은 스레드 재진입 허용)
        self._lock = threading.RLock()

        # 대시보드용 상태 기록기
        self.state_writer = BotStateWriter()

        # 텔레그램 실시간 알림
        from config.config import Config
        self.telegram = TelegramNotifier(
            bot_token=Config.TELEGRAM_BOT_TOKEN,
            chat_id=Config.TELEGRAM_CHAT_ID,
            throttle_seconds=30.0,
        )

        # AI 분석 결과 캐시 (대시보드 표시용)
        self._last_ai_info: Optional[Dict] = None

        # 시장 상태(Regime) 감지기
        self.regime_detector = RegimeDetector()
        self._current_regime: Optional[RegimeInfo] = None

        # 현재 ATR (트레일링 스톱 동적 계산용)
        self._current_atr: float = 0.0

        # ── 3계층 리스크 아키텍처 상수 ──
        # Layer 1: 거래소 하드 SL (생존 레이어)
        self.LAYER1_SL_ATR_NORMAL: float = 1.8   # WS 정상 시
        self.LAYER1_SL_ATR_TIGHT: float = 1.0    # WS 단절 폴백
        # Layer 2: WS 소프트 트레일링 (전략 레이어)
        self.LAYER2_ACTIVATION_ATR: float = 0.8   # 수익 활성화 임계
        self.LAYER2_OFFSET_ATR: float = 0.4       # 고점 대비 오프셋
        self._ws_sl_mode: str = 'normal'           # 'normal' | 'tight'
        # 캔들 중복 추론/스냅샷 방지 (라이브 전용)
        self._last_inference_candle_ts: float = 0.0
        self._last_snapshot_candle_ts: float = 0.0

        # 사이클별 구조화 로그 컨텍스트
        self._cycle_ctx: Dict[str, Any] = {}

        # 놓친 기회 스캐너 (Missed Opportunity Scanner)
        from collections import deque
        self._price_ring: deque = deque(maxlen=300)  # (epoch_sec, price) — 최대 5분 (10s 간격 × 30)
        self._missed_opportunities: List[Dict] = []  # 최근 50건
        self._last_missed_check: float = 0.0

        # 스마트 주문 실행 엔진
        self.execution_manager = ExecutionManager(
            client=self.client,
            symbol=self.symbol,
            paper_trading=self.paper_trading,
            cascade_conf_getter=lambda: self._cycle_ctx.get('cascade_conf', 0),
            candle_low_high_getter=lambda: self._get_candle_low_high(),
        )

        # 히스토리컬 백테스트 모드
        self._historical_mode: bool = False
        self._historical_df: Optional[pd.DataFrame] = None
        self._hist_cursor: int = 0
        self._hist_train_counter: int = 0
        self._backtest_allow_training: bool = True  # OOS에서 False
        self._backtest_is_pretrain: bool = True  # historical periodic training의 is_pretrain 제어
        self._sim_time: float = 0.0  # 시뮬레이션 시간 (backtest: 캔들 시간, live: time.time())

        # v2 학습 파이프라인: 링 버퍼 + 지연 라벨링
        from ai.sample_buffer import SampleRingBuffer, PriceLog
        self.price_log = PriceLog(maxlen=50000)
        self.ring_buffers: Dict[str, SampleRingBuffer] = {}
        # 1m 기본 버퍼
        self.ring_buffers['1m'] = SampleRingBuffer('1m')
        # MTF 버퍼
        for tf in self.mtf_intervals:
            if tf not in self.ring_buffers:
                self.ring_buffers[tf] = SampleRingBuffer(tf)

        # TF별 라벨 카운트 (inverse frequency 계산용)
        self._label_counts: Dict[str, Dict[str, int]] = {}

        # 통합 모델 학습 파이프라인: 3-TF 동시 스냅샷 버퍼
        self._unified_pending_snapshots: deque = deque(maxlen=2000)
        self._UNIFIED_LOOK_AHEAD_S: int = 900  # 15분 (3m × 5캔들)

        # 통합 모델 진단: 최근 예측 히스토리 (출력 분산 측정용)
        self._unified_pred_history: deque = deque(maxlen=200)

        # ── AR(1) 전략 상태 ──
        # 방향 신호: 이전 비-NEUTRAL 라벨 반복 (80% accuracy 확인됨)
        self._ar1_last_direction: str = 'NEUTRAL'          # 마지막 비-NEUTRAL 라벨
        self._ar1_label_history: deque = deque(maxlen=300)  # Rolling AC(1) — 거래 기반 300개 (NEUTRAL 제외)

        logger.info("=" * 70)
        logger.info("🤖 AI 선물 자동 거래 봇 초기화")
        logger.info("=" * 70)
        logger.info(f"  심볼: {symbol}")
        logger.info(f"  간격: {interval}")
        logger.info(f"  초기 자본: {initial_balance:.2f} USDT")
        logger.info(f"  기본 레버리지: {default_leverage}x (범위: {min_leverage}x - {max_leverage}x)")
        logger.info(f"  테스트넷: {testnet}")
        logger.info(f"  페이퍼 트레이딩: {paper_trading}")
        logger.info(f"  AI 모델 사용: {use_ai_model}")
        if use_ai_model:
            logger.info(f"  AI 가중치: {ai_model_weight:.0%}")
            logger.info(f"  AI 학습 주기: {self.model_training_interval_minutes}분")
        if self.use_multi_timeframe:
            logger.info(f"  MTF 시간대: {self.mtf_intervals}")
        if self.use_tick_model:
            logger.info(f"  ⚡ 틱 모델: 활성화 (10초 마이크로캔들)")
            logger.info(f"  ⚡ 틱 학습 주기: {self.tick_training_interval_minutes}분")
        logger.info("=" * 70)

        # 모델 저장 디렉토리 생성
        import os
        os.makedirs(os.path.dirname(self.model_save_path) if os.path.dirname(self.model_save_path) else "models", exist_ok=True)

    def initialize(self) -> bool:
        """봇 초기화"""
        try:
            logger.info("봇 초기화 중...")

            # 1. API 연결 테스트
            if not self.client.ping():
                logger.error("선물 API 연결 실패")
                return False

            # 2. 레버리지 설정
            if not self.paper_trading:
                try:
                    self.client.set_leverage(self.symbol, self.current_leverage)
                    self.client.set_margin_type(self.symbol, 'ISOLATED')
                except Exception as e:
                    logger.warning(f"레버리지 설정 경고: {e}")

            # 3. 실시간 데이터 매니저 생성 (초기 데이터 로드 전에 생성해야 함)
            logger.info("실시간 데이터 스트림 시작 중...")
            _max_candles = PRETRAIN_CANDLE_LIMITS.get(self.interval, 4320) + 100
            self.data_manager = RealTimeDataManager(
                symbol=self.symbol,
                interval=self.interval,
                testnet=self.testnet,
                max_candles=_max_candles
            )

            # 4. 초기 데이터 로드 (data_manager에 저장)
            logger.info("초기 시장 데이터 로드 중...")
            self._load_initial_data()

            # 4.5. Layer 2/1 콜백 등록 (start 전에 등록해야 함)
            self.data_manager.register_mark_price_callback(self._on_mark_price_update)
            self.data_manager.set_ws_state_callbacks(
                on_disconnect=self._on_ws_disconnect,
                on_reconnect=self._on_ws_reconnect,
            )

            # 5. WebSocket 스트림 시작
            self.data_manager.start()

            # 6. 사용자 데이터 스트림 (실제 거래 시)
            if not self.paper_trading:
                self.user_stream = UserDataStream(
                    futures_client=self.client,
                    on_account_update=self._on_account_update,
                    on_order_update=self._on_order_update,
                    testnet=self.testnet
                )
                self.user_stream.start()

            # 잠시 대기 (WebSocket 연결 안정화)
            time.sleep(2)

            logger.info("✅ 봇 초기화 완료")
            return True

        except Exception as e:
            logger.error(f"봇 초기화 실패: {e}")
            self.state = BotState.ERROR
            return False

    def _load_initial_data(self):
        """초기 히스토리컬 데이터 로드"""
        try:
            # REST API로 초기 캔들 데이터 로드 (확장된 범위)
            _target = PRETRAIN_CANDLE_LIMITS.get(self.interval, 4320)
            logger.info(f"초기 데이터 로드 중: {self.interval} {_target}개 캔들 목표...")

            klines = self.client.get_klines_paginated(
                symbol=self.symbol,
                interval=self.interval,
                total_candles=_target
            )

            for kline in klines:
                candle = {
                    'timestamp': kline[0],
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'is_closed': True
                }
                if self.data_manager:
                    self.data_manager.candles.append(candle)

            logger.info(f"초기 데이터 로드 완료: {len(klines)}개 캔들")

            # 통합 모델 로드 (사전학습된 모델이 있을 때)
            if self.use_ai_model and self.unified_analyzer:
                ready_path = f"models/unified_ready_{self.symbol}.pt"
                if self.unified_analyzer.load_model(ready_path):
                    logger.info(f"✅ 통합 모델 로드 완료: {ready_path}")
                else:
                    logger.warning(
                        f"사전학습 모델 없음 ({ready_path}), "
                        f"랜덤 초기화 상태로 시작 (Shadow Mode 보호 활성)"
                    )

        except Exception as e:
            logger.error(f"초기 데이터 로드 실패: {e}")

    @staticmethod
    def _expand_training_buffer(analyzer, expected_samples: int) -> int:
        """사전학습용 training_buffer 임시 확장 (deque.maxlen은 read-only이므로 교체)"""
        from collections import deque
        original_maxlen = analyzer.training_buffer.maxlen
        if expected_samples > original_maxlen:
            old_items = list(analyzer.training_buffer)
            analyzer.training_buffer = deque(old_items, maxlen=expected_samples + 100)
        return original_maxlen

    @staticmethod
    def _restore_training_buffer(analyzer, original_maxlen: int):
        """사전학습 후 training_buffer 원래 크기로 복원 (최근 샘플 유지)"""
        from collections import deque
        if analyzer.training_buffer.maxlen != original_maxlen:
            recent = list(analyzer.training_buffer)[-original_maxlen:]
            analyzer.training_buffer = deque(recent, maxlen=original_maxlen)

    def run_backtest(self, start_date: str, end_date: str, step: int = 1):
        """히스토리컬 백테스트: 과거 데이터로 모델 학습 + 거래 시뮬레이션

        Args:
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            step: 캔들 스킵 간격 (1=매 캔들, 5=5분마다)
        """
        import math

        logger.info("=" * 70)
        logger.info("📊 히스토리컬 백테스트 시작")
        logger.info(f"  심볼: {self.symbol}, 기간: {start_date} ~ {end_date}, step: {step}")
        logger.info("=" * 70)

        # 1. 과거 1m 데이터 다운로드
        logger.info("📥 과거 데이터 다운로드 중...")
        df = self.historical_collector.collect_klines(
            symbol=self.symbol,
            interval='1m',
            start_date=start_date,
            end_date=end_date,
            save_to_csv=True
        )

        if df.empty or len(df) < 200:
            logger.error(f"데이터 부족: {len(df)}개 캔들 (최소 200개 필요)")
            return

        # timestamp를 인덱스로 설정 (리샘플링용)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        # 필요한 컬럼만 유지 (trades, taker_buy_base: 피처용)
        _KEEP = ['open', 'high', 'low', 'close', 'volume', 'trades', 'taker_buy_base']
        df = df[[c for c in _KEEP if c in df.columns]].copy()

        logger.info(f"✅ 데이터 준비 완료: {len(df)}개 캔들 ({df.index[0]} ~ {df.index[-1]})")

        # 2. 히스토리컬 모드 활성화
        self._historical_mode = True
        self._historical_df = df
        self._hist_cursor = 0
        self._hist_train_counter = 0
        self._backtest_tick_df = None
        self._hist_tick_train_counter = 0

        # 3. API 연결 테스트 (데이터 다운로드에 필요했으므로 이미 확인됨)
        # 모델/분석기 초기화 (WebSocket 없이)
        logger.info("🧠 AI 모델 초기화 중...")
        import os
        os.makedirs(os.path.dirname(self.model_save_path) if os.path.dirname(self.model_save_path) else "models", exist_ok=True)

        # ── IS/OOS 분할 먼저 계산 (사전학습 데이터 누수 방지) ──
        total_candles = len(df)
        start_cursor = 70  # v2 피처: sequence_length(60) + 여유분
        oos_split = int(total_candles * 0.6)  # 60% 지점
        oos_split = max(oos_split, start_cursor + 100)  # 최소 IS 구간 보장
        logger.info(f"📐 Walk-Forward 분할 계산: IS=0~{oos_split} ({oos_split}캔들) | "
                     f"OOS={oos_split}~{total_candles} ({total_candles - oos_split}캔들)")

        # 4. 사전 학습 — IS 데이터만 사용 (OOS 데이터 누수 방지)
        self._hist_cursor = min(200, len(df))
        if self.use_ai_model:
            logger.info(f"🧠 사전 학습 시작 (IS 데이터만 사용: {oos_split}캔들)...")
            full_df = df.iloc[:oos_split].copy()  # IS 구간만 사용 (OOS 누수 방지)

            # 1m 모델 사전 학습 — Triple Barrier 라벨링 (라이브와 동일)
            _1m_bt_analyzer = self._get_analyzer_for_tf('1m')
            if _1m_bt_analyzer:
                try:
                    from ai.market_analyzer import TrainingSample
                    from ai.sample_buffer import CANDLE_LOOK_AHEAD, triple_barrier_label_precompute
                    from ai.feature_builder import TFFeatureBuilder

                    look_ahead = CANDLE_LOOK_AHEAD['1m']  # 15캔들
                    seq_len = _1m_bt_analyzer.sequence_length

                    logger.info(f"  1m 피처 사전 계산 중... ({len(full_df)}행, look_ahead={look_ahead}캔들)")
                    all_features = _1m_bt_analyzer.prepare_features(full_df)
                    close_arr = full_df['close'].values
                    high_arr = full_df['high'].values
                    low_arr = full_df['low'].values
                    n_rows = len(all_features)

                    # ATR 사전 계산 (Triple Barrier 배리어 스케일링용)
                    atr_series = TFFeatureBuilder.compute_raw_atr(full_df)
                    atr_arr = atr_series.values

                    samples_added = 0
                    for i in range(seq_len, n_rows - look_ahead):
                        sequence = all_features[i - seq_len:i].copy()
                        entry_price = close_arr[i - 1]
                        atr_val = atr_arr[i]

                        # Triple Barrier: 캔들별 H/L 체크
                        direction, net_log_return, barrier_type, _timing = triple_barrier_label_precompute(
                            entry_price=entry_price,
                            atr=atr_val,
                            high_arr=high_arr[i:i + look_ahead],
                            low_arr=low_arr[i:i + look_ahead],
                            close_arr=close_arr[i:i + look_ahead],
                        )

                        # 2-class: NEUTRAL(TIME/CONFLICT/cost미달) → 학습 제외
                        if direction == 'NEUTRAL':
                            continue

                        exit_price = entry_price * math.exp(net_log_return)
                        net_pct = (exit_price / entry_price - 1) * 100 - self.trading_fee_rate * 2 * 100
                        net_log_return = net_log_return - 2 * self.trading_fee_rate

                        sample = TrainingSample(
                            features=sequence,
                            actual_direction=direction,
                            actual_price_change=net_pct,
                            timestamp=full_df.index[i],
                            sample_weight=max(0.3, _timing),
                            future_log_return=net_log_return
                        )
                        _1m_bt_analyzer.training_buffer.append(sample)
                        samples_added += 1

                    logger.info(f"  1m 샘플: {samples_added}개 (버퍼={len(_1m_bt_analyzer.training_buffer)})")

                    if samples_added >= _1m_bt_analyzer.min_samples_for_training:
                        result = _1m_bt_analyzer.batch_train(epochs=5, is_pretrain=True)
                        if result.get('status') == 'completed':
                            _1m_bt_analyzer.save_model(self.model_save_path)
                            logger.info(f"  ✅ 1m 사전 학습 완료: 정확도={result.get('direction_accuracy', 0):.1%}, "
                                       f"저장={self.model_save_path}")
                        else:
                            logger.warning(f"  ⚠️ 1m 사전 학습 실패: {result.get('reason', result.get('error', 'unknown'))}")
                    else:
                        logger.info(f"  ⚠️ 1m 사전 학습 스킵: 샘플 부족 ({samples_added})")
                except Exception as e:
                    import traceback
                    logger.error(f"  1m 사전 학습 에러: {e}")
                    traceback.print_exc()



        # 5. Walk-Forward OOS 검증 구조 (IS/OOS 분할은 Step 4 이전에 계산됨)
        # ── Phase A: In-Sample (IS) 거래 + 학습 ──
        logger.info(f"📐 Walk-Forward 실행: IS={start_cursor}~{oos_split} ({oos_split - start_cursor}캔들) | "
                     f"OOS={oos_split}~{total_candles} ({total_candles - oos_split}캔들)")

        # ── Phase A: In-Sample (IS) 거래 + 학습 ──
        self._hist_cursor = start_cursor
        self.stats = TradingStats()
        self.trade_history = []
        self.current_balance = self.initial_balance
        self.current_position = None
        self.daily_pnl = 0.0
        self._backtest_allow_training = True  # IS: 학습 허용

        logger.info("")
        logger.info(f"🚀 [IS 구간] 백테스트 시작: {start_cursor} → {oos_split} (step={step})")

        cycle_count = 0
        is_progress_interval = max(1, (oos_split - start_cursor) // 10)

        while self._hist_cursor < oos_split:
            try:
                cycle_count += 1
                if not self._check_risk_limits():
                    logger.warning("⚠️ IS 리스크 한도 도달")
                    break
                self._execute_trading_cycle(cycle_count)
                if cycle_count % is_progress_interval == 0:
                    progress = (self._hist_cursor - start_cursor) / (oos_split - start_cursor) * 100
                    current_pnl = self.current_balance - self.initial_balance
                    logger.info(f"📊 [IS] {progress:.0f}% | PnL: {current_pnl:+.2f} | 거래: {self.stats.total_trades}건")
                self._hist_cursor += step
            except Exception as e:
                logger.error(f"IS 루프 오류: {e}", exc_info=True)
                self._hist_cursor += step

        # IS 포지션 강제 청산
        if self.current_position:
            is_close_price = df.iloc[min(self._hist_cursor, total_candles - 1)]['close']
            self._force_close_position(is_close_price)

        # IS 결과 스냅샷
        is_stats = TradingStats(
            total_trades=self.stats.total_trades,
            winning_trades=self.stats.winning_trades,
            losing_trades=self.stats.losing_trades,
            total_pnl=self.stats.total_pnl,
            best_trade=self.stats.best_trade,
            worst_trade=self.stats.worst_trade,
            max_drawdown=self.stats.max_drawdown,
            consecutive_losses=0,
        )
        is_balance = self.current_balance
        is_trades = list(self.trade_history)

        # ── Phase B: Out-of-Sample (OOS) 거래 — 학습 동결 ──
        self.stats = TradingStats()
        self.trade_history = []
        self.current_balance = self.initial_balance  # 동일 자본으로 리셋
        self.current_position = None
        self.daily_pnl = 0.0
        self._backtest_allow_training = False  # OOS: 학습 금지
        self._hist_tick_train_counter = 0  # 10s 학습 카운터 리셋 (tick_df는 유지: 연속성)

        logger.info("")
        logger.info(f"🔒 [OOS 구간] 백테스트 시작: {oos_split} → {total_candles} (학습 동결)")

        oos_cycle = 0
        oos_progress_interval = max(1, (total_candles - oos_split) // 10)

        while self._hist_cursor < total_candles:
            try:
                oos_cycle += 1
                if not self._check_risk_limits():
                    logger.warning("⚠️ OOS 리스크 한도 도달")
                    break
                self._execute_trading_cycle(oos_cycle)
                if oos_cycle % oos_progress_interval == 0:
                    progress = (self._hist_cursor - oos_split) / (total_candles - oos_split) * 100
                    current_pnl = self.current_balance - self.initial_balance
                    logger.info(f"📊 [OOS] {progress:.0f}% | PnL: {current_pnl:+.2f} | 거래: {self.stats.total_trades}건")
                self._hist_cursor += step
            except Exception as e:
                logger.error(f"OOS 루프 오류: {e}", exc_info=True)
                self._hist_cursor += step

        # OOS 포지션 강제 청산
        if self.current_position:
            last_price = df.iloc[-1]['close']
            self._force_close_position(last_price)

        oos_stats = self.stats
        oos_balance = self.current_balance
        oos_trades = self.trade_history

        # 7. IS vs OOS 비교 결과 출력
        self._print_walkforward_results(
            start_date, end_date, total_candles,
            is_stats, is_balance, is_trades, oos_split - start_cursor,
            oos_stats, oos_balance, oos_trades, total_candles - oos_split,
        )

        # 8. 모델 저장
        if self.use_ai_model:
            if self.use_multi_timeframe and self.mtf_analyzer:
                # MTF 모드: mtf_analyzer가 1m 포함 모든 TF 저장 (standalone 저장 불필요)
                for interval, analyzer in self.mtf_analyzer.analyzers.items():
                    save_path = f"models/transformer_{self.symbol}_{interval}_base.pt"
                    analyzer.save_model(save_path)
                logger.info("💾 MTF 모델 저장 완료")
            elif self.market_analyzer:
                # 단일 TF 모드: standalone만 저장
                self.market_analyzer.save_model(self.model_save_path)
                logger.info(f"💾 1m 모델 저장: {self.model_save_path}")
            if self.use_tick_model and self.tick_analyzer:
                self.tick_analyzer.save_model(self.tick_model_save_path)
                logger.info(f"💾 10s 틱 모델 저장: {self.tick_model_save_path}")

        # 9. 히스토리컬 모드 해제
        self._historical_mode = False
        self._historical_df = None
        self._hist_cursor = 0
        self._backtest_allow_training = True
        self._backtest_tick_df = None
        self._hist_tick_train_counter = 0

        logger.info("")
        logger.info("✅ 히스토리컬 백테스트 완료!")
        logger.info("   → 저장된 모델로 paper/live 트레이딩에서 이어서 학습 가능")

    def _force_close_position(self, price: float):
        """백테스트 종료 시 열린 포지션 강제 청산.

        _close_position과 동일한 PnL/수수료 기준 사용 (왕복 수수료).
        """
        if not self.current_position:
            return

        pos = self.current_position
        side = pos['side']
        entry_price = pos['entry_price']
        quantity = pos['quantity']
        leverage = pos.get('leverage', self.current_leverage)

        if side == 'LONG':
            gross_pnl = (price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - price) * quantity

        # 왕복 수수료 (_close_position과 동일)
        fee = (entry_price + price) * quantity * self.trading_fee_rate
        net_pnl = gross_pnl - fee

        margin = entry_price * quantity / leverage
        pnl_pct = net_pnl / margin * 100 if margin > 0 else 0.0

        with self._lock:
            self.current_balance += net_pnl
            self.stats.total_pnl += net_pnl
            # total_trades는 _open_position에서 이미 증가됨 — 중복 증가 방지

            if net_pnl > 0:
                self.stats.winning_trades += 1
                self.stats.best_trade = max(self.stats.best_trade, net_pnl)
            else:
                self.stats.losing_trades += 1
                self.stats.worst_trade = min(self.stats.worst_trade, net_pnl)

        self.trade_history.append({
            'side': side,
            'entry_price': entry_price,
            'exit_price': price,
            'quantity': quantity,
            'leverage': leverage,
            'gross_pnl': gross_pnl,
            'fee': fee,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'reason': 'Force Close (Backtest End)',
            'entry_time': pos.get('entry_time'),
            'exit_time': datetime.now(),
            'balance_after': self.current_balance,
        })

        with self._lock:
            self.current_position = None

    def _print_backtest_results(self, start_date: str, end_date: str, total_candles: int):
        """백테스트 결과 요약 출력"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 히스토리컬 백테스트 결과")
        logger.info("=" * 70)
        logger.info(f"  기간: {start_date} ~ {end_date} ({total_candles}캔들)")
        logger.info(f"  초기 자본: {self.initial_balance:.2f} USDT")
        logger.info(f"  최종 자본: {self.current_balance:.2f} USDT")

        total_return = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        logger.info(f"  총 수익률: {total_return:+.2f}%")
        logger.info(f"  총 손익: {self.stats.total_pnl:+.2f} USDT")
        logger.info(f"  총 거래: {self.stats.total_trades}건")

        if self.stats.total_trades > 0:
            win_rate = self.stats.winning_trades / self.stats.total_trades * 100
            logger.info(f"  승률: {win_rate:.1f}% ({self.stats.winning_trades}승 {self.stats.losing_trades}패)")
            logger.info(f"  최고 수익 거래: {self.stats.best_trade:+.2f} USDT")
            logger.info(f"  최악 손실 거래: {self.stats.worst_trade:+.2f} USDT")

            # 평균 수익/손실
            wins = [t['pnl'] for t in self.trade_history if t.get('pnl', 0) > 0]
            losses = [t['pnl'] for t in self.trade_history if t.get('pnl', 0) < 0]
            if wins:
                logger.info(f"  평균 수익: {sum(wins)/len(wins):+.2f} USDT")
            if losses:
                logger.info(f"  평균 손실: {sum(losses)/len(losses):+.2f} USDT")

            # Profit Factor
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 0
            if total_losses > 0:
                profit_factor = total_wins / total_losses
                logger.info(f"  Profit Factor: {profit_factor:.2f}")

            # ── 핵심 3대 지표 ──
            logger.info("-" * 70)
            logger.info("🎯 핵심 전략 지표")

            # ① Conditional TP rate: P(TP | trade)
            tp_trades = [t for t in self.trade_history if t.get('reason') == 'Take Profit']
            tp_rate = len(tp_trades) / self.stats.total_trades * 100
            logger.info(f"  ① TP Rate: {tp_rate:.1f}% ({len(tp_trades)}/{self.stats.total_trades})")

            # ② R multiple: avg_win / avg_loss
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = abs(sum(losses) / len(losses)) if losses else 0
            r_multiple = avg_win / avg_loss if avg_loss > 0 else float('inf')
            logger.info(f"  ② R Multiple: {r_multiple:.2f}  (avg_win={avg_win:+.2f} / avg_loss={avg_loss:.2f})")

            # ③ Net EV per trade: mean(pnl_i)
            all_pnls = [t['pnl'] for t in self.trade_history if 'pnl' in t]
            net_ev = sum(all_pnls) / len(all_pnls) if all_pnls else 0
            logger.info(f"  ③ Net EV/trade: {net_ev:+.4f} USDT")
            logger.info("-" * 70)

        # 최대 드로다운 계산
        if self.trade_history:
            equity_curve = [self.initial_balance]
            for t in self.trade_history:
                equity_curve.append(t.get('balance_after', equity_curve[-1]))
            peak = equity_curve[0]
            max_dd = 0
            for eq in equity_curve:
                peak = max(peak, eq)
                dd = (peak - eq) / peak
                max_dd = max(max_dd, dd)
            logger.info(f"  최대 드로다운: {max_dd:.1%}")

        # AI 학습 통계 — MTF 모드에서는 mtf_analyzer가 모든 TF 포함
        if self.use_ai_model:
            logger.info("-" * 70)
            logger.info("🧠 AI 학습 통계")
            if self.use_multi_timeframe and self.mtf_analyzer:
                mtf_stats = self.mtf_analyzer.get_training_stats()
                for iv, st in mtf_stats.items():
                    logger.info(f"  {iv} 학습 횟수: {st['training_count']}, "
                               f"버퍼: {st['buffer_size']}, "
                               f"방향 정확도: {st.get('prediction_accuracy', {}).get('direction_accuracy', 0):.1%}")
            elif self.market_analyzer:
                ts = self.market_analyzer.get_training_stats()
                logger.info(f"  1m 학습 횟수: {ts['training_count']}, "
                           f"버퍼: {ts['buffer_size']}, "
                           f"방향 정확도: {ts.get('prediction_accuracy', {}).get('direction_accuracy', 0):.1%}")

        # equity curve CSV 저장
        if self.trade_history:
            try:
                eq_df = pd.DataFrame(self.trade_history)
                eq_path = f"backtesting/results/backtest_{self.symbol}_{start_date}_{end_date}.csv"
                os.makedirs(os.path.dirname(eq_path), exist_ok=True)
                eq_df.to_csv(eq_path, index=False)
                logger.info(f"  📁 결과 저장: {eq_path}")
            except Exception as e:
                logger.warning(f"결과 저장 실패: {e}")

        logger.info("=" * 70)

    def _print_walkforward_results(
        self, start_date, end_date, total_candles,
        is_stats, is_balance, is_trades, is_candles,
        oos_stats, oos_balance, oos_trades, oos_candles,
    ):
        """Walk-Forward IS vs OOS 비교 결과 출력"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 Walk-Forward 백테스트 결과 (IS vs OOS)")
        logger.info("=" * 70)
        logger.info(f"  기간: {start_date} ~ {end_date} ({total_candles}캔들)")
        logger.info(f"  분할: IS={is_candles}캔들 (60%) | OOS={oos_candles}캔들 (40%)")
        logger.info("-" * 70)

        def _section(label, stats, balance, trades, candles):
            ret = (balance - self.initial_balance) / self.initial_balance * 100
            logger.info(f"  [{label}]")
            logger.info(f"    수익률: {ret:+.2f}%  |  PnL: {stats.total_pnl:+.2f} USDT")
            logger.info(f"    거래: {stats.total_trades}건  |  "
                         f"승: {stats.winning_trades}  패: {stats.losing_trades}")
            if stats.total_trades > 0:
                wr = stats.winning_trades / stats.total_trades * 100
                logger.info(f"    승률: {wr:.1f}%")

                wins = [t['pnl'] for t in trades if t.get('pnl', 0) > 0]
                losses = [t['pnl'] for t in trades if t.get('pnl', 0) < 0]
                avg_win = sum(wins) / len(wins) if wins else 0
                avg_loss = abs(sum(losses) / len(losses)) if losses else 0
                rr = avg_win / avg_loss if avg_loss > 0 else float('inf')
                logger.info(f"    평균 수익: {avg_win:+.2f}  |  평균 손실: {-avg_loss:+.2f}  |  RR: {rr:.2f}")

                # Profit Factor
                total_wins = sum(wins) if wins else 0
                total_losses_abs = abs(sum(losses)) if losses else 0
                pf = total_wins / total_losses_abs if total_losses_abs > 0 else float('inf')
                logger.info(f"    Profit Factor: {pf:.2f}")

                # Expectancy (기대값)
                expectancy = stats.total_pnl / stats.total_trades
                logger.info(f"    기대값/거래: {expectancy:+.2f} USDT")

            # Max Drawdown
            if trades:
                equity_curve = [self.initial_balance]
                for t in trades:
                    equity_curve.append(t.get('balance_after', equity_curve[-1]))
                peak = equity_curve[0]
                max_dd = 0
                for eq in equity_curve:
                    peak = max(peak, eq)
                    dd = (peak - eq) / peak
                    max_dd = max(max_dd, dd)
                logger.info(f"    최대 드로다운: {max_dd:.1%}")

        _section("IS (In-Sample)", is_stats, is_balance, is_trades, is_candles)
        logger.info("-" * 70)
        _section("OOS (Out-of-Sample)", oos_stats, oos_balance, oos_trades, oos_candles)

        # ── IS vs OOS 품질 비교 ──
        logger.info("-" * 70)
        logger.info("  [IS vs OOS 비교]")

        is_ret = (is_balance - self.initial_balance) / self.initial_balance * 100
        oos_ret = (oos_balance - self.initial_balance) / self.initial_balance * 100
        if is_stats.total_trades > 0 and oos_stats.total_trades > 0:
            is_wr = is_stats.winning_trades / is_stats.total_trades * 100
            oos_wr = oos_stats.winning_trades / oos_stats.total_trades * 100
            wr_decay = oos_wr - is_wr
            ret_decay = oos_ret - is_ret
            logger.info(f"    수익률 변화: IS={is_ret:+.2f}% → OOS={oos_ret:+.2f}% (차이: {ret_decay:+.2f}%)")
            logger.info(f"    승률 변화: IS={is_wr:.1f}% → OOS={oos_wr:.1f}% (차이: {wr_decay:+.1f}%)")

            # 오버핏 경고
            if is_ret > 0 and oos_ret < 0:
                logger.warning("    ⚠️ 오버피팅 의심: IS 수익 → OOS 손실")
            elif is_ret > 0 and oos_ret > 0 and oos_ret < is_ret * 0.3:
                logger.warning("    ⚠️ 오버피팅 가능성: OOS 수익이 IS의 30% 미만")
            elif oos_ret >= is_ret * 0.5:
                logger.info("    ✅ OOS 유지율 양호 (IS 대비 50%+)")

            if oos_wr < 45:
                logger.warning("    ⚠️ OOS 승률 < 45% — 피처/모델 재검토 필요")
        else:
            logger.info(f"    IS 수익률: {is_ret:+.2f}% ({is_stats.total_trades}거래)")
            logger.info(f"    OOS 수익률: {oos_ret:+.2f}% ({oos_stats.total_trades}거래)")

        # AI 학습 통계 — MTF 모드에서는 mtf_analyzer가 모든 TF 포함
        if self.use_ai_model:
            logger.info("-" * 70)
            logger.info("  [AI 학습 통계]")
            if self.use_multi_timeframe and self.mtf_analyzer:
                mtf_stats = self.mtf_analyzer.get_training_stats()
                for iv, st in mtf_stats.items():
                    logger.info(f"    {iv} 학습: {st['training_count']}회, 버퍼: {st['buffer_size']}, "
                               f"방향 정확도: {st.get('prediction_accuracy', {}).get('direction_accuracy', 0):.1%}")
            elif self.market_analyzer:
                ts = self.market_analyzer.get_training_stats()
                logger.info(f"    1m 학습 횟수: {ts['training_count']}, "
                           f"버퍼: {ts['buffer_size']}, "
                           f"방향 정확도: {ts.get('prediction_accuracy', {}).get('direction_accuracy', 0):.1%}")

        # equity curve CSV 저장 (IS + OOS 합본)
        all_trades = is_trades + oos_trades
        if all_trades:
            try:
                # IS/OOS 구분 태그 추가
                for t in is_trades:
                    t['phase'] = 'IS'
                for t in oos_trades:
                    t['phase'] = 'OOS'
                eq_df = pd.DataFrame(all_trades)
                eq_path = f"backtesting/results/walkforward_{self.symbol}_{start_date}_{end_date}.csv"
                import os
                os.makedirs(os.path.dirname(eq_path), exist_ok=True)
                eq_df.to_csv(eq_path, index=False)
                logger.info(f"  📁 결과 저장: {eq_path}")
            except Exception as e:
                logger.warning(f"결과 저장 실패: {e}")

        logger.info("=" * 70)

    # ═══════════════════════════════════════════════════════════════════
    # 4-Stage Pipeline: SSL → SFT → Backtest → Walk-Forward
    # ═══════════════════════════════════════════════════════════════════

    def _cleanup_historical_mode(self):
        """히스토리컬 모드 해제 및 상태 정리."""
        self._historical_mode = False
        self._historical_df = None
        self._hist_cursor = 0
        self._hist_train_counter = 0
        self._backtest_allow_training = True
        self._backtest_is_pretrain = True
        self._backtest_tick_df = None
        self._hist_tick_train_counter = 0

    def _save_ssl_models(self):
        """SSL base 모델 저장: backbone + reconstruction_head + mask_token만 저장."""
        import os
        os.makedirs("models", exist_ok=True)

        supervised_heads = ('price_head', 'direction_head', 'confidence_head',
                            'position_head', 'timing_head')

        def _save_checkpoint(model, save_path, timeframe):
            backbone_state = {
                k: v for k, v in model.state_dict().items()
                if not any(k.startswith(h) for h in supervised_heads)
            }
            checkpoint = {
                'backbone_state': backbone_state,
                'arch_type': type(model).__name__,
                'timeframe': timeframe,
            }
            torch.save(checkpoint, save_path)
            logger.info(f"💾 SSL base 저장: {save_path} ({len(backbone_state)} params)")

        if self.use_multi_timeframe and self.mtf_analyzer:
            for interval, analyzer in self.mtf_analyzer.analyzers.items():
                _save_checkpoint(analyzer.model,
                                 f"models/ssl_base_{self.symbol}_{interval}.pt", interval)
        elif self.market_analyzer:
            _save_checkpoint(self.market_analyzer.model,
                             f"models/ssl_base_{self.symbol}_1m.pt", '1m')

        if self.use_tick_model and self.tick_analyzer:
            _save_checkpoint(self.tick_analyzer.model,
                             f"models/ssl_base_{self.symbol}_10s.pt", '10s')

    def _load_ssl_models(self):
        """SSL base 모델 로드: backbone state만 로드하여 모델에 적용."""
        import os

        def _load_checkpoint(analyzer, load_path, timeframe):
            if not os.path.exists(load_path):
                logger.warning(f"SSL base 모델 없음: {load_path}")
                return False
            checkpoint = torch.load(load_path, map_location=analyzer.device, weights_only=True)
            model_state = analyzer.model.state_dict()
            # backbone state만 적용 (supervised heads는 기존 가중치 유지)
            loaded_keys = 0
            for k, v in checkpoint['backbone_state'].items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
                    loaded_keys += 1
            analyzer.model.load_state_dict(model_state)
            analyzer._ssl_pretrained = True
            logger.info(f"📥 SSL base 로드: {load_path} ({loaded_keys} params, "
                        f"arch={checkpoint.get('arch_type', '?')})")
            return True

        if self.use_multi_timeframe and self.mtf_analyzer:
            for interval, analyzer in self.mtf_analyzer.analyzers.items():
                _load_checkpoint(analyzer,
                                 f"models/ssl_base_{self.symbol}_{interval}.pt", interval)
        elif self.market_analyzer:
            _load_checkpoint(self.market_analyzer,
                             f"models/ssl_base_{self.symbol}_1m.pt", '1m')

        if self.use_tick_model and self.tick_analyzer:
            _load_checkpoint(self.tick_analyzer,
                             f"models/ssl_base_{self.symbol}_10s.pt", '10s')

    def _save_all_models(self, prefix: str = 'trading_ready'):
        """모든 모델을 지정 prefix로 저장."""
        import os
        os.makedirs("models", exist_ok=True)

        if self.use_multi_timeframe and self.mtf_analyzer:
            for interval, analyzer in self.mtf_analyzer.analyzers.items():
                save_path = f"models/{prefix}_{self.symbol}_{interval}.pt"
                analyzer.save_model(save_path)
        elif self.market_analyzer:
            save_path = f"models/{prefix}_{self.symbol}_1m.pt"
            self.market_analyzer.save_model(save_path)

        if self.use_tick_model and self.tick_analyzer:
            save_path = f"models/{prefix}_{self.symbol}_10s.pt"
            self.tick_analyzer.save_model(save_path)

        logger.info(f"💾 모델 저장 완료 (prefix={prefix})")

    def _load_all_models(self, prefix: str = 'trading_ready'):
        """지정 prefix 모델 로드."""
        import os

        if self.use_multi_timeframe and self.mtf_analyzer:
            for interval, analyzer in self.mtf_analyzer.analyzers.items():
                load_path = f"models/{prefix}_{self.symbol}_{interval}.pt"
                if os.path.exists(load_path):
                    analyzer._load_model(load_path)
                    logger.info(f"📥 {interval} 모델 로드: {load_path}")
                else:
                    logger.warning(f"모델 없음: {load_path}")
        elif self.market_analyzer:
            load_path = f"models/{prefix}_{self.symbol}_1m.pt"
            if os.path.exists(load_path):
                self.market_analyzer._load_model(load_path)
                logger.info(f"📥 1m 모델 로드: {load_path}")

        if self.use_tick_model and self.tick_analyzer:
            load_path = f"models/{prefix}_{self.symbol}_10s.pt"
            if os.path.exists(load_path):
                self.tick_analyzer._load_model(load_path)
                logger.info(f"📥 10s 모델 로드: {load_path}")

    def _reset_for_fold(self):
        """Walk-Forward fold 간 trading state만 리셋 (모델 가중치 유지)."""

        self.stats = TradingStats()
        self.trade_history = []
        self.current_balance = self.initial_balance
        self.current_position = None
        self.daily_pnl = 0.0
        self._hist_cursor = 0
        self._hist_train_counter = 0
        self._hist_tick_train_counter = 0
        self._backtest_tick_df = None

    def _collect_phase_metrics(self, trades: list, balance: float) -> dict:
        """거래 내역에서 성과 지표 수집."""
        initial = self.initial_balance
        pnl = balance - initial
        ret_pct = pnl / initial * 100 if initial > 0 else 0.0
        total = len(trades)
        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) < 0]
        win_rate = len(wins) / total * 100 if total > 0 else 0.0

        total_win_pnl = sum(t['pnl'] for t in wins) if wins else 0.0
        total_loss_pnl = abs(sum(t['pnl'] for t in losses)) if losses else 0.0
        profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float('inf')

        # Max drawdown
        max_dd = 0.0
        if trades:
            equity = [initial]
            for t in trades:
                equity.append(t.get('balance_after', equity[-1]))
            peak = equity[0]
            for eq in equity:
                peak = max(peak, eq)
                dd = (peak - eq) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

        # 핵심 3대 지표
        tp_trades = [t for t in trades if t.get('reason') == 'Take Profit']
        tp_rate = len(tp_trades) / total * 100 if total > 0 else 0.0

        win_pnls = [t['pnl'] for t in wins]
        loss_pnls = [t['pnl'] for t in losses]
        avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
        avg_loss = abs(sum(loss_pnls) / len(loss_pnls)) if loss_pnls else 0.0
        r_multiple = avg_win / avg_loss if avg_loss > 0 else float('inf')

        all_pnls = [t['pnl'] for t in trades if 'pnl' in t]
        net_ev = sum(all_pnls) / len(all_pnls) if all_pnls else 0.0

        return {
            'total_trades': total,
            'win_rate': win_rate,
            'return_pct': ret_pct,
            'pnl': pnl,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'balance': balance,
            'wins': len(wins),
            'losses': len(losses),
            'tp_rate': tp_rate,
            'r_multiple': r_multiple,
            'net_ev': net_ev,
        }

    def _print_walkforward_aggregate(self, fold_results: list):
        """Walk-Forward fold별 + 집계 결과 출력."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 Walk-Forward 결과 (Fold별)")
        logger.info("=" * 80)
        logger.info(f"{'Fold':>5} | {'Train Period':>25} | {'Test Period':>25} | "
                    f"{'Return%':>8} | {'WinRate':>7} | {'PF':>6} | {'MaxDD':>6} | {'Trades':>6}")
        logger.info("-" * 80)

        returns = []
        for i, fold in enumerate(fold_results):
            ret = fold.get('return_pct', 0)
            wr = fold.get('win_rate', 0)
            pf = fold.get('profit_factor', 0)
            mdd = fold.get('max_drawdown', 0)
            trades = fold.get('total_trades', 0)
            train_period = fold.get('train_period', '?')
            test_period = fold.get('test_period', '?')
            returns.append(ret)

            pf_str = f"{pf:.2f}" if pf < 100 else "inf"
            logger.info(f"  {i+1:>3} | {train_period:>25} | {test_period:>25} | "
                        f"{ret:>+7.2f}% | {wr:>6.1f}% | {pf_str:>6} | {mdd:>5.1%} | {trades:>6}")

        logger.info("-" * 80)

        # 집계 통계
        n_folds = len(fold_results)
        avg_ret = sum(returns) / n_folds if n_folds > 0 else 0
        profitable_folds = sum(1 for r in returns if r > 0)
        pf_ratio = profitable_folds / n_folds if n_folds > 0 else 0

        logger.info(f"  집계: {n_folds} folds | 평균 수익률: {avg_ret:+.2f}% | "
                    f"수익 fold: {profitable_folds}/{n_folds} ({pf_ratio:.0%})")

        # 핵심 3대 지표 집계
        tp_rates = [f.get('tp_rate', 0) for f in fold_results]
        r_mults = [f.get('r_multiple', 0) for f in fold_results if f.get('r_multiple', 0) < float('inf')]
        net_evs = [f.get('net_ev', 0) for f in fold_results]
        logger.info("-" * 80)
        logger.info("🎯 핵심 전략 지표 (평균)")
        logger.info(f"  ① TP Rate: {sum(tp_rates)/len(tp_rates):.1f}%" if tp_rates else "  ① TP Rate: N/A")
        logger.info(f"  ② R Multiple: {sum(r_mults)/len(r_mults):.2f}" if r_mults else "  ② R Multiple: N/A")
        logger.info(f"  ③ Net EV/trade: {sum(net_evs)/len(net_evs):+.4f} USDT" if net_evs else "  ③ Net EV/trade: N/A")

        if pf_ratio >= 0.5:
            logger.info("  ✅ 50%+ fold 수익 → 과적합 아닐 가능성 높음")
        else:
            logger.warning("  ⚠️ 50% 미만 fold 수익 → 모델 재검토 필요")
        logger.info("=" * 80)

    def run_backtest_new(self, start_date: str, end_date: str, step: int = 1) -> dict:
        """백테스트: trading_ready 모델로 라이브와 동일한 거래 시뮬레이션.

        입력: models/trading_ready_{symbol}_{tf}.pt
        기존 run_backtest()과 달리 사전학습 없이 모델 로드 후 바로 시뮬레이션.

        Args:
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            step: 캔들 스킵 간격
        """


        logger.info("=" * 70)
        logger.info("📊 [Stage 3] Backtest 시작 (라이브 유사)")
        logger.info(f"  심볼: {self.symbol}, 기간: {start_date} ~ {end_date}, step: {step}")
        logger.info("=" * 70)

        # 1. 데이터 다운로드
        logger.info("📥 과거 데이터 다운로드 중...")
        df = self.historical_collector.collect_klines(
            symbol=self.symbol, interval='1m',
            start_date=start_date, end_date=end_date, save_to_csv=True
        )

        if df.empty or len(df) < 200:
            logger.error(f"데이터 부족: {len(df)}개 캔들 (최소 200개 필요)")
            return {'status': 'failed', 'reason': 'insufficient_data'}

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()

        logger.info(f"✅ 데이터 준비 완료: {len(df)}개 캔들")

        # 2. trading_ready 모델 로드
        logger.info("📥 trading_ready 모델 로드 중...")
        self._load_all_models(prefix='trading_ready')

        # 3. Historical mode (라이브 유사)
        self._historical_mode = True
        self._historical_df = df
        self._backtest_is_pretrain = False  # 라이브와 동일 (롤백 활성)
        self._backtest_allow_training = True  # periodic training 활성

        start_cursor = 70
        total_candles = len(df)
        self._hist_cursor = start_cursor

        self.stats = TradingStats()
        self.trade_history = []
        self.current_balance = self.initial_balance
        self.current_position = None
        self.daily_pnl = 0.0

        # 4. 시뮬레이션 루프
        logger.info(f"🚀 백테스트 시작: {start_cursor} → {total_candles} (step={step})")

        cycle_count = 0
        progress_interval = max(1, (total_candles - start_cursor) // 20)

        while self._hist_cursor < total_candles:
            try:
                cycle_count += 1
                if not self._check_risk_limits():
                    logger.warning("⚠️ 리스크 한도 도달")
                    break
                self._execute_trading_cycle(cycle_count)
                if cycle_count % progress_interval == 0:
                    progress = (self._hist_cursor - start_cursor) / (total_candles - start_cursor) * 100
                    current_pnl = self.current_balance - self.initial_balance
                    logger.info(f"📊 {progress:.0f}% | PnL: {current_pnl:+.2f} | "
                                f"거래: {self.stats.total_trades}건")
                self._hist_cursor += step
            except Exception as e:
                logger.error(f"백테스트 루프 오류: {e}", exc_info=True)
                self._hist_cursor += step

        # 포지션 강제 청산
        if self.current_position:
            last_price = df.iloc[-1]['close']
            self._force_close_position(last_price)

        # 5. 결과 수집 및 출력
        metrics = self._collect_phase_metrics(self.trade_history, self.current_balance)

        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 Backtest 결과")
        logger.info("=" * 70)
        logger.info(f"  기간: {start_date} ~ {end_date} ({total_candles}캔들)")
        logger.info(f"  수익률: {metrics['return_pct']:+.2f}%  |  PnL: {metrics['pnl']:+.2f} USDT")
        logger.info(f"  거래: {metrics['total_trades']}건  |  "
                    f"승: {metrics['wins']}  패: {metrics['losses']}")
        logger.info(f"  승률: {metrics['win_rate']:.1f}%  |  "
                    f"PF: {metrics['profit_factor']:.2f}  |  MaxDD: {metrics['max_drawdown']:.1%}")
        logger.info("=" * 70)

        # 6. 정리
        self._cleanup_historical_mode()

        return metrics

    # ── Stage 4: Walk-Forward Optimization ──

    def run_walkforward(self, start_date: str, end_date: str,
                        train_months: int = 3, test_months: int = 1,
                        step: int = 1) -> dict:
        """Walk-Forward: SSL base 모델 공유, Fold마다 SFT → Backtest 반복.

        SSL base 모델은 run_ssl_pretrain()으로 사전에 생성해둬야 함.
        각 fold에서 SFT만 반복하고 backtest로 검증.
        모델 가중치는 fold간 이어받기 (누적 적응).

        Args:
            start_date: 전체 시작일 (YYYY-MM-DD)
            end_date: 전체 종료일 (YYYY-MM-DD)
            train_months: 학습 윈도우 (월)
            test_months: 테스트 윈도우 (월)
            step: 백테스트 캔들 스킵 간격
        """
        import math
        from datetime import datetime as dt


        logger.info("=" * 70)
        logger.info("🔄 [Stage 4] Walk-Forward Optimization 시작")
        logger.info(f"  심볼: {self.symbol}")
        logger.info(f"  전체 기간: {start_date} ~ {end_date}")
        logger.info(f"  윈도우: train={train_months}개월, test={test_months}개월")
        logger.info("=" * 70)

        # 0. SSL base 모델 로드
        logger.info("📥 SSL base 모델 로드 중...")
        self._load_ssl_models()

        # 1. 전체 데이터 다운로드
        logger.info("📥 전체 기간 데이터 다운로드 중...")
        df_full = self.historical_collector.collect_klines(
            symbol=self.symbol, interval='1m',
            start_date=start_date, end_date=end_date, save_to_csv=True
        )

        if df_full.empty or len(df_full) < 200:
            logger.error(f"데이터 부족: {len(df_full)}개 캔들")
            return {'status': 'failed', 'reason': 'insufficient_data'}

        if 'timestamp' in df_full.columns:
            df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
            df_full.set_index('timestamp', inplace=True)
        _KEEP = ['open', 'high', 'low', 'close', 'volume', 'trades', 'taker_buy_base']
        df_full = df_full[[c for c in _KEEP if c in df_full.columns]].copy()

        logger.info(f"✅ 데이터 준비 완료: {len(df_full)}개 캔들 "
                    f"({df_full.index[0]} ~ {df_full.index[-1]})")

        # 2. Fold 생성 (월 기반 슬라이딩 윈도우)
        try:
            from dateutil.relativedelta import relativedelta
        except ImportError:
            # dateutil 없으면 30일 기반 근사치 사용
            class relativedelta:
                def __init__(self, months=0):
                    self._days = months * 30
                def __radd__(self, other):
                    from datetime import timedelta
                    return other + timedelta(days=self._days)

        full_start = pd.Timestamp(start_date)
        full_end = pd.Timestamp(end_date)

        folds = []
        fold_start = full_start
        while True:
            train_end = fold_start + relativedelta(months=train_months)
            test_end = train_end + relativedelta(months=test_months)

            if test_end > full_end:
                # 마지막 fold: test 기간이 남은 데이터까지
                if train_end < full_end:
                    test_end = full_end
                else:
                    break

            folds.append({
                'train_start': fold_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
            })

            fold_start = fold_start + relativedelta(months=test_months)

            if fold_start >= full_end:
                break

        if not folds:
            logger.error("Fold를 생성할 수 없습니다 (기간이 너무 짧음)")
            return {'status': 'failed', 'reason': 'insufficient_period'}

        logger.info(f"📐 {len(folds)}개 fold 생성:")
        for i, f in enumerate(folds):
            logger.info(f"  Fold {i+1}: Train {f['train_start'].date()} ~ {f['train_end'].date()} | "
                        f"Test {f['test_start'].date()} ~ {f['test_end'].date()}")

        # 3. Fold 순회
        fold_results = []
        import os
        os.makedirs(os.path.dirname(self.model_save_path) if os.path.dirname(self.model_save_path) else "models", exist_ok=True)

        for fold_idx, fold in enumerate(folds):
            logger.info("")
            logger.info(f"{'='*60}")
            logger.info(f"🔄 Fold {fold_idx+1}/{len(folds)}")
            logger.info(f"  Train: {fold['train_start'].date()} ~ {fold['train_end'].date()}")
            logger.info(f"  Test:  {fold['test_start'].date()} ~ {fold['test_end'].date()}")
            logger.info(f"{'='*60}")

            # 3a. Trading state만 리셋 (모델 가중치는 이전 fold에서 이어받기)
            self._reset_for_fold()

            # 3b. Train/Test 슬라이싱
            train_df = df_full[fold['train_start']:fold['train_end']].copy()
            test_df = df_full[fold['test_start']:fold['test_end']].copy()

            if len(train_df) < 100 or len(test_df) < 50:
                logger.warning(f"  Fold {fold_idx+1}: 데이터 부족 "
                               f"(train={len(train_df)}, test={len(test_df)}) → 스킵")
                continue

            # ── 3c. SFT (Train 구간) ──
            logger.info(f"  🎯 SFT: {len(train_df)}캔들로 학습...")
            self._historical_mode = True
            self._historical_df = train_df
            self._backtest_is_pretrain = True
            self._hist_cursor = min(200, len(train_df))

            if self.use_ai_model:
                # 1m SFT
                _1m_analyzer = self._get_analyzer_for_tf('1m')
                if _1m_analyzer:
                    try:
                        from ai.market_analyzer import TrainingSample
                        from ai.sample_buffer import CANDLE_LOOK_AHEAD, triple_barrier_label_precompute
                        from ai.feature_builder import TFFeatureBuilder

                        look_ahead = CANDLE_LOOK_AHEAD['1m']
                        seq_len = _1m_analyzer.sequence_length
                        all_features = _1m_analyzer.prepare_features(train_df)
                        close_arr = train_df['close'].values
                        high_arr = train_df['high'].values
                        low_arr = train_df['low'].values
                        n_rows = len(all_features)
                        atr_arr = TFFeatureBuilder.compute_raw_atr(train_df).values

                        samples_added = 0
                        for i in range(seq_len, n_rows - look_ahead):
                            sequence = all_features[i - seq_len:i].copy()
                            entry_price = close_arr[i - 1]
                            atr_val = atr_arr[i]

                            direction, net_log_return, barrier_type, _timing = triple_barrier_label_precompute(
                                entry_price=entry_price, atr=atr_val,
                                high_arr=high_arr[i:i + look_ahead],
                                low_arr=low_arr[i:i + look_ahead],
                                close_arr=close_arr[i:i + look_ahead],
                            )

                            # 2-class: NEUTRAL → 학습 제외
                            if direction == 'NEUTRAL':
                                continue

                            exit_price = entry_price * math.exp(net_log_return)
                            net_pct = (exit_price / entry_price - 1) * 100 - self.trading_fee_rate * 2 * 100
                            net_log_return = net_log_return - 2 * self.trading_fee_rate

                            sample = TrainingSample(
                                features=sequence,
                                actual_direction=direction,
                                actual_price_change=net_pct,
                                timestamp=train_df.index[i],
                                sample_weight=max(0.3, _timing),
                                future_log_return=net_log_return
                            )
                            _1m_analyzer.training_buffer.append(sample)
                            samples_added += 1

                        if samples_added >= _1m_analyzer.min_samples_for_training:
                            result = _1m_analyzer.batch_train(epochs=5, is_pretrain=True)
                            logger.info(f"  1m SFT: {result.get('status')} "
                                        f"(acc={result.get('direction_accuracy', 0):.1%})")
                    except Exception as e:
                        logger.error(f"  1m SFT 에러: {e}")

                # MTF SFT
                if self.use_multi_timeframe and self.mtf_analyzer:
                    self._pretrain_mtf_with_history()

                # 10s Tick SFT
                if self.use_tick_model and self.tick_analyzer:
                    self._pretrain_tick_with_history_backtest(train_df)

            # ── 3d. Backtest (Test 구간) ──
            logger.info(f"  📊 Backtest: {len(test_df)}캔들로 검증...")
            self._historical_df = test_df
            self._hist_cursor = 70
            self._backtest_is_pretrain = False  # 라이브 유사
            self._backtest_allow_training = True
            self._hist_train_counter = 0
            self._hist_tick_train_counter = 0

            self.stats = TradingStats()
            self.trade_history = []
            self.current_balance = self.initial_balance
            self.current_position = None
            self.daily_pnl = 0.0

            cycle_count = 0
            while self._hist_cursor < len(test_df):
                try:
                    cycle_count += 1
                    if not self._check_risk_limits():
                        break
                    self._execute_trading_cycle(cycle_count)
                    self._hist_cursor += step
                except Exception as e:
                    logger.error(f"  Fold {fold_idx+1} BT 오류: {e}")
                    self._hist_cursor += step

            # 포지션 강제 청산
            if self.current_position:
                last_price = test_df.iloc[-1]['close']
                self._force_close_position(last_price)

            # 3e. Fold 결과 수집
            metrics = self._collect_phase_metrics(self.trade_history, self.current_balance)
            metrics['train_period'] = f"{fold['train_start'].date()} ~ {fold['train_end'].date()}"
            metrics['test_period'] = f"{fold['test_start'].date()} ~ {fold['test_end'].date()}"
            fold_results.append(metrics)

            logger.info(f"  Fold {fold_idx+1} 결과: "
                        f"Return={metrics['return_pct']:+.2f}%, "
                        f"WR={metrics['win_rate']:.1f}%, "
                        f"Trades={metrics['total_trades']}")

        # 4. 집계 결과 출력
        self._print_walkforward_aggregate(fold_results)

        # 5. CSV 저장
        try:
            eq_df = pd.DataFrame(fold_results)
            csv_path = f"backtesting/results/walkforward_{self.symbol}_{start_date}_{end_date}.csv"
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            eq_df.to_csv(csv_path, index=False)
            logger.info(f"📁 Walk-Forward 결과 저장: {csv_path}")
        except Exception as e:
            logger.warning(f"결과 저장 실패: {e}")

        # 6. 정리
        self._cleanup_historical_mode()

        return {'status': 'completed', 'folds': fold_results}

    # ── 통합 모델 파이프라인 ────────────────────────────────────

    def _collect_unified_data(self, df_1m: pd.DataFrame):
        """1m 캔들 → 15m/3m 리샘플. micro는 df_1m 그대로."""
        agg_map = {'open': 'first', 'high': 'max', 'low': 'min',
                   'close': 'last', 'volume': 'sum'}
        # trades, taker_buy_base가 있으면 합산
        if 'trades' in df_1m.columns:
            agg_map['trades'] = 'sum'
        if 'taker_buy_base' in df_1m.columns:
            agg_map['taker_buy_base'] = 'sum'

        df_15m = df_1m.resample('15min').agg(agg_map).dropna()
        df_3m = df_1m.resample('3min').agg(agg_map).dropna()

        return df_15m, df_3m, df_1m

    def _extract_unified_ssl_features(self, df_15m, df_3m, df_1m):
        """통합 모델용 SSL feature 시퀀스 추출."""
        import numpy as np
        features_dict = {'macro': [], 'mid': [], 'micro': []}
        ua = self.unified_analyzer

        macro_feat = ua.feature_builder.prepare_features(df_15m, '15m')
        mid_feat = ua.feature_builder.prepare_features(df_3m, '3m')
        micro_feat = ua.feature_builder.prepare_features(df_1m, '1m')

        # Sliding window
        for i in range(ua.macro_seq_len, len(macro_feat)):
            features_dict['macro'].append(macro_feat[i - ua.macro_seq_len:i])

        for i in range(ua.mid_seq_len, len(mid_feat)):
            features_dict['mid'].append(mid_feat[i - ua.mid_seq_len:i])

        for i in range(ua.micro_seq_len, len(micro_feat)):
            features_dict['micro'].append(micro_feat[i - ua.micro_seq_len:i])

        return features_dict

    def run_ssl_pretrain_unified(self, start_date: str, end_date: str,
                                 epochs: int = 10, mask_ratio: float = 0.15) -> dict:
        """통합 모델 SSL 사전학습.

        1m 데이터 다운로드 → 15m/3m 리샘플(+1m micro) → 인코더별 SSL → backbone SSL.
        결과물: models/unified_ssl_{symbol}.pt
        """
        if not self.unified_analyzer:
            logger.error("통합 모델이 활성화되지 않았습니다")
            return {'status': 'failed', 'reason': 'no_unified_model'}

        logger.info("=" * 70)
        logger.info("🧠 [통합 Stage 1] SSL Pre-training 시작")
        logger.info(f"  심볼: {self.symbol}, 기간: {start_date} ~ {end_date}")
        logger.info("=" * 70)

        # 1. 데이터 수집
        df = self.historical_collector.collect_klines(
            symbol=self.symbol, interval='1m',
            start_date=start_date, end_date=end_date, save_to_csv=True
        )
        if df.empty or len(df) < 500:
            logger.error(f"데이터 부족: {len(df)}개")
            return {'status': 'failed', 'reason': 'insufficient_data'}

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        _KEEP = ['open', 'high', 'low', 'close', 'volume', 'trades', 'taker_buy_base']
        df = df[[c for c in _KEEP if c in df.columns]].copy()
        logger.info(f"✅ 1m 데이터: {len(df)}개 캔들")

        # 2. 리샘플
        df_15m, df_3m, df_1m = self._collect_unified_data(df)
        logger.info(f"  15m: {len(df_15m)}, 3m: {len(df_3m)}, 1m(micro): {len(df_1m)}")

        # 3. SSL feature 추출
        features_dict = self._extract_unified_ssl_features(df_15m, df_3m, df_1m)
        for k, v in features_dict.items():
            logger.info(f"  {k}: {len(v)}개 시퀀스")

        # 4. SSL 학습
        result = self.unified_analyzer.ssl_pretrain(
            features_dict, epochs=epochs, mask_ratio=mask_ratio
        )

        # 5. 저장
        ssl_path = f"models/unified_ssl_{self.symbol}.pt"
        self.unified_analyzer.save_encoders(ssl_path)
        logger.info(f"✅ SSL 인코더 저장: {ssl_path}")

        return result

    def run_sft_unified(self, start_date: str, end_date: str) -> dict:
        """통합 모델 SFT 파인튜닝.

        SSL base 로드 → 인코더 해동 SFT → 인코더 동결 → MoE+Heads 학습.
        결과물: models/unified_ready_{symbol}.pt
        """
        if not self.unified_analyzer:
            return {'status': 'failed', 'reason': 'no_unified_model'}

        logger.info("=" * 70)
        logger.info("🎯 [통합 Stage 2] SFT Fine-tuning 시작")
        logger.info(f"  심볼: {self.symbol}, 기간: {start_date} ~ {end_date}")
        logger.info("=" * 70)

        # 1. SSL base 로드
        ssl_path = f"models/unified_ssl_{self.symbol}.pt"
        if self.unified_analyzer.load_encoders(ssl_path):
            logger.info(f"✅ SSL 인코더 로드: {ssl_path}")
        else:
            logger.warning("SSL 인코더 없음, 랜덤 초기화로 진행")

        # 2. 데이터 수집
        df = self.historical_collector.collect_klines(
            symbol=self.symbol, interval='1m',
            start_date=start_date, end_date=end_date, save_to_csv=True
        )
        if df.empty or len(df) < 200:
            return {'status': 'failed', 'reason': 'insufficient_data'}

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        _KEEP = ['open', 'high', 'low', 'close', 'volume', 'trades', 'taker_buy_base']
        df = df[[c for c in _KEEP if c in df.columns]].copy()

        # 3. 리샘플
        df_15m, df_3m, df_1m = self._collect_unified_data(df)

        # 4. Triple Barrier 라벨링 → 학습 샘플 생성 (timing_score 포함)
        import numpy as np
        from ai.market_analyzer import DIR_MAP
        from ai.sample_buffer import triple_barrier_label_precompute, CANDLE_LOOK_AHEAD
        from ai.feature_builder import TFFeatureBuilder
        ua = self.unified_analyzer

        macro_feat = ua.feature_builder.prepare_features(df_15m, '15m')
        mid_feat = ua.feature_builder.prepare_features(df_3m, '3m')
        micro_feat = ua.feature_builder.prepare_features(df_1m, '1m')

        # 3m 기준으로 라벨 생성 (중간 시간대) — Triple Barrier + timing_score
        look_ahead = CANDLE_LOOK_AHEAD['3m']  # 5캔들
        close_3m = df_3m['close'].values
        high_3m = df_3m['high'].values
        low_3m = df_3m['low'].values
        atr_series = TFFeatureBuilder.compute_raw_atr(df_3m)

        samples = []
        macro_step = max(1, len(macro_feat) // len(mid_feat))
        micro_step = max(1, len(micro_feat) // len(mid_feat))

        for i in range(max(ua.mid_seq_len, 10), len(mid_feat) - look_ahead):
            if i + look_ahead >= len(close_3m):
                break

            entry_price = close_3m[i]
            atr_val = float(atr_series.iloc[i]) if i < len(atr_series) else 0.0

            direction, net_lr, barrier_type, timing_score = triple_barrier_label_precompute(
                entry_price=entry_price,
                atr=atr_val,
                high_arr=high_3m[i + 1:i + 1 + look_ahead],
                low_arr=low_3m[i + 1:i + 1 + look_ahead],
                close_arr=close_3m[i + 1:i + 1 + look_ahead],
            )

            # 2-class: NEUTRAL → 학습 제외
            if direction == 'NEUTRAL':
                continue

            future_ret = (close_3m[i + look_ahead] - close_3m[i]) / close_3m[i] if close_3m[i] > 0 else 0

            # Align indices
            macro_idx = min(int(i * macro_step), len(macro_feat) - 1)
            micro_idx = min(int(i * micro_step), len(micro_feat) - 1)

            if (macro_idx < ua.macro_seq_len or
                    i < ua.mid_seq_len or
                    micro_idx < ua.micro_seq_len):
                continue

            sample = UnifiedTrainingSample(
                macro_features=macro_feat[macro_idx - ua.macro_seq_len:macro_idx],
                mid_features=mid_feat[i - ua.mid_seq_len:i],
                micro_features=micro_feat[micro_idx - ua.micro_seq_len:micro_idx],
                actual_direction=direction,
                actual_price_change=future_ret,
                timestamp=df_3m.index[i] if hasattr(df_3m.index[i], 'timestamp') else datetime.now(),
                sample_weight=max(0.3, timing_score),
                future_log_return=net_lr,
                actual_timing=timing_score,
            )
            ua.add_training_sample(sample)
            samples.append(sample)

        logger.info(f"  학습 샘플: {len(samples)}개")
        dir_counts = {}
        for s in samples:
            dir_counts[s.actual_direction] = dir_counts.get(s.actual_direction, 0) + 1
        logger.info(f"  방향 분포: {dir_counts}")

        if len(samples) < 20:
            return {'status': 'failed', 'reason': 'insufficient_samples'}

        # 5. SFT Phase 1: 인코더 해동 + 전체 학습 (10 epoch)
        logger.info("━" * 50)
        logger.info("  [Phase 1] 인코더 해동 SFT (10 epoch)")
        result1 = ua.batch_train(epochs=10, is_pretrain=True, freeze_encoders=False)
        logger.info(f"  Phase 1 결과: {result1.get('status')}, acc={result1.get('dir_accuracy', 0):.1%}")

        # 6. SFT Phase 2: 인코더 동결 + MoE+Heads 학습 (15 epoch)
        logger.info("━" * 50)
        logger.info("  [Phase 2] 인코더 동결, MoE+Heads 학습 (15 epoch)")
        result2 = ua.batch_train(epochs=15, is_pretrain=True, freeze_encoders=True)
        logger.info(f"  Phase 2 결과: {result2.get('status')}, acc={result2.get('dir_accuracy', 0):.1%}")

        # 7. 저장
        ready_path = f"models/unified_ready_{self.symbol}.pt"
        ua.save_model(ready_path)
        logger.info(f"✅ 통합 모델 저장: {ready_path}")

        # LoRA 상태
        lora_norm = sum(p.data.norm().item() for p in ua.model.online_lora.parameters())
        logger.info(f"  LoRA norm: {lora_norm:.4f}")

        return {'status': 'completed', 'phase1': result1, 'phase2': result2}

    def run_backtest_unified(
        self, start_date: str, end_date: str, step: int = 1,
        allow_training: bool = True, label: str = "Stage 3"
    ) -> dict:
        """통합 모델 백테스트.

        unified_ready 모델 로드 → 시뮬레이션.

        Args:
            allow_training: True=LoRA 온라인 학습 ON, False=추론만 (베이스 모델 평가)
            label: 로그 표시용 단계 이름
        """
        if not self.unified_analyzer:
            return {'status': 'failed', 'reason': 'no_unified_model'}

        _mode_str = "LoRA 온라인 학습" if allow_training else "추론 전용 (베이스 모델 평가)"
        logger.info("=" * 70)
        logger.info(f"📊 [통합 {label}] Backtest 시작 — {_mode_str}")
        logger.info(f"  심볼: {self.symbol}, 기간: {start_date} ~ {end_date}")
        logger.info("=" * 70)

        # 1. 모델 로드
        ready_path = f"models/unified_ready_{self.symbol}.pt"
        if self.unified_analyzer.load_model(ready_path):
            logger.info(f"✅ 모델 로드: {ready_path}")
        else:
            logger.warning("trading_ready 모델 없음, 현재 모델로 진행")

        # 2. 데이터 수집
        df = self.historical_collector.collect_klines(
            symbol=self.symbol, interval='1m',
            start_date=start_date, end_date=end_date, save_to_csv=True
        )
        if df.empty or len(df) < 200:
            return {'status': 'failed', 'reason': 'insufficient_data'}

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        _KEEP = ['open', 'high', 'low', 'close', 'volume', 'trades', 'taker_buy_base']
        df = df[[c for c in _KEEP if c in df.columns]].copy()

        # 3. Historical mode — 상태 초기화 (재실행 대비)
        self._historical_mode = True
        self._historical_df = df
        self._backtest_is_pretrain = False
        self._backtest_allow_training = allow_training
        _min_cursor = 1000
        self._hist_cursor = min(_min_cursor, len(df) - 200)

        # 거래 상태 초기화 (이전 백테스트 잔여 포지션 방지)
        self.trade_history = []
        self.current_position = None
        self.current_balance = self.initial_balance
        self.stats = type(self.stats)()  # TradingStats 리셋
        if hasattr(self.unified_analyzer, 'shadow_validator'):
            self.unified_analyzer.shadow_validator.force_shadow(f"백테스트 시작 ({label})")

        # 4. 시뮬레이션
        logger.info(f"📈 시뮬레이션 시작: {len(df)}개 캔들, step={step}")
        cycle = 0
        while self._hist_cursor < len(df):
            try:
                self._execute_trading_cycle(cycle)
            except Exception as e:
                logger.debug(f"Cycle {cycle} 오류: {e}")
            self._hist_cursor += step
            cycle += 1

        # 5. 결과
        metrics = self._collect_phase_metrics(self.trade_history, self.current_balance)

        # Shadow Validator 최종 통계
        sv_status = self.unified_analyzer.shadow_validator.get_status()
        metrics['shadow_accuracy'] = sv_status['recent_accuracy']
        metrics['shadow_pf'] = sv_status['recent_profit_factor']
        metrics['shadow_resolved'] = sv_status['total_resolved']

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"📊 통합 모델 백테스트 결과 [{label}] — {_mode_str}")
        logger.info("=" * 70)
        logger.info(f"  총 거래: {metrics.get('total_trades', 0)}")
        logger.info(f"  순수익: {metrics.get('total_pnl', 0):.2f} USDT")
        logger.info(f"  수익률: {metrics.get('return_pct', 0):.2f}%")
        logger.info(f"  승률: {metrics.get('win_rate', 0):.1f}%")
        logger.info(f"  PF: {metrics.get('profit_factor', 0):.2f}")
        logger.info(f"  최대 DD: {metrics.get('max_drawdown', 0) * 100:.2f}%")
        logger.info("-" * 70)
        logger.info("🎯 핵심 전략 지표")
        logger.info(f"  ① TP Rate: {metrics.get('tp_rate', 0):.1f}%")
        logger.info(f"  ② R Multiple: {metrics.get('r_multiple', 0):.2f}")
        logger.info(f"  ③ Net EV/trade: {metrics.get('net_ev', 0):+.4f} USDT")
        logger.info("-" * 70)
        logger.info(f"  Shadow 정확도: {sv_status['recent_accuracy']:.1%}")
        logger.info(f"  Shadow PF: {sv_status['recent_profit_factor']:.2f}")

        # LoRA 상태
        _l_norm = sum(p.data.norm().item() for p in self.unified_analyzer.model.online_lora.parameters())
        logger.info(f"  LoRA norm: {_l_norm:.4f}")

        self._cleanup_historical_mode()
        return metrics

    def run_unified_pipeline(self, start_date: str, end_date: str,
                             ssl_days: int = 365, sft_days: int = 180,
                             bt_days: int = 30) -> dict:
        """통합 풀 파이프라인: SSL → SFT → Backtest."""
        from datetime import datetime, timedelta
        now = datetime.now()

        # 데이터 분리: SSL/SFT 끝 = 백테스트 시작 (누출 방지)
        # [SSL ──────][SFT ─────][BT ──]
        #             ^sft_start ^bt_start ^now
        bt_end = (now - timedelta(days=1)).strftime('%Y-%m-%d')  # 전날까지만 (당일 불완전 데이터 방지 + 캐시 안정)
        bt_start = (now - timedelta(days=bt_days)).strftime('%Y-%m-%d')
        sft_end = bt_start
        sft_start = (now - timedelta(days=bt_days + sft_days)).strftime('%Y-%m-%d')
        ssl_end = sft_end
        ssl_start = (now - timedelta(days=bt_days + ssl_days)).strftime('%Y-%m-%d')

        logger.info("=" * 70)
        logger.info(f"🚀 통합 파이프라인: {self.symbol}")
        logger.info(f"  [1] SSL: {ssl_start} ~ {ssl_end}")
        logger.info(f"  [2] SFT: {sft_start} ~ {sft_end}")
        logger.info(f"  [3] BT:  {bt_start} ~ {bt_end}")
        logger.info("=" * 70)

        # Stage 1: SSL
        logger.info("━" * 70)
        logger.info("  [1/3] SSL Pre-training...")
        ssl_result = self.run_ssl_pretrain_unified(ssl_start, ssl_end, epochs=10)

        # Stage 2: SFT
        logger.info("━" * 70)
        logger.info("  [2/3] SFT Fine-tuning...")
        sft_result = self.run_sft_unified(sft_start, sft_end)

        # Stage 2.5: 베이스 모델 평가 (학습 OFF — 순수 SFT 성능 측정)
        logger.info("━" * 70)
        logger.info("  [2.5/4] 베이스 모델 평가 (LoRA 학습 OFF)...")
        base_result = self.run_backtest_unified(
            bt_start, bt_end, step=1,
            allow_training=False, label="Stage 2.5 베이스"
        )

        # Stage 3: Backtest (LoRA 온라인 학습 ON)
        logger.info("━" * 70)
        logger.info("  [3/4] Backtest (LoRA 학습 ON)...")
        bt_result = self.run_backtest_unified(
            bt_start, bt_end, step=1,
            allow_training=True, label="Stage 3 LoRA"
        )

        # Stage 4: 비교 리포트
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 베이스 vs LoRA 비교")
        logger.info("=" * 70)
        for key in ['total_trades', 'return_pct', 'win_rate', 'profit_factor',
                     'max_drawdown', 'shadow_accuracy', 'shadow_pf']:
            b = base_result.get(key, 0)
            l = bt_result.get(key, 0)
            if key == 'max_drawdown':
                logger.info(f"  {'max_drawdown_pct':25s}: 베이스={b*100:.1f}%  LoRA={l*100:.1f}%")
            else:
                _fmt = '.1f' if 'rate' in key or 'pct' in key or 'accuracy' in key else '.2f'
                logger.info(f"  {key:25s}: 베이스={b:{_fmt}}  LoRA={l:{_fmt}}")

        logger.info("")
        logger.info("✅ 통합 파이프라인 완료!")
        return {
            'ssl': ssl_result, 'sft': sft_result,
            'base_eval': base_result, 'backtest': bt_result,
        }

    # ── 통합 Walk-Forward 검증 ──────────────────────────────────

    def run_unified_walkforward(self, start_date: str, end_date: str,
                                 train_months: int = 9, test_months: int = 3,
                                 step: int = 1) -> dict:
        """통합 모델 Walk-Forward 검증.

        Train(9개월) → Test(3개월), 겹침 0으로 롤링.
        각 fold마다: 모델 초기화 → SSL → SFT → Backtest(추론 전용).

        Args:
            start_date: 전체 시작일 (YYYY-MM-DD)
            end_date: 전체 종료일 (YYYY-MM-DD)
            train_months: 학습 윈도우 (기본 9개월)
            test_months: 테스트 윈도우 (기본 3개월)
            step: 백테스트 캔들 스킵 간격
        """
        import math
        import numpy as np
        import os
        from datetime import datetime, timedelta
        from ai.sample_buffer import triple_barrier_label_precompute, CANDLE_LOOK_AHEAD
        from ai.feature_builder import TFFeatureBuilder
        from ai.unified_analyzer import UnifiedTrainingSample

        try:
            from dateutil.relativedelta import relativedelta
        except ImportError:
            class relativedelta:
                def __init__(self, months=0):
                    self._days = months * 30
                def __radd__(self, other):
                    return other + timedelta(days=self._days)

        if not self.unified_analyzer:
            logger.error("통합 모델이 활성화되지 않았습니다")
            return {'status': 'failed', 'reason': 'no_unified_model'}

        logger.info("=" * 70)
        logger.info("🔄 [통합 Walk-Forward] 시작")
        logger.info(f"  심볼: {self.symbol}")
        logger.info(f"  전체 기간: {start_date} ~ {end_date}")
        logger.info(f"  윈도우: Train={train_months}개월, Test={test_months}개월")
        logger.info(f"  겹침: 0 (Train과 Test 완전 분리)")
        logger.info("=" * 70)

        # ── 1. 전체 데이터 다운로드 (한 번만) ──
        logger.info("📥 전체 기간 데이터 다운로드 중...")
        df_full = self.historical_collector.collect_klines(
            symbol=self.symbol, interval='1m',
            start_date=start_date, end_date=end_date, save_to_csv=True
        )

        if df_full.empty or len(df_full) < 500:
            logger.error(f"데이터 부족: {len(df_full)}개 캔들")
            return {'status': 'failed', 'reason': 'insufficient_data'}

        if 'timestamp' in df_full.columns:
            df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
            df_full.set_index('timestamp', inplace=True)
        _KEEP = ['open', 'high', 'low', 'close', 'volume', 'trades', 'taker_buy_base']
        df_full = df_full[[c for c in _KEEP if c in df_full.columns]].copy()

        logger.info(f"✅ 데이터: {len(df_full)}개 캔들 "
                    f"({df_full.index[0]} ~ {df_full.index[-1]})")

        # ── 2. Fold 생성 (test_months 단위로 롤링, 겹침 0) ──
        full_start = pd.Timestamp(start_date)
        full_end = pd.Timestamp(end_date)

        folds = []
        fold_start = full_start
        while True:
            train_end = fold_start + relativedelta(months=train_months)
            test_start = train_end  # 겹침 0
            test_end = test_start + relativedelta(months=test_months)

            if test_end > full_end:
                if test_start < full_end:
                    test_end = full_end
                else:
                    break

            folds.append({
                'train_start': fold_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
            })

            # 롤링: test_months만큼 전진
            fold_start = fold_start + relativedelta(months=test_months)
            # 다음 fold가 전체 범위를 넘으면 중단
            next_train_end = fold_start + relativedelta(months=train_months)
            if next_train_end >= full_end:
                break

        if not folds:
            logger.error("Fold 생성 불가 (최소 train+test 기간 필요)")
            return {'status': 'failed', 'reason': 'insufficient_period'}

        logger.info(f"\n📐 {len(folds)}개 fold 생성:")
        for i, f in enumerate(folds):
            logger.info(
                f"  Fold {i+1}: Train {f['train_start'].date()} ~ {f['train_end'].date()} | "
                f"Test {f['test_start'].date()} ~ {f['test_end'].date()}"
            )

        # ── 3. Fold 순회 ──
        ua = self.unified_analyzer
        fold_results = []

        for fold_idx, fold in enumerate(folds):
            logger.info("")
            logger.info("━" * 70)
            logger.info(f"🔄 Fold {fold_idx+1}/{len(folds)}")
            logger.info(f"  Train: {fold['train_start'].date()} ~ {fold['train_end'].date()}")
            logger.info(f"  Test:  {fold['test_start'].date()} ~ {fold['test_end'].date()}")
            logger.info("━" * 70)

            # 3a. 데이터 슬라이싱
            train_df = df_full[fold['train_start']:fold['train_end']].copy()
            test_df = df_full[fold['test_start']:fold['test_end']].copy()

            if len(train_df) < 1000 or len(test_df) < 200:
                logger.warning(
                    f"  Fold {fold_idx+1}: 데이터 부족 "
                    f"(train={len(train_df)}, test={len(test_df)}) → 스킵"
                )
                continue

            logger.info(f"  데이터: Train={len(train_df)}캔들, Test={len(test_df)}캔들")

            # 3b. 모델 초기화 (매 fold 독립 — 이전 fold 영향 제거)
            from ai.market_analyzer import UnifiedTradingModel
            ua.model = UnifiedTradingModel(
                macro_features=ua.macro_features,
                mid_features=ua.mid_features,
                micro_features=ua.micro_features,
            ).to(ua.device)
            ua.training_buffer.clear()
            ua.replay_buffer.clear()
            ua._training_count = 0
            ua._best_val_loss = float('inf')
            ua._consecutive_degrades = 0
            ua._checkpoint_state = None
            ua.shadow_validator.force_shadow(f"WF Fold {fold_idx+1} 초기화")

            # 3c. SSL on Train
            logger.info(f"  🧠 [SSL] {len(train_df)}캔들로 사전학습...")
            df_15m_tr, df_3m_tr, df_1m_tr = self._collect_unified_data(train_df)
            features_dict = self._extract_unified_ssl_features(
                df_15m_tr, df_3m_tr, df_1m_tr
            )
            ssl_result = ua.ssl_pretrain(features_dict, epochs=10, mask_ratio=0.15)
            logger.info(f"  SSL 완료: {ssl_result.get('status', '?')}")

            # 3d. SFT on Train (Triple Barrier 라벨링 → Phase 1 + Phase 2)
            logger.info(f"  🎯 [SFT] 라벨링 + 파인튜닝...")
            macro_feat = ua.feature_builder.prepare_features(df_15m_tr, '15m')
            mid_feat = ua.feature_builder.prepare_features(df_3m_tr, '3m')
            micro_feat = ua.feature_builder.prepare_features(df_1m_tr, '1m')

            look_ahead = CANDLE_LOOK_AHEAD['3m']
            close_3m = df_3m_tr['close'].values
            high_3m = df_3m_tr['high'].values
            low_3m = df_3m_tr['low'].values
            atr_series = TFFeatureBuilder.compute_raw_atr(df_3m_tr)

            samples_added = 0
            dir_counts = {'UP': 0, 'DOWN': 0, 'NEUTRAL': 0}
            macro_step = max(1, len(macro_feat) // len(mid_feat))
            micro_step = max(1, len(micro_feat) // len(mid_feat))

            for i in range(max(ua.mid_seq_len, 10), len(mid_feat) - look_ahead):
                if i + look_ahead >= len(close_3m):
                    break

                entry_price = close_3m[i]
                atr_val = (
                    float(atr_series.iloc[i]) if i < len(atr_series) else 0.0
                )

                direction, net_lr, barrier_type, timing_score = (
                    triple_barrier_label_precompute(
                        entry_price=entry_price, atr=atr_val,
                        high_arr=high_3m[i + 1:i + 1 + look_ahead],
                        low_arr=low_3m[i + 1:i + 1 + look_ahead],
                        close_arr=close_3m[i + 1:i + 1 + look_ahead],
                    )
                )

                # 2-class: NEUTRAL 샘플 제외 (TIME/CONFLICT/cost미달)
                if direction == 'NEUTRAL':
                    continue

                future_ret = (
                    (close_3m[i + look_ahead] - close_3m[i]) / close_3m[i]
                    if close_3m[i] > 0 else 0
                )

                macro_idx = min(int(i * macro_step), len(macro_feat) - 1)
                micro_idx = min(int(i * micro_step), len(micro_feat) - 1)

                if (macro_idx < ua.macro_seq_len or
                        i < ua.mid_seq_len or
                        micro_idx < ua.micro_seq_len):
                    continue

                sample = UnifiedTrainingSample(
                    macro_features=macro_feat[
                        macro_idx - ua.macro_seq_len:macro_idx
                    ],
                    mid_features=mid_feat[i - ua.mid_seq_len:i],
                    micro_features=micro_feat[
                        micro_idx - ua.micro_seq_len:micro_idx
                    ],
                    actual_direction=direction,
                    actual_price_change=future_ret,
                    timestamp=(
                        df_3m_tr.index[i]
                        if hasattr(df_3m_tr.index[i], 'timestamp')
                        else datetime.now()
                    ),
                    sample_weight=max(0.3, timing_score),
                    future_log_return=net_lr,
                    actual_timing=timing_score,
                )
                ua.add_training_sample(sample)
                dir_counts[direction] = dir_counts.get(direction, 0) + 1
                samples_added += 1

            logger.info(f"  SFT 샘플: {samples_added}개 | 분포: {dir_counts}")

            if samples_added < 20:
                logger.warning(f"  Fold {fold_idx+1}: SFT 샘플 부족 → 스킵")
                continue

            # Phase 1: 인코더 해동 SFT
            result1 = ua.batch_train(
                epochs=5, is_pretrain=True, freeze_encoders=False
            )
            logger.info(
                f"  Phase 1: acc={result1.get('dir_accuracy', 0):.1%}"
            )

            # Phase 2: 인코더 동결 + MoE/Heads
            result2 = ua.batch_train(
                epochs=10, is_pretrain=True, freeze_encoders=True
            )
            logger.info(
                f"  Phase 2: acc={result2.get('dir_accuracy', 0):.1%}"
            )

            # LoRA 상태
            _lora_norm = sum(p.data.norm().item() for p in ua.model.online_lora.parameters())
            logger.info(f"  LoRA norm: {_lora_norm:.4f}")

            # 3e. Backtest on Test (추론 전용 — 학습 OFF)
            logger.info(
                f"  📊 [Backtest] {len(test_df)}캔들 검증 (학습 OFF)..."
            )
            self._historical_mode = True
            self._historical_df = test_df
            self._backtest_is_pretrain = False
            self._backtest_allow_training = False  # 학습 OFF — 순수 OOS
            _min_cursor = min(1000, len(test_df) - 200)
            self._hist_cursor = max(200, _min_cursor)

            self.trade_history = []
            self.current_position = None
            self.current_balance = self.initial_balance
            self.stats = type(self.stats)()
            self.daily_pnl = 0.0
            ua.shadow_validator.force_shadow(f"WF Fold {fold_idx+1} BT")

            cycle = 0
            while self._hist_cursor < len(test_df):
                try:
                    self._execute_trading_cycle(cycle)
                except Exception as e:
                    logger.debug(f"  Fold {fold_idx+1} BT 오류: {e}")
                self._hist_cursor += step
                cycle += 1

            if self.current_position:
                last_price = test_df.iloc[-1]['close']
                self._force_close_position(last_price)

            # 3f. Fold 결과 수집
            metrics = self._collect_phase_metrics(
                self.trade_history, self.current_balance
            )
            metrics['train_period'] = (
                f"{fold['train_start'].date()} ~ {fold['train_end'].date()}"
            )
            metrics['test_period'] = (
                f"{fold['test_start'].date()} ~ {fold['test_end'].date()}"
            )

            sv_status = ua.shadow_validator.get_status()
            metrics['shadow_accuracy'] = sv_status['recent_accuracy']
            metrics['shadow_pf'] = sv_status['recent_profit_factor']

            _lora_n = sum(p.data.norm().item() for p in ua.model.online_lora.parameters())
            metrics['lora_norm'] = round(_lora_n, 4)

            fold_results.append(metrics)

            logger.info(
                f"  Fold {fold_idx+1} 결과: "
                f"Return={metrics['return_pct']:+.2f}%, "
                f"WR={metrics['win_rate']:.1f}%, "
                f"PF={metrics['profit_factor']:.2f}, "
                f"Trades={metrics['total_trades']}, "
                f"Shadow={metrics['shadow_accuracy']:.1%}"
            )

        # ── 4. 집계 출력 ──
        self._print_walkforward_aggregate(fold_results)

        # ── 5. CSV 저장 ──
        try:
            eq_df = pd.DataFrame(fold_results)
            csv_path = (
                f"backtesting/results/"
                f"unified_wf_{self.symbol}_{start_date}_{end_date}.csv"
            )
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            eq_df.to_csv(csv_path, index=False)
            logger.info(f"📁 결과 저장: {csv_path}")
        except Exception as e:
            logger.warning(f"결과 저장 실패: {e}")

        self._cleanup_historical_mode()

        logger.info("")
        logger.info("✅ 통합 Walk-Forward 완료!")
        return {'status': 'completed', 'folds': fold_results}

    def start(self):
        """봇 시작"""
        if self.state == BotState.RUNNING:
            logger.warning("봇이 이미 실행 중입니다")
            return

        if not self.initialize():
            logger.error("봇 초기화 실패로 시작 불가")
            return

        self.state = BotState.RUNNING
        self._stop_event.clear()

        self._trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self._trading_thread.start()

        logger.info("🚀 AI 자동 거래 봇 시작!")

    def stop(self):
        """봇 중지"""
        logger.info("봇 중지 요청...")

        self._stop_event.set()
        self.state = BotState.STOPPED

        # 스레드 종료 대기
        if self._trading_thread and self._trading_thread.is_alive():
            self._trading_thread.join(timeout=10)

        # WebSocket 종료
        if self.data_manager:
            self.data_manager.stop()
        if self.user_stream:
            self.user_stream.stop()

        # 열린 포지션 경고
        with self._lock:
            if self.current_position:
                logger.warning(f"⚠️ 열린 포지션 있음: {self.current_position}")

        self._print_final_report()
        logger.info("봇 중지 완료")

    def pause(self):
        """봇 일시 정지"""
        with self._lock:
            if self.state == BotState.RUNNING:
                self.state = BotState.PAUSED
                logger.info("⏸️ 봇 일시 정지")

    def resume(self):
        """봇 재개"""
        with self._lock:
            if self.state == BotState.PAUSED:
                self.state = BotState.RUNNING
                logger.info("▶️ 봇 재개")

    def _trading_loop(self):
        """메인 거래 루프"""
        logger.info("거래 루프 시작")

        cycle_count = 0

        while not self._stop_event.is_set():
            try:
                if self.state != BotState.RUNNING:
                    time.sleep(1)
                    continue

                cycle_count += 1

                # 일일 리셋 체크
                self._check_daily_reset()

                # WebSocket 데이터 신선도 확인 + 재연결
                if self.data_manager:
                    self.data_manager.check_and_reconnect(stale_threshold_seconds=60)

                # 리스크 체크
                if not self._check_risk_limits():
                    time.sleep(60)
                    continue

                # 거래 사이클 실행
                self._execute_trading_cycle(cycle_count)

                # 대시보드용 상태 기록
                self.state_writer.periodic_write(self)

                # 다음 사이클 대기 (캔들 간격에 맞춤)
                wait_time = self._calculate_wait_time()
                time.sleep(wait_time)

            except Exception as e:
                logger.opt(exception=True).error(f"거래 루프 오류: {e}")
                time.sleep(10)

        logger.info("거래 루프 종료")

    def _execute_trading_cycle(self, cycle_count: int):
        """1회 거래 사이클 실행"""
        import time as _tmod
        _cycle_ts = _tmod.time()
        _cycle_id = f"C{cycle_count}-{int(_cycle_ts) % 100000}"
        # 이전 사이클의 cascade 데이터를 보존 (대시보드가 빈 데이터 표시 방지)
        _prev_cascade = {}
        for _keep_key in ('cascade_dir', 'cascade_conf', 'cascade_fire',
                          'fee_ev_ratio', 'final_score', 'hold_threshold',
                          'regime', 'regime_conf', 'action', 'execution_mode',
                          'vol_factor', 'vol_z'):
            if _keep_key in self._cycle_ctx:
                _prev_cascade[_keep_key] = self._cycle_ctx[_keep_key]
        self._cycle_ctx = {'id': _cycle_id, 'cycle': cycle_count, 'ts': _cycle_ts}
        self._cycle_ctx.update(_prev_cascade)

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"📊 거래 사이클 #{cycle_count} [{_cycle_id}]")
        logger.info("=" * 70)

        # 1. 시장 데이터 수집
        df = self._get_market_data()
        if df is None or len(df) < 100:
            logger.warning("시장 데이터 부족")
            return

        current_price = df['close'].iloc[-1]
        logger.info(f"현재 가격: {current_price:.2f} USDT")

        # 시뮬레이션 시간 업데이트 (backtest: 캔들 타임스탬프, live: wall-clock)
        import time as _time
        if self._historical_mode:
            self._sim_time = df.index[-1].timestamp()  # epoch seconds
        else:
            self._sim_time = _time.time()

        # 🔹 1.1 OBI 업데이트 (매 사이클: EMA 누적)
        if self._historical_mode:
            self._update_pseudo_obi(df)
        else:
            self._update_obi_data()

        # ── 캔들 변경 감지 (라이브 추론 중복 방지) ──
        _candle_ts = float(df.index[-1].timestamp())
        _is_new_candle = self._historical_mode or (_candle_ts != self._last_inference_candle_ts)
        if _is_new_candle:
            self._last_inference_candle_ts = _candle_ts

        # 1.5. AI 학습 샘플 수집 (PriceLog: 매 사이클, SnapshotEntry: 캔들당 1회)
        self._snapshot_to_ring_buffers(df, current_price)
        if self._historical_mode:
            self._populate_backtest_price_log(df)
        self._label_matured_samples()

        # 1.9. Shadow Validator 가격 업데이트 (가상 거래 해소)
        if self.unified_analyzer:
            _sv_sim = self._sim_time if self._historical_mode else None
            self.unified_analyzer.shadow_validator.update_prices(current_price, sim_time=_sv_sim)

        if not _is_new_candle:
            # ── 같은 캔들: SL/TP 안전망(Layer 2.5)만, 추론 스킵 ──
            if self.current_position:
                self._check_sl_tp(current_price, df=df)
            self._execute_periodic_training()
            if not self._historical_mode:
                self._price_ring.append((_tmod.time(), current_price))
                self._check_missed_opportunity(current_price, _tmod.time())
            return

        # ═══ 새 캔들: 전체 분석 파이프라인 ═══

        # 🔹 1.2 시장 상태(Regime) 감지
        try:
            self._current_regime = self.regime_detector.detect_regime(df)
            regime = self._current_regime.regime
            regime_conf = self._current_regime.confidence
            logger.info(f"  🌐 시장 상태: {regime.value} (신뢰도: {regime_conf:.0%})")
        except Exception as e:
            logger.debug(f"Regime 감지 실패: {e}")
            self._current_regime = None

        # 🔹 1.3 변동성 soft factor (Z-score → sigmoid)
        import math as _math
        _, current_vol = self.position_manager.check_volatility_filter(df)
        _close = df['close']
        _log_ret = (_close / _close.shift(1)).apply(_math.log)
        _vol_series = _log_ret.rolling(10).std().dropna()
        if len(_vol_series) >= 30:
            _vol_mean = _vol_series.rolling(60).mean().iloc[-1]
            _vol_std = _vol_series.rolling(60).std().iloc[-1]
            if _vol_std > 0 and _vol_mean == _vol_mean:  # NaN check
                _vol_z = (current_vol - _vol_mean) / _vol_std
            else:
                _vol_z = 0.0
        else:
            _vol_z = 0.0
        _raw_sig = 1.0 / (1.0 + _math.exp(-max(-5, min(5, _vol_z))))
        _vol_factor = 0.5 + 0.5 * _raw_sig  # [0.5, 1.0]
        self._cycle_ctx['vol_z'] = float(round(_vol_z, 3))
        self._cycle_ctx['vol_factor'] = float(round(_vol_factor, 3))
        logger.info(f"  📉 변동성: vol={current_vol:.6f}, z={_vol_z:+.2f}, factor={_vol_factor:.2f}")

        # 2. AI 레버리지 분석
        leverage_rec = self._analyze_leverage(df)

        # 2.5. 추론 시작 시점 기록 (Freshness Gate용)
        import time as _time_mod
        self._inference_start_price = current_price
        self._inference_start_ts = _time_mod.time()

        # 3. AI 포지션 분석
        position_decision = self._analyze_position(df)

        # 3.5 ATR 저장 (트레일링 스톱 동적 계산용)
        if 'atr' in df.columns and len(df) > 0:
            self._current_atr = df['atr'].iloc[-1]

        # 4. 포지션 관리
        self._manage_position(position_decision, leverage_rec, current_price, df)

        # 5. 주기적 AI 학습 실행
        self._execute_periodic_training()


        # 6. 상태 출력 — 히스토리컬에서는 100캔들마다만
        if not self._historical_mode or self._hist_cursor % 100 == 0:
            self._print_status()

        # 7. 사이클 구조화 요약 로그 (라이브만, backtest는 너무 많음)
        if not self._historical_mode and self._cycle_ctx.get('action'):
            import json as _json
            _ctx = self._cycle_ctx
            _summary = {
                'id': _ctx.get('id'),
                'price': float(current_price),
                'regime': _ctx.get('regime'),
                'cascade_dir': _ctx.get('cascade_dir'),
                'cascade_conf': _ctx.get('cascade_conf'),
                'cascade_fire': _ctx.get('cascade_fire'),
                'fee_ev': _ctx.get('fee_ev_ratio'),
                'final': _ctx.get('final_score'),
                'action': _ctx.get('action'),
                'side': _ctx.get('side'),
                'conf': _ctx.get('confidence'),
                'size': _ctx.get('size_pct'),
                'exec_mode': _ctx.get('execution_mode'),
                'chased': bool(_ctx['was_chased']) if _ctx.get('was_chased') is not None else None,
            }
            logger.info(f"[CYCLE_SUMMARY] {_json.dumps(_summary, ensure_ascii=False)}")

            # 텔레그램 사이클 요약 (throttle 적용)
            self.telegram.notify_cycle(_ctx, current_price)

        # 8. 놓친 기회 스캐너 (라이브만)
        if not self._historical_mode:
            import time as _t_missed
            self._price_ring.append((_t_missed.time(), current_price))
            self._check_missed_opportunity(current_price, _t_missed.time())

    # ═══════════════════════════════════════════════════════════════
    # 놓친 기회 스캐너 (Missed Opportunity Scanner)
    # ═══════════════════════════════════════════════════════════════

    def _check_missed_opportunity(self, current_price: float, now: float):
        """과거 5분 동안 피크 변동폭(high-low) 0.3% 이상인데 포지션이 없었으면 기록

        매 사이클마다 호출. 최소 60초 간격으로 실행.
        """
        # 60초 쿨다운
        if now - self._last_missed_check < 60.0:
            return
        self._last_missed_check = now

        # 포지션이 있으면 놓친 게 아님
        if self.current_position:
            return

        # 5분 구간 내 가격 수집
        cutoff = now - 300.0  # 5분
        window_prices = [px for ts, px in self._price_ring if ts >= cutoff]

        if len(window_prices) < 2:
            return

        # 피크 대조: 구간 내 최고/최저
        peak_high = max(window_prices)
        peak_low = min(window_prices)

        if peak_low <= 0:
            return

        # 변동폭 = (최고 - 최저) / 최저
        swing_pct = (peak_high - peak_low) / peak_low

        if swing_pct < 0.003:  # 0.3% 미만이면 무시
            return

        # 방향: 최고가가 최저가보다 나중이면 상승, 아니면 하락
        high_ts = max((ts for ts, px in self._price_ring if ts >= cutoff and px == peak_high), default=0)
        low_ts = max((ts for ts, px in self._price_ring if ts >= cutoff and px == peak_low), default=0)
        direction = "상승" if high_ts > low_ts else "하락"

        # 당시 AI 판단 사유 수집
        ctx = self._cycle_ctx
        block_reason = self._identify_block_reason(ctx)

        missed = {
            'time': datetime.now().isoformat(),
            'epoch': now,
            'direction': direction,
            'swing_pct': round(swing_pct * 100, 3),  # % (피크 변동폭)
            'peak_high': float(round(peak_high, 2)),
            'peak_low': float(round(peak_low, 2)),
            'cascade_conf': ctx.get('cascade_conf', 0),
            'cascade_dir': ctx.get('cascade_dir'),
            'fee_ev_ratio': ctx.get('fee_ev_ratio') or 0,
            'block_reason': block_reason,
        }

        # 최근 50건만 유지
        self._missed_opportunities.append(missed)
        if len(self._missed_opportunities) > 50:
            self._missed_opportunities = self._missed_opportunities[-50:]

        logger.warning(
            f"  🚨 [기회 놓침] {self.symbol} {swing_pct:.1%} {direction} 구간 발생! "
            f"(L={peak_low:.2f} H={peak_high:.2f}) "
            f"AI: conf={ctx.get('cascade_conf', 0):.0%} (사유: {block_reason})"
        )

        # 텔레그램 알림
        self.telegram._send_async(
            f"🚨 <b>기회 놓침</b> {self.symbol}\n"
            f"{swing_pct:.1%} {direction} (L={peak_low:.2f} → H={peak_high:.2f})\n"
            f"사유: {block_reason}\n"
            f"cascade_conf: {ctx.get('cascade_conf', 0):.0%} | "
            f"fee_ev: {ctx.get('fee_ev_ratio') or 0:.1f}x"
        )

    def _identify_block_reason(self, ctx: Dict) -> str:
        """사이클 컨텍스트에서 진입 차단 사유를 식별"""
        reasons = []

        cascade_conf = ctx.get('cascade_conf') or 0
        fee_ev = ctx.get('fee_ev_ratio') or 0
        cascade_fire = ctx.get('cascade_fire', False)

        # cascade가 발화하지 않음
        if not cascade_fire:
            if cascade_conf < 0.3:
                reasons.append(f"Cascade 미발화 (conf={cascade_conf:.0%})")
            else:
                reasons.append(f"Cascade 미달 (conf={cascade_conf:.0%})")

        # Fee EV 미달
        if fee_ev > 0 and fee_ev < 1.2:
            reasons.append(f"Fee EV 미달 ({fee_ev:.1f}x < 1.2x)")

        # Regime 감쇠
        regime = ctx.get('regime')
        if regime in ('RANGING', 'LOW_VOLATILITY'):
            reasons.append(f"Regime 감쇠 ({regime})")

        if not reasons:
            reasons.append("복합 요인 (임계값 미달)")

        return " | ".join(reasons)

    def _update_obi_data(self):
        """🔹 WebSocket bookTicker에서 OBI(Orderbook Imbalance) 업데이트

        실시간 WebSocket bookTicker 스트림의 best bid/ask qty를 사용하여
        AI 분석기의 OBI 캐시를 업데이트합니다.
        REST API 폴백: WebSocket 데이터가 없으면 REST로 조회.
        """
        if not self.use_ai_model or not self.market_analyzer:
            return

        try:
            # WebSocket bookTicker 우선 사용
            if self.data_manager:
                bt = self.data_manager.get_book_ticker()
                if bt['update_time'] > 0:
                    bid_vol = bt['bid_qty']
                    ask_vol = bt['ask_qty']
                    self.market_analyzer.update_obi(bid_vol, ask_vol)
                    obi_val = self.market_analyzer._obi_cache
                    # v2 feature_builder OBI 캐시도 업데이트
                    self.feature_builder.update_obi(obi_val)
                    logger.debug(f"OBI(WS): bid={bid_vol:.4f}, ask={ask_vol:.4f}, OBI={obi_val:.4f}")
                    return

            # 폴백: REST API
            depth = self.client.get_order_book(self.symbol, limit=5)
            if not depth or 'bids' not in depth or 'asks' not in depth:
                return

            bid_vol = sum(float(bid[1]) for bid in depth['bids'])
            ask_vol = sum(float(ask[1]) for ask in depth['asks'])
            self.market_analyzer.update_obi(bid_vol, ask_vol)
            obi_val = self.market_analyzer._obi_cache
            # v2 feature_builder OBI 캐시도 업데이트
            self.feature_builder.update_obi(obi_val)
            logger.debug(f"OBI(REST): bid={bid_vol:.4f}, ask={ask_vol:.4f}, OBI={obi_val:.4f}")

        except Exception as e:
            logger.debug(f"OBI 업데이트 실패: {e}")

    def _get_candle_low_high(self) -> tuple:
        """현재 캔들의 (low, high) 반환 — ExecutionManager paper passive 용"""
        if self._historical_mode and self._historical_df is not None and self._hist_cursor > 0:
            row = self._historical_df.iloc[self._hist_cursor - 1]
            return (row['low'], row['high'])
        # Live 모드에서는 마지막 수집된 데이터 사용
        if self.data_manager:
            price = self.data_manager.get_current_price()
            return (price * 0.999, price * 1.001)  # 근사치
        return (0.0, 0.0)

    def _update_pseudo_obi(self, df: pd.DataFrame):
        """Backtest용 pseudo OBI: wick pressure proxy [-1, 1]

        candle의 (close-low)/(high-low) 비율로 매수/매도 압력 근사.
        Live의 bid/ask OBI와 완벽히 같지 않지만 피처 분포 일관성 확보.
        """
        if not self.use_ai_model or len(df) < 1:
            return

        row = df.iloc[-1]
        h, l, c = row['high'], row['low'], row['close']
        hl_range = h - l

        if hl_range > 1e-10:
            pseudo_obi = 2.0 * (c - l) / hl_range - 1.0
        else:
            pseudo_obi = 0.0

        # feature_builder: EMA 평활 적용됨 (alpha=0.3)
        self.feature_builder.update_obi(pseudo_obi)

        # market_analyzer: v1 path 호환 (prepare_features에서 _obi_cache 사용)
        if self.market_analyzer:
            self.market_analyzer._obi_cache = self.feature_builder._obi_cache

    def _populate_backtest_price_log(self, df: pd.DataFrame):
        """백테스트: 1m 캔들에서 PriceLog 보충 (라벨링 해상도용)."""
        if len(df) < 1:
            return
        row = df.iloc[-1]
        base_ts = df.index[-1].timestamp()
        self.price_log.append(base_ts, float(row['close']),
                              float(row['high']), float(row['low']))

    def _get_market_data(self) -> Optional[pd.DataFrame]:
        """시장 데이터 수집"""
        # AI 예측에 필요한 최소 캔들 수 (v2: sequence_length + 여유분)
        min_candles_for_model = 70  # sequence_length(60) + 10

        # 히스토리컬 모드: 슬라이딩 윈도우 반환
        # 통합 모델: 60 × 15m = 900 1m 캔들 + 피처 워밍업 여유
        if self._historical_mode and self._historical_df is not None:
            _window_size = 1500
            start = max(0, self._hist_cursor - _window_size)
            window = self._historical_df.iloc[start:self._hist_cursor].copy()
            if len(window) >= min_candles_for_model:
                return window
            return None

        try:
            if self.data_manager:
                df = self.data_manager.get_candles_df()
                if len(df) >= min_candles_for_model:
                    return df
                elif len(df) > 0:
                    logger.info(f"WebSocket 캔들 부족 ({len(df)}/{min_candles_for_model}), REST API 보충")

            # 폴백: REST API 사용 (통합 모델: 15m×60 리샘플 → 1000개 필요)
            _rest_limit = 1000
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=_rest_limit
            )

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']:
                df[col] = df[col].astype(float)
            df['trades'] = df['trades'].astype(int)

            df.set_index('timestamp', inplace=True)
            return df

        except Exception as e:
            logger.error(f"시장 데이터 수집 실패: {e}")
            return None

    def _get_mtf_market_data(self) -> Dict[str, pd.DataFrame]:
        """Multi-Timeframe 시장 데이터 수집"""
        dataframes = {}

        # 히스토리컬 모드: 1m 데이터를 리샘플링하여 MTF 생성
        if self._historical_mode and self._historical_df is not None:
            window = self._historical_df.iloc[:self._hist_cursor]
            resample_map = {'1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min'}
            for interval in self.mtf_intervals:
                rule = resample_map.get(interval)
                if rule and len(window) > 0:
                    try:
                        _agg = {'open': 'first', 'high': 'max',
                                'low': 'min', 'close': 'last', 'volume': 'sum'}
                        if 'trades' in window.columns:
                            _agg['trades'] = 'sum'
                        if 'taker_buy_base' in window.columns:
                            _agg['taker_buy_base'] = 'sum'
                        resampled = window.resample(rule).agg(_agg).dropna()
                        if len(resampled) >= 30:  # v2 피처는 최소 데이터로 동작
                            dataframes[interval] = resampled
                    except Exception as e:
                        logger.warning(f"MTF 리샘플링 실패 ({interval}): {e}")
            return dataframes

        for interval in self.mtf_intervals:
            try:
                klines = self.client.get_klines(
                    symbol=self.symbol,
                    interval=interval,
                    limit=200
                )

                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])

                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']:
                    df[col] = df[col].astype(float)
                df['trades'] = df['trades'].astype(int)

                df.set_index('timestamp', inplace=True)
                dataframes[interval] = df

            except Exception as e:
                logger.warning(f"MTF 데이터 수집 실패 ({interval}): {e}")

        return dataframes

    def _analyze_leverage(self, df: pd.DataFrame) -> LeverageRecommendation:
        """AI 레버리지 분석"""
        logger.info("")
        logger.info("🔧 [AI 레버리지 분석]")

        # 확신도 기반 signal_confidence (이전 예측, 없으면 0.5)
        _lev_conf = 0.5
        if self._last_unified_prediction:
            _lev_conf = (self._last_unified_prediction.direction_confidence * 0.6 +
                         self._last_unified_prediction.timing_confidence * 0.4)
        recommendation = self.leverage_optimizer.recommend_leverage(
            df=df,
            signal_confidence=_lev_conf,
            position_side='LONG' if not self.current_position else self.current_position.get('side', 'LONG')
        )

        # 연속 손실/드로다운에 따른 조정
        adjusted_leverage = self.leverage_optimizer.adjust_leverage_for_consecutive_losses(
            recommendation.recommended_leverage,
            self.stats.consecutive_losses
        )

        drawdown = (self.initial_balance - self.current_balance) / self.initial_balance
        adjusted_leverage = self.leverage_optimizer.adjust_leverage_for_drawdown(
            adjusted_leverage,
            drawdown
        )

        # 레버리지 업데이트 (포지션 보유 중에는 변경 불가 — 실거래 동일)
        if adjusted_leverage != self.current_leverage:
            if self.current_position:
                logger.debug(f"  레버리지 변경 보류: 포지션 보유 중 ({self.current_leverage}x → {adjusted_leverage}x 대기)")
            else:
                self.current_leverage = adjusted_leverage
                logger.info(f"  📌 레버리지 변경: {self.current_leverage}x")

                if not self.paper_trading:
                    try:
                        self.client.set_leverage(self.symbol, self.current_leverage)
                    except Exception as e:
                        logger.error(f"레버리지 설정 실패: {e}")

        logger.info(f"  추천 레버리지: {recommendation.recommended_leverage}x")
        logger.info(f"  최대 안전 레버리지: {recommendation.max_safe_leverage}x")
        logger.info(f"  리스크 레벨: {recommendation.risk_level.name}")
        logger.info(f"  시장 상태: {recommendation.market_condition}")
        logger.info(f"  변동성: {recommendation.volatility_score:.1%}")
        logger.info(f"  트렌드 강도: {recommendation.trend_strength:.1%}")

        for reason in recommendation.reasons:
            logger.info(f"    → {reason}")

        return recommendation

    def _calculate_higher_tf_trend(self, df: pd.DataFrame) -> str:
        """상위 시간대(15분봉 기준) 추세 계산

        1분봉 데이터를 15분 단위로 리샘플링하여 추세 판단.
        EMA20 기울기 + 가격 위치로 추세 결정.

        Returns:
            'UP', 'DOWN', 'NEUTRAL'
        """
        try:
            if len(df) < 60:  # 최소 1시간 데이터 필요
                return 'NEUTRAL'

            # 최근 60개 캔들 사용 (1시간)
            recent_df = df.tail(60).copy()

            # EMA 계산 (20개 캔들 = 20분)
            ema_short = recent_df['close'].ewm(span=10, adjust=False).mean()
            ema_long = recent_df['close'].ewm(span=30, adjust=False).mean()

            current_price = recent_df['close'].iloc[-1]
            ema_short_now = ema_short.iloc[-1]
            ema_long_now = ema_long.iloc[-1]

            # EMA 기울기 (최근 10개 봉)
            ema_slope = (ema_short.iloc[-1] - ema_short.iloc[-10]) / ema_short.iloc[-10] * 100

            # 추세 판단
            # 상승: 가격 > EMA_short > EMA_long AND EMA 기울기 양수
            # 하락: 가격 < EMA_short < EMA_long AND EMA 기울기 음수
            if current_price > ema_short_now > ema_long_now and ema_slope > 0.05:
                return 'UP'
            elif current_price < ema_short_now < ema_long_now and ema_slope < -0.05:
                return 'DOWN'
            else:
                return 'NEUTRAL'

        except Exception as e:
            logger.debug(f"상위 TF 추세 계산 실패: {e}")
            return 'NEUTRAL'

    def _analyze_position(self, df: pd.DataFrame) -> PositionDecision:
        """AI 포지션 분석 (기술적 지표 + Transformer)"""
        logger.info("")
        logger.info("🎯 [AI 포지션 분석]")

        # 0. 상위 시간대 추세 계산 (역추세 진입 제한용)
        higher_tf_trend = self._calculate_higher_tf_trend(df)
        if higher_tf_trend != 'NEUTRAL':
            logger.info(f"  📊 상위 TF 추세: {higher_tf_trend}")

        # 1. 기술적 지표 분석 (상위 TF 추세 전달)
        decision = self.position_manager.decide_position(
            df=df,
            current_position=self.current_position,
            leverage=self.current_leverage,
            higher_tf_trend=higher_tf_trend
        )

        # ═══════════════════════════════════════════════════
        # 2. AI 분석 — 통합 모델
        # ═══════════════════════════════════════════════════
        return self._analyze_position_unified(df, decision)
    def _analyze_position_unified(
        self, df: pd.DataFrame, decision: 'PositionDecision'
    ) -> 'PositionDecision':
        """통합 모델 전용 포지션 분석 — cascade pipeline 우회.

        1m 캔들 → 15m/3m 리샘플(+1m micro) → unified_analyzer.predict()
        → 방향/신뢰도/타이밍 직접 사용 → 성숙도/Shadow gate 적용
        """
        import math as _math

        logger.info("  🧬 [통합 모델 분석]")

        # 1. 데이터 준비: 1m → 15m/3m 리샘플, micro=1m
        try:
            df_15m, df_3m, df_1m = self._collect_unified_data(df)
            logger.info(f"    데이터: 15m={len(df_15m)}캔들, 3m={len(df_3m)}캔들, 1m={len(df_1m)}캔들")
        except Exception as e:
            logger.warning(f"    데이터 리샘플 실패: {e}")
            decision.action = PositionAction.HOLD
            decision.side = 'NONE'
            decision.confidence = 0.0
            decision.position_size_pct = 0.0
            return decision

        # 2. 통합 모델 예측 (백테스트: 캔들 시뮬레이션 시간 전달)
        _sim_time = None
        if self._historical_mode and hasattr(df_1m.index, 'max'):
            _sim_time = df_1m.index.max().timestamp()
        up = self.unified_analyzer.predict(df_15m, df_3m, df_1m, sim_time=_sim_time)
        if up is None:
            logger.warning("    통합 모델 예측 실패 (데이터 부족 또는 오류)")
            decision.action = PositionAction.HOLD
            decision.side = 'NONE'
            decision.confidence = 0.0
            decision.position_size_pct = 0.0
            return decision

        self._last_unified_prediction = up
        self.last_prediction_direction = up.predicted_direction
        self.last_prediction_price = df['close'].iloc[-1]

        # 로그 출력
        _dir_emoji = '🟢' if up.predicted_direction == 'UP' else ('🔴' if up.predicted_direction == 'DOWN' else '⚪')
        logger.info(f"    {_dir_emoji} 예측: {up.predicted_direction} "
                     f"(dir_conf={up.direction_confidence:.1%}, "
                     f"timing={up.timing_confidence:.1%}, "
                     f"change={up.price_change_pct:+.2f}%)")
        logger.info(f"    Direction probs: D={up.direction_probs['down']:.1%} "
                     f"U={up.direction_probs['up']:.1%}")

        # 진단용 예측 히스토리 기록
        self._unified_pred_history.append({
            'dir': up.predicted_direction,
            'probs': (up.direction_probs['down'], up.direction_probs['up']),
        })

        # 3. 방향 결정 — DL 2-class (UP/DOWN) 주도
        # Shadow live 이전: AR(1) fallback (DL 미검증, cold-start 보호)
        # Shadow live 이후: DL 단독 (2-class UP/DOWN → LONG/SHORT)
        sv = self.unified_analyzer.shadow_validator

        # DL 예측 → 방향
        _dl_dir = up.predicted_direction   # 'UP', 'DOWN', 'NEUTRAL'
        _dl_conf = up.direction_confidence

        # AC(1) — size multiplier용 (DL/AR1 모드 공통, 반드시 if/else 이전에 계산)
        _ac1 = self._compute_rolling_ac1()

        if sv.is_live:
            # DL 주도: direction_confidence 직접 사용
            if _dl_dir == 'UP':
                cascade_dir = 'LONG'
            elif _dl_dir == 'DOWN':
                cascade_dir = 'SHORT'
            else:
                cascade_dir = 'NEUTRAL'
            cascade_conf = max(0.30, min(1.0, _dl_conf))
            logger.info(f"    🧠 DL: {_dl_dir} → {cascade_dir} (conf={_dl_conf:.1%})")
        else:
            # AR(1) fallback — Shadow 기간 cold-start 보호
            _ar1_dir = self._ar1_last_direction
            if _ar1_dir == 'UP':
                cascade_dir = 'LONG'
            elif _ar1_dir == 'DOWN':
                cascade_dir = 'SHORT'
            else:
                cascade_dir = 'NEUTRAL'
            cascade_conf = max(0.30, min(1.0, _ac1))
            if _ac1 < 0.30:
                cascade_dir = 'NEUTRAL'
            logger.info(f"    📈 AR(1) fallback: {_ar1_dir} → {cascade_dir}, AC(1)={_ac1:.3f}")

        logger.info(f"    📊 DL probs: D={up.direction_probs['down']:.1%} "
                    f"U={up.direction_probs['up']:.1%} "
                    f"(timing={up.timing_confidence:.1%})")

        # final_score 계산
        if cascade_dir == 'LONG':
            final_score = cascade_conf
        elif cascade_dir == 'SHORT':
            final_score = -cascade_conf
        else:
            final_score = 0.0

        # 4. Shadow Mode 게이트 (콜드스타트 보호 — 유일한 보호 레이어)
        # sv는 위에서 이미 정의됨
        _sv_status = sv.get_status()
        _ai_accuracy = _sv_status['recent_accuracy']
        _pred_count = _sv_status['resolved_count'] + _sv_status['pending_count']
        self._last_ai_accuracy = round(_ai_accuracy, 4)
        self._last_ai_maturity = 1.0 if sv.is_live else 0.0

        if final_score != 0.0 and not sv.is_live:
            if self._historical_mode:
                # 백테스트: Shadow 통계만 기록, 거래 차단 안 함 (성능 측정용)
                logger.debug(
                    f"    📊 Shadow Stats: {_sv_status['state']} "
                    f"(해소={_sv_status['resolved_count']}/{sv.config.min_samples}, "
                    f"acc={_sv_status['recent_accuracy']:.1%}, "
                    f"pf={_sv_status['recent_profit_factor']:.2f})"
                )
            else:
                # 라이브: Shadow 연성 게이트 — 완전 차단 대신 신호 스케일링
                _scale = sv.config.demoted_signal_scale if sv.state.value == 'demoted' else 0.0
                if _scale > 0:
                    final_score *= _scale
                    logger.info(
                        f"    ⏳ Shadow 연성 게이트: {_sv_status['state']} "
                        f"(해소={_sv_status['resolved_count']}/{sv.config.min_samples}, "
                        f"acc={_sv_status['recent_accuracy']:.1%}) → 신호 ×{_scale:.1f}"
                    )
                else:
                    # SHADOW 상태 (완전 미검증): 차단
                    logger.info(
                        f"    ⏳ Shadow Mode: {_sv_status['state']} "
                        f"(해소={_sv_status['resolved_count']}/{sv.config.min_samples}) → 거래 차단"
                    )
                    final_score = 0.0

        # 6. 변동성 스케일링
        _vol_factor = self._cycle_ctx.get('vol_factor', 1.0)

        # 7. 레짐 임계값 — 찍기성 거래 차단 강화
        _regime = self._current_regime
        hold_threshold = 0.25
        if _regime:
            _rv = _regime.regime.value
            if _rv == 'ranging':
                hold_threshold = 0.30
            elif _rv == 'volatile':
                hold_threshold = 0.30
            elif _rv in ('trending_up', 'trending_down'):
                hold_threshold = 0.20
            logger.info(f"    📊 레짐 {_rv} → 임계값 {hold_threshold}")

        max_pos_pct = self.position_manager.max_position_pct
        final_direction = 'NEUTRAL'
        if abs(final_score) >= hold_threshold:
            final_direction = 'LONG' if final_score > 0 else 'SHORT'

        # ── 추세 필터: 역추세 진입 하드 차단 ──
        # EMA(60) 기반 — 하락 추세에서 LONG 차단, 상승 추세에서 SHORT 차단
        _trend = self._calculate_higher_tf_trend(df)
        if final_direction == 'LONG' and _trend == 'DOWN':
            logger.info(f"    🚫 추세 필터: 하락 추세에서 LONG 차단 → NEUTRAL")
            final_direction = 'NEUTRAL'
        elif final_direction == 'SHORT' and _trend == 'UP':
            logger.info(f"    🚫 추세 필터: 상승 추세에서 SHORT 차단 → NEUTRAL")
            final_direction = 'NEUTRAL'

        logger.info(f"    최종 스코어: {final_score:+.3f} "
                     f"(shadow={_sv_status['state']}, preds={_pred_count}, acc={_ai_accuracy:.1%}, "
                     f"vol_adj=×{_vol_factor:.2f}, trend={_trend})")

        # 8. AC(1) 기반 포지션 사이즈 스케일링 (DL timing_confidence 병행 사용)
        # AC(1) 저하 구간: 50% 감소 / 정상: DL timing_confidence 반영
        _ac1_size_mult = 0.5 if 0.30 <= _ac1 < 0.50 else 1.0
        _conf_size_mult = min(1.5, max(0.5,
            up.timing_confidence  # timing_confidence만 사용 (방향 conf는 AR(1)으로 대체)
        )) * _ac1_size_mult
        logger.info(
            f"    🎯 Size mult: AC(1)={_ac1:.3f}×{_ac1_size_mult:.1f}, "
            f"timing={up.timing_confidence:.2f} → "
            f"size_mult={_conf_size_mult:.2f}x"
        )

        # 9. 최종 결정 — 기존 포지션 상태 인식
        consec_penalty = max(0.3, 1.0 - self.stats.consecutive_losses * 0.15)
        has_position = self.current_position is not None
        position_side = self.current_position.get('side') if has_position else None

        if has_position:
            # ── 포지션 보유 중: TP/SL에만 의존 ──
            # Signal-based exit 시도 결과: WinRate 32%, PF 0.62 (vs TP/SL only 45%, 0.69)
            # Label flip 빈도(40%) × 15min 간격 → TP 도달 전 close → 수수료 churning
            decision.action = PositionAction.HOLD
            decision.side = position_side
            decision.confidence = abs(final_score)
            decision.position_size_pct = 0.0
        else:
            # ── 포지션 없음: 신규 진입 ──
            # Kelly 사이징: 충분한 이력(20건+) 있으면 Kelly, 없으면 score 기반 fallback
            _MIN_ENTRY_PCT = 0.03  # 최소 진입 3% (거래 지속 보장)
            if len(self.trade_history) >= 20:
                _size_base = min(self._get_kelly_size(), max_pos_pct)
            else:
                _size_base = min(abs(final_score), max_pos_pct) * consec_penalty
            _size_base = max(_size_base, _MIN_ENTRY_PCT)  # floor 적용
            if final_direction == 'LONG':
                decision.action = PositionAction.OPEN_LONG
                decision.side = 'LONG'
                decision.confidence = min(1.0, abs(final_score))
                decision.position_size_pct = _size_base * _vol_factor * _conf_size_mult
            elif final_direction == 'SHORT':
                decision.action = PositionAction.OPEN_SHORT
                decision.side = 'SHORT'
                decision.confidence = min(1.0, abs(final_score))
                decision.position_size_pct = _size_base * _vol_factor * _conf_size_mult
            else:
                decision.action = PositionAction.HOLD
                decision.side = 'NONE'
                decision.confidence = abs(final_score)
                decision.position_size_pct = 0.0

            # SL/TP 재계산: AI가 결정한 최종 방향으로 SL/TP 갱신
            # (decide_position은 TA 방향으로 SL/TP를 계산하므로 AI 방향과 불일치 가능)
            if decision.action in (PositionAction.OPEN_LONG, PositionAction.OPEN_SHORT):
                _price = df['close'].iloc[-1]
                _atr = df['atr'].iloc[-1] if 'atr' in df.columns else _price * 0.005
                decision.stop_loss, decision.take_profit = \
                    self.position_manager._calculate_sl_tp(
                        _price, decision.side, _atr, self.current_leverage
                    )

        # SL/TP: position_manager._calculate_sl_tp()가 sample_buffer 상수로 계산 완료
        # 여기서는 로깅만 (학습 라벨 ↔ 트레이딩 배리어 일치 보장)
        if decision.action in (PositionAction.OPEN_LONG, PositionAction.OPEN_SHORT):
            _price = df['close'].iloc[-1]
            _sl_dist = abs(decision.stop_loss - _price) if decision.stop_loss else 0
            _tp_dist = abs(decision.take_profit - _price) if decision.take_profit else 0
            _atr = self._current_atr if self._current_atr > 0 else _price * 0.005
            logger.info(f"    SL/TP: SL={decision.stop_loss:.2f}, TP={decision.take_profit:.2f} "
                        f"(ATR={_atr:.2f}, sl_dist={_sl_dist:.2f}, tp_dist={_tp_dist:.2f})")

        logger.info(f"    최종: {decision.action.value} ({decision.confidence:.1%})")

        # 10. cycle_ctx + 대시보드 캐시 업데이트
        self._cycle_ctx.update({
            'cascade_dir': final_direction,
            'cascade_conf': float(round(cascade_conf, 3)),
            'cascade_fire': True,  # 통합 모델은 cascade 불필요
            'fee_ev_ratio': 0.0,  # 통합 모델은 fee EV 미사용 (로깅용 유지)
            'timing_confidence': float(round(up.timing_confidence, 3)),
            'final_score': float(round(final_score, 3)),
            'hold_threshold': float(round(hold_threshold, 3)),
            'regime': _regime.regime.value if _regime else None,
            'regime_conf': float(round(_regime.confidence, 2)) if _regime else None,
            'action': decision.action.value,
            'side': decision.side,
            'confidence': float(round(decision.confidence, 3)),
            'size_pct': float(round(decision.position_size_pct, 4)),
            'ai_maturity': float(self._last_ai_maturity),
            'ai_accuracy': float(round(_ai_accuracy, 4)),
            'pred_count': _pred_count,
        })

        # 대시보드용 AI 정보
        self._last_ai_info = {
            'direction': up.predicted_direction,
            'confidence': up.direction_confidence,
            'predicted_price': up.predicted_price,
            'price_change_pct': up.price_change_pct,
            'timing_confidence': up.timing_confidence,
            'direction_probs': up.direction_probs,
            'regime': _regime.regime.value if _regime else None,
            'regime_confidence': _regime.confidence if _regime else None,
        }

        return decision

    def _snapshot_to_ring_buffers(self, df: pd.DataFrame, current_price: float):
        """매 사이클 피처 스냅샷을 링 버퍼에 저장 (필터링 없음)

        1m/MTF 모델용 피처를 계산하여 각 TF 링 버퍼에 저장.
        PriceLog에도 현재 가격 기록.
        backtest: 캔들 타임스탬프 사용 (ring buffer 지연 라벨링이 정상 동작하도록)
        """
        from ai.sample_buffer import SnapshotEntry
        from ai.feature_builder import TFFeatureBuilder

        if not self.use_ai_model:
            return

        now = self._sim_time  # backtest: 캔들 시간, live: wall-clock
        # 1m 캔들 high/low → PriceLog wick 감지 (사전학습과 동일한 해상도)
        _candle_high = float(df.iloc[-1]['high']) if len(df) > 0 else current_price
        _candle_low = float(df.iloc[-1]['low']) if len(df) > 0 else current_price
        self.price_log.append(now, current_price, _candle_high, _candle_low)

        # 라이브 캔들 중복 방지: 같은 캔들이면 SnapshotEntry skip (PriceLog은 매 사이클 기록)
        if not self._historical_mode:
            _snap_candle_ts = float(df.index[-1].timestamp()) if len(df) > 0 else 0.0
            if _snap_candle_ts == self._last_snapshot_candle_ts:
                return
            self._last_snapshot_candle_ts = _snap_candle_ts

        # 변동성 Z-score 계산 (live soft weight용)
        volatility_z = 0.0
        if len(df) >= 30:
            _returns = df['close'].pct_change().dropna()
            _rolling_vol = _returns.rolling(20, min_periods=5).std()
            _vol_mean = _rolling_vol.mean()
            _vol_std = _rolling_vol.std()
            if _vol_std > 1e-10:
                volatility_z = float((_rolling_vol.iloc[-1] - _vol_mean) / _vol_std)
                volatility_z = max(-4.0, min(4.0, volatility_z))

        # 1m 링 버퍼 — ATR 포함
        _1m_analyzer = self._get_analyzer_for_tf('1m')
        if _1m_analyzer and len(df) >= _1m_analyzer.sequence_length + 5:
            try:
                features = self.feature_builder.prepare_features(df, '1m')
                seq = features[-_1m_analyzer.sequence_length:].copy()
                # Raw ATR for Triple Barrier labeling
                _atr_val = 0.0
                if len(df) >= 15:
                    _atr_s = TFFeatureBuilder.compute_raw_atr(df)
                    _last_atr = _atr_s.iloc[-1]
                    _atr_val = float(_last_atr) if not pd.isna(_last_atr) else 0.0
                entry = SnapshotEntry(timestamp=now, price=current_price, features=seq,
                                      volatility_z=volatility_z, atr=_atr_val)
                self.ring_buffers['1m'].add_snapshot(entry)
            except Exception as e:
                logger.debug(f"1m 스냅샷 실패: {e}")

        # MTF 링 버퍼 — ATR 포함
        if self.use_multi_timeframe and self.mtf_analyzer:
            try:
                dataframes = self._get_mtf_market_data()
                for interval, mtf_df in dataframes.items():
                    if interval not in self.ring_buffers:
                        continue
                    analyzer = self.mtf_analyzer.analyzers.get(interval)
                    if not analyzer or len(mtf_df) < analyzer.sequence_length + 5:
                        continue

                    features = self.feature_builder.prepare_features(mtf_df, interval)
                    seq = features[-analyzer.sequence_length:].copy()
                    # Raw ATR for Triple Barrier labeling
                    _atr_val = 0.0
                    if len(mtf_df) >= 15:
                        _atr_s = TFFeatureBuilder.compute_raw_atr(mtf_df)
                        _last_atr = _atr_s.iloc[-1]
                        _atr_val = float(_last_atr) if not pd.isna(_last_atr) else 0.0
                    entry = SnapshotEntry(timestamp=now, price=current_price, features=seq,
                                          volatility_z=volatility_z, atr=_atr_val)
                    self.ring_buffers[interval].add_snapshot(entry)
            except Exception as e:
                logger.debug(f"MTF 스냅샷 실패: {e}")

        # ── 통합 모델 3-TF 동시 스냅샷 ──
        if self.unified_analyzer:
            self._snapshot_unified_features(df, current_price, now, volatility_z)

    def _snapshot_unified_features(
        self, df: pd.DataFrame, current_price: float,
        now: float, volatility_z: float,
    ):
        """통합 모델용 3-TF 동시 피처 스냅샷."""
        try:
            from ai.feature_builder import TFFeatureBuilder

            df_15m, df_3m, df_1m = self._collect_unified_data(df)
            ua = self.unified_analyzer
            fb = ua.feature_builder

            macro_feat = fb.prepare_features(df_15m, '15m')
            mid_feat = fb.prepare_features(df_3m, '3m')
            micro_feat = fb.prepare_features(df_1m, '1m')

            if (len(macro_feat) < ua.macro_seq_len
                    or len(mid_feat) < ua.mid_seq_len
                    or len(micro_feat) < ua.micro_seq_len):
                return

            macro_seq = macro_feat[-ua.macro_seq_len:].copy()
            mid_seq = mid_feat[-ua.mid_seq_len:].copy()
            micro_seq = micro_feat[-ua.micro_seq_len:].copy()

            # ATR from 3m data (Triple Barrier labeling용)
            atr_val = 0.0
            if len(df_3m) >= 15:
                atr_s = TFFeatureBuilder.compute_raw_atr(df_3m)
                _last = atr_s.iloc[-1]
                atr_val = float(_last) if not pd.isna(_last) else 0.0

            self._unified_pending_snapshots.append({
                'timestamp': now,
                'price': current_price,
                'macro_features': macro_seq,
                'mid_features': mid_seq,
                'micro_features': micro_seq,
                'atr': atr_val,
                'volatility_z': volatility_z,
            })
        except Exception as e:
            logger.debug(f"통합 모델 스냅샷 실패: {e}")

    def _label_matured_samples(self):
        """look-ahead 경과한 스냅샷을 라벨링하여 학습 버퍼에 추가"""
        import math as _math
        import numpy as np

        if not self.use_ai_model:
            return

        now = self._sim_time  # backtest: 캔들 시간, live: wall-clock

        for tf, ring_buf in self.ring_buffers.items():
            labeled = ring_buf.label_matured_entries(self.price_log, now)

            if not labeled:
                continue

            # ── TF별 라벨 카운트 업데이트 (inverse frequency 계산용) ──
            if tf not in self._label_counts:
                self._label_counts[tf] = {'UP': 0, 'DOWN': 0, 'NEUTRAL': 0}
            for sample in labeled:
                self._label_counts[tf][sample.direction] = self._label_counts[tf].get(sample.direction, 0) + 1

            # Inverse frequency weight: 희소 클래스 → 높은 가중치
            _counts = self._label_counts[tf]
            _total = sum(_counts.values())
            _inv_freq = {}
            for _d in ('UP', 'DOWN', 'NEUTRAL'):
                _c = max(_counts.get(_d, 0), 1)
                _inv_freq[_d] = _total / (3.0 * _c)

            for sample in labeled:
                # 1) soft weight = sigmoid(volatility_z) — Live/Backtest 분포 일관성
                sample.sample_weight = 1.0 / (1.0 + _math.exp(-sample.sample_weight))

                # 2) Inverse frequency 가중치 — 희소 라벨 부스트
                sample.sample_weight *= _inv_freq.get(sample.direction, 1.0)

                # 3) 수수료 데드존 감쇠 — 완전 제거 대신 가중치 약화
                _fee_deadzone = self.trading_fee_rate * 2  # 왕복 수수료
                if abs(sample.log_return) < _fee_deadzone:
                    if self._historical_mode:
                        sample.sample_weight *= 0.7  # backtest: 약한 감쇠
                    else:
                        sample.sample_weight *= 0.3  # live: 강한 감쇠

                # 해당 TF의 analyzer를 찾아서 training_buffer에 추가
                analyzer = self._get_analyzer_for_tf(tf)
                if not analyzer:
                    continue

                from ai.market_analyzer import TrainingSample
                from datetime import datetime

                # 시뮬레이션 시간 기반 타임스탬프 (recency decay 정확하게 적용하기 위해)
                sample_dt = datetime.fromtimestamp(sample.timestamp)

                if tf == '10s':
                    # ── 10s 라벨링: 사전학습과 동일한 방식 ──
                    # 사전학습: 단순 방향(UP/DOWN only) + 노이즈 필터
                    # Triple Barrier의 NEUTRAL을 사용하지 않음

                    # 노이즈 필터: |log_return| < 0.1 × ATR ratio → skip
                    _atr_r = max(self._tick_atr_ratio_cache, 1e-10)
                    _micro_thr = max(1e-6, _atr_r * 0.1)
                    if abs(sample.log_return) < _micro_thr:
                        continue  # 사전학습과 동일하게 노이즈 스킵

                    # 단순 방향 라벨 (사전학습과 동일: NEUTRAL 없음)
                    _tick_direction = 'UP' if sample.log_return > 0 else 'DOWN'

                    # 동적 timing_label: sigmoid(return/atr × regime_factor)
                    _regime_f = 1.0 / (1.0 + _math.exp(-self._tick_volatility_z_cache))
                    _tl = 1.0 / (1.0 + _math.exp(-(sample.log_return / _atr_r * _regime_f)))

                    # 틱 분석기 training_buffer (dict 형식)
                    analyzer.training_buffer.append({
                        'features': sample.features,
                        'direction': _tick_direction,
                        'price_change': sample.price_change,
                        'timing_label': _tl,
                        'sample_weight': sample.sample_weight,
                        'timestamp': sample.timestamp,  # epoch seconds
                    })
                else:
                    # MarketAnalyzer training_buffer (TrainingSample 형식)
                    ts = TrainingSample(
                        features=sample.features,
                        actual_direction=sample.direction,
                        actual_price_change=sample.price_change,
                        timestamp=sample_dt,
                        sample_weight=sample.sample_weight,
                        future_log_return=sample.log_return,
                    )
                    analyzer.training_buffer.append(ts)

            logger.debug(f"v2 라벨링 완료: {tf} +{len(labeled)}개 (pending={ring_buf.pending_count}), "
                        f"누적분포={self._label_counts[tf]}")

        # ── 통합 모델 3-TF 라벨링 ──
        if self.unified_analyzer and self._unified_pending_snapshots:
            self._label_unified_snapshots()

    def _label_unified_snapshots(self):
        """통합 모델용 3-TF 스냅샷 지연 라벨링 → unified_analyzer.training_buffer"""
        import math as _math
        import numpy as np
        from ai.sample_buffer import (
            compute_barriers, compute_trading_costs, compute_net_label,
            BARRIER_K_TP, BARRIER_K_SL, MAE_QUALITY_THRESHOLD,
        )
        from ai.unified_analyzer import UnifiedTrainingSample

        now = self._sim_time
        look_ahead = self._UNIFIED_LOOK_AHEAD_S
        labeled_count = 0

        while self._unified_pending_snapshots:
            snap = self._unified_pending_snapshots[0]
            elapsed = now - snap['timestamp']
            if elapsed < look_ahead:
                break

            self._unified_pending_snapshots.popleft()
            entry_price = snap['price']
            atr = snap['atr']
            if entry_price <= 0:
                continue

            # PriceLog에서 look-ahead 윈도우 가격 범위 조회
            target_end = snap['timestamp'] + look_ahead
            price_range = self.price_log.lookup_range(snap['timestamp'] + 1, target_end)
            if not price_range:
                continue

            # Triple Barrier
            tp_barrier, sl_barrier = compute_barriers(entry_price, atr, BARRIER_K_TP, BARRIER_K_SL)
            exit_price = entry_price
            barrier_type = 'TIME'
            hit_elapsed = -1.0
            running_min_lo = entry_price  # MAE 추적 (UP 방향용)
            running_max_hi = entry_price  # MAE 추적 (DOWN 방향용)

            for ts, px, hi, lo in price_range:
                running_min_lo = min(running_min_lo, lo)
                running_max_hi = max(running_max_hi, hi)
                hit_tp = hi >= tp_barrier
                hit_sl = lo <= sl_barrier
                if hit_tp and hit_sl:
                    exit_price, barrier_type = sl_barrier, 'SL'
                    hit_elapsed = ts - snap['timestamp']
                    break
                elif hit_tp:
                    exit_price, barrier_type = tp_barrier, 'TP'
                    hit_elapsed = ts - snap['timestamp']
                    break
                elif hit_sl:
                    exit_price, barrier_type = sl_barrier, 'SL'
                    hit_elapsed = ts - snap['timestamp']
                    break
            else:
                exit_price = price_range[-1][1]

            gross_lr = np.log(exit_price / entry_price)
            _, total_cost = compute_trading_costs(atr, entry_price)
            atr_pct = atr / entry_price if atr > 0 else 0.0
            direction, net_lr = compute_net_label(gross_lr, total_cost, barrier_type, atr_pct)

            # MAE 품질 필터 — sample_buffer.py Phase 4와 동일 로직 (라이브 라벨링에 적용)
            # TP 샘플: 진입 후 최대 역행 / SL거리 > threshold → 운 좋은 생존이므로 NEUTRAL
            # SL 샘플: 진입 후 최대 역행 / TP거리 > threshold → 과도한 역행이므로 NEUTRAL
            if direction != 'NEUTRAL':
                _sl_dist = entry_price - sl_barrier
                _tp_dist = tp_barrier - entry_price
                if direction == 'UP' and _sl_dist > 1e-10:
                    _mae_ratio = (entry_price - running_min_lo) / _sl_dist
                    if _mae_ratio > MAE_QUALITY_THRESHOLD:
                        direction = 'NEUTRAL'
                        net_lr = 0.0
                        barrier_type = 'MAE_REJECT'
                elif direction == 'DOWN' and _tp_dist > 1e-10:
                    _mae_ratio = (running_max_hi - entry_price) / _tp_dist
                    if _mae_ratio > MAE_QUALITY_THRESHOLD:
                        direction = 'NEUTRAL'
                        net_lr = 0.0
                        barrier_type = 'MAE_REJECT'

            # Timing score
            timing_score = max(0.0, 1.0 - (hit_elapsed / look_ahead)) if hit_elapsed >= 0 else 0.0

            # Sample weight: sigmoid(volatility_z)
            raw_w = 1.0 / (1.0 + _math.exp(-snap['volatility_z']))

            # Fee deadzone attenuation
            _fee_deadzone = self.trading_fee_rate * 2
            if abs(net_lr) < _fee_deadzone:
                raw_w *= 0.3 if not self._historical_mode else 0.7

            future_ret = (exit_price - entry_price) / entry_price

            from datetime import datetime as _dt
            sample = UnifiedTrainingSample(
                macro_features=snap['macro_features'],
                mid_features=snap['mid_features'],
                micro_features=snap['micro_features'],
                actual_direction=direction,
                actual_price_change=future_ret,
                timestamp=_dt.fromtimestamp(snap['timestamp']),
                sample_weight=raw_w,
                future_log_return=net_lr,
                actual_timing=timing_score,
            )
            self.unified_analyzer.add_training_sample(sample)
            labeled_count += 1

            # ── AR(1) 추적: 비-NEUTRAL 거래만 저장 (거래 기반 window 300) ──
            if direction != 'NEUTRAL':
                self._ar1_last_direction = direction
                self._ar1_label_history.append(1 if direction == 'UP' else -1)

        if labeled_count > 0:
            logger.debug(
                f"통합 모델 라벨링: +{labeled_count}개, "
                f"buffer={len(self.unified_analyzer.training_buffer)}, "
                f"AR(1)_last={self._ar1_last_direction}"
            )

    def _get_analyzer_for_tf(self, timeframe: str):
        """TF에 해당하는 analyzer 반환

        MTF 모드에서는 1m도 mtf_analyzer.analyzers['1m']을 사용해야
        periodic_train_all()에서 ring buffer 샘플이 학습에 반영됨.
        """
        if timeframe == '10s':
            return self.tick_analyzer
        # MTF 모드: 모든 TF(1m 포함)를 mtf_analyzer에서 가져옴
        if self.mtf_analyzer and timeframe in self.mtf_analyzer.analyzers:
            return self.mtf_analyzer.analyzers[timeframe]
        # 단일 TF 모드 fallback
        if timeframe == '1m':
            return self.market_analyzer
        return None

    # ── AR(1) 전략 헬퍼 ──────────────────────────────────────

    def _compute_rolling_ac1(self, min_samples: int = 30) -> float:
        """Rolling AC(1) — AR(1) 레짐 신뢰도 지표.

        Returns:
            float: 자기상관 계수 (-1.0 ~ 1.0).
                   샘플 부족 시 백테스트에서 확인된 평균값(0.60) 반환.
        """
        import numpy as np
        hist = list(self._ar1_label_history)  # 이미 NEUTRAL 제외됨 (거래 기반)
        if len(hist) < min_samples:
            return 0.60  # 콜드스타트: 백테스트 확인 평균값
        arr = np.array(hist, dtype=float)
        if np.std(arr) < 1e-10:
            return 1.0  # 모두 같은 방향 (완전 추세)
        corr = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
        return float(np.clip(corr, -1.0, 1.0))

    def _compute_current_mdd(self) -> float:
        """현재 최대 낙폭 (0 ~ -1.0) — trade_history balance_after 기준."""
        if not self.trade_history:
            return 0.0
        peak = self.initial_balance
        max_dd = 0.0
        for t in self.trade_history:
            bal = t.get('balance_after', peak)
            if bal > peak:
                peak = bal
            if peak > 0:
                dd = (bal - peak) / peak
                if dd < max_dd:
                    max_dd = dd
        return max_dd

    # ── 레거시 학습 샘플 수집 (v1 — 호환 유지) ──────────────

    def _compute_rolling_r_values(self, window: int = 50):
        """
        Rolling R values for EV computation.

        Returns:
            (R_long, R_short, R_loss) — 평균 수익률 (%)
        """
        if not self.trade_history:
            return 0.15, 0.15, 0.10  # cold start defaults

        recent = self.trade_history[-window:]
        long_wins = [t['pnl_pct'] for t in recent if t.get('side') == 'LONG' and t.get('pnl_pct', 0) > 0]
        short_wins = [t['pnl_pct'] for t in recent if t.get('side') == 'SHORT' and t.get('pnl_pct', 0) > 0]
        losses = [abs(t['pnl_pct']) for t in recent if t.get('pnl_pct', 0) < 0]

        R_long = sum(long_wins) / len(long_wins) if long_wins else 0.15
        R_short = sum(short_wins) / len(short_wins) if short_wins else 0.15
        R_loss = sum(losses) / len(losses) if losses else 0.10

        return R_long, R_short, R_loss

    # ── Cascade Pipeline 헬퍼 ──────────────────────────────────

    def _execute_periodic_training(self):
        """주기적 AI 학습 실행"""
        if not self.use_ai_model:
            return

        # OOS 구간에서는 학습 동결
        if self._historical_mode and not getattr(self, '_backtest_allow_training', True):
            return

        # 히스토리컬 모드: 캔들 카운터 기반 학습 (30캔들 = 실시간 30분)
        if self._historical_mode:
            self._hist_train_counter += 1
            if self._hist_train_counter < 30:
                return
            self._hist_train_counter = 0

            try:
                if self.unified_analyzer:
                    result = self.unified_analyzer.periodic_train(min_interval_minutes=0)
                    if result and result.get('status') == 'completed':
                        logger.info(f"🧠 [히스토리컬 학습] 통합모델: "
                                   f"샘플={result.get('samples', 0)}, "
                                   f"정확도={result.get('dir_accuracy', 0):.1%}")
                    else:
                        logger.debug(f"🧠 [히스토리컬 학습] 통합모델 스킵: {result}")
                elif self.use_multi_timeframe and self.mtf_analyzer:
                    results = self.mtf_analyzer.periodic_train_all(
                        min_interval_minutes=0, force=True,
                        is_pretrain=self._backtest_is_pretrain
                    )
                    if results:
                        trained = [iv for iv, r in results.items()
                                   if r and r.get('status') == 'completed']
                        if trained:
                            logger.info(f"🧠 [히스토리컬 학습] MTF 완료: {', '.join(trained)}")
                elif self.market_analyzer:
                    result = self.market_analyzer.periodic_train(
                        min_interval_minutes=0, force=True,
                        save_path=self.model_save_path,
                        is_pretrain=self._backtest_is_pretrain
                    )
                    if result and result.get('status') == 'completed':
                        logger.info(f"🧠 [히스토리컬 학습] 1m 완료: "
                                   f"샘플={result.get('samples', 0)}, "
                                   f"정확도={result.get('direction_accuracy', 0):.1%}")
            except Exception as e:
                logger.warning(f"히스토리컬 학습 실패: {e}")
            return

        try:
            # 초기 학습 가속: training_count < 5 → 10분, 이후 → 기본 30분
            _WARMUP_THRESHOLD = 5
            _WARMUP_INTERVAL = 10  # 분

            # ── 통합 모델 (LoRA 온라인 학습) ──
            if self.unified_analyzer:
                _tc = getattr(self.unified_analyzer, '_training_count', 0)
                _effective_interval = (
                    _WARMUP_INTERVAL if _tc < _WARMUP_THRESHOLD
                    else self.model_training_interval_minutes
                )
                result = self.unified_analyzer.periodic_train(
                    min_interval_minutes=_effective_interval
                )
                if result and result.get('status') == 'completed':
                    logger.info("")
                    logger.info("🧠 [통합 모델 주기적 학습 완료]")
                    logger.info(f"  샘플={result.get('samples', 0)}, "
                               f"정확도={result.get('dir_accuracy', 0):.1%}, "
                               f"학습횟수={result.get('training_count', 0)}")
                    self._save_unified_model_versioned()
                return  # unified는 legacy 분기 skip

            _primary_tc = 0
            _pa = self._get_analyzer_for_tf('1m')
            if _pa:
                _primary_tc = getattr(_pa, 'training_count', 0)
            _effective_interval = _WARMUP_INTERVAL if _primary_tc < _WARMUP_THRESHOLD else self.model_training_interval_minutes

            # Multi-Timeframe 모드: 각 시간대별 개별 학습
            if self.use_multi_timeframe and self.mtf_analyzer:
                results = self.mtf_analyzer.periodic_train_all(
                    min_interval_minutes=_effective_interval
                )

                if results:
                    logger.info("")
                    logger.info("🧠 [Multi-Timeframe AI 주기적 학습 완료]")
                    for interval, result in results.items():
                        if result and result.get('status') == 'completed':
                            logger.info(f"  ✅ {interval}: 샘플={result.get('samples', 0)}, "
                                       f"정확도={result.get('direction_accuracy', 0):.1%}")
                        elif result and result.get('status') == 'failed':
                            logger.warning(f"  ❌ {interval}: {result.get('error', 'unknown error')}")

            # 단일 시간대 모드: 기존 방식 (초기 가속 동일 적용)
            elif self.market_analyzer:
                result = self.market_analyzer.periodic_train(
                    min_interval_minutes=_effective_interval,
                    save_path=self.model_save_path
                )

                if result:
                    logger.info("")
                    logger.info("🧠 [AI 주기적 학습 완료]")
                    logger.info(f"  상태: {result.get('status')}")
                    logger.info(f"  학습 샘플: {result.get('samples', 0)}개")
                    logger.info(f"  평균 손실: {result.get('avg_loss', 0):.4f}")
                    logger.info(f"  방향 정확도: {result.get('direction_accuracy', 0):.1%}")
                    logger.info(f"  총 학습 횟수: {result.get('training_count', 0)}")

                    if result.get('model_saved'):
                        logger.info(f"  모델 저장: {result['model_saved']}")

        except Exception as e:
            logger.warning(f"AI 주기적 학습 실패: {e}")

    def _save_unified_model_versioned(self):
        """통합 모델 버전관리 저장.

        1. models/unified_ready_{symbol}.pt — latest (라이브 부트용, 항상 덮어쓰기)
        2. models/checkpoints/unified_{symbol}_{timestamp}.pt — 타임스탬프 버전
        3. 오래된 체크포인트 정리 (최근 5개 유지)
        """
        if not self.unified_analyzer:
            return
        try:
            # 1) Latest (라이브 부트 시 로드 대상)
            ready_path = f"models/unified_ready_{self.symbol}.pt"
            self.unified_analyzer.save_model(ready_path)

            # 2) Timestamped checkpoint
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            ckpt_dir = "models/checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = f"{ckpt_dir}/unified_{self.symbol}_{ts}.pt"

            import shutil
            shutil.copy2(ready_path, ckpt_path)
            logger.info(f"💾 통합 모델 저장: {ready_path} + {ckpt_path}")

            # 3) 오래된 체크포인트 정리 (최근 5개 유지)
            import glob
            pattern = f"{ckpt_dir}/unified_{self.symbol}_*.pt"
            ckpts = sorted(glob.glob(pattern))
            if len(ckpts) > 5:
                for old in ckpts[:-5]:
                    try:
                        os.remove(old)
                        logger.debug(f"🗑️ 오래된 체크포인트 삭제: {old}")
                    except OSError:
                        pass
        except Exception as e:
            logger.warning(f"통합 모델 버전 저장 실패: {e}")

    def _manage_position(
        self,
        decision: PositionDecision,
        leverage_rec: LeverageRecommendation,
        current_price: float,
        df=None,
    ):
        """포지션 관리"""
        logger.info("")
        logger.info("💼 [포지션 관리]")

        # 쿨다운 체크
        if self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time).total_seconds()
            if elapsed < self.trade_cooldown:
                logger.info(f"  쿨다운 중 ({self.trade_cooldown - elapsed:.0f}초 남음)")
                return

        # 적응형 신뢰도 임계값 (방향별 차등 적용)
        action = decision.action
        side = decision.side

        # 방향에 따라 베이스 임계값 선택
        if side == 'LONG':
            base_threshold = self.min_confidence_long
        elif side == 'SHORT':
            base_threshold = self.min_confidence_short
        else:
            base_threshold = (self.min_confidence_long + self.min_confidence_short) / 2

        # 액션 타입에 따라 조정
        if action in [PositionAction.OPEN_LONG, PositionAction.OPEN_SHORT]:
            effective_threshold = base_threshold * 0.8  # 진입: 80%
        elif action in [PositionAction.DECREASE_LONG, PositionAction.DECREASE_SHORT,
                        PositionAction.CLOSE_LONG, PositionAction.CLOSE_SHORT]:
            effective_threshold = base_threshold * 0.5  # 청산/축소: 50%
        else:
            effective_threshold = base_threshold * 0.6  # 증액: 60%

        if decision.confidence < effective_threshold:
            logger.info(f"  신뢰도 부족 ({decision.confidence:.1%} < {effective_threshold:.1%})")
            # 신뢰도 부족해도 기존 포지션의 SL/TP는 반드시 체크
            if self.current_position:
                self._check_sl_tp(current_price, df=df)
            return

        # ── 최소 보유 시간 게이트 (SL/TP는 통과, 신호 기반 청산만 제한) ──
        if action in (PositionAction.CLOSE_LONG, PositionAction.CLOSE_SHORT) and self.current_position:
            entry_cycle = self.current_position.get('entry_cycle', 0)
            min_hold = self.current_position.get('min_hold_ticks', 3)
            current_cycle = self._cycle_ctx.get('cycle', 0)
            held_ticks = current_cycle - entry_cycle
            if held_ticks < min_hold:
                logger.info(f"  ⏳ 최소 보유 미달 ({held_ticks}/{min_hold}틱) → HOLD 전환")
                action = PositionAction.HOLD
                decision.action = PositionAction.HOLD

        # 액션 실행
        if action == PositionAction.OPEN_LONG:
            self._open_position('LONG', decision, current_price)

        elif action == PositionAction.OPEN_SHORT:
            self._open_position('SHORT', decision, current_price)

        elif action == PositionAction.CLOSE_LONG:
            self._close_position(current_price, "매도 신호")

        elif action == PositionAction.CLOSE_SHORT:
            self._close_position(current_price, "매수 신호")

        elif action in [PositionAction.INCREASE_LONG, PositionAction.INCREASE_SHORT]:
            self._increase_position(current_price, decision)

        elif action in [PositionAction.DECREASE_LONG, PositionAction.DECREASE_SHORT]:
            self._decrease_position(current_price, decision)

        elif action == PositionAction.HOLD:
            logger.info("  포지션 유지")

        # 액션 종류와 무관하게 포지션 보유 시 항상 SL/TP 체크
        # 단, 방금 진입한 사이클에서는 건너뜀 (진입가 == 현재가 → 오발동 방지)
        if self.current_position and action not in (
            PositionAction.OPEN_LONG, PositionAction.OPEN_SHORT
        ):
            self._check_sl_tp(current_price, df=df)

    def _open_position(self, side: str, decision: PositionDecision, price: float):
        """포지션 진입 — ExecutionManager 경유 스마트 체결

        cascade_conf / fee_ev 기반으로 체결 모드 자동 선택:
        - AGGRESSIVE_TAKER: conf≥0.8 + fee_ev≥3.0 → 시장가
        - PASSIVE_MAKER:    conf 0.5~0.8 + fee_ev≥1.2 → Post-Only 지정가 + OBI 미세조정
        - FALLBACK_MARKET:  기타 → 기본 시장가

        공유 상태 변경은 _lock 보호.
        """
        if self.current_position:
            logger.warning("  이미 포지션이 있습니다")
            return

        # ── Freshness Gate: 추론 중 가격 변동 방어 ──
        if not self._historical_mode and self.data_manager:
            import time as _time_mod
            _STALE_THRESHOLD_PCT = 0.002   # 0.2% = 20 bps
            _STALE_MAX_AGE_S = 5.0
            try:
                fresh_price, age = self.data_manager.get_freshest_price()
                inference_price = getattr(self, '_inference_start_price', price)
                if fresh_price > 0 and age < _STALE_MAX_AGE_S and inference_price > 0:
                    drift_pct = abs(fresh_price - inference_price) / inference_price
                    if drift_pct > _STALE_THRESHOLD_PCT:
                        adverse = (side == 'LONG' and fresh_price > inference_price) or \
                                  (side == 'SHORT' and fresh_price < inference_price)
                        if adverse and drift_pct > _STALE_THRESHOLD_PCT * 2:
                            logger.warning(
                                f"  ⏱️ Freshness Gate: 가격 drift {drift_pct:.3%} "
                                f"(adverse, {inference_price:.2f}→{fresh_price:.2f}) → SKIP"
                            )
                            return
                        else:
                            logger.info(
                                f"  ⏱️ Freshness Gate: 가격 갱신 "
                                f"{price:.2f}→{fresh_price:.2f} (drift={drift_pct:.3%})"
                            )
                            # SL/TP도 가격 변동만큼 조정
                            price_shift = fresh_price - price
                            price = fresh_price
                            if decision.stop_loss and price_shift != 0:
                                decision.stop_loss += price_shift
                            if decision.take_profit and price_shift != 0:
                                decision.take_profit += price_shift
            except Exception as e:
                logger.debug(f"  Freshness Gate 스킵: {e}")

        # 포지션 크기 계산
        position_value = self.current_balance * decision.position_size_pct
        quantity = position_value / price * self.current_leverage

        # 최소 수량 보장: 잔고 부족 시에도 최소 거래 유지 (학습 데이터 확보)
        if quantity < 0.001:
            quantity = 0.001
            position_value = quantity * price / max(self.current_leverage, 1)
            logger.info(f"  ⚠️ 최소 수량 적용: qty=0.001, value={position_value:.2f} USDT")

        logger.info(f"  🚀 {side} 포지션 진입!")
        logger.info(f"     가격: {price:.2f}")
        logger.info(f"     수량: {quantity:.4f}")
        logger.info(f"     레버리지: {self.current_leverage}x")
        logger.info(f"     금액: {position_value:.2f} USDT")

        # 텔레그램 진입 알림
        self.telegram.notify_entry(
            side=side, price=price, size_pct=decision.position_size_pct,
            cascade_conf=self._cycle_ctx.get('cascade_conf') or 0,
            fee_ev=self._cycle_ctx.get('fee_ev_ratio') or 0,
            sl=decision.stop_loss, tp=decision.take_profit,
        )

        # 진입 시점 AI 판단 정보 스냅샷 (기본 AI + cascade 컨텍스트)
        entry_ai_info = dict(self._last_ai_info) if self._last_ai_info else {}
        if self._cycle_ctx:
            entry_ai_info['cycle'] = {
                # Cascade 핵심
                'cascade_dir': self._cycle_ctx.get('cascade_dir'),
                'cascade_conf': self._cycle_ctx.get('cascade_conf'),
                'cascade_fire': self._cycle_ctx.get('cascade_fire'),
                'final_score': self._cycle_ctx.get('final_score'),
                'hold_threshold': self._cycle_ctx.get('hold_threshold'),
                # AI 성숙도
                'ai_accuracy': getattr(self, '_last_ai_accuracy', 0),
                'ai_maturity': getattr(self, '_last_ai_maturity', 0),
                'pred_count': self._cycle_ctx.get('pred_count', 0),
                # 변동성
                'vol_factor': self._cycle_ctx.get('vol_factor'),
                'vol_z': self._cycle_ctx.get('vol_z'),
                # 수수료 EV
                'fee_ev_ratio': self._cycle_ctx.get('fee_ev_ratio'),
                # 레짐
                'regime': self._cycle_ctx.get('regime'),
                'regime_conf': self._cycle_ctx.get('regime_conf'),
                # TA 신호 (참고용)
                'ta_signals': self._cycle_ctx.get('ta_signals', {}),
            }

        # ── ExecutionManager를 통한 스마트 체결 ──
        import time as _time
        bt = self.data_manager.get_book_ticker() if self.data_manager else None
        bt_age = (_time.time() - bt['update_time']) if bt and bt.get('update_time', 0) > 0 else 999.0
        best_bid = bt['bid_price'] if bt and bt.get('update_time', 0) > 0 else price
        best_ask = bt['ask_price'] if bt and bt.get('update_time', 0) > 0 else price

        exec_ctx = ExecutionContext(
            symbol=self.symbol,
            side=side,
            quantity=self.client.round_quantity(self.symbol, quantity) if not self.paper_trading else quantity,
            cascade_conf=self._cycle_ctx.get('cascade_conf') or 0,
            fee_ev_ratio=self._cycle_ctx.get('fee_ev_ratio') or 0,
            obi=self.feature_builder._obi_cache if self.feature_builder else 0.0,
            best_bid=best_bid,
            best_ask=best_ask,
            book_ticker_age=bt_age,
            timing_confidence=self._cycle_ctx.get('timing_confidence') or 0.0,
        )

        exec_result = self.execution_manager.execute(exec_ctx)

        if not exec_result.success:
            logger.warning(f"  ❌ 체결 실패: {exec_result.error}")
            return

        # 체결 결과 로그
        logger.info(f"  ✅ [{exec_result.mode.value}] {side} 체결: "
                     f"{exec_result.filled_quantity:.4f} @ {exec_result.filled_price:.2f}")
        if exec_result.was_chased:
            logger.info(f"     🏃 추격 체결 발생")
        if exec_result.fee_saved_bps > 0:
            logger.info(f"     💰 수수료 절약: ~{exec_result.fee_saved_bps:.1f}bps")

        # 사이클 컨텍스트에 체결 모드 기록
        self._cycle_ctx['execution_mode'] = exec_result.mode.value
        self._cycle_ctx['was_chased'] = exec_result.was_chased

        # 포지션 등록
        # 최소 보유 시간: confidence 기반 3~5틱 가변
        _entry_conf = getattr(decision, 'confidence', 0.5)
        _min_hold = max(3, min(5, 3 + round(_entry_conf * 2)))

        with self._lock:
            self.current_position = {
                'side': side,
                'entry_price': exec_result.filled_price,
                'quantity': exec_result.filled_quantity,
                'leverage': self.current_leverage,
                'stop_loss': decision.stop_loss,
                'take_profit': decision.take_profit,
                'entry_time': datetime.now(),
                'entry_cycle': self._cycle_ctx.get('cycle', 0),
                'min_hold_ticks': _min_hold,
                'pyramid_count': 0,
                'entry_ai_info': entry_ai_info,
                'execution_mode': exec_result.mode.value,
                'order_ids': exec_result.order_ids,
                # Layer 2 WS 트레일링 상태
                '_trail_hwm': exec_result.filled_price,
                '_trail_sl': None,
                '_ws_closing': False,
                'exchange_sl': None,
            }

        if not self.paper_trading:
            # SL/TP 주문 설정
            self._set_sl_tp_orders(decision)

        self.last_trade_time = datetime.now()
        self.stats.total_trades += 1

    def _increase_position(self, price: float, decision: PositionDecision):
        """피라미딩: 기존 포지션에 같은 방향 추가 진입"""
        if not self.current_position:
            logger.warning("  피라미딩 대상 포지션이 없습니다")
            return

        position = self.current_position
        side = position['side']
        old_qty = position['quantity']
        old_entry = position['entry_price']
        pyramid_count = position.get('pyramid_count', 0)

        # 추가 수량 계산
        add_value = self.current_balance * decision.position_size_pct
        add_qty = add_value / price * self.current_leverage

        if add_qty < 0.001:
            logger.warning("  피라미딩 수량이 너무 작습니다")
            return

        logger.info(f"  📐 {side} 피라미딩 #{pyramid_count+1}!")
        logger.info(f"     가격: {price:.2f}")
        logger.info(f"     추가 수량: {add_qty:.4f}")
        logger.info(f"     추가 금액: {add_value:.2f} USDT")

        if self.paper_trading:
            # 평균 진입가 재계산
            new_qty = old_qty + add_qty
            new_entry = (old_entry * old_qty + price * add_qty) / new_qty
            position['quantity'] = new_qty
            position['entry_price'] = new_entry
            position['pyramid_count'] = pyramid_count + 1

            # SL을 손익분기점으로 이동 (피라미딩 시 리스크 관리)
            if side == 'LONG':
                breakeven_sl = new_entry * 0.998  # 0.2% 아래
                if position.get('stop_loss') and breakeven_sl > position['stop_loss']:
                    position['stop_loss'] = breakeven_sl
            else:
                breakeven_sl = new_entry * 1.002  # 0.2% 위
                if position.get('stop_loss') and breakeven_sl < position['stop_loss']:
                    position['stop_loss'] = breakeven_sl

            logger.info(f"  📝 [페이퍼] 피라미딩 완료: 평균가 {new_entry:.2f}, 총 수량 {new_qty:.4f}")
        else:
            try:
                order_side = 'BUY' if side == 'LONG' else 'SELL'
                result = self.client.place_market_order(
                    symbol=self.symbol,
                    side=order_side,
                    quantity=self.client.round_quantity(self.symbol, add_qty)
                )
                logger.info(f"  ✅ 피라미딩 주문 체결: {result.get('orderId')}")

                exec_qty = float(result.get('executedQty', add_qty))
                exec_price = float(result.get('avgPrice', price))
                new_qty = old_qty + exec_qty
                new_entry = (old_entry * old_qty + exec_price * exec_qty) / new_qty
                position['quantity'] = new_qty
                position['entry_price'] = new_entry
                position['pyramid_count'] = pyramid_count + 1

                # SL을 손익분기점으로 이동
                if side == 'LONG':
                    breakeven_sl = new_entry * 0.998
                    if position.get('stop_loss') and breakeven_sl > position['stop_loss']:
                        position['stop_loss'] = breakeven_sl
                else:
                    breakeven_sl = new_entry * 1.002
                    if position.get('stop_loss') and breakeven_sl < position['stop_loss']:
                        position['stop_loss'] = breakeven_sl

            except Exception as e:
                logger.error(f"  ❌ 피라미딩 주문 실패: {e}")
                return

        self.last_trade_time = datetime.now()

    # ─── Kelly Criterion ────────────────────────────────────────

    def _update_kelly(self):
        """거래 완료 후 Kelly 최적 비율 업데이트 (EMA smoothing)."""
        recent = self.trade_history[-self._kelly_window:]
        if len(recent) < 20:
            return  # 최소 20건은 있어야 유의미

        wins = [t['pnl'] for t in recent if t.get('pnl', 0) > 0]
        losses = [t['pnl'] for t in recent if t.get('pnl', 0) < 0]

        if not wins or not losses:
            return

        p = len(wins) / len(recent)       # 승률
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        b = avg_win / avg_loss if avg_loss > 0 else 1.0  # 수익/손실 비

        # Kelly formula: f* = (p*b - q) / b, where q = 1-p
        kelly_raw = (p * b - (1 - p)) / b if b > 0 else 0.0
        kelly_raw = max(0.0, kelly_raw)  # 음수면 0 (베팅하지 않음)

        # Half-Kelly (안전 마진)
        kelly_half = kelly_raw * 0.5

        # EMA smoothing (급변 방지)
        self._kelly_smoothed = (
            self._kelly_ema_alpha * kelly_half +
            (1 - self._kelly_ema_alpha) * self._kelly_smoothed
        )

        # Floor + Hard cap
        self._kelly_smoothed = max(self._kelly_min_floor, self._kelly_smoothed)
        self._kelly_smoothed = min(self._kelly_smoothed, self._kelly_max_cap)

        logger.info(
            f"    📊 Kelly: p={p:.2f}, b={b:.2f}, "
            f"raw={kelly_raw:.3f}, half={kelly_half:.3f}, "
            f"smoothed={self._kelly_smoothed:.3f} ({len(recent)}건)"
        )

    def _get_kelly_size(self) -> float:
        """현재 Kelly 기반 포지션 사이즈 비율 반환."""
        return self._kelly_smoothed

    def _close_position(self, price: float, reason: str):
        """포지션 청산"""
        if not self.current_position:
            logger.warning("  청산할 포지션이 없습니다")
            return

        position = self.current_position
        side = position['side']
        entry_price = position['entry_price']
        quantity = position['quantity']

        # PnL 계산 (수수료 포함)
        if side == 'LONG':
            gross_pnl = (price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - price) * quantity

        # 왕복 수수료: 진입 + 청산
        fee = (entry_price + price) * quantity * self.trading_fee_rate
        pnl = gross_pnl - fee

        margin = entry_price * quantity / position['leverage']
        pnl_pct = pnl / margin * 100 if margin > 0 else 0.0

        logger.info(f"  💰 {side} 포지션 청산!")
        logger.info(f"     진입가: {entry_price:.2f}")
        logger.info(f"     청산가: {price:.2f}")
        logger.info(f"     총손익: {gross_pnl:.2f} → 수수료: {fee:.2f} → 순손익: {pnl:.2f} USDT ({pnl_pct:.2f}%)")
        logger.info(f"     사유: {reason}")

        # 텔레그램 청산 알림
        self.telegram.notify_exit(
            side=side, entry_price=entry_price, exit_price=price,
            pnl=pnl, pnl_pct=pnl_pct / 100, reason=reason,
        )

        if self.paper_trading:
            logger.info("  📝 [페이퍼 트레이딩] 청산 기록됨")
        else:
            try:
                result = self.client.close_position(self.symbol)
                logger.info(f"  ✅ 청산 완료: {result}")
            except Exception as e:
                from api.futures_client import BinanceAPIError
                if isinstance(e, BinanceAPIError) and e.is_position_not_found:
                    logger.warning(f"  ⚠️ 거래소에 포지션 없음 — 로컬 상태 정리: {e.message}")
                else:
                    logger.error(f"  ❌ 청산 실패: {e}")
                    return

            # 잔여 거래소 주문 취소 (orphaned SL 방지)
            try:
                self.client.cancel_all_orders(self.symbol)
            except Exception as e:
                logger.warning(f"  ⚠️ 잔여 주문 취소 실패 (non-critical): {e}")

        # 통계 업데이트 (lock: get_status 등 외부 스레드 읽기 보호)
        with self._lock:
            self.current_balance += pnl
            self.daily_pnl += pnl

            if pnl > 0:
                self.stats.winning_trades += 1
                self.stats.consecutive_losses = 0
                if pnl > self.stats.best_trade:
                    self.stats.best_trade = pnl
            else:
                self.stats.losing_trades += 1
                self.stats.consecutive_losses += 1
                if pnl < self.stats.worst_trade:
                    self.stats.worst_trade = pnl

            self.stats.total_pnl += pnl

            # 드로다운 업데이트
            drawdown = (self.initial_balance - self.current_balance) / self.initial_balance
            if drawdown > self.stats.max_drawdown:
                self.stats.max_drawdown = drawdown

        # 10s drift 감지: 거래 결과 기록
        if self.hybrid_analyzer:
            self.hybrid_analyzer.drift_detector.record_outcome(pnl > 0)

        # 거래 기록 (진입/청산 시점 AI 판단 정보 포함)
        exit_ai_info = dict(self._last_ai_info) if self._last_ai_info else {}
        if self._cycle_ctx:
            exit_ai_info['cycle'] = {
                # Cascade 핵심
                'cascade_dir': self._cycle_ctx.get('cascade_dir'),
                'cascade_conf': self._cycle_ctx.get('cascade_conf'),
                'cascade_fire': self._cycle_ctx.get('cascade_fire'),
                'final_score': self._cycle_ctx.get('final_score'),
                'hold_threshold': self._cycle_ctx.get('hold_threshold'),
                # AI 성숙도
                'ai_accuracy': getattr(self, '_last_ai_accuracy', 0),
                'ai_maturity': getattr(self, '_last_ai_maturity', 0),
                'pred_count': self._cycle_ctx.get('pred_count', 0),
                # 변동성
                'vol_factor': self._cycle_ctx.get('vol_factor'),
                'vol_z': self._cycle_ctx.get('vol_z'),
                # 수수료 EV
                'fee_ev_ratio': self._cycle_ctx.get('fee_ev_ratio'),
                # 레짐
                'regime': self._cycle_ctx.get('regime'),
                'regime_conf': self._cycle_ctx.get('regime_conf'),
                # TA 신호 (참고용)
                'ta_signals': self._cycle_ctx.get('ta_signals', {}),
            }
        self.trade_history.append({
            'side': side,
            'entry_price': entry_price,
            'exit_price': price,
            'quantity': quantity,
            'leverage': position['leverage'],
            'gross_pnl': gross_pnl,
            'fee': fee,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'balance_after': self.current_balance,
            'entry_ai_info': position.get('entry_ai_info', {}),
            'exit_ai_info': exit_ai_info,
        })

        with self._lock:
            self.current_position = None
        self.last_trade_time = datetime.now()

        # Kelly Criterion 업데이트
        self._update_kelly()

        # 대시보드에 거래 내역 즉시 기록
        self.state_writer.write_trade_history(self.trade_history)

    def _decrease_position(self, current_price: float, decision: PositionDecision):
        """포지션 50% 축소"""
        if not self.current_position:
            logger.warning("  축소할 포지션이 없습니다")
            return

        position = self.current_position
        side = position['side']
        entry_price = position['entry_price']
        original_qty = position['quantity']
        reduce_qty = original_qty * 0.5  # 50% 축소

        # 축소분 PnL 계산 (수수료 포함)
        if side == 'LONG':
            gross_pnl = (current_price - entry_price) * reduce_qty
        else:
            gross_pnl = (entry_price - current_price) * reduce_qty

        # 축소분 수수료: 진입시 해당 비중 + 청산
        fee = (entry_price + current_price) * reduce_qty * self.trading_fee_rate
        pnl = gross_pnl - fee

        margin = entry_price * reduce_qty / position['leverage']
        pnl_pct = pnl / margin * 100 if margin > 0 else 0.0

        logger.info(f"  📉 {side} 포지션 50% 축소!")
        logger.info(f"     축소 수량: {reduce_qty:.4f} → 잔여: {original_qty - reduce_qty:.4f}")
        logger.info(f"     축소분: 총{gross_pnl:.2f} → 수수료{fee:.2f} → 순{pnl:.2f} USDT ({pnl_pct:.2f}%)")

        if self.paper_trading:
            logger.info("  📝 [페이퍼 트레이딩] 포지션 축소 기록됨")
        else:
            try:
                close_side = 'SELL' if side == 'LONG' else 'BUY'
                result = self.client.place_market_order(
                    symbol=self.symbol,
                    side=close_side,
                    quantity=self.client.round_quantity(self.symbol, reduce_qty)
                )
                logger.info(f"  ✅ 축소 체결: {result.get('orderId')}")
            except Exception as e:
                logger.error(f"  ❌ 축소 실패: {e}")
                return

        # 잔고 및 통계 업데이트
        self.current_balance += pnl
        self.daily_pnl += pnl
        self.stats.total_pnl += pnl

        # 포지션 수량 갱신
        self.current_position['quantity'] = original_qty - reduce_qty

        # 남은 수량이 너무 작으면 전체 청산
        if self.current_position['quantity'] < 0.001:
            self.current_position = None
            logger.info("  잔여 수량 부족 → 포지션 완전 청산")
            # 거래소 SL 주문도 취소
            if not self.paper_trading and not self._historical_mode:
                try:
                    self.client.cancel_all_orders(self.symbol)
                except Exception:
                    pass
        elif not self.paper_trading and not self._historical_mode:
            # 부분청산: 거래소 SL 수량 동기화
            self._update_exchange_sl(
                self.LAYER1_SL_ATR_TIGHT if self._ws_sl_mode == 'tight' else self.LAYER1_SL_ATR_NORMAL
            )

        self.last_trade_time = datetime.now()

        # 거래 기록 (축소 거래)
        self.trade_history.append({
            'side': side,
            'entry_price': entry_price,
            'exit_price': current_price,
            'quantity': reduce_qty,
            'leverage': position['leverage'],
            'gross_pnl': gross_pnl,
            'fee': fee,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': '포지션 축소 (AI 권고)',
            'entry_time': position['entry_time'],
            'exit_time': datetime.now()
        })

        self.state_writer.write_trade_history(self.trade_history)

    # ══════════════════════════════════════════════════════════════
    # Layer 2: WebSocket markPrice 트레일링 스톱 (전략 레이어)
    # ══════════════════════════════════════════════════════════════

    def _on_mark_price_update(self, price: float):
        """Layer 2: WS markPrice 틱마다 호출 (~1초, WS 스레드).

        고점 워터마크 갱신 + 트레일링 SL 체크만 수행.
        AI 추론, 부분청산 등 무거운 판단은 메인 루프(Layer 2.5)에서 처리.
        """
        if self._historical_mode:
            return

        should_close = False

        with self._lock:
            position = self.current_position
            if position is None:
                return
            if position.get('_ws_closing'):
                return

            side = position['side']
            entry_price = position['entry_price']
            atr = self._current_atr

            if atr <= 0 or entry_price <= 0:
                return

            activation_dist = atr * self.LAYER2_ACTIVATION_ATR
            offset_dist = atr * self.LAYER2_OFFSET_ATR

            if side == 'LONG':
                profit = price - entry_price
                current_hwm = position.get('_trail_hwm', entry_price)
                new_hwm = max(current_hwm, price)
                position['_trail_hwm'] = new_hwm

                if profit >= activation_dist:
                    candidate_sl = new_hwm - offset_dist
                    current_trail_sl = position.get('_trail_sl') or 0.0
                    if candidate_sl > current_trail_sl:
                        position['_trail_sl'] = candidate_sl

                    trail_sl = position.get('_trail_sl') or 0.0
                    if trail_sl > 0 and price <= trail_sl:
                        logger.info(
                            f"  [L2-WS] Trailing Stop LONG: "
                            f"price={price:.2f} <= trail_sl={trail_sl:.2f} "
                            f"(hwm={new_hwm:.2f}, atr={atr:.2f})"
                        )
                        position['_ws_closing'] = True
                        should_close = True

            else:  # SHORT
                profit = entry_price - price
                current_hwm = position.get('_trail_hwm', entry_price)
                new_hwm = min(current_hwm, price)
                position['_trail_hwm'] = new_hwm

                if profit >= activation_dist:
                    candidate_sl = new_hwm + offset_dist
                    current_trail_sl = position.get('_trail_sl') or float('inf')
                    if candidate_sl < current_trail_sl:
                        position['_trail_sl'] = candidate_sl

                    trail_sl = position.get('_trail_sl')
                    if trail_sl is not None and trail_sl < float('inf') and price >= trail_sl:
                        logger.info(
                            f"  [L2-WS] Trailing Stop SHORT: "
                            f"price={price:.2f} >= trail_sl={trail_sl:.2f} "
                            f"(hwm={new_hwm:.2f}, atr={atr:.2f})"
                        )
                        position['_ws_closing'] = True
                        should_close = True

        # lock 밖에서 청산 (RLock이지만 중첩 회피)
        if should_close:
            self._close_position(price, "Layer2 WS Trailing Stop")

    # ══════════════════════════════════════════════════════════════
    # Layer 1: WS 연결 상태 변화 → 거래소 SL 조정
    # ══════════════════════════════════════════════════════════════

    def _on_ws_disconnect(self):
        """WS 단절 시 거래소 SL을 1.0 ATR로 타이트하게."""
        if self.paper_trading or self._historical_mode:
            return
        logger.warning("[L1] WS 단절 — 거래소 SL 타이트닝 (1.0 ATR)")
        self._ws_sl_mode = 'tight'
        self._update_exchange_sl(self.LAYER1_SL_ATR_TIGHT)

    def _on_ws_reconnect(self):
        """WS 재연결 시 거래소 SL을 1.8 ATR로 복원."""
        if self.paper_trading or self._historical_mode:
            return
        logger.info("[L1] WS 재연결 — 거래소 SL 복원 (1.8 ATR)")
        self._ws_sl_mode = 'normal'
        self._update_exchange_sl(self.LAYER1_SL_ATR_NORMAL)

    def _update_exchange_sl(self, atr_multiplier: float):
        """거래소 SL 주문 취소 후 새 레벨로 재설정."""
        if self.paper_trading or self._historical_mode:
            return

        with self._lock:
            position = self.current_position
            if position is None:
                return
            side = position['side']
            entry_price = position['entry_price']
            quantity = position['quantity']

        atr = self._current_atr
        if atr <= 0:
            logger.warning("[L1] ATR 없음 — 거래소 SL 갱신 불가")
            return

        hard_sl = self.position_manager._calculate_hard_sl(
            price=entry_price,
            side=side,
            atr=atr,
            leverage=self.current_leverage,
            atr_multiplier=atr_multiplier,
        )
        if hard_sl is None:
            return

        try:
            self.client.cancel_all_orders(self.symbol)
            sl_side = 'SELL' if side == 'LONG' else 'BUY'
            self.client.place_stop_market_order(
                symbol=self.symbol,
                side=sl_side,
                quantity=self.client.round_quantity(self.symbol, quantity),
                stop_price=hard_sl,
                reduce_only=True,
            )
            logger.info(f"  [L1] 거래소 SL 갱신: {hard_sl:.2f} ({atr_multiplier}x ATR)")

            with self._lock:
                if self.current_position:
                    self.current_position['exchange_sl'] = hard_sl

        except Exception as e:
            logger.error(f"[L1] 거래소 SL 갱신 실패: {e}")

    # ══════════════════════════════════════════════════════════════
    # Layer 2.5: 메인 루프 SL/TP 체크 (안전망 + TP)
    # ══════════════════════════════════════════════════════════════

    def _check_sl_tp(self, current_price: float, df=None):
        """Layer 2.5: 메인 루프 SL/TP 안전망 (5~10초 주기).

        트레일링 워터마크 갱신은 _on_mark_price_update(Layer 2)가 담당.
        이 메서드는:
        - _trail_sl (WS가 세팅)이 있으면 그 값으로 SL 체크
        - 없으면 원래 strategy stop_loss로 체크
        - TP 체크 (내부 관리)
        - 백테스트: intrabar high/low로 SL/TP 터치 판정 (close보다 정확)
        - 백테스트: Layer 2 trailing stop 시뮬레이션 (HWM + ATR 기반)
        """
        if not self.current_position:
            return

        position = self.current_position
        side = position['side']
        entry_price = position['entry_price']
        strategy_sl = position.get('stop_loss')
        take_profit = position.get('take_profit')

        # 백테스트: intrabar high/low 사용 (캔들 내부 가격 변동 반영)
        candle_low = current_price
        candle_high = current_price
        if self._historical_mode and df is not None and len(df) > 0:
            candle_low = df['low'].iloc[-1]
            candle_high = df['high'].iloc[-1]

        # ── 백테스트 trailing stop 시뮬레이션 (Layer 2 대체) ──
        if self._historical_mode:
            atr = self._current_atr
            if atr > 0:
                activation_dist = atr * self.LAYER2_ACTIVATION_ATR  # 0.8 × ATR
                offset_dist = atr * self.LAYER2_OFFSET_ATR          # 0.4 × ATR

                if side == 'LONG':
                    hwm = max(position.get('_trail_hwm', entry_price), candle_high)
                    position['_trail_hwm'] = hwm
                    if hwm - entry_price >= activation_dist:
                        candidate_sl = hwm - offset_dist
                        current_trail = position.get('_trail_sl') or 0.0
                        if candidate_sl > current_trail:
                            position['_trail_sl'] = candidate_sl
                else:  # SHORT
                    hwm = min(position.get('_trail_hwm', entry_price), candle_low)
                    position['_trail_hwm'] = hwm
                    if entry_price - hwm >= activation_dist:
                        candidate_sl = hwm + offset_dist
                        current_trail = position.get('_trail_sl') or float('inf')
                        if candidate_sl < current_trail:
                            position['_trail_sl'] = candidate_sl

        trail_sl = position.get('_trail_sl')

        # 유효 SL: 트레일링이 활성화됐으면 그 값, 아니면 전략 SL
        effective_sl = trail_sl if trail_sl is not None else strategy_sl

        if side == 'LONG':
            sl_touched = bool(effective_sl and
                              (candle_low if self._historical_mode else current_price) <= effective_sl)
            tp_touched = bool(take_profit and
                              (candle_high if self._historical_mode else current_price) >= take_profit)

            if sl_touched and tp_touched and self._historical_mode:
                # 양쪽 다 터치 — entry로부터 거리 비교 (가까운 쪽이 먼저 체결)
                sl_dist = entry_price - effective_sl
                tp_dist = take_profit - entry_price
                if tp_dist <= sl_dist:
                    sl_touched = False   # TP가 더 가깝거나 같으면 TP 우선
                else:
                    tp_touched = False   # SL이 더 가까우면 SL 우선

            if sl_touched:
                reason = "Layer2.5 Trailing Stop" if trail_sl is not None else "Stop Loss"
                exit_price = effective_sl if self._historical_mode else current_price
                logger.info(f"  [L2.5] {reason}: low={candle_low:.2f} <= sl={effective_sl:.2f}")
                self._close_position(exit_price, reason)
                return
            if tp_touched:
                exit_price = take_profit if self._historical_mode else current_price
                logger.info(f"  🎯 Take Profit 도달! high={candle_high:.2f} >= tp={take_profit:.2f}")
                self._close_position(exit_price, "Take Profit")
                return

        else:  # SHORT
            sl_touched = bool(effective_sl and
                              (candle_high if self._historical_mode else current_price) >= effective_sl)
            tp_touched = bool(take_profit and
                              (candle_low if self._historical_mode else current_price) <= take_profit)

            if sl_touched and tp_touched and self._historical_mode:
                sl_dist = effective_sl - entry_price
                tp_dist = entry_price - take_profit
                if tp_dist <= sl_dist:
                    sl_touched = False
                else:
                    tp_touched = False

            if sl_touched:
                reason = "Layer2.5 Trailing Stop" if trail_sl is not None else "Stop Loss"
                exit_price = effective_sl if self._historical_mode else current_price
                logger.info(f"  [L2.5] {reason}: high={candle_high:.2f} >= sl={effective_sl:.2f}")
                self._close_position(exit_price, reason)
                return
            if tp_touched:
                exit_price = take_profit if self._historical_mode else current_price
                logger.info(f"  🎯 Take Profit 도달! low={candle_low:.2f} <= tp={take_profit:.2f}")
                self._close_position(exit_price, "Take Profit")
                return

    def _set_sl_tp_orders(self, decision: PositionDecision):
        """Layer 1: 거래소 하드 SL 설정 (생존 레이어, catastrophic protection only).

        TP는 거래소에 걸지 않음 (내부 관리 — 생존 문제가 아님).
        SL은 1.8 ATR로 넓게 설정 → WS 트레일링(Layer 2)이 먼저 반응.
        """
        if self.paper_trading or not self.current_position:
            return

        side = self.current_position['side']
        entry_price = self.current_position['entry_price']
        quantity = self.current_position['quantity']
        atr = self._current_atr

        # 하드 SL 계산 (1.8 ATR)
        if atr > 0:
            hard_sl = self.position_manager._calculate_hard_sl(
                price=entry_price,
                side=side,
                atr=atr,
                leverage=self.current_leverage,
                atr_multiplier=self.LAYER1_SL_ATR_NORMAL,
            )
        else:
            # ATR 미산출 시 전략 SL로 폴백
            hard_sl = decision.stop_loss
            logger.warning("[L1] ATR 미산출 — 전략 SL로 폴백")

        if hard_sl is None:
            logger.warning("[L1] hard SL 계산 불가 — 거래소 SL 미설정")
            return

        try:
            sl_side = 'SELL' if side == 'LONG' else 'BUY'
            self.client.place_stop_market_order(
                symbol=self.symbol,
                side=sl_side,
                quantity=self.client.round_quantity(self.symbol, quantity),
                stop_price=hard_sl,
                reduce_only=True
            )
            logger.info(f"  [L1] 거래소 하드 SL 설정: {hard_sl:.2f} ({self.LAYER1_SL_ATR_NORMAL}x ATR)")

            with self._lock:
                if self.current_position:
                    self.current_position['exchange_sl'] = hard_sl

        except Exception as e:
            logger.error(f"[L1] 거래소 SL 설정 실패: {e}")

    def _check_risk_limits(self) -> bool:
        """리스크 한도 체크 — 일일 손실 + 연속 손실 + 드로다운"""
        # 1) 일일 손실 한도 (5%)
        if self.initial_balance > 0:
            daily_loss_pct = abs(min(0, self.daily_pnl)) / self.initial_balance
            if daily_loss_pct >= self.max_daily_loss_pct:
                logger.warning(f"⚠️ 일일 손실 한도 도달 ({daily_loss_pct:.1%}/{self.max_daily_loss_pct:.1%}) — 거래 중지")
                return False

        # 2) 최대 드로다운 (15%)
        drawdown = (self.initial_balance - self.current_balance) / self.initial_balance
        if drawdown >= self.max_drawdown_pct:
            logger.warning(f"⚠️ 최대 드로다운 도달 ({drawdown:.1%}) — 거래 중지")
            return False

        # 3) 연속 손실 circuit breaker (7회 이상 → 포지션 크기 50% 감소, 10회 → 중지)
        if self.stats.consecutive_losses >= 10:
            logger.warning(f"⚠️ 연속 손실 {self.stats.consecutive_losses}회 — 거래 중지 (circuit breaker)")
            return False
        elif self.stats.consecutive_losses >= 7:
            logger.warning(f"⚠️ 연속 손실 {self.stats.consecutive_losses}회 — 포지션 크기 50% 축소")

        return True

    def _check_daily_reset(self):
        """일일 리셋 체크"""
        today = datetime.now().date()
        if today > self.daily_reset_date:
            logger.info(f"📅 일일 리셋 (어제 손익: {self.daily_pnl:.2f} USDT)")
            self.daily_pnl = 0.0
            self.daily_reset_date = today

    def _calculate_wait_time(self) -> float:
        """다음 사이클까지 대기 시간 계산 (코인 시장 민첩 대응)"""
        # 기본 대기 시간: 10초 (빠른 거래 확인)
        base_wait = 10

        # 변동성에 따라 추가 조정 (높은 변동성 = 더 자주 체크)
        if self.data_manager:
            df = self.data_manager.get_candles_df()
            if len(df) >= 20:
                volatility = self.leverage_optimizer.calculate_volatility_score(df)
                if volatility > 0.7:
                    base_wait = 5   # 고변동성: 5초
                elif volatility > 0.4:
                    base_wait = 7   # 중변동성: 7초
                # 저변동성: 기본 10초 유지

        return base_wait

    def _on_account_update(self, data: Dict):
        """계정 업데이트 콜백"""
        logger.debug(f"계정 업데이트: {data}")

    def _on_order_update(self, data: Dict):
        """주문 업데이트 콜백"""
        logger.info(f"주문 업데이트: {data}")

    def _print_status(self):
        """현재 상태 출력"""
        logger.info("")
        logger.info("📈 [현재 상태]")
        logger.info(f"  잔고: {self.current_balance:.2f} USDT")
        logger.info(f"  총 손익: {self.stats.total_pnl:.2f} USDT ({self.stats.total_pnl/self.initial_balance*100:.2f}%)")
        logger.info(f"  일일 손익: {self.daily_pnl:.2f} USDT")
        logger.info(f"  기본 레버리지: {self.default_leverage}x / AI 조정 레버리지: {self.current_leverage}x")

        if self.current_position:
            pos = self.current_position
            current_price = self.data_manager.get_current_price() if self.data_manager else 0
            if pos['side'] == 'LONG':
                unrealized_pnl = (current_price - pos['entry_price']) * pos['quantity']
            else:
                unrealized_pnl = (pos['entry_price'] - current_price) * pos['quantity']

            logger.info(f"  포지션: {pos['side']} {pos['quantity']:.4f} @ {pos['entry_price']:.2f}")
            logger.info(f"  미실현 손익: {unrealized_pnl:.2f} USDT")
        else:
            logger.info("  포지션: 없음")

        logger.info(f"  거래 횟수: {self.stats.total_trades}")
        if self.stats.total_trades > 0:
            win_rate = self.stats.winning_trades / self.stats.total_trades * 100
            logger.info(f"  승률: {win_rate:.1f}%")
        logger.info(f"  연속 손실: {self.stats.consecutive_losses}")
        logger.info(f"  최대 드로다운: {self.stats.max_drawdown:.1%}")

        # AI 학습 상태
        if self.unified_analyzer:
            self._print_unified_diagnostics()
        else:
            _status_analyzer = self._get_analyzer_for_tf('1m') if self.use_ai_model else None
            if _status_analyzer:
                training_stats = _status_analyzer.get_training_stats()
                logger.info(f"  🧠 AI 학습 버퍼: {training_stats['buffer_size']}개")
                logger.info(f"  🧠 AI 학습 횟수: {training_stats['training_count']}")
                pred_acc = training_stats.get('prediction_accuracy', {})
                if pred_acc.get('direction_accuracy', 0) > 0:
                    logger.info(f"  🧠 AI 예측 정확도: {pred_acc['direction_accuracy']:.1%}")
        logger.info("")

    def _print_unified_diagnostics(self):
        """통합 모델 진단: 출력 분산 + 라우터 분포 + 결정 분포."""
        import numpy as _np
        import math as _math

        ua = self.unified_analyzer
        hist = self._unified_pred_history
        n = len(hist)

        logger.info(f"  🧠 [통합 모델 진단] (최근 {n}건)")
        logger.info(f"    학습 버퍼: {len(ua.training_buffer)}/{ua.training_buffer.maxlen}, "
                     f"학습 횟수: {ua._training_count}")

        if n < 5:
            return

        # 1) 결정 분포 (최근 N건 중 UP/DOWN/NEUTRAL 비율)
        dir_counts = {'UP': 0, 'DOWN': 0, 'NEUTRAL': 0}
        for h in hist:
            dir_counts[h['dir']] = dir_counts.get(h['dir'], 0) + 1
        logger.info(
            f"    결정분포: UP={dir_counts.get('UP',0)}/{n} "
            f"DOWN={dir_counts.get('DOWN',0)}/{n} "
            f"NEUTRAL={dir_counts.get('NEUTRAL',0)}/{n}"
        )

        # 2) 출력 분산 (direction probs의 std — 핵심 지표)
        probs = _np.array([h['probs'] for h in hist])
        prob_mean = probs.mean(axis=0)
        prob_std = probs.std(axis=0)
        logger.info(
            f"    출력분포: S={prob_mean[0]:.3f}±{prob_std[0]:.3f}, "
            f"H={prob_mean[1]:.3f}±{prob_std[1]:.3f}, "
            f"L={prob_mean[2]:.3f}±{prob_std[2]:.3f}"
        )
        total_std = prob_std.mean()
        if total_std < 0.02:
            logger.warning(f"    ⚠️ 출력 분산 극히 낮음 ({total_std:.4f}) — 모델이 입력에 반응하지 않음")

        # 3) LoRA 진단
        _lora_n = sum(p.data.norm().item() for p in ua.model.online_lora.parameters())
        logger.info(f"    LoRA norm: {_lora_n:.4f}")

    def _print_final_report(self):
        """최종 리포트 출력"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 최종 거래 리포트")
        logger.info("=" * 70)
        logger.info(f"  초기 자본: {self.initial_balance:.2f} USDT")
        logger.info(f"  최종 자본: {self.current_balance:.2f} USDT")
        logger.info(f"  총 손익: {self.stats.total_pnl:.2f} USDT ({self.stats.total_pnl/self.initial_balance*100:.2f}%)")
        logger.info(f"  총 거래 횟수: {self.stats.total_trades}")

        if self.stats.total_trades > 0:
            win_rate = self.stats.winning_trades / self.stats.total_trades * 100
            logger.info(f"  승률: {win_rate:.1f}% ({self.stats.winning_trades}승 {self.stats.losing_trades}패)")

        logger.info(f"  최대 드로다운: {self.stats.max_drawdown:.1%}")
        logger.info(f"  최고 수익 거래: {self.stats.best_trade:.2f} USDT")
        logger.info(f"  최악 손실 거래: {self.stats.worst_trade:.2f} USDT")

        # AI 학습 통계 — MTF 모드에서는 실제 활성 analyzer 참조
        _report_analyzer = self._get_analyzer_for_tf('1m') if self.use_ai_model else None
        if _report_analyzer:
            logger.info("-" * 70)
            logger.info("🧠 AI 학습 통계")
            training_stats = _report_analyzer.get_training_stats()
            logger.info(f"  총 학습 횟수: {training_stats['training_count']}")
            logger.info(f"  학습 버퍼 크기: {training_stats['buffer_size']}개")
            pred_acc = training_stats.get('prediction_accuracy', {})
            logger.info(f"  최종 방향 예측 정확도: {pred_acc.get('direction_accuracy', 0):.1%}")
            logger.info(f"  최종 가격 MAE: {pred_acc.get('price_mae', 0):.2f}")
        logger.info("=" * 70)

    def _collect_model_stats(self) -> Dict:
        """모델별/시간대별 학습 통계 수집 (대시보드 모델 탭용)"""
        result = {'timeframes': {}, 'tick': None, 'unified': None}
        try:

            # 통합 모델 통계
            if self.unified_analyzer:
                ua = self.unified_analyzer
                sv_status = ua.shadow_validator.get_status()
                _lora_norm = 0.0
                try:
                    _lora_norm = sum(p.data.norm().item() for p in ua.model.online_lora.parameters())
                except Exception:
                    pass
                result['unified'] = {
                    'shadow_state': sv_status['state'],
                    'shadow_resolved': sv_status['resolved_count'],
                    'shadow_pending': sv_status['pending_count'],
                    'shadow_accuracy': sv_status['recent_accuracy'],
                    'shadow_profit_factor': sv_status['recent_profit_factor'],
                    'shadow_promotions': sv_status['promotions'],
                    'shadow_demotions': sv_status['demotions'],
                    'lora_norm': round(_lora_norm, 4),
                }

            # PriceLog 통계
            result['price_log_size'] = len(self.price_log) if self.price_log else 0

        except Exception:
            pass
        return result

    def get_status(self) -> Dict:
        """현재 상태 반환 (대시보드 포함) — 스레드 안전"""
        with self._lock:
            # 미실현 손익 계산
            unrealized_pnl = 0.0
            current_price = 0.0
            if self.data_manager:
                current_price = self.data_manager.get_current_price()
            if self.current_position and current_price:
                pos = self.current_position
                if pos['side'] == 'LONG':
                    unrealized_pnl = (current_price - pos['entry_price']) * pos['quantity']
                else:
                    unrealized_pnl = (pos['entry_price'] - current_price) * pos['quantity']

            return {
                'state': self.state.value,
                'symbol': self.symbol,
                'initial_balance': self.initial_balance,
                'current_balance': self.current_balance,
                'total_pnl': self.stats.total_pnl,
                'daily_pnl': self.daily_pnl,
                'default_leverage': self.default_leverage,
                'current_leverage': self.current_leverage,
                'current_position': dict(self.current_position) if self.current_position else None,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'stats': {
                    'total_trades': self.stats.total_trades,
                    'winning_trades': self.stats.winning_trades,
                    'losing_trades': self.stats.losing_trades,
                    'win_rate': self.stats.winning_trades / self.stats.total_trades * 100 if self.stats.total_trades > 0 else 0,
                    'max_drawdown': self.stats.max_drawdown,
                    'consecutive_losses': self.stats.consecutive_losses,
                    'best_trade': self.stats.best_trade,
                    'worst_trade': self.stats.worst_trade
                },
                'ai_info': dict(self._last_ai_info) if self._last_ai_info else None,
                'cycle_ctx': {
                    'use_unified_model': True,
                    'cascade_dir': self._cycle_ctx.get('cascade_dir'),
                    'cascade_conf': self._cycle_ctx.get('cascade_conf', 0),
                    'cascade_fire': self._cycle_ctx.get('cascade_fire', False),
                    'fee_ev_ratio': self._cycle_ctx.get('fee_ev_ratio') or 0,
                    'final_score': self._cycle_ctx.get('final_score') or 0,
                    'hold_threshold': self._cycle_ctx.get('hold_threshold', 0),
                    'regime': self._cycle_ctx.get('regime'),
                    'regime_conf': self._cycle_ctx.get('regime_conf'),
                    'action': self._cycle_ctx.get('action'),
                    'execution_mode': self._cycle_ctx.get('execution_mode'),
                    'vol_factor': self._cycle_ctx.get('vol_factor'),
                    'vol_z': self._cycle_ctx.get('vol_z'),
                    'ai_maturity': self._cycle_ctx.get('ai_maturity'),
                    'ai_accuracy': self._cycle_ctx.get('ai_accuracy'),
                    'pred_count': self._cycle_ctx.get('pred_count'),
                } if self._cycle_ctx else None,
                'data_status': self.data_manager.get_status() if self.data_manager else None,
                'missed_opportunities': list(self._missed_opportunities[-10:]),  # 최근 10건
                'model_stats': self._collect_model_stats(),
            }
