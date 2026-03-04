"""시스템 설정 관리"""
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리
BASE_DIR = Path(__file__).resolve().parent.parent

# .env 파일 로드
load_dotenv(BASE_DIR / '.env')


class Config:
    """전역 설정 클래스"""

    # 프로젝트 루트 디렉토리
    BASE_DIR = BASE_DIR

    # 바이낸스 API 설정
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
    # API 키가 없으면 자동으로 테스트넷 사용 (안전장치)
    _testnet_env = os.getenv('BINANCE_TESTNET', '').lower()
    if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
        IS_TESTNET = True  # API 키 없으면 무조건 테스트넷
    elif _testnet_env in ('true', '1', 'yes'):
        IS_TESTNET = True
    elif _testnet_env in ('false', '0', 'no'):
        IS_TESTNET = False
    else:
        # 명시하지 않았으면 테스트넷이 기본
        IS_TESTNET = True

    # 바이낸스 엔드포인트
    if IS_TESTNET:
        SPOT_BASE_URL = "https://testnet.binance.vision"
        WSS_BASE_URL = "wss://testnet.binance.vision/ws"
    else:
        SPOT_BASE_URL = "https://api.binance.com"
        WSS_BASE_URL = "wss://stream.binance.com:9443/ws"

    # 환경 설정
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

    # 데이터베이스 설정
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'cointrade')
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', '')

    # SQLAlchemy 연결 문자열
    DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

    # Redis 설정
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))

    # 거래 설정
    TRADING_SYMBOLS: List[str] = os.getenv('TRADING_SYMBOLS', 'BTCUSDT,ETHUSDT').split(',')
    DEFAULT_TIMEFRAME = os.getenv('DEFAULT_TIMEFRAME', '1m')
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 10000.0))

    # 리스크 관리 파라미터
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 0.2))  # 20%
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', 0.05))  # 5%
    STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', 0.01))  # 1%
    MAX_LEVERAGE = 3.0
    MAX_CONCURRENT_POSITIONS = 5

    # API Rate Limits (바이낸스 공식)
    RATE_LIMIT_REQUESTS_PER_MINUTE = 1200
    RATE_LIMIT_ORDERS_PER_SECOND = 10
    RATE_LIMIT_ORDERS_PER_DAY = 100000

    # 데이터 수집 설정
    HISTORICAL_DATA_DAYS = 365  # 1년치 데이터
    DATA_COLLECTION_INTERVAL = 60  # 초 단위

    # 모델 설정
    MODEL_SAVE_DIR = BASE_DIR / 'models' / 'saved'
    MODEL_CHECKPOINT_DIR = BASE_DIR / 'models' / 'checkpoints'
    LOOKBACK_PERIOD = 60  # 모델 입력 시퀀스 길이

    # 백테스팅 설정
    BACKTEST_START_DATE = '2023-01-01'
    BACKTEST_END_DATE = '2024-12-31'
    TRADING_FEE_MAKER = 0.001  # 0.1%
    TRADING_FEE_TAKER = 0.001  # 0.1%
    SLIPPAGE = 0.0005  # 0.05%

    # 선물 거래 수수료/슬리피지 (모델 학습용)
    FUTURES_FEE = float(os.getenv('FUTURES_FEE', 0.0004))  # 0.04% (바이낸스 선물 기본)
    FUTURES_SLIPPAGE = float(os.getenv('FUTURES_SLIPPAGE', 0.0005))  # 0.05%
    FUTURES_TOTAL_COST = FUTURES_FEE + FUTURES_SLIPPAGE  # 편도 0.09%

    # 틱 모델 라벨링 threshold (10초봉: 0.1% 미만은 노이즈)
    TICK_LABEL_THRESHOLD = float(os.getenv('TICK_LABEL_THRESHOLD', 0.001))  # 0.1%

    # 변동성 필터 임계값
    VOLATILITY_FILTER_THRESHOLD = float(os.getenv('VOL_FILTER_THRESHOLD', 0.0005))

    # 확률 기반 진입 임계값
    ENTRY_PROB_HIGH = float(os.getenv('ENTRY_PROB_HIGH', 0.65))
    ENTRY_PROB_LOW = float(os.getenv('ENTRY_PROB_LOW', 0.35))

    # 알림 설정
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')

    # 로깅 설정
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_DIR = BASE_DIR / 'logs'
    LOG_FILE = LOG_DIR / 'trading.log'

    # 디렉토리 생성
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 생성"""
        directories = [
            cls.MODEL_SAVE_DIR,
            cls.MODEL_CHECKPOINT_DIR,
            cls.LOG_DIR,
            BASE_DIR / 'data' / 'raw',
            BASE_DIR / 'data' / 'processed',
            BASE_DIR / 'backtesting' / 'results',
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls):
        """설정 유효성 검사"""
        errors = []

        if not cls.BINANCE_API_KEY:
            errors.append("BINANCE_API_KEY가 설정되지 않았습니다.")

        if not cls.BINANCE_SECRET_KEY:
            errors.append("BINANCE_SECRET_KEY가 설정되지 않았습니다.")

        if cls.MAX_POSITION_SIZE > 1.0 or cls.MAX_POSITION_SIZE < 0:
            errors.append("MAX_POSITION_SIZE는 0과 1 사이여야 합니다.")

        if errors:
            raise ValueError("설정 오류:\n" + "\n".join(errors))

        return True

    @classmethod
    def print_config(cls):
        """현재 설정 출력 (민감한 정보 제외)"""
        print("=" * 60)
        print("바이낸스 자동거래 시스템 설정")
        print("=" * 60)
        print(f"환경: {cls.ENVIRONMENT}")
        print(f"테스트넷: {cls.IS_TESTNET}")
        print(f"거래 심볼: {', '.join(cls.TRADING_SYMBOLS)}")
        print(f"초기 자본: ${cls.INITIAL_CAPITAL:,.2f}")
        print(f"최대 포지션 크기: {cls.MAX_POSITION_SIZE * 100}%")
        print(f"일일 최대 손실: {cls.MAX_DAILY_LOSS * 100}%")
        print(f"손절매: {cls.STOP_LOSS_PCT * 100}%")
        print("=" * 60)


# 애플리케이션 시작 시 디렉토리 생성
Config.create_directories()
