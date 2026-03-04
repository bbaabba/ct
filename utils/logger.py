"""로깅 설정"""
import sys
from pathlib import Path
from loguru import logger
from config.config import Config

# 전역 초기화 플래그 — setup_logger()가 여러 모듈에서 호출되어도
# 파일 핸들러는 1회만 등록 (다중 핸들러 → Windows 파일 락 에러 방지)
_initialized = False


def setup_logger(name: str = "cointrade", level: str = None):
    """
    로거 설정 (환경별 자동 최적화)

    Args:
        name: 로거 이름
        level: 로그 레벨 (None이면 환경에 따라 자동 설정)

    Returns:
        설정된 로거
    """
    global _initialized
    if _initialized:
        return logger

    _initialized = True

    # 기존 핸들러 제거
    logger.remove()

    # 환경별 로그 레벨 자동 설정
    if level is None:
        env = Config.ENVIRONMENT.lower()
        if env == 'production':
            level = 'WARNING'  # 프로덕션: 경고 이상만
        elif env == 'staging':
            level = 'INFO'     # 스테이징: 정보 이상
        elif env == 'development':
            level = 'DEBUG'    # 개발: 모든 로그
        else:
            level = Config.LOG_LEVEL  # 설정 파일 기본값

    # 환경별 포맷 설정
    if Config.ENVIRONMENT.lower() == 'production':
        # 프로덕션: 간결한 포맷
        console_format = "<level>{level: <8}</level> | {message}"
        file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    else:
        # 개발/스테이징: 상세한 포맷
        console_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"

    # 콘솔 출력
    logger.add(
        sys.stdout,
        format=console_format,
        level=level,
        colorize=True,
        filter=lambda record: record["level"].no < 40  # ERROR 이상은 stderr로
    )

    # 에러는 stderr로 출력
    logger.add(
        sys.stderr,
        format="<red>{time:YYYY-MM-DD HH:mm:ss}</red> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="ERROR",
        colorize=True
    )

    # 파일 출력
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 일반 로그 — {time} 플레이스홀더로 rotation 시 rename 없이 새 파일 생성 (Windows 호환)
    _retention = "30 days" if Config.ENVIRONMENT.lower() == 'development' else "90 days"
    logger.add(
        str(Config.LOG_DIR / "trading.{time:YYYY-MM-DD_HH-mm-ss}.log"),
        format=file_format,
        level=level,
        rotation="100 MB",
        retention=_retention,
        compression="zip",
        enqueue=True,
    )

    # 에러 로그 (별도 파일, 장기 보관)
    logger.add(
        str(Config.LOG_DIR / "error.{time:YYYY-MM-DD_HH-mm-ss}.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="50 MB",
        retention="180 days",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    # 성과 로그 (거래 결과 추적용)
    logger.add(
        str(Config.LOG_DIR / "performance.{time:YYYY-MM-DD_HH-mm-ss}.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        level="INFO",
        rotation="10 MB",
        retention="365 days",
        compression="zip",
        filter=lambda record: "PERFORMANCE" in record["extra"],
    )

    logger.info(f"로거 '{name}' 초기화 완료 (레벨: {level}, 환경: {Config.ENVIRONMENT})")
    return logger
