"""예외 처리 유틸리티"""
import functools
import time
import traceback
from typing import Callable, Any, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


def graceful_degradation(fallback_value=None, log_error=True):
    """
    예외 발생 시 graceful degradation을 제공하는 데코레이터

    Args:
        fallback_value: 예외 발생 시 반환할 기본값
        log_error: 에러 로깅 여부

    Example:
        @graceful_degradation(fallback_value=pd.DataFrame())
        def get_data():
            # 데이터 가져오기
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    # 상세 에러 로깅
                    error_msg = f"{func.__name__} 실행 중 예외 발생: {type(e).__name__}: {e}"
                    logger.error(error_msg)
                    logger.debug(f"스택 트레이스:\n{traceback.format_exc()}")

                    # 인자 정보 (민감 정보 제외)
                    safe_args = args[:2] if len(args) > 0 else []
                    safe_kwargs = {k: v for k, v in kwargs.items() if k not in ['password', 'secret', 'key']}
                    logger.debug(f"Args: {safe_args}, Kwargs: {list(safe_kwargs.keys())}")

                # graceful degradation: 기본값 반환
                return fallback_value

        return wrapper
    return decorator


def with_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0,
               exceptions: tuple = (Exception,)):
    """
    재시도 데코레이터 (지수 백오프 지원)

    Args:
        max_retries: 최대 재시도 횟수
        delay: 초기 재시도 간 대기 시간 (초)
        backoff: 백오프 배율 (2.0 = 지수 백오프)
        exceptions: 재시도할 예외 타입들

    Example:
        @with_retry(max_retries=3, delay=1.0)
        def fetch_api_data():
            # API 호출
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"{func.__name__} 재시도 {attempt + 1}/{max_retries}: "
                            f"{type(e).__name__}: {e} (다음 시도까지 {wait_time:.1f}초 대기)"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"{func.__name__} {max_retries}회 재시도 후 실패: "
                            f"{type(e).__name__}: {e}"
                        )

            # 모든 재시도 실패 시 마지막 예외 발생
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def circuit_breaker(failure_threshold: int = 5, timeout: float = 60.0):
    """
    Circuit Breaker 패턴 구현

    연속 실패 시 일정 시간 동안 함수 실행을 중단하여 시스템 보호

    Args:
        failure_threshold: 차단 기준 실패 횟수
        timeout: 차단 후 대기 시간 (초)

    Example:
        @circuit_breaker(failure_threshold=5, timeout=60)
        def call_external_api():
            # 외부 API 호출
            pass
    """
    def decorator(func):
        func.failure_count = 0
        func.last_failure_time = None
        func.is_open = False

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Circuit이 열려있는지 확인
            if func.is_open:
                if time.time() - func.last_failure_time < timeout:
                    logger.warning(
                        f"{func.__name__} Circuit Breaker 열림 상태 "
                        f"(남은 시간: {timeout - (time.time() - func.last_failure_time):.1f}초)"
                    )
                    raise Exception(f"Circuit Breaker 열림: {func.__name__}")
                else:
                    # 타임아웃 경과, 반쯤 열린 상태로 전환
                    logger.info(f"{func.__name__} Circuit Breaker 반쯤 열림 상태로 전환")
                    func.is_open = False
                    func.failure_count = 0

            try:
                result = func(*args, **kwargs)
                # 성공 시 카운터 리셋
                func.failure_count = 0
                return result
            except Exception as e:
                func.failure_count += 1
                func.last_failure_time = time.time()

                if func.failure_count >= failure_threshold:
                    func.is_open = True
                    logger.error(
                        f"{func.__name__} Circuit Breaker 열림: "
                        f"{failure_threshold}회 연속 실패"
                    )

                raise

        return wrapper
    return decorator


class ErrorRecovery:
    """에러 복구 전략 관리"""

    @staticmethod
    def safe_division(a: float, b: float, default: float = 0.0) -> float:
        """안전한 나눗셈"""
        try:
            if b == 0:
                logger.warning(f"0으로 나눔 방지: {a} / {b} = {default}")
                return default
            return a / b
        except Exception as e:
            logger.error(f"나눗셈 오류: {e}")
            return default

    @staticmethod
    def safe_list_access(lst: list, index: int, default: Any = None) -> Any:
        """안전한 리스트 접근"""
        try:
            return lst[index]
        except (IndexError, TypeError) as e:
            logger.debug(f"리스트 접근 오류: {e}, 기본값 반환")
            return default

    @staticmethod
    def safe_dict_access(d: dict, key: str, default: Any = None) -> Any:
        """안전한 딕셔너리 접근"""
        try:
            return d.get(key, default)
        except (AttributeError, TypeError) as e:
            logger.debug(f"딕셔너리 접근 오류: {e}, 기본값 반환")
            return default