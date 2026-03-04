"""API Rate Limiter 구현"""
import time
from collections import deque
from typing import Optional


class RateLimiter:
    """
    바이낸스 API Rate Limit 관리
    - 분당 최대 요청 수: 1200
    - 초당 최대 주문 수: 10
    """

    def __init__(self, max_requests: int = 1200, time_window: int = 60):
        """
        Args:
            max_requests: 시간 윈도우 내 최대 요청 수
            time_window: 시간 윈도우 (초)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()

    def can_proceed(self) -> bool:
        """현재 요청 가능 여부 확인"""
        current_time = time.time()

        # 시간 윈도우 밖의 요청 제거
        while self.requests and self.requests[0] < current_time - self.time_window:
            self.requests.popleft()

        return len(self.requests) < self.max_requests

    def record_request(self):
        """요청 기록"""
        self.requests.append(time.time())

    def wait_if_needed(self) -> Optional[float]:
        """
        필요시 대기
        Returns:
            대기 시간 (초), 대기하지 않으면 None
        """
        if not self.can_proceed():
            oldest_request = self.requests[0]
            wait_time = self.time_window - (time.time() - oldest_request)

            if wait_time > 0:
                time.sleep(wait_time + 0.1)  # 0.1초 버퍼
                return wait_time

        self.record_request()
        return None

    def get_remaining_requests(self) -> int:
        """남은 요청 가능 수"""
        current_time = time.time()

        # 시간 윈도우 밖의 요청 제거
        while self.requests and self.requests[0] < current_time - self.time_window:
            self.requests.popleft()

        return self.max_requests - len(self.requests)

    def reset(self):
        """Rate Limiter 초기화"""
        self.requests.clear()
