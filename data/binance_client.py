"""바이낸스 API 클라이언트"""
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.config import Config
from utils.rate_limiter import RateLimiter
from utils.logger import setup_logger

logger = setup_logger(__name__)


class BinanceClient:
    """바이낸스 REST API 클라이언트"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        testnet: bool = True
    ):
        """
        Args:
            api_key: API 키 (None이면 Config에서 가져옴)
            secret_key: Secret 키 (None이면 Config에서 가져옴)
            testnet: 테스트넷 사용 여부
        """
        self.api_key = api_key or Config.BINANCE_API_KEY
        self.secret_key = secret_key or Config.BINANCE_SECRET_KEY
        self.testnet = testnet

        # 엔드포인트 설정
        if testnet:
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = Config.SPOT_BASE_URL

        # Rate Limiter 초기화
        self.request_limiter = RateLimiter(
            max_requests=Config.RATE_LIMIT_REQUESTS_PER_MINUTE,
            time_window=60
        )
        self.order_limiter = RateLimiter(
            max_requests=Config.RATE_LIMIT_ORDERS_PER_SECOND,
            time_window=1
        )

        # HTTP 세션 설정 (재시도 로직 포함)
        self.session = self._create_session()

        logger.info(f"바이낸스 클라이언트 초기화 (테스트넷: {testnet})")

    def _create_session(self) -> requests.Session:
        """재시도 로직이 포함된 HTTP 세션 생성"""
        session = requests.Session()

        # 재시도 전략
        retry_strategy = Retry(
            total=5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "DELETE"],
            backoff_factor=2  # 2, 4, 8, 16, 32초
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    def _create_signature(self, params: Dict[str, Any]) -> str:
        """HMAC SHA256 서명 생성"""
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """
        API 요청 실행

        Args:
            method: HTTP 메서드 (GET, POST, DELETE 등)
            endpoint: API 엔드포인트
            params: 요청 파라미터
            signed: 서명 필요 여부

        Returns:
            API 응답 (JSON)
        """
        # Rate Limit 체크
        wait_time = self.request_limiter.wait_if_needed()
        if wait_time:
            logger.warning(f"Rate limit 도달. {wait_time:.2f}초 대기")

        # 파라미터 준비
        params = params or {}

        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
            params['signature'] = self._create_signature(params)

        # 헤더 설정
        headers = {}
        if self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key

        # 요청 실행
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params if method == 'GET' else None,
                data=params if method in ['POST', 'DELETE'] else None,
                headers=headers,
                timeout=10
            )

            # Rate Limit 정보 로깅
            if 'X-MBX-USED-WEIGHT-1M' in response.headers:
                used_weight = response.headers['X-MBX-USED-WEIGHT-1M']
                logger.debug(f"사용된 가중치: {used_weight}/6000")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP 오류: {e}")
            logger.error(f"응답 내용: {e.response.text if e.response else 'N/A'}")
            raise

        except requests.exceptions.RequestException as e:
            logger.error(f"요청 오류: {e}")
            raise

    # ============ 시장 데이터 API ============

    def ping(self) -> bool:
        """서버 연결 테스트"""
        try:
            self._request('GET', '/api/v3/ping')
            logger.info("바이낸스 서버 연결 성공")
            return True
        except Exception as e:
            logger.error(f"서버 연결 실패: {e}")
            return False

    def get_server_time(self) -> int:
        """서버 시간 조회"""
        response = self._request('GET', '/api/v3/time')
        return response['serverTime']

    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """거래소 정보 조회"""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()

        return self._request('GET', '/api/v3/exchangeInfo', params)

    def get_ticker_price(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """현재 가격 조회"""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()

        return self._request('GET', '/api/v3/ticker/price', params)

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[List]:
        """
        캔들스틱 데이터 조회

        Args:
            symbol: 심볼 (예: BTCUSDT)
            interval: 간격 (1m, 5m, 15m, 1h, 4h, 1d 등)
            limit: 개수 (최대 1000)
            start_time: 시작 시간 (밀리초)
            end_time: 종료 시간 (밀리초)

        Returns:
            캔들스틱 데이터 리스트
        """
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': min(limit, 1000)
        }

        if start_time:
            params['startTime'] = start_time

        if end_time:
            params['endTime'] = end_time

        return self._request('GET', '/api/v3/klines', params)

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        호가창 조회

        Args:
            symbol: 심볼
            limit: 개수 (5, 10, 20, 50, 100, 500, 1000, 5000)

        Returns:
            호가창 데이터
        """
        params = {
            'symbol': symbol.upper(),
            'limit': limit
        }

        return self._request('GET', '/api/v3/depth', params)

    # ============ 계정 관련 API ============

    def get_account_info(self) -> Dict[str, Any]:
        """계정 정보 조회"""
        return self._request('GET', '/api/v3/account', signed=True)

    def get_balance(self, asset: Optional[str] = None) -> Dict[str, Any]:
        """잔고 조회"""
        account_info = self.get_account_info()

        if asset:
            for balance in account_info['balances']:
                if balance['asset'] == asset.upper():
                    return balance
            return {'asset': asset.upper(), 'free': '0', 'locked': '0'}

        return account_info['balances']

    # ============ 주문 관련 API ============

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> Dict[str, Any]:
        """
        시장가 주문

        Args:
            symbol: 심볼
            side: BUY 또는 SELL
            quantity: 수량

        Returns:
            주문 결과
        """
        # 주문 Rate Limit 체크
        self.order_limiter.wait_if_needed()

        params = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'type': 'MARKET',
            'quantity': quantity
        }

        logger.info(f"시장가 주문: {side} {quantity} {symbol}")

        return self._request('POST', '/api/v3/order', params, signed=True)

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        time_in_force: str = 'GTC'
    ) -> Dict[str, Any]:
        """
        지정가 주문

        Args:
            symbol: 심볼
            side: BUY 또는 SELL
            quantity: 수량
            price: 가격
            time_in_force: GTC, IOC, FOK

        Returns:
            주문 결과
        """
        # 주문 Rate Limit 체크
        self.order_limiter.wait_if_needed()

        params = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'type': 'LIMIT',
            'timeInForce': time_in_force,
            'quantity': quantity,
            'price': price
        }

        logger.info(f"지정가 주문: {side} {quantity} {symbol} @ {price}")

        return self._request('POST', '/api/v3/order', params, signed=True)

    def place_stop_loss_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        stop_price: float
    ) -> Dict[str, Any]:
        """
        손절 지정가 주문

        Args:
            symbol: 심볼
            side: BUY 또는 SELL
            quantity: 수량
            price: 체결 희망 가격
            stop_price: 손절 트리거 가격

        Returns:
            주문 결과
        """
        self.order_limiter.wait_if_needed()

        params = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'type': 'STOP_LOSS_LIMIT',
            'timeInForce': 'GTC',
            'quantity': quantity,
            'price': price,
            'stopPrice': stop_price
        }

        logger.info(f"손절 주문: {side} {quantity} {symbol} @ {price} (Stop: {stop_price})")

        return self._request('POST', '/api/v3/order', params, signed=True)

    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """주문 취소"""
        params = {
            'symbol': symbol.upper(),
            'orderId': order_id
        }

        logger.info(f"주문 취소: {symbol} - {order_id}")

        return self._request('DELETE', '/api/v3/order', params, signed=True)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """미체결 주문 조회"""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()

        return self._request('GET', '/api/v3/openOrders', params, signed=True)

    def get_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """주문 상태 조회"""
        params = {
            'symbol': symbol.upper(),
            'orderId': order_id
        }

        return self._request('GET', '/api/v3/order', params, signed=True)
