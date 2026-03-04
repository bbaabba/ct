"""바이낸스 선물(Futures) API 클라이언트"""
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


class BinanceAPIError(Exception):
    """바이낸스 API 오류 (에러 코드 + 메시지 포함)"""

    # 재시도 가능한 오류 코드
    RETRYABLE = {-1001, -1003, -1015, -1021}
    # 주문 관련 비복구 오류 (재시도 금지)
    ORDER_FATAL = {-2010, -2013, -2014, -2015, -2019, -2022, -4028}

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")

    @property
    def is_retryable(self) -> bool:
        """서버 과부하/타임스탬프 등 일시적 오류 → 재시도 가능"""
        return self.code in self.RETRYABLE

    @property
    def is_insufficient_balance(self) -> bool:
        return self.code == -2019

    @property
    def is_invalid_quantity(self) -> bool:
        return self.code in (-1111, -4028)

    @property
    def is_position_not_found(self) -> bool:
        return self.code in (-2013, -2014, -2015)

    @property
    def is_timestamp_error(self) -> bool:
        return self.code == -1021


class BinanceFuturesClient:
    """바이낸스 USDⓈ-M 선물 REST API 클라이언트"""

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

        # 선물 엔드포인트 설정
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"
            self.wss_url = "wss://stream.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"
            self.wss_url = "wss://fstream.binance.com"

        # Rate Limiter 초기화
        self.request_limiter = RateLimiter(
            max_requests=Config.RATE_LIMIT_REQUESTS_PER_MINUTE,
            time_window=60
        )
        self.order_limiter = RateLimiter(
            max_requests=Config.RATE_LIMIT_ORDERS_PER_SECOND,
            time_window=1
        )

        # HTTP 세션 설정
        self.session = self._create_session()

        logger.info(f"바이낸스 선물 클라이언트 초기화 (테스트넷: {testnet})")

    def _create_session(self) -> requests.Session:
        """재시도 로직이 포함된 HTTP 세션 생성"""
        session = requests.Session()

        retry_strategy = Retry(
            total=5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "DELETE"],
            backoff_factor=2
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
        """API 요청 실행"""
        wait_time = self.request_limiter.wait_if_needed()
        if wait_time:
            logger.warning(f"Rate limit 도달. {wait_time:.2f}초 대기")

        params = params or {}

        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
            params['signature'] = self._create_signature(params)

        headers = {}
        if self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key

        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params if method == 'GET' else None,
                data=params if method in ['POST', 'DELETE', 'PUT'] else None,
                headers=headers,
                timeout=10
            )

            # 바이낸스 API 에러 응답 파싱 (HTTP 4xx도 JSON body에 code/msg 포함)
            if response.status_code >= 400:
                try:
                    body = response.json()
                    code = body.get('code', -response.status_code)
                    msg = body.get('msg', response.text)
                    api_err = BinanceAPIError(code, msg)

                    if api_err.is_retryable:
                        logger.warning(f"바이낸스 일시 오류 [{code}]: {msg} → 재시도 가능")
                    elif api_err.is_insufficient_balance:
                        logger.error(f"잔고 부족 [{code}]: {msg}")
                    elif api_err.is_timestamp_error:
                        logger.error(f"타임스탬프 오류 [{code}]: {msg} → 서버 시간 동기화 필요")
                    elif api_err.is_invalid_quantity:
                        logger.error(f"수량 오류 [{code}]: {msg}")
                    elif api_err.is_position_not_found:
                        logger.warning(f"포지션 없음 [{code}]: {msg}")
                    else:
                        logger.error(f"바이낸스 API 오류 [{code}]: {msg}")

                    raise api_err
                except (ValueError, KeyError):
                    pass  # JSON 파싱 실패 시 아래 raise_for_status로 폴백

            response.raise_for_status()
            return response.json()

        except BinanceAPIError:
            raise  # 이미 로깅됨, 그대로 전파

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
            self._request('GET', '/fapi/v1/ping')
            logger.info("바이낸스 선물 서버 연결 성공")
            return True
        except Exception as e:
            logger.error(f"서버 연결 실패: {e}")
            return False

    def get_server_time(self) -> int:
        """서버 시간 조회"""
        response = self._request('GET', '/fapi/v1/time')
        return response['serverTime']

    def get_exchange_info(self) -> Dict[str, Any]:
        """거래소 정보 조회"""
        return self._request('GET', '/fapi/v1/exchangeInfo')

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """특정 심볼 정보 조회"""
        exchange_info = self.get_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol.upper():
                return s
        return None

    def get_ticker_price(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """현재 가격 조회"""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        return self._request('GET', '/fapi/v1/ticker/price', params)

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[List]:
        """캔들스틱 데이터 조회"""
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': min(limit, 1500)
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        return self._request('GET', '/fapi/v1/klines', params)

    def get_klines_paginated(
        self,
        symbol: str,
        interval: str,
        total_candles: int,
        end_time: Optional[int] = None
    ) -> List[List]:
        """대량 캔들스틱 데이터 페이지네이션 조회

        1500개 초과 시 과거 방향으로 반복 호출하여 수집.
        시간순(오래된→최신) 정렬된 결과를 반환합니다.

        Args:
            symbol: 거래 심볼
            interval: 캔들 간격
            total_candles: 수집할 총 캔들 수
            end_time: 마지막 캔들 시간 (밀리초). None이면 현재 시간
        """
        if total_candles <= 1500:
            return self.get_klines(
                symbol=symbol, interval=interval,
                limit=total_candles, end_time=end_time
            )

        all_klines: List[List] = []
        batch_size = 1500
        _end_time = end_time or int(time.time() * 1000)
        collected = 0

        while collected < total_candles:
            remaining = total_candles - collected
            klines = self.get_klines(
                symbol=symbol,
                interval=interval,
                limit=min(batch_size, remaining),
                end_time=_end_time
            )

            if not klines:
                break

            all_klines = klines + all_klines  # prepend (시간순 정렬)
            collected += len(klines)
            _end_time = klines[0][0] - 1  # 첫 캔들 직전으로 이동

            if len(klines) < min(batch_size, remaining):
                break  # 더 이상 데이터 없음

            time.sleep(0.2)

        logger.info(f"페이지네이션 수집 완료: {symbol} {interval} {len(all_klines)}캔들 ({(collected - 1) // batch_size + 1}회 호출)")
        return all_klines

    def get_mark_price(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """마크 가격 및 펀딩비율 조회"""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        return self._request('GET', '/fapi/v1/premiumIndex', params)

    def get_funding_rate(self, symbol: str, limit: int = 100) -> List[Dict]:
        """펀딩비율 히스토리 조회"""
        params = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        return self._request('GET', '/fapi/v1/fundingRate', params)

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """호가창 조회"""
        params = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        return self._request('GET', '/fapi/v1/depth', params)

    # ============ 계정 관련 API ============

    def get_account_info(self) -> Dict[str, Any]:
        """선물 계정 정보 조회"""
        return self._request('GET', '/fapi/v2/account', signed=True)

    def get_balance(self) -> List[Dict[str, Any]]:
        """선물 잔고 조회"""
        return self._request('GET', '/fapi/v2/balance', signed=True)

    def get_usdt_balance(self) -> Dict[str, float]:
        """USDT 잔고만 조회"""
        balances = self.get_balance()
        for b in balances:
            if b['asset'] == 'USDT':
                return {
                    'balance': float(b['balance']),
                    'available': float(b['availableBalance']),
                    'unrealized_pnl': float(b.get('crossUnPnl', 0))
                }
        return {'balance': 0.0, 'available': 0.0, 'unrealized_pnl': 0.0}

    def get_position_info(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """포지션 정보 조회"""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        return self._request('GET', '/fapi/v2/positionRisk', params, signed=True)

    def get_active_positions(self) -> List[Dict[str, Any]]:
        """활성 포지션만 조회 (수량 > 0)"""
        positions = self.get_position_info()
        return [p for p in positions if float(p['positionAmt']) != 0]

    # ============ 레버리지 및 마진 설정 ============

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        레버리지 설정

        Args:
            symbol: 거래 심볼
            leverage: 레버리지 배율 (1-125)
        """
        params = {
            'symbol': symbol.upper(),
            'leverage': leverage
        }
        result = self._request('POST', '/fapi/v1/leverage', params, signed=True)
        logger.info(f"레버리지 설정: {symbol} = {leverage}x")
        return result

    def set_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """
        마진 타입 설정

        Args:
            symbol: 거래 심볼
            margin_type: ISOLATED (격리) 또는 CROSSED (교차)
        """
        params = {
            'symbol': symbol.upper(),
            'marginType': margin_type.upper()
        }
        try:
            result = self._request('POST', '/fapi/v1/marginType', params, signed=True)
            logger.info(f"마진 타입 설정: {symbol} = {margin_type}")
            return result
        except requests.exceptions.HTTPError as e:
            # 이미 같은 타입이면 에러가 발생하지만 무시
            if 'No need to change margin type' in str(e):
                logger.info(f"마진 타입 이미 설정됨: {symbol} = {margin_type}")
                return {'code': 0, 'msg': 'Already set'}
            raise

    def get_leverage_brackets(self, symbol: Optional[str] = None) -> List[Dict]:
        """레버리지 등급 조회"""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        return self._request('GET', '/fapi/v1/leverageBracket', params, signed=True)

    # ============ 주문 관련 API ============

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        position_side: str = 'BOTH',
        reduce_only: bool = False,
        time_in_force: str = 'GTC',
        close_position: bool = False
    ) -> Dict[str, Any]:
        """
        선물 주문

        Args:
            symbol: 거래 심볼
            side: BUY 또는 SELL
            order_type: LIMIT, MARKET, STOP, STOP_MARKET, TAKE_PROFIT, TAKE_PROFIT_MARKET
            quantity: 수량 (close_position=True면 불필요)
            price: 가격 (LIMIT 주문시)
            stop_price: 스탑 가격 (STOP, TAKE_PROFIT 주문시)
            position_side: BOTH (단방향), LONG, SHORT (양방향 모드)
            reduce_only: 포지션 축소만 허용
            time_in_force: GTC, IOC, FOK, GTX
            close_position: 전체 포지션 청산 여부
        """
        self.order_limiter.wait_if_needed()

        params = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'type': order_type.upper(),
            'positionSide': position_side.upper()
        }

        if quantity is not None and not close_position:
            params['quantity'] = quantity

        if price is not None:
            params['price'] = price

        if stop_price is not None:
            params['stopPrice'] = stop_price

        if order_type.upper() == 'LIMIT':
            params['timeInForce'] = time_in_force

        if reduce_only:
            params['reduceOnly'] = 'true'

        if close_position:
            params['closePosition'] = 'true'

        logger.info(f"선물 주문: {side} {quantity} {symbol} @ {order_type}")
        return self._request('POST', '/fapi/v1/order', params, signed=True)

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        position_side: str = 'BOTH',
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """시장가 주문"""
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type='MARKET',
            quantity=quantity,
            position_side=position_side,
            reduce_only=reduce_only
        )

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        position_side: str = 'BOTH',
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """지정가 주문"""
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type='LIMIT',
            quantity=quantity,
            price=price,
            position_side=position_side,
            reduce_only=reduce_only
        )

    def place_stop_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        position_side: str = 'BOTH',
        reduce_only: bool = True
    ) -> Dict[str, Any]:
        """스탑 마켓 주문 (손절용)"""
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type='STOP_MARKET',
            quantity=quantity,
            stop_price=stop_price,
            position_side=position_side,
            reduce_only=reduce_only
        )

    def place_take_profit_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        position_side: str = 'BOTH',
        reduce_only: bool = True
    ) -> Dict[str, Any]:
        """익절 마켓 주문"""
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type='TAKE_PROFIT_MARKET',
            quantity=quantity,
            stop_price=stop_price,
            position_side=position_side,
            reduce_only=reduce_only
        )

    def close_position(self, symbol: str, position_side: str = 'BOTH') -> Dict[str, Any]:
        """
        포지션 전체 청산

        Args:
            symbol: 거래 심볼
            position_side: BOTH, LONG, SHORT
        """
        positions = self.get_position_info(symbol)

        for pos in positions:
            if pos['symbol'] == symbol.upper():
                position_amt = float(pos['positionAmt'])

                if position_amt > 0:
                    # 롱 포지션 청산 → SELL
                    return self.place_market_order(
                        symbol=symbol,
                        side='SELL',
                        quantity=abs(position_amt),
                        position_side=position_side,
                        reduce_only=True
                    )
                elif position_amt < 0:
                    # 숏 포지션 청산 → BUY
                    return self.place_market_order(
                        symbol=symbol,
                        side='BUY',
                        quantity=abs(position_amt),
                        position_side=position_side,
                        reduce_only=True
                    )

        logger.warning(f"청산할 포지션 없음: {symbol}")
        return {'msg': 'No position to close'}

    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """주문 취소"""
        params = {
            'symbol': symbol.upper(),
            'orderId': order_id
        }
        logger.info(f"주문 취소: {symbol} - {order_id}")
        return self._request('DELETE', '/fapi/v1/order', params, signed=True)

    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """해당 심볼의 모든 주문 취소"""
        params = {'symbol': symbol.upper()}
        logger.info(f"전체 주문 취소: {symbol}")
        return self._request('DELETE', '/fapi/v1/allOpenOrders', params, signed=True)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """미체결 주문 조회"""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        return self._request('GET', '/fapi/v1/openOrders', params, signed=True)

    def get_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """주문 상태 조회"""
        params = {
            'symbol': symbol.upper(),
            'orderId': order_id
        }
        return self._request('GET', '/fapi/v1/order', params, signed=True)

    def get_all_orders(
        self,
        symbol: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """전체 주문 내역 조회"""
        params = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        return self._request('GET', '/fapi/v1/allOrders', params, signed=True)

    def get_trades(
        self,
        symbol: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """체결 내역 조회"""
        params = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        return self._request('GET', '/fapi/v1/userTrades', params, signed=True)

    # ============ 유틸리티 메서드 ============

    def calculate_liquidation_price(
        self,
        side: str,
        entry_price: float,
        leverage: int,
        maintenance_margin_rate: float = 0.004
    ) -> float:
        """
        청산가 계산

        Args:
            side: LONG 또는 SHORT
            entry_price: 진입가
            leverage: 레버리지
            maintenance_margin_rate: 유지마진율 (기본 0.4%)

        Returns:
            예상 청산가
        """
        if side.upper() == 'LONG':
            # 롱 청산가 = 진입가 × (1 - 1/레버리지 + 유지마진율)
            liq_price = entry_price * (1 - 1/leverage + maintenance_margin_rate)
        else:
            # 숏 청산가 = 진입가 × (1 + 1/레버리지 - 유지마진율)
            liq_price = entry_price * (1 + 1/leverage - maintenance_margin_rate)

        return round(liq_price, 2)

    def calculate_pnl(
        self,
        side: str,
        entry_price: float,
        current_price: float,
        quantity: float,
        leverage: int = 1
    ) -> Dict[str, float]:
        """
        손익 계산

        Returns:
            {
                'pnl': 손익 (USDT),
                'pnl_pct': 손익률 (%),
                'roe': 자본수익률 (레버리지 적용)
            }
        """
        if side.upper() == 'LONG':
            pnl = (current_price - entry_price) * quantity
        else:
            pnl = (entry_price - current_price) * quantity

        # 투자금 = 진입가 × 수량 / 레버리지
        investment = entry_price * quantity / leverage
        pnl_pct = (pnl / (entry_price * quantity)) * 100
        roe = (pnl / investment) * 100 if investment > 0 else 0

        return {
            'pnl': round(pnl, 4),
            'pnl_pct': round(pnl_pct, 2),
            'roe': round(roe, 2)
        }

    def get_symbol_precision(self, symbol: str) -> Dict[str, int]:
        """심볼의 수량/가격 정밀도 조회"""
        info = self.get_symbol_info(symbol)
        if info:
            for f in info['filters']:
                if f['filterType'] == 'PRICE_FILTER':
                    price_precision = len(f['tickSize'].rstrip('0').split('.')[-1])
                if f['filterType'] == 'LOT_SIZE':
                    qty_precision = len(f['stepSize'].rstrip('0').split('.')[-1])
            return {
                'price_precision': price_precision,
                'quantity_precision': qty_precision
            }
        return {'price_precision': 2, 'quantity_precision': 3}

    def round_quantity(self, symbol: str, quantity: float) -> float:
        """수량을 심볼 정밀도에 맞게 반올림"""
        precision = self.get_symbol_precision(symbol)
        return round(quantity, precision['quantity_precision'])

    def round_price(self, symbol: str, price: float) -> float:
        """가격을 심볼 정밀도에 맞게 반올림"""
        precision = self.get_symbol_precision(symbol)
        return round(price, precision['price_precision'])
