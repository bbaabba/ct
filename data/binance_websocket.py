"""바이낸스 WebSocket 클라이언트"""
import json
import threading
import time
from typing import Callable, List, Optional, Dict, Any
import websocket

from config.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class BinanceWebSocket:
    """바이낸스 WebSocket 스트림 클라이언트"""

    def __init__(
        self,
        streams: List[str],
        on_message_callback: Callable[[Dict[str, Any]], None],
        on_error_callback: Optional[Callable[[Exception], None]] = None,
        testnet: bool = True
    ):
        """
        Args:
            streams: 구독할 스트림 리스트 (예: ['btcusdt@kline_1m', 'ethusdt@kline_1m'])
            on_message_callback: 메시지 수신 시 호출할 콜백 함수
            on_error_callback: 에러 발생 시 호출할 콜백 함수
            testnet: 테스트넷 사용 여부
        """
        self.streams = streams
        self.on_message_callback = on_message_callback
        self.on_error_callback = on_error_callback
        self.testnet = testnet

        # WebSocket URL 생성
        if testnet:
            # 테스트넷은 단일 스트림 방식 사용
            base_url = "wss://testnet.binance.vision/ws"
            # 첫 번째 스트림만 사용 (테스트넷 제한)
            self.url = f"{base_url}/{streams[0]}"
        else:
            # 실전은 복합 스트림 방식
            base_url = "wss://stream.binance.com:9443/stream"
            stream_path = '/'.join(streams)
            self.url = f"{base_url}?streams={stream_path}"

        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10

        logger.info(f"WebSocket 클라이언트 초기화: {len(streams)}개 스트림")

    def on_message(self, ws: websocket.WebSocketApp, message: str):
        """메시지 수신 핸들러"""
        try:
            data = json.loads(message)
            self.on_message_callback(data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
        except Exception as e:
            logger.error(f"메시지 처리 오류: {e}")

    def on_error(self, ws: websocket.WebSocketApp, error: Exception):
        """에러 핸들러"""
        logger.error(f"WebSocket 오류: {error}")

        if self.on_error_callback:
            try:
                self.on_error_callback(error)
            except Exception as e:
                logger.error(f"에러 콜백 실행 오류: {e}")

    def on_close(self, ws: websocket.WebSocketApp, close_status_code: int, close_msg: str):
        """연결 종료 핸들러"""
        logger.warning(f"WebSocket 연결 종료: {close_status_code} - {close_msg}")

        if self.is_running:
            # 자동 재연결
            self.reconnect()

    def on_open(self, ws: websocket.WebSocketApp):
        """연결 성공 핸들러"""
        logger.info(f"WebSocket 연결 성공: {len(self.streams)}개 스트림")
        self.reconnect_attempts = 0  # 재연결 카운터 리셋

    def reconnect(self):
        """재연결 시도"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"최대 재연결 시도 횟수 초과 ({self.max_reconnect_attempts})")
            self.is_running = False
            return

        self.reconnect_attempts += 1
        wait_time = min(2 ** self.reconnect_attempts, 60)  # 지수 백오프 (최대 60초)

        logger.info(f"재연결 시도 {self.reconnect_attempts}/{self.max_reconnect_attempts} ({wait_time}초 후)")
        time.sleep(wait_time)

        self.connect()

    def connect(self):
        """WebSocket 연결"""
        if self.is_running and self.ws_thread and self.ws_thread.is_alive():
            logger.warning("이미 WebSocket이 실행 중입니다")
            return

        self.is_running = True

        # WebSocketApp 생성
        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )

        # 별도 스레드에서 실행
        self.ws_thread = threading.Thread(target=self._run_forever, daemon=True)
        self.ws_thread.start()

        logger.info("WebSocket 연결 시작")

    def _run_forever(self):
        """WebSocket 무한 실행"""
        try:
            self.ws.run_forever(
                ping_interval=20,  # 20초마다 ping
                ping_timeout=10    # 10초 타임아웃
            )
        except Exception as e:
            logger.error(f"WebSocket 실행 오류: {e}")
            if self.is_running:
                self.reconnect()

    def close(self):
        """WebSocket 연결 종료"""
        logger.info("WebSocket 연결 종료 중...")
        self.is_running = False

        if self.ws:
            self.ws.close()

        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5)

        logger.info("WebSocket 연결 종료 완료")

    @staticmethod
    def create_kline_stream(symbol: str, interval: str) -> str:
        """
        캔들스틱 스트림 이름 생성

        Args:
            symbol: 심볼 (예: BTCUSDT)
            interval: 간격 (1m, 5m, 15m, 1h, 4h, 1d 등)

        Returns:
            스트림 이름 (예: btcusdt@kline_1m)
        """
        return f"{symbol.lower()}@kline_{interval}"

    @staticmethod
    def create_trade_stream(symbol: str) -> str:
        """거래 스트림 이름 생성"""
        return f"{symbol.lower()}@trade"

    @staticmethod
    def create_ticker_stream(symbol: str) -> str:
        """24시간 티커 스트림 이름 생성"""
        return f"{symbol.lower()}@ticker"

    @staticmethod
    def create_book_ticker_stream(symbol: str) -> str:
        """최적 호가 스트림 이름 생성"""
        return f"{symbol.lower()}@bookTicker"

    @staticmethod
    def create_depth_stream(symbol: str, levels: int = 5) -> str:
        """
        호가창 스트림 이름 생성

        Args:
            symbol: 심볼
            levels: 레벨 (5, 10, 20)

        Returns:
            스트림 이름
        """
        return f"{symbol.lower()}@depth{levels}"


class OrderBookManager:
    """로컬 Order Book 관리자"""

    def __init__(self, symbol: str, binance_client):
        """
        Args:
            symbol: 심볼
            binance_client: BinanceClient 인스턴스
        """
        self.symbol = symbol
        self.client = binance_client

        self.bids: Dict[float, float] = {}  # {price: quantity}
        self.asks: Dict[float, float] = {}
        self.last_update_id = 0
        self.initialized = False
        self.buffer = []  # 초기화 전 메시지 버퍼

        logger.info(f"OrderBookManager 초기화: {symbol}")

    def initialize(self):
        """REST API로 초기 스냅샷 가져오기"""
        logger.info(f"{self.symbol} Order Book 초기화 중...")

        snapshot = self.client.get_order_book(self.symbol, limit=1000)

        self.last_update_id = snapshot['lastUpdateId']

        # 초기 호가 설정
        self.bids = {float(price): float(qty) for price, qty in snapshot['bids']}
        self.asks = {float(price): float(qty) for price, qty in snapshot['asks']}

        self.initialized = True
        logger.info(f"{self.symbol} Order Book 초기화 완료 (Update ID: {self.last_update_id})")

        # 버퍼된 메시지 처리
        for buffered_data in self.buffer:
            self.update(buffered_data)
        self.buffer.clear()

    def update(self, depth_data: Dict[str, Any]):
        """
        Diff Depth Stream으로부터 업데이트

        Args:
            depth_data: WebSocket 메시지 데이터
        """
        if not self.initialized:
            self.buffer.append(depth_data)
            return

        # 스트림 데이터 추출
        if 'data' in depth_data:
            data = depth_data['data']
        else:
            data = depth_data

        first_update_id = data['U']
        last_update_id = data['u']

        # 중복 또는 오래된 업데이트 무시
        if last_update_id <= self.last_update_id:
            return

        # 연속성 확인
        if first_update_id > self.last_update_id + 1:
            logger.warning(f"{self.symbol} Order Book 동기화 오류. 재초기화...")
            self.initialized = False
            self.initialize()
            return

        # Bids 업데이트
        for price, qty in data['b']:
            price, qty = float(price), float(qty)
            if qty == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        # Asks 업데이트
        for price, qty in data['a']:
            price, qty = float(price), float(qty)
            if qty == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

        self.last_update_id = last_update_id

    def get_best_bid(self) -> tuple:
        """최고 매수 호가"""
        if self.bids:
            best_price = max(self.bids.keys())
            return best_price, self.bids[best_price]
        return None, None

    def get_best_ask(self) -> tuple:
        """최저 매도 호가"""
        if self.asks:
            best_price = min(self.asks.keys())
            return best_price, self.asks[best_price]
        return None, None

    def get_spread(self) -> Optional[float]:
        """스프레드 계산"""
        best_bid, _ = self.get_best_bid()
        best_ask, _ = self.get_best_ask()

        if best_bid and best_ask:
            return best_ask - best_bid
        return None

    def get_mid_price(self) -> Optional[float]:
        """중간 가격 계산"""
        best_bid, _ = self.get_best_bid()
        best_ask, _ = self.get_best_ask()

        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None
