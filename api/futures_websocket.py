"""바이낸스 선물 WebSocket 클라이언트"""
import json
import threading
import time
import math
from typing import Callable, List, Optional, Dict, Any
from collections import deque
import websocket

from utils.logger import setup_logger

logger = setup_logger(__name__)


class FuturesWebSocket:
    """바이낸스 USDⓈ-M 선물 WebSocket 스트림 클라이언트"""

    def __init__(
        self,
        streams: List[str],
        on_message_callback: Callable[[Dict[str, Any]], None],
        on_error_callback: Optional[Callable[[Exception], None]] = None,
        testnet: bool = True
    ):
        """
        Args:
            streams: 구독할 스트림 리스트
            on_message_callback: 메시지 수신 콜백
            on_error_callback: 에러 발생 콜백
            testnet: 테스트넷 사용 여부
        """
        self.streams = streams
        self.on_message_callback = on_message_callback
        self.on_error_callback = on_error_callback
        self.testnet = testnet

        # WebSocket URL
        if testnet:
            base_url = "wss://stream.binancefuture.com"
        else:
            base_url = "wss://fstream.binance.com"

        if len(streams) == 1:
            self.url = f"{base_url}/ws/{streams[0]}"
        else:
            stream_path = '/'.join(streams)
            self.url = f"{base_url}/stream?streams={stream_path}"

        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10

        logger.info(f"선물 WebSocket 초기화: {len(streams)}개 스트림")

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
            self.reconnect()

    def on_open(self, ws: websocket.WebSocketApp):
        """연결 성공 핸들러"""
        logger.info(f"선물 WebSocket 연결 성공")
        self.reconnect_attempts = 0

    def reconnect(self):
        """재연결 시도"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"최대 재연결 시도 횟수 초과")
            self.is_running = False
            return

        self.reconnect_attempts += 1
        wait_time = min(2 ** self.reconnect_attempts, 60)

        logger.info(f"재연결 시도 {self.reconnect_attempts}/{self.max_reconnect_attempts} ({wait_time}초 후)")
        time.sleep(wait_time)

        self.connect()

    def connect(self):
        """WebSocket 연결"""
        if self.is_running and self.ws_thread and self.ws_thread.is_alive():
            logger.warning("이미 WebSocket이 실행 중입니다")
            return

        self.is_running = True

        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )

        self.ws_thread = threading.Thread(target=self._run_forever, daemon=True)
        self.ws_thread.start()

        logger.info("선물 WebSocket 연결 시작")

    def _run_forever(self):
        """WebSocket 무한 실행"""
        try:
            self.ws.run_forever(
                ping_interval=20,
                ping_timeout=10
            )
        except Exception as e:
            logger.error(f"WebSocket 실행 오류: {e}")
            if self.is_running:
                self.reconnect()

    def close(self):
        """WebSocket 연결 종료"""
        logger.info("선물 WebSocket 연결 종료 중...")
        self.is_running = False

        if self.ws:
            self.ws.close()

        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5)

        logger.info("선물 WebSocket 연결 종료 완료")

    # ============ 스트림 헬퍼 메서드 ============

    @staticmethod
    def create_kline_stream(symbol: str, interval: str) -> str:
        """캔들스틱 스트림"""
        return f"{symbol.lower()}@kline_{interval}"

    @staticmethod
    def create_mark_price_stream(symbol: str) -> str:
        """마크가격 스트림 (1초마다)"""
        return f"{symbol.lower()}@markPrice"

    @staticmethod
    def create_mark_price_all_stream() -> str:
        """전체 마크가격 스트림"""
        return "!markPrice@arr"

    @staticmethod
    def create_ticker_stream(symbol: str) -> str:
        """24시간 티커 스트림"""
        return f"{symbol.lower()}@ticker"

    @staticmethod
    def create_agg_trade_stream(symbol: str) -> str:
        """집계 거래 스트림"""
        return f"{symbol.lower()}@aggTrade"

    @staticmethod
    def create_book_ticker_stream(symbol: str) -> str:
        """최적 호가 스트림"""
        return f"{symbol.lower()}@bookTicker"

    @staticmethod
    def create_liquidation_stream(symbol: str) -> str:
        """청산 주문 스트림"""
        return f"{symbol.lower()}@forceOrder"

    @staticmethod
    def create_depth_stream(symbol: str, levels: int = 5, speed: str = '100ms') -> str:
        """호가창 스트림"""
        return f"{symbol.lower()}@depth{levels}@{speed}"


class TickDataCollector:
    """
    markPrice 업데이트를 3초 OHLCV 마이크로캔들로 집계.

    WebSocket markPrice 스트림(~1초 간격)을 받아서
    지정된 interval_seconds 단위로 OHLCV 캔들을 생성합니다.
    """

    def __init__(self, interval_seconds: int = 3, max_candles: int = 3000):
        """
        Args:
            interval_seconds: 마이크로캔들 집계 간격 (기본 3초)
            max_candles: 최대 저장 캔들 수 (~2.5시간 at 3초)
        """
        self.interval_seconds = interval_seconds
        self.max_candles = max_candles
        self.candles: deque = deque(maxlen=max_candles)

        # 현재 구축 중인 캔들 상태
        self._current_open: Optional[float] = None
        self._current_high: float = 0.0
        self._current_low: float = float('inf')
        self._current_close: float = 0.0
        self._current_tick_count: int = 0
        self._current_window_id: Optional[int] = None  # timestamp_ms // (interval * 1000)

        self._lock = threading.Lock()

    def on_mark_price(self, price: float, timestamp_ms: int) -> Optional[Dict]:
        """
        markPrice 업데이트 수신. 윈도우가 바뀌면 완성된 캔들 반환.

        Args:
            price: 마크 가격
            timestamp_ms: 서버 타임스탬프 (밀리초)

        Returns:
            완성된 캔들 dict (윈도우 교체 시) 또는 None
        """
        window_id = timestamp_ms // (self.interval_seconds * 1000)

        with self._lock:
            # 첫 번째 틱
            if self._current_window_id is None:
                self._current_window_id = window_id
                self._current_open = price
                self._current_high = price
                self._current_low = price
                self._current_close = price
                self._current_tick_count = 1
                return None

            # 같은 윈도우 내 업데이트
            if window_id == self._current_window_id:
                self._current_high = max(self._current_high, price)
                self._current_low = min(self._current_low, price)
                self._current_close = price
                self._current_tick_count += 1
                return None

            # 윈도우 전환 → 이전 캔들 완성
            completed = self._finalize_candle()

            # 새 윈도우 시작
            self._current_window_id = window_id
            self._current_open = price
            self._current_high = price
            self._current_low = price
            self._current_close = price
            self._current_tick_count = 1

            return completed

    def _finalize_candle(self) -> Dict:
        """현재 캔들을 완성하고 저장."""
        candle = {
            'timestamp': self._current_window_id * self.interval_seconds * 1000,
            'open': self._current_open,
            'high': self._current_high,
            'low': self._current_low,
            'close': self._current_close,
            'volume': float(self._current_tick_count),  # 틱 카운트를 볼륨으로 사용
        }
        self.candles.append(candle)
        return candle

    def get_candles_df(self) -> 'pd.DataFrame':
        """마이크로캔들 DataFrame 반환."""
        import pandas as pd

        with self._lock:
            candles_list = list(self.candles)
            # 현재 미완성 캔들도 포함
            if self._current_open is not None and self._current_tick_count > 0:
                candles_list.append({
                    'timestamp': self._current_window_id * self.interval_seconds * 1000,
                    'open': self._current_open,
                    'high': self._current_high,
                    'low': self._current_low,
                    'close': self._current_close,
                    'volume': float(self._current_tick_count),
                })

        if not candles_list:
            return pd.DataFrame()

        df = pd.DataFrame(candles_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def get_candle_count(self) -> int:
        """축적된 마이크로캔들 수 반환."""
        return len(self.candles)


class RealTimeDataManager:
    """
    실시간 데이터 관리자

    WebSocket 스트림에서 받은 데이터를 관리하고
    거래봇에 제공합니다.
    """

    def __init__(
        self,
        symbol: str,
        interval: str = '1m',
        testnet: bool = True,
        max_candles: int = 500
    ):
        """
        Args:
            symbol: 거래 심볼
            interval: 캔들스틱 간격
            testnet: 테스트넷 사용 여부
            max_candles: 저장할 최대 캔들 수
        """
        self.symbol = symbol.upper()
        self.interval = interval
        self.testnet = testnet
        self.max_candles = max_candles

        # 데이터 저장소
        self.candles = deque(maxlen=max_candles)
        self.current_candle: Optional[Dict] = None
        self.mark_price: float = 0.0
        self.funding_rate: float = 0.0
        self.last_update_time: float = 0

        # 통계
        self.message_count = 0
        self.error_count = 0

        # WebSocket
        self.ws: Optional[FuturesWebSocket] = None

        # 3초 마이크로캔들 수집기
        self.tick_collector: TickDataCollector = TickDataCollector(
            interval_seconds=3,
            max_candles=3000  # ~2.5시간
        )

        # bookTicker (최적 호가) 데이터
        self._best_bid_qty: float = 0.0
        self._best_ask_qty: float = 0.0
        self._best_bid_price: float = 0.0
        self._best_ask_price: float = 0.0
        self._book_ticker_update_time: float = 0.0

        # 스레드 안전
        self._lock = threading.Lock()

        # markPrice 콜백 (Layer 2 WS 트레일링용 — lock 밖에서 호출됨)
        self._mark_price_callbacks: List[Callable[[float], None]] = []
        # WS 연결 상태 콜백 (Layer 1 SL 조정용)
        self._on_ws_disconnect_cb: Optional[Callable[[], None]] = None
        self._on_ws_reconnect_cb: Optional[Callable[[], None]] = None
        self._ws_connected: bool = False

        logger.info(f"RealTimeDataManager 초기화: {symbol} {interval}")

    def register_mark_price_callback(self, callback: Callable[[float], None]) -> None:
        """markPrice 틱마다 호출될 콜백 등록 (WS 스레드에서 실행, lock 밖에서 호출)."""
        self._mark_price_callbacks.append(callback)

    def set_ws_state_callbacks(
        self,
        on_disconnect: Optional[Callable[[], None]] = None,
        on_reconnect: Optional[Callable[[], None]] = None,
    ) -> None:
        """WS 연결/단절 이벤트 콜백 등록."""
        self._on_ws_disconnect_cb = on_disconnect
        self._on_ws_reconnect_cb = on_reconnect

    def start(self):
        """실시간 데이터 수신 시작"""
        self._connect_ws()
        logger.info(f"실시간 데이터 수신 시작: {self.symbol}")

    def _connect_ws(self):
        """WebSocket 연결 (재연결용 내부 메서드)"""
        streams = [
            FuturesWebSocket.create_kline_stream(self.symbol, self.interval),
            FuturesWebSocket.create_mark_price_stream(self.symbol),
            FuturesWebSocket.create_book_ticker_stream(self.symbol),
        ]

        self.ws = FuturesWebSocket(
            streams=streams,
            on_message_callback=self._handle_message,
            on_error_callback=self._handle_error,
            testnet=self.testnet
        )

        self.ws.connect()

    def stop(self):
        """실시간 데이터 수신 중지"""
        if self.ws:
            self.ws.close()
        logger.info(f"실시간 데이터 수신 중지: {self.symbol}")

    def check_and_reconnect(self, stale_threshold_seconds: float = 60) -> bool:
        """
        데이터 신선도 확인 후 필요시 재연결.

        Args:
            stale_threshold_seconds: 이 시간(초) 이상 데이터가 없으면 재연결

        Returns:
            재연결 수행 여부
        """
        if self.last_update_time == 0:
            return False

        elapsed = time.time() - self.last_update_time
        if elapsed < stale_threshold_seconds:
            # WS 정상 — 연결 복구 콜백 (단절→정상 전환 시)
            if not self._ws_connected:
                self._ws_connected = True
                if self._on_ws_reconnect_cb:
                    try:
                        self._on_ws_reconnect_cb()
                    except Exception as e:
                        logger.error(f"WS reconnect 콜백 오류: {e}")
            return False

        logger.warning(f"⚠️ WebSocket 데이터 {elapsed:.0f}초 동안 수신 없음 → 재연결 시도")

        # WS 단절 콜백 (정상→단절 전환 시)
        if self._ws_connected:
            self._ws_connected = False
            if self._on_ws_disconnect_cb:
                try:
                    self._on_ws_disconnect_cb()
                except Exception as e:
                    logger.error(f"WS disconnect 콜백 오류: {e}")

        try:
            # 기존 연결 종료
            if self.ws:
                try:
                    self.ws.close()
                except Exception:
                    pass

            time.sleep(2)

            # 재연결
            self._connect_ws()
            logger.info("✅ WebSocket 재연결 완료")
            return True

        except Exception as e:
            logger.error(f"WebSocket 재연결 실패: {e}")
            return False

    def _handle_message(self, data: Dict[str, Any]):
        """WebSocket 메시지 처리"""
        self.message_count += 1
        _mp_val: Optional[float] = None  # markPrice 콜백 센티널

        try:
            # 복합 스트림 처리
            if 'stream' in data:
                stream_name = data['stream']
                payload = data['data']
            else:
                stream_name = data.get('e', 'unknown')
                payload = data

            event_type = payload.get('e', stream_name)

            with self._lock:
                if event_type == 'kline':
                    self._process_kline(payload)
                elif event_type == 'markPriceUpdate':
                    self._process_mark_price(payload)
                    _mp_val = self.mark_price  # lock 안에서 캡처
                elif event_type == 'bookTicker':
                    self._process_book_ticker(payload)

            self.last_update_time = time.time()

            # WS 연결 상태 추적
            if not self._ws_connected:
                self._ws_connected = True

            # markPrice 콜백을 lock 밖에서 호출 (데드락 방지)
            if _mp_val is not None:
                for cb in self._mark_price_callbacks:
                    try:
                        cb(_mp_val)
                    except Exception as cb_err:
                        logger.error(f"markPrice 콜백 오류: {cb_err}")

        except Exception as e:
            logger.error(f"메시지 처리 오류: {e}")
            self.error_count += 1

    def _process_kline(self, data: Dict):
        """캔들스틱 데이터 처리"""
        kline = data['k']

        candle = {
            'timestamp': kline['t'],
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'is_closed': kline['x']
        }

        if kline['x']:  # 캔들 완성
            self.candles.append(candle)
            self.current_candle = None
            logger.debug(f"캔들 완성: {candle['close']}")
        else:
            self.current_candle = candle

    def _process_mark_price(self, data: Dict):
        """마크 가격 처리 + 10초 마이크로캔들 집계"""
        self.mark_price = float(data['p'])
        self.funding_rate = float(data.get('r', 0))

        # 틱 데이터 수집기에 전달
        timestamp_ms = int(data.get('E', time.time() * 1000))
        self.tick_collector.on_mark_price(self.mark_price, timestamp_ms)

    def _process_book_ticker(self, data: Dict):
        """최적 호가 (bookTicker) 처리 → OBI 계산용 bid/ask qty 저장"""
        self._best_bid_price = float(data.get('b', 0))
        self._best_bid_qty = float(data.get('B', 0))
        self._best_ask_price = float(data.get('a', 0))
        self._best_ask_qty = float(data.get('A', 0))
        self._book_ticker_update_time = time.time()

    def get_book_ticker(self) -> Dict[str, float]:
        """최적 호가 데이터 반환 (OBI 계산용)

        Returns:
            {'bid_qty': float, 'ask_qty': float,
             'bid_price': float, 'ask_price': float,
             'update_time': float}
        """
        with self._lock:
            return {
                'bid_qty': self._best_bid_qty,
                'ask_qty': self._best_ask_qty,
                'bid_price': self._best_bid_price,
                'ask_price': self._best_ask_price,
                'update_time': self._book_ticker_update_time,
            }

    def _handle_error(self, error: Exception):
        """에러 처리"""
        self.error_count += 1
        logger.error(f"WebSocket 에러: {error}")

    def get_current_price(self) -> float:
        """현재 가격 반환"""
        with self._lock:
            if self.current_candle:
                return self.current_candle['close']
            elif self.candles:
                return self.candles[-1]['close']
            return self.mark_price

    def get_mark_price(self) -> float:
        """마크 가격 반환"""
        return self.mark_price

    def get_freshest_price(self) -> tuple:
        """최신 마크가격과 나이(초) 반환. (mark_price, age_seconds).

        WebSocket markPrice 스트림의 최신 값과
        마지막 업데이트로부터 경과 시간을 반환합니다.
        아직 가격이 수신되지 않았으면 (0.0, inf)를 반환합니다.
        """
        with self._lock:
            if self.mark_price <= 0 or self.last_update_time == 0:
                return 0.0, float('inf')
            age = time.time() - self.last_update_time
            return self.mark_price, age

    def get_funding_rate(self) -> float:
        """펀딩비율 반환"""
        return self.funding_rate

    def get_candles_df(self) -> 'pd.DataFrame':
        """캔들 데이터를 DataFrame으로 반환"""
        import pandas as pd

        with self._lock:
            candles_list = list(self.candles)
            if self.current_candle:
                candles_list.append(self.current_candle)

        if not candles_list:
            return pd.DataFrame()

        df = pd.DataFrame(candles_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    def get_tick_candles_df(self) -> 'pd.DataFrame':
        """10초 마이크로캔들 DataFrame 반환"""
        return self.tick_collector.get_candles_df()

    def get_status(self) -> Dict:
        """상태 정보 반환"""
        return {
            'symbol': self.symbol,
            'interval': self.interval,
            'candle_count': len(self.candles),
            'tick_candle_count': self.tick_collector.get_candle_count(),
            'current_price': self.get_current_price(),
            'mark_price': self.mark_price,
            'funding_rate': self.funding_rate,
            'message_count': self.message_count,
            'error_count': self.error_count,
            'last_update': self.last_update_time,
            'is_connected': self.ws.is_running if self.ws else False
        }


class UserDataStream:
    """
    사용자 데이터 스트림 (계정 업데이트, 주문 업데이트)
    """

    def __init__(
        self,
        futures_client,
        on_account_update: Optional[Callable] = None,
        on_order_update: Optional[Callable] = None,
        testnet: bool = True
    ):
        """
        Args:
            futures_client: BinanceFuturesClient 인스턴스
            on_account_update: 계정 업데이트 콜백
            on_order_update: 주문 업데이트 콜백
            testnet: 테스트넷 사용 여부
        """
        self.client = futures_client
        self.on_account_update = on_account_update
        self.on_order_update = on_order_update
        self.testnet = testnet

        self.listen_key: Optional[str] = None
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.keepalive_thread: Optional[threading.Thread] = None
        self.is_running = False

        if testnet:
            self.base_url = "wss://stream.binancefuture.com/ws"
        else:
            self.base_url = "wss://fstream.binance.com/ws"

        logger.info("UserDataStream 초기화")

    def start(self):
        """사용자 데이터 스트림 시작"""
        # Listen Key 생성
        self.listen_key = self._create_listen_key()
        if not self.listen_key:
            logger.error("Listen Key 생성 실패")
            return

        self.is_running = True

        # WebSocket 연결
        url = f"{self.base_url}/{self.listen_key}"

        self.ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )

        self.ws_thread = threading.Thread(target=self._run_forever, daemon=True)
        self.ws_thread.start()

        # Keep-alive 스레드
        self.keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self.keepalive_thread.start()

        logger.info("사용자 데이터 스트림 시작")

    def stop(self):
        """사용자 데이터 스트림 중지"""
        self.is_running = False

        if self.ws:
            self.ws.close()

        if self.listen_key:
            self._delete_listen_key()

        logger.info("사용자 데이터 스트림 중지")

    def _create_listen_key(self) -> Optional[str]:
        """Listen Key 생성"""
        try:
            response = self.client._request('POST', '/fapi/v1/listenKey', signed=True)
            return response.get('listenKey')
        except Exception as e:
            logger.error(f"Listen Key 생성 실패: {e}")
            return None

    def _extend_listen_key(self):
        """Listen Key 연장"""
        try:
            self.client._request('PUT', '/fapi/v1/listenKey', signed=True)
            logger.debug("Listen Key 연장 성공")
        except Exception as e:
            logger.error(f"Listen Key 연장 실패: {e}")

    def _delete_listen_key(self):
        """Listen Key 삭제"""
        try:
            self.client._request('DELETE', '/fapi/v1/listenKey', signed=True)
            logger.debug("Listen Key 삭제 성공")
        except Exception as e:
            logger.error(f"Listen Key 삭제 실패: {e}")

    def _keepalive_loop(self):
        """Keep-alive 루프 (30분마다)"""
        while self.is_running:
            time.sleep(30 * 60)  # 30분
            if self.is_running:
                self._extend_listen_key()

    def _run_forever(self):
        """WebSocket 실행"""
        try:
            self.ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            logger.error(f"UserDataStream 오류: {e}")

    def _on_message(self, ws, message: str):
        """메시지 처리"""
        try:
            data = json.loads(message)
            event_type = data.get('e')

            if event_type == 'ACCOUNT_UPDATE':
                if self.on_account_update:
                    self.on_account_update(data)
            elif event_type == 'ORDER_TRADE_UPDATE':
                if self.on_order_update:
                    self.on_order_update(data)
            elif event_type == 'MARGIN_CALL':
                logger.warning(f"⚠️ 마진 콜 경고: {data}")

        except Exception as e:
            logger.error(f"UserDataStream 메시지 처리 오류: {e}")

    def _on_error(self, ws, error):
        """에러 처리"""
        logger.error(f"UserDataStream 오류: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """연결 종료"""
        logger.warning(f"UserDataStream 연결 종료: {close_status_code}")

    def _on_open(self, ws):
        """연결 성공"""
        logger.info("UserDataStream 연결 성공")
