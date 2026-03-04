# 바이낸스 기반 AI 자동거래 시스템 구축 가이드

> **참조 문서**: [Binance Developers Documentation](https://developers.binance.com/docs)

---

## 📋 목차

1. [시스템 개요](#시스템-개요)
2. [바이낸스 API 통합](#바이낸스-api-통합)
3. [AI 모델 아키텍처](#ai-모델-아키텍처)
4. [전략 시스템](#전략-시스템)
5. [전략 스위칭 메커니즘](#전략-스위칭-메커니즘)
6. [구현 체크리스트](#구현-체크리스트)

---

## 🎯 시스템 개요

### 핵심 목표
- **바이낸스 거래소** 기반 실시간 암호화폐 자동거래
- **RNN/LSTM 모델**을 활용한 가격 예측 및 거래 신호 생성
- **다중 전략** 운용 및 **동적 전략 스위칭**
- **리스크 관리** 및 **포트폴리오 최적화**

### 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    바이낸스 거래소                              │
│              (Spot & Futures Trading)                        │
└────────────┬────────────────────────────────┬────────────────┘
             │                                │
        [REST API]                      [WebSocket API]
             │                                │
             ▼                                ▼
┌─────────────────────────────────────────────────────────────┐
│                  데이터 수집 레이어                             │
│  ├─ Market Data Collector (OHLCV, Order Book, Trades)       │
│  ├─ Account Data Collector (Balance, Positions, Orders)     │
│  └─ Data Preprocessor (Normalization, Feature Engineering)  │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                  데이터 저장소                                 │
│  ├─ TimescaleDB (시계열 데이터: OHLCV, Indicators)            │
│  ├─ Redis (실시간 캐싱: Live prices, Signals)                │
│  └─ PostgreSQL (거래 기록, 포트폴리오, 전략 성과)               │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                  AI 모델 레이어                                │
│  ├─ RNN/LSTM Price Prediction Models                        │
│  ├─ Multi-Timeframe Models (1h, 4h, 1d)                     │
│  ├─ Ensemble Models (Weighted Average, Stacking)            │
│  └─ Reinforcement Learning Agent (Strategy Selector)        │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                  전략 레이어                                   │
│  ├─ Technical Strategies (MA, RSI, MACD, BB)                │
│  ├─ ML-Based Strategies (LSTM Signals)                      │
│  ├─ Market-Neutral Strategies (Arbitrage, Mean Reversion)   │
│  └─ Strategy Performance Tracker                            │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│              전략 스위칭 시스템                                 │
│  ├─ Market Regime Detector (Trending/Range/Volatile)        │
│  ├─ Performance-Based Selector (Sharpe, Win Rate)           │
│  └─ Meta-Learning Agent (RL-based Strategy Allocation)      │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│              실행 및 리스크 관리                                │
│  ├─ Order Manager (Market, Limit, Stop Orders)              │
│  ├─ Position Manager (Sizing, Leverage Control)             │
│  ├─ Risk Manager (Stop Loss, Daily Loss Limit)              │
│  └─ Portfolio Manager (Rebalancing, Diversification)        │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│            모니터링 및 알림 시스템                              │
│  ├─ Grafana Dashboard (실시간 성과 모니터링)                   │
│  ├─ Prometheus Metrics (시스템 상태 추적)                     │
│  └─ Slack/Telegram Alerts (거래 알림, 에러 통지)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔌 바이낸스 API 통합

### 1. API 기본 정보

**공식 문서**: [Binance REST API](https://developers.binance.com/docs/binance-spot-api-docs/rest-api)

#### Base Endpoints
```python
# Production 엔드포인트
SPOT_BASE_URL = "https://api.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"

# 대체 엔드포인트 (성능 향상, 안정성 낮음)
ALTERNATIVE_URLS = [
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api-gcp.binance.com"
]

# 공개 데이터 전용 (인증 불필요)
PUBLIC_DATA_URL = "https://data-api.binance.vision"
```

#### WebSocket Endpoints
**공식 문서**: [WebSocket Streams](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams)

```python
# 실시간 시장 데이터 스트림
WSS_SPOT_STREAM = "wss://stream.binance.com:9443/ws"
WSS_SPOT_STREAM_ALT = "wss://stream.binance.com:443/ws"

# 복합 스트림 (여러 심볼 동시 구독)
WSS_COMBINED_STREAM = "wss://stream.binance.com:9443/stream"
```

### 2. 인증 및 API 키 관리

#### API 키 생성
1. 바이낸스 계정 로그인
2. **API Management** 페이지 접속
3. **Create API** 클릭
4. API 키 이름 설정 (예: "AutoTradingBot")
5. **Enable Reading**, **Enable Spot & Margin Trading** 권한 설정
6. **Restrict access to trusted IPs only** (보안 강화)

#### 인증 방식
**공식 문서**: [Authentication Methods](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/authentication)

바이낸스는 3가지 서명 방식 지원:
- **HMAC SHA256** (가장 일반적)
- **RSA**
- **Ed25519**

```python
# HMAC SHA256 서명 예제
import hmac
import hashlib
import time
import requests

API_KEY = "your_api_key"
SECRET_KEY = "your_secret_key"
BASE_URL = "https://api.binance.com"

def create_signature(query_string):
    return hmac.new(
        SECRET_KEY.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def get_account_info():
    endpoint = "/api/v3/account"
    timestamp = int(time.time() * 1000)

    params = {
        'timestamp': timestamp,
        'recvWindow': 5000  # 요청 유효 시간 (밀리초)
    }

    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = create_signature(query_string)

    url = f"{BASE_URL}{endpoint}?{query_string}&signature={signature}"

    headers = {
        'X-MBX-APIKEY': API_KEY
    }

    response = requests.get(url, headers=headers)
    return response.json()
```

#### 환경 변수 관리
```python
# .env 파일
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here
BINANCE_TESTNET=False  # True면 테스트넷 사용

# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv('BINANCE_API_KEY')
    SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
    IS_TESTNET = os.getenv('BINANCE_TESTNET', 'False') == 'True'

    BASE_URL = "https://testnet.binance.vision" if IS_TESTNET else "https://api.binance.com"
```

### 3. API Rate Limits

**공식 문서**: [Rate Limits](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/limits)

#### 제한 사항
```python
# IP 기반 제한
RATE_LIMITS = {
    'requests_per_minute': 1200,      # 분당 요청 수
    'orders_per_second': 10,          # 초당 주문 수
    'orders_per_day': 100000,         # 일일 주문 수
    'weight_per_minute': 6000         # 가중치 기반 제한
}

# 응답 헤더를 통한 현재 사용량 확인
# X-MBX-USED-WEIGHT-1M: 1분간 사용된 가중치
# X-MBX-ORDER-COUNT-10S: 10초간 주문 수
# X-MBX-ORDER-COUNT-1M: 1분간 주문 수
```

#### Rate Limit 처리 전략
```python
import time
from collections import deque

class RateLimiter:
    def __init__(self, max_requests=1200, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()

    def can_proceed(self):
        current_time = time.time()

        # 시간 윈도우 밖의 요청 제거
        while self.requests and self.requests[0] < current_time - self.time_window:
            self.requests.popleft()

        return len(self.requests) < self.max_requests

    def record_request(self):
        self.requests.append(time.time())

    def wait_if_needed(self):
        if not self.can_proceed():
            oldest_request = self.requests[0]
            wait_time = self.time_window - (time.time() - oldest_request)
            if wait_time > 0:
                time.sleep(wait_time + 0.1)  # 0.1초 버퍼

        self.record_request()

# 사용 예제
rate_limiter = RateLimiter(max_requests=1200, time_window=60)

def safe_api_call(func, *args, **kwargs):
    rate_limiter.wait_if_needed()
    return func(*args, **kwargs)
```

#### 429 에러 처리 (Rate Limit Exceeded)
```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_session_with_retry():
    session = requests.Session()

    retry_strategy = Retry(
        total=5,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "OPTIONS"],
        backoff_factor=2  # 2, 4, 8, 16, 32초 대기
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    return session
```

### 4. 실시간 데이터 수집 (WebSocket)

**공식 문서**: [WebSocket Streams](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams)

#### WebSocket 스트림 종류

```python
# 1. Aggregate Trade Streams (집계 거래)
# 동일 주문에서 발생한 여러 거래를 하나로 집계
STREAM_AGGTRADE = "{symbol}@aggTrade"  # 예: btcusdt@aggTrade

# 2. Trade Streams (개별 거래)
STREAM_TRADE = "{symbol}@trade"

# 3. Kline/Candlestick Streams (캔들스틱)
STREAM_KLINE = "{symbol}@kline_{interval}"
# interval: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

# 4. Individual Symbol Mini Ticker (24시간 미니 통계)
STREAM_MINI_TICKER = "{symbol}@miniTicker"

# 5. Individual Symbol Ticker (24시간 전체 통계)
STREAM_TICKER = "{symbol}@ticker"

# 6. Individual Symbol Book Ticker (최적 호가)
STREAM_BOOK_TICKER = "{symbol}@bookTicker"

# 7. Partial Book Depth Streams (부분 호가창)
STREAM_DEPTH = "{symbol}@depth{levels}"  # levels: 5, 10, 20
STREAM_DEPTH_UPDATE = "{symbol}@depth@100ms"  # 100ms 또는 1000ms

# 8. Diff Depth Stream (호가창 변경 사항)
STREAM_DIFF_DEPTH = "{symbol}@depth"
```

#### WebSocket 연결 관리
```python
import websocket
import json
import threading

class BinanceWebSocket:
    def __init__(self, streams, on_message_callback):
        """
        streams: 구독할 스트림 리스트 (예: ['btcusdt@kline_1m', 'ethusdt@kline_1m'])
        on_message_callback: 메시지 수신 시 호출할 콜백 함수
        """
        self.streams = streams
        self.on_message_callback = on_message_callback
        self.ws = None

        # 복합 스트림 URL 생성
        stream_path = '/'.join(streams)
        self.url = f"wss://stream.binance.com:9443/stream?streams={stream_path}"

    def on_message(self, ws, message):
        data = json.loads(message)
        self.on_message_callback(data)

    def on_error(self, ws, error):
        print(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket Closed: {close_status_code} - {close_msg}")
        # 자동 재연결
        self.reconnect()

    def on_open(self, ws):
        print(f"WebSocket Connected: {self.streams}")

    def reconnect(self):
        print("Reconnecting WebSocket...")
        time.sleep(5)
        self.connect()

    def connect(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )

        # 별도 스레드에서 실행
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

    def close(self):
        if self.ws:
            self.ws.close()

# 사용 예제
def handle_kline_message(data):
    """캔들스틱 데이터 처리"""
    if 'data' in data:
        kline = data['data']['k']
        print(f"Symbol: {kline['s']}, Close: {kline['c']}, Volume: {kline['v']}")

streams = ['btcusdt@kline_1m', 'ethusdt@kline_1m', 'bnbusdt@kline_1m']
ws_client = BinanceWebSocket(streams, handle_kline_message)
ws_client.connect()
```

#### 로컬 Order Book 관리
**공식 문서**: [Managing Local Order Book](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams#how-to-manage-a-local-order-book-correctly)

```python
import requests
from collections import OrderedDict

class OrderBookManager:
    def __init__(self, symbol):
        self.symbol = symbol
        self.bids = OrderedDict()  # {price: quantity}
        self.asks = OrderedDict()
        self.last_update_id = 0
        self.initialized = False
        self.buffer = []  # 초기화 전 메시지 버퍼

    def initialize(self):
        """REST API로 초기 스냅샷 가져오기"""
        url = f"https://api.binance.com/api/v3/depth?symbol={self.symbol}&limit=1000"
        response = requests.get(url)
        data = response.json()

        self.last_update_id = data['lastUpdateId']

        # 초기 호가 설정
        for price, qty in data['bids']:
            self.bids[float(price)] = float(qty)

        for price, qty in data['asks']:
            self.asks[float(price)] = float(qty)

        self.initialized = True

        # 버퍼된 메시지 처리
        for buffered_data in self.buffer:
            self.update(buffered_data)
        self.buffer.clear()

    def update(self, depth_data):
        """Diff Depth Stream으로부터 업데이트"""
        if not self.initialized:
            self.buffer.append(depth_data)
            return

        # U: 첫 번째 업데이트 ID, u: 마지막 업데이트 ID
        first_update_id = depth_data['U']
        last_update_id = depth_data['u']

        # 중복 또는 오래된 업데이트 무시
        if last_update_id <= self.last_update_id:
            return

        # 연속성 확인
        if first_update_id > self.last_update_id + 1:
            print(f"Order book out of sync. Reinitializing...")
            self.initialized = False
            self.initialize()
            return

        # Bids 업데이트
        for price, qty in depth_data['b']:
            price, qty = float(price), float(qty)
            if qty == 0:
                self.bids.pop(price, None)  # 수량 0이면 제거
            else:
                self.bids[price] = qty

        # Asks 업데이트
        for price, qty in depth_data['a']:
            price, qty = float(price), float(qty)
            if qty == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

        self.last_update_id = last_update_id

        # 정렬 유지 (가격 순)
        self.bids = OrderedDict(sorted(self.bids.items(), reverse=True))
        self.asks = OrderedDict(sorted(self.asks.items()))

    def get_best_bid(self):
        """최고 매수 호가"""
        if self.bids:
            return list(self.bids.items())[0]
        return None, None

    def get_best_ask(self):
        """최저 매도 호가"""
        if self.asks:
            return list(self.asks.items())[0]
        return None, None

    def get_spread(self):
        """스프레드 계산"""
        best_bid, _ = self.get_best_bid()
        best_ask, _ = self.get_best_ask()
        if best_bid and best_ask:
            return best_ask - best_bid
        return None
```

### 5. 주문 실행

**공식 문서**: [New Order](https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api)

#### 주문 타입
```python
ORDER_TYPES = {
    'MARKET': '시장가 주문 (즉시 체결)',
    'LIMIT': '지정가 주문 (특정 가격에만 체결)',
    'STOP_LOSS': '손절 주문',
    'STOP_LOSS_LIMIT': '손절 지정가 주문',
    'TAKE_PROFIT': '익절 주문',
    'TAKE_PROFIT_LIMIT': '익절 지정가 주문',
    'LIMIT_MAKER': 'Maker로만 체결되는 지정가 주문'
}

TIME_IN_FORCE = {
    'GTC': 'Good Till Cancel (취소할 때까지 유효)',
    'IOC': 'Immediate or Cancel (즉시 체결 또는 취소)',
    'FOK': 'Fill or Kill (전량 체결 또는 취소)'
}
```

#### 주문 실행 클래스
```python
import hmac
import hashlib
import time
import requests

class BinanceOrderExecutor:
    def __init__(self, api_key, secret_key, base_url="https://api.binance.com"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.session = create_session_with_retry()

    def _create_signature(self, params):
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def place_market_order(self, symbol, side, quantity):
        """
        시장가 주문
        side: 'BUY' 또는 'SELL'
        quantity: 주문 수량
        """
        endpoint = "/api/v3/order"

        params = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'type': 'MARKET',
            'quantity': quantity,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }

        params['signature'] = self._create_signature(params)

        headers = {'X-MBX-APIKEY': self.api_key}

        response = self.session.post(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=headers
        )

        return response.json()

    def place_limit_order(self, symbol, side, quantity, price, time_in_force='GTC'):
        """
        지정가 주문
        price: 주문 가격
        time_in_force: 'GTC', 'IOC', 'FOK'
        """
        endpoint = "/api/v3/order"

        params = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'type': 'LIMIT',
            'timeInForce': time_in_force,
            'quantity': quantity,
            'price': price,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }

        params['signature'] = self._create_signature(params)

        headers = {'X-MBX-APIKEY': self.api_key}

        response = self.session.post(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=headers
        )

        return response.json()

    def place_stop_loss_limit_order(self, symbol, side, quantity, price, stop_price):
        """
        손절 지정가 주문
        price: 체결 희망 가격
        stop_price: 손절 트리거 가격
        """
        endpoint = "/api/v3/order"

        params = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'type': 'STOP_LOSS_LIMIT',
            'timeInForce': 'GTC',
            'quantity': quantity,
            'price': price,
            'stopPrice': stop_price,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }

        params['signature'] = self._create_signature(params)

        headers = {'X-MBX-APIKEY': self.api_key}

        response = self.session.post(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=headers
        )

        return response.json()

    def cancel_order(self, symbol, order_id):
        """주문 취소"""
        endpoint = "/api/v3/order"

        params = {
            'symbol': symbol.upper(),
            'orderId': order_id,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }

        params['signature'] = self._create_signature(params)

        headers = {'X-MBX-APIKEY': self.api_key}

        response = self.session.delete(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=headers
        )

        return response.json()

    def get_open_orders(self, symbol=None):
        """미체결 주문 조회"""
        endpoint = "/api/v3/openOrders"

        params = {
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }

        if symbol:
            params['symbol'] = symbol.upper()

        params['signature'] = self._create_signature(params)

        headers = {'X-MBX-APIKEY': self.api_key}

        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=headers
        )

        return response.json()

    def get_account_info(self):
        """계정 정보 조회 (잔고 등)"""
        endpoint = "/api/v3/account"

        params = {
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }

        params['signature'] = self._create_signature(params)

        headers = {'X-MBX-APIKEY': self.api_key}

        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=headers
        )

        return response.json()
```

### 6. 파이썬 라이브러리 활용

**공식 SDK**: [binance-connector-python](https://github.com/binance/binance-connector-python)

```bash
# 설치
pip install binance-connector
```

```python
from binance.spot import Spot
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient

# REST API 클라이언트
client = Spot(api_key='your_api_key', api_secret='your_secret_key')

# 계정 정보
account_info = client.account()

# 시장가 주문
order = client.new_order(
    symbol='BTCUSDT',
    side='BUY',
    type='MARKET',
    quantity=0.001
)

# WebSocket 클라이언트
def message_handler(_, message):
    print(message)

ws_client = SpotWebsocketStreamClient(on_message=message_handler)
ws_client.kline(symbol='btcusdt', interval='1m')
```

---

## 🧠 AI 모델 아키텍처

### 1. RNN/LSTM 모델 설계

#### 1.1 모델 구조

```python
import torch
import torch.nn as nn

class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        """
        input_size: 입력 특징 개수 (예: 20개 기술 지표)
        hidden_size: LSTM 은닉 유닛 개수
        num_layers: LSTM 레이어 개수
        dropout: 드롭아웃 비율
        """
        super(LSTMPricePredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Fully Connected 레이어
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)  # 가격 예측 (회귀)

    def forward(self, x):
        """
        x: (batch_size, sequence_length, input_size)
        """
        # LSTM 출력
        lstm_out, _ = self.lstm(x)

        # 마지막 타임스텝의 출력만 사용
        last_output = lstm_out[:, -1, :]

        # Fully Connected
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class LSTMDirectionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        """가격 방향 예측 (상승/하락/횡보)"""
        super(LSTMDirectionClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 3)  # 3개 클래스: 상승(2), 횡보(1), 하락(0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]

        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)

        return out


class MultiTimeframeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        """다중 타임프레임 LSTM (1h, 4h, 1d)"""
        super(MultiTimeframeLSTM, self).__init__()

        # 각 타임프레임별 LSTM
        self.lstm_1h = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_4h = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_1d = nn.LSTM(input_size, hidden_size, batch_first=True)

        # 통합 레이어
        self.fc = nn.Linear(hidden_size * 3, 1)

    def forward(self, x_1h, x_4h, x_1d):
        """
        x_1h: 1시간 봉 데이터
        x_4h: 4시간 봉 데이터
        x_1d: 1일 봉 데이터
        """
        _, (h_1h, _) = self.lstm_1h(x_1h)
        _, (h_4h, _) = self.lstm_4h(x_4h)
        _, (h_1d, _) = self.lstm_1d(x_1d)

        # 마지막 은닉 상태 결합
        combined = torch.cat([h_1h[-1], h_4h[-1], h_1d[-1]], dim=1)

        output = self.fc(combined)
        return output
```

#### 1.2 특징 엔지니어링

```python
import pandas as pd
import talib

class FeatureEngineering:
    @staticmethod
    def calculate_technical_indicators(df):
        """
        기술 지표 계산
        df: OHLCV 데이터프레임
        """
        # 이동평균
        df['MA_7'] = talib.SMA(df['close'], timeperiod=7)
        df['MA_25'] = talib.SMA(df['close'], timeperiod=25)
        df['MA_99'] = talib.SMA(df['close'], timeperiod=99)

        # 지수이동평균
        df['EMA_12'] = talib.EMA(df['close'], timeperiod=12)
        df['EMA_26'] = talib.EMA(df['close'], timeperiod=26)

        # 볼린저 밴드
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
            df['close'], timeperiod=20
        )
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

        # RSI (Relative Strength Index)
        df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)

        # MACD
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )

        # Stochastic
        df['STOCH_k'], df['STOCH_d'] = talib.STOCH(
            df['high'], df['low'], df['close']
        )

        # ATR (Average True Range)
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        # ADX (Average Directional Index)
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # OBV (On Balance Volume)
        df['OBV'] = talib.OBV(df['close'], df['volume'])

        # 가격 변화율
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # 변동성
        df['volatility'] = df['returns'].rolling(window=20).std()

        # 거래량 이동평균
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        return df

    @staticmethod
    def create_sequences(data, lookback=60, forecast_horizon=1):
        """
        시계열 데이터를 LSTM 입력 형태로 변환
        lookback: 과거 몇 개 시점을 볼 것인가
        forecast_horizon: 몇 시점 후를 예측할 것인가
        """
        X, y = [], []

        for i in range(lookback, len(data) - forecast_horizon + 1):
            X.append(data[i-lookback:i])
            y.append(data[i+forecast_horizon-1]['close'])  # 종가 예측

        return np.array(X), np.array(y)

    @staticmethod
    def normalize_data(df, columns):
        """데이터 정규화"""
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])

        return df, scaler
```

#### 1.3 학습 파이프라인

```python
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class CryptoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ModelTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.best_loss = float('inf')

    def train(self, train_loader, val_loader, epochs=100, lr=0.001):
        # 손실 함수 및 옵티마이저
        criterion = nn.HuberLoss()  # 이상치에 강건
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Early Stopping
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # 학습 모드
            self.model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                predictions = self.model(X_batch)
                loss = criterion(predictions.squeeze(), y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()

            # 검증
            val_loss = self.validate(val_loader, criterion)

            # 학습률 조정
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")

            # Early Stopping 체크
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                patience_counter = 0
                # 최적 모델 저장
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    def validate(self, val_loader, criterion):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(X_batch)
                loss = criterion(predictions.squeeze(), y_batch)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()


# 학습 실행 예제
def train_model(df):
    # 특징 엔지니어링
    fe = FeatureEngineering()
    df = fe.calculate_technical_indicators(df)

    # 결측치 제거
    df = df.dropna()

    # 특징 선택
    feature_columns = [
        'close', 'volume', 'MA_7', 'MA_25', 'RSI_14', 'MACD',
        'MACD_signal', 'BB_width', 'ATR', 'ADX', 'volatility'
    ]

    # 정규화
    df_normalized, scaler = fe.normalize_data(df, feature_columns)

    # 시퀀스 생성
    X, y = fe.create_sequences(
        df_normalized[feature_columns].values,
        lookback=60,
        forecast_horizon=1
    )

    # Train/Val/Test 분할
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, shuffle=False)

    # DataLoader 생성
    train_dataset = CryptoDataset(X_train, y_train)
    val_dataset = CryptoDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 모델 초기화
    model = LSTMPricePredictor(input_size=len(feature_columns), hidden_size=128, num_layers=2)

    # 학습
    trainer = ModelTrainer(model)
    trainer.train(train_loader, val_loader, epochs=100, lr=0.001)

    return model, scaler
```

#### 1.4 온라인 학습 (Incremental Learning)

```python
class OnlineLearner:
    def __init__(self, model, lr=0.0001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.HuberLoss()
        self.update_counter = 0

    def update(self, X_new, y_new):
        """
        새로운 데이터로 모델 업데이트
        X_new: 새로운 입력 데이터
        y_new: 새로운 타겟 값
        """
        self.model.train()

        X_tensor = torch.FloatTensor(X_new).unsqueeze(0)
        y_tensor = torch.FloatTensor([y_new])

        prediction = self.model(X_tensor)
        loss = self.criterion(prediction.squeeze(), y_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1

        # 100번마다 모델 저장
        if self.update_counter % 100 == 0:
            torch.save(self.model.state_dict(), f'online_model_{self.update_counter}.pth')

        return loss.item()

    def should_retrain(self, recent_losses, threshold=0.05):
        """
        성능 저하 감지 → 전체 재학습 트리거
        recent_losses: 최근 N개의 loss 값들
        threshold: 재학습 트리거 임계값
        """
        if len(recent_losses) < 50:
            return False

        avg_loss = np.mean(recent_losses)
        return avg_loss > threshold
```

### 2. 앙상블 모델

```python
class EnsemblePredictor:
    def __init__(self, models, weights=None):
        """
        models: 모델 리스트 [model_1h, model_4h, model_1d]
        weights: 가중치 리스트 (None이면 균등 가중)
        """
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)

    def predict(self, X_list):
        """
        X_list: 각 모델에 대한 입력 리스트
        """
        predictions = []

        for model, X in zip(self.models, X_list):
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).unsqueeze(0)
                pred = model(X_tensor).item()
                predictions.append(pred)

        # 가중 평균
        ensemble_pred = np.average(predictions, weights=self.weights)

        return ensemble_pred, predictions

    def update_weights(self, recent_performances):
        """
        성능 기반 가중치 업데이트
        recent_performances: 각 모델의 최근 성능 (MAE, RMSE 등)
        """
        # 성능이 좋을수록 높은 가중치 (역수 사용)
        inverse_errors = [1/max(perf, 0.001) for perf in recent_performances]
        total = sum(inverse_errors)
        self.weights = [inv/total for inv in inverse_errors]
```

---

## 📊 전략 시스템

### 1. 전략 인터페이스

```python
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, name):
        self.name = name
        self.trades = []
        self.performance_metrics = {}

    @abstractmethod
    def generate_signal(self, market_data, portfolio):
        """
        거래 신호 생성
        Returns: -1 (매도), 0 (관망), 1 (매수)
        """
        pass

    @abstractmethod
    def calculate_position_size(self, signal, portfolio, risk_params):
        """
        포지션 크기 계산
        Returns: 주문 수량
        """
        pass

    def record_trade(self, trade):
        """거래 기록"""
        self.trades.append(trade)

    def evaluate_performance(self, time_window='7d'):
        """전략 성과 평가"""
        recent_trades = self._get_recent_trades(time_window)

        if not recent_trades:
            return None

        returns = [t['pnl'] for t in recent_trades]

        metrics = {
            'total_return': sum(returns),
            'win_rate': len([r for r in returns if r > 0]) / len(returns),
            'avg_win': np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0,
            'avg_loss': np.mean([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0,
            'profit_factor': abs(sum([r for r in returns if r > 0]) / sum([r for r in returns if r < 0])) if any(r < 0 for r in returns) else float('inf'),
            'sharpe_ratio': self._calculate_sharpe(returns),
            'max_drawdown': self._calculate_max_drawdown(returns)
        }

        self.performance_metrics = metrics
        return metrics

    def _calculate_sharpe(self, returns, risk_free_rate=0.0):
        """샤프 비율 계산"""
        if len(returns) < 2:
            return 0

        excess_returns = np.array(returns) - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0

    def _calculate_max_drawdown(self, returns):
        """최대 낙폭 계산"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return np.min(drawdown) if len(drawdown) > 0 else 0

    def _get_recent_trades(self, time_window):
        """최근 거래 가져오기"""
        # 시간 윈도우 파싱 (예: '7d', '24h')
        # 실제 구현 시 타임스탬프 필터링
        return self.trades[-100:]  # 간단히 최근 100개
```

### 2. 기술적 분석 전략

```python
class MovingAverageCrossover(BaseStrategy):
    def __init__(self, fast_period=7, slow_period=25):
        super().__init__(name=f"MA_Cross_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signal(self, market_data, portfolio):
        """
        이동평균 크로스오버 전략
        빠른 MA가 느린 MA를 상향 돌파: 매수 신호
        빠른 MA가 느린 MA를 하향 돌파: 매도 신호
        """
        df = market_data

        # 이동평균 계산
        fast_ma = df['close'].rolling(window=self.fast_period).mean()
        slow_ma = df['close'].rolling(window=self.slow_period).mean()

        # 현재 및 이전 값
        fast_current = fast_ma.iloc[-1]
        fast_prev = fast_ma.iloc[-2]
        slow_current = slow_ma.iloc[-1]
        slow_prev = slow_ma.iloc[-2]

        # 크로스오버 감지
        if fast_prev <= slow_prev and fast_current > slow_current:
            return 1  # 골든 크로스 (매수)
        elif fast_prev >= slow_prev and fast_current < slow_current:
            return -1  # 데드 크로스 (매도)
        else:
            return 0  # 관망

    def calculate_position_size(self, signal, portfolio, risk_params):
        """
        Kelly Criterion 기반 포지션 사이징
        """
        if signal == 0:
            return 0

        # 최근 성과 기반 Kelly Criterion
        win_rate = self.performance_metrics.get('win_rate', 0.5)
        avg_win = self.performance_metrics.get('avg_win', 0.01)
        avg_loss = abs(self.performance_metrics.get('avg_loss', -0.01))

        if avg_loss == 0:
            kelly_fraction = 0.1
        else:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

        # Kelly의 25% 사용 (보수적)
        kelly_fraction = max(0, min(kelly_fraction * 0.25, 0.2))

        # 포트폴리오 가치 대비 포지션 크기
        portfolio_value = portfolio['total_value']
        position_value = portfolio_value * kelly_fraction

        # 현재 가격
        current_price = portfolio['current_price']

        # 수량 계산
        quantity = position_value / current_price

        return quantity


class RSIMeanReversion(BaseStrategy):
    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        super().__init__(name=f"RSI_MeanRev_{rsi_period}")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signal(self, market_data, portfolio):
        """
        RSI 평균 회귀 전략
        RSI < 30: 과매도 → 매수
        RSI > 70: 과매수 → 매도
        """
        df = market_data

        # RSI 계산
        rsi = talib.RSI(df['close'], timeperiod=self.rsi_period)
        current_rsi = rsi.iloc[-1]

        if current_rsi < self.oversold:
            return 1  # 과매도 → 매수
        elif current_rsi > self.overbought:
            return -1  # 과매수 → 매도
        else:
            return 0

    def calculate_position_size(self, signal, portfolio, risk_params):
        # RSI 거리 기반 포지션 사이징
        df = portfolio['market_data']
        rsi = talib.RSI(df['close'], timeperiod=self.rsi_period).iloc[-1]

        if signal == 1:
            # 과매도일수록 큰 포지션
            distance = (self.oversold - rsi) / self.oversold
        elif signal == -1:
            # 과매수일수록 큰 포지션
            distance = (rsi - self.overbought) / (100 - self.overbought)
        else:
            return 0

        base_fraction = 0.1
        position_fraction = base_fraction * (1 + distance)
        position_fraction = min(position_fraction, 0.2)  # 최대 20%

        portfolio_value = portfolio['total_value']
        current_price = portfolio['current_price']

        quantity = (portfolio_value * position_fraction) / current_price
        return quantity


class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, bb_period=20, num_std=2):
        super().__init__(name=f"BB_Strategy_{bb_period}")
        self.bb_period = bb_period
        self.num_std = num_std

    def generate_signal(self, market_data, portfolio):
        """
        볼린저 밴드 전략
        가격이 하단 밴드 터치: 매수
        가격이 상단 밴드 터치: 매도
        """
        df = market_data

        upper, middle, lower = talib.BBANDS(
            df['close'],
            timeperiod=self.bb_period,
            nbdevup=self.num_std,
            nbdevdn=self.num_std
        )

        current_price = df['close'].iloc[-1]
        current_lower = lower.iloc[-1]
        current_upper = upper.iloc[-1]

        # 밴드 폭 확인 (변동성)
        bb_width = (current_upper - current_lower) / middle.iloc[-1]

        # 변동성이 너무 낮으면 거래하지 않음
        if bb_width < 0.02:
            return 0

        if current_price <= current_lower:
            return 1  # 하단 밴드 터치 → 매수
        elif current_price >= current_upper:
            return -1  # 상단 밴드 터치 → 매도
        else:
            return 0

    def calculate_position_size(self, signal, portfolio, risk_params):
        df = portfolio['market_data']

        upper, middle, lower = talib.BBANDS(
            df['close'],
            timeperiod=self.bb_period,
            nbdevup=self.num_std,
            nbdevdn=self.num_std
        )

        current_price = df['close'].iloc[-1]

        # 밴드로부터의 거리 기반 포지션 사이징
        if signal == 1:
            distance = (current_price - lower.iloc[-1]) / (middle.iloc[-1] - lower.iloc[-1])
        elif signal == -1:
            distance = (upper.iloc[-1] - current_price) / (upper.iloc[-1] - middle.iloc[-1])
        else:
            return 0

        position_fraction = 0.15 * (1 - distance)  # 밴드에 가까울수록 큰 포지션
        position_fraction = max(0.05, min(position_fraction, 0.2))

        portfolio_value = portfolio['total_value']
        current_price = portfolio['current_price']

        quantity = (portfolio_value * position_fraction) / current_price
        return quantity
```

### 3. ML 기반 전략

```python
class LSTMStrategy(BaseStrategy):
    def __init__(self, model, scaler, threshold=0.02):
        super().__init__(name="LSTM_Prediction")
        self.model = model
        self.scaler = scaler
        self.threshold = threshold  # 예측 변화율 임계값

    def generate_signal(self, market_data, portfolio):
        """
        LSTM 모델 예측 기반 신호 생성
        """
        # 특징 추출
        fe = FeatureEngineering()
        df = fe.calculate_technical_indicators(market_data.copy())

        feature_columns = [
            'close', 'volume', 'MA_7', 'MA_25', 'RSI_14', 'MACD',
            'MACD_signal', 'BB_width', 'ATR', 'ADX', 'volatility'
        ]

        # 정규화
        features = df[feature_columns].iloc[-60:].values
        features_normalized = self.scaler.transform(features)

        # 예측
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(features_normalized).unsqueeze(0)
            prediction = self.model(X_tensor).item()

        # 역정규화
        current_price = df['close'].iloc[-1]
        predicted_price = prediction  # 이미 역정규화된 값

        # 변화율 계산
        price_change = (predicted_price - current_price) / current_price

        # 신호 생성
        if price_change > self.threshold:
            return 1  # 상승 예측 → 매수
        elif price_change < -self.threshold:
            return -1  # 하락 예측 → 매도
        else:
            return 0  # 관망

    def calculate_position_size(self, signal, portfolio, risk_params):
        # 예측 신뢰도 기반 포지션 사이징
        # 실제 구현 시 모델의 불확실성 추정 활용 (예: 드롭아웃 기반)
        base_fraction = 0.15

        portfolio_value = portfolio['total_value']
        current_price = portfolio['current_price']

        quantity = (portfolio_value * base_fraction) / current_price
        return quantity
```

### 4. 강화학습 전략 (선택사항)

```python
import gym
from stable_baselines3 import PPO

class TradingEnvironment(gym.Env):
    """
    강화학습 거래 환경
    """
    def __init__(self, df, initial_balance=10000):
        super(TradingEnvironment, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0

        # 행동 공간: 0 (관망), 1 (매수), 2 (매도)
        self.action_space = gym.spaces.Discrete(3)

        # 상태 공간: [가격, 지표들, 포지션, 잔고]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 60  # 최소 lookback
        self.portfolio_value = self.initial_balance

        return self._get_observation()

    def _get_observation(self):
        """현재 상태 반환"""
        row = self.df.iloc[self.current_step]

        obs = [
            row['close'] / 10000,  # 정규화
            row['volume'] / 1000000,
            row['RSI_14'] / 100,
            row['MACD'],
            # ... 기타 지표들
            self.position,
            self.balance / self.initial_balance
        ]

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']

        # 행동 실행
        if action == 1:  # 매수
            if self.position == 0 and self.balance > 0:
                self.position = self.balance / current_price
                self.balance = 0

        elif action == 2:  # 매도
            if self.position > 0:
                self.balance = self.position * current_price
                self.position = 0

        # 다음 스텝
        self.current_step += 1

        # 포트폴리오 가치 계산
        next_price = self.df.iloc[self.current_step]['close']
        self.portfolio_value = self.balance + self.position * next_price

        # 보상 계산
        reward = (self.portfolio_value - self.initial_balance) / self.initial_balance

        # 종료 조건
        done = self.current_step >= len(self.df) - 1

        return self._get_observation(), reward, done, {}


class RLStrategy(BaseStrategy):
    def __init__(self, model_path):
        super().__init__(name="RL_PPO")
        self.agent = PPO.load(model_path)

    def generate_signal(self, market_data, portfolio):
        # 상태 구성
        state = self._construct_state(market_data, portfolio)

        # 행동 예측
        action, _ = self.agent.predict(state, deterministic=True)

        # 0: 관망, 1: 매수, 2: 매도
        if action == 1:
            return 1
        elif action == 2:
            return -1
        else:
            return 0

    def _construct_state(self, market_data, portfolio):
        # TradingEnvironment와 동일한 상태 구성
        row = market_data.iloc[-1]

        state = [
            row['close'] / 10000,
            row['volume'] / 1000000,
            row['RSI_14'] / 100,
            # ...
            portfolio.get('position', 0),
            portfolio.get('balance', 10000) / 10000
        ]

        return np.array(state, dtype=np.float32)

    def calculate_position_size(self, signal, portfolio, risk_params):
        # 고정 비율
        base_fraction = 0.2
        portfolio_value = portfolio['total_value']
        current_price = portfolio['current_price']

        quantity = (portfolio_value * base_fraction) / current_price
        return quantity
```

---

## 🔄 전략 스위칭 메커니즘

### 1. 시장 레짐 감지

```python
class MarketRegimeDetector:
    @staticmethod
    def detect_regime(market_data):
        """
        시장 상황 분류
        Returns: 'trending', 'range_bound', 'high_volatility', 'low_liquidity'
        """
        df = market_data

        # ADX (추세 강도)
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]

        # ATR (변동성)
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        avg_atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).mean()

        # Bollinger Band Width
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        bb_width = ((upper - lower) / middle).iloc[-1]

        # Volume
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].mean()

        # 추세장 감지
        if adx > 25 and df['close'].iloc[-1] > df['close'].rolling(50).mean().iloc[-1]:
            return 'trending_up'
        elif adx > 25 and df['close'].iloc[-1] < df['close'].rolling(50).mean().iloc[-1]:
            return 'trending_down'

        # 횡보장 감지
        elif adx < 20 and bb_width < 0.05:
            return 'range_bound'

        # 고변동성 감지
        elif atr > avg_atr * 1.5:
            return 'high_volatility'

        # 저유동성 감지
        elif current_volume < avg_volume * 0.5:
            return 'low_liquidity'

        else:
            return 'normal'

    @staticmethod
    def get_regime_features(market_data):
        """
        시장 레짐 특징 벡터 반환 (ML 모델용)
        """
        df = market_data

        features = {
            'adx': talib.ADX(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1],
            'atr_ratio': talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1] / df['close'].iloc[-1],
            'volume_ratio': df['volume'].iloc[-1] / df['volume'].mean(),
            'price_vs_ma50': (df['close'].iloc[-1] - df['close'].rolling(50).mean().iloc[-1]) / df['close'].iloc[-1],
            'volatility': df['close'].pct_change().std()
        }

        return features
```

### 2. 성과 기반 전략 선택

```python
class StrategySelector:
    def __init__(self, strategies):
        """
        strategies: 전략 객체들의 리스트
        """
        self.strategies = strategies
        self.strategy_scores = {s.name: 0.5 for s in strategies}
        self.current_strategy = strategies[0]
        self.evaluation_window = '7d'

    def select_best_strategy(self, market_data, portfolio):
        """
        최적 전략 선택
        """
        # 시장 레짐 감지
        regime = MarketRegimeDetector.detect_regime(market_data)

        # 각 전략 성과 평가
        for strategy in self.strategies:
            performance = strategy.evaluate_performance(self.evaluation_window)

            if performance:
                score = self._calculate_strategy_score(performance, regime, strategy)
                self.strategy_scores[strategy.name] = score

        # 최고 점수 전략 선택
        best_strategy_name = max(self.strategy_scores, key=self.strategy_scores.get)
        best_strategy = next(s for s in self.strategies if s.name == best_strategy_name)

        # 전략 전환 비용 고려
        if best_strategy != self.current_strategy:
            if self._should_switch(best_strategy, portfolio):
                print(f"Strategy switching: {self.current_strategy.name} → {best_strategy.name}")
                self.current_strategy = best_strategy

        return self.current_strategy

    def _calculate_strategy_score(self, performance, regime, strategy):
        """
        전략 점수 계산
        """
        # 기본 성과 메트릭
        sharpe_normalized = min(max(performance['sharpe_ratio'], -2), 2) / 4 + 0.5
        win_rate_normalized = performance['win_rate']
        profit_factor_normalized = min(performance['profit_factor'], 3) / 3
        drawdown_normalized = 1 - min(abs(performance['max_drawdown']), 0.5) / 0.5

        # 시장 레짐 매칭
        regime_match = self._get_regime_match_score(strategy, regime)

        # 종합 점수
        score = (
            0.3 * sharpe_normalized +
            0.2 * win_rate_normalized +
            0.2 * profit_factor_normalized +
            0.15 * drawdown_normalized +
            0.15 * regime_match
        )

        return score

    def _get_regime_match_score(self, strategy, regime):
        """
        전략과 시장 레짐 매칭 점수
        """
        # 전략별 최적 레짐 정의
        strategy_regime_map = {
            'MA_Cross': {'trending_up': 1.0, 'trending_down': 1.0, 'range_bound': 0.3},
            'RSI_MeanRev': {'range_bound': 1.0, 'trending_up': 0.4, 'trending_down': 0.4},
            'BB_Strategy': {'range_bound': 0.9, 'high_volatility': 0.7},
            'LSTM_Prediction': {'normal': 0.8, 'trending_up': 0.7, 'trending_down': 0.7}
        }

        # 전략 이름에서 베이스 추출
        strategy_base = strategy.name.split('_')[0] + '_' + strategy.name.split('_')[1]

        if strategy_base in strategy_regime_map:
            return strategy_regime_map[strategy_base].get(regime, 0.5)

        return 0.5

    def _should_switch(self, new_strategy, portfolio):
        """
        전략 전환 여부 결정
        """
        current_score = self.strategy_scores[self.current_strategy.name]
        new_score = self.strategy_scores[new_strategy.name]

        # 최소 개선 임계값
        min_improvement = 0.15

        # 전환 비용 (슬리피지, 수수료 등)
        switching_cost = 0.001  # 0.1%

        # 현재 포지션 존재 여부
        has_position = portfolio.get('position', 0) > 0

        if has_position:
            # 포지션이 있으면 더 큰 개선 필요
            min_improvement += switching_cost

        return new_score > current_score + min_improvement
```

### 3. 메타 학습 시스템 (강화학습 기반)

```python
import numpy as np
from collections import deque

class MetaStrategyAgent:
    """
    강화학습 기반 전략 가중치 학습
    """
    def __init__(self, num_strategies, state_dim=10):
        self.num_strategies = num_strategies
        self.state_dim = state_dim

        # Simple Q-learning (실제로는 DQN, PPO 등 사용 가능)
        self.q_table = {}
        self.epsilon = 0.1  # 탐험 비율
        self.alpha = 0.01   # 학습률
        self.gamma = 0.95   # 할인율

        self.memory = deque(maxlen=1000)

    def get_state_key(self, market_features, strategy_performances):
        """
        상태를 이산화하여 Q-table 키 생성
        """
        # 실제로는 더 정교한 상태 표현 필요
        regime = MarketRegimeDetector.detect_regime(market_features)
        best_perf_idx = np.argmax([p['sharpe_ratio'] for p in strategy_performances])

        return f"{regime}_{best_perf_idx}"

    def select_strategy_weights(self, market_features, strategy_performances):
        """
        전략 가중치 선택 (행동)
        """
        state_key = self.get_state_key(market_features, strategy_performances)

        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            # 탐험: 랜덤 가중치
            weights = np.random.dirichlet(np.ones(self.num_strategies))
        else:
            # 활용: Q-value가 높은 행동 선택
            if state_key in self.q_table:
                action_values = self.q_table[state_key]
                best_action_idx = np.argmax(action_values)
                weights = self._action_to_weights(best_action_idx)
            else:
                # 처음 보는 상태: 균등 가중치
                weights = np.ones(self.num_strategies) / self.num_strategies

        return weights

    def update(self, state, action, reward, next_state):
        """
        Q-learning 업데이트
        """
        state_key = self.get_state_key(state['market_features'], state['strategy_performances'])
        next_state_key = self.get_state_key(next_state['market_features'], next_state['strategy_performances'])

        # Q-table 초기화
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_strategies)

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.num_strategies)

        # Q-learning 업데이트
        action_idx = self._weights_to_action(action)

        current_q = self.q_table[state_key][action_idx]
        max_next_q = np.max(self.q_table[next_state_key])

        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q

        # 메모리 저장
        self.memory.append((state_key, action_idx, reward, next_state_key))

    def _action_to_weights(self, action_idx):
        """
        행동 인덱스를 전략 가중치로 변환
        """
        # 간단히 특정 전략에 높은 가중치 부여
        weights = np.ones(self.num_strategies) * 0.1
        weights[action_idx] = 0.6
        weights /= weights.sum()

        return weights

    def _weights_to_action(self, weights):
        """
        가중치를 행동 인덱스로 변환
        """
        return np.argmax(weights)
```

### 4. 통합 전략 관리자

```python
class StrategyManager:
    def __init__(self, strategies, use_meta_learning=True):
        self.strategies = strategies
        self.selector = StrategySelector(strategies)
        self.use_meta_learning = use_meta_learning

        if use_meta_learning:
            self.meta_agent = MetaStrategyAgent(num_strategies=len(strategies))

        self.current_weights = np.ones(len(strategies)) / len(strategies)
        self.performance_history = []

    def generate_signal(self, market_data, portfolio):
        """
        통합 신호 생성
        """
        if self.use_meta_learning:
            # 메타 학습: 여러 전략의 신호를 가중 결합
            signals = []

            for strategy in self.strategies:
                signal = strategy.generate_signal(market_data, portfolio)
                signals.append(signal)

            # 가중 평균 신호
            weighted_signal = np.dot(signals, self.current_weights)

            # 임계값 기반 최종 신호
            if weighted_signal > 0.5:
                final_signal = 1
            elif weighted_signal < -0.5:
                final_signal = -1
            else:
                final_signal = 0

        else:
            # 단일 최적 전략 선택
            best_strategy = self.selector.select_best_strategy(market_data, portfolio)
            final_signal = best_strategy.generate_signal(market_data, portfolio)

        return final_signal

    def execute_trade(self, signal, market_data, portfolio, order_executor):
        """
        거래 실행
        """
        if signal == 0:
            return None

        # 포지션 크기 계산
        if self.use_meta_learning:
            # 가중 평균 포지션 크기
            position_sizes = []
            for strategy in self.strategies:
                size = strategy.calculate_position_size(signal, portfolio, {})
                position_sizes.append(size)

            quantity = np.dot(position_sizes, self.current_weights)
        else:
            best_strategy = self.selector.current_strategy
            quantity = best_strategy.calculate_position_size(signal, portfolio, {})

        # 리스크 체크
        if not self._risk_check(signal, quantity, portfolio):
            print("Risk check failed. Trade cancelled.")
            return None

        # 주문 실행
        symbol = portfolio['symbol']
        side = 'BUY' if signal > 0 else 'SELL'

        try:
            order_result = order_executor.place_market_order(symbol, side, quantity)

            # 거래 기록
            trade = {
                'timestamp': time.time(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': order_result['fills'][0]['price'],
                'order_id': order_result['orderId']
            }

            # 모든 전략에 기록
            for strategy in self.strategies:
                strategy.record_trade(trade)

            return trade

        except Exception as e:
            print(f"Order execution failed: {e}")
            return None

    def _risk_check(self, signal, quantity, portfolio):
        """
        리스크 체크
        """
        current_price = portfolio['current_price']
        portfolio_value = portfolio['total_value']

        # 최대 포지션 크기 (포트폴리오의 20%)
        max_position_value = portfolio_value * 0.2
        position_value = quantity * current_price

        if position_value > max_position_value:
            return False

        # 일일 최대 손실 체크 (5%)
        daily_pnl = portfolio.get('daily_pnl', 0)
        if daily_pnl < -portfolio_value * 0.05:
            return False

        return True

    def update_meta_agent(self, market_data, portfolio_return):
        """
        메타 에이전트 업데이트
        """
        if not self.use_meta_learning:
            return

        # 상태 구성
        regime_features = MarketRegimeDetector.get_regime_features(market_data)
        strategy_performances = [s.evaluate_performance() for s in self.strategies]

        state = {
            'market_features': market_data,
            'strategy_performances': strategy_performances
        }

        # 보상 계산
        reward = portfolio_return

        # 다음 상태 (실제로는 다음 시점 데이터 사용)
        next_state = state  # 간단히 동일하게 설정

        # 업데이트
        self.meta_agent.update(state, self.current_weights, reward, next_state)

        # 새로운 가중치 선택
        self.current_weights = self.meta_agent.select_strategy_weights(
            market_data, strategy_performances
        )
```

---

## ✅ 구현 체크리스트

이제 상세한 구현 체크리스트를 별도 파일로 작성합니다.
