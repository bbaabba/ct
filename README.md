# 🤖 바이낸스 AI 자동거래 시스템

AI 기반 암호화폐 자동거래 시스템 - LSTM 모델과 다중 전략 스위칭

## 📋 프로젝트 개요

바이낸스 거래소와 연동하여 AI 모델을 활용한 자동 거래를 수행하는 시스템입니다.

### 주요 기능

- ✅ **바이낸스 API 통합**: REST API 및 WebSocket 실시간 데이터 수집
- 🧠 **AI 모델**: LSTM 기반 가격 예측 및 거래 신호 생성
- 📊 **다중 전략**: 기술적 분석, ML 기반, 강화학습 전략
- 🔄 **전략 스위칭**: 시장 상황에 따른 동적 전략 선택
- 🛡️ **리스크 관리**: Kelly Criterion, Stop-Loss, 포지션 사이징
- 📈 **백테스팅**: 역사적 데이터 기반 전략 성과 검증
- 📱 **모니터링**: 실시간 성과 추적 및 알림

## 🚀 시작하기

### 1. 환경 설정

```bash
# 저장소 클론
cd workspace/cointrade

# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (Mac/Linux)
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env.example 파일을 .env로 복사
copy .env.example .env  # Windows
cp .env.example .env    # Mac/Linux
```

`.env` 파일을 열고 바이낸스 API 키를 설정하세요:

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here
BINANCE_TESTNET=True  # 처음에는 테스트넷 사용
```

### 3. 바이낸스 API 키 발급

#### 테스트넷 (권장 - 처음 시작 시)

1. [바이낸스 테스트넷](https://testnet.binance.vision/) 접속
2. API 키 생성
3. `.env` 파일에 키 입력
4. `BINANCE_TESTNET=True` 설정

#### 실전 거래소

1. 바이낸스 계정 로그인
2. API Management 페이지 접속
3. Create API 클릭
4. 권한 설정: Reading, Spot & Margin Trading
5. IP 화이트리스트 설정 (보안 강화)
6. `.env` 파일에 키 입력
7. `BINANCE_TESTNET=False` 설정

### 4. 연결 테스트

```python
from config.config import Config
from data.binance_client import BinanceClient

# 설정 확인
Config.print_config()

# API 연결 테스트
client = BinanceClient(testnet=True)
print("서버 연결:", client.ping())

# 계정 정보 조회
account = client.get_account_info()
print("계정 정보:", account)

# 현재 가격 조회
price = client.get_ticker_price('BTCUSDT')
print("BTC 가격:", price)
```

## 📁 프로젝트 구조

```
cointrade/
├── config/                   # 설정 관리
│   ├── __init__.py
│   └── config.py            # 전역 설정 클래스
├── data/                    # 데이터 수집 및 관리
│   ├── binance_client.py    # REST API 클라이언트
│   ├── binance_websocket.py # WebSocket 클라이언트
│   ├── collectors/          # 데이터 수집기
│   ├── processors/          # 데이터 전처리
│   └── storage/             # 데이터 저장소
├── models/                  # AI 모델
│   ├── rnn/                 # LSTM 모델
│   ├── training/            # 학습 파이프라인
│   └── inference/           # 추론
├── strategies/              # 거래 전략
│   ├── base.py              # 기본 인터페이스
│   ├── technical/           # 기술적 분석 전략
│   └── ml_based/            # ML 기반 전략
├── execution/               # 거래 실행
│   ├── order_manager.py     # 주문 관리
│   ├── portfolio.py         # 포트폴리오 관리
│   └── risk_manager.py      # 리스크 관리
├── strategy_selector/       # 전략 선택
│   ├── regime_detector.py   # 시장 분석
│   └── selector.py          # 전략 선택기
├── backtesting/             # 백테스팅
│   └── engine.py
├── monitoring/              # 모니터링
│   └── dashboard.py
├── utils/                   # 유틸리티
│   ├── rate_limiter.py      # Rate Limiter
│   └── logger.py            # 로깅
├── tests/                   # 테스트
├── notebooks/               # 분석 노트북
├── main.py                  # 메인 실행
├── requirements.txt         # 의존성
├── .env.example             # 환경 변수 예제
└── README.md                # 본 문서
```

## 🎯 사용 예제

### 예제 1: 실시간 가격 모니터링

```python
from data.binance_websocket import BinanceWebSocket

def handle_kline(data):
    if 'data' in data:
        k = data['data']['k']
        print(f"{k['s']} - 종가: {k['c']}, 거래량: {k['v']}")

# 1분 봉 스트림 구독
streams = [
    BinanceWebSocket.create_kline_stream('BTCUSDT', '1m'),
    BinanceWebSocket.create_kline_stream('ETHUSDT', '1m')
]

ws = BinanceWebSocket(streams, handle_kline, testnet=True)
ws.connect()

# 종료 시
# ws.close()
```

### 예제 2: 역사적 데이터 수집

```python
from data.binance_client import BinanceClient
import pandas as pd

client = BinanceClient(testnet=True)

# 1시간 봉 500개 가져오기
klines = client.get_klines('BTCUSDT', '1h', limit=500)

# DataFrame으로 변환
df = pd.DataFrame(klines, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
    'taker_buy_quote', 'ignore'
])

# 타입 변환
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

print(df.head())
```

### 예제 3: 주문 실행 (테스트넷)

```python
from data.binance_client import BinanceClient

client = BinanceClient(testnet=True)

# 계정 잔고 확인
balance = client.get_balance('USDT')
print(f"USDT 잔고: {balance['free']}")

# 시장가 매수 (소액 테스트)
order = client.place_market_order(
    symbol='BTCUSDT',
    side='BUY',
    quantity=0.001  # 0.001 BTC
)
print("주문 결과:", order)

# 미체결 주문 확인
open_orders = client.get_open_orders('BTCUSDT')
print("미체결 주문:", open_orders)
```

## 📖 상세 문서

더 자세한 내용은 다음 문서를 참고하세요:

- [BINANCE_TRADING_SYSTEM.md](BINANCE_TRADING_SYSTEM.md) - 시스템 아키텍처 및 기술 가이드
- [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) - 단계별 구현 체크리스트
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - 전체 구현 계획

## ⚙️ 주요 설정

### 리스크 관리 파라미터

`.env` 파일에서 다음 파라미터를 조정할 수 있습니다:

```env
MAX_POSITION_SIZE=0.2  # 포트폴리오의 20%
MAX_DAILY_LOSS=0.05    # 일일 최대 손실 5%
STOP_LOSS_PCT=0.02     # 개별 거래 손절 2%
INITIAL_CAPITAL=10000.0  # 초기 자본
```

### 거래 심볼

```env
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT
DEFAULT_TIMEFRAME=1h
```

## 🛡️ 보안 주의사항

1. **절대 API 키를 공유하지 마세요**
   - `.env` 파일은 `.gitignore`에 포함되어 있습니다
   - Git에 커밋하지 않도록 주의하세요

2. **IP 화이트리스트 설정**
   - 바이낸스 API 설정에서 신뢰할 수 있는 IP만 허용

3. **최소 권한 원칙**
   - API 키에 필요한 최소 권한만 부여
   - Withdrawal 권한은 절대 활성화하지 마세요

4. **테스트넷 먼저 사용**
   - 실전 거래 전 테스트넷에서 충분히 테스트

5. **소액으로 시작**
   - 실전 거래 시작 시 소액으로 시작 ($100-$500)
   - 시스템 안정성 확인 후 점진적으로 증액

## 📊 백테스팅

```python
# TODO: 백테스팅 예제 (Phase 1 완료 후 추가)
```

## 🧠 AI 모델

```python
# TODO: AI 모델 사용 예제 (Phase 2 완료 후 추가)
```

## 📈 모니터링

```python
# TODO: 모니터링 대시보드 (Phase 5 완료 후 추가)
```

## 🐛 문제 해결

### API 연결 실패

```python
# 서버 시간 동기화 확인
from data.binance_client import BinanceClient
import time

client = BinanceClient(testnet=True)
server_time = client.get_server_time()
local_time = int(time.time() * 1000)

print(f"서버 시간: {server_time}")
print(f"로컬 시간: {local_time}")
print(f"차이: {abs(server_time - local_time)}ms")

# 차이가 5000ms 이상이면 시스템 시간 동기화 필요
```

### Rate Limit 오류

- 요청 빈도를 줄이세요
- `RateLimiter` 클래스가 자동으로 관리합니다
- 429 오류 발생 시 자동 재시도

### WebSocket 연결 끊김

- 자동 재연결 로직이 구현되어 있습니다
- 최대 10회 재시도 (지수 백오프)

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

## ⚠️ 면책 조항

**이 시스템은 교육 및 연구 목적으로 제공됩니다. 암호화폐 거래는 높은 리스크를 수반하며, 투자 손실이 발생할 수 있습니다. 실전 거래 전 충분한 테스트와 리스크 관리가 필수입니다.**

- 본 소프트웨어 사용으로 인한 재정적 손실에 대해 개발자는 책임지지 않습니다
- 투자 결정은 본인의 책임하에 이루어져야 합니다
- 실전 거래 전 반드시 테스트넷에서 충분히 테스트하세요
- 감당할 수 있는 금액만 투자하세요

## 📞 문의

문제가 발생하거나 질문이 있으시면 Issue를 등록해주세요.

## 🗺️ 로드맵

- [x] Phase 1: 기초 인프라 구축 (진행 중)
- [ ] Phase 2: AI 모델 개발
- [ ] Phase 3: 거래 전략 시스템
- [ ] Phase 4: 전략 스위칭
- [ ] Phase 5: 실전 배포

---

**Happy Trading! 🚀**
