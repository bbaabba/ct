# 레버리지 및 숏 포지션 구현 체크리스트

바이낸스 선물(Futures) 거래 시스템 구현을 위한 단계별 체크리스트입니다.

---

## 📋 구현 개요

| 항목 | 현재 상태 | 목표 |
|------|----------|------|
| 거래 방식 | 현물(Spot) Long Only | 선물(Futures) Long/Short |
| 레버리지 | 미지원 (1x) | 1x ~ 20x 지원 |
| 포지션 방향 | 매수(Long)만 가능 | 매수/매도(Short) 모두 가능 |
| 마진 타입 | 없음 | 격리/교차 마진 선택 |
| 청산 관리 | SL/TP만 | 강제청산가 계산 포함 |

---

## Phase 1: 선물 API 인프라 구축 (예상 소요: 3-4일)

### 1.1 바이낸스 선물 API 클라이언트 생성

- [ ] `api/futures_client.py` 파일 생성
- [ ] 선물 API 엔드포인트 연결
  - [ ] Testnet: `https://testnet.binancefuture.com`
  - [ ] Mainnet: `https://fapi.binance.com`
- [ ] WebSocket 엔드포인트 연결
  - [ ] Testnet: `wss://stream.binancefuture.com`
  - [ ] Mainnet: `wss://fstream.binance.com`
- [ ] 기본 메서드 구현
  - [ ] `get_account_info()` - 선물 계좌 정보 조회
  - [ ] `get_position_info()` - 현재 포지션 조회
  - [ ] `get_leverage_brackets()` - 레버리지 등급 조회
  - [ ] `set_leverage()` - 레버리지 설정
  - [ ] `set_margin_type()` - 마진 타입 설정 (ISOLATED/CROSSED)
  - [ ] `get_funding_rate()` - 펀딩비율 조회

### 1.2 선물 주문 메서드 구현

- [ ] `place_order()` - 기본 주문
  - [ ] LIMIT (지정가)
  - [ ] MARKET (시장가)
  - [ ] STOP_MARKET (스탑마켓)
  - [ ] TAKE_PROFIT_MARKET (익절마켓)
  - [ ] TRAILING_STOP_MARKET (트레일링스탑)
- [ ] `close_position()` - 포지션 청산
- [ ] `cancel_order()` - 주문 취소
- [ ] `cancel_all_orders()` - 전체 주문 취소
- [ ] `get_open_orders()` - 미체결 주문 조회
- [ ] `get_order_history()` - 주문 내역 조회

### 1.3 Config 업데이트

- [ ] `config/config.py` 수정
  ```python
  # 선물 거래 설정
  FUTURES_BASE_URL = "https://fapi.binance.com"
  FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"
  FUTURES_WSS_URL = "wss://fstream.binance.com/ws"

  # 레버리지 설정
  DEFAULT_LEVERAGE = 5
  MAX_LEVERAGE = 20

  # 마진 타입
  DEFAULT_MARGIN_TYPE = "ISOLATED"  # ISOLATED or CROSSED

  # 선물 전용 리스크 설정
  FUTURES_MAX_POSITION_SIZE = 0.3  # 30%
  LIQUIDATION_BUFFER = 0.05  # 청산가 대비 5% 버퍼
  ```

---

## Phase 2: 포지션 관리 시스템 개편 (예상 소요: 3-4일)

### 2.1 Position 데이터 구조 확장

- [ ] `execution/position.py` 신규 생성
  ```python
  @dataclass
  class FuturesPosition:
      symbol: str
      side: str  # 'LONG' or 'SHORT'
      entry_price: float
      quantity: float
      leverage: int
      margin_type: str  # 'ISOLATED' or 'CROSSED'
      unrealized_pnl: float
      liquidation_price: float
      margin: float  # 사용된 마진
      entry_time: datetime
      stop_loss: Optional[float]
      take_profit: Optional[float]
  ```

### 2.2 청산가(Liquidation Price) 계산기 구현

- [ ] `execution/liquidation_calculator.py` 생성
- [ ] 롱 포지션 청산가 계산
  ```
  청산가(Long) = 진입가 × (1 - 1/레버리지 + 유지마진율)
  ```
- [ ] 숏 포지션 청산가 계산
  ```
  청산가(Short) = 진입가 × (1 + 1/레버리지 - 유지마진율)
  ```
- [ ] 유지마진율(Maintenance Margin Rate) 테이블 구현
- [ ] 청산 위험도 알림 기능

### 2.3 리스크 관리자 확장

- [ ] `execution/risk_manager.py` 수정
- [ ] 레버리지 고려 포지션 크기 계산
  ```python
  def calculate_leveraged_position_size(
      self, price, leverage, confidence, margin_type
  ):
      # 레버리지 적용된 실제 포지션 크기
      # 청산가 버퍼 고려
      # 마진 요구량 계산
  ```
- [ ] 숏 포지션용 SL/TP 계산 (방향 반전)
  - [ ] Long: SL = entry × (1 - sl_pct), TP = entry × (1 + tp_pct)
  - [ ] Short: SL = entry × (1 + sl_pct), TP = entry × (1 - tp_pct)
- [ ] 강제청산 방지 로직
  - [ ] 청산가 접근 시 자동 포지션 축소
  - [ ] 마진 추가 알림

---

## Phase 3: 거래 봇 로직 수정 (예상 소요: 2-3일)

### 3.1 TradingBot 클래스 수정

- [ ] `trading_bot/bot.py` 수정
- [ ] 선물 클라이언트 통합
  ```python
  def __init__(self, ..., trading_mode='futures'):
      if trading_mode == 'futures':
          self.client = FuturesClient(testnet=testnet)
      else:
          self.client = BinanceClient(testnet=testnet)
  ```
- [ ] 숏 포지션 진입 로직 추가
  ```python
  if signal == Signal.SELL and self.current_position is None:
      # 숏 포지션 진입 (기존: 무시됨)
      self.open_short_position(signal, current_price)
  ```
- [ ] 레버리지 설정 메서드
  ```python
  def set_leverage(self, symbol, leverage):
      self.client.set_leverage(symbol, leverage)
  ```

### 3.2 포지션 진입/청산 로직 분리

- [ ] `open_long_position()` - 롱 포지션 진입
- [ ] `open_short_position()` - 숏 포지션 진입
- [ ] `close_long_position()` - 롱 포지션 청산
- [ ] `close_short_position()` - 숏 포지션 청산
- [ ] 포지션 방향별 손익 계산
  - [ ] Long PnL = (현재가 - 진입가) × 수량
  - [ ] Short PnL = (진입가 - 현재가) × 수량

### 3.3 양방향 SL/TP 처리

- [ ] `check_stop_loss_take_profit()` 수정
  ```python
  if position.side == 'LONG':
      if current_price <= position.stop_loss:  # SL 하향 돌파
          return 'STOP_LOSS'
      if current_price >= position.take_profit:  # TP 상향 돌파
          return 'TAKE_PROFIT'
  elif position.side == 'SHORT':
      if current_price >= position.stop_loss:  # SL 상향 돌파
          return 'STOP_LOSS'
      if current_price <= position.take_profit:  # TP 하향 돌파
          return 'TAKE_PROFIT'
  ```

---

## Phase 4: 전략 시스템 확장 (예상 소요: 2-3일)

### 4.1 Signal Enum 확장

- [ ] `strategies/base.py` 수정
  ```python
  class Signal(Enum):
      BUY = 1       # 롱 진입
      SELL = -1     # 숏 진입 (기존: 롱 청산용)
      HOLD = 0      # 관망
      CLOSE = 2     # 포지션 청산 (신규)

  class PositionSide(Enum):
      LONG = 'LONG'
      SHORT = 'SHORT'
      BOTH = 'BOTH'  # 헤지 모드용
  ```

### 4.2 전략별 숏 신호 생성 로직

- [ ] `strategies/technical/ma_crossover.py` 수정
  - [ ] 데드크로스(Dead Cross) = 숏 신호
  - [ ] 골든크로스(Golden Cross) = 롱 신호
- [ ] `strategies/technical/rsi_strategy.py` 수정
  - [ ] RSI > 70 (과매수) = 숏 신호
  - [ ] RSI < 30 (과매도) = 롱 신호
- [ ] `strategies/technical/bollinger_bands.py` 수정
  - [ ] 상단밴드 돌파 = 숏 신호
  - [ ] 하단밴드 돌파 = 롱 신호

### 4.3 앙상블 전략 수정

- [ ] `strategy_selector/strategy_selector.py` 수정
- [ ] 숏 신호 가중치 계산
- [ ] 롱/숏 신호 충돌 해결 로직
  ```python
  # 롱과 숏 신호가 동시에 나올 경우
  if long_score > short_score + threshold:
      final_signal = Signal.BUY
  elif short_score > long_score + threshold:
      final_signal = Signal.SELL
  else:
      final_signal = Signal.HOLD
  ```

---

## Phase 5: 펀딩비 관리 시스템 (예상 소요: 1-2일)

### 5.1 펀딩비 모니터링

- [ ] `execution/funding_manager.py` 생성
- [ ] 펀딩비 조회 메서드
  ```python
  def get_funding_rate(self, symbol) -> Dict:
      # 현재 펀딩비율
      # 다음 펀딩 시간
      # 예상 펀딩비용
  ```
- [ ] 펀딩 시간 (8시간마다: 00:00, 08:00, 16:00 UTC)

### 5.2 펀딩비 기반 의사결정

- [ ] 높은 펀딩비 경고 (> ±0.1%)
- [ ] 펀딩비 수익 전략 (역방향 포지션 고려)
- [ ] 펀딩 직전 포지션 조정 옵션

---

## Phase 6: 백테스팅 시스템 업데이트 (예상 소요: 2-3일)

### 6.1 백테스트 엔진 수정

- [ ] `backtesting/backtest_engine.py` 수정
- [ ] 숏 포지션 시뮬레이션
- [ ] 레버리지 적용 수익 계산
  ```python
  leveraged_pnl = base_pnl * leverage
  ```
- [ ] 펀딩비 시뮬레이션
- [ ] 강제청산 시뮬레이션

### 6.2 성과 지표 추가

- [ ] Long/Short 별 승률
- [ ] Long/Short 별 평균 수익률
- [ ] 레버리지 효율성 지표
- [ ] 최대 손실(MDD) - 레버리지 포함

---

## Phase 7: 안전장치 및 모니터링 (예상 소요: 2-3일)

### 7.1 안전장치 구현

- [ ] `execution/safety_guard.py` 생성
- [ ] 최대 레버리지 제한
  ```python
  MAX_ALLOWED_LEVERAGE = 10  # 시스템 최대 허용
  ```
- [ ] 청산가 접근 경고 (현재가가 청산가의 80% 도달 시)
- [ ] 일일 최대 손실 도달 시 거래 중단
- [ ] 연속 손실 시 레버리지 자동 감소
  ```python
  if consecutive_losses >= 3:
      leverage = max(1, current_leverage - 1)
  ```

### 7.2 실시간 모니터링

- [ ] 포지션 상태 대시보드 데이터
  ```python
  {
      'position': 'LONG',
      'entry_price': 50000,
      'current_price': 51000,
      'leverage': 5,
      'unrealized_pnl': 500,
      'unrealized_pnl_pct': 10.0,
      'liquidation_price': 42000,
      'distance_to_liquidation': 17.6%,
      'margin_ratio': 5.0%
  }
  ```
- [ ] 텔레그램/슬랙 알림 연동
  - [ ] 포지션 진입/청산 알림
  - [ ] 청산 위험 경고
  - [ ] 일일 손익 리포트

---

## Phase 8: 테스트 및 검증 (예상 소요: 3-4일)

### 8.1 단위 테스트

- [ ] `tests/test_futures_client.py`
  - [ ] API 연결 테스트
  - [ ] 주문 테스트 (테스트넷)
  - [ ] 레버리지 설정 테스트
- [ ] `tests/test_liquidation.py`
  - [ ] 청산가 계산 정확도
  - [ ] 다양한 레버리지 시나리오
- [ ] `tests/test_short_position.py`
  - [ ] 숏 포지션 손익 계산
  - [ ] 숏 SL/TP 동작

### 8.2 통합 테스트

- [ ] `tests/test_futures_integration.py`
  - [ ] 전체 거래 사이클 (롱)
  - [ ] 전체 거래 사이클 (숏)
  - [ ] 레버리지 변경 시나리오
  - [ ] 강제청산 시뮬레이션

### 8.3 백테스트 검증

- [ ] 동일 기간 현물 vs 선물 성과 비교
- [ ] 다양한 레버리지별 리스크/리턴 분석
- [ ] 숏 전략 포함 시 MDD 개선 확인

---

## Phase 9: 문서화 및 배포 준비 (예상 소요: 1-2일)

### 9.1 문서화

- [ ] `docs/FUTURES_TRADING_GUIDE.md` 작성
  - [ ] 선물 거래 기본 개념
  - [ ] 레버리지 사용 주의사항
  - [ ] 청산 메커니즘 설명
- [ ] `docs/API_REFERENCE.md` 업데이트
- [ ] `README.md` 업데이트

### 9.2 설정 템플릿

- [ ] `.env.example` 업데이트
  ```env
  # 선물 거래 설정
  TRADING_MODE=futures
  DEFAULT_LEVERAGE=5
  MARGIN_TYPE=ISOLATED
  MAX_LEVERAGE=10
  ```

### 9.3 마이그레이션 가이드

- [ ] 현물 → 선물 전환 절차
- [ ] 데이터베이스 스키마 변경사항
- [ ] 기존 백테스트 결과 호환성

---

## ⚠️ 주의사항 및 리스크

### 레버리지 거래 위험성

1. **강제청산 위험**: 레버리지가 높을수록 청산가가 진입가에 가까워짐
2. **변동성 증폭**: 가격 변동이 레버리지 배수만큼 증폭됨
3. **펀딩비 부담**: 포지션 유지 시 8시간마다 펀딩비 발생

### 권장 설정 (초보자)

```python
DEFAULT_LEVERAGE = 3      # 낮은 레버리지 시작
MAX_LEVERAGE = 5          # 최대 제한
MARGIN_TYPE = "ISOLATED"  # 격리마진 (다른 포지션 영향 없음)
STOP_LOSS_PCT = 0.02      # 2% 손절
LIQUIDATION_BUFFER = 0.10 # 청산가 대비 10% 버퍼
```

### 권장 설정 (경험자)

```python
DEFAULT_LEVERAGE = 5
MAX_LEVERAGE = 10
MARGIN_TYPE = "CROSSED"   # 교차마진 (유연한 마진 활용)
STOP_LOSS_PCT = 0.015     # 1.5% 손절
LIQUIDATION_BUFFER = 0.05 # 청산가 대비 5% 버퍼
```

---

## 📊 예상 일정

| Phase | 내용 | 예상 소요 | 누적 |
|-------|------|----------|------|
| Phase 1 | 선물 API 인프라 | 3-4일 | 3-4일 |
| Phase 2 | 포지션 관리 시스템 | 3-4일 | 6-8일 |
| Phase 3 | 거래 봇 로직 | 2-3일 | 8-11일 |
| Phase 4 | 전략 시스템 확장 | 2-3일 | 10-14일 |
| Phase 5 | 펀딩비 관리 | 1-2일 | 11-16일 |
| Phase 6 | 백테스팅 업데이트 | 2-3일 | 13-19일 |
| Phase 7 | 안전장치/모니터링 | 2-3일 | 15-22일 |
| Phase 8 | 테스트 및 검증 | 3-4일 | 18-26일 |
| Phase 9 | 문서화/배포 | 1-2일 | 19-28일 |

**총 예상 소요 기간: 3-4주**

---

## 📁 신규/수정 파일 목록

### 신규 생성 파일
```
api/futures_client.py
execution/position.py
execution/liquidation_calculator.py
execution/funding_manager.py
execution/safety_guard.py
tests/test_futures_client.py
tests/test_liquidation.py
tests/test_short_position.py
tests/test_futures_integration.py
docs/FUTURES_TRADING_GUIDE.md
```

### 수정 필요 파일
```
config/config.py
execution/risk_manager.py
trading_bot/bot.py
strategies/base.py
strategies/technical/ma_crossover.py
strategies/technical/rsi_strategy.py
strategies/technical/bollinger_bands.py
strategy_selector/strategy_selector.py
backtesting/backtest_engine.py
main.py
.env.example
README.md
```

---

## 체크리스트 사용법

1. 각 Phase를 순차적으로 진행
2. 체크박스([ ])를 완료 시 [x]로 변경
3. 각 Phase 완료 후 테스트 진행
4. Phase 8에서 전체 통합 테스트 수행

**작성일**: 2025-01-26
**최종 수정일**: 2025-01-26
**버전**: 1.0
