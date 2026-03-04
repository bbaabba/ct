# 웹 대시보드 구현 체크리스트

> **목표**: 실시간 차트, 거래 진입점, 손익 정보를 표시하는 웹 대시보드 구현

## 📋 전체 개요

### 주요 기능
- 📊 실시간 가격 차트 (TradingView 스타일)
- 🎯 진입/청산 포인트 표시
- 💰 실시간 손익 계산 및 표시
- 📈 성과 메트릭 대시보드
- 🔔 알림 시스템
- 📱 반응형 디자인 (모바일 지원)

---

## Phase 1: 백엔드 API 개발

### 1.1 FastAPI 프로젝트 설정
- [ ] FastAPI 프로젝트 구조 생성
  ```
  web/
  ├── api/
  │   ├── __init__.py
  │   ├── main.py          # FastAPI 앱
  │   ├── routes/
  │   │   ├── trading.py   # 거래 관련 API
  │   │   ├── chart.py     # 차트 데이터 API
  │   │   ├── strategy.py  # 전략 관련 API
  │   │   └── websocket.py # 실시간 업데이트
  │   ├── models/
  │   │   ├── response.py  # API 응답 모델
  │   │   └── schemas.py   # Pydantic 스키마
  │   └── services/
  │       ├── trading_service.py
  │       └── chart_service.py
  ├── static/
  │   ├── css/
  │   ├── js/
  │   └── img/
  └── templates/
      └── index.html
  ```

- [ ] 필요한 패키지 설치
  ```bash
  pip install fastapi uvicorn websockets python-socketio aiofiles
  pip install plotly dash pandas-ta
  ```

### 1.2 데이터 모델 정의
- [ ] **거래 정보 모델** (`TradeInfo`)
  ```python
  class TradeInfo(BaseModel):
      id: str
      symbol: str
      side: str  # 'BUY' or 'SELL'
      entry_price: float
      exit_price: Optional[float]
      quantity: float
      entry_time: datetime
      exit_time: Optional[datetime]
      pnl: Optional[float]
      pnl_percentage: Optional[float]
      stop_loss: float
      take_profit: float
      strategy: str
      status: str  # 'OPEN', 'CLOSED', 'PENDING'
  ```

- [ ] **차트 데이터 모델** (`ChartData`)
  ```python
  class ChartData(BaseModel):
      timestamp: List[datetime]
      open: List[float]
      high: List[float]
      low: List[float]
      close: List[float]
      volume: List[float]
  ```

- [ ] **포지션 정보 모델** (`PositionInfo`)
  ```python
  class PositionInfo(BaseModel):
      symbol: str
      side: str
      entry_price: float
      current_price: float
      quantity: float
      unrealized_pnl: float
      unrealized_pnl_pct: float
      duration: timedelta
  ```

### 1.3 REST API 엔드포인트
- [ ] **GET /api/trades** - 전체 거래 내역
  - Query params: `limit`, `offset`, `status`, `symbol`
  - Response: `List[TradeInfo]`

- [ ] **GET /api/trades/{trade_id}** - 특정 거래 상세
  - Response: `TradeInfo`

- [ ] **GET /api/positions** - 현재 오픈 포지션
  - Response: `List[PositionInfo]`

- [ ] **GET /api/chart/{symbol}** - 차트 데이터
  - Query params: `interval`, `start_date`, `end_date`
  - Response: `ChartData`

- [ ] **GET /api/performance** - 성과 메트릭
  - Response: `PerformanceMetrics`
  ```python
  class PerformanceMetrics(BaseModel):
      total_trades: int
      win_rate: float
      total_pnl: float
      total_pnl_pct: float
      sharpe_ratio: float
      max_drawdown: float
      avg_win: float
      avg_loss: float
      profit_factor: float
      daily_pnl: List[dict]  # 일별 손익
  ```

- [ ] **GET /api/strategies** - 전략 목록 및 상태
  - Response: `List[StrategyInfo]`

- [ ] **GET /api/system/status** - 시스템 상태
  - Response: `SystemStatus`
  ```python
  class SystemStatus(BaseModel):
      is_running: bool
      uptime: timedelta
      last_cycle_time: datetime
      api_health: bool
      database_health: bool
      active_strategies: List[str]
  ```

### 1.4 WebSocket 실시간 업데이트
- [ ] **WS /ws/prices** - 실시간 가격 스트림
  ```python
  {
    "type": "price_update",
    "symbol": "BTCUSDT",
    "price": 92000.50,
    "timestamp": "2026-01-13T10:00:00Z"
  }
  ```

- [ ] **WS /ws/trades** - 실시간 거래 업데이트
  ```python
  {
    "type": "trade_opened" | "trade_closed",
    "trade": TradeInfo
  }
  ```

- [ ] **WS /ws/signals** - 실시간 거래 신호
  ```python
  {
    "type": "signal",
    "symbol": "BTCUSDT",
    "signal": "BUY" | "SELL" | "HOLD",
    "confidence": 0.85,
    "strategy": "MA_Crossover",
    "timestamp": "2026-01-13T10:00:00Z"
  }
  ```

---

## Phase 2: 프론트엔드 개발

### 2.1 기술 스택 선택
- [ ] **옵션 1: React + TypeScript** (추천)
  - 라이브러리: TradingView Lightweight Charts, Recharts
  - 상태 관리: Redux Toolkit 또는 Zustand
  - UI 프레임워크: Material-UI 또는 Ant Design

- [ ] **옵션 2: Vue.js 3 + TypeScript**
  - 라이브러리: Apache ECharts, TradingView
  - 상태 관리: Pinia
  - UI 프레임워크: Vuetify

- [ ] **옵션 3: Next.js** (SSR 지원)
  - React 기반, 서버사이드 렌더링
  - 최적의 성능과 SEO

### 2.2 차트 컴포넌트
- [ ] **실시간 캔들스틱 차트**
  - TradingView Lightweight Charts 통합
  - 다중 타임프레임 지원 (1m, 5m, 15m, 1h, 4h, 1d)
  - 기술적 지표 오버레이
    - [ ] 이동평균선 (MA, EMA)
    - [ ] RSI
    - [ ] Bollinger Bands
    - [ ] MACD
    - [ ] 볼륨

- [ ] **진입/청산 포인트 마커**
  - 매수 진입: 녹색 화살표 (↑)
  - 매도 진입: 빨간색 화살표 (↓)
  - 청산: X 마크
  - 호버 시 상세 정보 표시
    ```
    진입: $92,000.00
    시간: 2026-01-13 10:30:00
    전략: MA_Crossover
    신뢰도: 85%
    ```

- [ ] **Stop-Loss / Take-Profit 라인**
  - 점선으로 표시
  - 드래그 가능 (수정 기능)
  - 현재 가격과의 거리 표시

### 2.3 거래 정보 패널
- [ ] **현재 포지션 카드**
  ```
  ┌─────────────────────────────────┐
  │ BTC/USDT - LONG                 │
  │ 진입: $92,000.00                │
  │ 현재: $92,500.00 (+0.54%)       │
  │ 수량: 0.1 BTC                   │
  │ 미실현 손익: +$50.00 (+0.54%)   │
  │ 보유 시간: 2시간 15분            │
  │ SL: $90,000 | TP: $95,000       │
  └─────────────────────────────────┘
  ```

- [ ] **거래 내역 테이블**
  | 시간 | 심볼 | 방향 | 진입가 | 청산가 | 수량 | 손익 | 전략 |
  |------|------|------|--------|--------|------|------|------|
  | ... | ... | ... | ... | ... | ... | ... | ... |

  - 정렬 가능
  - 필터링 (심볼, 전략, 손익)
  - 페이지네이션
  - CSV 내보내기 버튼

### 2.4 성과 대시보드
- [ ] **주요 메트릭 카드**
  ```
  [총 손익]      [승률]        [총 거래]
   $1,250        68.5%          127
  (+12.5%)       ■■■□□         ↑ 5

  [Sharpe]      [Max DD]      [Profit Factor]
   1.85          -8.2%          2.3
  ```

- [ ] **손익 그래프**
  - 누적 손익 라인 차트
  - 일별 손익 바 차트
  - 시간대별 성과 히트맵

- [ ] **전략별 성과 비교**
  - 파이 차트: 전략별 손익 기여도
  - 바 차트: 전략별 승률 비교
  - 테이블: 전략별 상세 메트릭

### 2.5 알림 시스템
- [ ] **브라우저 알림**
  - 거래 진입/청산 시
  - Stop-Loss / Take-Profit 도달 시
  - 시스템 에러 발생 시

- [ ] **알림 센터 UI**
  - 알림 목록 (최근 50개)
  - 읽음/안읽음 상태
  - 알림 필터링

---

## Phase 3: 실시간 통합

### 3.1 WebSocket 클라이언트 구현
- [ ] **가격 업데이트 구독**
  ```typescript
  const ws = new WebSocket('ws://localhost:8000/ws/prices');
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateChart(data);
  };
  ```

- [ ] **자동 재연결 로직**
  - 연결 끊김 시 지수 백오프로 재시도
  - 최대 5회 재시도
  - 연결 상태 UI 표시

- [ ] **데이터 버퍼링**
  - 네트워크 지연 시 데이터 큐잉
  - 순서 보장

### 3.2 상태 관리
- [ ] **전역 상태 구조**
  ```typescript
  interface AppState {
    trades: Trade[];
    positions: Position[];
    chartData: ChartData;
    performance: PerformanceMetrics;
    systemStatus: SystemStatus;
    ui: {
      selectedSymbol: string;
      selectedTimeframe: string;
      isLoading: boolean;
    };
  }
  ```

- [ ] **액션 정의**
  - `fetchTrades`, `fetchPositions`
  - `updatePrice`, `updatePosition`
  - `addTrade`, `closeTrade`

### 3.3 데이터 캐싱
- [ ] **로컬 스토리지**
  - 사용자 설정 (선호 심볼, 타임프레임)
  - 차트 레이아웃

- [ ] **인메모리 캐싱**
  - 최근 차트 데이터
  - 거래 내역 (최근 100개)

---

## Phase 4: 고급 기능

### 4.1 사용자 인터페이스 개선
- [ ] **다크/라이트 테마**
  - 테마 전환 버튼
  - 시스템 설정 따르기 옵션

- [ ] **레이아웃 커스터마이제이션**
  - 드래그 앤 드롭으로 위젯 배치
  - 레이아웃 저장/불러오기
  - 사전 정의된 레이아웃 템플릿

- [ ] **반응형 디자인**
  - 데스크톱: 멀티 컬럼 레이아웃
  - 태블릿: 2컬럼 레이아웃
  - 모바일: 싱글 컬럼, 스와이프 네비게이션

### 4.2 분석 도구
- [ ] **백테스트 결과 시각화**
  - 차트에 백테스트 거래 오버레이
  - 시뮬레이션 모드

- [ ] **성과 분석 리포트**
  - PDF 내보내기
  - 주간/월간 리포트 자동 생성

- [ ] **비교 모드**
  - 여러 전략 동시 비교
  - 다중 심볼 비교

### 4.3 제어 기능
- [ ] **수동 거래 인터페이스**
  - 시장가/지정가 주문
  - Stop-Loss / Take-Profit 설정
  - 포지션 크기 계산기

- [ ] **봇 제어**
  - 시작/정지 버튼
  - 긴급 정지 (모든 포지션 청산)
  - 전략 활성화/비활성화

- [ ] **파라미터 실시간 조정**
  - 리스크 설정 변경
  - 전략 파라미터 조정
  - 실시간 적용 또는 다음 사이클부터 적용

---

## Phase 5: 배포 및 보안

### 5.1 배포 설정
- [ ] **Docker 컨테이너화**
  ```dockerfile
  # Dockerfile.web
  FROM node:18 as frontend
  WORKDIR /app
  COPY frontend/package*.json ./
  RUN npm install
  COPY frontend/ ./
  RUN npm run build

  FROM python:3.11
  WORKDIR /app
  COPY --from=frontend /app/dist ./static
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  CMD ["uvicorn", "web.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

- [ ] **Docker Compose**
  ```yaml
  version: '3.8'
  services:
    web:
      build: .
      ports:
        - "8000:8000"
      environment:
        - DATABASE_URL=postgresql://...
        - REDIS_URL=redis://redis:6379
      depends_on:
        - postgres
        - redis

    postgres:
      image: postgres:15
      volumes:
        - postgres_data:/var/lib/postgresql/data

    redis:
      image: redis:7
      volumes:
        - redis_data:/data

  volumes:
    postgres_data:
    redis_data:
  ```

### 5.2 보안
- [ ] **인증 시스템**
  - JWT 토큰 기반 인증
  - API 키 관리
  - 역할 기반 접근 제어 (RBAC)

- [ ] **HTTPS 설정**
  - Let's Encrypt SSL 인증서
  - NGINX 리버스 프록시

- [ ] **Rate Limiting**
  - API 요청 제한
  - WebSocket 연결 제한

- [ ] **입력 검증**
  - Pydantic 스키마 검증
  - SQL Injection 방지
  - XSS 방지

### 5.3 모니터링
- [ ] **로깅**
  - 구조화된 로깅 (JSON)
  - 로그 집계 (ELK Stack 또는 Grafana Loki)

- [ ] **메트릭**
  - Prometheus 메트릭 수집
  - Grafana 대시보드

- [ ] **알림**
  - 시스템 다운 알림
  - 성과 이상 감지 알림

---

## Phase 6: 테스트 및 최적화

### 6.1 테스트
- [ ] **단위 테스트**
  - API 엔드포인트 테스트
  - 컴포넌트 테스트 (Jest)

- [ ] **통합 테스트**
  - API + DB 통합 테스트
  - WebSocket 통신 테스트

- [ ] **E2E 테스트**
  - Playwright 또는 Cypress
  - 주요 사용자 플로우 테스트

### 6.2 성능 최적화
- [ ] **프론트엔드**
  - 코드 스플리팅
  - 레이지 로딩
  - 이미지 최적화
  - 캐싱 전략

- [ ] **백엔드**
  - 데이터베이스 쿼리 최적화
  - 인덱싱
  - 캐싱 (Redis)
  - 연결 풀링

- [ ] **WebSocket**
  - 메시지 압축
  - 데이터 샘플링 (초당 업데이트 제한)

---

## 📚 참고 리소스

### 기술 문서
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [TradingView Lightweight Charts](https://www.tradingview.com/lightweight-charts/)
- [WebSocket API 가이드](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)

### 오픈소스 참고 프로젝트
- [FreqTrade UI](https://github.com/freqtrade/frequi)
- [Jesse Trading Bot](https://github.com/jesse-ai/jesse)
- [Binance Trading Bot Dashboard](https://github.com/chrisleekr/binance-trading-bot)

### 디자인 영감
- TradingView
- Binance 거래소
- CoinMarketCap
- CryptoQuant

---

## 🎯 우선순위 로드맵

### Sprint 1 (1-2주)
1. ✅ FastAPI 기본 구조
2. ✅ REST API 엔드포인트 (거래, 포지션)
3. ✅ 간단한 프론트엔드 (차트 + 거래 목록)

### Sprint 2 (2-3주)
1. ✅ WebSocket 실시간 업데이트
2. ✅ 차트 진입/청산 포인트 표시
3. ✅ 성과 대시보드

### Sprint 3 (3-4주)
1. ✅ 고급 차트 기능 (지표, 마커)
2. ✅ 알림 시스템
3. ✅ 반응형 디자인

### Sprint 4 (4-5주)
1. ✅ 제어 기능 (봇 시작/정지)
2. ✅ 배포 설정
3. ✅ 보안 강화

---

## 📝 구현 예시 코드

### FastAPI 엔드포인트 예시
```python
# web/api/routes/trading.py
from fastapi import APIRouter, Depends
from typing import List
from ..models.schemas import TradeInfo, PositionInfo

router = APIRouter(prefix="/api", tags=["trading"])

@router.get("/trades", response_model=List[TradeInfo])
async def get_trades(
    limit: int = 50,
    offset: int = 0,
    symbol: Optional[str] = None
):
    # 거래 내역 조회 로직
    trades = trading_service.get_trades(limit, offset, symbol)
    return trades

@router.get("/positions", response_model=List[PositionInfo])
async def get_positions():
    # 현재 포지션 조회
    positions = trading_service.get_open_positions()
    return positions
```

### WebSocket 예시
```python
# web/api/routes/websocket.py
from fastapi import WebSocket
import asyncio

@router.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 실시간 가격 데이터 전송
            price_data = await get_latest_price()
            await websocket.send_json(price_data)
            await asyncio.sleep(1)  # 1초마다 업데이트
    except WebSocketDisconnect:
        logger.info("WebSocket 연결 종료")
```

### React 차트 컴포넌트 예시
```typescript
// frontend/src/components/TradingChart.tsx
import { createChart } from 'lightweight-charts';
import { useEffect, useRef } from 'react';

export function TradingChart({ data, trades }) {
  const chartContainerRef = useRef();

  useEffect(() => {
    const chart = createChart(chartContainerRef.current, {
      width: 800,
      height: 400,
    });

    const candlestickSeries = chart.addCandlestickSeries();
    candlestickSeries.setData(data);

    // 진입/청산 마커 추가
    const markers = trades.map(trade => ({
      time: trade.entry_time,
      position: trade.side === 'BUY' ? 'belowBar' : 'aboveBar',
      color: trade.side === 'BUY' ? '#26a69a' : '#ef5350',
      shape: 'arrowUp',
      text: `${trade.side} @ $${trade.entry_price}`,
    }));

    candlestickSeries.setMarkers(markers);

    return () => chart.remove();
  }, [data, trades]);

  return <div ref={chartContainerRef} />;
}
```

---

## ✅ 완료 체크

각 항목 완료 시 `- [ ]`를 `- [x]`로 변경하여 진행 상황을 추적하세요.

**추정 개발 기간**: 4-6주 (1명 기준)
**추정 개발 기간**: 2-3주 (2-3명 팀)

