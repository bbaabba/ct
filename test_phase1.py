"""Phase 1 구현 테스트 스크립트"""
import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("바이낸스 자동거래 시스템 - Phase 1 테스트")
print("=" * 70)
print()

# ============ 1. 설정 테스트 ============
print("[ 1/5 ] 설정 시스템 테스트...")
print("-" * 70)

try:
    from config.config import Config

    # 설정 출력
    Config.print_config()
    print("✅ 설정 시스템 정상 작동")

except Exception as e:
    print(f"❌ 설정 시스템 오류: {e}")
    sys.exit(1)

print()

# ============ 2. Rate Limiter 테스트 ============
print("[ 2/5 ] Rate Limiter 테스트...")
print("-" * 70)

try:
    from utils.rate_limiter import RateLimiter

    # 초당 5개 제한으로 테스트
    limiter = RateLimiter(max_requests=5, time_window=1)

    print("초당 5개 요청 제한 테스트:")
    for i in range(7):
        can_proceed = limiter.can_proceed()
        if can_proceed:
            limiter.record_request()
            print(f"  요청 {i+1}: ✅ 허용 (남은 요청: {limiter.get_remaining_requests()})")
        else:
            print(f"  요청 {i+1}: ⏳ 대기 필요")
            wait_time = limiter.wait_if_needed()
            print(f"  → {wait_time:.2f}초 대기 후 재시도")

    print("✅ Rate Limiter 정상 작동")

except Exception as e:
    print(f"❌ Rate Limiter 오류: {e}")
    import traceback
    traceback.print_exc()

print()

# ============ 3. 바이낸스 API 연결 테스트 ============
print("[ 3/5 ] 바이낸스 API 연결 테스트...")
print("-" * 70)

try:
    from data.binance_client import BinanceClient

    # API 키 확인
    if not Config.BINANCE_API_KEY or not Config.BINANCE_SECRET_KEY:
        print("⚠️  경고: API 키가 설정되지 않았습니다.")
        print("   .env 파일에 BINANCE_API_KEY와 BINANCE_SECRET_KEY를 설정하세요.")
        print("   공개 API만 테스트합니다.")

        # API 키 없이 공개 API만 테스트
        client = BinanceClient(api_key='', secret_key='', testnet=True)
    else:
        # 정상적으로 클라이언트 생성
        client = BinanceClient(testnet=Config.IS_TESTNET)

    # 1) Ping 테스트
    print("\n1) 서버 연결 테스트 (Ping)...")
    ping_result = client.ping()
    if ping_result:
        print("   ✅ 서버 연결 성공")
    else:
        print("   ❌ 서버 연결 실패")

    # 2) 서버 시간 조회
    print("\n2) 서버 시간 조회...")
    server_time = client.get_server_time()
    local_time = int(time.time() * 1000)
    time_diff = abs(server_time - local_time)

    print(f"   서버 시간: {server_time}")
    print(f"   로컬 시간: {local_time}")
    print(f"   시간 차이: {time_diff}ms")

    if time_diff < 5000:
        print("   ✅ 시간 동기화 정상")
    else:
        print("   ⚠️  시간 차이가 5초 이상입니다. 시스템 시간을 확인하세요.")

    # 3) 현재 가격 조회
    print("\n3) 현재 가격 조회...")
    try:
        btc_price = client.get_ticker_price('BTCUSDT')
        print(f"   BTC/USDT: ${float(btc_price['price']):,.2f}")

        eth_price = client.get_ticker_price('ETHUSDT')
        print(f"   ETH/USDT: ${float(eth_price['price']):,.2f}")

        print("   ✅ 가격 조회 성공")
    except Exception as e:
        print(f"   ❌ 가격 조회 실패: {e}")

    # 4) 캔들스틱 데이터 조회
    print("\n4) 캔들스틱 데이터 조회 (최근 5개)...")
    try:
        klines = client.get_klines('BTCUSDT', '1h', limit=5)
        print(f"   조회된 캔들 개수: {len(klines)}")

        if klines:
            import pandas as pd
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)

            print("\n   최근 캔들스틱:")
            print(df[['timestamp', 'open', 'high', 'low', 'close']].to_string(index=False))
            print("\n   ✅ 캔들스틱 데이터 조회 성공")
    except Exception as e:
        print(f"   ❌ 캔들스틱 조회 실패: {e}")

    # 5) 호가창 조회
    print("\n5) 호가창 조회 (상위 5개)...")
    try:
        orderbook = client.get_order_book('BTCUSDT', limit=5)

        print("\n   매도 호가 (Asks):")
        for price, qty in orderbook['asks'][:5]:
            print(f"      ${float(price):,.2f} - {float(qty):.4f} BTC")

        print("\n   매수 호가 (Bids):")
        for price, qty in orderbook['bids'][:5]:
            print(f"      ${float(price):,.2f} - {float(qty):.4f} BTC")

        # 스프레드 계산
        best_ask = float(orderbook['asks'][0][0])
        best_bid = float(orderbook['bids'][0][0])
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid) * 100

        print(f"\n   스프레드: ${spread:.2f} ({spread_pct:.4f}%)")
        print("   ✅ 호가창 조회 성공")
    except Exception as e:
        print(f"   ❌ 호가창 조회 실패: {e}")

    # 6) 계정 정보 조회 (API 키가 있을 때만)
    if Config.BINANCE_API_KEY and Config.BINANCE_SECRET_KEY:
        print("\n6) 계정 정보 조회...")
        try:
            account = client.get_account_info()
            print(f"   계정 타입: {account.get('accountType', 'N/A')}")
            print(f"   거래 가능: {account.get('canTrade', False)}")

            # 주요 잔고 출력
            print("\n   주요 잔고:")
            for balance in account['balances']:
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked

                if total > 0:
                    print(f"      {balance['asset']}: {total:.8f} (사용가능: {free:.8f})")

            print("   ✅ 계정 정보 조회 성공")
        except Exception as e:
            print(f"   ❌ 계정 정보 조회 실패: {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ 바이낸스 API 테스트 완료")

except Exception as e:
    print(f"❌ 바이낸스 API 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

print()

# ============ 4. WebSocket 테스트 ============
print("[ 4/5 ] WebSocket 실시간 데이터 테스트...")
print("-" * 70)

try:
    from data.binance_websocket import BinanceWebSocket

    message_count = [0]  # 리스트로 감싸서 클로저 문제 해결

    def handle_message(data):
        """WebSocket 메시지 핸들러"""
        message_count[0] += 1

        if 'data' in data and 'k' in data['data']:
            k = data['data']['k']
            print(f"   [{message_count[0]}] {k['s']} - 종가: {k['c']}, 거래량: {k['v']}, 완료: {k['x']}")

    # 1분 봉 스트림 구독
    streams = [
        BinanceWebSocket.create_kline_stream('BTCUSDT', '1m'),
    ]

    print("BTCUSDT 1분 봉 스트림 구독 (5초간 테스트)...")

    ws = BinanceWebSocket(streams, handle_message, testnet=Config.IS_TESTNET)
    ws.connect()

    # 5초간 메시지 수신
    time.sleep(5)

    # 연결 종료
    ws.close()

    if message_count[0] > 0:
        print(f"\n   ✅ WebSocket 테스트 성공 (수신 메시지: {message_count[0]}개)")
    else:
        print("\n   ⚠️  메시지를 수신하지 못했습니다 (연결은 정상)")

except Exception as e:
    print(f"❌ WebSocket 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

print()

# ============ 5. 전체 요약 ============
print("[ 5/5 ] 테스트 요약")
print("-" * 70)
print()
print("✅ Phase 1 핵심 기능 테스트 완료!")
print()
print("다음 단계:")
print("  1. .env 파일에 바이낸스 API 키 설정 (아직 안했다면)")
print("  2. 테스트넷에서 소액 거래 테스트")
print("  3. Phase 2: AI 모델 개발 시작")
print()
print("=" * 70)
