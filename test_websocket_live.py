"""WebSocket 실전 연결 테스트 (공개 데이터만)"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.binance_websocket import BinanceWebSocket

print("=" * 70)
print("WebSocket 실시간 데이터 테스트 (실전 서버)")
print("=" * 70)

message_count = [0]

def handle_message(data):
    """메시지 핸들러"""
    message_count[0] += 1

    # 스트림 데이터 추출
    if 'stream' in data and 'data' in data:
        stream_name = data['stream']
        k = data['data']['k']
        print(f"[{message_count[0]}] {k['s']} - 시간: {k['t']}, 종가: {k['c']}, 거래량: {k['v']}")
    elif 'e' in data and data['e'] == 'kline':
        # 단일 스트림 형식
        k = data['k']
        print(f"[{message_count[0]}] {k['s']} - 시간: {k['t']}, 종가: {k['c']}, 거래량: {k['v']}")

# 실전 서버의 1분 봉 스트림
streams = [
    BinanceWebSocket.create_kline_stream('btcusdt', '1m'),
]

print(f"\n스트림 구독: {streams}")
print("10초간 메시지 수신 테스트...\n")

ws = BinanceWebSocket(streams, handle_message, testnet=False)
ws.connect()

# 10초 대기
time.sleep(10)

ws.close()

print(f"\n✅ 테스트 완료! 총 {message_count[0]}개 메시지 수신")
print("=" * 70)
