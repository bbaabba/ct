"""AI 베이스 모델 사전 학습 스크립트

역사적 데이터를 사용하여 AI 모델을 처음부터 학습합니다.
실시간 거래 전에 이 스크립트로 베이스 모델을 만들어야 합니다.

사용법:
    python scripts/train_base_model.py --symbol BTCUSDT --days 90 --epochs 20
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import requests
from tqdm import tqdm

from ai.market_analyzer import MarketAnalyzer
from utils.logger import setup_logger
from config.config import Config

logger = setup_logger(__name__)


class HistoricalDataCollector:
    """바이낸스에서 역사적 데이터 수집"""

    def __init__(self, symbol: str = 'BTCUSDT', is_testnet: bool = False):
        self.symbol = symbol
        if is_testnet:
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = "https://api.binance.com"

    def fetch_klines(
        self,
        interval: str = '1m',
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """캔들 데이터 가져오기"""
        endpoint = f"{self.base_url}/api/v3/klines"

        params = {
            'symbol': self.symbol,
            'interval': interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)

        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            return df

        except Exception as e:
            logger.error(f"데이터 가져오기 실패: {e}")
            return pd.DataFrame()

    def fetch_historical_data(
        self,
        interval: str = '1m',
        days: int = 90
    ) -> pd.DataFrame:
        """지정된 기간의 역사적 데이터 수집"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        all_data = []
        current_start = start_time

        logger.info(f"📥 {self.symbol} {interval} 데이터 수집 시작: {days}일치")

        # 1000개씩 나눠서 가져오기 (바이낸스 제한)
        with tqdm(total=days, desc="데이터 수집") as pbar:
            while current_start < end_time:
                df = self.fetch_klines(
                    interval=interval,
                    start_time=current_start,
                    end_time=end_time,
                    limit=1000
                )

                if df.empty:
                    break

                all_data.append(df)

                # 다음 시작 시간
                current_start = df['timestamp'].iloc[-1] + timedelta(minutes=1)

                # 진행률 업데이트
                elapsed_days = (current_start - start_time).days
                pbar.update(elapsed_days - pbar.n)

                # API 레이트 리미트 대비 (1초에 최대 10 요청)
                import time
                time.sleep(0.15)

        if not all_data:
            logger.error("데이터 수집 실패")
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

        logger.info(f"✅ 총 {len(result)}개 캔들 수집 완료")
        return result


def create_training_samples(
    df: pd.DataFrame,
    analyzer: MarketAnalyzer,
    future_bars: int = 5
) -> List[Dict]:
    """역사적 데이터에서 학습 샘플 생성

    Args:
        df: OHLCV 데이터프레임
        analyzer: AI 분석기
        future_bars: 미래 몇 봉 후의 결과를 라벨로 사용할지

    Returns:
        학습 샘플 리스트
    """
    samples = []
    min_rows = analyzer.sequence_length + 100 + future_bars

    if len(df) < min_rows:
        logger.warning(f"데이터 부족: {len(df)} < {min_rows}")
        return samples

    logger.info(f"🔄 학습 샘플 생성 시작 (미래 {future_bars}봉 기준)")

    # 각 타임스텝마다 샘플 생성 (미래 데이터 접근을 위해 future_bars만큼 제외)
    for i in tqdm(range(min_rows, len(df) - future_bars), desc="샘플 생성"):
        # 과거 데이터 (시퀀스 + 지표 계산용)
        historical_df = df.iloc[:i].copy()

        # 미래 가격 변화
        current_price = df.iloc[i - 1]['close']
        future_price = df.iloc[i - 1 + future_bars]['close']
        price_change_pct = (future_price / current_price - 1) * 100

        # 방향 라벨
        if price_change_pct > 0.05:  # 0.05% 이상 상승
            direction = 'UP'
        elif price_change_pct < -0.05:  # 0.05% 이상 하락
            direction = 'DOWN'
        else:
            direction = 'NEUTRAL'

        # 샘플 가중치 (큰 변화일수록 높은 가중치)
        abs_change = abs(price_change_pct)
        if abs_change > 1.0:  # 1% 이상 변화
            weight = 3.0
        elif abs_change > 0.5:  # 0.5% 이상 변화
            weight = 2.0
        elif abs_change > 0.2:  # 0.2% 이상 변화
            weight = 1.5
        else:
            weight = 1.0

        # 샘플 추가
        analyzer.add_training_sample(
            df=historical_df,
            actual_direction=direction,
            actual_price_change=price_change_pct,
            sample_weight=weight
        )

        samples.append({
            'timestamp': df.iloc[i - 1]['timestamp'],
            'direction': direction,
            'price_change': price_change_pct,
            'weight': weight
        })

    logger.info(f"✅ 총 {len(samples)}개 샘플 생성 완료")

    # 샘플 통계
    from collections import Counter
    direction_counts = Counter(s['direction'] for s in samples)
    logger.info(f"📊 샘플 분포: {dict(direction_counts)}")

    return samples


def train_base_model(
    symbol: str = 'BTCUSDT',
    days: int = 90,
    epochs: int = 20,
    batch_size: int = 64,
    is_testnet: bool = False
) -> str:
    """베이스 모델 학습

    Args:
        symbol: 거래 심볼
        days: 학습할 데이터 일수
        epochs: 학습 에폭
        batch_size: 배치 크기
        is_testnet: 테스트넷 사용 여부

    Returns:
        저장된 모델 경로
    """
    logger.info("=" * 70)
    logger.info(f"AI 베이스 모델 사전 학습 시작")
    logger.info(f"심볼: {symbol}, 기간: {days}일, 에폭: {epochs}")
    logger.info("=" * 70)

    # 1. 데이터 수집
    collector = HistoricalDataCollector(symbol=symbol, is_testnet=is_testnet)
    df = collector.fetch_historical_data(interval='1m', days=days)

    if df.empty:
        raise ValueError("데이터 수집 실패")

    # 2. AI 분석기 초기화 (새 모델)
    analyzer = MarketAnalyzer(
        sequence_length=60,
        hidden_size=128,
        num_layers=2,
        dropout=0.4,
        model_path=None  # 새 모델
    )
    analyzer.set_timeframe('1m')

    # 3. 학습 샘플 생성
    samples = create_training_samples(df, analyzer, future_bars=5)

    if len(samples) < 100:
        raise ValueError(f"샘플 부족: {len(samples)} < 100")

    # 4. 모델 학습
    logger.info(f"🚀 모델 학습 시작 (에폭: {epochs}, 배치: {batch_size})")

    result = analyzer.batch_train(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-4  # 새 모델이므로 높은 학습률
    )

    if result.get('status') != 'completed':
        raise ValueError(f"학습 실패: {result}")

    logger.info(f"✅ 학습 완료: loss={result['avg_loss']:.4f}, "
               f"accuracy={result['direction_accuracy']:.1%}")

    # 5. 모델 저장
    Config.create_directories()
    model_path = Config.MODEL_SAVE_DIR / f'transformer_{symbol}_1m_base.pt'
    analyzer.save_model(str(model_path))

    logger.info("=" * 70)
    logger.info(f"✅ 베이스 모델 저장 완료: {model_path}")
    logger.info("=" * 70)

    return str(model_path)


def main():
    parser = argparse.ArgumentParser(description='AI 베이스 모델 사전 학습')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='거래 심볼 (기본: BTCUSDT)')
    parser.add_argument('--days', type=int, default=90,
                       help='학습 데이터 일수 (기본: 90일)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='학습 에폭 (기본: 20)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='배치 크기 (기본: 64)')
    parser.add_argument('--testnet', action='store_true',
                       help='테스트넷 사용')

    args = parser.parse_args()

    try:
        model_path = train_base_model(
            symbol=args.symbol,
            days=args.days,
            epochs=args.epochs,
            batch_size=args.batch_size,
            is_testnet=args.testnet
        )

        print("\n" + "=" * 70)
        print("✅ 베이스 모델 학습 완료!")
        print(f"📁 모델 경로: {model_path}")
        print("\n다음 단계:")
        print("1. 이 모델을 futures_bot.py에서 로드하여 사용")
        print("2. 실시간 거래 데이터로 파인튜닝 자동 진행")
        print("=" * 70)

    except Exception as e:
        logger.error(f"❌ 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
