"""역사적 데이터 수집기"""
import time
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
from pathlib import Path

from data.binance_client import BinanceClient
from config.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class HistoricalDataCollector:
    """바이낸스 역사적 데이터 수집 및 저장"""

    def __init__(self, client=None, cache_ttl: int = 300):
        """
        Args:
            client: get_klines()를 가진 클라이언트 (BinanceClient 또는 BinanceFuturesClient).
                    None이면 BinanceClient(spot) 생성 — 선물 봇은 반드시 futures client를 전달할 것.
            cache_ttl: 캐시 유효 시간 (초), 기본 5분
        """
        self.client = client or BinanceClient(testnet=Config.IS_TESTNET)
        self.data_dir = Config.BASE_DIR / 'data' / 'raw'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 메모리 캐시 (key: "symbol_interval_days", value: (data, timestamp))
        self._cache = {}
        self.cache_ttl = cache_ttl

        logger.info(f"역사적 데이터 수집기 초기화 완료 (캐시 TTL: {cache_ttl}초)")

    def collect_klines(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        save_to_csv: bool = True
    ) -> pd.DataFrame:
        """
        캔들스틱 데이터 수집

        Args:
            symbol: 심볼 (예: BTCUSDT)
            interval: 간격 (1m, 5m, 15m, 1h, 4h, 1d 등)
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (None이면 현재)
            save_to_csv: CSV 파일로 저장 여부

        Returns:
            DataFrame (timestamp, open, high, low, close, volume)
        """
        logger.info(f"{symbol} {interval} 데이터 수집 시작: {start_date} ~ {end_date or '현재'}")

        # ── 캐시된 CSV가 1주일 이내면 바로 로드 (API 호출 건너뛰기) ──
        if save_to_csv and end_date:
            filename = f"{symbol}_{interval}_{start_date}_{end_date}.csv"
            filepath = self.data_dir / filename
            if filepath.exists():
                file_age = time.time() - filepath.stat().st_mtime
                if file_age < 7 * 86400:  # 7일 = 604800초
                    try:
                        df = self.load_csv(str(filepath))
                        if len(df) > 0:
                            days_ago = file_age / 86400
                            logger.info(f"📦 캐시 CSV 로드: {filepath} ({len(df)}개 캔들, {days_ago:.1f}일 전 수집)")
                            return df
                    except Exception as e:
                        logger.warning(f"캐시 CSV 로드 실패, 재수집: {e}")
                else:
                    logger.info(f"⏰ 캐시 만료 ({file_age/86400:.1f}일 경과), 재수집: {filepath.name}")

        # 날짜를 밀리초 타임스탬프로 변환
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)

        if end_date:
            # end_date 당일 23:59:59까지 포함 (자정 = 당일 제외 방지)
            end_ts = int((datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)

        all_klines = []
        current_ts = start_ts

        # 바이낸스는 한 번에 최대 1000개까지만 가져올 수 있음
        limit = 1000

        while current_ts < end_ts:
            try:
                # 데이터 요청
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    start_time=current_ts,
                    end_time=end_ts
                )

                if not klines:
                    break

                all_klines.extend(klines)

                # 다음 요청의 시작 시간을 마지막 캔들 이후로 설정
                current_ts = klines[-1][6] + 1  # close_time + 1ms

                logger.info(f"  수집됨: {len(klines)}개 (총: {len(all_klines)}개)")

                # Rate Limit 고려하여 잠시 대기
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"데이터 수집 오류: {e}")
                break

        # DataFrame 변환
        df = self._klines_to_dataframe(all_klines)

        logger.info(f"✅ 총 {len(df)}개 캔들 수집 완료")

        # CSV 저장
        if save_to_csv and not df.empty:
            filename = f"{symbol}_{interval}_{start_date}_{end_date or 'now'}.csv"
            filepath = self.data_dir / filename
            df.to_csv(filepath, index=False)
            logger.info(f"💾 파일 저장: {filepath}")

        return df

    def _klines_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """캔들스틱 데이터를 DataFrame으로 변환"""
        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # 타입 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                          'quote_volume', 'taker_buy_base', 'taker_buy_quote']
        df[numeric_columns] = df[numeric_columns].astype(float)

        df['trades'] = df['trades'].astype(int)

        # 불필요한 컬럼 제거
        df = df.drop(['ignore'], axis=1)

        return df

    def collect_multiple_symbols(
        self,
        symbols: List[str],
        interval: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> dict:
        """
        여러 심볼의 데이터를 한 번에 수집

        Args:
            symbols: 심볼 리스트
            interval: 간격
            start_date: 시작 날짜
            end_date: 종료 날짜

        Returns:
            {symbol: DataFrame} 딕셔너리
        """
        logger.info(f"{len(symbols)}개 심볼 데이터 수집 시작")

        results = {}

        for symbol in symbols:
            try:
                df = self.collect_klines(symbol, interval, start_date, end_date)
                results[symbol] = df
                logger.info(f"✅ {symbol}: {len(df)}개")

                # 심볼 간 대기 (Rate Limit)
                time.sleep(1)

            except Exception as e:
                logger.error(f"❌ {symbol} 수집 실패: {e}")
                results[symbol] = pd.DataFrame()

        logger.info(f"✅ 전체 수집 완료: {len([s for s, df in results.items() if not df.empty])}/{len(symbols)}")

        return results

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        저장된 CSV 파일 로드

        Args:
            filepath: 파일 경로

        Returns:
            DataFrame
        """
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        if 'close_time' in df.columns:
            df['close_time'] = pd.to_datetime(df['close_time'])

        logger.info(f"📂 파일 로드: {filepath} ({len(df)}개)")

        return df

    def get_recent_data(
        self,
        symbol: str,
        interval: str,
        days: int = 30,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        최근 N일간의 데이터 수집 (캐싱 지원)

        Args:
            symbol: 심볼
            interval: 간격
            days: 일수
            use_cache: 캐시 사용 여부

        Returns:
            DataFrame
        """
        # 캐시 키 생성
        cache_key = f"{symbol}_{interval}_{days}"

        # 캐시 확인
        if use_cache and cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            age = time.time() - cached_time

            if age < self.cache_ttl:
                logger.info(f"📦 캐시 사용: {symbol} (나이: {age:.1f}초)")
                return cached_data.copy()
            else:
                logger.debug(f"⏰ 캐시 만료: {symbol} (나이: {age:.1f}초)")

        # 데이터 수집
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(f"{symbol} 최근 {days}일 데이터 수집")

        df = self.collect_klines(
            symbol=symbol,
            interval=interval,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            save_to_csv=True
        )

        # 캐시에 저장
        if use_cache:
            self._cache[cache_key] = (df, time.time())
            logger.debug(f"💾 캐시 저장: {cache_key}")

        return df

    def clear_cache(self, symbol: str = None):
        """캐시 삭제"""
        if symbol:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(symbol)]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"🗑️  캐시 삭제: {symbol} ({len(keys_to_remove)}개)")
        else:
            self._cache.clear()
            logger.info("🗑️  전체 캐시 삭제")

    def update_existing_data(
        self,
        symbol: str,
        interval: str,
        existing_csv: str
    ) -> pd.DataFrame:
        """
        기존 데이터에 최신 데이터 추가

        Args:
            symbol: 심볼
            interval: 간격
            existing_csv: 기존 CSV 파일 경로

        Returns:
            업데이트된 DataFrame
        """
        # 기존 데이터 로드
        existing_df = self.load_csv(existing_csv)

        if existing_df.empty:
            logger.warning("기존 데이터가 비어있습니다")
            return existing_df

        # 마지막 타임스탬프 이후부터 수집
        last_timestamp = existing_df['timestamp'].max()
        start_date = (last_timestamp + timedelta(seconds=1)).strftime('%Y-%m-%d')

        logger.info(f"업데이트 수집: {start_date} ~ 현재")

        # 새 데이터 수집
        new_df = self.collect_klines(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            save_to_csv=False
        )

        if new_df.empty:
            logger.info("새로운 데이터 없음")
            return existing_df

        # 데이터 병합
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)

        # 중복 제거 (timestamp 기준)
        updated_df = updated_df.drop_duplicates(subset=['timestamp'], keep='last')
        updated_df = updated_df.sort_values('timestamp').reset_index(drop=True)

        # 저장
        updated_df.to_csv(existing_csv, index=False)
        logger.info(f"✅ 데이터 업데이트 완료: {len(existing_df)} → {len(updated_df)}개 (+{len(updated_df) - len(existing_df)})")

        return updated_df
