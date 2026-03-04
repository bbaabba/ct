"""특징 엔지니어링 및 기술 지표 계산"""
import numpy as np
import pandas as pd
import talib
from typing import List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureEngineering:
    """기술 지표 계산 및 특징 생성"""

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 기술 지표 계산

        Args:
            df: OHLCV 데이터프레임

        Returns:
            기술 지표가 추가된 DataFrame
        """
        logger.info(f"기술 지표 계산 시작 (데이터: {len(df)}개)")

        df = df.copy()

        # 1. 이동평균 (Moving Averages)
        df = FeatureEngineering._calculate_moving_averages(df)

        # 2. 모멘텀 지표 (Momentum Indicators)
        df = FeatureEngineering._calculate_momentum_indicators(df)

        # 3. 변동성 지표 (Volatility Indicators)
        df = FeatureEngineering._calculate_volatility_indicators(df)

        # 4. 거래량 지표 (Volume Indicators)
        df = FeatureEngineering._calculate_volume_indicators(df)

        # 5. 추세 지표 (Trend Indicators)
        df = FeatureEngineering._calculate_trend_indicators(df)

        # 6. 가격 변화 지표
        df = FeatureEngineering._calculate_price_changes(df)

        # 7. 추가 특징
        df = FeatureEngineering._calculate_additional_features(df)

        # 결측치 처리
        initial_count = len(df)
        df = df.dropna()
        dropped = initial_count - len(df)

        if dropped > 0:
            logger.info(f"결측치 제거: {dropped}개 (남은 데이터: {len(df)}개)")

        logger.info(f"✅ 기술 지표 계산 완료: {len(df.columns)}개 컬럼")

        return df

    @staticmethod
    def _calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """이동평균 계산"""
        # Simple Moving Averages
        df['SMA_7'] = talib.SMA(df['close'], timeperiod=7)
        df['SMA_25'] = talib.SMA(df['close'], timeperiod=25)
        df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
        df['SMA_99'] = talib.SMA(df['close'], timeperiod=99)

        # Exponential Moving Averages
        df['EMA_7'] = talib.EMA(df['close'], timeperiod=7)
        df['EMA_12'] = talib.EMA(df['close'], timeperiod=12)
        df['EMA_26'] = talib.EMA(df['close'], timeperiod=26)
        df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)

        # Weighted Moving Average
        df['WMA_14'] = talib.WMA(df['close'], timeperiod=14)

        return df

    @staticmethod
    def _calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """모멘텀 지표 계산"""
        # RSI (Relative Strength Index)
        df['RSI_7'] = talib.RSI(df['close'], timeperiod=7)
        df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
        df['RSI_21'] = talib.RSI(df['close'], timeperiod=21)

        # MACD (Moving Average Convergence Divergence)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
            df['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )

        # Stochastic Oscillator
        df['STOCH_k'], df['STOCH_d'] = talib.STOCH(
            df['high'],
            df['low'],
            df['close'],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3
        )

        # CCI (Commodity Channel Index)
        df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)

        # Williams %R
        df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)

        # ROC (Rate of Change)
        df['ROC'] = talib.ROC(df['close'], timeperiod=10)

        # MOM (Momentum)
        df['MOM'] = talib.MOM(df['close'], timeperiod=10)

        return df

    @staticmethod
    def _calculate_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """변동성 지표 계산"""
        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
            df['close'],
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2
        )
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_pct'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # ATR (Average True Range)
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['ATR_pct'] = df['ATR'] / df['close']

        # NATR (Normalized ATR)
        df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)

        # True Range
        df['TRANGE'] = talib.TRANGE(df['high'], df['low'], df['close'])

        return df

    @staticmethod
    def _calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """거래량 지표 계산"""
        # OBV (On Balance Volume)
        df['OBV'] = talib.OBV(df['close'], df['volume'])

        # AD (Accumulation/Distribution)
        df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])

        # ADOSC (Chaikin A/D Oscillator)
        df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])

        # 거래량 이동평균
        df['volume_SMA_7'] = talib.SMA(df['volume'], timeperiod=7)
        df['volume_SMA_20'] = talib.SMA(df['volume'], timeperiod=20)

        # 거래량 비율
        df['volume_ratio'] = df['volume'] / df['volume_SMA_20']

        return df

    @staticmethod
    def _calculate_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """추세 지표 계산"""
        # ADX (Average Directional Index)
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # +DI, -DI
        df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)

        # AROON
        df['AROON_down'], df['AROON_up'] = talib.AROON(df['high'], df['low'], timeperiod=14)

        # SAR (Parabolic SAR)
        df['SAR'] = talib.SAR(df['high'], df['low'])

        return df

    @staticmethod
    def _calculate_price_changes(df: pd.DataFrame) -> pd.DataFrame:
        """가격 변화 지표"""
        # 단순 수익률
        df['returns'] = df['close'].pct_change()

        # 로그 수익률
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # 변화율 (다양한 기간)
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)

        # 고가-저가 비율
        df['high_low_ratio'] = df['high'] / df['low']

        # 종가 위치 (고가-저가 범위 내)
        df['close_location'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        return df

    @staticmethod
    def _calculate_additional_features(df: pd.DataFrame) -> pd.DataFrame:
        """추가 특징"""
        # 변동성 (Rolling Standard Deviation)
        df['volatility_7'] = df['returns'].rolling(window=7).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()

        # 가격 vs 이동평균 차이
        df['price_vs_sma7'] = (df['close'] - df['SMA_7']) / df['SMA_7']
        df['price_vs_sma25'] = (df['close'] - df['SMA_25']) / df['SMA_25']
        df['price_vs_sma50'] = (df['close'] - df['SMA_50']) / df['SMA_50']

        # MA 크로스오버 신호
        df['ma_cross_7_25'] = (df['SMA_7'] > df['SMA_25']).astype(int)
        df['ma_cross_25_50'] = (df['SMA_25'] > df['SMA_50']).astype(int)

        # 시간 특징 (있으면)
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['day_of_month'] = pd.to_datetime(df['timestamp']).dt.day

        return df

    @staticmethod
    def create_sequences(
        data: np.ndarray,
        target: np.ndarray,
        lookback: int = 60,
        forecast_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        시계열 데이터를 모델 입력 형태로 변환

        Args:
            data: 입력 특징 배열 (n_samples, n_features)
            target: 타겟 배열 (n_samples,)
            lookback: 과거 몇 개 시점을 볼 것인가
            forecast_horizon: 몇 시점 후를 예측할 것인가

        Returns:
            X: (n_samples, lookback, n_features)
            y: (n_samples,)
        """
        X, y = [], []

        for i in range(lookback, len(data) - forecast_horizon + 1):
            X.append(data[i - lookback:i])
            y.append(target[i + forecast_horizon - 1])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"시퀀스 생성 완료: X {X.shape}, y {y.shape}")

        return X, y

    @staticmethod
    def normalize_data(
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'minmax'
    ) -> Tuple[pd.DataFrame, object]:
        """
        데이터 정규화

        Args:
            df: DataFrame
            columns: 정규화할 컬럼 리스트
            method: 'minmax' 또는 'standard'

        Returns:
            정규화된 DataFrame, Scaler 객체
        """
        df = df.copy()

        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"지원하지 않는 정규화 방법: {method}")

        df[columns] = scaler.fit_transform(df[columns])

        logger.info(f"데이터 정규화 완료: {method} 방식, {len(columns)}개 컬럼")

        return df, scaler

    @staticmethod
    def get_feature_columns(exclude_base: bool = True) -> List[str]:
        """
        특징 컬럼 이름 리스트 반환

        Args:
            exclude_base: 기본 OHLCV 컬럼 제외 여부

        Returns:
            특징 컬럼 리스트
        """
        feature_columns = [
            # Moving Averages
            'SMA_7', 'SMA_25', 'SMA_50', 'SMA_99',
            'EMA_7', 'EMA_12', 'EMA_26', 'EMA_50',
            'WMA_14',

            # Momentum
            'RSI_7', 'RSI_14', 'RSI_21',
            'MACD', 'MACD_signal', 'MACD_hist',
            'STOCH_k', 'STOCH_d',
            'CCI', 'WILLR', 'ROC', 'MOM',

            # Volatility
            'BB_width', 'BB_pct',
            'ATR', 'ATR_pct', 'NATR',

            # Volume
            'OBV', 'AD', 'ADOSC',
            'volume_ratio',

            # Trend
            'ADX', 'PLUS_DI', 'MINUS_DI',
            'AROON_down', 'AROON_up',

            # Price Changes
            'returns', 'log_returns',
            'price_change_1', 'price_change_5', 'price_change_10',
            'high_low_ratio', 'close_location',

            # Additional
            'volatility_7', 'volatility_20',
            'price_vs_sma7', 'price_vs_sma25', 'price_vs_sma50',
            'ma_cross_7_25', 'ma_cross_25_50',
        ]

        if not exclude_base:
            feature_columns = ['open', 'high', 'low', 'close', 'volume'] + feature_columns

        return feature_columns

    @staticmethod
    def prepare_ml_data(
        df: pd.DataFrame,
        target_column: str = 'close',
        lookback: int = 60,
        forecast_horizon: int = 1,
        normalize: bool = True,
        train_size: float = 0.7,
        val_size: float = 0.15
    ) -> dict:
        """
        머신러닝 학습용 데이터 준비 (전체 파이프라인)

        Args:
            df: 원본 DataFrame
            target_column: 예측 대상 컬럼
            lookback: 모델 입력 시퀀스 길이
            forecast_horizon: 예측 시점
            normalize: 정규화 여부
            train_size: 학습 데이터 비율
            val_size: 검증 데이터 비율

        Returns:
            {
                'X_train', 'y_train',
                'X_val', 'y_val',
                'X_test', 'y_test',
                'scaler', 'feature_columns'
            }
        """
        logger.info("머신러닝 데이터 준비 시작")

        # 1. 기술 지표 계산
        df = FeatureEngineering.calculate_all_indicators(df)

        # 2. 특징 컬럼 선택
        feature_columns = FeatureEngineering.get_feature_columns(exclude_base=False)
        feature_columns = [col for col in feature_columns if col in df.columns]

        logger.info(f"사용 특징: {len(feature_columns)}개")

        # 3. 정규화
        scaler = None
        if normalize:
            df, scaler = FeatureEngineering.normalize_data(df, feature_columns, method='minmax')

        # 4. 시퀀스 생성
        features = df[feature_columns].values
        target = df[target_column].values

        X, y = FeatureEngineering.create_sequences(
            features, target, lookback, forecast_horizon
        )

        # 5. Train/Val/Test 분할
        n_samples = len(X)
        train_end = int(n_samples * train_size)
        val_end = int(n_samples * (train_size + val_size))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info(f"데이터 분할 완료:")
        logger.info(f"  Train: {len(X_train)} ({train_size*100:.0f}%)")
        logger.info(f"  Val:   {len(X_val)} ({val_size*100:.0f}%)")
        logger.info(f"  Test:  {len(X_test)} ({(1-train_size-val_size)*100:.0f}%)")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler,
            'feature_columns': feature_columns
        }
