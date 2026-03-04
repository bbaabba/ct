"""시장 상태(Regime) 감지 시스템"""
import pandas as pd
import numpy as np
import talib
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)


class MarketRegime(Enum):
    """시장 상태 타입"""
    TRENDING_UP = "trending_up"        # 상승 추세
    TRENDING_DOWN = "trending_down"    # 하락 추세
    RANGING = "ranging"                # 횡보
    HIGH_VOLATILITY = "high_volatility"  # 고변동성
    LOW_VOLATILITY = "low_volatility"    # 저변동성


@dataclass
class RegimeInfo:
    """시장 상태 정보"""
    regime: MarketRegime
    confidence: float  # 0.0 ~ 1.0
    metrics: Dict[str, float]
    timestamp: pd.Timestamp


class RegimeDetector:
    """
    시장 상태 감지기

    다양한 지표를 종합하여 현재 시장 상태를 판단합니다.
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Args:
            params: 감지 파라미터
        """
        default_params = {
            # ADX 임계값
            'adx_trending_threshold': 25,
            'adx_strong_trend_threshold': 40,

            # 변동성 임계값 (ATR 기준)
            'volatility_high_threshold': 0.03,  # 3%
            'volatility_low_threshold': 0.01,   # 1%

            # 추세 강도 (가격 vs MA)
            'trend_strength_threshold': 0.02,   # 2%

            # 분석 기간
            'lookback_period': 50
        }

        if params:
            default_params.update(params)

        self.params = default_params
        logger.info(f"RegimeDetector 초기화: {self.params}")

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """시장 상태 판단을 위한 지표 계산"""
        df = data.copy()

        # 1. ADX (추세 강도)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)

        # 2. 이동평균
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)

        # 3. ATR (변동성)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_pct'] = df['atr'] / df['close']

        # 4. Bollinger Bands (변동성)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # 5. 가격 변동성 (Rolling Std)
        df['returns'] = df['close'].pct_change()
        df['volatility_20'] = df['returns'].rolling(window=20).std()

        # 6. Linear Regression (추세 방향)
        df['lr_slope'] = self._calculate_linear_regression_slope(df['close'], window=20)

        return df

    def _calculate_linear_regression_slope(self, series: pd.Series, window: int = 20) -> pd.Series:
        """선형 회귀 기울기 계산"""
        slopes = []

        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i - window + 1:i + 1].values
                x = np.arange(window)

                # 선형 회귀
                slope, _ = np.polyfit(x, y, 1)
                slopes.append(slope)

        return pd.Series(slopes, index=series.index)

    def detect_regime(self, data: pd.DataFrame) -> RegimeInfo:
        """
        시장 상태 감지

        Args:
            data: OHLCV 데이터프레임

        Returns:
            RegimeInfo 객체
        """
        if len(data) < self.params['lookback_period']:
            logger.warning(f"데이터 부족: {len(data)} < {self.params['lookback_period']}")
            return RegimeInfo(
                regime=MarketRegime.RANGING,
                confidence=0.0,
                metrics={},
                timestamp=pd.Timestamp.now()
            )

        # 지표 계산
        df = self.calculate_indicators(data)
        df = df.dropna()

        if len(df) == 0:
            return RegimeInfo(
                regime=MarketRegime.RANGING,
                confidence=0.0,
                metrics={},
                timestamp=pd.Timestamp.now()
            )

        # 현재 시점 데이터
        current = df.iloc[-1]

        # 메트릭 수집
        metrics = {
            'adx': current['adx'],
            'plus_di': current['plus_di'],
            'minus_di': current['minus_di'],
            'atr_pct': current['atr_pct'],
            'bb_width': current['bb_width'],
            'volatility_20': current['volatility_20'],
            'lr_slope': current['lr_slope'],
            'price_vs_sma20': (current['close'] - current['sma_20']) / current['sma_20'],
            'price_vs_sma50': (current['close'] - current['sma_50']) / current['sma_50']
        }

        # 1차: 변동성 판단
        regime_scores = {
            MarketRegime.TRENDING_UP: 0.0,
            MarketRegime.TRENDING_DOWN: 0.0,
            MarketRegime.RANGING: 0.0,
            MarketRegime.HIGH_VOLATILITY: 0.0,
            MarketRegime.LOW_VOLATILITY: 0.0
        }

        # 변동성 점수
        if current['atr_pct'] > self.params['volatility_high_threshold']:
            regime_scores[MarketRegime.HIGH_VOLATILITY] += 2.0
        elif current['atr_pct'] < self.params['volatility_low_threshold']:
            regime_scores[MarketRegime.LOW_VOLATILITY] += 2.0

        if current['bb_width'] > 0.1:  # 볼린저 밴드 폭이 넓음
            regime_scores[MarketRegime.HIGH_VOLATILITY] += 1.0
        elif current['bb_width'] < 0.05:
            regime_scores[MarketRegime.LOW_VOLATILITY] += 1.0

        # 추세 점수 (ADX 기반)
        if current['adx'] > self.params['adx_strong_trend_threshold']:
            # 강한 추세
            if current['plus_di'] > current['minus_di']:
                regime_scores[MarketRegime.TRENDING_UP] += 3.0
            else:
                regime_scores[MarketRegime.TRENDING_DOWN] += 3.0

        elif current['adx'] > self.params['adx_trending_threshold']:
            # 중간 강도 추세
            if current['plus_di'] > current['minus_di']:
                regime_scores[MarketRegime.TRENDING_UP] += 2.0
            else:
                regime_scores[MarketRegime.TRENDING_DOWN] += 2.0
        else:
            # 약한 추세 → 횡보
            regime_scores[MarketRegime.RANGING] += 2.0

        # 선형 회귀 기울기
        if current['lr_slope'] > 0:
            regime_scores[MarketRegime.TRENDING_UP] += 1.0
        elif current['lr_slope'] < 0:
            regime_scores[MarketRegime.TRENDING_DOWN] += 1.0
        else:
            regime_scores[MarketRegime.RANGING] += 1.0

        # 가격 vs 이동평균
        price_vs_sma20 = metrics['price_vs_sma20']
        price_vs_sma50 = metrics['price_vs_sma50']

        if price_vs_sma20 > self.params['trend_strength_threshold'] and \
           price_vs_sma50 > self.params['trend_strength_threshold']:
            regime_scores[MarketRegime.TRENDING_UP] += 1.5
        elif price_vs_sma20 < -self.params['trend_strength_threshold'] and \
             price_vs_sma50 < -self.params['trend_strength_threshold']:
            regime_scores[MarketRegime.TRENDING_DOWN] += 1.5
        else:
            regime_scores[MarketRegime.RANGING] += 1.0

        # 최종 판단
        regime = max(regime_scores, key=regime_scores.get)
        max_score = regime_scores[regime]
        total_score = sum(regime_scores.values())

        confidence = max_score / total_score if total_score > 0 else 0.0

        regime_info = RegimeInfo(
            regime=regime,
            confidence=confidence,
            metrics=metrics,
            timestamp=current.name if isinstance(current.name, pd.Timestamp) else pd.Timestamp.now()
        )

        logger.info(f"시장 상태: {regime.value}, 신뢰도: {confidence:.2%}")
        logger.info(f"  ADX: {metrics['adx']:.2f}, ATR%: {metrics['atr_pct']:.2%}")

        return regime_info

    def get_suitable_strategies(self, regime: MarketRegime) -> list:
        """
        시장 상태에 적합한 전략 추천

        Args:
            regime: 시장 상태

        Returns:
            추천 전략 리스트
        """
        strategy_map = {
            MarketRegime.TRENDING_UP: ['MA_Crossover', 'Transformer'],
            MarketRegime.TRENDING_DOWN: ['MA_Crossover', 'Transformer'],
            MarketRegime.RANGING: ['RSI', 'BollingerBands'],
            MarketRegime.HIGH_VOLATILITY: ['BollingerBands', 'RSI'],
            MarketRegime.LOW_VOLATILITY: ['MA_Crossover', 'Transformer']
        }

        return strategy_map.get(regime, ['MA_Crossover'])
