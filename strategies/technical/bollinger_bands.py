"""볼린저 밴드 전략"""
import pandas as pd
import numpy as np
import talib

from strategies.base import BaseStrategy, Signal, TradeSignal
from utils.logger import setup_logger

logger = setup_logger(__name__)


class BollingerBandsStrategy(BaseStrategy):
    """
    볼린저 밴드 전략

    하단 밴드 터치/돌파: 매수 신호 (반등 기대)
    상단 밴드 터치/돌파: 매도 신호 (조정 기대)

    파라미터:
        - period: 볼린저 밴드 기간 (기본: 20)
        - std_dev: 표준편차 배수 (기본: 2.0)
        - touch_threshold: 밴드 터치 기준 (기본: 0.02 = 2%)
        - min_confidence: 최소 신뢰도 (기본: 0.6)
    """

    def __init__(self, params=None):
        default_params = {
            'period': 20,
            'std_dev': 2.0,
            'touch_threshold': 0.02,
            'min_confidence': 0.6,
            'max_position_pct': 0.2
        }

        if params:
            default_params.update(params)

        super().__init__(name='BollingerBands', params=default_params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """볼린저 밴드 및 보조 지표 계산"""
        df = data.copy()

        period = self.params['period']
        std_dev = self.params['std_dev']

        # 볼린저 밴드 계산
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'],
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev
        )

        # 밴드 폭 (변동성 지표)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # 가격의 밴드 내 위치 (0 ~ 1)
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # RSI (추가 필터)
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)

        # 거래량
        df['volume_ma'] = talib.SMA(df['volume'], timeperiod=20)
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        return df

    def generate_signal(self, data: pd.DataFrame) -> TradeSignal:
        """거래 신호 생성"""
        if not self.validate_data(data):
            return TradeSignal(
                signal=Signal.HOLD,
                confidence=0.0,
                price=data['close'].iloc[-1] if len(data) > 0 else 0.0,
                timestamp=pd.Timestamp.now(),
                reason="데이터 유효성 검사 실패"
            )

        # 지표 계산
        df = self.calculate_indicators(data)

        # 결측치 제거
        df = df.dropna()

        if len(df) < 2:
            return TradeSignal(
                signal=Signal.HOLD,
                confidence=0.0,
                price=data['close'].iloc[-1],
                timestamp=pd.Timestamp.now(),
                reason="데이터 부족"
            )

        # 현재 및 이전 데이터
        current = df.iloc[-1]
        previous = df.iloc[-2]

        touch_threshold = self.params['touch_threshold']

        # 신호 결정
        signal = Signal.HOLD
        reason = "밴드 중간 구간"
        confidence = 0.0

        # 하단 밴드 터치 → 매수 신호
        lower_touch = (current['close'] - current['bb_lower']) / current['bb_lower']
        if lower_touch < touch_threshold:
            signal = Signal.BUY
            reason = f"볼린저 하단 밴드 터치 (BB%: {current['bb_pct']:.2%})"

            # 신뢰도 계산
            # 1) 밴드 하단에 가까울수록 높음
            band_proximity = 1.0 - current['bb_pct']

            # 2) RSI 과매도 시 추가 가중치
            rsi_factor = 1.0
            if current['rsi'] < 30:
                rsi_factor = 1.3
            elif current['rsi'] < 40:
                rsi_factor = 1.1

            # 3) 거래량 증가 시 추가 가중치
            volume_factor = min(current['volume_ratio'], 1.5) / 1.5

            confidence = band_proximity * rsi_factor * volume_factor
            confidence = min(confidence, 1.0)

        # 상단 밴드 터치 → 매도 신호
        upper_touch = (current['bb_upper'] - current['close']) / current['bb_upper']
        if upper_touch < touch_threshold:
            signal = Signal.SELL
            reason = f"볼린저 상단 밴드 터치 (BB%: {current['bb_pct']:.2%})"

            # 신뢰도 계산
            # 1) 밴드 상단에 가까울수록 높음
            band_proximity = current['bb_pct']

            # 2) RSI 과매수 시 추가 가중치
            rsi_factor = 1.0
            if current['rsi'] > 70:
                rsi_factor = 1.3
            elif current['rsi'] > 60:
                rsi_factor = 1.1

            # 3) 거래량 증가 시 추가 가중치
            volume_factor = min(current['volume_ratio'], 1.5) / 1.5

            confidence = band_proximity * rsi_factor * volume_factor
            confidence = min(confidence, 1.0)

        # 밴드 폭 축소 (변동성 감소) → 큰 움직임 대기
        if current['bb_width'] < 0.05:
            reason = f"밴드 폭 축소 (변동성 낮음: {current['bb_width']:.2%})"
            confidence = 0.0

        trade_signal = TradeSignal(
            signal=signal,
            confidence=confidence,
            price=current['close'],
            timestamp=current.name if isinstance(current.name, pd.Timestamp) else pd.Timestamp.now(),
            reason=reason,
            metadata={
                'bb_upper': current['bb_upper'],
                'bb_middle': current['bb_middle'],
                'bb_lower': current['bb_lower'],
                'bb_width': current['bb_width'],
                'bb_pct': current['bb_pct'],
                'rsi': current['rsi'],
                'volume_ratio': current['volume_ratio']
            }
        )

        logger.info(f"{self.name} 신호: {signal.name}, 신뢰도: {confidence:.2%}, 이유: {reason}")

        return trade_signal
