"""이동평균 교차 전략"""
import pandas as pd
import numpy as np
import talib

from strategies.base import BaseStrategy, Signal, TradeSignal
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MACrossoverStrategy(BaseStrategy):
    """
    이동평균 교차 전략

    골든크로스(단기 MA > 장기 MA): 매수 신호
    데드크로스(단기 MA < 장기 MA): 매도 신호

    파라미터:
        - fast_period: 단기 이동평균 기간 (기본: 7)
        - slow_period: 장기 이동평균 기간 (기본: 25)
        - ma_type: 이동평균 유형 ('SMA', 'EMA', 'WMA') (기본: 'EMA')
        - min_confidence: 최소 신뢰도 (기본: 0.6)
    """

    def __init__(self, params=None):
        default_params = {
            'fast_period': 7,
            'slow_period': 25,
            'ma_type': 'EMA',
            'min_confidence': 0.3,  # 최소 30% (앙상블과 함께 사용 시)
            'max_position_pct': 0.2
        }

        if params:
            default_params.update(params)

        super().__init__(name='MA_Crossover', params=default_params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """이동평균 계산"""
        df = data.copy()

        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']
        ma_type = self.params['ma_type']

        # 이동평균 계산
        if ma_type == 'SMA':
            df['fast_ma'] = talib.SMA(df['close'], timeperiod=fast_period)
            df['slow_ma'] = talib.SMA(df['close'], timeperiod=slow_period)
        elif ma_type == 'EMA':
            df['fast_ma'] = talib.EMA(df['close'], timeperiod=fast_period)
            df['slow_ma'] = talib.EMA(df['close'], timeperiod=slow_period)
        elif ma_type == 'WMA':
            df['fast_ma'] = talib.WMA(df['close'], timeperiod=fast_period)
            df['slow_ma'] = talib.WMA(df['close'], timeperiod=slow_period)
        else:
            raise ValueError(f"지원하지 않는 MA 유형: {ma_type}")

        # MA 차이 계산 (정규화)
        df['ma_diff'] = (df['fast_ma'] - df['slow_ma']) / df['slow_ma']

        # 거래량 확인 (추가 검증용)
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

        # 현재 및 이전 시점 데이터
        current = df.iloc[-1]
        previous = df.iloc[-2]

        # 신호 결정
        signal = Signal.HOLD
        reason = "교차 없음"
        confidence = 0.0

        # 골든크로스 (매수 신호)
        if previous['fast_ma'] <= previous['slow_ma'] and current['fast_ma'] > current['slow_ma']:
            signal = Signal.BUY
            reason = f"골든크로스 발생 (Fast MA: {current['fast_ma']:.2f} > Slow MA: {current['slow_ma']:.2f})"

            # 신뢰도 계산
            ma_diff_strength = abs(current['ma_diff'])
            volume_strength = min(current['volume_ratio'] / 2.0, 1.0)

            confidence = (ma_diff_strength * 100 + volume_strength) / 2.0
            confidence = min(confidence, 1.0)

        # 데드크로스 (매도 신호)
        elif previous['fast_ma'] >= previous['slow_ma'] and current['fast_ma'] < current['slow_ma']:
            signal = Signal.SELL
            reason = f"데드크로스 발생 (Fast MA: {current['fast_ma']:.2f} < Slow MA: {current['slow_ma']:.2f})"

            # 신뢰도 계산
            ma_diff_strength = abs(current['ma_diff'])
            volume_strength = min(current['volume_ratio'] / 2.0, 1.0)

            confidence = (ma_diff_strength * 100 + volume_strength) / 2.0
            confidence = min(confidence, 1.0)

        # 추세 지속 시에도 신호 생성
        else:
            if current['fast_ma'] > current['slow_ma']:
                # 상승 추세 지속 → 매수 신호
                signal = Signal.BUY

                # MA 간격에 비례한 신뢰도 (개선된 계산식)
                ma_spread = abs((current['fast_ma'] - current['slow_ma']) / current['slow_ma'])
                # 스케일링 팩터 증가: 0.1% 차이 → 10% 신뢰도
                trend_strength = min(ma_spread * 100, 0.5)  # 최대 50%

                # 거래량 고려 (거래량이 평균보다 높으면 boost)
                volume_boost = 0.0
                if current['volume_ratio'] > 1.0:
                    volume_boost = min((current['volume_ratio'] - 1.0) * 0.3, 0.15)  # boost 증가

                # 기본 신뢰도 추가 (최소 30%)
                base_confidence = 0.3
                confidence = min(base_confidence + trend_strength + volume_boost, 0.75)
                reason = f"상승 추세 지속 중 (강도: {confidence:.2%})"
            else:
                # 하락 추세 지속 → 매도 신호
                signal = Signal.SELL

                ma_spread = abs((current['slow_ma'] - current['fast_ma']) / current['fast_ma'])
                # 스케일링 팩터 증가
                trend_strength = min(ma_spread * 100, 0.5)  # 최대 50%

                volume_boost = 0.0
                if current['volume_ratio'] > 1.0:
                    volume_boost = min((current['volume_ratio'] - 1.0) * 0.3, 0.15)

                # 기본 신뢰도 추가 (최소 30%)
                base_confidence = 0.3
                confidence = min(base_confidence + trend_strength + volume_boost, 0.75)
                reason = f"하락 추세 지속 중 (강도: {confidence:.2%})"

        trade_signal = TradeSignal(
            signal=signal,
            confidence=confidence,
            price=current['close'],
            timestamp=current.name if isinstance(current.name, pd.Timestamp) else pd.Timestamp.now(),
            reason=reason,
            metadata={
                'fast_ma': current['fast_ma'],
                'slow_ma': current['slow_ma'],
                'ma_diff': current['ma_diff'],
                'volume_ratio': current['volume_ratio']
            }
        )

        logger.info(f"{self.name} 신호: {signal.name}, 신뢰도: {confidence:.2%}, 이유: {reason}")

        return trade_signal
