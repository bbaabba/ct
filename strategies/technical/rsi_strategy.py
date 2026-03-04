"""RSI 평균회귀 전략"""
import pandas as pd
import numpy as np
import talib

from strategies.base import BaseStrategy, Signal, TradeSignal
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RSIStrategy(BaseStrategy):
    """
    RSI 평균회귀 전략

    과매도 구간(RSI < 30): 매수 신호
    과매수 구간(RSI > 70): 매도 신호

    파라미터:
        - rsi_period: RSI 기간 (기본: 14)
        - oversold_level: 과매도 기준 (기본: 30)
        - overbought_level: 과매수 기준 (기본: 70)
        - min_confidence: 최소 신뢰도 (기본: 0.6)
    """

    def __init__(self, params=None):
        default_params = {
            'rsi_period': 14,
            'oversold_level': 30,
            'overbought_level': 70,
            'min_confidence': 0.6,
            'max_position_pct': 0.2
        }

        if params:
            default_params.update(params)

        super().__init__(name='RSI_MeanReversion', params=default_params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """RSI 및 보조 지표 계산"""
        df = data.copy()

        rsi_period = self.params['rsi_period']

        # RSI 계산
        df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_period)

        # RSI 변화율
        df['rsi_change'] = df['rsi'].diff()

        # 볼륨 확인
        df['volume_ma'] = talib.SMA(df['volume'], timeperiod=20)
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # ATR (변동성 확인)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_pct'] = df['atr'] / df['close']

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

        # 현재 데이터
        current = df.iloc[-1]
        previous = df.iloc[-2]

        oversold = self.params['oversold_level']
        overbought = self.params['overbought_level']

        # 신호 결정
        signal = Signal.HOLD
        reason = "중립 구간"
        confidence = 0.0

        # 과매도 → 매수 신호
        if current['rsi'] < oversold:
            signal = Signal.BUY
            reason = f"RSI 과매도 ({current['rsi']:.2f} < {oversold})"

            # 신뢰도 계산: RSI가 낮을수록, 상승 전환 시 높음
            oversold_strength = (oversold - current['rsi']) / oversold
            turning_up = 1.0 if current['rsi_change'] > 0 else 0.5

            confidence = (oversold_strength + turning_up) / 2.0
            confidence = min(confidence, 1.0)

        # 과매수 → 매도 신호
        elif current['rsi'] > overbought:
            signal = Signal.SELL
            reason = f"RSI 과매수 ({current['rsi']:.2f} > {overbought})"

            # 신뢰도 계산: RSI가 높을수록, 하락 전환 시 높음
            overbought_strength = (current['rsi'] - overbought) / (100 - overbought)
            turning_down = 1.0 if current['rsi_change'] < 0 else 0.5

            confidence = (overbought_strength + turning_down) / 2.0
            confidence = min(confidence, 1.0)

        # RSI가 중립 구간 진입
        elif previous['rsi'] < oversold and current['rsi'] >= oversold:
            signal = Signal.BUY
            reason = "RSI 과매도 구간 탈출"
            confidence = 0.3

        elif previous['rsi'] > overbought and current['rsi'] <= overbought:
            signal = Signal.SELL
            reason = "RSI 과매수 구간 탈출"
            confidence = 0.3

        # 중립 구간에서도 방향성 신호 생성 (개선)
        else:
            # RSI가 50 이상이면 상승 압력
            if current['rsi'] >= 50:
                signal = Signal.BUY
                # 50에서 70 사이: 신뢰도 개선된 계산식
                rsi_strength = (current['rsi'] - 50) / 20  # 0~1
                base_confidence = 0.3  # 기본 신뢰도 30%
                rsi_bonus = rsi_strength * 0.3  # RSI 강도에 따라 최대 30% 추가
                confidence = base_confidence + rsi_bonus  # 최대 60%
                reason = f"RSI 중립-강세 ({current['rsi']:.2f})"

            # RSI가 50 미만이면 하락 압력
            else:
                signal = Signal.SELL
                # 30에서 50 사이: 신뢰도 개선된 계산식
                rsi_strength = (50 - current['rsi']) / 20
                base_confidence = 0.3  # 기본 신뢰도 30%
                rsi_bonus = rsi_strength * 0.3  # RSI 강도에 따라 최대 30% 추가
                confidence = base_confidence + rsi_bonus  # 최대 60%
                reason = f"RSI 중립-약세 ({current['rsi']:.2f})"

            # RSI 변화 방향 고려 (momentum boost)
            momentum_multiplier = 1.0
            if current['rsi_change'] > 5:  # 큰 상승
                momentum_multiplier = 1.3
            elif current['rsi_change'] > 0:  # 작은 상승
                momentum_multiplier = 1.15
            elif current['rsi_change'] < -5:  # 큰 하락
                momentum_multiplier = 1.3
            elif current['rsi_change'] < 0:  # 작은 하락
                momentum_multiplier = 1.15

            confidence = min(confidence * momentum_multiplier, 0.7)  # 최대 70%

        # 거래량 및 변동성 고려
        if signal != Signal.HOLD:
            volume_multiplier = min(current['volume_ratio'], 1.5) / 1.5
            confidence = confidence * volume_multiplier

        trade_signal = TradeSignal(
            signal=signal,
            confidence=confidence,
            price=current['close'],
            timestamp=current.name if isinstance(current.name, pd.Timestamp) else pd.Timestamp.now(),
            reason=reason,
            metadata={
                'rsi': current['rsi'],
                'rsi_change': current['rsi_change'],
                'volume_ratio': current['volume_ratio'],
                'atr_pct': current['atr_pct']
            }
        )

        logger.info(f"{self.name} 신호: {signal.name}, 신뢰도: {confidence:.2%}, 이유: {reason}")

        return trade_signal
