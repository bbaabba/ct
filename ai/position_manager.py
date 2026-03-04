"""AI 기반 포지션 관리 시스템"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from utils.logger import setup_logger

logger = setup_logger(__name__)


class PositionAction(Enum):
    """포지션 액션"""
    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    INCREASE_LONG = "INCREASE_LONG"
    INCREASE_SHORT = "INCREASE_SHORT"
    DECREASE_LONG = "DECREASE_LONG"
    DECREASE_SHORT = "DECREASE_SHORT"
    HOLD = "HOLD"


@dataclass
class PositionDecision:
    """포지션 결정 결과"""
    action: PositionAction
    side: str  # 'LONG', 'SHORT', 'NONE'
    confidence: float
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size_pct: float  # 자본 대비 포지션 크기 비율
    reasons: List[str]
    signals: Dict[str, float]  # 각 지표별 신호 강도


class AIPositionManager:
    """
    AI 기반 포지션 관리자

    다양한 기술적 지표와 시장 상황을 종합 분석하여
    포지션 진입/청산을 자동으로 결정합니다.
    """

    # 기본 기술적 지표 가중치 (RSI/MACD 중복 축소, MA/Bollinger 강화)
    ALL_INDICATORS = {
        'trend_ma': 0.20,      # 이동평균 트렌드 (강화)
        'rsi': 0.08,           # RSI (축소: MACD와 중복)
        'macd': 0.08,          # MACD (축소: RSI와 중복)
        'bollinger': 0.18,     # 볼린저 밴드 (강화)
        'volume': 0.08,        # 거래량
        'momentum': 0.08,      # 모멘텀
        'support_resistance': 0.12,  # 지지/저항
        'stochastic': 0.06,    # 스토캐스틱
        'adx': 0.08,           # ADX (추세 강도)
        'obv': 0.04,           # OBV (거래량 흐름)
    }

    def __init__(
        self,
        min_confidence_long: float = 0.55,
        min_confidence_short: float = 0.50,
        max_position_pct: float = 0.3,
        stop_loss_pct: float = 0.01,
        take_profit_pct: float = 0.02,
        risk_reward_ratio: float = 2.0,
        enabled_indicators: list = None,
        volatility_threshold: float = 0.0005,  # 🔹 변동성 필터 임계값
    ):
        """
        Args:
            min_confidence_long: LONG 신호 최소 신뢰도 (LONG 편향 방지)
            min_confidence_short: SHORT 신호 최소 신뢰도
            max_position_pct: 최대 포지션 크기 (자본 대비)
            stop_loss_pct: 기본 손절 비율
            take_profit_pct: 기본 익절 비율
            risk_reward_ratio: 리스크/리워드 비율
            enabled_indicators: 활성화할 지표 목록 (기본: 핵심 4개)
            volatility_threshold: 🔹 변동성 필터 최소 임계값
        """
        self.min_confidence_long = min_confidence_long
        self.min_confidence_short = min_confidence_short
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.risk_reward_ratio = risk_reward_ratio

        # 🔹 변동성 필터 임계값
        self.volatility_threshold = volatility_threshold

        # 활성 지표 설정 (기본: 핵심 4개)
        self.enabled_indicators = enabled_indicators or ['trend_ma', 'rsi', 'macd', 'bollinger']

        # 활성 지표에 대한 가중치 재계산 (합계가 1이 되도록 정규화)
        self.signal_weights = self._calculate_normalized_weights()

        # 지표 건강성 모니터링: 최근 출력값 히스토리 (고착 감지용)
        self._indicator_history: Dict[str, list] = {k: [] for k in self.enabled_indicators}
        self._indicator_health_window = 20  # 최근 20회 체크

        logger.info(f"AIPositionManager 초기화: 활성 지표 {len(self.enabled_indicators)}개 "
                   f"({', '.join(self.enabled_indicators)})")

    def _calculate_normalized_weights(self) -> Dict[str, float]:
        """활성 지표에 대한 정규화된 가중치 계산"""
        # 활성 지표의 원래 가중치 합계
        active_weights = {
            k: v for k, v in self.ALL_INDICATORS.items()
            if k in self.enabled_indicators
        }
        total = sum(active_weights.values())

        # 정규화 (합계 = 1.0)
        if total > 0:
            return {k: v / total for k, v in active_weights.items()}
        else:
            # fallback: 균등 분배
            count = len(self.enabled_indicators)
            return {k: 1.0 / count for k in self.enabled_indicators}

    def check_volatility_filter(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """🔹 변동성 필터: 로그수익률 변동성이 임계값 이상일 때만 거래 허용

        Args:
            df: OHLCV 데이터프레임

        Returns:
            (거래 허용 여부, 현재 변동성)
        """
        if len(df) < 15:
            return True, 0.0

        # 로그수익률 계산
        log_ret = np.log(df['close'] / df['close'].shift(1))
        current_vol = log_ret.rolling(10).std().iloc[-1]

        if np.isnan(current_vol):
            return True, 0.0

        is_tradeable = current_vol > self.volatility_threshold
        return is_tradeable, current_vol

    def set_enabled_indicators(self, indicators: list):
        """활성 지표 변경 (런타임 변경 가능)"""
        valid = [i for i in indicators if i in self.ALL_INDICATORS]
        if len(valid) < 2:
            logger.warning("최소 2개 지표 필요, 기본값 유지")
            return
        self.enabled_indicators = valid
        self.signal_weights = self._calculate_normalized_weights()
        logger.info(f"활성 지표 변경: {', '.join(valid)}")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        df = df.copy()

        # 이동평균
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['ma_25'] = df['close'].rolling(window=25).mean()
        df['ma_99'] = df['close'].rolling(window=99).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 볼린저 밴드
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()

        # 거래량 이동평균
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # 모멘텀
        df['momentum'] = df['close'].pct_change(periods=10) * 100

        # 스토캐스틱 (%K, %D)
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = ((df['close'] - low_14) / (high_14 - low_14).replace(0, 1e-10)) * 100
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # ADX (Average Directional Index)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr_h = df['high'] - df['low']
        tr_hc = np.abs(df['high'] - df['close'].shift())
        tr_lc = np.abs(df['low'] - df['close'].shift())
        tr14 = pd.concat([tr_h, tr_hc, tr_lc], axis=1).max(axis=1).rolling(window=14).mean()

        plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr14.replace(0, 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr14.replace(0, 1e-10))
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)) * 100
        df['adx'] = dx.rolling(window=14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # OBV (On Balance Volume)
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        df['obv_ma'] = df['obv'].rolling(window=20).mean()

        return df

    def analyze_trend_ma(self, df: pd.DataFrame) -> Tuple[float, str]:
        """이동평균 기반 트렌드 분석"""
        current = df.iloc[-1]

        score = 0
        reasons = []

        # 가격과 MA 관계
        if current['close'] > current['ma_7'] > current['ma_25'] > current['ma_99']:
            score = 0.9
            reasons.append("강한 상승 트렌드 (가격 > MA7 > MA25 > MA99)")
        elif current['close'] > current['ma_7'] > current['ma_25']:
            score = 0.7
            reasons.append("상승 트렌드")
        elif current['close'] > current['ma_7']:
            score = 0.55
            reasons.append("약한 상승")
        elif current['close'] < current['ma_7'] < current['ma_25'] < current['ma_99']:
            score = 0.1
            reasons.append("강한 하락 트렌드")
        elif current['close'] < current['ma_7'] < current['ma_25']:
            score = 0.3
            reasons.append("하락 트렌드")
        elif current['close'] < current['ma_7']:
            score = 0.45
            reasons.append("약한 하락")
        else:
            score = 0.5
            reasons.append("횡보")

        return score, '; '.join(reasons)

    def analyze_rsi(self, df: pd.DataFrame) -> Tuple[float, str]:
        """RSI 분석 — 연속 매핑 방식

        기존 문제: 대부분 0.42~0.55 사이에만 머뭄 (사실상 중립)
        개선: RSI 값을 연속적 점수로 매핑 (0.2~0.8 범위)
        """
        rsi = df['rsi'].iloc[-1]
        rsi_prev = df['rsi'].iloc[-2] if len(df) > 1 else rsi

        # RSI(0~100)를 점수(0.8~0.2)로 연속 매핑
        # RSI 30 → 0.71, RSI 50 → 0.50, RSI 70 → 0.29
        # 평균회귀 관점: 낮은 RSI = 매수, 높은 RSI = 매도
        score = 0.80 - (rsi / 100.0) * 0.60  # 0.20 ~ 0.80 범위

        # RSI 극단값 보정: 30~70 범위 밖에서 더 강한 신호
        if rsi < 30:
            score += (30 - rsi) / 100.0 * 0.15  # 추가 매수 신호 (최대 +0.045)
        elif rsi > 70:
            score -= (rsi - 70) / 100.0 * 0.15  # 추가 매도 신호

        # 구간별 사유
        if rsi < 25:
            reason = f"RSI {rsi:.1f}: 과매도 (반등 가능)"
        elif rsi < 40:
            reason = f"RSI {rsi:.1f}: 약세 영역"
        elif rsi > 75:
            reason = f"RSI {rsi:.1f}: 과매수 (조정 가능)"
        elif rsi > 60:
            reason = f"RSI {rsi:.1f}: 강세 영역"
        else:
            reason = f"RSI {rsi:.1f}: 중립"

        # RSI 다이버전스 체크
        if rsi > rsi_prev and df['close'].iloc[-1] < df['close'].iloc[-2]:
            score += 0.05
            reason += " (상승 다이버전스)"
        elif rsi < rsi_prev and df['close'].iloc[-1] > df['close'].iloc[-2]:
            score -= 0.05
            reason += " (하락 다이버전스)"

        return min(max(score, 0.15), 0.85), reason

    def analyze_macd(self, df: pd.DataFrame) -> Tuple[float, str]:
        """MACD 분석"""
        current = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else current

        macd = current['macd']
        signal = current['macd_signal']
        hist = current['macd_hist']
        prev_hist = prev['macd_hist']

        score = 0.5
        reasons = []

        # MACD 크로스
        if macd > signal and prev['macd'] <= prev['macd_signal']:
            score = 0.8
            reasons.append("MACD 골든크로스")
        elif macd < signal and prev['macd'] >= prev['macd_signal']:
            score = 0.2
            reasons.append("MACD 데드크로스")

        # 히스토그램 변화
        if hist > 0:
            if hist > prev_hist:
                score += 0.1
                reasons.append("상승 모멘텀 강화")
            score = min(score + 0.1, 1)
        else:
            if hist < prev_hist:
                score -= 0.1
                reasons.append("하락 모멘텀 강화")
            score = max(score - 0.1, 0)

        # 제로라인 기준
        if macd > 0 and signal > 0:
            score += 0.05
            reasons.append("MACD 양의 영역")
        elif macd < 0 and signal < 0:
            score -= 0.05
            reasons.append("MACD 음의 영역")

        return min(max(score, 0), 1), '; '.join(reasons) if reasons else "MACD 중립"

    def analyze_bollinger(self, df: pd.DataFrame) -> Tuple[float, str]:
        """볼린저 밴드 분석 — 연속 점수 방식

        기존 문제: band_position이 0.2~0.8이면 항상 0.50 → 사실상 무용
        개선: 밴드 위치를 연속적 점수로 변환 (평균회귀 관점)
        """
        current = df.iloc[-1]
        close = current['close']
        upper = current['bb_upper']
        lower = current['bb_lower']

        # 밴드 내 위치 (0: 하단, 0.5: 중간, 1: 상단)
        band_position = (close - lower) / (upper - lower) if upper != lower else 0.5
        band_position = max(0.0, min(1.0, band_position))

        # 연속 점수: 밴드 위치를 평균회귀 관점으로 매핑
        # band_position 0.0 → score 0.85 (하단 = 매수)
        # band_position 0.5 → score 0.50 (중간 = 중립)
        # band_position 1.0 → score 0.15 (상단 = 매도)
        score = 0.85 - band_position * 0.70  # 0.15 ~ 0.85 연속 범위

        # 밴드 방향 추세 반영 (최근 5봉 밴드 중간선 기울기)
        if len(df) >= 5:
            mid_slope = (df['bb_middle'].iloc[-1] - df['bb_middle'].iloc[-5]) / df['bb_middle'].iloc[-5]
            # 상승 추세 시 약간 매수, 하락 추세 시 약간 매도 보정
            score += mid_slope * 50  # ±0.05 정도 보정
            score = max(0.15, min(0.85, score))

        # 점수 구간별 사유
        if band_position < 0.2:
            reason = f"하단 밴드 근접 (위치: {band_position:.0%})"
        elif band_position > 0.8:
            reason = f"상단 밴드 근접 (위치: {band_position:.0%})"
        elif band_position < 0.4:
            reason = f"밴드 하부 (위치: {band_position:.0%})"
        elif band_position > 0.6:
            reason = f"밴드 상부 (위치: {band_position:.0%})"
        else:
            reason = f"밴드 중간 (위치: {band_position:.0%})"

        # 밴드 수축 (스퀴즈) 체크
        if len(df) >= 20:
            avg_width = df['bb_width'].rolling(20).mean().iloc[-1]
            if avg_width > 0 and current['bb_width'] < avg_width * 0.8:
                reason += " - 밴드 수축 (변동성 확대 예상)"

        return score, reason

    def analyze_volume(self, df: pd.DataFrame) -> Tuple[float, str]:
        """거래량 분석"""
        current = df.iloc[-1]
        vol_ratio = current['volume_ratio']

        # 가격 변화 방향
        price_change = (current['close'] - df['close'].iloc[-2]) / df['close'].iloc[-2]

        if vol_ratio > 2.0:
            if price_change > 0:
                score = 0.8
                reason = f"거래량 급증({vol_ratio:.1f}x) + 상승 → 강한 매수세"
            else:
                score = 0.2
                reason = f"거래량 급증({vol_ratio:.1f}x) + 하락 → 강한 매도세"
        elif vol_ratio > 1.5:
            if price_change > 0:
                score = 0.65
                reason = f"거래량 증가({vol_ratio:.1f}x) + 상승"
            else:
                score = 0.35
                reason = f"거래량 증가({vol_ratio:.1f}x) + 하락"
        elif vol_ratio < 0.5:
            score = 0.5
            reason = f"거래량 부족({vol_ratio:.1f}x) - 신뢰도 낮음"
        else:
            score = 0.5
            reason = f"거래량 정상({vol_ratio:.1f}x)"

        return score, reason

    def analyze_momentum(self, df: pd.DataFrame) -> Tuple[float, str]:
        """모멘텀 분석"""
        momentum = df['momentum'].iloc[-1]

        if momentum > 10:
            score = 0.85
            reason = f"강한 상승 모멘텀({momentum:.1f}%)"
        elif momentum > 5:
            score = 0.7
            reason = f"상승 모멘텀({momentum:.1f}%)"
        elif momentum > 0:
            score = 0.55
            reason = f"약한 상승 모멘텀({momentum:.1f}%)"
        elif momentum > -5:
            score = 0.45
            reason = f"약한 하락 모멘텀({momentum:.1f}%)"
        elif momentum > -10:
            score = 0.3
            reason = f"하락 모멘텀({momentum:.1f}%)"
        else:
            score = 0.15
            reason = f"강한 하락 모멘텀({momentum:.1f}%)"

        return score, reason

    def analyze_stochastic(self, df: pd.DataFrame) -> Tuple[float, str]:
        """스토캐스틱 분석 - 역추세 신호 완화 버전

        RSI와 마찬가지로 점수 범위를 제한하여 추세 지표를 완전히 상쇄하지 않음
        """
        k = df['stoch_k'].iloc[-1]
        d = df['stoch_d'].iloc[-1]
        k_prev = df['stoch_k'].iloc[-2] if len(df) > 1 else k

        # 점수 범위를 0.25~0.75로 제한 (극단값에서만 0.2/0.8)
        if k < 15 and d < 15:
            score = 0.75  # 극심한 과매도 (이전 0.85)
            reason = f"Stoch K={k:.0f}/D={d:.0f}: 극심한 과매도 (반등 신호)"
        elif k < 25 and d < 25:
            score = 0.65  # 과매도
            reason = f"Stoch K={k:.0f}/D={d:.0f}: 과매도"
        elif k < 35:
            score = 0.58
            reason = f"Stoch K={k:.0f}/D={d:.0f}: 약한 과매도"
        elif k > 85 and d > 85:
            score = 0.25  # 극심한 과매수 (이전 0.15)
            reason = f"Stoch K={k:.0f}/D={d:.0f}: 극심한 과매수 (조정 신호)"
        elif k > 75 and d > 75:
            score = 0.35  # 과매수
            reason = f"Stoch K={k:.0f}/D={d:.0f}: 과매수"
        elif k > 65:
            score = 0.42
            reason = f"Stoch K={k:.0f}/D={d:.0f}: 약한 과매수"
        else:
            score = 0.5
            reason = f"Stoch K={k:.0f}/D={d:.0f}: 중립"

        # %K가 %D를 상향 돌파 (골든크로스) - 보너스 축소
        if k > d and k_prev <= df['stoch_d'].iloc[-2]:
            score = min(score + 0.05, 0.80)  # 이전 0.1
            reason += " (골든크로스)"
        elif k < d and k_prev >= df['stoch_d'].iloc[-2]:
            score = max(score - 0.05, 0.20)  # 이전 0.1
            reason += " (데드크로스)"

        return min(max(score, 0.2), 0.8), reason

    def analyze_adx(self, df: pd.DataFrame) -> Tuple[float, str]:
        """ADX (추세 강도) 분석"""
        adx = df['adx'].iloc[-1]
        plus_di = df['plus_di'].iloc[-1]
        minus_di = df['minus_di'].iloc[-1]

        # ADX가 높으면 추세가 강함 → 추세 방향에 따라 점수 배정
        if adx > 25:
            # 강한 추세 존재
            if plus_di > minus_di:
                score = 0.5 + min((adx - 25) / 50, 0.4)  # 최대 0.9
                reason = f"ADX={adx:.0f}: 강한 상승 추세 (+DI={plus_di:.0f} > -DI={minus_di:.0f})"
            else:
                score = 0.5 - min((adx - 25) / 50, 0.4)  # 최소 0.1
                reason = f"ADX={adx:.0f}: 강한 하락 추세 (-DI={minus_di:.0f} > +DI={plus_di:.0f})"
        elif adx < 15:
            # 약한 추세 (횡보)
            score = 0.5
            reason = f"ADX={adx:.0f}: 추세 약함 (횡보)"
        else:
            # 보통 추세
            if plus_di > minus_di:
                score = 0.55
                reason = f"ADX={adx:.0f}: 약한 상승 추세"
            else:
                score = 0.45
                reason = f"ADX={adx:.0f}: 약한 하락 추세"

        return score, reason

    def analyze_obv(self, df: pd.DataFrame) -> Tuple[float, str]:
        """OBV (On Balance Volume) 분석"""
        obv_current = df['obv'].iloc[-1]
        obv_ma = df['obv_ma'].iloc[-1]

        # OBV 기울기 (최근 10봉)
        if len(df) >= 10:
            obv_slope = (df['obv'].iloc[-1] - df['obv'].iloc[-10]) / max(abs(df['obv'].iloc[-10]), 1)
        else:
            obv_slope = 0

        # 가격 방향
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 5 else 0

        if obv_current > obv_ma and obv_slope > 0:
            if price_change > 0:
                score = 0.7
                reason = f"OBV 상승 + 가격 상승 (매수세 확인)"
            else:
                score = 0.65
                reason = f"OBV 상승 + 가격 하락 (상승 다이버전스)"
        elif obv_current < obv_ma and obv_slope < 0:
            if price_change < 0:
                score = 0.3
                reason = f"OBV 하락 + 가격 하락 (매도세 확인)"
            else:
                score = 0.35
                reason = f"OBV 하락 + 가격 상승 (하락 다이버전스)"
        else:
            score = 0.5
            reason = "OBV 중립"

        return score, reason

    def analyze_support_resistance(self, df: pd.DataFrame) -> Tuple[float, str]:
        """지지/저항 분석"""
        current_price = df['close'].iloc[-1]

        # 최근 고점/저점
        recent_high = df['high'].rolling(20).max().iloc[-1]
        recent_low = df['low'].rolling(20).min().iloc[-1]

        # 피봇 포인트 (간단 버전)
        prev_high = df['high'].iloc[-2]
        prev_low = df['low'].iloc[-2]
        prev_close = df['close'].iloc[-2]
        pivot = (prev_high + prev_low + prev_close) / 3

        # 지지/저항 레벨
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high

        # 현재 가격 위치 분석
        if current_price > recent_high * 0.99:
            score = 0.65  # 저항 근접 (돌파 가능성)
            reason = "최근 고점 근접 (저항)"
        elif current_price < recent_low * 1.01:
            score = 0.65  # 지지 근접 (반등 가능성)
            reason = "최근 저점 근접 (지지)"
        elif current_price > pivot:
            if current_price > r1:
                score = 0.7
                reason = "R1 저항 돌파 (강세)"
            else:
                score = 0.55
                reason = "피봇 상방"
        else:
            if current_price < s1:
                score = 0.3
                reason = "S1 지지 이탈 (약세)"
            else:
                score = 0.45
                reason = "피봇 하방"

        return score, reason

    def decide_position(
        self,
        df: pd.DataFrame,
        current_position: Optional[Dict] = None,
        leverage: int = 1,
        higher_tf_trend: Optional[str] = None
    ) -> PositionDecision:
        """
        포지션 결정

        Args:
            df: OHLCV 데이터프레임
            current_position: 현재 포지션 정보 {'side': 'LONG'/'SHORT', 'entry_price': float}
            leverage: 적용 레버리지
            higher_tf_trend: 상위 시간대 추세 ('UP', 'DOWN', 'NEUTRAL', None)
                             역추세 진입 시 신뢰도 하향 조정

        Returns:
            PositionDecision 객체
        """
        # 지표 계산
        df = self.calculate_indicators(df)

        if len(df) < 100:
            return PositionDecision(
                action=PositionAction.HOLD,
                side='NONE',
                confidence=0.0,
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                position_size_pct=0.0,
                reasons=["데이터 부족"],
                signals={}
            )

        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]

        # 활성 지표만 분석 (enabled_indicators에 있는 것만)
        signals = {}
        reasons = []

        # 지표 분석 함수 매핑
        indicator_analyzers = {
            'trend_ma': (self.analyze_trend_ma, 'MA'),
            'rsi': (self.analyze_rsi, 'RSI'),
            'macd': (self.analyze_macd, 'MACD'),
            'bollinger': (self.analyze_bollinger, 'BB'),
            'volume': (self.analyze_volume, 'VOL'),
            'momentum': (self.analyze_momentum, 'MOM'),
            'support_resistance': (self.analyze_support_resistance, 'S/R'),
            'stochastic': (self.analyze_stochastic, 'STOCH'),
            'adx': (self.analyze_adx, 'ADX'),
            'obv': (self.analyze_obv, 'OBV'),
        }

        # 활성 지표만 분석
        for indicator_name in self.enabled_indicators:
            if indicator_name in indicator_analyzers:
                analyzer, label = indicator_analyzers[indicator_name]
                score, reason = analyzer(df)
                signals[indicator_name] = score
                reasons.append(f"[{label}] {reason}")

                # 지표 건강성 히스토리 업데이트
                if indicator_name in self._indicator_history:
                    history = self._indicator_history[indicator_name]
                    history.append(score)
                    if len(history) > self._indicator_health_window:
                        history.pop(0)

        # 지표 건강성 체크: 분산이 극도로 낮은 지표 감지 및 가중치 제거
        effective_weights = dict(self.signal_weights)
        stuck_indicators = []
        for ind_name, history in self._indicator_history.items():
            if len(history) >= 10:  # 최소 10회 데이터
                variance = sum((v - sum(history)/len(history))**2 for v in history) / len(history)
                if variance < 0.001:  # 분산 < 0.001 → 사실상 고착
                    effective_weights[ind_name] = 0.0
                    stuck_indicators.append(ind_name)

        if stuck_indicators:
            # 고착 지표 제외 후 가중치 재정규화
            active_total = sum(v for v in effective_weights.values())
            if active_total > 0:
                effective_weights = {k: v / active_total for k, v in effective_weights.items()}
            logger.warning(f"⚠️ 고착 지표 감지 → 가중치 제거: {stuck_indicators}")

        # 가중 평균 점수 계산 (건강한 지표만 반영)
        total_score = sum(
            signals[key] * effective_weights.get(key, 0)
            for key in signals
        )

        # 액션 결정
        action, side = self._determine_action(
            total_score, current_position
        )

        # 신뢰도 계산 v4: 더 적극적인 거래를 위한 상향 조정
        #
        # 핵심 변경:
        # 1. 기존 공식의 구조적 한계 (최대 ~59%) → 80%까지 상향 가능
        # 2. 스케일링 팩터 도입 (1.4배) + 바닥값 상향 (0.30)
        # 3. 강한 신호 보너스 강화

        # 1) 개별 지표 확신도: 각 지표가 0.5에서 벗어난 평균 크기 (가장 중요)
        individual_strength = sum(abs(v - 0.5) for v in signals.values()) / max(len(signals), 1) * 2

        # 2) 방향 편향: 종합 점수가 0.5에서 벗어난 정도
        directional_bias = abs(total_score - 0.5) * 2

        # 3) 지표 일치도 (방향과 무관하게 강한 신호가 많은지)
        strong_signals = sum(1 for v in signals.values() if abs(v - 0.5) > 0.15)
        signal_clarity = strong_signals / max(len(signals), 1)

        # 결합: 개별확신 50% + 방향편향 30% + 신호명확성 20%
        raw_confidence = (
            individual_strength * 0.50 +
            directional_bias * 0.30 +
            signal_clarity * 0.20
        )

        # 스케일링: 기존 최대 0.59 → 1.4배 스케일업 + 바닥값 0.30
        # 결과: 기존 0.59 → 0.30 + 0.59*1.0 = 0.89 (상한 0.85로 제한)
        # 기존 0.40 → 0.30 + 0.40*1.0 = 0.70
        # 기존 0.25 → 0.30 + 0.25*1.0 = 0.55
        confidence = 0.30 + raw_confidence * 1.0  # 바닥값 30% + 원래값

        # 강한 신호 보너스 강화
        very_strong_signals = sum(1 for v in signals.values() if abs(v - 0.5) > 0.25)
        if very_strong_signals >= 3:
            confidence = min(1.0, confidence * 1.15)  # 15% 추가 보너스
        elif very_strong_signals >= 2:
            confidence = min(1.0, confidence * 1.08)  # 8% 추가 보너스

        # 상한 제한 (과도한 확신 방지)
        confidence = min(0.85, confidence)

        # 역추세 진입 제한: 상위 시간대 추세와 반대 방향 진입 시 신뢰도 하향
        # 예: 상위 TF가 UP인데 SHORT 진입 → 신뢰도 60%로 감소
        counter_trend_penalty = 1.0
        if higher_tf_trend and action in [PositionAction.OPEN_LONG, PositionAction.OPEN_SHORT]:
            if higher_tf_trend == 'UP' and action == PositionAction.OPEN_SHORT:
                counter_trend_penalty = 0.6
                reasons.append("[⚠️ 역추세] 상위TF 상승 중 SHORT 진입 → 신뢰도 하향")
            elif higher_tf_trend == 'DOWN' and action == PositionAction.OPEN_LONG:
                counter_trend_penalty = 0.6
                reasons.append("[⚠️ 역추세] 상위TF 하락 중 LONG 진입 → 신뢰도 하향")

        confidence = confidence * counter_trend_penalty
        # 패널티 적용 후 최소값 보장 (후속 AI 패널티와 누적 시 과도한 하락 방지)
        if counter_trend_penalty < 1.0:
            min_floor = self.min_confidence_short * 0.85  # 0.50 * 0.85 = 0.425
            confidence = max(confidence, min_floor)
        confidence = min(1.0, max(0.0, confidence))

        # 포지션 크기 계산 (방향별 최소 신뢰도 적용)
        position_size_pct = self._calculate_position_size(
            confidence, leverage, side
        )

        # SL/TP 계산
        stop_loss, take_profit = self._calculate_sl_tp(
            current_price, side, atr, leverage
        )

        decision = PositionDecision(
            action=action,
            side=side,
            confidence=confidence,
            entry_price=current_price if action in [PositionAction.OPEN_LONG, PositionAction.OPEN_SHORT] else None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size_pct,
            reasons=reasons,
            signals=signals
        )

        logger.info(f"포지션 결정: {action.value} (신뢰도: {confidence:.1%}, 점수: {total_score:.3f})")

        return decision

    def _determine_action(
        self,
        score: float,
        current_position: Optional[Dict]
    ) -> Tuple[PositionAction, str]:
        """점수 기반 액션 결정 (진입/HOLD만 — 청산은 final_score 파이프라인에서 처리)"""
        has_position = current_position is not None
        position_side = current_position.get('side') if has_position else None

        # 포지션 보유 시: TA 단독 청산/축소 하지 않음 → HOLD
        # 청산 판단은 _analyze_position()의 final_score 기반으로 통합
        if has_position:
            return PositionAction.HOLD, position_side or 'NONE'

        # 포지션 없을 때: 진입 신호
        if score > 0.55:
            return PositionAction.OPEN_LONG, 'LONG'
        elif score < 0.45:
            return PositionAction.OPEN_SHORT, 'SHORT'
        else:
            return PositionAction.HOLD, 'NONE'

    def _calculate_position_size(
        self,
        confidence: float,
        leverage: int,
        side: str = 'NONE'
    ) -> float:
        """포지션 크기 계산

        v3: 방향별 최소 신뢰도 차등 적용
        - LONG: 더 높은 최소 신뢰도 요구 (LONG 편향 방지)
        - SHORT: 기본 최소 신뢰도
        - 레버리지 패널티 완화 유지
        """
        # 방향별 최소 신뢰도 체크
        if side == 'LONG':
            min_threshold = self.min_confidence_long
        elif side == 'SHORT':
            min_threshold = self.min_confidence_short
        else:
            min_threshold = max(self.min_confidence_long, self.min_confidence_short)

        if confidence < min_threshold:
            return 0.0

        # 신뢰도에 비례하여 포지션 크기 결정
        base_size = self.max_position_pct * confidence

        # 레버리지에 따른 조정 (완화된 패널티: 0.1 → 0.02)
        # 2x leverage: 0.98배 (기존 0.91배)
        # 5x leverage: 0.93배 (기존 0.71배)
        # 10x leverage: 0.85배 (기존 0.53배)
        leverage_factor = 1.0 / (1 + (leverage - 1) * 0.02)
        adjusted_size = base_size * leverage_factor

        return min(adjusted_size, self.max_position_pct)

    def _calculate_sl_tp(
        self,
        price: float,
        side: str,
        atr: float,
        leverage: int
    ) -> Tuple[Optional[float], Optional[float]]:
        """SL/TP 계산 — sample_buffer 배리어 상수와 동일 기준 사용."""
        if side == 'NONE':
            return None, None

        from ai.sample_buffer import BARRIER_K_TP, BARRIER_K_SL, BARRIER_MIN_TP_PCT, BARRIER_MIN_SL_PCT

        sl_distance = max(atr * BARRIER_K_SL, price * BARRIER_MIN_SL_PCT)
        tp_distance = max(atr * BARRIER_K_TP, price * BARRIER_MIN_TP_PCT)

        if side == 'LONG':
            stop_loss = price - sl_distance
            take_profit = price + tp_distance
        else:  # SHORT
            stop_loss = price + sl_distance
            take_profit = price - tp_distance

        return round(stop_loss, 2), round(take_profit, 2)

    def _calculate_hard_sl(
        self,
        price: float,
        side: str,
        atr: float,
        leverage: int,
        atr_multiplier: float = 1.8
    ) -> Optional[float]:
        """거래소 하드 SL 가격 (catastrophic protection only).

        Layer 1 생존 레이어용. 전략 SL보다 넓게 설정하여
        WS 트레일링이 먼저 반응하도록 함.

        Args:
            atr_multiplier: 1.8 (WS 정상) / 1.0 (WS 단절 폴백)
        """
        if side == 'NONE' or atr <= 0:
            return None

        leverage_factor = 1.0 / (1 + (leverage - 1) * 0.05)
        sl_distance = max(atr * atr_multiplier * leverage_factor, price * self.stop_loss_pct)

        if side == 'LONG':
            return round(price - sl_distance, 2)
        else:
            return round(price + sl_distance, 2)

    def should_close_position(
        self,
        current_price: float,
        position: Dict,
        df: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        포지션 청산 여부 판단

        Args:
            current_price: 현재 가격
            position: 포지션 정보 {'side', 'entry_price', 'stop_loss', 'take_profit'}
            df: 최신 데이터

        Returns:
            (청산 여부, 청산 사유)
        """
        entry_price = position['entry_price']
        side = position['side']
        stop_loss = position.get('stop_loss')
        take_profit = position.get('take_profit')

        # SL/TP 체크
        if side == 'LONG':
            if stop_loss and current_price <= stop_loss:
                return True, f"Stop Loss 도달 ({current_price:.2f} <= {stop_loss:.2f})"
            if take_profit and current_price >= take_profit:
                return True, f"Take Profit 도달 ({current_price:.2f} >= {take_profit:.2f})"
        else:  # SHORT
            if stop_loss and current_price >= stop_loss:
                return True, f"Stop Loss 도달 ({current_price:.2f} >= {stop_loss:.2f})"
            if take_profit and current_price <= take_profit:
                return True, f"Take Profit 도달 ({current_price:.2f} <= {take_profit:.2f})"

        # 반전 신호 체크
        df = self.calculate_indicators(df)
        decision = self.decide_position(df, position)

        if side == 'LONG' and decision.action == PositionAction.CLOSE_LONG:
            return True, f"반전 신호 감지 (신뢰도: {decision.confidence:.1%})"
        elif side == 'SHORT' and decision.action == PositionAction.CLOSE_SHORT:
            return True, f"반전 신호 감지 (신뢰도: {decision.confidence:.1%})"

        return False, "유지"
