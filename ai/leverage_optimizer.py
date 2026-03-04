"""AI 기반 레버리지 최적화 시스템"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from utils.logger import setup_logger

logger = setup_logger(__name__)


class RiskLevel(Enum):
    """리스크 레벨"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class LeverageRecommendation:
    """레버리지 추천 결과"""
    recommended_leverage: int
    max_safe_leverage: int
    risk_level: RiskLevel
    confidence: float
    reasons: list
    volatility_score: float
    trend_strength: float
    market_condition: str


class AILeverageOptimizer:
    """
    AI 기반 레버리지 최적화기

    시장 상황, 변동성, 트렌드 강도를 분석하여
    최적의 레버리지를 자동으로 결정합니다.
    """

    def __init__(
        self,
        min_leverage: int = 1,
        max_leverage: int = 20,
        default_leverage: int = 5,
        risk_tolerance: float = 0.5  # 0.0 (보수적) ~ 1.0 (공격적)
    ):
        """
        Args:
            min_leverage: 최소 레버리지
            max_leverage: 최대 레버리지
            default_leverage: 기본 레버리지
            risk_tolerance: 리스크 허용도
        """
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        self.default_leverage = default_leverage
        self.risk_tolerance = risk_tolerance

        # 레버리지 결정 가중치
        self.weights = {
            'volatility': 0.35,      # 변동성 (높을수록 낮은 레버리지)
            'trend_strength': 0.25,  # 트렌드 강도 (강할수록 높은 레버리지)
            'volume': 0.15,          # 거래량 (많을수록 안정적)
            'market_regime': 0.15,   # 시장 상태
            'confidence': 0.10       # 신호 신뢰도
        }

        # 연속 손실 레버리지 감소 상태 (중복 조정 방지)
        self._consec_loss_reduced: bool = False

        logger.info(f"AILeverageOptimizer 초기화 (범위: {min_leverage}x-{max_leverage}x)")

    def calculate_volatility_score(self, df: pd.DataFrame) -> float:
        """
        변동성 점수 계산 (0-1, 높을수록 변동성 높음)

        Args:
            df: OHLCV 데이터프레임

        Returns:
            변동성 점수
        """
        if len(df) < 20:
            return 0.5

        # 1. ATR 기반 변동성
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))

        tr = np.maximum(np.maximum(tr1, tr2), tr3)[1:]
        atr = np.mean(tr[-14:])
        atr_pct = atr / close[-1] * 100

        # 2. 표준편차 기반 변동성
        returns = np.diff(np.log(close))
        std_returns = np.std(returns[-20:]) * np.sqrt(24)  # 일간 변동성 환산

        # 3. 최근 가격 범위
        recent_range = (max(high[-10:]) - min(low[-10:])) / close[-1] * 100

        # 종합 변동성 점수 (0-1 정규화)
        vol_score = 0
        vol_score += min(atr_pct / 5.0, 1.0) * 0.4  # ATR 5% 이상이면 최대
        vol_score += min(std_returns / 0.05, 1.0) * 0.3  # 일간 5% 이상이면 최대
        vol_score += min(recent_range / 10.0, 1.0) * 0.3  # 10% 범위 이상이면 최대

        return min(max(vol_score, 0.0), 1.0)

    def calculate_trend_strength(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        트렌드 강도 계산

        Returns:
            (트렌드 강도 0-1, 트렌드 방향 'BULLISH'/'BEARISH'/'NEUTRAL')
        """
        if len(df) < 50:
            return 0.5, 'NEUTRAL'

        close = df['close'].values

        # 이동평균 계산
        ma_short = np.mean(close[-10:])
        ma_mid = np.mean(close[-20:])
        ma_long = np.mean(close[-50:])

        current_price = close[-1]

        # 트렌드 방향 결정
        bullish_signals = 0
        bearish_signals = 0

        if current_price > ma_short:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if ma_short > ma_mid:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if ma_mid > ma_long:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if current_price > ma_long:
            bullish_signals += 1
        else:
            bearish_signals += 1

        # 트렌드 강도 계산
        total_signals = bullish_signals + bearish_signals
        strength = abs(bullish_signals - bearish_signals) / total_signals

        # ADX 유사 지표 (간소화)
        high = df['high'].values[-14:]
        low = df['low'].values[-14:]

        dm_plus = np.maximum(np.diff(high), 0)
        dm_minus = np.maximum(-np.diff(low), 0)

        di_plus = np.mean(dm_plus)
        di_minus = np.mean(dm_minus)

        if di_plus + di_minus > 0:
            dx = abs(di_plus - di_minus) / (di_plus + di_minus)
            strength = (strength + dx) / 2

        # 트렌드 방향
        if bullish_signals > bearish_signals:
            direction = 'BULLISH'
        elif bearish_signals > bullish_signals:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'

        return min(max(strength, 0.0), 1.0), direction

    def calculate_volume_score(self, df: pd.DataFrame) -> float:
        """
        거래량 점수 계산 (높을수록 유동성 좋음)
        """
        if len(df) < 20:
            return 0.5

        volume = df['volume'].values

        # 최근 거래량 vs 평균
        recent_vol = np.mean(volume[-5:])
        avg_vol = np.mean(volume[-20:])

        if avg_vol == 0:
            return 0.5

        vol_ratio = recent_vol / avg_vol

        # 거래량이 평균의 0.5배 ~ 2배 사이면 안정적
        if vol_ratio < 0.5:
            score = vol_ratio  # 거래량 부족
        elif vol_ratio > 2.0:
            score = 1.0 - min((vol_ratio - 2.0) / 3.0, 0.5)  # 과도한 거래량은 불안정
        else:
            score = 0.7 + (1 - abs(1 - vol_ratio)) * 0.3  # 정상 범위

        return min(max(score, 0.0), 1.0)

    def analyze_market_condition(
        self,
        df: pd.DataFrame,
        volatility_score: float,
        trend_strength: float
    ) -> Tuple[str, float]:
        """
        시장 상황 종합 분석

        Returns:
            (시장 상태, 안정성 점수)
        """
        # 시장 상태 분류
        if volatility_score > 0.7:
            if trend_strength > 0.6:
                condition = "VOLATILE_TRENDING"
                stability = 0.3
            else:
                condition = "VOLATILE_CHOPPY"
                stability = 0.2
        elif volatility_score > 0.4:
            if trend_strength > 0.5:
                condition = "NORMAL_TRENDING"
                stability = 0.7
            else:
                condition = "NORMAL_RANGING"
                stability = 0.5
        else:
            if trend_strength > 0.6:
                condition = "CALM_TRENDING"
                stability = 0.9
            else:
                condition = "CALM_RANGING"
                stability = 0.6

        return condition, stability

    def recommend_leverage(
        self,
        df: pd.DataFrame,
        signal_confidence: float = 0.5,
        position_side: str = 'LONG'
    ) -> LeverageRecommendation:
        """
        최적 레버리지 추천

        Args:
            df: OHLCV 데이터프레임
            signal_confidence: 전략 신호 신뢰도 (0-1)
            position_side: 포지션 방향

        Returns:
            LeverageRecommendation 객체
        """
        reasons = []

        # 1. 변동성 분석
        volatility_score = self.calculate_volatility_score(df)

        # 2. 트렌드 분석
        trend_strength, trend_direction = self.calculate_trend_strength(df)

        # 3. 거래량 분석
        volume_score = self.calculate_volume_score(df)

        # 4. 시장 상황 분석
        market_condition, stability = self.analyze_market_condition(
            df, volatility_score, trend_strength
        )

        # 5. 레버리지 점수 계산
        leverage_score = 0

        # 변동성: 낮을수록 높은 레버리지 가능
        vol_contribution = (1 - volatility_score) * self.weights['volatility']
        leverage_score += vol_contribution

        # 트렌드: 강할수록 높은 레버리지 가능
        trend_contribution = trend_strength * self.weights['trend_strength']
        leverage_score += trend_contribution

        # 거래량: 안정적일수록 높은 레버리지 가능
        vol_contribution = volume_score * self.weights['volume']
        leverage_score += vol_contribution

        # 시장 상태: 안정적일수록 높은 레버리지 가능
        market_contribution = stability * self.weights['market_regime']
        leverage_score += market_contribution

        # 신호 신뢰도
        conf_contribution = signal_confidence * self.weights['confidence']
        leverage_score += conf_contribution

        # 리스크 허용도 적용
        leverage_score = leverage_score * (0.5 + self.risk_tolerance * 0.5)

        # 6. 최종 레버리지 계산
        leverage_range = self.max_leverage - self.min_leverage
        raw_leverage = self.min_leverage + leverage_score * leverage_range

        # 트렌드 방향과 포지션 방향 일치 보너스
        if (trend_direction == 'BULLISH' and position_side == 'LONG') or \
           (trend_direction == 'BEARISH' and position_side == 'SHORT'):
            raw_leverage *= 1.1
            reasons.append(f"트렌드 방향({trend_direction})과 포지션 방향 일치 → 레버리지 +10%")
        elif trend_direction != 'NEUTRAL':
            raw_leverage *= 0.9
            reasons.append(f"트렌드 역방향 거래 → 레버리지 -10%")

        # 변동성 경고
        if volatility_score > 0.7:
            raw_leverage *= 0.7
            reasons.append(f"높은 변동성({volatility_score:.1%}) → 레버리지 대폭 감소")
        elif volatility_score > 0.5:
            reasons.append(f"중간 변동성({volatility_score:.1%}) → 적절한 레버리지")
        else:
            reasons.append(f"낮은 변동성({volatility_score:.1%}) → 레버리지 여유 있음")

        # 트렌드 강도 피드백
        if trend_strength > 0.7:
            reasons.append(f"강한 트렌드({trend_strength:.1%}) → 추세 추종에 유리")
        elif trend_strength < 0.3:
            raw_leverage *= 0.85
            reasons.append(f"약한 트렌드({trend_strength:.1%}) → 횡보장 주의")

        # 정수로 반올림
        recommended = int(round(raw_leverage))
        recommended = max(self.min_leverage, min(self.max_leverage, recommended))

        # 최대 안전 레버리지 (변동성 기반)
        max_safe = int(round((1 - volatility_score * 0.7) * self.max_leverage))
        max_safe = max(self.min_leverage, min(self.max_leverage, max_safe))

        # 리스크 레벨 결정
        risk_level = self._determine_risk_level(
            recommended, volatility_score, trend_strength
        )

        recommendation = LeverageRecommendation(
            recommended_leverage=recommended,
            max_safe_leverage=max_safe,
            risk_level=risk_level,
            confidence=leverage_score,
            reasons=reasons,
            volatility_score=volatility_score,
            trend_strength=trend_strength,
            market_condition=market_condition
        )

        logger.info(f"레버리지 추천: {recommended}x (최대안전: {max_safe}x, 상태: {market_condition})")

        return recommendation

    def _determine_risk_level(
        self,
        leverage: int,
        volatility: float,
        trend_strength: float
    ) -> RiskLevel:
        """리스크 레벨 결정"""
        # 종합 리스크 점수
        risk_score = 0

        # 레버리지 기여
        lev_ratio = leverage / self.max_leverage
        risk_score += lev_ratio * 0.4

        # 변동성 기여
        risk_score += volatility * 0.4

        # 트렌드 약함 = 리스크 증가
        risk_score += (1 - trend_strength) * 0.2

        if risk_score < 0.2:
            return RiskLevel.VERY_LOW
        elif risk_score < 0.4:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MEDIUM
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def adjust_leverage_for_consecutive_losses(
        self,
        current_leverage: int,
        consecutive_losses: int
    ) -> int:
        """
        연속 손실에 따른 레버리지 조정

        Args:
            current_leverage: 현재 레버리지
            consecutive_losses: 연속 손실 횟수

        Returns:
            조정된 레버리지
        """
        # 연속 승리 시 감소 플래그 리셋
        if consecutive_losses == 0:
            self._consec_loss_reduced = False

        if consecutive_losses >= 5:
            # 5연패 이상: 최소 레버리지 (항상 적용)
            new_leverage = self.min_leverage
        elif consecutive_losses >= 3 and not self._consec_loss_reduced:
            # 3연패 (최초 1회만): 레버리지 50% 감소
            new_leverage = max(self.min_leverage, current_leverage // 2)
            self._consec_loss_reduced = True
        elif consecutive_losses >= 2:
            # 2연패: 레버리지 1단계 감소
            new_leverage = max(self.min_leverage, current_leverage - 1)
        else:
            new_leverage = current_leverage

        if new_leverage != current_leverage:
            logger.warning(f"연속 손실({consecutive_losses}회)로 레버리지 조정: {current_leverage}x → {new_leverage}x")

        return new_leverage

    def adjust_leverage_for_drawdown(
        self,
        current_leverage: int,
        drawdown_pct: float
    ) -> int:
        """
        드로다운에 따른 레버리지 조정

        Args:
            current_leverage: 현재 레버리지
            drawdown_pct: 드로다운 비율 (0-1)

        Returns:
            조정된 레버리지
        """
        if drawdown_pct > 0.20:  # 20% 이상 손실
            new_leverage = self.min_leverage
        elif drawdown_pct > 0.15:  # 15% 이상 손실
            new_leverage = max(self.min_leverage, current_leverage // 2)
        elif drawdown_pct > 0.10:  # 10% 이상 손실
            new_leverage = max(self.min_leverage, current_leverage - 2)
        elif drawdown_pct > 0.05:  # 5% 이상 손실
            new_leverage = max(self.min_leverage, current_leverage - 1)
        else:
            new_leverage = current_leverage

        if new_leverage != current_leverage:
            logger.warning(f"드로다운({drawdown_pct:.1%})으로 레버리지 조정: {current_leverage}x → {new_leverage}x")

        return new_leverage
