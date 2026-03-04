"""MoE 기반 확신도 스케일링 — 포지션 사이즈 및 레버리지 연동.

MoE 라우터 출력(방향 확신도, 라우터 집중도, 타이밍 확신도)을
종합하여 포지션 크기와 레버리지 조절에 활용.

사용:
    scaler = MoEConfidenceScaler()
    conf = scaler.compute(
        direction_confidence=0.82,
        router_sharpness=0.65,
        timing_confidence=0.71,
    )
    position_size *= conf.size_multiplier   # 0.5x ~ 1.5x
    leverage_optimizer(signal_confidence=conf.leverage_confidence)
"""
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class MoEConfidence:
    """MoE 종합 확신도 결과."""
    combined_score: float        # [0, 1] 종합 확신도
    direction_confidence: float  # max(p_short, p_long)
    router_sharpness: float      # 1 - normalized_entropy (0=균일, 1=단일)
    timing_confidence: float     # timing_head sigmoid 출력
    size_multiplier: float       # 포지션 크기 배율 (0.5x ~ 1.5x)
    leverage_confidence: float   # 레버리지 옵티마이저에 전달할 값


class MoEConfidenceScaler:
    """MoE 모델 출력을 종합하여 포지션/레버리지 스케일링 생성."""

    def __init__(
        self,
        direction_weight: float = 0.50,
        sharpness_weight: float = 0.30,
        timing_weight: float = 0.20,
        min_size_mult: float = 0.5,
        max_size_mult: float = 1.5,
    ):
        assert abs(direction_weight + sharpness_weight + timing_weight - 1.0) < 0.01
        self.direction_weight = direction_weight
        self.sharpness_weight = sharpness_weight
        self.timing_weight = timing_weight
        self.min_size_mult = min_size_mult
        self.max_size_mult = max_size_mult

    def compute(
        self,
        direction_confidence: float,
        router_sharpness: float,
        timing_confidence: float,
    ) -> MoEConfidence:
        """MoE 출력 3가지를 종합 확신도로 변환.

        Args:
            direction_confidence: max(p_short, p_long) from softmax. [0.33, 1.0]
            router_sharpness: 1 - normalized_entropy. [0, 1]
            timing_confidence: timing_head sigmoid 출력. [0, 1]

        Returns:
            MoEConfidence with combined_score, size_multiplier, leverage_confidence.
        """
        # direction_confidence를 [0.33, 1.0] → [0, 1]로 정규화
        dir_norm = max(0.0, min(1.0, (direction_confidence - 0.33) / 0.67))

        combined = (
            self.direction_weight * dir_norm
            + self.sharpness_weight * max(0.0, min(1.0, router_sharpness))
            + self.timing_weight * max(0.0, min(1.0, timing_confidence))
        )
        combined = max(0.0, min(1.0, combined))

        # 포지션 크기 배율: [0, 1] → [min_size_mult, max_size_mult]
        size_mult = self.min_size_mult + combined * (self.max_size_mult - self.min_size_mult)

        return MoEConfidence(
            combined_score=round(combined, 4),
            direction_confidence=round(direction_confidence, 4),
            router_sharpness=round(router_sharpness, 4),
            timing_confidence=round(timing_confidence, 4),
            size_multiplier=round(size_mult, 4),
            leverage_confidence=round(combined, 4),
        )
