"""AI 시장 분석기 - TCN/GRU 기반 선물 거래 시스템"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os

from utils.logger import setup_logger
from collections import deque

logger = setup_logger(__name__)

# 2-class direction mapping: DOWN=0, UP=1 (NEUTRAL은 학습 제외, 추론에서만 사용)
DIR_MAP = {'DOWN': 0, 'UP': 1}
DIR_MAP_INV = {0: 'DOWN', 1: 'UP'}


class CostSensitiveFocalLoss(nn.Module):
    """
    Cost-Sensitive Focal Loss — 2-class (DOWN=0, UP=1)

    Cost Matrix (actual x predicted):
                  Pred DOWN  Pred UP
    Actual DOWN    0.0        1.0    <- 역방향 = 최대 비용
    Actual UP      1.0        0.0    <- 역방향 = 최대 비용

    Loss = alpha[y] * (1-p_y)^gamma * CE(y) + beta * sum_{j!=y} cost[y,j] * p_j
    """

    def __init__(self, cost_matrix=None, alpha=None, gamma=2.0, beta=0.5,
                 label_smoothing: float = 0.05, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if cost_matrix is None:
            cost_matrix = torch.tensor([
                [0.0, 1.0],   # actual DOWN: 역방향(UP)=1.0
                [1.0, 0.0],   # actual UP: 역방향(DOWN)=1.0
            ], dtype=torch.float32)
        self.register_buffer('cost_matrix', cost_matrix)

        if alpha is None:
            alpha = torch.tensor([1.0, 1.0], dtype=torch.float32)
        self.register_buffer('alpha', alpha)

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch, 2) raw logits (before softmax)
            targets: (batch,) class indices {0=DOWN, 1=UP}
        Returns:
            scalar loss (if reduction='mean') or (batch,) per-sample loss
        """
        ce = nn.functional.cross_entropy(
            logits, targets, reduction='none',
            label_smoothing=self.label_smoothing
        )
        p_t = torch.exp(-ce)
        focal = self.alpha[targets] * (1 - p_t) ** self.gamma * ce

        probs = nn.functional.softmax(logits, dim=1)
        cost_penalty = (self.cost_matrix[targets] * probs).sum(dim=1)

        total = focal + self.beta * cost_penalty

        if self.reduction == 'mean':
            return total.mean()
        elif self.reduction == 'sum':
            return total.sum()
        return total


class PnLLoss(nn.Module):
    """
    수수료 + 슬리피지 포함 PnL 기반 손실 함수

    loss = -((tanh(outputs) * future_returns - |tanh(outputs)| * (fee + slippage))).mean()

    핵심: 모델이 수수료를 넘는 수익을 내도록 학습
    - tanh(outputs): soft position [-1, 1]
    - future_returns: 다음 캔들의 로그 수익률
    - fee + slippage: 편도 거래 비용
    """
    def __init__(self, fee: float = 0.0004, slippage: float = 0.0005):
        super(PnLLoss, self).__init__()
        self.total_cost = fee + slippage

    def forward(self, outputs: torch.Tensor, future_returns: torch.Tensor) -> torch.Tensor:
        """
        Args:
            outputs: 모델 raw 출력 (batch,)
            future_returns: 다음 캔들의 로그 수익률 (batch,)
        """
        position = torch.tanh(outputs)
        gross_pnl = position * future_returns
        cost = torch.abs(position) * self.total_cost
        net_pnl = gross_pnl - cost
        return -net_pnl.mean()


@dataclass
class TrainingSample:
    """학습용 샘플 데이터"""
    features: np.ndarray  # (sequence_length, num_features)
    actual_direction: str  # 'UP', 'DOWN', 'NEUTRAL'
    actual_price_change: float
    timestamp: datetime
    sample_weight: float = 1.0  # 수익성 기반 가중치 (수수료 초과 시 높은 가중치)
    future_log_return: float = 0.0  # 다음 캔들 로그 수익률 (PnL Loss용)


@dataclass
class ModelPrediction:
    """모델 예측 결과"""
    predicted_price: float
    predicted_direction: str  # 'UP', 'DOWN', 'NEUTRAL'
    direction_confidence: float
    price_change_pct: float
    attention_weights: Optional[np.ndarray]
    feature_importance: Dict[str, float]
    timestamp: datetime
    direction_probs: Optional[Dict[str, float]] = None  # {'short': p, 'hold': p, 'long': p}


class RevIN(nn.Module):
    """Reversible Instance Normalization - 시계열 분포 변동 대응

    암호화폐처럼 가격 레벨이 급변하는 비정상(non-stationary) 시계열에서
    입력 정규화 → Transformer 처리 후 역정규화로 분포 안정성 확보
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        # forward 시 저장 (denorm용)
        self.mean: Optional[torch.Tensor] = None
        self.stdev: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
            mode: 'norm' (정규화) 또는 'denorm' (역정규화)
        """
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
        return x


# ─── TCN 공통 빌딩블록 ─────────────────────────────────────────

class Chomp1d(nn.Module):
    """인과성 보장: 오른쪽 패딩(미래) 제거"""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN 단일 블록: dilated causal conv x2 + residual"""

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """TCN 블록 스택: 지수적 dilation으로 receptive field 확장"""

    def __init__(self, num_inputs: int, num_channels: List[int],
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size,
                                        dilation=2 ** i, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, channels, seq_len) → (batch, channels, seq_len)"""
        return self.network(x)


# ─── TradingTCN (1m, 3m용) ──────────────────────────────────────

class TradingTCN(nn.Module):
    """TCN 기반 거래 모델 — 단기 패턴 인식 (1m, 3m)

    아키텍처:
        Input (batch, seq_len, features)
        → RevIN normalize
        → Conv1d transpose → (batch, features, seq_len)
        → TCN (3 blocks, dilation=[1,2,4], channels=[128,128,128])
        → Attention Pooling → context (batch, 128)
        → Timeframe Embedding add
        → price_head, direction_head, confidence_head, position_head

    인터페이스: forward(x), forward_with_position(x), set_timeframe(tf)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.4,
        kernel_size: int = 3
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # Timeframe embedding
        self.TIMEFRAME_MAP = {'1m': 0, '3m': 1, '5m': 2, '15m': 3, '1h': 4, '4h': 5}
        self.tf_embedding = nn.Embedding(len(self.TIMEFRAME_MAP), hidden_size)
        self._timeframe_idx = 0

        # RevIN
        self.revin = RevIN(num_features=input_size, affine=True)

        # TCN: input_size channels → hidden_size channels
        num_channels = [hidden_size] * num_layers  # [128, 128, 128]
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # Attention Pooling (시퀀스 → 단일 벡터)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # 가격 예측 헤드
        self.price_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # 방향 헤드 (2-class: DOWN=0, UP=1)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

        # 신뢰도 헤드 (sigmoid: 0~1)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Soft Position 헤드 (tanh: -1~1)
        self.position_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # ─── SSL Masked Prediction 구조 ───
        self.mask_token = nn.Parameter(torch.randn(1, 1, input_size) * 0.02)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

        logger.info(
            f"TradingTCN 초기화: input={input_size}, hidden={hidden_size}, "
            f"blocks={num_layers}, kernel={kernel_size}, dilation=[{','.join(str(2**i) for i in range(num_layers))}]"
        )

    def set_timeframe(self, timeframe: str):
        self._timeframe_idx = self.TIMEFRAME_MAP.get(timeframe, 0)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone forward: RevIN → TCN → per-timestep hidden.

        Args:
            x: (batch, seq_len, features) — RevIN 적용 전 원본 또는 마스킹된 입력
        Returns:
            (batch, seq_len, hidden_size)
        """
        x = self.revin(x, mode='norm')
        x_conv = x.permute(0, 2, 1)        # (batch, features, seq_len)
        tcn_out = self.tcn(x_conv)          # (batch, hidden, seq_len)
        return tcn_out.permute(0, 2, 1)     # (batch, seq_len, hidden)

    def _compute_context(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        # 1~2. Backbone
        hidden = self._encode(x)  # (batch, seq_len, hidden)

        # 3. Attention Pooling
        attn_scores = self.attention(hidden)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(hidden * attn_weights, dim=1)  # (batch, hidden)

        # 4. Timeframe Embedding
        tf_idx = torch.tensor([self._timeframe_idx], device=x.device).expand(batch_size)
        context = context + self.tf_embedding(tf_idx)

        attn_expanded = attn_weights.squeeze(-1)  # (batch, seq_len)

        return context, attn_expanded

    def pretrain_forward(self, x: torch.Tensor, mask_ratio: float = 0.15):
        """SSL Masked Prediction forward.

        Args:
            x: (batch, seq_len, features) — 원본 feature
            mask_ratio: 마스킹 비율
        Returns:
            reconstruction: (batch, seq_len, features)
            mask: (batch, seq_len) bool — True=마스킹 위치
        """
        batch_size, seq_len, _ = x.shape
        mask = torch.rand(batch_size, seq_len, device=x.device) < mask_ratio
        mask[:, -1] = False  # 최신 timestep 보호

        x_masked = x.clone()
        x_masked[mask] = self.mask_token.squeeze(0).squeeze(0)

        hidden = self._encode(x_masked)
        reconstruction = self.reconstruction_head(hidden)
        return reconstruction, mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            price_pred: (batch, 1)
            direction_logits: (batch, 3)
            confidence: (batch, 1)
            attention_weights: (batch, seq_len)
        """
        context, attn_weights = self._compute_context(x)
        price_pred = self.price_head(context)
        direction_logits = self.direction_head(context)
        confidence = torch.sigmoid(self.confidence_head(context))
        return price_pred, direction_logits, confidence, attn_weights

    def forward_with_position(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            price_pred, direction_logits, confidence, attention_weights, soft_position
        """
        context, attn_weights = self._compute_context(x)
        price_pred = self.price_head(context)
        direction_logits = self.direction_head(context)
        confidence = torch.sigmoid(self.confidence_head(context))
        soft_position = torch.tanh(self.position_head(context))
        return price_pred, direction_logits, confidence, attn_weights, soft_position


# ─── TradingGRU (15m, 5m용) ─────────────────────────────────────

class TradingGRU(nn.Module):
    """GRU 기반 거래 모델 — 매크로 추세 추적 (15m, 5m)

    아키텍처:
        Input (batch, seq_len, features)
        → RevIN normalize
        → 2-layer Bidirectional GRU (hidden=64, output=128)
        → Dropout
        → Attention Pooling → context (batch, 128)
        → Timeframe Embedding add
        → price_head, direction_head, confidence_head, position_head

    인터페이스: forward(x), forward_with_position(x), set_timeframe(tf)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4
    ):
        super().__init__()

        self.hidden_size = hidden_size
        gru_hidden = hidden_size // 2  # BiGRU: 64 x 2 = 128

        # Timeframe embedding
        self.TIMEFRAME_MAP = {'1m': 0, '3m': 1, '5m': 2, '15m': 3, '1h': 4, '4h': 5}
        self.tf_embedding = nn.Embedding(len(self.TIMEFRAME_MAP), hidden_size)
        self._timeframe_idx = 3  # default: 15m

        # RevIN
        self.revin = RevIN(num_features=input_size, affine=True)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.gru_dropout = nn.Dropout(dropout)

        # Attention Pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # 가격 예측 헤드
        self.price_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # 방향 헤드 (2-class: DOWN=0, UP=1)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

        # 신뢰도 헤드
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Soft Position 헤드
        self.position_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # ─── SSL Masked Prediction 구조 ───
        self.mask_token = nn.Parameter(torch.randn(1, 1, input_size) * 0.02)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

        logger.info(
            f"TradingGRU 초기화: input={input_size}, hidden={hidden_size} (BiGRU {gru_hidden}x2), "
            f"layers={num_layers}"
        )

    def set_timeframe(self, timeframe: str):
        self._timeframe_idx = self.TIMEFRAME_MAP.get(timeframe, 0)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone forward: RevIN → BiGRU → per-timestep hidden."""
        x = self.revin(x, mode='norm')
        gru_out, _ = self.gru(x)          # (batch, seq_len, hidden_size)
        return self.gru_dropout(gru_out)

    def _compute_context(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        # 1~2. Backbone
        hidden = self._encode(x)

        # 3. Attention Pooling
        attn_scores = self.attention(hidden)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(hidden * attn_weights, dim=1)

        # 4. Timeframe Embedding
        tf_idx = torch.tensor([self._timeframe_idx], device=x.device).expand(batch_size)
        context = context + self.tf_embedding(tf_idx)

        attn_expanded = attn_weights.squeeze(-1)
        return context, attn_expanded

    def pretrain_forward(self, x: torch.Tensor, mask_ratio: float = 0.15):
        """SSL Masked Prediction forward."""
        batch_size, seq_len, _ = x.shape
        mask = torch.rand(batch_size, seq_len, device=x.device) < mask_ratio
        mask[:, -1] = False

        x_masked = x.clone()
        x_masked[mask] = self.mask_token.squeeze(0).squeeze(0)

        hidden = self._encode(x_masked)
        reconstruction = self.reconstruction_head(hidden)
        return reconstruction, mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            price_pred: (batch, 1)
            direction_logits: (batch, 3)
            confidence: (batch, 1)
            attention_weights: (batch, seq_len)
        """
        context, attn_weights = self._compute_context(x)
        price_pred = self.price_head(context)
        direction_logits = self.direction_head(context)
        confidence = torch.sigmoid(self.confidence_head(context))
        return price_pred, direction_logits, confidence, attn_weights

    def forward_with_position(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            price_pred, direction_logits, confidence, attention_weights, soft_position
        """
        context, attn_weights = self._compute_context(x)
        price_pred = self.price_head(context)
        direction_logits = self.direction_head(context)
        confidence = torch.sigmoid(self.confidence_head(context))
        soft_position = torch.tanh(self.position_head(context))
        return price_pred, direction_logits, confidence, attn_weights, soft_position


# ─── TickTCN (10s용) ─────────────────────────────────────────────

class TickTCN(nn.Module):
    """TCN 기반 틱 모델 — 10초 마이크로캔들 전용 경량 모델

    아키텍처:
        Input (batch, 30, 11)
        → RevIN normalize
        → Conv1d transpose
        → TCN (2 blocks, dilation=[1,2], channels=[64,64])
        → Attention Pooling → context (batch, 64)
        → timing_head, direction_head, confidence_head

    인터페이스: forward(x) → (timing, direction, confidence, attn_weights)
    """

    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        kernel_size: int = 3
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # RevIN
        self.revin = RevIN(num_features=input_size, affine=True)

        # TCN
        num_channels = [hidden_size] * num_layers  # [64, 64]
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # Attention Pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # 타이밍 신뢰도 헤드 (sigmoid: 0~1)
        self.timing_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 방향 헤드 (2-class: sigmoid → 1.0=UP, 0.0=DOWN)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 신뢰도 헤드 (sigmoid: 0~1)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # ─── SSL Masked Prediction 구조 ───
        self.mask_token = nn.Parameter(torch.randn(1, 1, input_size) * 0.02)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, input_size)
        )

        logger.info(
            f"TickTCN 초기화: input={input_size}, hidden={hidden_size}, "
            f"blocks={num_layers}, kernel={kernel_size}"
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone forward: RevIN → TCN → per-timestep hidden."""
        x = self.revin(x, mode='norm')
        x_conv = x.permute(0, 2, 1)
        tcn_out = self.tcn(x_conv)
        return tcn_out.permute(0, 2, 1)  # (batch, seq_len, hidden)

    def pretrain_forward(self, x: torch.Tensor, mask_ratio: float = 0.15):
        """SSL Masked Prediction forward."""
        batch_size, seq_len, _ = x.shape
        mask = torch.rand(batch_size, seq_len, device=x.device) < mask_ratio
        mask[:, -1] = False

        x_masked = x.clone()
        x_masked[mask] = self.mask_token.squeeze(0).squeeze(0)

        hidden = self._encode(x_masked)
        reconstruction = self.reconstruction_head(hidden)
        return reconstruction, mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len=30, features=11)
        Returns:
            timing: (batch, 1)
            direction: (batch, 1)
            confidence: (batch, 1)
            attn_weights: (batch, seq_len)
        """
        # 1~2. Backbone
        hidden = self._encode(x)

        # 3. Attention Pooling
        attn_scores = self.attention(hidden)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(hidden * attn_weights, dim=1)

        # 4. Output heads
        timing = self.timing_head(context)
        direction = self.direction_head(context)
        confidence = self.confidence_head(context)

        attn_expanded = attn_weights.squeeze(-1)
        return timing, direction, confidence, attn_expanded


# ─── MoE 컴포넌트 (Global Experts) ──────────────────────────────

class LoRAExpert(nn.Module):
    """Low-Rank Adapter — 국면 전문가 1개.

    A(dim→rank) + B(rank→dim) 구조로 가벼운 residual delta 생성.
    초기값: B=0 → delta=0 (학습 전 원본 유지).
    Bounded output: tanh * max_delta로 base 표현 변경 폭 제한.
    """

    def __init__(self, dim: int, rank: int = 8, max_delta: float = 0.5):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.max_delta = max_delta
        nn.init.kaiming_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.up(self.down(x))
        return torch.tanh(raw) * self.max_delta



# ─── 멀티-스케일 통합 모델 ──────────────────────────────────────

class UnifiedTradingModel(nn.Module):
    """Multi-Scale TCN + GRU Temporal Pool 통합 트레이딩 모델.

    3개의 눈(Eye Encoder) + 깊은 공유 몸통(Shared TCN Backbone) + GRU Pool + Heads.

    아키텍처:
        Micro-Eye (1m, 30bars):  RevIN → Conv1d(11→64) → BN → ReLU
        Mid-Eye   (3m, 40bars):  RevIN → Conv1d(11→64) → BN → ReLU → AdaptivePool(30)
        Macro-Eye (15m, 60bars): RevIN → Conv1d(11→64) → BN → ReLU → AdaptivePool(30)
        → Channel Concat (batch, 192, 30)
        → Shared TCN (4 blocks, dilation=[1,2,4,8]) → (batch, 128, 30)
        → GRU Temporal Pool (128→64) → last hidden (batch, 64)
        → LoRA Adapter (64→rank→64) → residual
        → Macro Direction Gate (15m trend bias → direction logits 보정)
        → Heads: direction(3), price(1), confidence(1), position(1), timing(1)
    """

    # GRU 출력 차원 (heads 입력)
    GRU_HIDDEN = 64

    def __init__(
        self,
        macro_features: int = 11,
        mid_features: int = 11,
        micro_features: int = 11,
        hidden_size: int = 128,
        eye_dim: int = 64,
        num_experts: int = 3,       # unused, kept for checkpoint compat
        lora_rank: int = 4,
        dropout: float = 0.25,
        common_seq_len: int = 30,
        entropy_coeff: float = 0.02  # unused, kept for checkpoint compat
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eye_dim = eye_dim
        self.common_seq_len = common_seq_len
        gru_h = self.GRU_HIDDEN

        # ── Micro-Eye (1m) ──
        self.micro_revin = RevIN(micro_features)
        self.micro_eye = nn.Sequential(
            nn.Conv1d(micro_features, eye_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(eye_dim),
            nn.ReLU()
        )

        # ── Mid-Eye (3m) ──
        self.mid_revin = RevIN(mid_features)
        self.mid_eye = nn.Sequential(
            nn.Conv1d(mid_features, eye_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(eye_dim),
            nn.ReLU()
        )
        self.mid_pool = nn.AdaptiveAvgPool1d(common_seq_len)

        # ── Macro-Eye (15m) ──
        self.macro_revin = RevIN(macro_features)
        self.macro_eye = nn.Sequential(
            nn.Conv1d(macro_features, eye_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(eye_dim),
            nn.ReLU()
        )
        self.macro_pool = nn.AdaptiveAvgPool1d(common_seq_len)

        # ── Shared TCN Backbone (4 Dilated Blocks) ──
        backbone_input = eye_dim * 3  # 192
        self.backbone = TemporalConvNet(
            num_inputs=backbone_input,
            num_channels=[hidden_size, hidden_size, hidden_size, hidden_size],
            kernel_size=3,
            dropout=dropout
        )

        # ── Attention Temporal Pool (GRU 대체) ──
        # GRU: backbone(3.3) → gru(0.1) → 97% variance 붕괴 확인됨
        # Attention Pool: 중요 timestep에 가중치 집중 → variance 보존
        self.temporal_attn = nn.Linear(hidden_size, 1)  # (B, S, 128) → (B, S, 1)
        self.temporal_proj = nn.Sequential(
            nn.Linear(hidden_size, gru_h),  # 128 → 64
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── LoRA Adapter (온라인 학습용, MoE 전문가 대체) ──
        self.online_lora = LoRAExpert(gru_h, lora_rank)

        # ── Macro Direction Gate (15m 추세 → 방향 편향) ──
        self.macro_direction_bias = nn.Linear(eye_dim, 2)
        self.macro_gate_scale = 0.1

        # ── LayerNorm 제거: variance 마스킹 방지 ──
        # 기존 LayerNorm은 gru_out=0.1을 1.0으로 "가짜 복구"해서 문제를 은폐
        # Attention Pool은 variance를 보존하므로 LayerNorm 불필요

        # ── Prediction Heads (64-dim 입력) ──
        self.direction_head = nn.Sequential(
            nn.Linear(gru_h, 32), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(32, 2)
        )
        self.price_head = nn.Sequential(
            nn.Linear(gru_h, 32), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(32, 1)
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(gru_h, 32), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(32, 1)
        )
        self.position_head = nn.Sequential(
            nn.Linear(gru_h, 32), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(32, 1)
        )
        self.timing_head = nn.Sequential(
            nn.Linear(gru_h, 32), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(32, 1)
        )

        # ── SSL Reconstruction (인코더별) ──
        self.micro_mask_token = nn.Parameter(torch.randn(1, 1, micro_features) * 0.02)
        self.micro_recon_head = nn.Sequential(
            nn.Linear(eye_dim, 32), nn.ReLU(), nn.Linear(32, micro_features)
        )
        self.mid_mask_token = nn.Parameter(torch.randn(1, 1, mid_features) * 0.02)
        self.mid_recon_head = nn.Sequential(
            nn.Linear(eye_dim, 32), nn.ReLU(), nn.Linear(32, mid_features)
        )
        self.macro_mask_token = nn.Parameter(torch.randn(1, 1, macro_features) * 0.02)
        self.macro_recon_head = nn.Sequential(
            nn.Linear(eye_dim, 32), nn.ReLU(), nn.Linear(32, macro_features)
        )

        # Backbone-level SSL reconstruction
        self.backbone_recon_head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Linear(64, backbone_input)  # 128 → 192
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"UnifiedTradingModel 초기화: eyes={eye_dim}×3, backbone={hidden_size}×4, "
            f"AttnPool={hidden_size}→{gru_h}, LoRA(r={lora_rank}), "
            f"params={total_params:,} (trainable={trainable:,})"
        )

    # ── Eye Encoders ──

    def _eye_micro(self, x: torch.Tensor) -> torch.Tensor:
        """1m: Conv1d → (batch, 64, 30)
        RevIN 비활성화: EWMA Z-score가 이미 정규화 수행, RevIN 중복은 regime 신호 소멸.
        """
        return self.micro_eye(x.permute(0, 2, 1))

    def _eye_mid(self, x: torch.Tensor) -> torch.Tensor:
        """3m: Conv1d → Pool → (batch, 64, 30)"""
        h = self.mid_eye(x.permute(0, 2, 1))
        return self.mid_pool(h)

    def _eye_macro(self, x: torch.Tensor) -> torch.Tensor:
        """15m: Conv1d → Pool → (batch, 64, 30)"""
        h = self.macro_eye(x.permute(0, 2, 1))
        return self.macro_pool(h)

    # ── Core Forward ──

    _repr_diag_counter = 0

    def _encode(self, macro_x: torch.Tensor, mid_x: torch.Tensor,
                micro_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """3개 눈 → Channel Concat → Shared Backbone → hidden."""
        micro_h = self._eye_micro(micro_x)   # (B, 64, 30)
        mid_h = self._eye_mid(mid_x)          # (B, 64, 30)
        macro_h = self._eye_macro(macro_x)    # (B, 64, 30)

        combined = torch.cat([micro_h, mid_h, macro_h], dim=1)  # (B, 192, 30)
        backbone_out = self.backbone(combined)  # (B, 128, 30)

        # ── Representation Variance 진단 (1000회마다) ──
        UnifiedTradingModel._repr_diag_counter += 1
        if UnifiedTradingModel._repr_diag_counter % 1000 == 1:
            with torch.no_grad():
                _var = lambda t: t.var().item()
                logger.info(
                    f"[REPR] input_var: micro={_var(micro_x):.4f} mid={_var(mid_x):.4f} macro={_var(macro_x):.4f} | "
                    f"eye_var: micro={_var(micro_h):.4f} mid={_var(mid_h):.4f} macro={_var(macro_h):.4f} | "
                    f"backbone={_var(backbone_out):.4f}"
                )

        return backbone_out.permute(0, 2, 1), macro_h    # (B, 30, 128), (B, 64, 30)

    def _temporal_pool(self, hidden: torch.Tensor) -> torch.Tensor:
        """Attention Temporal Pooling — 중요 timestep 가중 합산.

        GRU는 backbone variance 3.3→0.1로 97% 붕괴.
        Attention: 학습된 가중치로 중요 timestep 선택 → variance 보존.

        Args:
            hidden: (batch, seq_len, hidden_size) — backbone output (B, 30, 128)
        Returns:
            context: (batch, gru_hidden) — attention-pooled hidden (B, 64)
        """
        # Attention scores
        scores = self.temporal_attn(hidden)           # (B, S, 1)
        weights = F.softmax(scores, dim=1)            # (B, S, 1) — timestep별 중요도
        pooled = (hidden * weights).sum(dim=1)        # (B, 128) — 가중합
        return self.temporal_proj(pooled)             # (B, 64) — projection

    def _apply_macro_gate(
        self, direction_logits: torch.Tensor, macro_h: torch.Tensor
    ) -> torch.Tensor:
        """15m 추세 기반 방향 편향 적용.

        매크로 눈의 마지막 타임스텝에서 방향 bias를 추출하여
        direction logits에 가산 → "큰 흐름을 거스르지 않기" 내재화.

        Args:
            direction_logits: (B, 3) — raw direction logits
            macro_h: (B, 64, 30) — 이미 계산된 macro eye 출력 (_encode에서 재사용)
        """
        macro_last = macro_h[:, :, -1]            # (B, 64) — 가장 최근 시점
        raw_bias = self.macro_direction_bias(macro_last)  # (B, 3)
        # 크기 제한: tanh로 [-1,1] 바운딩 후 스케일링 → 방향 붕괴 방지
        bias = torch.tanh(raw_bias) * self.macro_gate_scale
        return direction_logits + bias

    def forward(
        self, macro_x: torch.Tensor, mid_x: torch.Tensor, micro_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            macro_x: (batch, 60, features) — 15m
            mid_x:   (batch, 40, features) — 3m
            micro_x: (batch, 30, features) — 1m
        Returns:
            price_pred: (batch, 1)
            direction_logits: (batch, 3)
            confidence: (batch, 1)
            macro_gate_bias: (batch, 3) — macro gate 적용된 logit shift
            aux_loss: scalar (항상 0, 하위 호환)
        """
        hidden, macro_h = self._encode(macro_x, mid_x, micro_x)
        context = self._temporal_pool(hidden)
        # LoRA 비활성화: SFT 베이스 모델 성능 먼저 확보
        # context = context + self.online_lora(context)
        h = context

        raw_dir = self.direction_head(h)
        direction_logits = self._apply_macro_gate(raw_dir, macro_h)

        return (
            self.price_head(h),
            direction_logits,
            torch.sigmoid(self.confidence_head(h)),
            torch.zeros(macro_x.shape[0], self.common_seq_len, device=macro_x.device),  # placeholder
            torch.tensor(0.0, device=macro_x.device),  # no aux_loss
        )

    def forward_with_position(
        self, macro_x: torch.Tensor, mid_x: torch.Tensor, micro_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            price_pred, direction_logits, confidence, placeholder,
            soft_position, timing, aux_loss (항상 0)
        """
        hidden, macro_h = self._encode(macro_x, mid_x, micro_x)
        context = self._temporal_pool(hidden)
        # LoRA 비활성화: SFT 베이스 모델 성능 먼저 확보
        # context = context + self.online_lora(context)
        h = context

        raw_dir = self.direction_head(h)
        direction_logits = self._apply_macro_gate(raw_dir, macro_h)

        # ── GRU/Head Variance 진단 (1000회마다, _encode와 동기) ──
        if UnifiedTradingModel._repr_diag_counter % 1000 == 1:
            with torch.no_grad():
                _var = lambda t: t.var().item()
                logger.info(
                    f"[REPR] attn_pool={_var(context):.4f} head_in(h)={_var(h):.4f} "
                    f"dir_logits={_var(direction_logits):.4f}"
                )

        # position_head: h.detach()로 PnL gradient가 shared h를 오염시키지 않도록 격리
        return (
            self.price_head(h),
            direction_logits,
            torch.sigmoid(self.confidence_head(h)),
            torch.zeros(macro_x.shape[0], self.common_seq_len, device=macro_x.device),
            torch.tanh(self.position_head(h.detach())),
            torch.sigmoid(self.timing_head(h)),
            torch.tensor(0.0, device=macro_x.device),
        )

    # ── SSL Pretrain (인코더별) ──

    def pretrain_forward_micro(self, x: torch.Tensor, mask_ratio: float = 0.15):
        """Micro-Eye SSL: 마스킹 → Conv1d → 복원."""
        B, S, _ = x.shape
        mask = torch.rand(B, S, device=x.device) < mask_ratio
        mask[:, -1] = False
        x_m = x.clone()
        x_m[mask] = self.micro_mask_token.squeeze(0).squeeze(0)
        h = self.micro_eye(self.micro_revin(x_m, mode='norm').permute(0, 2, 1))
        recon = self.micro_recon_head(h.permute(0, 2, 1))  # (B, S, features)
        return recon, mask

    def pretrain_forward_mid(self, x: torch.Tensor, mask_ratio: float = 0.15):
        """Mid-Eye SSL: 마스킹 → Conv1d → 복원."""
        B, S, _ = x.shape
        mask = torch.rand(B, S, device=x.device) < mask_ratio
        mask[:, -1] = False
        x_m = x.clone()
        x_m[mask] = self.mid_mask_token.squeeze(0).squeeze(0)
        h = self.mid_eye(self.mid_revin(x_m, mode='norm').permute(0, 2, 1))
        recon = self.mid_recon_head(h.permute(0, 2, 1))  # (B, S, features)
        return recon, mask

    def pretrain_forward_macro(self, x: torch.Tensor, mask_ratio: float = 0.15):
        """Macro-Eye SSL: 마스킹 → Conv1d → 복원."""
        B, S, _ = x.shape
        mask = torch.rand(B, S, device=x.device) < mask_ratio
        mask[:, -1] = False
        x_m = x.clone()
        x_m[mask] = self.macro_mask_token.squeeze(0).squeeze(0)
        h = self.macro_eye(self.macro_revin(x_m, mode='norm').permute(0, 2, 1))
        recon = self.macro_recon_head(h.permute(0, 2, 1))  # (B, S, features)
        return recon, mask

    def pretrain_forward_backbone(
        self, macro_x: torch.Tensor, mid_x: torch.Tensor, micro_x: torch.Tensor,
        mask_ratio: float = 0.15
    ):
        """Backbone SSL: 3-input 통합 마스킹 → backbone → 복원.

        각 Eye의 출력(192-dim)을 backbone이 처리 후 복원.
        Returns:
            recon: (batch, common_seq_len, 192)
            mask: (batch, common_seq_len)
        """
        # Eyes (마스킹 없이 정상 인코딩)
        micro_h = self._eye_micro(micro_x)   # (B, 64, 30)
        mid_h = self._eye_mid(mid_x)          # (B, 64, 30)
        macro_h = self._eye_macro(macro_x)    # (B, 64, 30)
        combined = torch.cat([micro_h, mid_h, macro_h], dim=1)  # (B, 192, 30)

        # Backbone 입력에 마스킹 적용
        B = combined.shape[0]
        S = combined.shape[2]  # 30
        mask = torch.rand(B, S, device=combined.device) < mask_ratio
        mask[:, -1] = False

        combined_masked = combined.clone()
        # 마스킹: 해당 timestep의 모든 192 채널을 0으로
        mask_expanded = mask.unsqueeze(1).expand_as(combined_masked)  # (B, 192, 30)
        combined_masked[mask_expanded] = 0.0

        backbone_out = self.backbone(combined_masked)  # (B, 128, 30)
        hidden = backbone_out.permute(0, 2, 1)          # (B, 30, 128)
        recon = self.backbone_recon_head(hidden)         # (B, 30, 192)

        # Target: 원본 combined (B, 192, 30) → permute → (B, 30, 192)
        target = combined.permute(0, 2, 1).detach()
        return recon, mask, target

    # ── Parameter Groups ──

    def eye_params(self) -> list:
        """3개 Eye 인코더 파라미터."""
        return (list(self.micro_revin.parameters()) +
                list(self.micro_eye.parameters()) +
                list(self.mid_revin.parameters()) +
                list(self.mid_eye.parameters()) +
                list(self.mid_pool.parameters()) +
                list(self.macro_revin.parameters()) +
                list(self.macro_eye.parameters()) +
                list(self.macro_pool.parameters()))

    def backbone_params(self) -> list:
        """공유 TCN Backbone 파라미터."""
        return list(self.backbone.parameters())

    def encoder_params(self) -> list:
        """동결 대상: Eyes + Backbone + Attention Pool."""
        return (self.eye_params() + self.backbone_params() +
                list(self.temporal_attn.parameters()) +
                list(self.temporal_proj.parameters()))

    def head_params(self) -> list:
        """예측 Head 파라미터 (Macro Gate 포함)."""
        return (list(self.macro_direction_bias.parameters()) +
                list(self.direction_head.parameters()) +
                list(self.price_head.parameters()) +
                list(self.confidence_head.parameters()) +
                list(self.position_head.parameters()) +
                list(self.timing_head.parameters()))

    def trainable_params(self) -> list:
        """SFT Phase 2 학습 대상: LoRA + 전체 Heads."""
        return list(self.online_lora.parameters()) + self.head_params()

    def online_trainable_params(self) -> list:
        """온라인 학습 대상: LoRA만 (512 params).

        direction_head 포함 모든 head를 동결하여 SFT 패턴 매칭 보존.
        LoRA는 hidden representation만 미세 조정 → head가 더 정확한 입력을 받음.
        """
        return list(self.online_lora.parameters())
