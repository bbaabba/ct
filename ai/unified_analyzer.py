"""통합 AI 분석기 — Multi-Scale TCN + Global MoE

3-Eye (Macro/Mid/Micro) 통합 모델 기반 시장 분석.
기존 MarketAnalyzer + TickAnalyzer + MTF + Hybrid를 단일 모델로 통합.
"""
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from pathlib import Path

from utils.logger import setup_logger
from ai.market_analyzer import (
    UnifiedTradingModel, CostSensitiveFocalLoss, PnLLoss,
    DIR_MAP, DIR_MAP_INV
)
from ai.shadow_validator import ShadowValidator

logger = setup_logger(__name__)


# ─── 데이터 구조 ────────────────────────────────────────────────

@dataclass
class UnifiedPrediction:
    """통합 모델 예측 결과."""
    predicted_price: float
    predicted_direction: str        # 'UP', 'DOWN', 'NEUTRAL'
    direction_confidence: float     # 0~1
    price_change_pct: float
    timing_confidence: float        # 0~1 진입 타이밍 품질
    soft_position: float            # -1~1
    direction_probs: Dict[str, float]  # {'short': p, 'hold': p, 'long': p}
    timestamp: datetime


@dataclass
class UnifiedTrainingSample:
    """통합 모델 학습 샘플."""
    macro_features: np.ndarray    # (60, features) — 15m
    mid_features: np.ndarray      # (40, features) — 3m
    micro_features: np.ndarray    # (30, features) — 1m
    actual_direction: str         # 'UP', 'DOWN', 'NEUTRAL'
    actual_price_change: float
    timestamp: datetime
    sample_weight: float = 1.0
    future_log_return: float = 0.0
    actual_timing: float = 0.5    # 배리어 도달 속도 (0~1)


# ─── UnifiedAnalyzer ─────────────────────────────────────────────

class UnifiedAnalyzer:
    """Multi-Scale TCN + GRU Temporal Pool 통합 분석기.

    기존 MarketAnalyzer + TickAnalyzer + MultiTimeframeAnalyzer + HybridAnalyzer를
    하나의 UnifiedTradingModel로 통합 운영.
    """

    def __init__(
        self,
        symbol: str = 'ETHUSDT',
        lora_rank: int = 4,
        device: str = 'auto',
        macro_seq_len: int = 60,
        mid_seq_len: int = 40,
        micro_seq_len: int = 30,
        # backward compat kwargs (ignored)
        num_experts: int = 3,
        entropy_coeff: float = 0.02,
    ):
        self.symbol = symbol
        self.macro_seq_len = macro_seq_len
        self.mid_seq_len = mid_seq_len
        self.micro_seq_len = micro_seq_len

        # Device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Feature builder
        from ai.feature_builder import TFFeatureBuilder
        self.feature_builder = TFFeatureBuilder()

        # Feature dimensions
        self.macro_features = self.feature_builder.get_input_dim('15m')
        self.mid_features = self.feature_builder.get_input_dim('3m')
        self.micro_features = self.feature_builder.get_input_dim('1m')

        # Model
        self.model = UnifiedTradingModel(
            macro_features=self.macro_features,
            mid_features=self.mid_features,
            micro_features=self.micro_features,
            lora_rank=lora_rank,
        ).to(self.device)

        # Training state
        # SFT: 6개월 3m 데이터 ≈ 12,000샘플 전부 수용
        # 온라인: 최근 데이터 위주로 자연 순환
        self.training_buffer: deque = deque(maxlen=15000)
        self.replay_buffer: deque = deque(maxlen=2000)
        self._replay_priorities: deque = deque(maxlen=2000)
        self._per_alpha: float = 0.6
        self._per_beta: float = 0.4
        self._replay_mix_ratio: float = 0.2

        # Anchor Memory — SFT 대표 샘플 보존 (catastrophic forgetting 방지)
        self._anchor_buffer: List[dict] = []  # [{macro, mid, micro, dir, price, log_ret, timing, weight}]
        self._anchor_max_size: int = 500
        self._anchor_mix_ratio: float = 0.3  # 온라인 학습 시 anchor 30% 혼합

        self.batch_size: int = 32
        self.training_epochs: int = 5
        self._training_count: int = 0
        self.is_pretrained: bool = False
        self._ssl_pretrained: bool = False

        # Checkpoint + Rollback
        self._checkpoint_state: Optional[dict] = None
        self._best_val_loss: float = float('inf')
        self._consecutive_degrades: int = 0
        self._max_degrades_before_rollback: int = 2   # 2연속 악화 시 즉시 롤백
        self._val_loss_tolerance: float = 0.01  # 1% 이내면 개선으로 인정 (엄격)

        # Timing
        self._last_train_time: Optional[datetime] = None

        # Shadow Validator (콜드스타트 보호)
        self.shadow_validator = ShadowValidator()

        logger.info(
            f"UnifiedAnalyzer 초기화: {symbol}, device={self.device}, "
            f"GRU pool, LoRA(r={lora_rank})"
        )

    # ─── Predict ──────────────────────────────────────────────

    def predict(
        self, df_15m: pd.DataFrame, df_3m: pd.DataFrame, df_1m: pd.DataFrame,
        sim_time: Optional[float] = None,
    ) -> Optional[UnifiedPrediction]:
        """3-시간대 입력 → 통합 예측.

        Args:
            df_15m: 15m OHLCV (최소 macro_seq_len+20 행)
            df_3m: 3m OHLCV
            df_1m: 1m OHLCV (micro eye)
            sim_time: 백테스트 시 캔들 epoch seconds (None이면 실시간)
        """
        try:
            # Feature extraction
            macro_feat = self.feature_builder.prepare_features(df_15m, '15m')
            mid_feat = self.feature_builder.prepare_features(df_3m, '3m')
            micro_feat = self.feature_builder.prepare_features(df_1m, '1m')

            if (len(macro_feat) < self.macro_seq_len or
                    len(mid_feat) < self.mid_seq_len or
                    len(micro_feat) < self.micro_seq_len):
                return None

            # Slice to sequence length
            macro_seq = macro_feat[-self.macro_seq_len:]
            mid_seq = mid_feat[-self.mid_seq_len:]
            micro_seq = micro_feat[-self.micro_seq_len:]

            # Tensor
            macro_x = torch.FloatTensor(macro_seq).unsqueeze(0).to(self.device)
            mid_x = torch.FloatTensor(mid_seq).unsqueeze(0).to(self.device)
            micro_x = torch.FloatTensor(micro_seq).unsqueeze(0).to(self.device)

            # Forward
            self.model.eval()
            with torch.no_grad():
                price_pred, dir_logits, conf, attn, soft_pos, timing, _ = \
                    self.model.forward_with_position(macro_x, mid_x, micro_x)

            # Post-process
            price_val = price_pred.item()
            conf_val = conf.item()
            timing_val = timing.item()
            soft_pos_val = soft_pos.item()

            # Direction — logit bias 보정 + temperature scaling
            # 1) Running mean으로 학습된 class bias 제거 (LONG bias 교정)
            #    warmup 후 평균 logit을 빼서 중립화 → 시장 적응형
            if not hasattr(self, '_logit_ema'):
                self._logit_ema = dir_logits[0].detach().clone()
                self._logit_ema_count = 0
            self._logit_ema_count += 1
            _ema_alpha = min(0.05, 2.0 / (self._logit_ema_count + 1))
            self._logit_ema = (1 - _ema_alpha) * self._logit_ema + _ema_alpha * dir_logits[0].detach()
            # warmup 50회 후부터 보정 적용
            if self._logit_ema_count >= 50:
                dir_logits = dir_logits - self._logit_ema.unsqueeze(0)

            # 2) Temperature scaling — 0.7: 편향 증폭 완화
            _INFERENCE_TEMPERATURE = 0.7
            dir_probs = F.softmax(dir_logits / _INFERENCE_TEMPERATURE, dim=-1)[0]
            p_down, p_up = dir_probs.tolist()  # 2-class: DOWN=0, UP=1

            # margin 기반 방향 결정 (|p_up - p_down| < margin → HOLD)
            _DIR_MARGIN = 0.10  # 2-class: 0.5±0.10 밖이어야 방향 판정
            if p_up - p_down >= _DIR_MARGIN:
                pred_dir = 'UP'
            elif p_down - p_up >= _DIR_MARGIN:
                pred_dir = 'DOWN'
            else:
                pred_dir = 'NEUTRAL'  # 추론에서만 HOLD (학습엔 없음)

            direction_confidence = max(p_down, p_up)

            # Price change estimate
            current_price = df_3m['close'].iloc[-1] if len(df_3m) > 0 else 0
            price_change_pct = price_val * 100  # normalized → %

            result = UnifiedPrediction(
                predicted_price=current_price * (1 + price_change_pct / 100),
                predicted_direction=pred_dir,
                direction_confidence=direction_confidence,
                price_change_pct=price_change_pct,
                timing_confidence=timing_val,
                soft_position=soft_pos_val,
                direction_probs={'down': p_down, 'up': p_up},
                timestamp=datetime.now()
            )

            # Shadow validator에 가상 예측 기록 (ATR 포함 → 가변 슬리피지 추정)
            if pred_dir != 'NEUTRAL' and current_price > 0:
                sv_dir = 'LONG' if pred_dir == 'UP' else 'SHORT'
                _sv_atr = 0.0
                if len(df_3m) >= 5:
                    _sv_atr = float((df_3m['high'] - df_3m['low']).tail(14).mean())
                self.shadow_validator.record_prediction(
                    sv_dir, direction_confidence, current_price,
                    atr=_sv_atr, sim_time=sim_time,
                )

            return result
        except Exception as e:
            logger.error(f"UnifiedAnalyzer predict 실패: {e}")
            return None

    # ─── Training Sample ──────────────────────────────────────

    def add_training_sample(self, sample: UnifiedTrainingSample):
        """학습 버퍼에 샘플 추가."""
        if sample.actual_direction not in DIR_MAP:
            return
        self.training_buffer.append(sample)

        # Replay buffer (극단적 움직임 우선 저장)
        abs_change = abs(sample.actual_price_change)
        if abs_change > 0.001 or len(self.replay_buffer) < 100:
            self.replay_buffer.append(sample)
            self._replay_priorities.append(abs_change + 1e-4)

    # ─── SSL Pretrain ─────────────────────────────────────────

    def ssl_pretrain(
        self, features_dict: Dict[str, list],
        epochs: int = 10, mask_ratio: float = 0.15, lr: float = 5e-4
    ) -> dict:
        """Self-Supervised Masked Prediction 사전학습.

        Args:
            features_dict: {'macro': [array(60,f)...], 'mid': [array(40,f)...], 'micro': [array(30,f)...]}
            epochs: SSL 학습 에폭
            mask_ratio: 마스킹 비율
            lr: 학습률
        Returns:
            {'status': str, 'losses': dict}
        """
        results = {}

        # Phase 1: 각 Eye 인코더 개별 SSL
        eye_configs = [
            ('micro', features_dict.get('micro', []),
             self.model.pretrain_forward_micro, self._get_micro_ssl_params()),
            ('mid', features_dict.get('mid', []),
             self.model.pretrain_forward_mid, self._get_mid_ssl_params()),
            ('macro', features_dict.get('macro', []),
             self.model.pretrain_forward_macro, self._get_macro_ssl_params()),
        ]

        for name, features, pretrain_fn, ssl_params in eye_configs:
            if not features or len(features) < 10:
                logger.warning(f"  {name}: 데이터 부족 ({len(features)}개), 건너뜀")
                continue

            X = np.array(features)
            X_t = torch.FloatTensor(X).to(self.device)
            dataset = TensorDataset(X_t)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            optimizer = torch.optim.AdamW(ssl_params, lr=lr, weight_decay=0.01)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

            self.model.train()
            final_loss = 0
            for epoch in range(epochs):
                epoch_loss = 0
                for (batch_x,) in loader:
                    optimizer.zero_grad()
                    recon, mask = pretrain_fn(batch_x, mask_ratio)
                    loss = F.mse_loss(recon[mask], batch_x[mask])
                    loss.backward()
                    nn.utils.clip_grad_norm_(ssl_params, max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                scheduler.step()
                final_loss = epoch_loss / max(len(loader), 1)
                logger.info(f"  SSL {name} Epoch {epoch+1}/{epochs}: loss={final_loss:.6f}")

            results[name] = {'status': 'completed', 'final_loss': final_loss, 'samples': len(X)}

        # Phase 2: Backbone 통합 SSL (3-input 동시 마스킹)
        macro_list = features_dict.get('macro', [])
        mid_list = features_dict.get('mid', [])
        micro_list = features_dict.get('micro', [])

        if macro_list and mid_list and micro_list:
            n_samples = min(len(macro_list), len(mid_list), len(micro_list))
            if n_samples >= 10:
                logger.info(f"  Backbone SSL: {n_samples}개 샘플")

                macro_t = torch.FloatTensor(np.array(macro_list[:n_samples])).to(self.device)
                mid_t = torch.FloatTensor(np.array(mid_list[:n_samples])).to(self.device)
                micro_t = torch.FloatTensor(np.array(micro_list[:n_samples])).to(self.device)

                dataset = TensorDataset(macro_t, mid_t, micro_t)
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

                backbone_params = (list(self.model.backbone.parameters()) +
                                   list(self.model.backbone_recon_head.parameters()))
                optimizer = torch.optim.AdamW(backbone_params, lr=lr * 0.5, weight_decay=0.01)
                scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

                self.model.train()
                for epoch in range(epochs):
                    epoch_loss = 0
                    for batch_macro, batch_mid, batch_micro in loader:
                        optimizer.zero_grad()
                        recon, mask, target = self.model.pretrain_forward_backbone(
                            batch_macro, batch_mid, batch_micro, mask_ratio
                        )
                        # mask: (B, 30), recon: (B, 30, 192), target: (B, 30, 192)
                        loss = F.mse_loss(recon[mask], target[mask])
                        loss.backward()
                        nn.utils.clip_grad_norm_(backbone_params, max_norm=1.0)
                        optimizer.step()
                        epoch_loss += loss.item()
                    scheduler.step()
                    avg_loss = epoch_loss / max(len(loader), 1)
                    logger.info(f"  SSL backbone Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

                results['backbone'] = {'status': 'completed', 'final_loss': avg_loss, 'samples': n_samples}

        self._ssl_pretrained = True
        self.model.eval()
        logger.info("✅ SSL Pre-training 완료")
        return {'status': 'completed', 'results': results}

    def _get_micro_ssl_params(self):
        return (list(self.model.micro_revin.parameters()) +
                list(self.model.micro_eye.parameters()) +
                list(self.model.micro_recon_head.parameters()) +
                [self.model.micro_mask_token])

    def _get_mid_ssl_params(self):
        return (list(self.model.mid_revin.parameters()) +
                list(self.model.mid_eye.parameters()) +
                list(self.model.mid_recon_head.parameters()) +
                [self.model.mid_mask_token])

    def _get_macro_ssl_params(self):
        return (list(self.model.macro_revin.parameters()) +
                list(self.model.macro_eye.parameters()) +
                list(self.model.macro_recon_head.parameters()) +
                [self.model.macro_mask_token])

    # ─── Batch Train ──────────────────────────────────────────

    def batch_train(
        self, epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        is_pretrain: bool = False,
        freeze_encoders: bool = True
    ) -> Dict[str, float]:
        """LoRA + Heads 학습.

        Args:
            epochs: 학습 에폭 (기본 5)
            batch_size: 배치 크기
            is_pretrain: True면 롤백 비활성화
            freeze_encoders: True면 Eyes+Backbone 동결 (온라인 모드)
        """
        epochs = epochs or self.training_epochs
        batch_size = batch_size or self.batch_size

        # 학습 데이터 수집
        samples = [s for s in self.training_buffer
                   if s.actual_direction in DIR_MAP]
        if len(samples) < 10:
            return {'status': 'insufficient_data', 'count': len(samples)}

        # Checkpoint
        self._checkpoint_state = copy.deepcopy(self.model.state_dict())

        # Extract features
        macro_list, mid_list, micro_list = [], [], []
        y_dir_list, y_price_list, weight_list, log_ret_list, y_timing_list = [], [], [], [], []

        for s in samples:
            macro_list.append(s.macro_features)
            mid_list.append(s.mid_features)
            micro_list.append(s.micro_features)
            y_dir_list.append(DIR_MAP[s.actual_direction])
            y_price_list.append(s.actual_price_change)
            weight_list.append(s.sample_weight)
            log_ret_list.append(s.future_log_return)
            y_timing_list.append(s.actual_timing)

        # PER (Prioritized Experience Replay)
        _per_indices = []
        if len(self.replay_buffer) >= 10:
            n_replay = max(1, int(len(samples) * self._replay_mix_ratio))
            priorities = np.array(list(self._replay_priorities), dtype=np.float64)
            valid_mask = priorities > 0
            if valid_mask.sum() >= n_replay:
                valid_pri = priorities[valid_mask]
                probs = valid_pri ** self._per_alpha
                probs /= probs.sum()
                valid_indices = np.where(valid_mask)[0]
                chosen = np.random.choice(len(valid_indices), size=n_replay, p=probs, replace=False)
                _per_indices = valid_indices[chosen].tolist()
                is_weights = (len(valid_indices) * probs[chosen]) ** (-self._per_beta)
                is_weights /= is_weights.max()
                for i, idx in enumerate(_per_indices):
                    rs = self.replay_buffer[idx]
                    macro_list.append(rs.macro_features)
                    mid_list.append(rs.mid_features)
                    micro_list.append(rs.micro_features)
                    y_dir_list.append(DIR_MAP[rs.actual_direction])
                    y_price_list.append(rs.actual_price_change)
                    weight_list.append(rs.sample_weight * is_weights[i])
                    log_ret_list.append(rs.future_log_return)
                    y_timing_list.append(rs.actual_timing)

        # Tensors
        macro_t = torch.FloatTensor(np.array(macro_list)).to(self.device)
        mid_t = torch.FloatTensor(np.array(mid_list)).to(self.device)
        micro_t = torch.FloatTensor(np.array(micro_list)).to(self.device)
        y_dir = torch.LongTensor(y_dir_list).to(self.device)
        y_price = torch.FloatTensor(y_price_list).to(self.device)
        weights = torch.FloatTensor(weight_list).to(self.device)
        y_log_ret = torch.FloatTensor(log_ret_list).to(self.device)
        y_timing = torch.FloatTensor(y_timing_list).to(self.device)

        # Recency decay
        now_ts = samples[-1].timestamp.timestamp() if samples else 0
        oldest_ts = samples[0].timestamp.timestamp() if samples else 0
        span_hours = max((now_ts - oldest_ts) / 3600, 0.01)
        half_life_hours = max(1.0, min(span_hours / 4, 720.0))

        recency_weights = []
        for s in samples:
            age_hours = max((now_ts - s.timestamp.timestamp()) / 3600, 0)
            rw = 2.0 ** (-age_hours / half_life_hours)
            recency_weights.append(max(0.1, min(rw, 1.0)))
        # Replay samples get weight 1.0
        recency_weights.extend([1.0] * (len(macro_list) - len(samples)))
        recency_t = torch.FloatTensor(recency_weights).to(self.device)
        weights = weights * recency_t

        # 온라인 모드 판별 (anchor 혼합 + optimizer 양쪽에서 사용)
        _is_online = freeze_encoders and not is_pretrain and self.is_pretrained and self._training_count > 0

        # Anchor Memory 혼합 (온라인 학습 시 SFT 패턴 상기)
        # anchor는 train set 앞쪽에 추가 → val split에 영향 없음
        _n_anchor_mixed = 0
        if _is_online and self._anchor_buffer:
            _n_anchor_mixed = self._mix_anchor_samples(
                macro_t, mid_t, micro_t, y_dir, y_price, y_log_ret, y_timing, weights
            )
            if _n_anchor_mixed > 0:
                # 텐서 재구성 (anchor가 앞쪽에 추가됨)
                macro_t, mid_t, micro_t, y_dir, y_price, y_log_ret, y_timing, weights = \
                    self._last_mixed_tensors

        # Train/Val split — 시간순 분할 + gap (과적합 감지 강화)
        # [train 70%][gap 10%][val 20%] — gap으로 인접 패턴 유사성 차단
        n = macro_t.shape[0]
        n_val = max(1, int(n * 0.2))
        n_gap = max(1, int(n * 0.1))
        n_train = n - n_val - n_gap
        train_idx = torch.arange(n_train)
        val_idx = torch.arange(n_train + n_gap, n)  # gap 건너뛰기

        # Logistic regression baseline (SFT 또는 5회마다)
        if is_pretrain or self._training_count % 5 == 0:
            try:
                self._logistic_baseline(
                    macro_t, mid_t, micro_t, y_dir, train_idx, val_idx
                )
            except Exception as e:
                logger.debug(f"  Logistic baseline 오류: {e}")

        # Deep feature diagnosis (SFT 시 1회만)
        if is_pretrain and self._training_count == 0:
            try:
                self._deep_feature_diagnosis(
                    macro_t, mid_t, micro_t, y_dir, train_idx, val_idx
                )
            except Exception as e:
                logger.debug(f"  Deep diagnosis 오류: {e}")

        # Encoder freezing
        if freeze_encoders:
            for p in self.model.encoder_params():
                p.requires_grad = False

        # Gradual unfreezing (SSL pretrained → first SFT)
        _use_gradual_unfreezing = self._ssl_pretrained and is_pretrain and not freeze_encoders
        _phase1_epochs = min(2, max(1, epochs // 3)) if _use_gradual_unfreezing else 0

        # Optimizer: 모드별 학습 파라미터 선택
        if freeze_encoders:
            if _is_online:
                # 온라인: LoRA만 (512p) — 모든 head 동결, hidden 표현만 미세 조정
                train_params = self.model.online_trainable_params()
            else:
                # SFT Phase 2: LoRA + 전체 heads
                train_params = self.model.trainable_params()
        elif _use_gradual_unfreezing:
            train_params = self.model.trainable_params()
        else:
            train_params = list(self.model.parameters())

        # LR 결정 — 30~50% 감소 (상수해 수렴 방지)
        if freeze_encoders:
            if _is_online:
                lr = 1e-5    # 온라인: 매우 보수적 (베이스 보존)
            else:
                lr = 1e-4    # SFT Phase 2 (3e-4→1e-4: Phase1과 동일, 안정적 전환)
        else:
            lr = 5e-5        # SFT Phase 1 (1e-4→5e-5: 50% 감소, 느린 수렴 유도)
        optimizer = torch.optim.AdamW(train_params, lr=lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

        # Loss functions — 2-class 균등 alpha + Focal gamma
        # gamma=2.0: 쉬운 샘플 down-weight, 어려운 경계 샘플 집중
        dir_alpha = torch.tensor([1.0, 1.0])
        dir_criterion = CostSensitiveFocalLoss(
            alpha=dir_alpha.to(self.device), gamma=2.0, reduction='none'
        )
        conf_criterion = nn.MSELoss(reduction='none')
        price_criterion = nn.MSELoss(reduction='none')
        timing_criterion = nn.MSELoss(reduction='none')
        pnl_criterion = PnLLoss()

        logger.info(f"  batch_train: n={n}, lr={lr:.1e}, freeze={freeze_encoders}, "
                    f"pretrain={is_pretrain}, patience={max(2, round(epochs * 0.25)) if is_pretrain else 2}")

        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        patience = max(4, round(epochs * 0.4)) if is_pretrain else 2  # 2→4, 25%→40%: 충분한 학습 시간

        for epoch in range(epochs):
            # Gradual unfreezing: phase transition
            if _use_gradual_unfreezing and epoch == _phase1_epochs:
                logger.info(f"  [SFT Phase 2] Backbone 해동: encoder_lr=5e-5, heads_lr=1e-4")
                for p in self.model.encoder_params():
                    p.requires_grad = True
                optimizer = torch.optim.AdamW([
                    {'params': self.model.encoder_params(), 'lr': 5e-5},
                    {'params': self.model.trainable_params(), 'lr': 1e-4},
                ], weight_decay=0.01)
                remaining = epochs - epoch
                scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-6)

            # Training — WeightedRandomSampler로 클래스 균형 배치 생성
            self.model.train()
            train_loss = 0
            n_batches = 0

            # 클래스별 가중치: 소수 클래스에 높은 샘플링 확률
            train_labels = y_dir[:n_train]
            class_counts = torch.bincount(train_labels, minlength=2).float().clamp(min=1)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[train_labels]
            sampler = WeightedRandomSampler(sample_weights, num_samples=n_train, replacement=True)
            sampled_indices = list(sampler)

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                idx = train_idx[sampled_indices[start:end]]

                b_macro = macro_t[idx]
                b_mid = mid_t[idx]
                b_micro = micro_t[idx]
                b_dir = y_dir[idx]
                b_price = y_price[idx]
                b_weights = weights[idx]
                b_log_ret = y_log_ret[idx]
                b_timing = y_timing[idx]

                optimizer.zero_grad()

                price_pred, dir_logits, conf, _, soft_pos, timing, _ = \
                    self.model.forward_with_position(b_macro, b_mid, b_micro)

                # Direction loss (focal + cost-sensitive)
                dir_loss = dir_criterion(dir_logits, b_dir)
                weighted_dir = (dir_loss * b_weights).mean()

                # Confidence loss (target = 정답 방향 확률)
                dir_probs = F.softmax(dir_logits.detach(), dim=-1)
                conf_target = dir_probs.gather(1, b_dir.unsqueeze(1)).squeeze(1)
                conf_loss = conf_criterion(conf.squeeze(-1), conf_target)
                weighted_conf = (conf_loss * b_weights).mean()

                # Price loss
                price_loss = price_criterion(price_pred.squeeze(-1), b_price)
                weighted_price = (price_loss * b_weights).mean()

                # PnL loss
                pnl_loss = pnl_criterion(soft_pos.squeeze(-1), b_log_ret)

                # Timing loss (배리어 도달 속도 예측)
                timing_loss = timing_criterion(timing.squeeze(-1), b_timing)
                weighted_timing = (timing_loss * b_weights).mean()

                # Batch-level Entropy Penalty — 배치 평균 분포의 쏠림 방지
                # 개별 확신은 보존, 배치 전체가 한 방향으로 collapse하는 것만 방지
                dir_probs_ent = F.softmax(dir_logits, dim=-1)
                mean_probs = dir_probs_ent.mean(dim=0)  # (3,) 배치 평균 클래스 분포
                batch_entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum()
                entropy_penalty = np.log(2) - batch_entropy  # 균일=0, 편향=높음

                # Loss 가중치 — entropy는 가벼운 정규화 (0.05)
                task_loss = (weighted_dir + 0.20 * pnl_loss + 0.15 * weighted_timing +
                             0.10 * weighted_conf + 0.10 * weighted_price)
                total = task_loss + 0.05 * entropy_penalty

                total.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += total.item()
                n_batches += 1

            avg_train = train_loss / max(n_batches, 1)

            # Validation
            self.model.eval()
            with torch.no_grad():
                v_idx = val_idx
                v_price, v_dir, v_conf, _, v_pos, v_timing, _ = \
                    self.model.forward_with_position(
                        macro_t[v_idx], mid_t[v_idx], micro_t[v_idx])

                v_dir_loss = dir_criterion(v_dir, y_dir[v_idx]).mean()
                v_price_loss = price_criterion(v_price.squeeze(-1), y_price[v_idx]).mean()
                v_pnl = pnl_criterion(v_pos.squeeze(-1), y_log_ret[v_idx])
                v_timing_loss = timing_criterion(v_timing.squeeze(-1), y_timing[v_idx]).mean()

                # Confidence validation loss (이전 누락 수정)
                v_dir_probs = F.softmax(v_dir, dim=-1)
                v_conf_target = v_dir_probs.gather(1, y_dir[v_idx].unsqueeze(1)).squeeze(1)
                v_conf_loss = conf_criterion(v_conf.squeeze(-1), v_conf_target).mean()

                # Batch-level Entropy Penalty (Validation)
                v_mean_probs = v_dir_probs.mean(dim=0)
                v_batch_entropy = -(v_mean_probs * torch.log(v_mean_probs + 1e-8)).sum()
                v_entropy_penalty = np.log(2) - v_batch_entropy

                # Validation loss = Training loss 가중치와 동일
                v_task_loss = (v_dir_loss + 0.05 * v_pnl + 0.15 * v_timing_loss +
                               0.10 * v_conf_loss + 0.10 * v_price_loss)
                val_loss = v_task_loss + 0.05 * v_entropy_penalty

            val_loss_val = val_loss.item()

            # Direction accuracy + 상세 진단
            with torch.no_grad():
                v_probs = F.softmax(v_dir, dim=-1)
                pred_dirs = v_probs.argmax(dim=-1)
                correct = (pred_dirs == y_dir[v_idx]).float().mean().item()

            logger.info(
                f"  Epoch {epoch+1}/{epochs}: train={avg_train:.4f}, "
                f"val={val_loss_val:.4f}, dir_acc={correct:.1%}"
            )

            # 마지막 epoch 또는 매 3 epoch마다 상세 진단 출력
            if epoch == epochs - 1 or (epoch + 1) % 3 == 0:
                self._log_training_diagnostics(
                    v_probs, pred_dirs, y_dir[v_idx], epoch + 1, epochs
                )

            # LR schedule
            scheduler.step()

            # Early stopping
            if val_loss_val < best_val_loss - 1e-4:
                best_val_loss = val_loss_val
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"  조기 종료 (patience={patience})")
                    break

        # Rollback check
        # 온라인 학습은 분포 변화 → 절대 기준 대신 상대 기준 사용
        # val_loss가 best 대비 tolerance% 이내면 개선으로 인정 (분포 적응 허용)
        rolled_back = False
        if not is_pretrain:
            _tol = self._best_val_loss * self._val_loss_tolerance
            if val_loss_val < self._best_val_loss + _tol:
                # 개선 또는 허용 범위 → 수용
                if val_loss_val < self._best_val_loss:
                    self._best_val_loss = val_loss_val
                self._consecutive_degrades = 0
            else:
                # 유의미한 악화
                self._consecutive_degrades += 1
                if self._consecutive_degrades >= self._max_degrades_before_rollback:
                    self.model.load_state_dict(self._checkpoint_state)
                    rolled_back = True
                    self._consecutive_degrades = 0  # 롤백 후 리셋 (연쇄 롤백 방지)
                    # best_val_loss를 약간 완화하여 분포 변화 수용
                    self._best_val_loss *= 1.01
                    logger.warning(f"  ⚠️ 성능 저하 → 롤백 (best_val={self._best_val_loss:.4f})")
        else:
            self._best_val_loss = float('inf')
            self._consecutive_degrades = 0

        # Unfreeze for next time if needed
        if freeze_encoders:
            for p in self.model.encoder_params():
                p.requires_grad = True

        if _use_gradual_unfreezing:
            self._ssl_pretrained = False

        self._training_count += 1
        self._last_train_time = datetime.now()
        self.is_pretrained = True
        self.model.eval()

        # SFT 완료 시 Anchor Memory 구축 (온라인 학습용 catastrophic forgetting 방지)
        if is_pretrain and not self._anchor_buffer:
            self._build_anchor_buffer(
                macro_t, mid_t, micro_t,
                y_dir, y_price, y_log_ret, y_timing, weights
            )

        # LoRA soft 리셋 (200회 학습마다, trading 중단 없이)
        if self._training_count % 200 == 0:
            self._soft_reset_lora()

        # PER beta annealing
        if _per_indices:
            self._per_beta = min(1.0, self._per_beta + 0.01)

        return {
            'status': 'rolled_back' if rolled_back else 'completed',
            'train_loss': avg_train,
            'val_loss': val_loss_val,
            'dir_accuracy': correct,
            'epochs_run': epoch + 1,
            'samples': len(samples),
            'training_count': self._training_count,
        }

    def _log_training_diagnostics(
        self, v_probs: torch.Tensor, pred_dirs: torch.Tensor,
        true_dirs: torch.Tensor, epoch: int, total_epochs: int
    ):
        """학습 진단: per-class metrics + router + output 분산."""
        import math as _math
        labels = {0: 'DOWN', 1: 'UP'}

        # 1) Per-class Precision / Recall / F1
        lines = [f"  📊 [진단 Epoch {epoch}/{total_epochs}]"]
        for cls_idx, cls_name in labels.items():
            tp = ((pred_dirs == cls_idx) & (true_dirs == cls_idx)).sum().item()
            fp = ((pred_dirs == cls_idx) & (true_dirs != cls_idx)).sum().item()
            fn = ((pred_dirs != cls_idx) & (true_dirs == cls_idx)).sum().item()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            actual_n = (true_dirs == cls_idx).sum().item()
            pred_n = (pred_dirs == cls_idx).sum().item()
            lines.append(
                f"    {cls_name:>5s}: P={prec:.2f} R={rec:.2f} F1={f1:.2f} "
                f"(pred={pred_n}, actual={actual_n})"
            )

        # 2) Output 분산 (direction probs — 2-class)
        prob_mean = v_probs.mean(dim=0).tolist()  # [p_down, p_up]
        prob_std = v_probs.std(dim=0).tolist()
        lines.append(
            f"    출력분포: D={prob_mean[0]:.3f}±{prob_std[0]:.3f}, "
            f"U={prob_mean[1]:.3f}±{prob_std[1]:.3f}"
        )

        # 3) AUC + p_up 분포 분석
        try:
            from sklearn.metrics import roc_auc_score as _auc
            p_up_np = v_probs[:, 1].cpu().numpy()
            y_np = true_dirs.cpu().numpy()
            if len(np.unique(y_np)) >= 2:
                auc = _auc(y_np, p_up_np)
                lines.append(f"    AUC={auc:.3f}, p_up: mean={np.mean(p_up_np):.3f}, std={np.std(p_up_np):.3f}")
                if auc < 0.55:
                    lines.append(f"    ⚠️ AUC < 0.55 — 모델 분별력 부족")
        except Exception:
            pass

        # 4) GRU + LoRA 상태
        lora_norm = sum(
            p.data.norm().item() for p in self.model.online_lora.parameters()
        )
        lines.append(f"    LoRA norm: {lora_norm:.4f}")

        for line in lines:
            logger.info(line)

    def _logistic_baseline(
        self, macro_t: torch.Tensor, mid_t: torch.Tensor,
        micro_t: torch.Tensor, y_dir: torch.Tensor,
        train_idx: torch.Tensor, val_idx: torch.Tensor
    ):
        """Logistic regression baseline — 피처 신호 존재 여부 판정.

        val_acc < 52%  → 피처에 신호 없음, 딥러닝 무의미
        val_acc ≥ 55%  → 신호 존재, 딥러닝 의미 있음
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("  sklearn 미설치 — logistic baseline 생략")
            return

        # Flatten: (n, seq, feat) → (n, seq*feat)
        X_macro = macro_t.cpu().numpy().reshape(macro_t.shape[0], -1)
        X_mid = mid_t.cpu().numpy().reshape(mid_t.shape[0], -1)
        X_micro = micro_t.cpu().numpy().reshape(micro_t.shape[0], -1)
        X = np.concatenate([X_macro, X_mid, X_micro], axis=1)
        y = y_dir.cpu().numpy()

        ti = train_idx.numpy()
        vi = val_idx.numpy()
        X_train, X_val = X[ti], X[vi]
        y_train, y_val = y[ti], y[vi]

        # 클래스 수 확인 (val에 단일 클래스면 AUC 불가)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            logger.info("  📏 Logistic baseline: 단일 클래스 — 생략")
            return

        # StandardScaler — lbfgs 수렴 보장
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
        lr.fit(X_train_s, y_train)

        train_acc = lr.score(X_train_s, y_train)
        val_acc = lr.score(X_val_s, y_val)

        # AUC
        try:
            val_proba = lr.predict_proba(X_val_s)[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)
        except Exception:
            val_auc = 0.0

        # Sigmoid 분포 — val set 예측 확률의 평균/표준편차
        val_mean = float(np.mean(val_proba)) if val_auc > 0 else 0.5
        val_std = float(np.std(val_proba)) if val_auc > 0 else 0.0

        logger.info(
            f"  📏 Logistic baseline: train={train_acc:.1%}, "
            f"val={val_acc:.1%}, AUC={val_auc:.3f}, "
            f"prob_dist={val_mean:.3f}±{val_std:.3f}"
        )

        if val_acc < 0.52:
            logger.warning(
                f"  ⚠️ Feature signal 약함 (val {val_acc:.1%} < 52%) "
                f"— 딥러닝 효과 제한적"
            )
        elif val_acc >= 0.55:
            logger.info(
                f"  ✅ Feature signal 존재 (val {val_acc:.1%} ≥ 55%) "
                f"— 딥러닝 의미 있음"
            )

    def _deep_feature_diagnosis(
        self, macro_t: torch.Tensor, mid_t: torch.Tensor,
        micro_t: torch.Tensor, y_dir: torch.Tensor,
        train_idx: torch.Tensor, val_idx: torch.Tensor
    ):
        """종합 피처 진단 — 신호/라벨/중요도 3종 분석.

        1) 단순 피처 신호 테스트 (last-step 핵심 피처 → logistic)
        2) 라벨 자기상관 분석 (lag-1 autocorrelation)
        3) 피처 중요도 (RandomForest feature importance)
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import roc_auc_score
        except ImportError:
            logger.warning("  sklearn 미설치 — 진단 생략")
            return

        ti = train_idx.numpy()
        vi = val_idx.numpy()
        y = y_dir.cpu().numpy()
        y_train, y_val = y[ti], y[vi]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            logger.info("  🔬 Deep diagnosis: 단일 클래스 — 생략")
            return

        logger.info("  ═══ 🔬 Deep Feature Diagnosis ═══")

        # ──────────────────────────────────────────────
        # 1) 단순 피처 신호 테스트
        #    각 스케일의 마지막 타임스텝 피처만 사용 (11개 × 3 = 33개)
        #    시계열 flatten 대신 "지금 상태"만으로 방향 예측 가능한지 테스트
        # ──────────────────────────────────────────────
        macro_np = macro_t.cpu().numpy()  # (n, 60, 11)
        mid_np = mid_t.cpu().numpy()      # (n, 40, 11)
        micro_np = micro_t.cpu().numpy()  # (n, 30, 11)

        # Last-step features: 각 스케일의 마지막 bar 피처
        X_last = np.concatenate([
            macro_np[:, -1, :],   # (n, 11) — 15m 마지막 bar
            mid_np[:, -1, :],     # (n, 11) — 3m 마지막 bar
            micro_np[:, -1, :],   # (n, 11) — 1m 마지막 bar
        ], axis=1)  # (n, 33)

        feature_names = [
            # macro (15m)
            'M_return', 'M_atr_ratio', 'M_bb_pos', 'M_vol_delta',
            'M_vol_change', 'M_mom_slope', 'M_htf_trend', 'M_vol_cycle',
            'M_regime', 'M_hl_struct', 'M_htf_bias',
            # mid (3m)
            'm_return', 'm_atr_ratio', 'm_bb_pos', 'm_vol_delta',
            'm_vol_change', 'm_mom_slope', 'm_di_diff', 'm_taker_buy',
            'm_trade_int', 'm_vwap_dev', 'm_htf_bias',
            # micro (1m)
            'u_return', 'u_atr_ratio', 'u_bb_pos', 'u_vol_delta',
            'u_vol_change', 'u_mom_slope', 'u_di_diff', 'u_taker_buy',
            'u_trade_int', 'u_vwap_dev', 'u_htf_bias',
        ]

        X_tr, X_va = X_last[ti], X_last[vi]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
        lr.fit(X_tr_s, y_train)
        lr_train = lr.score(X_tr_s, y_train)
        lr_val = lr.score(X_va_s, y_val)
        try:
            lr_proba = lr.predict_proba(X_va_s)[:, 1]
            lr_auc = roc_auc_score(y_val, lr_proba)
        except Exception:
            lr_auc = 0.0

        logger.info(
            f"  [1/3] 단순피처(last-step 33개) Logistic: "
            f"train={lr_train:.1%}, val={lr_val:.1%}, AUC={lr_auc:.3f}"
        )
        if lr_val < 0.52:
            logger.warning(f"        → 라벨이 이 피처로 예측 불가능 — 라벨링 재검토 필요")
        elif lr_val >= 0.55:
            logger.info(f"        → 신호 존재! 현재 모델이 이를 활용 못하는 것")

        # 스케일별 분리 테스트
        for name, start, end in [('Macro(15m)', 0, 11), ('Mid(3m)', 11, 22), ('Micro(1m)', 22, 33)]:
            X_sc_tr = scaler.fit_transform(X_tr[:, start:end])
            X_sc_va = scaler.transform(X_va[:, start:end])
            lr_sc = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
            lr_sc.fit(X_sc_tr, y_train)
            sc_val = lr_sc.score(X_sc_va, y_val)
            try:
                sc_auc = roc_auc_score(y_val, lr_sc.predict_proba(X_sc_va)[:, 1])
            except Exception:
                sc_auc = 0.0
            logger.info(f"        {name}: val={sc_val:.1%}, AUC={sc_auc:.3f}")

        # ──────────────────────────────────────────────
        # 2) 라벨 자기상관 분석
        #    시간순 라벨의 lag-1,2,3 autocorrelation
        #    ~0이면 라벨이 coin flip (예측 불가능)
        # ──────────────────────────────────────────────
        y_series = y.astype(float)  # 0/1
        n_total = len(y_series)
        y_centered = y_series - y_series.mean()

        autocorrs = []
        var = np.sum(y_centered ** 2)
        for lag in [1, 2, 3, 5, 10]:
            if lag < n_total and var > 0:
                ac = np.sum(y_centered[:-lag] * y_centered[lag:]) / var
                autocorrs.append((lag, ac))

        ac_str = ", ".join([f"lag{lag}={ac:+.3f}" for lag, ac in autocorrs])
        logger.info(f"  [2/3] 라벨 자기상관: {ac_str}")

        # 유의 기준: |ac| > 2/sqrt(n) = 약 0.03~0.05 (n=2000~4000)
        threshold = 2.0 / np.sqrt(n_total)
        significant = [(lag, ac) for lag, ac in autocorrs if abs(ac) > threshold]
        if not significant:
            logger.warning(
                f"        → 모든 lag에서 자기상관 ≈ 0 (임계={threshold:.3f}) "
                f"— 라벨이 사실상 random"
            )
        else:
            sig_str = ", ".join([f"lag{lag}={ac:+.3f}" for lag, ac in significant])
            logger.info(f"        → 유의한 자기상관: {sig_str}")

        # UP/DOWN 비율
        up_ratio = y_series.mean()
        logger.info(f"        라벨 분포: UP={up_ratio:.1%}, DOWN={1 - up_ratio:.1%}")

        # 연속 패턴 (runs test 간이 버전)
        changes = np.sum(np.diff(y_series) != 0)
        expected_changes = 2 * up_ratio * (1 - up_ratio) * (n_total - 1)
        logger.info(
            f"        방향전환: {changes}회 (random 기대={expected_changes:.0f})"
        )

        # ──────────────────────────────────────────────
        # 3) 피처 중요도 (RandomForest)
        #    어떤 피처가 그나마 신호가 있는지 파악
        # ──────────────────────────────────────────────
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
        )
        rf.fit(X_tr_s, y_train)
        rf_train = rf.score(X_tr_s, y_train)
        rf_val = rf.score(X_va_s, y_val)
        try:
            rf_auc = roc_auc_score(y_val, rf.predict_proba(X_va_s)[:, 1])
        except Exception:
            rf_auc = 0.0

        logger.info(
            f"  [3/3] RandomForest(d=5): "
            f"train={rf_train:.1%}, val={rf_val:.1%}, AUC={rf_auc:.3f}"
        )

        # Top-10 important features
        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        logger.info("        Top-10 features:")
        for rank, idx in enumerate(top_idx, 1):
            fname = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
            logger.info(f"          {rank:2d}. {fname:15s} = {importances[idx]:.4f}")

        # ──────────────────────────────────────────────
        # 종합 판정
        # ──────────────────────────────────────────────
        logger.info("  ─── 종합 판정 ───")
        if lr_val < 0.52 and rf_val < 0.52:
            logger.warning(
                "  🔴 피처+라벨 모두 신호 없음 → 라벨링 방식 재설계 필요"
            )
        elif lr_val < 0.52 and rf_val >= 0.52:
            logger.info(
                "  🟡 비선형 신호 존재 (RF > Logistic) → 모델 구조는 적합, "
                "피처 선별 필요"
            )
        elif lr_val >= 0.55:
            logger.info(
                "  🟢 선형 신호 존재 → 현재 모델이 학습 못하는 것, "
                "학습 파이프라인 점검"
            )
        else:
            logger.info(
                "  🟡 약한 신호 (52-55%) → 피처 보강 또는 라벨 품질 개선 필요"
            )

    # ─── Periodic Train ───────────────────────────────────────

    def periodic_train(self, min_interval_minutes: int = 30) -> Optional[dict]:
        """주기적 온라인 학습 — LoRA 비활성화로 중단."""
        # LoRA 비활성화: 베이스 모델 성능 먼저 확보 후 재활성화
        return None

    # ─── LoRA Management ──────────────────────────────────────

    def _reset_lora(self):
        """LoRA 어댑터 가중치 리셋 (과적합 시 초기화)."""
        nn.init.kaiming_uniform_(self.model.online_lora.down.weight)
        nn.init.zeros_(self.model.online_lora.up.weight)
        logger.info("  LoRA 어댑터 리셋 완료")
        self.shadow_validator.force_shadow("LoRA reset")

    def _soft_reset_lora(self):
        """LoRA soft 리셋 — up만 zero화, trading 중단 없음.

        up=0 → delta=0 → 베이스 모델 출력으로 복귀.
        down weight는 보존 (학습된 특성 공간 유지).
        shadow validator는 건드리지 않음 (LIVE 유지 가능).
        """
        nn.init.zeros_(self.model.online_lora.up.weight)
        logger.info(f"  LoRA soft 리셋 (training_count={self._training_count})")

    # ─── Encoder Freeze/Unfreeze ──────────────────────────────

    def freeze_encoders(self):
        """Eyes + Backbone 동결."""
        for p in self.model.encoder_params():
            p.requires_grad = False

    def unfreeze_encoders(self):
        """Eyes + Backbone 해제."""
        for p in self.model.encoder_params():
            p.requires_grad = True

    # ─── Anchor Memory ──────────────────────────────────────────

    def _build_anchor_buffer(
        self, macro_t, mid_t, micro_t, y_dir, y_price, y_log_ret, y_timing, weights
    ):
        """SFT 완료 후 대표 샘플을 anchor buffer에 저장.

        모델이 높은 confidence로 맞춘 샘플 = base 패턴의 대표.
        """
        n = macro_t.shape[0]
        if n == 0:
            return

        # 모델 예측으로 confidence 높은 정답 샘플 선별
        self.model.eval()
        with torch.no_grad():
            _, dir_logits, conf, _, _, _, _ = self.model.forward_with_position(
                macro_t, mid_t, micro_t
            )
            pred_dirs = dir_logits.argmax(dim=1)
            correct_mask = (pred_dirs == y_dir)
            conf_scores = conf.squeeze(-1)

        # 정답이면서 confidence 높은 순으로 정렬
        scores = torch.where(correct_mask, conf_scores, torch.zeros_like(conf_scores))
        _, top_indices = scores.sort(descending=True)
        n_anchor = min(self._anchor_max_size, int(n * 0.5))  # 최대 50% 또는 500건

        self._anchor_buffer = []
        for idx in top_indices[:n_anchor].cpu().tolist():
            self._anchor_buffer.append({
                'macro': macro_t[idx].cpu().numpy(),
                'mid': mid_t[idx].cpu().numpy(),
                'micro': micro_t[idx].cpu().numpy(),
                'dir': y_dir[idx].item(),
                'price': y_price[idx].item(),
                'log_ret': y_log_ret[idx].item(),
                'timing': y_timing[idx].item(),
                'weight': weights[idx].item(),
            })

        logger.info(
            f"  Anchor Memory 구축: {len(self._anchor_buffer)}건 "
            f"(정답+고신뢰 상위 샘플, 전체 {n}건 중)"
        )

    def _mix_anchor_samples(
        self, macro_t, mid_t, micro_t, y_dir, y_price, y_log_ret, y_timing, weights
    ) -> int:
        """온라인 학습 시 anchor 샘플을 혼합하여 텐서 재구성.

        anchor는 train set 앞쪽에 추가 → val split(뒤 20%)에 포함 안 됨.
        Returns: 혼합된 anchor 샘플 수
        """
        if not self._anchor_buffer:
            self._last_mixed_tensors = (macro_t, mid_t, micro_t, y_dir, y_price, y_log_ret, y_timing, weights)
            return 0

        n_recent = macro_t.shape[0]
        n_anchor = max(1, int(n_recent * self._anchor_mix_ratio / (1 - self._anchor_mix_ratio)))
        n_anchor = min(n_anchor, len(self._anchor_buffer))

        # anchor에서 랜덤 샘플링
        indices = np.random.choice(len(self._anchor_buffer), size=n_anchor, replace=False)

        a_macro, a_mid, a_micro = [], [], []
        a_dir, a_price, a_lr, a_timing, a_weight = [], [], [], [], []

        for idx in indices:
            a = self._anchor_buffer[idx]
            a_macro.append(a['macro'])
            a_mid.append(a['mid'])
            a_micro.append(a['micro'])
            a_dir.append(a['dir'])
            a_price.append(a['price'])
            a_lr.append(a['log_ret'])
            a_timing.append(a['timing'])
            a_weight.append(a['weight'])

        dev = macro_t.device
        # anchor를 앞쪽에 concat (train set에만 포함)
        self._last_mixed_tensors = (
            torch.cat([torch.FloatTensor(np.array(a_macro)).to(dev), macro_t]),
            torch.cat([torch.FloatTensor(np.array(a_mid)).to(dev), mid_t]),
            torch.cat([torch.FloatTensor(np.array(a_micro)).to(dev), micro_t]),
            torch.cat([torch.LongTensor(a_dir).to(dev), y_dir]),
            torch.cat([torch.FloatTensor(a_price).to(dev), y_price]),
            torch.cat([torch.FloatTensor(a_lr).to(dev), y_log_ret]),
            torch.cat([torch.FloatTensor(a_timing).to(dev), y_timing]),
            torch.cat([torch.FloatTensor(a_weight).to(dev), weights]),
        )
        return n_anchor

    # ─── Save / Load ──────────────────────────────────────────

    def save_model(self, path: str):
        """전체 모델 저장 (enhanced metadata)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model_state': self.model.state_dict(),
            'training_count': self._training_count,
            'is_pretrained': self.is_pretrained,
            'best_val_loss': self._best_val_loss,
            'symbol': self.symbol,
            # Enhanced metadata
            'saved_at': datetime.now().isoformat(),
            'architecture': {
                'macro_features': self.macro_features,
                'mid_features': self.mid_features,
                'micro_features': self.micro_features,
                'macro_seq_len': self.macro_seq_len,
                'mid_seq_len': self.mid_seq_len,
                'micro_seq_len': self.micro_seq_len,
            },
            'shadow_state': self.shadow_validator.to_dict(),
            'consecutive_degrades': self._consecutive_degrades,
            'anchor_buffer': self._anchor_buffer,
        }
        torch.save(checkpoint, path)
        logger.info(f"모델 저장: {path} (count={self._training_count}, anchor={len(self._anchor_buffer)})")

    def load_model(self, path: str) -> bool:
        """전체 모델 로드 (backward-compatible)."""
        if not Path(path).exists():
            logger.warning(f"모델 파일 없음: {path}")
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            missing, unexpected = self.model.load_state_dict(
                checkpoint['model_state'], strict=False
            )
            if missing:
                logger.info(f"  새로 초기화된 레이어: {missing}")
            if unexpected:
                logger.info(f"  무시된 레이어: {unexpected}")
            self._training_count = checkpoint.get('training_count', 0)
            self.is_pretrained = checkpoint.get('is_pretrained', True)
            self._best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self._consecutive_degrades = checkpoint.get('consecutive_degrades', 0)

            # Shadow validator 상태 복원 (backward-compatible)
            shadow_state = checkpoint.get('shadow_state')
            if shadow_state:
                self.shadow_validator.from_dict(shadow_state)
                logger.info(
                    f"  Shadow 상태 복원: {shadow_state.get('state', '?')}, "
                    f"resolved={len(shadow_state.get('virtual_trades', []))}"
                )

            # Anchor Memory 복원
            anchor_data = checkpoint.get('anchor_buffer', [])
            if anchor_data:
                self._anchor_buffer = anchor_data
                logger.info(f"  Anchor Memory 복원: {len(self._anchor_buffer)}건")

            saved_at = checkpoint.get('saved_at', 'unknown')
            logger.info(f"모델 로드: {path} (count={self._training_count}, saved={saved_at})")
            return True
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return False

    def save_encoders(self, path: str):
        """인코더(Eyes+Backbone)만 저장 — hot-swap용."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        encoder_state = {}
        for name, param in self.model.named_parameters():
            # Eyes + Backbone + Attention (MoE, Heads 제외)
            if any(name.startswith(prefix) for prefix in [
                'micro_revin', 'micro_eye', 'mid_revin', 'mid_eye', 'mid_pool',
                'macro_revin', 'macro_eye', 'macro_pool',
                'backbone', 'attention'
            ]):
                encoder_state[name] = param.data.clone()

        torch.save({
            'encoder_state': encoder_state,
            'symbol': self.symbol,
        }, path)
        logger.info(f"인코더 저장: {path} ({len(encoder_state)} params)")

    def load_encoders(self, path: str) -> bool:
        """인코더만 로드 — hot-swap용. MoE/Heads는 유지."""
        if not Path(path).exists():
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            encoder_state = checkpoint['encoder_state']

            model_state = self.model.state_dict()
            loaded = 0
            for name, value in encoder_state.items():
                if name in model_state and model_state[name].shape == value.shape:
                    model_state[name] = value
                    loaded += 1

            self.model.load_state_dict(model_state, strict=False)
            logger.info(f"인코더 로드: {path} ({loaded} params)")
            self._ssl_pretrained = True
            return True
        except Exception as e:
            logger.error(f"인코더 로드 실패: {e}")
            return False
