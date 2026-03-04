"""
수익률 기반 AI 학습 스크립트 (Profit-Weighted Training)

목표: 65%+ 정확도, Profit Factor > 1.5
핵심 개선:
1. Profit-Weighted Loss: 큰 수익/손실 예측에 가중치
2. Asymmetric Loss: 틀렸을 때 큰 손실 페널티
3. 수익률 회귀 가중치 증가 (0.1 → 0.4)
4. Profit Factor 기반 모델 선택
5. Kelly Criterion 기반 신뢰도 캘리브레이션
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from api.futures_client import BinanceFuturesClient
from ai.market_analyzer import MarketAnalyzer, TrainingSample
from utils.logger import setup_logger


@dataclass
class ProfitMetrics:
    """수익률 기반 평가 메트릭"""
    accuracy: float
    profit_factor: float  # 총수익 / 총손실
    total_return: float   # 레버리지 포함 총 수익률
    win_rate: float       # 방향 맞춘 비율
    avg_win: float        # 평균 수익
    avg_loss: float       # 평균 손실
    kelly_optimal: float  # Kelly 최적 베팅 비율
    sharpe_ratio: float   # 샤프 비율


class ProfitWeightedLoss(nn.Module):
    """
    수익률 기반 가중치 손실 함수 (2-class direction + confidence)

    - 큰 수익/손실 예측에 높은 가중치
    - 틀렸을 때 큰 손실 페널티 (비대칭)
    """

    def __init__(
        self,
        direction_weight: float = 0.4,
        confidence_weight: float = 0.15,
        return_weight: float = 0.35,
        pnl_weight: float = 0.1,
        profit_power: float = 1.2,      # 수익률 가중치 지수
        loss_penalty: float = 1.5,      # 손실 페널티 배율
    ):
        super().__init__()
        self.direction_weight = direction_weight
        self.confidence_weight = confidence_weight
        self.return_weight = return_weight
        self.pnl_weight = pnl_weight
        self.profit_power = profit_power
        self.loss_penalty = loss_penalty

        self.direction_criterion = nn.BCELoss(reduction='none')
        self.confidence_criterion = nn.MSELoss(reduction='none')
        self.return_criterion = nn.HuberLoss(reduction='none', delta=1.0)

    def forward(
        self,
        direction_out: torch.Tensor,    # (batch, 1) sigmoid
        confidence_out: torch.Tensor,   # (batch, 1) sigmoid
        price_pred: torch.Tensor,       # (batch, 1)
        target_direction: torch.Tensor, # (batch,) float: UP=1.0, DOWN=0.0, NEUTRAL=0.5
        target_confidence: torch.Tensor,# (batch,) float: UP/DOWN=1.0, NEUTRAL=0.0
        target_return: torch.Tensor,    # (batch,)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """수익률 가중치 적용 손실 계산 (2-class)"""
        # 1. Direction loss (BCE)
        dir_loss = self.direction_criterion(direction_out.squeeze(), target_direction)

        # 2. Confidence loss (MSE)
        conf_loss = self.confidence_criterion(confidence_out.squeeze(), target_confidence)

        # 3. Return loss (Huber — 이상치에 강건)
        return_loss = self.return_criterion(price_pred.squeeze(), target_return)

        # 4. Edge 기반 예측 정확도 계산
        edge = direction_out.squeeze() - 0.5
        predicted_up = (edge > 0.025).float()
        predicted_down = (edge < -0.025).float()
        actual_up = (target_direction > 0.75).float()
        actual_down = (target_direction < 0.25).float()
        is_correct = (predicted_up * actual_up + predicted_down * actual_down).clamp(0, 1)

        # 5. 수익률 기반 샘플 가중치 (절대값이 클수록 중요)
        abs_return = torch.abs(target_return)
        profit_weights = (1.0 + abs_return) ** self.profit_power
        profit_weights = profit_weights / profit_weights.mean()

        # 6. 비대칭 손실 가중치 (틀렸을 때 + 큰 손실 = 큰 페널티)
        wrong_penalty = (1.0 - is_correct) * (1.0 + abs_return * self.loss_penalty)
        sample_weights = profit_weights * (1.0 + wrong_penalty)

        # 7. 가중 평균 손실
        weighted_dir_loss = (dir_loss * sample_weights).mean()
        weighted_conf_loss = (conf_loss * profit_weights).mean()
        weighted_return_loss = (return_loss * profit_weights).mean()

        # 8. 총 손실
        total_loss = (
            self.direction_weight * weighted_dir_loss +
            self.confidence_weight * weighted_conf_loss +
            self.return_weight * weighted_return_loss
        )

        metrics = {
            'dir_loss': weighted_dir_loss.item(),
            'conf_loss': weighted_conf_loss.item(),
            'return_loss': weighted_return_loss.item(),
            'total_loss': total_loss.item(),
            'avg_profit_weight': profit_weights.mean().item()
        }

        return total_loss, metrics


def calculate_profit_metrics(
    predictions: List[int],      # 예측 방향 (0=UP, 1=NEUTRAL, 2=DOWN)
    actuals: List[int],          # 실제 방향
    returns: List[float],        # 실제 수익률 (%)
    confidences: List[float],    # 예측 신뢰도
    leverage: float = 3.0        # 기본 레버리지
) -> ProfitMetrics:
    """
    수익률 기반 종합 메트릭 계산
    """
    n = len(predictions)
    if n == 0:
        return ProfitMetrics(0, 0, 0, 0, 0, 0, 0, 0)

    correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
    accuracy = correct / n

    # 거래 시뮬레이션 (방향 예측 기반)
    wins = []
    losses = []
    trade_returns = []

    for pred, actual, ret, conf in zip(predictions, actuals, returns, confidences):
        # NEUTRAL 예측은 거래 안함
        if pred == 1:  # NEUTRAL
            continue

        # 예측 방향에 따른 포지션
        position = 1 if pred == 0 else -1  # UP=Long, DOWN=Short

        # 실제 수익률 (레버리지 적용)
        # 신뢰도 기반 레버리지 조절 (Kelly 스타일)
        adjusted_leverage = leverage * min(conf, 1.0)
        trade_return = position * ret * adjusted_leverage / 100

        trade_returns.append(trade_return)

        if trade_return > 0:
            wins.append(trade_return)
        else:
            losses.append(abs(trade_return))

    if not trade_returns:
        return ProfitMetrics(accuracy, 0, 0, 0, 0, 0, 0, 0)

    total_wins = sum(wins) if wins else 0
    total_losses = sum(losses) if losses else 0.0001  # 0 방지

    profit_factor = total_wins / total_losses
    total_return = sum(trade_returns) * 100  # %로 변환
    win_rate = len(wins) / len(trade_returns) if trade_returns else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    # Kelly Criterion: f* = (bp - q) / b
    # b = avg_win / avg_loss, p = win_rate, q = 1 - p
    if avg_loss > 0 and avg_win > 0:
        b = avg_win / avg_loss
        kelly_optimal = (b * win_rate - (1 - win_rate)) / b
        kelly_optimal = max(0, min(kelly_optimal, 1.0))  # 0~1 범위
    else:
        kelly_optimal = 0

    # Sharpe Ratio (간단 버전)
    if len(trade_returns) > 1:
        sharpe_ratio = np.mean(trade_returns) / (np.std(trade_returns) + 1e-6) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    return ProfitMetrics(
        accuracy=accuracy,
        profit_factor=profit_factor,
        total_return=total_return,
        win_rate=win_rate,
        avg_win=avg_win * 100,
        avg_loss=avg_loss * 100,
        kelly_optimal=kelly_optimal,
        sharpe_ratio=sharpe_ratio
    )


def fetch_historical_data(
    symbol: str,
    interval: str,
    days: int,
    testnet: bool = True
) -> pd.DataFrame:
    """히스토리컬 데이터 수집"""
    print(f"\n📊 데이터 수집 중: {symbol} {interval} ({days}일)")

    client = BinanceFuturesClient(testnet=testnet)

    interval_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15,
        '30m': 30, '1h': 60, '4h': 240, '1d': 1440
    }
    minutes = interval_minutes.get(interval, 60)
    total_candles = int(days * 24 * 60 / minutes)

    print(f"   예상 캔들 수: {total_candles:,}개")

    all_klines = []
    batch_size = 1000
    end_time = int(datetime.now().timestamp() * 1000)

    collected = 0
    while collected < total_candles:
        try:
            klines = client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=min(batch_size, total_candles - collected),
                end_time=end_time
            )

            if not klines:
                break

            all_klines = klines + all_klines  # 시간순 정렬
            collected += len(klines)

            # 다음 배치의 끝 시간 설정 (klines는 리스트의 리스트, [0]은 timestamp)
            end_time = klines[0][0] - 1

            print(f"   수집 중... {collected:,}/{total_candles:,} ({collected/total_candles*100:.1f}%)")

            time.sleep(0.1)

        except Exception as e:
            print(f"   ⚠️ 데이터 수집 중 오류: {e}")
            break

    if not all_klines:
        return pd.DataFrame()

    # 데이터프레임 변환 (Binance klines 형식: 리스트의 리스트)
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]

    print(f"\n✅ 데이터 수집 완료: {len(df):,}개 캔들")
    print(f"   기간: {df.index[0]} ~ {df.index[-1]}")

    return df


def prepare_profit_weighted_samples(
    df: pd.DataFrame,
    market_analyzer: MarketAnalyzer,
    lookahead: int = 1,
    threshold_pct: float = 0.1,
    balance_classes: bool = True,
    profit_threshold: float = 0.5  # 큰 수익 기준 (%)
) -> Tuple[List[TrainingSample], np.ndarray]:
    """
    수익률 기반 가중치를 포함한 학습 샘플 생성

    Returns:
        samples: TrainingSample 리스트
        sample_weights: 샘플별 가중치 (profit-weighted)
    """
    print(f"\n🔧 수익률 가중 샘플 생성 중...")
    print(f"   방향 임계값: ±{threshold_pct}%")
    print(f"   큰 수익 기준: ±{profit_threshold}%")

    samples_by_direction = {'UP': [], 'DOWN': [], 'NEUTRAL': []}
    sequence_length = market_analyzer.sequence_length
    min_data = sequence_length + 100

    features = market_analyzer.prepare_features(df)
    total_samples = len(df) - min_data - lookahead

    for i in range(min_data, len(df) - lookahead):
        try:
            sequence = features[i - sequence_length:i].copy()

            current_price = df['close'].iloc[i]
            future_price = df['close'].iloc[i + lookahead]
            price_change = (future_price / current_price - 1) * 100

            # 방향 결정
            if price_change > threshold_pct:
                direction = 'UP'
            elif price_change < -threshold_pct:
                direction = 'DOWN'
            else:
                direction = 'NEUTRAL'

            sample = TrainingSample(
                features=sequence,
                actual_direction=direction,
                actual_price_change=price_change,
                timestamp=df.index[i]
            )
            samples_by_direction[direction].append(sample)

        except Exception as e:
            continue

    # 방향 분포 확인
    up_count = len(samples_by_direction['UP'])
    down_count = len(samples_by_direction['DOWN'])
    neutral_count = len(samples_by_direction['NEUTRAL'])
    total_count = up_count + down_count + neutral_count

    print(f"\n📊 원본 샘플 분포:")
    print(f"   UP: {up_count:,} ({up_count/total_count*100:.1f}%)")
    print(f"   DOWN: {down_count:,} ({down_count/total_count*100:.1f}%)")
    print(f"   NEUTRAL: {neutral_count:,} ({neutral_count/total_count*100:.1f}%)")

    # 큰 수익 샘플 통계
    big_moves_up = sum(1 for s in samples_by_direction['UP']
                       if abs(s.actual_price_change) > profit_threshold)
    big_moves_down = sum(1 for s in samples_by_direction['DOWN']
                         if abs(s.actual_price_change) > profit_threshold)
    print(f"\n📈 큰 변동 (>{profit_threshold}%) 샘플:")
    print(f"   UP: {big_moves_up:,}개")
    print(f"   DOWN: {big_moves_down:,}개")

    # 클래스 밸런싱 (수익률 가중)
    if balance_classes:
        min_count = min(up_count, down_count, neutral_count)
        max_count = max(up_count, down_count, neutral_count)
        target_count = max(min_count * 2, (min_count + max_count) // 2)
        target_count = min(target_count, max_count)

        print(f"\n⚖️ 클래스 밸런싱 (타겟: {target_count:,}개/클래스)")

        balanced_samples = []
        sample_weights = []

        for direction, samples_list in samples_by_direction.items():
            # 수익률 크기로 정렬 (큰 것 우선)
            sorted_samples = sorted(
                samples_list,
                key=lambda s: abs(s.actual_price_change),
                reverse=True
            )

            if len(sorted_samples) >= target_count:
                # 상위 50%는 그대로, 나머지는 수익률 가중 샘플링
                top_half = target_count // 2
                bottom_half = target_count - top_half

                selected = sorted_samples[:top_half]
                remaining = sorted_samples[top_half:]

                if remaining:
                    # 수익률 기반 확률로 샘플링
                    probs = np.array([abs(s.actual_price_change) + 0.1 for s in remaining])
                    probs = probs / probs.sum()
                    indices = np.random.choice(
                        len(remaining),
                        size=min(bottom_half, len(remaining)),
                        replace=False,
                        p=probs
                    )
                    selected.extend([remaining[i] for i in indices])
            else:
                # 오버샘플링 (큰 수익 샘플 더 많이)
                selected = samples_list.copy()
                while len(selected) < target_count:
                    probs = np.array([abs(s.actual_price_change) + 0.1 for s in samples_list])
                    probs = probs / probs.sum()
                    idx = np.random.choice(len(samples_list), p=probs)
                    selected.append(samples_list[idx])

            # 샘플 가중치 계산 (수익률 기반)
            for s in selected:
                weight = (1.0 + abs(s.actual_price_change)) ** 1.2
                sample_weights.append(weight)

            balanced_samples.extend(selected)
            print(f"   {direction}: {len(samples_list):,} → {len(selected):,}")

        # 셔플 (가중치와 함께)
        combined = list(zip(balanced_samples, sample_weights))
        random.shuffle(combined)
        balanced_samples, sample_weights = zip(*combined)

        # 가중치 정규화
        sample_weights = np.array(sample_weights)
        sample_weights = sample_weights / sample_weights.mean()

        print(f"\n✅ 샘플 생성 완료: {len(balanced_samples):,}개")
        print(f"   평균 가중치: {sample_weights.mean():.2f}")
        print(f"   최대 가중치: {sample_weights.max():.2f}")

        return list(balanced_samples), sample_weights

    else:
        all_samples = []
        for samples_list in samples_by_direction.values():
            all_samples.extend(samples_list)
        random.shuffle(all_samples)

        # 균일 가중치
        sample_weights = np.ones(len(all_samples))

        return all_samples, sample_weights


def train_profit_weighted_model(
    symbol: str = 'BTCUSDT',
    interval: str = '1h',
    days: int = None,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 5e-5,
    validation_split: float = 0.2,
    save_path: str = None,
    profit_power: float = 1.2,
    loss_penalty: float = 1.5
) -> Optional[Dict]:
    """
    수익률 가중 AI 모델 학습
    """
    # 간격별 최적 설정
    settings = {
        '1m': {'days': 14, 'threshold': 0.05, 'seq_len': 60, 'profit_threshold': 0.3},
        '3m': {'days': 21, 'threshold': 0.08, 'seq_len': 60, 'profit_threshold': 0.4},
        '5m': {'days': 30, 'threshold': 0.1, 'seq_len': 60, 'profit_threshold': 0.5},
        '15m': {'days': 60, 'threshold': 0.15, 'seq_len': 48, 'profit_threshold': 0.7},
        '1h': {'days': 180, 'threshold': 0.3, 'seq_len': 48, 'profit_threshold': 1.0},
        '4h': {'days': 365, 'threshold': 0.5, 'seq_len': 36, 'profit_threshold': 1.5},
    }

    opt = settings.get(interval, settings['1h'])
    days = days or opt['days']
    threshold_pct = opt['threshold']
    sequence_length = opt['seq_len']
    profit_threshold = opt['profit_threshold']

    print()
    print("=" * 70)
    print("🧠 수익률 가중 AI 학습 (Profit-Weighted Training)")
    print("=" * 70)
    print(f"  심볼: {symbol}")
    print(f"  간격: {interval}")
    print(f"  학습 기간: {days}일")
    print(f"  에폭: {epochs}")
    print(f"  배치 크기: {batch_size}")
    print(f"  학습률: {learning_rate}")
    print(f"  수익률 가중치 지수: {profit_power}")
    print(f"  손실 페널티: {loss_penalty}x")
    print("=" * 70)

    # 1. 데이터 수집
    df = fetch_historical_data(symbol, interval, days, testnet=True)

    if len(df) < 200:
        print("❌ 데이터가 부족합니다.")
        return None

    # 2. AI 분석기 초기화
    print("\n🔧 AI 분석기 초기화...")
    market_analyzer = MarketAnalyzer(
        sequence_length=sequence_length,
        hidden_size=128,
        num_layers=2,
        dropout=0.3  # 과적합 방지 강화
    )
    market_analyzer.set_timeframe(interval)

    # 3. 수익률 가중 샘플 생성
    samples, sample_weights = prepare_profit_weighted_samples(
        df, market_analyzer,
        threshold_pct=threshold_pct,
        profit_threshold=profit_threshold,
        balance_classes=True
    )

    if len(samples) < 100:
        print("❌ 학습 샘플이 부족합니다.")
        return None

    # 4. 학습/검증 분할
    split_idx = int(len(samples) * (1 - validation_split))
    train_samples = samples[:split_idx]
    train_weights = sample_weights[:split_idx]
    val_samples = samples[split_idx:]
    val_weights = sample_weights[split_idx:]

    print(f"\n📊 데이터 분할:")
    print(f"   학습: {len(train_samples):,}개")
    print(f"   검증: {len(val_samples):,}개")

    # 5. 학습 데이터 준비 (2-class direction + confidence)
    direction_map = {'UP': 1.0, 'NEUTRAL': 0.5, 'DOWN': 0.0}
    confidence_map = {'UP': 1.0, 'NEUTRAL': 0.0, 'DOWN': 1.0}
    direction_map_metrics = {'UP': 0, 'NEUTRAL': 1, 'DOWN': 2}  # 메트릭 계산용

    train_features = np.array([s.features for s in train_samples])
    train_directions = np.array([direction_map[s.actual_direction] for s in train_samples], dtype=np.float32)
    train_confidences = np.array([confidence_map[s.actual_direction] for s in train_samples], dtype=np.float32)
    train_returns = np.array([s.actual_price_change for s in train_samples])

    X_train = torch.FloatTensor(train_features)
    y_dir = torch.FloatTensor(train_directions)
    y_conf = torch.FloatTensor(train_confidences)
    y_ret = torch.FloatTensor(train_returns)

    # Weighted Random Sampler (수익률 가중)
    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(train_weights),
        num_samples=len(train_samples),
        replacement=True
    )

    train_dataset = TensorDataset(X_train, y_dir, y_conf, y_ret)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    # 검증 데이터
    val_features = np.array([s.features for s in val_samples])
    val_directions = np.array([direction_map[s.actual_direction] for s in val_samples], dtype=np.float32)
    val_confidences = np.array([confidence_map[s.actual_direction] for s in val_samples], dtype=np.float32)
    val_returns = np.array([s.actual_price_change for s in val_samples])
    val_directions_metrics = np.array([direction_map_metrics[s.actual_direction] for s in val_samples])

    X_val = torch.FloatTensor(val_features).to(market_analyzer.device)
    y_val_dir = torch.FloatTensor(val_directions).to(market_analyzer.device)
    y_val_conf = torch.FloatTensor(val_confidences).to(market_analyzer.device)
    y_val_ret = torch.FloatTensor(val_returns).to(market_analyzer.device)

    # 6. 손실 함수 및 옵티마이저
    criterion = ProfitWeightedLoss(
        direction_weight=0.4,
        confidence_weight=0.15,
        return_weight=0.35,
        pnl_weight=0.1,
        profit_power=profit_power,
        loss_penalty=loss_penalty
    )

    optimizer = torch.optim.AdamW(
        market_analyzer.model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # 7. 학습 루프
    print("\n🚀 학습 시작...")
    print("-" * 70)

    best_profit_factor = 0.0
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 10

    for epoch in range(epochs):
        # 학습 모드
        market_analyzer.model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_dir, batch_conf, batch_ret in train_loader:
            batch_x = batch_x.to(market_analyzer.device)
            batch_dir = batch_dir.to(market_analyzer.device)
            batch_conf = batch_conf.to(market_analyzer.device)
            batch_ret = batch_ret.to(market_analyzer.device)

            optimizer.zero_grad()

            # Forward (4 returns: price, direction, confidence, attention)
            price_pred, direction_out, confidence_out, _ = market_analyzer.model(batch_x)

            # Profit-Weighted Loss
            loss, metrics = criterion(
                direction_out, confidence_out, price_pred,
                batch_dir, batch_conf, batch_ret
            )

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(market_analyzer.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += metrics['total_loss']
            # Edge 기반 정확도
            edge = (direction_out.squeeze() - 0.5)
            pred_up = (edge > 0.025).long()
            actual_up = (batch_dir > 0.75).long()
            train_correct += (pred_up == actual_up).sum().item()
            train_total += len(batch_dir)

        scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # 검증 모드
        market_analyzer.model.eval()
        with torch.no_grad():
            _, val_dir_out, val_conf_out, _ = market_analyzer.model(X_val)

            # Edge*confidence 기반 예측
            val_edge = (val_dir_out.squeeze() - 0.5).cpu().numpy()
            val_conf_raw = val_conf_out.squeeze().cpu().numpy()
            # 기존 형식 변환 (0=UP, 1=NEUTRAL, 2=DOWN)
            val_pred = np.where(val_edge > 0.025, 0, np.where(val_edge < -0.025, 2, 1))
            val_conf_values = np.clip(np.abs(val_edge) * val_conf_raw * 2, 0, 1)

            metrics = calculate_profit_metrics(
                predictions=val_pred.tolist(),
                actuals=val_directions_metrics.tolist(),
                returns=val_returns.tolist(),
                confidences=val_conf_values.tolist(),
                leverage=3.0
            )

        current_lr = optimizer.param_groups[0]['lr']

        print(f"   에폭 {epoch+1:2d}/{epochs}: "
              f"Loss={avg_train_loss:.4f}, "
              f"Acc={metrics.accuracy:.1%}, "
              f"PF={metrics.profit_factor:.2f}, "
              f"Kelly={metrics.kelly_optimal:.2f}, "
              f"LR={current_lr:.2e}")

        # Profit Factor 기준 최고 모델 저장
        if metrics.profit_factor > best_profit_factor and metrics.accuracy > 0.45:
            best_profit_factor = metrics.profit_factor
            best_accuracy = metrics.accuracy
            best_model_state = {
                k: v.cpu().clone()
                for k, v in market_analyzer.model.state_dict().items()
            }
            patience_counter = 0
            print(f"         ✅ 최고 모델 갱신! (PF={best_profit_factor:.2f}, Acc={best_accuracy:.1%})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n⏹️ 조기 종료 (patience={early_stop_patience})")
                break

    print("-" * 70)

    # 8. 최고 모델 복원 및 저장
    if best_model_state:
        market_analyzer.model.load_state_dict(best_model_state)
        market_analyzer.model.to(market_analyzer.device)

    if save_path is None:
        save_path = f"models/transformer_{symbol}_{interval}_profit.pt"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    market_analyzer.save_model(save_path)

    print(f"\n✅ 학습 완료!")
    print(f"   모델 저장: {save_path}")

    # 9. 최종 평가
    print("\n📊 최종 모델 평가:")
    market_analyzer.model.eval()

    with torch.no_grad():
        _, val_dir_out, val_conf_out, _ = market_analyzer.model(X_val)
        val_edge = (val_dir_out.squeeze() - 0.5).cpu().numpy()
        val_conf_raw = val_conf_out.squeeze().cpu().numpy()
        val_pred = np.where(val_edge > 0.025, 0, np.where(val_edge < -0.025, 2, 1))
        val_conf_values = np.clip(np.abs(val_edge) * val_conf_raw * 2, 0, 1)

        final_metrics = calculate_profit_metrics(
            predictions=val_pred.tolist(),
            actuals=val_directions_metrics.tolist(),
            returns=val_returns.tolist(),
            confidences=val_conf_values.tolist(),
            leverage=3.0
        )

    print(f"   방향 정확도: {final_metrics.accuracy:.1%}")
    print(f"   Profit Factor: {final_metrics.profit_factor:.2f}")
    print(f"   총 수익률: {final_metrics.total_return:.1f}%")
    print(f"   승률: {final_metrics.win_rate:.1%}")
    print(f"   평균 수익: {final_metrics.avg_win:.2f}%")
    print(f"   평균 손실: {final_metrics.avg_loss:.2f}%")
    print(f"   Kelly 최적: {final_metrics.kelly_optimal:.2f}")
    print(f"   Sharpe Ratio: {final_metrics.sharpe_ratio:.2f}")

    # 65% 목표 달성 여부
    if final_metrics.accuracy >= 0.65:
        print("\n🎉 목표 달성! 65% 이상 정확도!")
    elif final_metrics.profit_factor >= 1.5:
        print("\n🎉 Profit Factor 1.5+ 달성! 수익성 있는 모델!")
    else:
        print(f"\n⚠️ 추가 개선 필요 (목표: Acc≥65% 또는 PF≥1.5)")

    return {
        'model_path': save_path,
        'accuracy': final_metrics.accuracy,
        'profit_factor': final_metrics.profit_factor,
        'total_return': final_metrics.total_return,
        'kelly_optimal': final_metrics.kelly_optimal,
        'sharpe_ratio': final_metrics.sharpe_ratio,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples)
    }


def train_all_intervals_profit(
    symbol: str = 'BTCUSDT',
    intervals: List[str] = None
) -> Dict:
    """여러 간격 모델 학습 (수익률 가중)"""
    if intervals is None:
        intervals = ['5m', '15m', '1h']

    print("\n" + "=" * 70)
    print("🧠 다중 시간대 수익률 가중 AI 학습")
    print("=" * 70)

    results = {}

    for interval in intervals:
        print(f"\n\n{'='*70}")
        print(f"📊 {interval} 모델 학습 시작")
        print("=" * 70)

        result = train_profit_weighted_model(
            symbol=symbol,
            interval=interval
        )

        if result:
            results[interval] = result
            print(f"\n✅ {interval} 완료: Acc={result['accuracy']:.1%}, PF={result['profit_factor']:.2f}")
        else:
            print(f"\n❌ {interval} 학습 실패")

    # 최종 요약
    print("\n\n" + "=" * 70)
    print("📋 학습 결과 요약 (수익률 가중)")
    print("=" * 70)

    for interval, result in results.items():
        status = "🎉" if result['accuracy'] >= 0.65 or result['profit_factor'] >= 1.5 else "⚠️"
        print(f"  {status} {interval:5s}: Acc={result['accuracy']:.1%}, "
              f"PF={result['profit_factor']:.2f}, "
              f"Kelly={result['kelly_optimal']:.2f}")

    print("=" * 70)

    return results


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='수익률 가중 AI 학습')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--interval', type=str, default='1h')
    parser.add_argument('--days', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--profit-power', type=float, default=1.2)
    parser.add_argument('--loss-penalty', type=float, default=1.5)
    parser.add_argument('--all', action='store_true', help='모든 간격 학습')
    parser.add_argument('--intervals', type=str, default=None)

    args = parser.parse_args()

    if args.all:
        train_all_intervals_profit(symbol=args.symbol)
        return

    if args.intervals:
        intervals = [i.strip() for i in args.intervals.split(',')]
        train_all_intervals_profit(symbol=args.symbol, intervals=intervals)
        return

    train_profit_weighted_model(
        symbol=args.symbol,
        interval=args.interval,
        days=args.days,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        profit_power=args.profit_power,
        loss_penalty=args.loss_penalty
    )


if __name__ == '__main__':
    main()
