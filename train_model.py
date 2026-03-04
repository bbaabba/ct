"""AI 기초 모델 학습 스크립트"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import os
import time
import random
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta

from api.futures_client import BinanceFuturesClient
from ai.market_analyzer import MarketAnalyzer, TrainingSample, TickAnalyzer
from utils.logger import setup_logger

logger = setup_logger(__name__)


def fetch_historical_data(
    symbol: str = 'BTCUSDT',
    interval: str = '1h',
    days: int = 90,
    testnet: bool = True
) -> pd.DataFrame:
    """
    바이낸스에서 히스토리컬 데이터 수집

    Args:
        symbol: 거래 심볼
        interval: 캔들 간격
        days: 수집할 일수
        testnet: 테스트넷 사용 여부

    Returns:
        OHLCV 데이터프레임
    """
    print(f"\n📊 히스토리컬 데이터 수집 중...")
    print(f"   심볼: {symbol}")
    print(f"   간격: {interval}")
    print(f"   기간: {days}일")

    client = BinanceFuturesClient(testnet=testnet)

    # 간격별 캔들 수 계산
    interval_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440
    }

    minutes_per_interval = interval_minutes.get(interval, 60)
    candles_per_day = 1440 // minutes_per_interval
    total_candles = candles_per_day * days

    print(f"   예상 캔들 수: {total_candles:,}개")

    all_klines = []
    batch_size = 1000  # 바이낸스 API 최대 limit

    # 현재 시간부터 과거로 수집
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

            # 다음 배치의 끝 시간 설정
            end_time = klines[0][0] - 1

            print(f"   수집 중... {collected:,}/{total_candles:,} ({collected/total_candles*100:.1f}%)")

            time.sleep(0.1)  # API 레이트 리밋 방지

        except Exception as e:
            logger.warning(f"데이터 수집 중 오류: {e}")
            break

    # 데이터프레임 변환
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


def prepare_training_samples(
    df: pd.DataFrame,
    market_analyzer: MarketAnalyzer,
    lookahead: int = 1,
    threshold_pct: float = 0.1,
    balance_classes: bool = True
) -> list:
    """
    학습용 샘플 생성 (클래스 밸런싱 지원)

    Args:
        df: OHLCV 데이터프레임
        market_analyzer: AI 분석기
        lookahead: 예측할 미래 캔들 수
        threshold_pct: 방향 결정 임계값 (%)
        balance_classes: 클래스 밸런싱 여부

    Returns:
        TrainingSample 리스트
    """
    print(f"\n🔧 학습 샘플 생성 중...")
    print(f"   방향 임계값: ±{threshold_pct}%")
    print(f"   클래스 밸런싱: {'예' if balance_classes else '아니오'}")

    samples_by_direction = {'UP': [], 'DOWN': [], 'NEUTRAL': []}
    sequence_length = market_analyzer.sequence_length
    min_data = sequence_length + 100

    # 전체 특징 추출
    features = market_analyzer.prepare_features(df)

    # 슬라이딩 윈도우로 샘플 생성
    total_samples = len(df) - min_data - lookahead

    for i in range(min_data, len(df) - lookahead):
        try:
            # 시퀀스 추출
            sequence = features[i - sequence_length:i].copy()

            # 실제 가격 변화 계산
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

            total_created = sum(len(v) for v in samples_by_direction.values())
            if total_created % 2000 == 0:
                print(f"   생성 중... {total_created:,}/{total_samples:,}")

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

    # 클래스 밸런싱
    if balance_classes:
        # 소수 클래스 기준으로 언더샘플링 + 오버샘플링 조합
        min_count = min(up_count, down_count, neutral_count)
        max_count = max(up_count, down_count, neutral_count)

        # 타겟 샘플 수: 중간값 사용 (너무 적지도, 너무 많지도 않게)
        target_count = max(min_count * 2, (min_count + max_count) // 2)
        target_count = min(target_count, max_count)  # 최대값 초과 방지

        print(f"\n⚖️ 클래스 밸런싱 (타겟: {target_count:,}개/클래스)")

        balanced_samples = []
        for direction, samples_list in samples_by_direction.items():
            if len(samples_list) >= target_count:
                # 언더샘플링: 랜덤 선택
                selected = random.sample(samples_list, target_count)
            else:
                # 오버샘플링: 복제
                selected = samples_list.copy()
                while len(selected) < target_count:
                    selected.extend(random.sample(
                        samples_list,
                        min(len(samples_list), target_count - len(selected))
                    ))
            balanced_samples.extend(selected)
            print(f"   {direction}: {len(samples_list):,} → {len(selected):,}")

        # 셔플
        random.shuffle(balanced_samples)

        print(f"\n✅ 밸런싱 완료: {len(balanced_samples):,}개")
        return balanced_samples
    else:
        # 밸런싱 없이 모든 샘플 반환
        all_samples = []
        for samples_list in samples_by_direction.values():
            all_samples.extend(samples_list)

        random.shuffle(all_samples)

        print(f"\n✅ 샘플 생성 완료: {len(all_samples):,}개")
        return all_samples


def get_optimal_settings(interval: str) -> dict:
    """
    간격별 최적 학습 설정 반환

    Args:
        interval: 캔들 간격

    Returns:
        최적 설정 딕셔너리
    """
    settings = {
        '1m': {
            'days': 14,  # 1분봉은 데이터가 많으므로 14일이면 충분
            'threshold_pct': 0.05,  # 작은 변동에도 반응
            'lookahead': 1,
            'sequence_length': 60,  # 60분 = 1시간 패턴
            'batch_size': 128,
            'epochs': 15,
            'learning_rate': 5e-5,
        },
        '3m': {
            'days': 21,
            'threshold_pct': 0.08,
            'lookahead': 1,
            'sequence_length': 60,
            'batch_size': 128,
            'epochs': 15,
            'learning_rate': 5e-5,
        },
        '5m': {
            'days': 30,
            'threshold_pct': 0.1,
            'lookahead': 1,
            'sequence_length': 60,
            'batch_size': 64,
            'epochs': 20,
            'learning_rate': 1e-4,
        },
        '15m': {
            'days': 60,
            'threshold_pct': 0.15,
            'lookahead': 1,
            'sequence_length': 48,  # 12시간 패턴
            'batch_size': 64,
            'epochs': 20,
            'learning_rate': 1e-4,
        },
        '1h': {
            'days': 180,  # 6개월
            'threshold_pct': 0.3,
            'lookahead': 1,
            'sequence_length': 48,  # 48시간 = 2일 패턴
            'batch_size': 32,
            'epochs': 30,
            'learning_rate': 1e-4,
        },
        '4h': {
            'days': 365,  # 1년
            'threshold_pct': 0.5,
            'lookahead': 1,
            'sequence_length': 36,  # 6일 패턴
            'batch_size': 32,
            'epochs': 30,
            'learning_rate': 1e-4,
        },
    }

    return settings.get(interval, settings['1h'])


def train_base_model(
    symbol: str = 'BTCUSDT',
    interval: str = '1h',
    days: int = None,  # None이면 간격별 최적값 사용
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    validation_split: float = 0.2,
    save_path: str = None,
    balance_classes: bool = True,
    threshold_pct: float = None
):
    """
    기초 모델 학습 (개선된 버전)

    Args:
        symbol: 거래 심볼
        interval: 캔들 간격
        days: 학습 데이터 일수 (None이면 자동 설정)
        epochs: 학습 에폭 수 (None이면 자동 설정)
        batch_size: 배치 크기 (None이면 자동 설정)
        learning_rate: 학습률 (None이면 자동 설정)
        validation_split: 검증 데이터 비율
        save_path: 모델 저장 경로
        balance_classes: 클래스 밸런싱 여부
        threshold_pct: 방향 결정 임계값 (None이면 자동 설정)
    """
    # 간격별 최적 설정 가져오기
    optimal = get_optimal_settings(interval)

    # None인 파라미터는 최적값으로 대체
    days = days or optimal['days']
    epochs = epochs or optimal['epochs']
    batch_size = batch_size or optimal['batch_size']
    learning_rate = learning_rate or optimal['learning_rate']
    threshold_pct = threshold_pct or optimal['threshold_pct']
    sequence_length = optimal['sequence_length']
    lookahead = optimal['lookahead']

    print()
    print("=" * 70)
    print("🧠 AI 기초 모델 학습 (개선된 버전)")
    print("=" * 70)
    print(f"  심볼: {symbol}")
    print(f"  간격: {interval}")
    print(f"  학습 기간: {days}일")
    print(f"  에폭: {epochs}")
    print(f"  배치 크기: {batch_size}")
    print(f"  학습률: {learning_rate}")
    print(f"  시퀀스 길이: {sequence_length}")
    print(f"  방향 임계값: ±{threshold_pct}%")
    print(f"  클래스 밸런싱: {'예' if balance_classes else '아니오'}")
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
        dropout=0.4
    )
    market_analyzer.set_timeframe(interval)

    # 3. 학습 샘플 생성 (클래스 밸런싱 적용)
    samples = prepare_training_samples(
        df, market_analyzer,
        lookahead=lookahead,
        threshold_pct=threshold_pct,
        balance_classes=balance_classes
    )

    if len(samples) < 100:
        print("❌ 학습 샘플이 부족합니다.")
        return None

    # 4. 학습/검증 분할
    split_idx = int(len(samples) * (1 - validation_split))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"\n📊 데이터 분할:")
    print(f"   학습: {len(train_samples):,}개")
    print(f"   검증: {len(val_samples):,}개")

    # 검증 세트 클래스 분포 확인
    val_directions = [s.actual_direction for s in val_samples]
    print(f"   검증 분포: UP={val_directions.count('UP')}, "
          f"DOWN={val_directions.count('DOWN')}, "
          f"NEUTRAL={val_directions.count('NEUTRAL')}")

    # 5. 직접 학습 실행 (버퍼 제한 없음)
    print("\n🚀 학습 시작...")
    print("-" * 70)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # 학습 데이터 준비 (2-class: UP=1, DOWN=0, NEUTRAL=0.5 + confidence)
    train_features = np.array([s.features for s in train_samples])
    train_dir_targets = np.array([
        1.0 if s.actual_direction == 'UP' else (0.0 if s.actual_direction == 'DOWN' else 0.5)
        for s in train_samples
    ])
    train_conf_targets = np.array([
        0.0 if s.actual_direction == 'NEUTRAL' else 1.0
        for s in train_samples
    ])
    train_price_changes = np.array([s.actual_price_change for s in train_samples])

    X_train = torch.FloatTensor(train_features)
    y_direction = torch.FloatTensor(train_dir_targets).unsqueeze(1)
    y_confidence = torch.FloatTensor(train_conf_targets).unsqueeze(1)
    y_price = torch.FloatTensor(train_price_changes)

    train_dataset = TensorDataset(X_train, y_direction, y_confidence, y_price)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 검증 데이터 준비
    val_features = np.array([s.features for s in val_samples])
    val_dir_targets = np.array([
        1.0 if s.actual_direction == 'UP' else (0.0 if s.actual_direction == 'DOWN' else 0.5)
        for s in val_samples
    ])
    val_conf_targets = np.array([
        0.0 if s.actual_direction == 'NEUTRAL' else 1.0
        for s in val_samples
    ])

    X_val = torch.FloatTensor(val_features).to(market_analyzer.device)
    y_val_dir = torch.FloatTensor(val_dir_targets).to(market_analyzer.device)
    y_val_conf = torch.FloatTensor(val_conf_targets).to(market_analyzer.device)

    # 옵티마이저 및 손실 함수 (2-class BCE + confidence MSE)
    optimizer = torch.optim.AdamW(market_analyzer.model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    dir_criterion = nn.BCELoss()
    conf_criterion = nn.MSELoss()
    price_criterion = nn.MSELoss()

    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 7

    for epoch in range(epochs):
        # 학습 모드
        market_analyzer.model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_dir, batch_conf, batch_price in train_loader:
            batch_x = batch_x.to(market_analyzer.device)
            batch_dir = batch_dir.to(market_analyzer.device)
            batch_conf = batch_conf.to(market_analyzer.device)
            batch_price = batch_price.to(market_analyzer.device)

            optimizer.zero_grad()

            # Forward (2-class direction + confidence)
            price_pred, dir_out, conf_out, _ = market_analyzer.model(batch_x)

            # Loss 계산
            dir_loss = dir_criterion(dir_out, batch_dir)
            c_loss = conf_criterion(conf_out, batch_conf)
            price_loss = price_criterion(price_pred.squeeze(), batch_price)
            loss = dir_loss + c_loss * 0.5 + price_loss * 0.1

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(market_analyzer.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            # edge * confidence 기반 accuracy
            edge = dir_out.squeeze() - 0.5
            score = edge * conf_out.squeeze() * 2
            pred_up = (score > 0.05).float()
            pred_down = (score < -0.05).float()
            actual_up = (batch_dir.squeeze() > 0.7).float()
            actual_down = (batch_dir.squeeze() < 0.3).float()
            train_correct += (pred_up * actual_up + pred_down * actual_down).sum().item()
            train_total += len(batch_dir)

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # 검증 모드
        market_analyzer.model.eval()
        with torch.no_grad():
            _, val_dir, val_conf, _ = market_analyzer.model(X_val)
            val_edge = val_dir.squeeze() - 0.5
            val_score = val_edge * val_conf.squeeze() * 2
            val_pred_up = (val_score > 0.05).float()
            val_pred_down = (val_score < -0.05).float()
            val_actual_up = (y_val_dir > 0.7).float()
            val_actual_down = (y_val_dir < 0.3).float()
            val_correct = (val_pred_up * val_actual_up + val_pred_down * val_actual_down).sum().item()
            val_accuracy = val_correct / len(y_val_dir)
            val_loss = dir_criterion(val_dir.squeeze(), y_val_dir).item()

        # 학습률 스케줄러
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"   에폭 {epoch+1:2d}/{epochs}: "
              f"Loss={avg_train_loss:.4f}, "
              f"Train Acc={train_accuracy:.1%}, "
              f"Val Acc={val_accuracy:.1%}, "
              f"LR={current_lr:.2e}")

        # 최고 모델 저장 (검증 정확도 기준)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in market_analyzer.model.state_dict().items()}
            patience_counter = 0
            print(f"         ✅ 최고 모델 갱신!")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n⏹️ 조기 종료 (patience={early_stop_patience})")
                break

    print("-" * 70)
    print(f"\n✅ 학습 완료!")
    print(f"   최고 검증 정확도: {best_val_accuracy:.1%}")

    # 6. 최고 모델 복원 및 저장
    if best_model_state:
        market_analyzer.model.load_state_dict(best_model_state)
        market_analyzer.model.to(market_analyzer.device)

    if save_path is None:
        save_path = f"models/transformer_{symbol}_{interval}_base.pt"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    market_analyzer.save_model(save_path)

    print(f"   모델 저장: {save_path}")

    # 7. 최종 평가
    print("\n📊 최종 모델 평가:")

    # 검증 세트로 최종 평가
    market_analyzer.model.eval()
    final_correct = 0
    predictions = {'UP': 0, 'DOWN': 0, 'NEUTRAL': 0}
    actuals = {'UP': 0, 'DOWN': 0, 'NEUTRAL': 0}

    with torch.no_grad():
        for sample in val_samples:
            x = torch.FloatTensor(sample.features).unsqueeze(0).to(market_analyzer.device)
            _, dir_out, conf_out, _ = market_analyzer.model(x)
            edge = dir_out.item() - 0.5
            score = edge * conf_out.item() * 2
            if score > 0.05:
                predicted_dir = 'UP'
            elif score < -0.05:
                predicted_dir = 'DOWN'
            else:
                predicted_dir = 'NEUTRAL'

            predictions[predicted_dir] += 1
            actuals[sample.actual_direction] += 1

            if predicted_dir == sample.actual_direction:
                final_correct += 1

    final_accuracy = final_correct / len(val_samples)

    print(f"   최종 정확도: {final_accuracy:.1%}")
    print(f"   예측 분포: UP={predictions['UP']}, DOWN={predictions['DOWN']}, NEUTRAL={predictions['NEUTRAL']}")
    print(f"   실제 분포: UP={actuals['UP']}, DOWN={actuals['DOWN']}, NEUTRAL={actuals['NEUTRAL']}")

    return {
        'model_path': save_path,
        'best_val_accuracy': best_val_accuracy,
        'final_accuracy': final_accuracy,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'epochs': epochs
    }


def train_all_intervals(symbol: str = 'BTCUSDT', intervals: list = None):
    """
    여러 간격의 모델을 한번에 학습

    Args:
        symbol: 거래 심볼
        intervals: 학습할 간격 리스트
    """
    if intervals is None:
        intervals = ['1m', '5m', '15m', '1h']

    print("\n" + "=" * 70)
    print("🧠 다중 시간대 AI 모델 학습")
    print("=" * 70)
    print(f"  심볼: {symbol}")
    print(f"  학습 간격: {', '.join(intervals)}")
    print("=" * 70)

    results = {}

    for interval in intervals:
        print(f"\n\n{'='*70}")
        print(f"📊 {interval} 모델 학습 시작")
        print("=" * 70)

        result = train_base_model(
            symbol=symbol,
            interval=interval,
            balance_classes=True
        )

        if result:
            results[interval] = result
            print(f"\n✅ {interval} 모델 학습 완료: {result['final_accuracy']:.1%}")
        else:
            print(f"\n❌ {interval} 모델 학습 실패")

    # 최종 요약
    print("\n\n" + "=" * 70)
    print("📋 학습 결과 요약")
    print("=" * 70)

    for interval, result in results.items():
        print(f"  {interval:5s}: {result['final_accuracy']:.1%} "
              f"(학습: {result['train_samples']:,}, 검증: {result['val_samples']:,})")

    print("=" * 70)

    return results


def generate_synthetic_tick_data(
    df_1m: pd.DataFrame,
    ticks_per_candle: int = 6
) -> pd.DataFrame:
    """
    1분봉 데이터에서 합성 10초 마이크로캔들 생성.

    각 1분봉을 6개의 10초 바로 분할하여 근사적 가격 경로 생성.

    Args:
        df_1m: 1분봉 OHLCV DataFrame
        ticks_per_candle: 1분봉당 마이크로캔들 수 (기본 6)

    Returns:
        합성 10초 마이크로캔들 DataFrame
    """
    print(f"\n🔄 합성 10초 데이터 생성 중 ({len(df_1m)}개 1분봉 → ~{len(df_1m)*ticks_per_candle}개 10초봉)...")

    micro_candles = []

    for i in range(len(df_1m)):
        row = df_1m.iloc[i]
        o, h, l, c, v = row['open'], row['high'], row['low'], row['close'], row['volume']

        # 1분봉의 타임스탬프 (인덱스)
        if hasattr(df_1m.index[i], 'timestamp'):
            base_ts = int(df_1m.index[i].timestamp() * 1000)
        else:
            base_ts = int(pd.Timestamp(df_1m.index[i]).timestamp() * 1000)

        # 가격 경로 생성: open → peak/trough → close
        # 간단한 접근: 선형 보간 + 고점/저점 삽입
        prices = [o]

        # 고점/저점 중 어느 것이 먼저 오는지 결정
        # 종가가 시가보다 높으면: 먼저 하락 후 상승 (보수적) 또는 그 반대
        mid_count = ticks_per_candle - 1

        if mid_count > 1:
            # 심플한 접근: open → (low or high) → (high or low) → close
            if c >= o:
                # 상승 캔들: open → low 근처 → high 근처 → close
                key_prices = [o, l, h, c]
            else:
                # 하락 캔들: open → high 근처 → low 근처 → close
                key_prices = [o, h, l, c]

            # key_prices를 ticks_per_candle 포인트로 보간
            x_key = np.linspace(0, 1, len(key_prices))
            x_interp = np.linspace(0, 1, ticks_per_candle + 1)  # +1 because we need boundaries
            interp_prices = np.interp(x_interp, x_key, key_prices)

            # 약간의 랜덤 노이즈 추가 (현실감)
            noise_scale = (h - l) * 0.02 if h > l else 0.0001
            noise = np.random.normal(0, noise_scale, len(interp_prices))
            noise[0] = 0  # 시가 보존
            noise[-1] = 0  # 종가 보존
            interp_prices = interp_prices + noise

            # 고저 범위 내로 클리핑
            interp_prices = np.clip(interp_prices, l * 0.999, h * 1.001)
        else:
            interp_prices = np.linspace(o, c, ticks_per_candle + 1)

        # 마이크로캔들 생성
        sub_volume = max(v / ticks_per_candle, 1.0)

        for j in range(ticks_per_candle):
            ts = base_ts + j * 10000  # 10초 간격 (밀리초)
            sub_open = interp_prices[j]
            sub_close = interp_prices[j + 1]
            sub_high = max(sub_open, sub_close) * (1 + abs(np.random.normal(0, 0.0001)))
            sub_low = min(sub_open, sub_close) * (1 - abs(np.random.normal(0, 0.0001)))

            # 원본 캔들 범위 내로 제한
            sub_high = min(sub_high, h * 1.001)
            sub_low = max(sub_low, l * 0.999)

            micro_candles.append({
                'timestamp': ts,
                'open': sub_open,
                'high': sub_high,
                'low': sub_low,
                'close': sub_close,
                'volume': sub_volume + np.random.uniform(-sub_volume*0.3, sub_volume*0.3),
            })

    df_tick = pd.DataFrame(micro_candles)
    df_tick['timestamp'] = pd.to_datetime(df_tick['timestamp'], unit='ms')
    df_tick.set_index('timestamp', inplace=True)

    # volume을 양수로 보정
    df_tick['volume'] = df_tick['volume'].clip(lower=1.0)

    print(f"✅ 합성 10초 데이터 생성 완료: {len(df_tick)}개 마이크로캔들")
    print(f"   기간: {df_tick.index[0]} ~ {df_tick.index[-1]}")

    return df_tick


def prepare_tick_training_samples(
    df_tick: pd.DataFrame,
    tick_analyzer: TickAnalyzer,
    lookahead: int = 1,
    threshold_pct: float = 0.01,
    balance_classes: bool = True
) -> list:
    """
    틱 학습 샘플 생성.

    Args:
        df_tick: 10초 마이크로캔들 DataFrame
        tick_analyzer: TickAnalyzer
        lookahead: 예측 대상 바 수
        threshold_pct: 방향 결정 임계값 (%)
        balance_classes: 클래스 밸런싱

    Returns:
        학습 샘플 리스트
    """
    print(f"\n🔧 틱 학습 샘플 생성 중...")
    print(f"   임계값: ±{threshold_pct}%, lookahead: {lookahead}")

    sequence_length = tick_analyzer.sequence_length
    min_data = sequence_length + 20
    features = tick_analyzer.prepare_features(df_tick)

    samples_by_dir = {'UP': [], 'DOWN': [], 'NEUTRAL': []}

    for i in range(min_data, len(df_tick) - lookahead):
        try:
            seq = features[i - sequence_length:i].copy()

            current_price = df_tick['close'].iloc[i]
            future_price = df_tick['close'].iloc[i + lookahead]
            pct_change = (future_price / current_price - 1) * 100

            if pct_change > threshold_pct:
                direction = 'UP'
            elif pct_change < -threshold_pct:
                direction = 'DOWN'
            else:
                direction = 'NEUTRAL'

            # timing_label: 방향과 일치하는 정도 (0.5=중립, 1.0=강한 매수 타이밍)
            if direction == 'UP':
                timing = min(1.0, 0.5 + abs(pct_change) * 10)
            elif direction == 'DOWN':
                timing = min(1.0, 0.5 + abs(pct_change) * 10)
            else:
                timing = 0.3  # 중립일 때는 타이밍 약함

            samples_by_dir[direction].append({
                'features': seq,
                'direction': direction,
                'price_change': pct_change,
                'timing_label': timing,
            })

        except Exception:
            continue

    up_n = len(samples_by_dir['UP'])
    down_n = len(samples_by_dir['DOWN'])
    neut_n = len(samples_by_dir['NEUTRAL'])
    total = up_n + down_n + neut_n

    print(f"\n📊 원본 분포: UP={up_n}, DOWN={down_n}, NEUTRAL={neut_n} (총 {total})")

    if balance_classes and total > 0:
        min_count = min(up_n, down_n, neut_n)
        max_count = max(up_n, down_n, neut_n)
        target = max(min_count * 2, (min_count + max_count) // 2)
        target = min(target, max_count)

        balanced = []
        for d, lst in samples_by_dir.items():
            if len(lst) >= target:
                balanced.extend(random.sample(lst, target))
            else:
                sel = lst.copy()
                while len(sel) < target:
                    sel.extend(random.sample(lst, min(len(lst), target - len(sel))))
                balanced.extend(sel)

        random.shuffle(balanced)
        print(f"⚖️ 밸런싱: {len(balanced)}개 (타겟 {target}/클래스)")
        return balanced
    else:
        all_samples = []
        for lst in samples_by_dir.values():
            all_samples.extend(lst)
        random.shuffle(all_samples)
        return all_samples


def train_tick_model(
    symbol: str = 'BTCUSDT',
    source_interval: str = '1m',
    days: int = 7,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    validation_split: float = 0.2,
    save_path: str = None,
    threshold_pct: float = 0.01
) -> Optional[dict]:
    """
    합성 데이터로 틱 AI 모델 학습.

    1분봉 히스토리컬 데이터를 10초 마이크로캔들로 변환 후 학습.

    Args:
        symbol: 거래 심볼
        source_interval: 소스 캔들 간격
        days: 히스토리컬 데이터 일수
        epochs: 학습 에폭
        batch_size: 배치 크기
        learning_rate: 학습률
        validation_split: 검증 비율
        save_path: 모델 저장 경로
        threshold_pct: 방향 결정 임계값

    Returns:
        학습 결과 dict
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print()
    print("=" * 70)
    print("🧠 틱 AI 모델 학습 (10초 마이크로캔들)")
    print("=" * 70)
    print(f"  심볼: {symbol}")
    print(f"  소스 간격: {source_interval}")
    print(f"  학습 기간: {days}일")
    print(f"  에폭: {epochs}")
    print(f"  배치: {batch_size}")
    print(f"  학습률: {learning_rate}")
    print(f"  방향 임계값: ±{threshold_pct}%")
    print("=" * 70)

    # 1. 1분봉 데이터 수집
    df_1m = fetch_historical_data(symbol, source_interval, days, testnet=True)
    if len(df_1m) < 500:
        print("❌ 데이터 부족")
        return None

    # 2. 합성 10초 데이터 생성
    df_tick = generate_synthetic_tick_data(df_1m)

    # 3. TickAnalyzer 초기화
    tick_analyzer = TickAnalyzer(sequence_length=30, hidden_size=64)

    # 4. 학습 샘플 생성
    samples = prepare_tick_training_samples(
        df_tick, tick_analyzer,
        lookahead=1,
        threshold_pct=threshold_pct,
        balance_classes=True
    )

    if len(samples) < 100:
        print("❌ 학습 샘플 부족")
        return None

    # 5. Train/Val 분할
    split_idx = int(len(samples) * (1 - validation_split))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"\n📊 데이터 분할: 학습 {len(train_samples)}, 검증 {len(val_samples)}")

    direction_map = {'UP': 1.0, 'NEUTRAL': 0.5, 'DOWN': 0.0}
    confidence_map = {'UP': 1.0, 'NEUTRAL': 0.0, 'DOWN': 1.0}

    X_train = torch.FloatTensor(np.array([s['features'] for s in train_samples]))
    y_dir_train = torch.FloatTensor([direction_map[s['direction']] for s in train_samples])
    y_conf_train = torch.FloatTensor([confidence_map[s['direction']] for s in train_samples])
    y_timing_train = torch.FloatTensor([s['timing_label'] for s in train_samples]).unsqueeze(1)

    X_val = torch.FloatTensor(np.array([s['features'] for s in val_samples])).to(tick_analyzer.device)
    y_dir_val = torch.FloatTensor([direction_map[s['direction']] for s in val_samples]).to(tick_analyzer.device)
    y_conf_val = torch.FloatTensor([confidence_map[s['direction']] for s in val_samples]).to(tick_analyzer.device)

    train_dataset = TensorDataset(X_train, y_dir_train, y_conf_train, y_timing_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 6. 학습
    print("\n🚀 학습 시작...")
    print("-" * 70)

    optimizer = torch.optim.AdamW(tick_analyzer.model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    dir_criterion = nn.BCELoss()
    conf_criterion = nn.MSELoss()
    timing_criterion = nn.MSELoss()

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 7

    for epoch in range(epochs):
        tick_analyzer.model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0

        for bx, bd, bc, bt in train_loader:
            bx = bx.to(tick_analyzer.device)
            bd = bd.to(tick_analyzer.device)
            bc = bc.to(tick_analyzer.device)
            bt = bt.to(tick_analyzer.device)

            optimizer.zero_grad()
            timing_pred, dir_out, conf_out, _ = tick_analyzer.model(bx)

            dir_loss = dir_criterion(dir_out.squeeze(), bd)
            conf_loss = conf_criterion(conf_out.squeeze(), bc)
            timing_loss = timing_criterion(timing_pred, bt)
            loss = dir_loss * 0.4 + conf_loss * 0.1 + timing_loss * 0.5
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tick_analyzer.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            # Edge 기반 정확도
            edge = (dir_out.squeeze() - 0.5)
            pred_up = (edge > 0.025).long()
            actual_up = (bd > 0.75).long()
            train_correct += (pred_up == actual_up).sum().item()
            train_total += len(bd)

        avg_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total

        # 검증
        tick_analyzer.model.eval()
        with torch.no_grad():
            _, val_dir_out, _, _ = tick_analyzer.model(X_val)
            val_edge = (val_dir_out.squeeze() - 0.5)
            val_pred_up = (val_edge > 0.025).long()
            val_actual_up = (y_dir_val > 0.75).long()
            val_acc = (val_pred_up == val_actual_up).sum().item() / len(y_dir_val)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f"   에폭 {epoch+1:2d}/{epochs}: "
              f"Loss={avg_loss:.4f}, Train={train_acc:.1%}, Val={val_acc:.1%}, LR={lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in tick_analyzer.model.state_dict().items()}
            patience_counter = 0
            print(f"         ✅ 최고 모델 갱신!")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n⏹️ 조기 종료 (patience={early_stop_patience})")
                break

    print("-" * 70)

    # 7. 최고 모델 복원 및 저장
    if best_model_state:
        tick_analyzer.model.load_state_dict(best_model_state)
        tick_analyzer.model.to(tick_analyzer.device)

    save_path = save_path or f"models/tick_transformer_{symbol}_10s.pt"
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    tick_analyzer.save_model(save_path)

    print(f"\n✅ 틱 모델 학습 완료!")
    print(f"   최고 검증 정확도: {best_val_acc:.1%}")
    print(f"   모델 저장: {save_path}")

    return {
        'model_path': save_path,
        'best_val_accuracy': best_val_acc,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'epochs': epochs
    }


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='AI 기초 모델 학습 (개선된 버전)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='거래 심볼')
    parser.add_argument('--interval', type=str, default='1h', help='캔들 간격 (1m, 5m, 15m, 1h, 4h)')
    parser.add_argument('--days', type=int, default=None, help='학습 데이터 일수 (자동 설정)')
    parser.add_argument('--epochs', type=int, default=None, help='학습 에폭 수 (자동 설정)')
    parser.add_argument('--batch-size', type=int, default=None, help='배치 크기 (자동 설정)')
    parser.add_argument('--lr', type=float, default=None, help='학습률 (자동 설정)')
    parser.add_argument('--threshold', type=float, default=None, help='방향 결정 임계값 %% (자동 설정)')
    parser.add_argument('--no-balance', action='store_true', help='클래스 밸런싱 비활성화')
    parser.add_argument('--all', action='store_true', help='모든 간격 (1m, 5m, 15m, 1h) 학습')
    parser.add_argument('--intervals', type=str, default=None, help='학습할 간격들 (쉼표 구분, 예: 1m,5m,15m)')
    parser.add_argument('--tick', action='store_true', help='10초 틱 AI 모델 학습')
    parser.add_argument('--tick-days', type=int, default=7, help='틱 모델 학습용 1분봉 데이터 일수 (기본: 7)')
    parser.add_argument('--tick-threshold', type=float, default=0.01, help='틱 방향 결정 임계값 %% (기본: 0.01)')

    args = parser.parse_args()

    # 틱 모델 학습
    if args.tick:
        result = train_tick_model(
            symbol=args.symbol,
            days=args.tick_days,
            epochs=args.epochs or 20,
            batch_size=args.batch_size or 64,
            learning_rate=args.lr or 1e-4,
            threshold_pct=args.tick_threshold
        )
        if result:
            print(f"\n🎉 틱 모델 학습 완료! 정확도: {result['best_val_accuracy']:.1%}")
        return

    # 모든 간격 학습
    if args.all:
        results = train_all_intervals(symbol=args.symbol)
        return

    # 특정 간격들 학습
    if args.intervals:
        intervals = [i.strip() for i in args.intervals.split(',')]
        results = train_all_intervals(symbol=args.symbol, intervals=intervals)
        return

    # 단일 간격 학습
    result = train_base_model(
        symbol=args.symbol,
        interval=args.interval,
        days=args.days,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        balance_classes=not args.no_balance,
        threshold_pct=args.threshold
    )

    if result:
        print("\n" + "=" * 70)
        print("🎉 기초 모델 학습 완료!")
        print("=" * 70)
        print(f"   모델 경로: {result['model_path']}")
        print(f"   학습 샘플: {result['train_samples']:,}개")
        print(f"   검증 샘플: {result['val_samples']:,}개")
        print(f"   최종 정확도: {result['final_accuracy']:.1%}")
        print("=" * 70)
        print("\n💡 봇에서 사용하려면:")
        print(f"   python main.py 실행 후 AI 모델 활성화")
        print(f"   또는 모델 경로 직접 지정: --model {result['model_path']}")


if __name__ == '__main__':
    main()
