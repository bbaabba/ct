"""LSTM 모델 학습 예제"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader

from data.binance_client import BinanceClient
from data.collectors.historical_collector import HistoricalDataCollector
from data.processors.feature_engineering import FeatureEngineering
from models.rnn.lstm_models import LSTMPricePredictor
from models.training.dataset import CryptoDataset
from models.training.trainer import ModelTrainer
from config.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """메인 학습 파이프라인"""

    print("=" * 70)
    print("LSTM 가격 예측 모델 학습")
    print("=" * 70)
    print()

    # ========== 1. 데이터 수집 ==========
    logger.info("[ 1/6 ] 데이터 수집 시작...")

    client = BinanceClient(testnet=Config.IS_TESTNET)
    collector = HistoricalDataCollector(client)

    # 최근 180일 데이터 수집
    df = collector.get_recent_data(
        symbol='BTCUSDT',
        interval='1h',
        days=180
    )

    logger.info(f"✅ 데이터 수집 완료: {len(df)}개")
    print()

    # ========== 2. 특징 엔지니어링 ==========
    logger.info("[ 2/6 ] 특징 엔지니어링...")

    # ML 데이터 준비 (전체 파이프라인)
    ml_data = FeatureEngineering.prepare_ml_data(
        df=df,
        target_column='close',
        lookback=60,
        forecast_horizon=1,
        normalize=True,
        train_size=0.7,
        val_size=0.15
    )

    X_train = ml_data['X_train']
    y_train = ml_data['y_train']
    X_val = ml_data['X_val']
    y_val = ml_data['y_val']
    X_test = ml_data['X_test']
    y_test = ml_data['y_test']
    scaler = ml_data['scaler']
    feature_columns = ml_data['feature_columns']

    logger.info(f"✅ 특징 엔지니어링 완료")
    logger.info(f"  특징 개수: {len(feature_columns)}")
    logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print()

    # ========== 3. 데이터셋 및 데이터로더 생성 ==========
    logger.info("[ 3/6 ] 데이터로더 생성...")

    train_dataset = CryptoDataset(X_train, y_train)
    val_dataset = CryptoDataset(X_val, y_val)
    test_dataset = CryptoDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    logger.info(f"✅ 데이터로더 생성 완료")
    print()

    # ========== 4. 모델 생성 ==========
    logger.info("[ 4/6 ] 모델 생성...")

    input_size = X_train.shape[2]  # 특징 개수

    model = LSTMPricePredictor(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )

    logger.info(f"✅ 모델 생성 완료")
    print()

    # ========== 5. 학습 ==========
    logger.info("[ 5/6 ] 모델 학습 시작...")

    trainer = ModelTrainer(model)

    save_path = Config.MODEL_SAVE_DIR / 'lstm_btc_1h_best.pth'

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,  # 테스트용으로 50 에폭
        lr=0.001,
        early_stopping_patience=10,
        save_path=str(save_path)
    )

    logger.info(f"✅ 학습 완료")
    print()

    # ========== 6. 평가 ==========
    logger.info("[ 6/6 ] 모델 평가...")

    # 최적 모델 로드
    trainer.load_model(str(save_path))

    # 테스트 세트 평가
    metrics = trainer.evaluate(test_loader)

    logger.info(f"✅ 평가 완료")
    print()

    # ========== 결과 요약 ==========
    print("=" * 70)
    print("학습 결과 요약")
    print("=" * 70)
    print(f"데이터:")
    print(f"  심볼: BTCUSDT")
    print(f"  간격: 1h")
    print(f"  기간: 최근 180일")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print()
    print(f"모델:")
    print(f"  타입: LSTM")
    print(f"  입력 크기: {input_size}")
    print(f"  Hidden Size: 128")
    print(f"  Layers: 2")
    print()
    print(f"성능:")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R²: {metrics['r2_score']:.4f}")
    print()
    print(f"모델 저장 위치: {save_path}")
    print("=" * 70)

    # 학습 이력 그래프 저장
    plot_path = Config.MODEL_SAVE_DIR / 'training_history.png'
    trainer.plot_training_history(save_path=str(plot_path))

    logger.info("🎉 모든 과정 완료!")


if __name__ == '__main__':
    main()
