"""LSTM 모델 기반 거래 전략"""
import pandas as pd
import numpy as np
import torch
from typing import Optional
from pathlib import Path

from strategies.base import BaseStrategy, Signal, TradeSignal
from data.processors.feature_engineering import FeatureEngineering
from models.rnn.lstm_models import LSTMPricePredictor
from utils.logger import setup_logger

logger = setup_logger(__name__)


class LSTMStrategy(BaseStrategy):
    """
    LSTM 가격 예측 기반 거래 전략

    학습된 LSTM 모델을 사용하여 다음 시점 가격을 예측하고,
    예측 방향과 신뢰도에 따라 거래 신호를 생성합니다.

    파라미터:
        - model_path: 학습된 모델 경로
        - lookback: LSTM 입력 시퀀스 길이 (기본: 60)
        - threshold: 가격 변화 기준 (기본: 0.005 = 0.5%)
        - min_confidence: 최소 신뢰도 (기본: 0.7)
        - max_position_pct: 최대 포지션 비율 (기본: 0.2)
    """

    def __init__(self, params=None):
        default_params = {
            'model_path': 'models/saved/lstm_btc_1h_best.pth',
            'lookback': 60,
            'threshold': 0.005,
            'min_confidence': 0.7,
            'max_position_pct': 0.2
        }

        if params:
            default_params.update(params)

        super().__init__(name='LSTM_Prediction', params=default_params)

        self.model: Optional[LSTMPricePredictor] = None
        self.scaler = None
        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, model_path: str):
        """학습된 모델 로드"""
        model_path = Path(model_path)

        if not model_path.exists():
            logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
            return False

        try:
            # 체크포인트 로드
            checkpoint = torch.load(model_path, map_location=self.device)

            # 모델 생성
            input_size = checkpoint.get('input_size', 50)
            hidden_size = checkpoint.get('hidden_size', 128)
            num_layers = checkpoint.get('num_layers', 2)
            dropout = checkpoint.get('dropout', 0.2)

            self.model = LSTMPricePredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )

            # 가중치 로드
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            # Scaler 및 feature columns 로드
            self.scaler = checkpoint.get('scaler')
            self.feature_columns = checkpoint.get('feature_columns')

            logger.info(f"✅ 모델 로드 완료: {model_path}")
            logger.info(f"   입력 크기: {input_size}, Hidden: {hidden_size}, Layers: {num_layers}")

            self.is_fitted = True
            return True

        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return False

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """전체 기술 지표 계산 (Phase 2에서 구현한 FeatureEngineering 사용)"""
        return FeatureEngineering.calculate_all_indicators(data)

    def prepare_input(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """LSTM 입력 데이터 준비"""
        try:
            # 지표 계산
            df = self.calculate_indicators(data)

            # 결측치 제거
            df = df.dropna()

            lookback = self.params['lookback']

            if len(df) < lookback:
                logger.warning(f"데이터 부족: {len(df)} < {lookback}")
                return None

            # Feature columns 선택
            if self.feature_columns is None:
                self.feature_columns = FeatureEngineering.get_feature_columns(exclude_base=False)

            available_features = [col for col in self.feature_columns if col in df.columns]

            if len(available_features) == 0:
                logger.error("사용 가능한 특징이 없습니다")
                return None

            # 최근 lookback 개의 데이터 선택
            recent_data = df[available_features].iloc[-lookback:].values

            # 정규화
            if self.scaler is not None:
                recent_data = self.scaler.transform(recent_data)

            # 배치 차원 추가: (1, lookback, features)
            recent_data = recent_data.reshape(1, lookback, -1)

            return recent_data

        except Exception as e:
            logger.error(f"입력 데이터 준비 실패: {e}")
            return None

    def predict_price(self, data: pd.DataFrame) -> Optional[float]:
        """다음 가격 예측"""
        if self.model is None:
            model_path = self.params['model_path']
            if not self.load_model(model_path):
                logger.error("모델 로드 실패")
                return None

        # 입력 데이터 준비
        X = self.prepare_input(data)

        if X is None:
            return None

        try:
            # 예측
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                prediction = self.model(X_tensor)

                predicted_price = prediction.cpu().numpy()[0][0]

                # 역정규화 (필요시)
                if self.scaler is not None:
                    # 스케일러가 전체 특징에 대해 fit되어 있으므로, close 컬럼 인덱스 찾기
                    if 'close' in self.feature_columns:
                        close_idx = self.feature_columns.index('close')
                        # 더미 배열 생성 후 역변환
                        dummy = np.zeros((1, len(self.feature_columns)))
                        dummy[0, close_idx] = predicted_price
                        inverse_transformed = self.scaler.inverse_transform(dummy)
                        predicted_price = inverse_transformed[0, close_idx]

            return float(predicted_price)

        except Exception as e:
            logger.error(f"가격 예측 실패: {e}")
            return None

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

        # 현재 가격
        current_price = data['close'].iloc[-1]

        # 다음 가격 예측
        predicted_price = self.predict_price(data)

        if predicted_price is None:
            return TradeSignal(
                signal=Signal.HOLD,
                confidence=0.0,
                price=current_price,
                timestamp=pd.Timestamp.now(),
                reason="예측 실패"
            )

        # 가격 변화율 계산
        price_change_pct = (predicted_price - current_price) / current_price

        threshold = self.params['threshold']

        # 신호 결정
        signal = Signal.HOLD
        reason = f"예측 변화율: {price_change_pct:.2%}"
        confidence = 0.0

        # 상승 예측 → 매수
        if price_change_pct > threshold:
            signal = Signal.BUY
            reason = f"가격 상승 예측 ({current_price:.2f} → {predicted_price:.2f}, +{price_change_pct:.2%})"

            # 신뢰도: 변화율이 클수록 높음
            confidence = min(abs(price_change_pct) / (threshold * 5), 1.0)

        # 하락 예측 → 매도
        elif price_change_pct < -threshold:
            signal = Signal.SELL
            reason = f"가격 하락 예측 ({current_price:.2f} → {predicted_price:.2f}, {price_change_pct:.2%})"

            # 신뢰도: 변화율이 클수록 높음
            confidence = min(abs(price_change_pct) / (threshold * 5), 1.0)

        # 변화 미미 → 홀드
        else:
            reason = f"변화 미미 ({price_change_pct:.2%})"

        trade_signal = TradeSignal(
            signal=signal,
            confidence=confidence,
            price=current_price,
            timestamp=pd.Timestamp.now(),
            reason=reason,
            metadata={
                'predicted_price': predicted_price,
                'price_change_pct': price_change_pct,
                'threshold': threshold
            }
        )

        logger.info(f"{self.name} 신호: {signal.name}, 신뢰도: {confidence:.2%}, 이유: {reason}")

        return trade_signal
