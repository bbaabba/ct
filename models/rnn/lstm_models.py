"""LSTM 기반 가격 예측 모델"""
import torch
import torch.nn as nn
from typing import Tuple

from utils.logger import setup_logger

logger = setup_logger(__name__)


class LSTMPricePredictor(nn.Module):
    """LSTM 기반 가격 예측 모델 (회귀)"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: 입력 특징 개수
            hidden_size: LSTM 은닉 유닛 개수
            num_layers: LSTM 레이어 개수
            dropout: 드롭아웃 비율
        """
        super(LSTMPricePredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Fully Connected 레이어
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)  # 가격 예측 (단일 값)

        logger.info(f"LSTMPricePredictor 초기화: input={input_size}, hidden={hidden_size}, layers={num_layers}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            (batch_size, 1) 예측 가격
        """
        # LSTM 출력
        lstm_out, _ = self.lstm(x)

        # 마지막 타임스텝의 출력만 사용
        last_output = lstm_out[:, -1, :]

        # Fully Connected
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class LSTMDirectionClassifier(nn.Module):
    """LSTM 기반 가격 방향 분류 모델"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3
    ):
        """
        Args:
            input_size: 입력 특징 개수
            hidden_size: LSTM 은닉 유닛 개수
            num_layers: LSTM 레이어 개수
            dropout: 드롭아웃 비율
            num_classes: 클래스 개수 (3: 상승/횡보/하락)
        """
        super(LSTMDirectionClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Fully Connected 레이어
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)  # 클래스 분류

        logger.info(f"LSTMDirectionClassifier 초기화: input={input_size}, hidden={hidden_size}, classes={num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            (batch_size, num_classes) 클래스 로짓
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]

        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class MultiTimeframeLSTM(nn.Module):
    """다중 타임프레임 LSTM 모델"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: 각 타임프레임의 입력 특징 개수
            hidden_size: LSTM 은닉 유닛 개수
            dropout: 드롭아웃 비율
        """
        super(MultiTimeframeLSTM, self).__init__()

        self.hidden_size = hidden_size

        # 각 타임프레임별 LSTM
        self.lstm_1h = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_4h = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_1d = nn.LSTM(input_size, hidden_size, batch_first=True)

        # 통합 레이어
        self.fc1 = nn.Linear(hidden_size * 3, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)

        logger.info(f"MultiTimeframeLSTM 초기화: input={input_size}, hidden={hidden_size}")

    def forward(
        self,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor,
        x_1d: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x_1h: (batch_size, seq_len, input_size) 1시간 봉
            x_4h: (batch_size, seq_len, input_size) 4시간 봉
            x_1d: (batch_size, seq_len, input_size) 1일 봉

        Returns:
            (batch_size, 1) 예측 가격
        """
        # 각 타임프레임 LSTM 처리
        _, (h_1h, _) = self.lstm_1h(x_1h)
        _, (h_4h, _) = self.lstm_4h(x_4h)
        _, (h_1d, _) = self.lstm_1d(x_1d)

        # 마지막 은닉 상태 결합
        combined = torch.cat([h_1h[-1], h_4h[-1], h_1d[-1]], dim=1)

        # Fully Connected
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class BidirectionalLSTM(nn.Module):
    """양방향 LSTM 모델"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: 입력 특징 개수
            hidden_size: LSTM 은닉 유닛 개수
            num_layers: LSTM 레이어 개수
            dropout: 드롭아웃 비율
        """
        super(BidirectionalLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # 양방향
        )

        # 양방향이므로 hidden_size * 2
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

        logger.info(f"BidirectionalLSTM 초기화: input={input_size}, hidden={hidden_size}, layers={num_layers}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            (batch_size, 1) 예측값
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]

        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class AttentionLSTM(nn.Module):
    """Attention 메커니즘을 가진 LSTM 모델"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: 입력 특징 개수
            hidden_size: LSTM 은닉 유닛 개수
            num_layers: LSTM 레이어 개수
            dropout: 드롭아웃 비율
        """
        super(AttentionLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Attention 레이어
        self.attention = nn.Linear(hidden_size, 1)

        # Fully Connected
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

        logger.info(f"AttentionLSTM 초기화: input={input_size}, hidden={hidden_size}, layers={num_layers}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            prediction: (batch_size, 1)
            attention_weights: (batch_size, sequence_length)
        """
        # LSTM 출력
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # Attention 가중치 계산
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)

        # Attention 적용
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden)

        # Fully Connected
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out, attention_weights.squeeze(-1)
