"""PyTorch Dataset 클래스"""
import torch
from torch.utils.data import Dataset
import numpy as np


class CryptoDataset(Dataset):
    """암호화폐 가격 데이터셋"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: 입력 데이터 (n_samples, sequence_length, n_features)
            y: 타겟 데이터 (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]


class MultiTimeframeDataset(Dataset):
    """다중 타임프레임 데이터셋"""

    def __init__(
        self,
        X_1h: np.ndarray,
        X_4h: np.ndarray,
        X_1d: np.ndarray,
        y: np.ndarray
    ):
        """
        Args:
            X_1h: 1시간 봉 데이터
            X_4h: 4시간 봉 데이터
            X_1d: 1일 봉 데이터
            y: 타겟 데이터
        """
        self.X_1h = torch.FloatTensor(X_1h)
        self.X_4h = torch.FloatTensor(X_4h)
        self.X_1d = torch.FloatTensor(X_1d)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X_1h)

    def __getitem__(self, idx: int) -> tuple:
        return self.X_1h[idx], self.X_4h[idx], self.X_1d[idx], self.y[idx]
