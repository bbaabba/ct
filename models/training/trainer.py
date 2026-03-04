"""모델 학습 트레이너"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import numpy as np
from pathlib import Path

from config.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelTrainer:
    """모델 학습 및 평가"""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: PyTorch 모델
            device: 'cuda' 또는 'cpu'
        """
        self.model = model.to(device)
        self.device = device
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        logger.info(f"ModelTrainer 초기화 (Device: {device})")
        logger.info(f"모델 파라미터 수: {self._count_parameters():,}")

    def _count_parameters(self) -> int:
        """모델의 학습 가능한 파라미터 수 계산"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        lr: float = 0.001,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        모델 학습

        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            epochs: 에폭 수
            lr: 학습률
            criterion: 손실 함수
            optimizer: 옵티마이저
            early_stopping_patience: Early Stopping 인내심
            save_path: 모델 저장 경로

        Returns:
            {'train_losses': [...], 'val_losses': [...]}
        """
        # 손실 함수 설정
        if criterion is None:
            criterion = nn.HuberLoss()  # 이상치에 강건

        # 옵티마이저 설정
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Early Stopping
        patience_counter = 0

        logger.info(f"학습 시작: {epochs} 에폭, LR={lr}")

        for epoch in range(epochs):
            # ===== 학습 단계 =====
            self.model.train()
            train_loss = 0.0

            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                predictions = self.model(X_batch)
                loss = criterion(predictions.squeeze(), y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (폭발 방지)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            # ===== 검증 단계 =====
            val_loss = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)

            # Learning Rate 조정
            scheduler.step(val_loss)

            # 로그 출력
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {avg_train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

            # ===== 모델 저장 =====
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                patience_counter = 0

                # 최적 모델 저장
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"✅ 최적 모델 저장 (Val Loss: {val_loss:.6f})")
            else:
                patience_counter += 1

            # ===== Early Stopping =====
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early Stopping at epoch {epoch+1}")
                break

        logger.info(f"학습 완료! 최적 Val Loss: {self.best_loss:.6f}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """
        검증 단계

        Args:
            val_loader: 검증 데이터 로더
            criterion: 손실 함수

        Returns:
            평균 검증 손실
        """
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(X_batch)
                loss = criterion(predictions.squeeze(), y_batch)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def predict(
        self,
        X: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        예측

        Args:
            X: 입력 데이터
            batch_size: 배치 크기

        Returns:
            예측값
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                X_tensor = torch.FloatTensor(batch).to(self.device)

                pred = self.model(X_tensor)
                predictions.append(pred.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def evaluate(
        self,
        test_loader: DataLoader,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """
        테스트 평가

        Args:
            test_loader: 테스트 데이터 로더
            criterion: 손실 함수

        Returns:
            {'loss': ..., 'mae': ..., 'rmse': ..., 'mape': ...}
        """
        if criterion is None:
            criterion = nn.MSELoss()

        self.model.eval()
        test_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(X_batch)
                loss = criterion(predictions.squeeze(), y_batch)

                test_loss += loss.item()
                all_predictions.append(predictions.squeeze().cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        # 전체 예측값과 타겟값
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)

        # 평가 메트릭 계산
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100

        # R² Score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + 1e-8))

        metrics = {
            'loss': test_loss / len(test_loader),
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2_score': r2_score
        }

        logger.info(f"테스트 평가 결과:")
        logger.info(f"  Loss: {metrics['loss']:.6f}")
        logger.info(f"  MAE: {metrics['mae']:.6f}")
        logger.info(f"  RMSE: {metrics['rmse']:.6f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  R²: {metrics['r2_score']:.4f}")

        return metrics

    def save_model(self, path: str):
        """
        모델 저장

        Args:
            path: 저장 경로
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, save_path)

        logger.info(f"모델 저장: {save_path}")

    def load_model(self, path: str):
        """
        모델 로드

        Args:
            path: 모델 경로
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        logger.info(f"모델 로드: {path}")
        logger.info(f"Best Loss: {self.best_loss:.6f}")

    def plot_training_history(self, save_path: Optional[str] = None):
        """
        학습 이력 시각화

        Args:
            save_path: 저장 경로 (None이면 화면 표시)
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(self.train_losses, label='Train Loss')
            plt.plot(self.val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training History')
            plt.legend()
            plt.grid(True)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"학습 이력 그래프 저장: {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib이 설치되지 않아 그래프를 그릴 수 없습니다")
