"""LSTM-based (or fallback) pedestrian trajectory predictor."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from src.config import LSTM_HISTORY_LENGTH, LSTM_PRED_STEPS, YOLO_DEVICE


Point = Tuple[float, float]

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


if nn is not None:
    class DummyLSTM(nn.Module):
        """Simple LSTM head for 2D trajectories."""

        def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_layers: int = 2, pred_steps: int = 15):
            super().__init__()
            self.pred_steps = pred_steps
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.head = nn.Linear(hidden_dim, input_dim)

        def forward(self, seq: torch.Tensor) -> torch.Tensor:
            outputs = []
            last_input = seq[:, -1:, :]
            _, (h, c) = self.lstm(seq)
            cur = last_input
            for _ in range(self.pred_steps):
                out, (h, c) = self.lstm(cur, (h, c))
                pred = self.head(out[:, -1:, :])
                outputs.append(pred)
                cur = pred
            return torch.cat(outputs, dim=1)


class TrajectoryPredictor:
    """Trajectory predictor that falls back to constant velocity when torch is unavailable."""

    def __init__(
        self,
        weights_path: str = "weights/lstm_pedestrian.pt",
        history_length: int = LSTM_HISTORY_LENGTH,
        pred_steps: int = LSTM_PRED_STEPS,
        device: str = YOLO_DEVICE,
    ) -> None:
        self.history_length = history_length
        self.pred_steps = pred_steps
        self.device = device
        self.model = None
        self.enabled = False

        if torch is None or nn is None:
            return

        try:
            self.model = DummyLSTM(pred_steps=pred_steps).to(device)
            weight_file = Path(weights_path)
            if weight_file.exists():
                self.model.load_state_dict(torch.load(weight_file, map_location=device))
                self.enabled = True
            self.model.eval()
        except Exception:
            self.model = None
            self.enabled = False

    def predict(self, history: Sequence[Point]) -> List[Point]:
        if len(history) < 2:
            return []

        trimmed = list(history)[-self.history_length :]
        if self.enabled and self.model is not None and torch is not None:
            return self._predict_with_model(trimmed)
        return self._predict_constant_velocity(trimmed)

    def _predict_with_model(self, history: Sequence[Point]) -> List[Point]:
        tensor = torch.tensor(history, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            future = self.model(tensor).squeeze(0).cpu().numpy()
        return [(float(x), float(y)) for x, y in future]

    def _predict_constant_velocity(self, history: Sequence[Point]) -> List[Point]:
        p1 = history[-2]
        p2 = history[-1]
        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]
        preds: List[Point] = []
        last_point = list(history)[-1]
        for _ in range(self.pred_steps):
            last_point = (last_point[0] + vx, last_point[1] + vy)
            preds.append(last_point)
        return preds
