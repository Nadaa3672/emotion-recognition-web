"""
Rete convoluzionale per il riconoscimento delle emozioni dal parlato.

Lo spettrogramma log-Mel viene trattato come un'immagine a tre canali: il
piano tempo-frequenza e le sue derivate prima e seconda lungo l'asse temporale,
che codificano la dinamica prosodica (come varia l'energia nel tempo) oltre alla
sua distribuzione istantanea.

Architettura: tre blocchi convoluzionali (convoluzione 3x3, batch normalization,
ReLU, max-pooling 2x2) con numero di canali crescente w, 2w, 4w; global average
pooling; dropout; strato lineare finale sulle otto classi.
"""
import numpy as np
import torch
import torch.nn as nn

from src import config


class EmotionCNN(nn.Module):
    def __init__(self, n_classes: int = 8, dropout: float = 0.3, width: int = 32,
                 in_channels: int = 3):
        super().__init__()
        w = int(width)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, w),
            block(w, w * 2),
            block(w * 2, w * 4),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(w * 4, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model(path=None, device=None):
    """
    Ricostruisce la rete a partire dal checkpoint salvato.

    Il checkpoint contiene, oltre ai pesi, gli iperparametri dell'architettura e
    le statistiche di normalizzazione calcolate sul solo insieme di
    addestramento: vanno riapplicate identiche in fase di inferenza, altrimenti
    la rete riceve dati su una scala diversa da quella che ha imparato.
    """
    path = path or config.MODEL_PATH
    device = device or torch.device("cpu")
    ck = torch.load(path, map_location=device, weights_only=False)

    model = EmotionCNN(
        n_classes=len(config.EMOTIONS),
        dropout=float(ck["hp"].get("dropout", 0.3)),
        width=int(ck["hp"].get("width", 32)),
        in_channels=int(ck.get("in_channels", 3)),
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    mean, std = ck["norm"]
    return model, float(mean), float(std)


def to_tensor(mel: np.ndarray, mean: float, std: float) -> torch.Tensor:
    """
    Trasforma uno spettrogramma (H, W) nel tensore (1, 3, H, W) atteso dalla rete.

    Standardizzazione con le statistiche dell'addestramento, poi impilamento con
    le derivate prima e seconda lungo l'asse temporale.
    """
    base = ((mel - mean) / (std + 1e-6)).astype(np.float32)
    d1 = np.gradient(base, axis=1).astype(np.float32)     # asse 1 = tempo
    d2 = np.gradient(d1, axis=1).astype(np.float32)
    x = np.stack([base, d1, d2], axis=0)[None, ...]       # (1, 3, H, W)
    return torch.from_numpy(x)


def n_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
