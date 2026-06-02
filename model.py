import torch
import torch.nn as nn


class ImageAnalyzer(nn.Module):
    """
    CNN pequena que analisa o estado atual da imagem e decide qual
    filtro aplicar a seguir (ou parar).

    Input:  tensor (B, 3, 128, 128) normalizado ImageNet
    Output: logits (B, n_actions)
    """

    def __init__(self, n_actions=8, input_channels=3):
        super().__init__()

        self.features = nn.Sequential(
            # 128 → 64
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 64 → 32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 32 → 16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 16 → 8
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
