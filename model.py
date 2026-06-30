import torch.nn as nn
import torchvision.models as models


class ImageAnalyzer(nn.Module):
    """
    ResNet-18 pré-treinado no ImageNet como extrator de features,
    com cabeça de classificação para decisão de filtros.

    Transfer learning: o backbone já conhece texturas, bordas e padrões
    visuais gerais. Só treinamos a cabeça para mapear essas features
    na decisão de qual filtro aplicar.

    Treino em duas fases (ver train.py):
      Fase 1 — backbone congelado, treina só a cabeça (rápido)
      Fase 2 — backbone descongelado, fine-tuning com lr menor
    """

    def __init__(self, n_actions=8, freeze_backbone=True):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove a cabeça original de 1000 classes do ImageNet
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Cabeça de decisão de filtros
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_actions),
        )

    def unfreeze_backbone(self):
        """Descongela o backbone para fine-tuning na fase 2."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.classifier(self.backbone(x))
