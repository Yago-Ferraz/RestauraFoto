# RestauraFoto

Sistema de restauração iterativa de fotografias históricas usando aprendizado profundo. Um agente baseado em ResNet-18 analisa a imagem e decide automaticamente qual filtro aplicar — e em qual intensidade — repetindo o processo até obter o melhor resultado possível.

---

## Instalação

**Requisitos:** Python 3.10 ou superior.

```bash
# 1. Clone o repositório
git clone https://github.com/Yago-Ferraz/RestauraFoto.git

# 2. Crie e ative o ambiente virtual
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Instale as dependências (PyTorch CPU incluído)
pip install -r requirements.txt
```

O `requirements.txt` já aponta para a versão CPU do PyTorch. Nenhuma GPU é necessária para rodar o projeto.

> Para usar GPU (treinamento mais rápido), substitua as linhas do torch no `requirements.txt` pela versão CUDA correspondente à sua placa. Veja em: https://pytorch.org/get-started/locally/

---

## Estrutura de arquivos

```
restaurafoto/
├── gui.py              # Interface gráfica Tkinter
├── demo.py             # Demo por linha de comando
├── agent.py            # Agente iterativo de restauração
├── model.py            # Arquitetura ResNet-18 adaptada
├── train.py            # Script de treinamento
├── dataset.py          # Geração de dados sintéticos e soft labels
├── filters.py          # 20 filtros de processamento de imagem
├── noise.py            # Funções de degradação sintética
├── metrics.py          # SSIM e PSNR
├── download_data.py    # Baixa o dataset DIV2K automaticamente
├── model.pth           # Pesos treinados (pronto para uso)
└── requirements.txt    # Dependências
```

---

## Como usar

### Linha de Comando

```bash
# Foto antiga real
python demo.py foto_antiga.jpg --mode real

# Teste sintético com ground truth
python demo.py foto_limpa.jpg --mode synthetic

# Usando checkpoint de treinamento em vez do modelo final
python demo.py foto.jpg --mode real --model checkpoint.pth
```

### Interface Gráfica

```bash
python gui.py
```

1. Clique em **Selecionar Foto** e escolha uma fotografia (`.jpg`, `.png`, `.webp`, etc.)
2. Escolha o modo:
   - **Real** — restaura a foto como está, sem referência
   - **Sintético** — aplica uma degradação artificial e depois restaura, calculando SSIM e PSNR
3. Ajuste o **máximo de iterações** (padrão: 6)
4. Clique em **Restaurar**
5. O painel direito mostra o resultado; use **Salvar Resultado** para exportar

## Treinamento

Para retreinar o modelo do zero (necessário se quiser usar seu próprio dataset):

```bash
# 1. Baixe o dataset DIV2K (800 imagens de alta resolução)
python download_data.py

# 2. Treine por 40 épocas (recomendado)
python train.py --data data/clean --epochs 40 --batch 128

# Para continuar de onde parou
python train.py --data data/clean --epochs 40 --batch 128 --resume
```

O treinamento salva `model.pth` automaticamente com os melhores pesos e gera `training_curves.png` com as curvas de loss e acurácia.

---

## Como o projeto foi feito

### Problema

Fotografias históricas sofrem de diversos tipos de degradação ao longo do tempo: amarelamento por oxidação química (sépia), desbotamento, granulado de filme, manchas de envelhecimento, subexposição, e danos físicos como arranhões. Restaurar essas fotos manualmente exige experiência e tempo. O objetivo do projeto é automatizar essa decisão: dado o estado atual da foto, qual filtro aplicar, com qual intensidade?

### Arquitetura geral

O sistema é composto por três camadas:

```
Foto degradada
      |
      v
+-------------+
|  ResNet-18  |  extrai features visuais (textura, cor, contraste)
|  (backbone) |
+------+------+
       |  vetor de 512 features
       v
+-------------+
|   Cabeca    |  Linear(512->256) -> ReLU -> Dropout(0.4) -> Linear(256->21)
|classificad. |
+------+------+
       |  21 logits (20 filtros + STOP)
       v
+-------------+
|   Agente    |  aplica o filtro, verifica qualidade, decide continuar
|  iterativo  |
+-------------+
```

### Modelo: ResNet-18 com Transfer Learning

A base da rede é uma **ResNet-18 pré-treinada no ImageNet**. A ideia do transfer learning é aproveitar que a rede já aprendeu a reconhecer texturas, bordas e padrões visuais gerais em milhões de imagens — e adaptar esse conhecimento para reconhecer tipos de degradação fotográfica.

O treinamento acontece em **duas fases**:

**Fase 1 (épocas 1–10):** o backbone é congelado. Apenas a cabeça de classificação é treinada com learning rate `1e-3`. Isso faz a cabeça aprender rapidamente os mapeamentos mais básicos sem distorcer os pesos do ImageNet.

**Fase 2 (épocas 11–40):** o backbone inteiro é descongelado. Um learning rate diferenciado é usado — `1e-5` para o backbone (ajuste fino e cuidadoso) e `1e-4` para a cabeça — com `CosineAnnealingLR` em ambas as fases.

A função de perda é **soft cross-entropy** (cross-entropy com distribuições de probabilidade, não rótulos duros), que permite ao modelo aprender que vários filtros podem ser igualmente bons para uma situação.

### Os 20 filtros

O sistema cobre todas as principais técnicas de processamento de imagem da disciplina, com **variantes de intensidade** para que o agente possa escolher o quão agressivo ser:

| # | Filtro | Técnica | Uso |
|---|--------|---------|-----|
| 0 | `mean` | Filtragem espacial passa-baixa | Suavização uniforme |
| 1 | `gaussian_soft` (σ=0.8) | Filtro gaussiano | Ruído leve |
| 2 | `gaussian_medium` (σ=1.5) | Filtro gaussiano | Ruído moderado |
| 3 | `gaussian_strong` (σ=2.5) | Filtro gaussiano | Ruído forte |
| 4 | `bilateral` | Filtragem não-linear | Preserva bordas |
| 5 | `median_fine` (k=3) | Filtro da mediana | Sal e pimenta leve |
| 6 | `median_strong` (k=7) | Filtro da mediana | Sal e pimenta forte |
| 7 | `high_pass_gentle` (α=0.4) | Passa-alta / Unsharp Mask | Nitidez leve |
| 8 | `high_pass_strong` (α=1.0) | Passa-alta / Unsharp Mask | Nitidez agressiva |
| 9 | `color_correction_gentle` | Processamento de cor | Cast leve (papel envelhecido, sépia suave) |
| 10 | `color_correction_strong` | Processamento de cor | Cast forte (sépia intensa, tom azul) |
| 11 | `clahe_subtle` (clip=1.5) | Histograma adaptativo (CLAHE) | Contraste moderado |
| 12 | `clahe_strong` (clip=3.0) | Histograma adaptativo (CLAHE) | Contraste agressivo |
| 13 | `gamma_bright` (γ=0.6) | Transformação de potência s=c·r^γ | Aclara moderado |
| 14 | `gamma_very_bright` (γ=0.35) | Transformação de potência s=c·r^γ | Aclara forte |
| 15 | `log_transform` | Transformação logarítmica s=c·log(1+r) | Revela detalhes nas sombras |
| 16 | `contrast_stretch` | Alargamento de contraste (percentil) | Normaliza histograma |
| 17 | `morph_open` | Abertura morfológica (erosão→dilatação) | Remove poeira e foxing |
| 18 | `morph_close` | Fechamento morfológico (dilatação→erosão) | Preenche arranhões finos |
| 19 | `fourier` | Rejeita-banda de Fourier | Remove textura periódica do papel |
| 20 | `STOP` | — | Encerra quando a foto está satisfatória |

O **filtro de Fourier** segue o pipeline completo da disciplina: padding 2M×2N para evitar aliasing, centralização do espectro por (-1)^(x+y), DFT, máscara rejeita-banda H(u,v), produto no domínio da frequência, IDFT, e remoção do padding.

### Geração de dados sintéticos e Soft Labels

Como não existe um dataset de "pares degradado/restaurado" com rótulos de qual filtro aplicar, os dados de treino são **gerados sinteticamente** em tempo real:

1. Uma imagem limpa do dataset DIV2K é escolhida
2. Uma degradação aleatória é aplicada (entre 17 cenários diferentes)
3. Todos os 20 filtros são testados na imagem degradada
4. A melhoria de cada filtro é medida pela **métrica híbrida perceptual**
5. Essa melhoria é convertida em uma distribuição de probabilidade via softmax com temperatura (soft label)
6. Se nenhum filtro melhora além de 0.003, STOP recebe probabilidade 1.0

**Distribuição de degradações (17 cenários):**

| Probabilidade | Degradação | Filtro esperado |
|--------------|-----------|----------------|
| 8% | Nenhuma (foto limpa) | STOP |
| 7% | Tom quente sutil (papel envelhecido, pintura) | `color_correction_gentle` |
| 7% | Tom quente moderado (fotos anos 70-80) | `color_correction_gentle` |
| 8% | Sépia moderada (0.3–0.6) | `color_correction_gentle` |
| 7% | Sépia forte (0.65–1.0) | `color_correction_strong` |
| 8% | Tom azulado (WB frio) | `color_correction_gentle` |
| 8% | Subexposição forte (fator 0.12–0.32) | `gamma_very_bright` / `log_transform` |
| 7% | Subexposição leve (fator 0.38–0.65) | `gamma_bright` |
| 8% | Desbotamento/fading | `clahe_subtle` / `clahe_strong` |
| 6% | Foto desfocada (motion blur) | `high_pass_gentle` / `high_pass_strong` |
| 7% | Granulado leve | `gaussian_soft` / `bilateral` |
| 6% | Granulado forte | `gaussian_strong` / `median_strong` |
| 5% | Sal e pimenta | `median_fine` / `median_strong` |
| 5% | Textura periódica | `fourier` |
| 3% | Manchas claras (poeira no scanner) | `morph_open` |
| 2% | Arranhões e fissuras escuros | `morph_close` |
| 2% | Degradação histórica composta | misto |

### Métrica híbrida perceptual

A métrica que guia os soft labels combina estrutura e fidelidade de cor:

```
score = 0.5 x SSIM  +  0.5 x acuracia_de_cor
```

O **SSIM** (Structural Similarity Index) mede semelhança estrutural, bordas e contraste entre a imagem filtrada e a referência limpa.

A **acurácia de cor** mede o desvio da média dos canais a/b do espaço CIE LAB:

```
cast_dist    = || mean_ab(filtrada) - mean_ab(referencia) ||_2
acuracia_cor = max(0,  1 - cast_dist / 20.0)
```

Usar a **média** dos canais a/b — e não o MSE pixel a pixel — é essencial para detectar dominantes de cor como sépia. O amarelamento desloca a média global dos canais sem alterar a variância local, tornando o MSE pixel a pixel insensível a ele. O desvio da média no espaço LAB captura exatamente esse deslocamento.

O cálculo é feito em resolução reduzida (64×64) para tornar a geração do dataset viável em CPU (~12x mais rápido que resolução original).

### Agente iterativo

O agente em `agent.py` implementa o loop de decisão com vários mecanismos de segurança:

**Detecção de foto P&B:** se os canais R/G/B têm desvio padrão entre suas médias < 8 (imagem em escala de cinza), `color_correction` é bloqueado. Aplicar correção de cor em fotos P&B criaria uma dominante artificial onde não havia nenhuma.

**Confiança mínima para filtros destrutivos:** `high_pass_strong` requer probabilidade ≥ 30% e `fourier` requer ≥ 45%. Esses filtros podem criar artefatos graves (amplificação de ruído, anéis de frequência) se aplicados com baixa confiança.

**Redirecionamento por SSIM consecutivo:** se um filtro distorce demais a imagem do passo atual (SSIM < 25% de similaridade), o agente não desiste — tenta a segunda melhor opção. Só para se a segunda também falhar.

**Guarda SSIM vs. original:** no modo real (sem ground truth), se o resultado acumulado difere mais de 20% da imagem de entrada (SSIM < 0.80), o agente para para evitar distorção progressiva imperceptível a cada passo.

**Limite de blurs:** no máximo 2 passes de filtros de suavização (gaussian, bilateral) por sessão, para não produzir imagens excessivamente borradas.

**Bloco anti-loop imediato:** a mesma ação não pode ser aplicada duas vezes consecutivas (A→A bloqueado), mas A→B→A é permitido — isso permite ao agente iterar entre variantes de intensidade para encontrar o equilíbrio correto (ex: `color_correction_gentle` → `color_correction_strong` → `color_correction_gentle`).

### Dataset

O projeto usa o **DIV2K** — 800 fotografias de alta resolução com conteúdo variado (paisagens, pessoas, objetos, arquitetura). Para cada imagem são geradas 25 amostras sintéticas diferentes durante o treinamento, totalizando 20.000 amostras. 85% é usado para treino e 15% para validação, com semente fixa (42) para reprodutibilidade.

---

## Dependências

| Pacote | Versão | Uso |
|--------|--------|-----|
| `torch` | 2.2.2 (CPU) | Rede neural, treinamento |
| `torchvision` | 0.17.2 | ResNet-18 pré-treinada no ImageNet |
| `opencv-python` | ≥ 4.8 | Filtros, leitura de imagens, conversão de cor |
| `scikit-image` | ≥ 0.21 | Cálculo de SSIM |
| `numpy` | ≥ 1.24 | Operações matriciais |
| `matplotlib` | ≥ 3.7 | Curvas de treinamento e visualizações |
| `Pillow` | ≥ 9.5 | Exibição de imagens na GUI Tkinter |
| `tqdm` | ≥ 4.65 | Barra de progresso durante o treino |

---

## Referências

- He, K. et al. *Deep Residual Learning for Image Recognition*. CVPR 2016.
- Wang, Z. et al. *Image Quality Assessment: From Error Visibility to Structural Similarity*. IEEE TIP 2004.
- Agustsson, E. & Timofte, R. *NTIRE 2017 Challenge on Single Image Super-Resolution*. CVPRW 2017. (Dataset DIV2K)
- Gonzalez, R. & Woods, R. *Digital Image Processing*. 4ª ed. Pearson 2018.
