# RestauraFoto

Restauração de fotografias históricas usando filtros clássicos de Processamento de Imagens guiados por uma rede neural de decisão.

**Disciplina:** Visão Computacional  
**Integrantes:** Khadidja Moraes Lopes da Silva, Raphael Cordeiro Bezerra Pereira, Yago Moura Ferraz

---

## O Problema

Fotografias antigas e digitalizadas sofrem degradação natural com o tempo:
- Amarelamento e tom sépia (oxidação química do papel)
- Desbotamento e baixo contraste
- Granulado de filme
- Manchas de envelhecimento (foxing)
- Vinheta nas bordas
- Arranhões físicos
- Textura periódica do papel fotográfico

O objetivo é estudar, aplicar e comparar filtros digitais clássicos para limpar essas degradações sem destruir bordas e detalhes importantes.

---

## Arquitetura

O sistema funciona como um agente iterativo:

```
Foto degradada
     │
     ▼
┌─────────────────────────────┐
│   ImageAnalyzer (CNN)        │  ← observa o estado atual da imagem
│   decide qual filtro aplicar │
└─────────────┬───────────────┘
              │ ação
              ▼
┌─────────────────────────────┐
│        FilterBank            │  ← aplica o filtro clássico escolhido
│  (filtros do domínio espacial│
│   frequência e morfológico)  │
└─────────────┬───────────────┘
              │
              ▼
       melhora? → repete
       não melhora? → PARA
```

A rede não restaura a imagem diretamente — ela **decide qual filtro clássico aplicar** a cada iteração, com base no estado visual atual da foto. O processo se repete até a rede emitir STOP ou a melhora de SSIM cair abaixo do limiar.

---

## Filtros Implementados

| # | Filtro | Degradação alvo |
|---|--------|----------------|
| 0 | Média | Suavização geral |
| 1 | Gaussiano | Granulado de filme |
| 2 | Mediana | Manchas e poeira (preserva bordas) |
| 3 | Correção de cor | Dominante amarela/sépia |
| 4 | Fechamento morfológico | Arranhões e buracos finos |
| 5 | Rejeita-banda (Fourier) | Textura periódica do papel |
| 6 | CLAHE | Desbotamento e baixo contraste local |
| 7 | STOP | — |

---

## Degradações Sintéticas de Treino

Para treinar a rede, imagens limpas recebem degradações que simulam o envelhecimento real de fotografias:

- **Sépia/amarelamento** — oxidação química simulada por transformação de cor
- **Desbotamento** — compressão de contraste com elevação dos pretos
- **Vinheta** — escurecimento das bordas por máscara gaussiana
- **Manchas de envelhecimento (foxing)** — blobs marrons gerados proceduralmente
- **Granulado de filme** — ruído correlacionado por canal
- **Arranhões** — linhas brancas aleatórias
- **Textura periódica** — padrão senoidal (simula papel fotográfico antigo)

70% das amostras usam degradação histórica composta; 30% usam degradações genéricas (gaussiano, sal-pimenta, periódico isolados).

---

## Métricas de Avaliação

**PSNR (Peak Signal-to-Noise Ratio):** Mede o erro pixel a pixel em escala logarítmica (dB). Quanto maior, mais próxima a imagem restaurada está do original.

**SSIM (Structural Similarity Index):** Compara luminância, contraste e estrutura local entre as imagens, imitando a percepção visual humana. Varia de 0 a 1; valores acima de 0.95 indicam alta qualidade.

O SSIM é usado como sinal de treino: para cada estado da imagem, o filtro que maximiza o ganho de SSIM recebe o label positivo.

---

## Estrutura do Projeto

```
RestauraFoto/
├── filters.py        — 7 filtros clássicos + ação STOP
├── noise.py          — geradores de degradação sintética
├── metrics.py        — SSIM e PSNR
├── model.py          — CNN de decisão (ImageAnalyzer)
├── dataset.py        — geração de amostras e labels via SSIM
├── agent.py          — RestorationAgent: loop iterativo
├── train.py          — pipeline de treino
├── demo.py           — visualização dos resultados
├── download_data.py  — download do dataset de treino
└── data/clean/       — imagens limpas para treino
```

---

## Como Usar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

Para usar GPU (recomendado):
```bash
# Verifique sua versão CUDA com: nvidia-smi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Baixar dataset de treino

```bash
python download_data.py
```

### 3. Treinar

```bash
python train.py --epochs 40
```

O modelo é salvo automaticamente em `model.pth` a cada vez que a acurácia de validação melhora.

### 4. Restaurar uma foto

```bash
# Foto histórica real (sem ground truth)
python demo.py "foto_antiga.jpg" --mode real

# Teste com degradação sintética conhecida (com métricas SSIM/PSNR)
python demo.py "foto.jpg" --mode synthetic
```

---

## Requisitos

- Python 3.9+
- PyTorch 2.0+
- OpenCV
- scikit-image
- GPU com CUDA (opcional, mas recomendado — testado em RTX 4050 6GB)
