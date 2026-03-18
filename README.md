#Transformer "From Scratch" — Lab 04

> **Aviso acadêmico:** Partes complementadas com IA, revisadas por Ingrid.

Implementação completa da arquitetura **Encoder-Decoder Transformer** em NumPy puro, construída sobre os módulos desenvolvidos nos Laboratórios 01, 02 e 03.  
O objetivo é verificar o **fluxo correto dos tensores** ao longo de todas as camadas — não a qualidade linguística da saída (os pesos são aleatórios, sem treinamento).

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Como Executar](#-como-executar)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Arquitetura Implementada](#-arquitetura-implementada)
- [Lógica Matemática](#-lógica-matemática)
- [Saída Esperada](#-saída-esperada)
- [Observações Pedagógicas](#-observações-pedagógicas)

---

## Visão Geral

Este laboratório integra todos os componentes do Transformer original
("Attention is All You Need", Vaswani et al., 2017):

| Componente | Arquivo | Tarefa |
|---|---|---|
| Scaled Dot-Product Attention | `transformer_lab04.py` | 1 |
| Position-wise FFN | `transformer_lab04.py` | 1 |
| Add & Norm (Residual) | `transformer_lab04.py` | 1 |
| Positional Encoding (sin/cos) | `transformer_lab04.py` | 1 |
| Encoder Stack (6 blocos) | `transformer_lab04.py` | 2 |
| Decoder Stack (6 blocos) | `transformer_lab04.py` | 3 |
| Loop Auto-Regressivo | `transformer_lab04.py` | 4 |

---

## Pré-requisitos

- **Python** `>= 3.10`
- **pip** `>= 21`
- **NumPy** `>= 1.24`

Verifique sua versão do Python:

```bash
python --version
```

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/<seu-usuario>/<seu-repo>.git
cd <seu-repo>
```

### 2. (Recomendado) Crie um ambiente virtual

```bash
# Criar
python -m venv .venv

# Ativar — Linux / macOS
source .venv/bin/activate

# Ativar — Windows
.venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install numpy
```

Ou, se o projeto tiver um `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Como Executar

Com o ambiente virtual ativo, rode:

```bash
python transformer_lab04.py
```

O script irá:

1. Instanciar o Encoder e o Decoder com pesos aleatórios.
2. Codificar a frase de entrada `["Thinking", "Machines"]` pelas 6 camadas do Encoder.
3. Executar o loop auto-regressivo no Decoder até o token `<EOS>`.
4. Imprimir as verificações de sanidade ao final.

### Exemplo de saída (resumida)

```
============================================================
Laboratório 04 — Transformer Completo 'From Scratch'
============================================================

Instanciando Encoder e Decoder...
  Encoder: 6 camadas | d_model=512 | d_ff=2048
  Decoder: 6 camadas | d_model=512 | vocab_size=10000

Frase de entrada (Encoder): ['Thinking', 'Machines']

TAREFA 4 — Loop Auto-Regressivo de Inferência
----------------------------------------
  Passo  1 | próximo token: 'palavra_XXXX' (id=..., prob=...)
  Passo  2 | próximo token: 'palavra_XXXX' (id=..., prob=...)
  Passo  3 | próximo token: 'palavra_XXXX' (id=..., prob=...)
  Passo  4 | próximo token: 'palavra_XXXX' (id=..., prob=...)
  Passo  5 | próximo token: '<EOS>'        (id=1,   prob=1.0000)

✓ Token <EOS> detectado. Geração encerrada.

VERIFICAÇÕES DE SANIDADE
✓ Máscara causal: futuro zerado? True
✓ Shape do Encoder preservado: (1, 1, 512)
✓ Probabilidades do Decoder somam 1? True (soma=1.000000)
```

> **Nota:** os tokens gerados (`palavra_XXXX`) variam a cada execução por causa
> do top-K sampling com pesos aleatórios. O comportamento determinístico é
> atingido mantendo o `np.random.seed(42)` no topo do arquivo.

---

## Estrutura do Projeto

```
.
├── transformer_lab04.py   # Código principal — Transformer completo
├── README.md              # Este arquivo
└── requirements.txt       # Dependências (opcional)
```

---

## 🏗️ Arquitetura Implementada

```
Entrada (token IDs)
       │
       ▼
┌──────────────────────────────────┐
│          ENCODER STACK           │
│  ┌────────────────────────────┐  │
│  │      EncoderBlock  ×6      │  │
│  │                            │  │
│  │  x ──► Self-Attention      │  │
│  │           │                │  │
│  │        Add & Norm          │  │
│  │           │                │  │
│  │          FFN               │  │
│  │           │                │  │
│  │        Add & Norm          │  │
│  └────────────────────────────┘  │
└──────────────┬───────────────────┘
               │  Memória  Z  (1, seq_enc, d_model)
               ▼
┌──────────────────────────────────┐
│          DECODER STACK           │
│  ┌────────────────────────────┐  │
│  │      DecoderBlock  ×6      │  │
│  │                            │  │
│  │  y ──► Masked Self-Att     │  │
│  │        (máscara causal)    │  │
│  │           │                │  │
│  │        Add & Norm          │  │
│  │           │                │  │
│  │        Cross-Attention     │  │
│  │        Q←dec  K,V←Z       │  │
│  │           │                │  │
│  │        Add & Norm          │  │
│  │           │                │  │
│  │          FFN               │  │
│  │           │                │  │
│  │        Add & Norm          │  │
│  └────────────────────────────┘  │
│                                  │
│   Linear  (d_model → vocab_size) │
│   Softmax → distribuição  P(w)   │
└──────────────┬───────────────────┘
               │
               ▼
        próximo token
   (loop até <EOS> ou max_steps)
```

---

## Lógica Matemática

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

- `Q`, `K`, `V` são projeções lineares da entrada.
- A divisão por `√d_k` evita que os scores cresçam demais em alta dimensão,
  mantendo os gradientes estáveis durante o treinamento.

### Add & Norm — Conexão Residual

```
Output = LayerNorm( x + Sublayer(x) )
```

- O atalho residual soma a entrada original à saída da subcamada,
  facilitando o fluxo de gradiente em redes profundas.
- O LayerNorm normaliza ao longo do eixo de features (`d_model`).

### Feed-Forward Network (FFN)

```
FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
```

- Expansão de dimensão: `d_model (512) → d_ff (2048) → d_model (512)`.
- A ReLU entre as duas projeções lineares introduce não-linearidade.

### Máscara Causal (Look-Ahead Mask)

```
M[i, j] =   0    se j ≤ i   →  passado / presente  (permitido)
M[i, j] = -inf   se j > i   →  futuro              (bloqueado)
```

- Somada aos scores **antes** do softmax: `exp(-inf) = 0` zera o triângulo
  superior, garantindo que a posição `i` só atenda a tokens `0 … i`.

### Cross-Attention — Ponte Encoder → Decoder

```
Q  =  decoder_state  @ W_Q    # "pergunta" do Decoder
K  =  encoder_output @ W_K    # índice da memória do Encoder
V  =  encoder_output @ W_V    # conteúdo da memória do Encoder
```

- O Decoder consulta **toda** a sequência de entrada em cada passo gerado.

### Positional Encoding (sin / cos)

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
```

### Loop Auto-Regressivo

```python
sequencia = ["<START>"]
while sequencia[-1] != "<EOS>" and len(sequencia) <= max_steps:
    probs         = Decoder(sequencia, Z)   # P(w | contexto, Z)
    proximo_token = top_k_sample(probs)     # amostragem
    sequencia.append(proximo_token)
```

---

## 📊 Saída Esperada (detalhada)

A execução imprime três blocos principais:

| Bloco | O que mostra |
|---|---|
| **Instanciação** | Hiperparâmetros do modelo (camadas, d_model, vocab) |
| **Inferência** | Cada token gerado com seu `id` e probabilidade |
| **Verificações de Sanidade** | 3 asserções automáticas (máscara, shapes, soma de probs) |

As três verificações devem **sempre** retornar `True`. Se alguma falhar,
há um bug no fluxo de tensores.

---
