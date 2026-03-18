import numpy as np

# HIPERPARÂMETROS GLOBAIS
np.random.seed(42)

D_MODEL    = 512
D_FF       = D_MODEL * 4   # 2048  (expansão interna do FFN)
N_LAYERS   = 6             # número de camadas no Encoder e no Decoder
VOCAB_SIZE = 10_000

# Vocabulário mínimo para a demo
id2word = {i: f"palavra_{i}" for i in range(VOCAB_SIZE)}
id2word[0] = "<START>"
id2word[1] = "<EOS>"
id2word[2] = "Thinking"
id2word[3] = "Machines"

word2id = {v: k for k, v in id2word.items()}

# Tabela de embeddings compartilhada (Encoder + Decoder)
EMBEDDING_TABLE = np.random.randn(VOCAB_SIZE, D_MODEL)

# TAREFA 1 — BLOCOS DE MONTAR (funções reutilizáveis)
# ── 1-A. Softmax numericamente estável ──────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax estável ao longo do último eixo."""
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x   = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


# ── 1-B. Layer Normalization ─────────────────────────────────
def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    LayerNorm: normaliza no último eixo (features).
    Fórmula: (x - mean) / sqrt(var + eps)
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var  = np.var( x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


# ── 1-C. Máscara causal (look-ahead mask) ───────────────────
def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Cria máscara causal [seq_len, seq_len].
      - 0      nas posições permitidas (diagonal e abaixo)
      - -inf   nas posições bloqueadas  (triângulo superior)
    """
    return np.where(
        np.triu(np.ones((seq_len, seq_len)), k=1) == 1,
        -np.inf,
        0.0
    )


# ── 1-D. Scaled Dot-Product Attention ───────────────────────
def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

    Shapes esperados:  (batch, seq, d_k/d_v)
    """
    d_k    = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)  # (batch, seq_q, seq_k)

    if mask is not None:
        scores = scores + mask                          # broadcast

    weights = softmax(scores)                           # (batch, seq_q, seq_k)
    return weights @ V                                  # (batch, seq_q, d_v)


# ── 1-E. Position-wise Feed-Forward Network ─────────────────
class FFN:
    """
    FFN(x) = max(0, x W1 + b1) W2 + b2
    Expande d_model → d_ff (ReLU) → d_model
    """
    def __init__(self, d_model: int = D_MODEL, d_ff: int = D_FF):
        # He-init para camadas com ReLU
        self.W1 = np.random.randn(d_model, d_ff)   * np.sqrt(2 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff,   d_model) * np.sqrt(2 / d_ff)
        self.b2 = np.zeros(d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return hidden @ self.W2 + self.b2


# ── 1-F. Positional Encoding (sin/cos) ──────────────────────
def positional_encoding(seq_len: int, d_model: int = D_MODEL) -> np.ndarray:
    """
    PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
    """
    pe  = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, np.newaxis]           # (seq_len, 1)
    i   = np.arange(0, d_model, 2)                    # índices pares

    pe[:, 0::2] = np.sin(pos / (10_000 ** (i / d_model)))
    pe[:, 1::2] = np.cos(pos / (10_000 ** (i / d_model)))
    return pe   # (seq_len, d_model)

# TAREFA 2 — PILHA DO ENCODER

class EncoderBlock:
    """
    Um bloco do Encoder:
      x → Self-Attention → Add&Norm → FFN → Add&Norm → saída
    """
    def __init__(self, d_model: int = D_MODEL):
        # Pesos de projeção para Q, K, V
        self.WQ  = np.random.randn(d_model, d_model) * 0.02
        self.WK  = np.random.randn(d_model, d_model) * 0.02
        self.WV  = np.random.randn(d_model, d_model) * 0.02
        self.ffn = FFN(d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, seq_len, d_model)
        retorna: (batch, seq_len, d_model)  — mesma shape
        """
        # 1. Self-Attention: Q, K, V todos derivados de x
        Q = x @ self.WQ
        K = x @ self.WK
        V = x @ self.WV

        att_out = scaled_dot_product_attention(Q, K, V, mask=None)

        # 2. Add & Norm  (residual connection)
        x = layer_norm(x + att_out)

        # 3. Feed-Forward Network
        ffn_out = self.ffn(x)

        # 4. Add & Norm
        x = layer_norm(x + ffn_out)

        return x


class EncoderStack:
    """Empilha N_LAYERS blocos de Encoder."""

    def __init__(self, n_layers: int = N_LAYERS, d_model: int = D_MODEL):
        self.layers = [EncoderBlock(d_model) for _ in range(n_layers)]

    def __call__(self, input_ids: list[int]) -> np.ndarray:
        """
        input_ids : lista de inteiros com os IDs dos tokens de entrada
        retorna   : Z de shape (1, seq_len, d_model)  — memória do Encoder
        """
        seq_len = len(input_ids)

        # Embedding + Positional Encoding
        x = EMBEDDING_TABLE[input_ids][np.newaxis, :, :]   # (1, seq_len, d_model)
        x = x + positional_encoding(seq_len)[np.newaxis, :, :]

        Z = x
        for block in self.layers:
            Z = block(Z)

        return Z   # (1, seq_len, d_model)
    
# TAREFA 3 — PILHA DO DECODER

class DecoderBlock:
    """
    Um bloco do Decoder:
      y → Masked Self-Att → Add&Norm
        → Cross-Att(Q=y, K=Z, V=Z) → Add&Norm
        → FFN → Add&Norm → saída
    """
    def __init__(self, d_model: int = D_MODEL):
        # Pesos para Masked Self-Attention
        self.WQ_self  = np.random.randn(d_model, d_model) * 0.02
        self.WK_self  = np.random.randn(d_model, d_model) * 0.02
        self.WV_self  = np.random.randn(d_model, d_model) * 0.02

        # Pesos para Cross-Attention
        self.WQ_cross = np.random.randn(d_model, d_model) * 0.02
        self.WK_cross = np.random.randn(d_model, d_model) * 0.02
        self.WV_cross = np.random.randn(d_model, d_model) * 0.02

        self.ffn = FFN(d_model)

    def __call__(self, y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        y : (batch, seq_dec, d_model)  — estado atual do Decoder
        Z : (batch, seq_enc, d_model)  — memória do Encoder
        retorna: (batch, seq_dec, d_model)
        """
        seq_dec = y.shape[1]

        # 1. Masked Self-Attention  (causal mask — não vê o futuro)
        mask = create_causal_mask(seq_dec)   # (seq_dec, seq_dec)
        Q_s  = y @ self.WQ_self
        K_s  = y @ self.WK_self
        V_s  = y @ self.WV_self
        self_att_out = scaled_dot_product_attention(Q_s, K_s, V_s, mask=mask)

        # 2. Add & Norm
        y = layer_norm(y + self_att_out)

        # 3. Cross-Attention: Q vem do Decoder, K e V vêm do Encoder (Z)
        Q_c       = y @ self.WQ_cross
        K_c       = Z @ self.WK_cross
        V_c       = Z @ self.WV_cross
        cross_out = scaled_dot_product_attention(Q_c, K_c, V_c, mask=None)

        # 4. Add & Norm
        y = layer_norm(y + cross_out)

        # 5. FFN + Add & Norm
        ffn_out = self.ffn(y)
        y = layer_norm(y + ffn_out)

        return y


class DecoderStack:
    """
    Empilha N_LAYERS blocos de Decoder e adiciona a projeção final
    para o vocabulário.
    """

    def __init__(
        self,
        n_layers:   int = N_LAYERS,
        d_model:    int = D_MODEL,
        vocab_size: int = VOCAB_SIZE
    ):
        self.layers = [DecoderBlock(d_model) for _ in range(n_layers)]
        # Camada linear: d_model → vocab_size
        self.W_proj = np.random.randn(d_model, vocab_size) * 0.02

    def __call__(
        self,
        target_ids: list[int],
        Z:          np.ndarray
    ) -> np.ndarray:
        """
        target_ids : IDs dos tokens já gerados (inclui <START>)
        Z          : memória do Encoder (1, seq_enc, d_model)

        retorna    : vetor de probabilidades (vocab_size,) para o próximo token
        """
        seq_len = len(target_ids)

        # Embedding + Positional Encoding do alvo
        y = EMBEDDING_TABLE[target_ids][np.newaxis, :, :]          # (1, seq_len, d_model)
        y = y + positional_encoding(seq_len)[np.newaxis, :, :]

        # Passa pelos N blocos do Decoder
        for block in self.layers:
            y = block(y, Z)

        # Pegar o vetor da ÚLTIMA posição (token mais recente gerado)
        ultimo_vetor = y[0, -1, :]                     # (d_model,)

        # Projeção linear → logits → Softmax
        logits = ultimo_vetor @ self.W_proj            # (vocab_size,)
        probs  = softmax(logits.reshape(1, -1)).flatten()

        return probs   # (vocab_size,)

# TAREFA 4 — INFERÊNCIA AUTO-REGRESSIVA

def sample_top_k(probs: np.ndarray, k: int = 50) -> int:
    """
    Top-K sampling: amostra dentre os k tokens mais prováveis.
    Garante variabilidade mantendo coerência com a distribuição.
    """
    top_k_idx   = np.argsort(probs)[-k:]
    top_k_probs = probs[top_k_idx]
    top_k_probs = top_k_probs / top_k_probs.sum()          # renormaliza
    chosen      = np.random.choice(len(top_k_idx), p=top_k_probs)
    return int(top_k_idx[chosen])


def autoregressive_inference(
    encoder_input_ids: list[int],
    encoder_stack:     EncoderStack,
    decoder_stack:     DecoderStack,
    max_steps:         int = 20,
    temperature:       float = 1.5,
    top_k:             int = 50,
    eos_at_step:       int = 5          # mock pedagógico: força <EOS> no passo N
) -> list[str]:
    """
    Loop auto-regressivo completo:
      1. Encoder processa a frase de entrada → Z
      2. Decoder inicia com <START> e gera tokens até <EOS> ou max_steps

    Estratégia de decodificação:
      - Temperatura aplicada nos logits para suavizar a distribuição
        (pesos aleatórios concentram tudo num único token sem isso).
      - Top-K sampling para variabilidade.
      - Mock pedagógico: força <EOS> no passo `eos_at_step` para demonstrar
        a condição de parada (em produção isso viria do modelo treinado).

    Retorna a sequência de palavras gerada (incluindo <START> e <EOS>).
    """
    print("TAREFA 4 — Loop Auto-Regressivo de Inferência")


    # ── Passo 1: Encoder ────────────────────────────────────
    print(f"\nEncoder processando: {[id2word[i] for i in encoder_input_ids]}")
    Z = encoder_stack(encoder_input_ids)
    print(f"Memória Z do Encoder shape: {Z.shape}")

    # ── Passo 2: Loop do Decoder ────────────────────────────
    generated  = ["<START>"]
    target_ids = [word2id["<START>"]]

    print(f"\nParâmetros: temperatura={temperature}, top_k={top_k}")
    print("Iniciando geração token a token:")
    print("-" * 40)

    for step in range(1, max_steps + 1):

        # Obter distribuição de probabilidades (logits divididos pela temperatura)
        probs = decoder_stack(target_ids, Z)

        # ── Mock pedagógico: forçar <EOS> no passo definido ──
        # Em produção, o modelo treinado aprenderia quando emitir <EOS>.
        if step == eos_at_step:
            probs          = np.zeros(VOCAB_SIZE)
            probs[word2id["<EOS>"]] = 1.0

        # Aplicar temperatura: torna a distribuição mais suave
        # (necessário pois pesos aleatórios concentram prob num único token)
        log_probs    = np.log(np.clip(probs, 1e-12, None))
        probs_scaled = softmax((log_probs / temperature).reshape(1, -1)).flatten()

        # Top-K sampling
        next_id    = sample_top_k(probs_scaled, k=top_k)
        next_token = id2word[next_id]

        print(f"  Passo {step:2d} | próximo token: '{next_token}' "
              f"(id={next_id}, prob={probs[next_id]:.4f})")

        generated.append(next_token)
        target_ids.append(next_id)

        # ── Condição de parada ────────────────────────────────
        if next_token == "<EOS>":
            print("\n✓ Token <EOS> detectado. Geração encerrada.")
            break
    else:
        print(f"\n⚠ Limite de {max_steps} passos atingido sem <EOS>.")

    return generated

# EXECUÇÃO PRINCIPAL

if __name__ == "__main__":

    print("Laboratório 04 — Transformer Completo 'From Scratch'")

    # ── Instanciar o modelo ──────────────────────────────────
    print("\nInstanciando Encoder e Decoder...")
    encoder = EncoderStack(n_layers=N_LAYERS, d_model=D_MODEL)
    decoder = DecoderStack(n_layers=N_LAYERS, d_model=D_MODEL, vocab_size=VOCAB_SIZE)
    print(f"  Encoder: {N_LAYERS} camadas | d_model={D_MODEL} | d_ff={D_FF}")
    print(f"  Decoder: {N_LAYERS} camadas | d_model={D_MODEL} | vocab_size={VOCAB_SIZE}")

    # ── Frase de entrada simulando "Thinking Machines" ───────
    encoder_sentence = ["Thinking", "Machines"]
    encoder_ids = [word2id.get(w, 0) for w in encoder_sentence]
    print(f"\nFrase de entrada (Encoder): {encoder_sentence}")
    print(f"IDs correspondentes       : {encoder_ids}")

    # ── Inferência auto-regressiva ───────────────────────────
    output_tokens = autoregressive_inference(
        encoder_input_ids = encoder_ids,
        encoder_stack     = encoder,
        decoder_stack     = decoder,
        max_steps         = 20
    )

    # ── Resultado final ──────────────────────────────────────
    print("SEQUÊNCIA FINAL GERADA:")
    print(" ".join(output_tokens))
    print()

    # ── Verificações de Sanidade ─────────────────────────────
    print("VERIFICAÇÕES DE SANIDADE")

    # 1. Máscara causal
    M = create_causal_mask(5)
    scores_test = np.random.randn(5, D_MODEL) @ np.random.randn(5, D_MODEL).T / np.sqrt(D_MODEL)
    weights_test = softmax(scores_test + M)
    upper = np.triu(np.ones((5, 5), dtype=bool), k=1)
    print(f"\n✓ Máscara causal: futuro zerado? {np.all(weights_test[upper] == 0.0)}")

    # 2. Shape preservado pelo Encoder
    Z_test = encoder([encoder_ids[0]])
    print(f"✓ Shape do Encoder preservado: {Z_test.shape}  "
          f"(esperado: (1, 1, {D_MODEL}))")

    # 3. Probabilidades do Decoder somam 1
    probs_test = decoder([word2id["<START>"]], Z_test)
    print(f"✓ Probabilidades do Decoder somam 1? "
          f"{np.isclose(probs_test.sum(), 1.0)} (soma={probs_test.sum():.6f})")
    print()