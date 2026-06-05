import { useState, useEffect, useRef, useCallback } from "react";

const MODULES = [
  {
    id: "intro",
    title: "What is a Transformer?",
    icon: "⚡",
    color: "#6366f1",
    level: "Beginner",
    readTime: "8 min",
    sections: [
      {
        title: "The Big Picture",
        content: `Transformers are neural networks that process sequences by paying attention to every part simultaneously — unlike RNNs which process tokens one at a time.

Introduced in "Attention Is All You Need" (Vaswani et al., 2017), transformers replaced recurrent architectures entirely for most NLP tasks and later became the backbone of GPT, BERT, T5, and virtually every modern LLM.`,
      },
      {
        title: "Why Not RNNs?",
        content: `RNNs had two critical problems:
• Vanishing gradients — information from early tokens gets diluted over long sequences
• Sequential processing — token N can't be computed until token N-1 is done, making parallelization impossible

Transformers solve both by operating on the full sequence at once using self-attention.`,
      },
    ],
    quiz: {
      question: "What was the key innovation of 'Attention Is All You Need'?",
      options: [
        "Using more layers in neural networks",
        "Replacing recurrence with self-attention for sequence modeling",
        "Introducing dropout regularization",
        "Using larger training datasets",
      ],
      answer: 1,
    },
  },
  {
    id: "attention",
    title: "Self-Attention",
    icon: "👁",
    color: "#ec4899",
    level: "Core",
    readTime: "12 min",
    sections: [
      {
        title: "Query, Key, Value",
        content: `Every token generates three vectors:
• Query (Q) — "What am I looking for?"
• Key   (K) — "What do I contain?"
• Value (V) — "What do I contribute?"

Attention(Q,K,V) = softmax(QKᵀ / √dₖ) × V

The dot product QKᵀ measures how relevant each token is to each other token. Dividing by √dₖ stabilizes gradients for large dimensions.`,
      },
      {
        title: "Intuition",
        content: `Consider "The cat sat on the mat because it was tired."

When encoding "it", the model needs to figure out that "it" refers to "cat". Self-attention lets "it" look across the entire sentence and assign high attention weight to "cat", pulling in the right contextual meaning.

This is the core superpower: arbitrary long-range dependencies resolved in a single layer.`,
      },
    ],
    quiz: {
      question: "Why do we divide by √dₖ in the attention formula?",
      options: [
        "To normalize the output between 0 and 1",
        "To prevent softmax inputs from getting too large (gradient stability)",
        "To match the embedding dimension",
        "It's a design choice with no mathematical justification",
      ],
      answer: 1,
    },
  },
  {
    id: "multihead",
    title: "Multi-Head Attention",
    icon: "🔀",
    color: "#f59e0b",
    level: "Core",
    readTime: "10 min",
    sections: [
      {
        title: "Why Multiple Heads?",
        content: `A single attention head can only capture one type of relationship at a time. Multi-head attention runs h parallel attention operations:

MultiHead(Q,K,V) = Concat(head₁,...,headₙ) × Wᴼ
headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)

Each head learns to attend to different aspects: one might track syntactic dependencies, another semantic roles, another coreference.`,
      },
      {
        title: "Projections",
        content: `Each head uses separate learned weight matrices WᵢQ, WᵢK, WᵢV to project the inputs into lower-dimensional subspaces.

In "Attention Is All You Need": dmodel = 512, h = 8 heads, so dₖ = dᵥ = 64 per head. The final Wᴼ projects the concatenated output back to dmodel.

Total parameters per MHA layer ≈ 4 × dmodel²`,
      },
    ],
    quiz: {
      question: "In the original paper (dmodel=512, h=8), what is dₖ per head?",
      options: ["512", "256", "64", "128"],
      answer: 2,
    },
  },
  {
    id: "positional",
    title: "Positional Encoding",
    icon: "📍",
    color: "#10b981",
    level: "Core",
    readTime: "8 min",
    sections: [
      {
        title: "The Problem",
        content: `Self-attention is permutation-invariant — "cat sat mat" and "mat cat sat" produce identical attention patterns without positional information.

We need to inject token order while preserving the ability to compute attention efficiently.`,
      },
      {
        title: "Sinusoidal Encoding",
        content: `The original paper uses sinusoidal functions:

PE(pos, 2i)   = sin(pos / 10000^(2i/dmodel))
PE(pos, 2i+1) = cos(pos / 10000^(2i/dmodel))

Properties:
• Deterministic — no learned parameters
• Unique encoding per position
• Relative positions have consistent geometric relationship
• Generalizes to longer sequences than seen in training

Modern models (GPT, LLaMA) typically use learned or RoPE positional encodings instead.`,
      },
    ],
    quiz: {
      question: "What problem does positional encoding solve?",
      options: [
        "It reduces the number of parameters",
        "Self-attention is permutation-invariant and needs order information",
        "It speeds up training",
        "It prevents overfitting",
      ],
      answer: 1,
    },
  },
  {
    id: "ffn",
    title: "Feed-Forward Networks",
    icon: "🧠",
    color: "#8b5cf6",
    level: "Core",
    readTime: "6 min",
    sections: [
      {
        title: "The FFN Sub-layer",
        content: `After multi-head attention, each position passes through an identical feed-forward network independently:

FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

In the original paper: dmodel=512, inner dimension dff=2048 (4×).

The FFN operates position-wise — each token is processed independently. The MHA handles cross-token relationships; the FFN handles per-token transformations.`,
      },
      {
        title: "What Does the FFN Learn?",
        content: `Research suggests FFN layers act as key-value memories. The first layer's weights act as "keys" that activate on certain input patterns; the second layer projects the activated patterns to output space.

This is why scaling the FFN dimension (the 4× factor) significantly improves model capacity — it increases the number of facts/patterns the model can store.`,
      },
    ],
    quiz: {
      question: "What is the typical inner dimension of the FFN relative to dmodel?",
      options: ["Same (1×)", "2×", "4×", "8×"],
      answer: 2,
    },
  },
  {
    id: "encoder",
    title: "Encoder Architecture",
    icon: "📦",
    color: "#06b6d4",
    level: "Architecture",
    readTime: "10 min",
    sections: [
      {
        title: "Encoder Stack",
        content: `The encoder is a stack of N=6 identical layers. Each layer has:

1. Multi-Head Self-Attention sub-layer
2. Feed-Forward Network sub-layer

Both use residual connections and layer normalization:
output = LayerNorm(x + Sublayer(x))

The encoder reads the full input sequence bidirectionally — every token can attend to every other token. Used in BERT, RoBERTa.`,
      },
      {
        title: "Layer Normalization",
        content: `LayerNorm normalizes across the feature dimension (not the batch):

LayerNorm(x) = (x - μ) / (σ + ε) × γ + β

Why layer norm after residual addition? It prevents the model from being sensitive to the scale of inputs, stabilizing training considerably for deep networks.

Pre-norm vs Post-norm: Modern LLMs often use Pre-LN (normalize before sublayer) for better gradient flow.`,
      },
    ],
    quiz: {
      question: "How many identical layers does the original encoder have?",
      options: ["4", "6", "8", "12"],
      answer: 1,
    },
  },
  {
    id: "decoder",
    title: "Decoder Architecture",
    icon: "🎯",
    color: "#f97316",
    level: "Architecture",
    readTime: "12 min",
    sections: [
      {
        title: "Decoder Stack",
        content: `The decoder has N=6 layers, each with THREE sub-layers:

1. Masked Multi-Head Self-Attention
2. Cross-Attention (encoder-decoder attention)
3. Feed-Forward Network

The mask in step 1 prevents positions from attending to future tokens — essential for autoregressive generation.`,
      },
      {
        title: "Cross-Attention",
        content: `In the cross-attention sub-layer:
• Queries come from the decoder's previous sub-layer
• Keys and Values come from the encoder's output

This is how the decoder "reads" the source sequence while generating the target. At each decoding step, the decoder attends to all encoder positions.

GPT-style decoder-only models remove this layer entirely, keeping only masked self-attention + FFN.`,
      },
    ],
    quiz: {
      question: "What is the purpose of masking in the decoder's self-attention?",
      options: [
        "To speed up computation",
        "To prevent attending to future tokens (autoregressive constraint)",
        "To reduce memory usage",
        "To handle variable-length sequences",
      ],
      answer: 1,
    },
  },
  {
    id: "training",
    title: "Training Transformers",
    icon: "🏋️",
    color: "#84cc16",
    level: "Advanced",
    readTime: "14 min",
    sections: [
      {
        title: "Optimization",
        content: `The original paper uses Adam optimizer with a custom learning rate schedule:

lrate = dmodel^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))

Warmup steps = 4000. The rate increases linearly during warmup, then decreases proportionally to inverse square root of step number. This prevents instability in early training when gradients are noisy.`,
      },
      {
        title: "Regularization",
        content: `Three regularization techniques:

1. Residual Dropout — applied to sublayer outputs before addition (p=0.1)
2. Attention Dropout — applied to attention weights after softmax
3. Label Smoothing — instead of one-hot labels, use ε=0.1 (0.1/V distributed to other tokens)

Label smoothing hurts perplexity but improves BLEU score — the model learns to be less confident on any single prediction.`,
      },
    ],
    quiz: {
      question: "What does label smoothing do during training?",
      options: [
        "Makes the model more confident in its predictions",
        "Distributes a small probability mass across all vocabulary tokens",
        "Increases the learning rate",
        "Reduces the number of parameters",
      ],
      answer: 1,
    },
  },
  {
    id: "paper",
    title: "Attention Is All You Need",
    icon: "📄",
    color: "#a78bfa",
    level: "Paper",
    readTime: "20 min",
    sections: [
      {
        title: "Paper Overview",
        content: `Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017)

Published at NeurIPS 2017. One of the most cited ML papers of all time.

Key claims:
• Transformers outperform recurrent and convolutional models on WMT translation
• Training is significantly faster due to parallelism
• Attention heads learn interpretable linguistic relationships`,
      },
      {
        title: "Results & Impact",
        content: `WMT 2014 English-German: 28.4 BLEU (new SOTA at the time)
WMT 2014 English-French: 41.0 BLEU

Training cost: base model ~$300, big model ~$6000 on P100 GPUs (2017 pricing).

Impact: spawned BERT (2018), GPT (2018), GPT-2/3/4, T5, PaLM, LLaMA, Claude, and virtually every modern foundation model. Possibly the highest ROI paper in ML history.`,
      },
    ],
    quiz: {
      question: "What was the BLEU score on WMT 2014 English-German (big model)?",
      options: ["24.1", "26.4", "28.4", "31.0"],
      answer: 2,
    },
  },
];

const ATTENTION_TOKENS = ["The", "cat", "sat", "on", "mat"];

function AttentionViz({ activeToken, setActiveToken }) {
  const weights = [
    [0.9, 0.05, 0.02, 0.02, 0.01],
    [0.15, 0.7, 0.08, 0.04, 0.03],
    [0.06, 0.3, 0.52, 0.07, 0.05],
    [0.04, 0.06, 0.18, 0.6, 0.12],
    [0.3, 0.04, 0.06, 0.1, 0.5],
  ];
  const colors = ["#6366f1", "#ec4899", "#f59e0b", "#10b981", "#8b5cf6"];

  return (
    <div style={{ padding: "1.5rem" }}>
      <p style={{ color: "#94a3b8", fontSize: 13, marginBottom: "1.2rem", textAlign: "center" }}>
        Click a token to see its attention distribution
      </p>
      <div style={{ display: "flex", justifyContent: "center", gap: 12, marginBottom: "2rem" }}>
        {ATTENTION_TOKENS.map((tok, i) => (
          <div
            key={tok}
            onClick={() => setActiveToken(i)}
            style={{
              padding: "10px 18px",
              borderRadius: 8,
              border: `2px solid ${activeToken === i ? colors[i] : "rgba(255,255,255,0.1)"}`,
              background: activeToken === i ? colors[i] + "33" : "rgba(255,255,255,0.04)",
              color: activeToken === i ? colors[i] : "#e2e8f0",
              cursor: "pointer",
              fontWeight: activeToken === i ? 700 : 400,
              transition: "all 0.2s",
              fontSize: 15,
            }}
          >
            {tok}
          </div>
        ))}
      </div>
      <div style={{ display: "flex", justifyContent: "center", gap: 12 }}>
        {ATTENTION_TOKENS.map((tok, i) => {
          const w = weights[activeToken][i];
          return (
            <div key={tok} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
              <div
                style={{
                  width: 48,
                  height: 80,
                  borderRadius: 6,
                  background: `rgba(${activeToken === i ? "99,102,241" : "148,163,184"},${w})`,
                  border: `1px solid rgba(255,255,255,${w * 0.4})`,
                  display: "flex",
                  alignItems: "flex-end",
                  justifyContent: "center",
                  paddingBottom: 6,
                  transition: "all 0.4s cubic-bezier(0.4,0,0.2,1)",
                }}
              >
                <span style={{ fontSize: 11, color: "rgba(255,255,255,0.7)", fontWeight: 600 }}>
                  {(w * 100).toFixed(0)}%
                </span>
              </div>
              <span style={{ fontSize: 12, color: "#64748b" }}>{tok}</span>
            </div>
          );
        })}
      </div>
      <p style={{ textAlign: "center", color: "#94a3b8", fontSize: 12, marginTop: "1.5rem" }}>
        When <strong style={{ color: colors[activeToken] }}>&quot;{ATTENTION_TOKENS[activeToken]}&quot;</strong> is the query, it attends to each key token with the shown weights
      </p>
    </div>
  );
}

function ArchDiagram() {
  const [hoveredBlock, setHoveredBlock] = useState(null);
  const blocks = [
    { id: "embed", label: "Input Embedding", sublabel: "+ Positional Encoding", y: 380, color: "#10b981" },
    { id: "mha1", label: "Multi-Head Attention", sublabel: "(Masked in Decoder)", y: 290, color: "#6366f1" },
    { id: "add1", label: "Add & Norm", sublabel: "LayerNorm(x + sublayer)", y: 230, color: "#475569" },
    { id: "ffn", label: "Feed-Forward", sublabel: "2-layer MLP, ReLU", y: 160, color: "#8b5cf6" },
    { id: "add2", label: "Add & Norm", sublabel: "LayerNorm(x + sublayer)", y: 100, color: "#475569" },
    { id: "out", label: "Linear + Softmax", sublabel: "Vocabulary logits", y: 20, color: "#ec4899" },
  ];

  return (
    <div style={{ position: "relative", height: 480, width: "100%" }}>
      <svg width="100%" height="480" viewBox="0 0 400 480" style={{ position: "absolute", top: 0, left: 0 }}>
        {blocks.map((b, i) => {
          if (i === blocks.length - 1) return null;
          const next = blocks[i + 1];
          return (
            <line
              key={b.id + "-arrow"}
              x1="200" y1={b.y + 38}
              x2="200" y2={next.y + 62}
              stroke="rgba(255,255,255,0.15)"
              strokeWidth="1.5"
              strokeDasharray="4 3"
            />
          );
        })}
        {blocks.map((b) => (
          <g key={b.id} onMouseEnter={() => setHoveredBlock(b.id)} onMouseLeave={() => setHoveredBlock(null)} style={{ cursor: "pointer" }}>
            <rect
              x="80" y={b.y}
              width="240" height="50"
              rx="8"
              fill={hoveredBlock === b.id ? b.color + "33" : "rgba(15,23,42,0.8)"}
              stroke={hoveredBlock === b.id ? b.color : "rgba(255,255,255,0.1)"}
              strokeWidth={hoveredBlock === b.id ? 1.5 : 1}
              style={{ transition: "all 0.2s" }}
            />
            <text x="200" y={b.y + 20} textAnchor="middle" fill={hoveredBlock === b.id ? b.color : "#e2e8f0"} fontSize="13" fontWeight="600">
              {b.label}
            </text>
            <text x="200" y={b.y + 37} textAnchor="middle" fill="#64748b" fontSize="11">
              {b.sublabel}
            </text>
          </g>
        ))}
        <text x="10" y="240" fill="#334155" fontSize="11" style={{ writingMode: "vertical-rl", transform: "rotate(180deg) translateX(0px) translateY(-20px)" }}>
          ×N layers
        </text>
        <rect x="8" y="80" width="24" height="340" rx="4" fill="none" stroke="#1e293b" strokeWidth="1.5" strokeDasharray="3 3" />
      </svg>
    </div>
  );
}

function ModuleCard({ module, isCompleted, onToggle, onOpen }) {
  const levelColors = { Beginner: "#10b981", Core: "#6366f1", Architecture: "#f59e0b", Advanced: "#ec4899", Paper: "#a78bfa" };
  return (
    <div
      style={{
        background: isCompleted ? "rgba(16,185,129,0.06)" : "rgba(255,255,255,0.03)",
        border: `1px solid ${isCompleted ? "rgba(16,185,129,0.3)" : "rgba(255,255,255,0.08)"}`,
        borderRadius: 12,
        padding: "1.2rem",
        cursor: "pointer",
        transition: "all 0.2s",
        position: "relative",
        overflow: "hidden",
      }}
      onMouseEnter={(e) => { e.currentTarget.style.borderColor = module.color + "66"; e.currentTarget.style.background = module.color + "0d"; }}
      onMouseLeave={(e) => { e.currentTarget.style.borderColor = isCompleted ? "rgba(16,185,129,0.3)" : "rgba(255,255,255,0.08)"; e.currentTarget.style.background = isCompleted ? "rgba(16,185,129,0.06)" : "rgba(255,255,255,0.03)"; }}
    >
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, ${module.color}, transparent)`, opacity: isCompleted ? 1 : 0.3 }} />
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 22 }}>{module.icon}</span>
          <div>
            <div style={{ fontWeight: 600, color: "#e2e8f0", fontSize: 14, lineHeight: 1.3 }}>{module.title}</div>
            <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>{module.readTime} read</div>
          </div>
        </div>
        <button
          onClick={(e) => { e.stopPropagation(); onToggle(); }}
          style={{
            width: 22, height: 22, borderRadius: "50%",
            border: `2px solid ${isCompleted ? "#10b981" : "rgba(255,255,255,0.2)"}`,
            background: isCompleted ? "#10b981" : "transparent",
            cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
            color: "white", fontSize: 11, flexShrink: 0,
          }}
        >
          {isCompleted && "✓"}
        </button>
      </div>
      <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
        <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 4, background: levelColors[module.level] + "22", color: levelColors[module.level], fontWeight: 600 }}>
          {module.level}
        </span>
      </div>
      <button
        onClick={onOpen}
        style={{
          width: "100%", padding: "8px", borderRadius: 8,
          background: module.color + "22", border: `1px solid ${module.color}44`,
          color: module.color, fontSize: 12, cursor: "pointer", fontWeight: 600,
        }}
      >
        {isCompleted ? "Review →" : "Study →"}
      </button>
    </div>
  );
}

function QuizBlock({ quiz, color }) {
  const [selected, setSelected] = useState(null);
  const [revealed, setRevealed] = useState(false);

  return (
    <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 12, padding: "1.5rem", marginTop: "2rem" }}>
      <div style={{ fontWeight: 700, color: color, fontSize: 13, marginBottom: "0.8rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>Quick Check</div>
      <div style={{ color: "#e2e8f0", fontWeight: 500, marginBottom: "1.2rem", fontSize: 15 }}>{quiz.question}</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {quiz.options.map((opt, i) => {
          let bg = "rgba(255,255,255,0.04)";
          let border = "rgba(255,255,255,0.1)";
          let textColor = "#cbd5e1";
          if (revealed) {
            if (i === quiz.answer) { bg = "rgba(16,185,129,0.15)"; border = "#10b981"; textColor = "#10b981"; }
            else if (i === selected && selected !== quiz.answer) { bg = "rgba(239,68,68,0.1)"; border = "#ef4444"; textColor = "#ef4444"; }
          } else if (i === selected) { bg = color + "22"; border = color; textColor = color; }
          return (
            <div
              key={i}
              onClick={() => !revealed && setSelected(i)}
              style={{ padding: "10px 14px", borderRadius: 8, border: `1px solid ${border}`, background: bg, color: textColor, fontSize: 14, cursor: revealed ? "default" : "pointer", transition: "all 0.2s" }}
            >
              {opt}
            </div>
          );
        })}
      </div>
      {!revealed && selected !== null && (
        <button
          onClick={() => setRevealed(true)}
          style={{ marginTop: 12, padding: "8px 20px", borderRadius: 8, background: color, border: "none", color: "white", fontWeight: 700, cursor: "pointer", fontSize: 13 }}
        >
          Check Answer
        </button>
      )}
      {revealed && (
        <div style={{ marginTop: 12, fontSize: 13, color: selected === quiz.answer ? "#10b981" : "#ef4444", fontWeight: 600 }}>
          {selected === quiz.answer ? "✓ Correct!" : `✗ Correct answer: "${quiz.options[quiz.answer]}"`}
        </div>
      )}
    </div>
  );
}

function ModuleReader({ module, onClose, onMarkComplete, isCompleted }) {
  const [section, setSection] = useState(0);

  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.85)", zIndex: 100,
      display: "flex", alignItems: "center", justifyContent: "center", padding: "1rem",
    }}>
      <div style={{
        background: "#0f172a", border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 16, width: "100%", maxWidth: 680, maxHeight: "90vh",
        overflow: "auto", position: "relative",
      }}>
        <div style={{ padding: "1.5rem", borderBottom: "1px solid rgba(255,255,255,0.08)", position: "sticky", top: 0, background: "#0f172a", zIndex: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <span style={{ fontSize: 26 }}>{module.icon}</span>
              <div>
                <div style={{ fontWeight: 700, color: "#e2e8f0", fontSize: 18 }}>{module.title}</div>
                <div style={{ fontSize: 12, color: "#64748b" }}>{module.readTime} · {module.level}</div>
              </div>
            </div>
            <button onClick={onClose} style={{ background: "rgba(255,255,255,0.08)", border: "none", color: "#94a3b8", borderRadius: 8, padding: "6px 12px", cursor: "pointer", fontSize: 13 }}>✕ Close</button>
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: "1rem" }}>
            {module.sections.map((s, i) => (
              <button
                key={i}
                onClick={() => setSection(i)}
                style={{
                  padding: "5px 12px", borderRadius: 6, border: "none", cursor: "pointer", fontSize: 12, fontWeight: 600,
                  background: section === i ? module.color : "rgba(255,255,255,0.06)",
                  color: section === i ? "white" : "#64748b",
                }}
              >
                {s.title}
              </button>
            ))}
          </div>
        </div>

        <div style={{ padding: "2rem" }}>
          <h2 style={{ color: module.color, fontSize: 20, fontWeight: 700, marginBottom: "1.2rem" }}>
            {module.sections[section].title}
          </h2>
          <div style={{ color: "#cbd5e1", lineHeight: 1.85, fontSize: 15, whiteSpace: "pre-line" }}>
            {module.sections[section].content}
          </div>

          {module.id === "attention" && section === 0 && (
            <div style={{ marginTop: "2rem", background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 12, overflow: "hidden" }}>
              <div style={{ padding: "1rem 1.5rem", borderBottom: "1px solid rgba(255,255,255,0.06)", fontWeight: 700, color: "#ec4899", fontSize: 13 }}>INTERACTIVE — ATTENTION VISUALIZATION</div>
              <AttentionViz activeToken={0} setActiveToken={() => {}} />
            </div>
          )}

          {section === module.sections.length - 1 && (
            <QuizBlock quiz={module.quiz} color={module.color} />
          )}

          <div style={{ display: "flex", gap: 12, marginTop: "2rem" }}>
            {section > 0 && (
              <button onClick={() => setSection(s => s - 1)} style={{ flex: 1, padding: "10px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.1)", background: "transparent", color: "#94a3b8", cursor: "pointer", fontSize: 13 }}>
                ← Previous
              </button>
            )}
            {section < module.sections.length - 1 ? (
              <button onClick={() => setSection(s => s + 1)} style={{ flex: 1, padding: "10px", borderRadius: 8, border: "none", background: module.color, color: "white", cursor: "pointer", fontWeight: 700, fontSize: 13 }}>
                Continue →
              </button>
            ) : (
              <button
                onClick={() => { onMarkComplete(); onClose(); }}
                style={{ flex: 1, padding: "10px", borderRadius: 8, border: "none", background: isCompleted ? "rgba(16,185,129,0.3)" : "#10b981", color: "white", cursor: "pointer", fontWeight: 700, fontSize: 13 }}
              >
                {isCompleted ? "✓ Completed" : "Mark as Learned ✓"}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function TransformerHub() {
  const [completed, setCompleted] = useState({});
  const [activeModule, setActiveModule] = useState(null);
  const [tab, setTab] = useState("learn");
  const [attentionToken, setAttentionToken] = useState(2);
  const [particles, setParticles] = useState([]);
  const canvasRef = useRef(null);
  const animRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const pts = Array.from({ length: 40 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r: Math.random() * 1.5 + 0.5,
      color: ["#6366f1", "#ec4899", "#8b5cf6", "#06b6d4"][Math.floor(Math.random() * 4)],
    }));

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      pts.forEach((p) => {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = p.color + "88";
        ctx.fill();
      });
      pts.forEach((a, i) => pts.slice(i + 1).forEach((b) => {
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < 100) {
          ctx.beginPath();
          ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = `rgba(99,102,241,${(1 - d / 100) * 0.15})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }));
      animRef.current = requestAnimationFrame(draw);
    }
    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, []);

  const totalCompleted = Object.values(completed).filter(Boolean).length;
  const progress = Math.round((totalCompleted / MODULES.length) * 100);

  const tabs = [
    { id: "learn", label: "📚 Learn" },
    { id: "viz", label: "👁 Attention Viz" },
    { id: "arch", label: "🏗 Architecture" },
    { id: "todo", label: `✅ Progress (${totalCompleted}/${MODULES.length})` },
    { id: "paper", label: "📄 Paper" },
  ];

  return (
    <div style={{ minHeight: "100vh", background: "#060b18", fontFamily: "'JetBrains Mono', 'Fira Code', monospace", color: "#e2e8f0", position: "relative", overflow: "hidden" }}>
      <canvas ref={canvasRef} style={{ position: "fixed", inset: 0, width: "100%", height: "100%", pointerEvents: "none", opacity: 0.6 }} />

      <div style={{ position: "relative", zIndex: 1, maxWidth: 900, margin: "0 auto", padding: "0 1rem" }}>

        {/* Header */}
        <div style={{ textAlign: "center", padding: "3rem 0 2rem" }}>
          <div style={{ display: "inline-block", padding: "4px 14px", borderRadius: 20, background: "rgba(99,102,241,0.15)", border: "1px solid rgba(99,102,241,0.3)", color: "#818cf8", fontSize: 12, fontWeight: 700, marginBottom: "1rem", letterSpacing: "0.08em" }}>
            TRANSFORMER ARCHITECTURE · DEEP LEARNING
          </div>
          <h1 style={{ fontSize: "clamp(2rem, 6vw, 3.5rem)", fontWeight: 900, lineHeight: 1.1, marginBottom: "0.8rem", background: "linear-gradient(135deg, #e2e8f0 0%, #818cf8 50%, #ec4899 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            Attention Is All You Need
          </h1>
          <p style={{ color: "#64748b", fontSize: 15, maxWidth: 520, margin: "0 auto 1.5rem", lineHeight: 1.6 }}>
            A complete interactive guide to transformers — from self-attention to full architecture, with the original paper built in.
          </p>
          <div style={{ display: "flex", justifyContent: "center", gap: "2rem", flexWrap: "wrap", color: "#475569", fontSize: 13, marginBottom: "1.5rem" }}>
            <span>📖 {MODULES.length} modules</span>
            <span>⚡ {MODULES.reduce((a, m) => a + parseInt(m.readTime), 0)} min total</span>
            <span>🧪 {MODULES.length} quizzes</span>
          </div>
          <div style={{ maxWidth: 320, margin: "0 auto" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: 12, color: "#64748b" }}>
              <span>Progress</span><span>{progress}% complete</span>
            </div>
            <div style={{ height: 4, background: "rgba(255,255,255,0.08)", borderRadius: 2, overflow: "hidden" }}>
              <div style={{ height: "100%", width: `${progress}%`, background: "linear-gradient(90deg, #6366f1, #ec4899)", borderRadius: 2, transition: "width 0.5s" }} />
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, background: "rgba(255,255,255,0.04)", padding: 4, borderRadius: 12, border: "1px solid rgba(255,255,255,0.06)", overflowX: "auto", marginBottom: "2rem" }}>
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                flex: "1 0 auto", padding: "8px 14px", borderRadius: 8, border: "none", cursor: "pointer", fontSize: 12, fontWeight: 700, whiteSpace: "nowrap",
                background: tab === t.id ? "rgba(99,102,241,0.25)" : "transparent",
                color: tab === t.id ? "#818cf8" : "#64748b",
                transition: "all 0.2s",
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Learn Tab */}
        {tab === "learn" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 16 }}>
              {MODULES.filter(m => m.id !== "paper").map((module) => (
                <ModuleCard
                  key={module.id}
                  module={module}
                  isCompleted={!!completed[module.id]}
                  onToggle={() => setCompleted(c => ({ ...c, [module.id]: !c[module.id] }))}
                  onOpen={() => setActiveModule(module)}
                />
              ))}
            </div>
          </div>
        )}

        {/* Attention Viz Tab */}
        {tab === "viz" && (
          <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, overflow: "hidden" }}>
            <div style={{ padding: "1.5rem", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
              <h2 style={{ color: "#ec4899", fontWeight: 700, fontSize: 18, marginBottom: 4 }}>Self-Attention Playground</h2>
              <p style={{ color: "#64748b", fontSize: 13, lineHeight: 1.5 }}>
                See how each token attends to others. The height of each bar represents the attention weight — how much the query token &quot;looks at&quot; each key token.
              </p>
            </div>
            <AttentionViz activeToken={attentionToken} setActiveToken={setAttentionToken} />
            <div style={{ padding: "1.5rem", borderTop: "1px solid rgba(255,255,255,0.06)", background: "rgba(236,72,153,0.05)" }}>
              <div style={{ fontWeight: 700, color: "#ec4899", fontSize: 12, marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>Formula</div>
              <div style={{ fontFamily: "monospace", background: "rgba(0,0,0,0.4)", padding: "12px 16px", borderRadius: 8, fontSize: 13, color: "#a78bfa", border: "1px solid rgba(167,139,250,0.2)" }}>
                Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
              </div>
              <p style={{ color: "#475569", fontSize: 12, marginTop: 10, lineHeight: 1.6 }}>
                Each token produces Q, K, V vectors via learned projections. The dot product of the query with all keys gives relevance scores; softmax normalizes them into weights; the weighted sum of values is the output.
              </p>
            </div>
          </div>
        )}

        {/* Architecture Tab */}
        {tab === "arch" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
            <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, overflow: "hidden" }}>
              <div style={{ padding: "1.2rem", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                <h3 style={{ color: "#6366f1", fontWeight: 700, fontSize: 15 }}>Encoder Layer</h3>
                <p style={{ color: "#475569", fontSize: 12, marginTop: 4 }}>Bidirectional self-attention</p>
              </div>
              <ArchDiagram />
            </div>
            <div>
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: "1.5rem", marginBottom: 16 }}>
                <h3 style={{ color: "#f59e0b", fontWeight: 700, fontSize: 14, marginBottom: "1rem" }}>Key Dimensions (Base Model)</h3>
                {[
                  ["dmodel", "512", "Embedding / model dim"],
                  ["dff", "2048", "FFN inner dimension"],
                  ["h", "8", "Attention heads"],
                  ["dₖ = dᵥ", "64", "Per-head key/value dim"],
                  ["N", "6", "Encoder/decoder layers"],
                  ["Params", "~65M", "Total parameters"],
                ].map(([k, v, desc]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "7px 0", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                    <div>
                      <span style={{ fontFamily: "monospace", color: "#6366f1", fontWeight: 700, fontSize: 13 }}>{k}</span>
                      <span style={{ color: "#475569", fontSize: 11, marginLeft: 8 }}>{desc}</span>
                    </div>
                    <span style={{ color: "#e2e8f0", fontWeight: 700, fontFamily: "monospace", fontSize: 14 }}>{v}</span>
                  </div>
                ))}
              </div>
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: "1.5rem" }}>
                <h3 style={{ color: "#10b981", fontWeight: 700, fontSize: 14, marginBottom: "1rem" }}>Modern Variants</h3>
                {[
                  { name: "BERT", desc: "Encoder-only, bidirectional", color: "#10b981" },
                  { name: "GPT-4", desc: "Decoder-only, autoregressive", color: "#6366f1" },
                  { name: "T5", desc: "Full encoder-decoder", color: "#f59e0b" },
                  { name: "LLaMA", desc: "Decoder + RoPE + SwiGLU", color: "#ec4899" },
                ].map((m) => (
                  <div key={m.name} style={{ display: "flex", gap: 10, alignItems: "center", padding: "7px 0", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                    <span style={{ width: 8, height: 8, borderRadius: "50%", background: m.color, flexShrink: 0 }} />
                    <span style={{ color: m.color, fontWeight: 700, fontSize: 13, width: 60 }}>{m.name}</span>
                    <span style={{ color: "#475569", fontSize: 12 }}>{m.desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Todo / Progress Tab */}
        {tab === "todo" && (
          <div>
            <div style={{ background: "rgba(16,185,129,0.06)", border: "1px solid rgba(16,185,129,0.2)", borderRadius: 16, padding: "1.5rem", marginBottom: "1.5rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                <h2 style={{ color: "#10b981", fontWeight: 700, fontSize: 18 }}>Your Learning Progress</h2>
                <span style={{ fontFamily: "monospace", fontWeight: 900, fontSize: 22, color: progress === 100 ? "#10b981" : "#6366f1" }}>{progress}%</span>
              </div>
              <div style={{ height: 6, background: "rgba(255,255,255,0.08)", borderRadius: 3, overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${progress}%`, background: "linear-gradient(90deg, #10b981, #6366f1)", borderRadius: 3, transition: "width 0.6s" }} />
              </div>
              <div style={{ display: "flex", gap: "2rem", marginTop: "1rem", fontSize: 13, color: "#64748b" }}>
                <span>✅ {totalCompleted} completed</span>
                <span>📖 {MODULES.length - totalCompleted} remaining</span>
                <span>⏱ ~{MODULES.filter(m => !completed[m.id]).reduce((a, m) => a + parseInt(m.readTime), 0)} min left</span>
              </div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {MODULES.map((module) => (
                <div
                  key={module.id}
                  style={{
                    display: "flex", alignItems: "center", gap: 14, padding: "12px 16px",
                    background: completed[module.id] ? "rgba(16,185,129,0.06)" : "rgba(255,255,255,0.02)",
                    border: `1px solid ${completed[module.id] ? "rgba(16,185,129,0.25)" : "rgba(255,255,255,0.06)"}`,
                    borderRadius: 10, cursor: "pointer",
                    transition: "all 0.2s",
                  }}
                  onClick={() => setCompleted(c => ({ ...c, [module.id]: !c[module.id] }))}
                >
                  <div style={{
                    width: 24, height: 24, borderRadius: "50%", border: `2px solid ${completed[module.id] ? "#10b981" : "rgba(255,255,255,0.2)"}`,
                    background: completed[module.id] ? "#10b981" : "transparent",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    color: "white", fontSize: 12, flexShrink: 0, transition: "all 0.2s",
                  }}>
                    {completed[module.id] && "✓"}
                  </div>
                  <span style={{ fontSize: 18 }}>{module.icon}</span>
                  <div style={{ flex: 1 }}>
                    <div style={{ color: completed[module.id] ? "#94a3b8" : "#e2e8f0", fontWeight: 600, fontSize: 14, textDecoration: completed[module.id] ? "line-through" : "none" }}>
                      {module.title}
                    </div>
                    <div style={{ fontSize: 11, color: "#475569" }}>{module.level} · {module.readTime}</div>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); setActiveModule(module); }}
                    style={{ padding: "5px 12px", borderRadius: 6, border: `1px solid ${module.color}44`, background: "transparent", color: module.color, fontSize: 11, cursor: "pointer", fontWeight: 700 }}
                  >
                    {completed[module.id] ? "Review" : "Study"}
                  </button>
                </div>
              ))}
            </div>

            {totalCompleted === MODULES.length && (
              <div style={{ textAlign: "center", padding: "3rem", background: "rgba(99,102,241,0.08)", border: "1px solid rgba(99,102,241,0.2)", borderRadius: 16, marginTop: "1.5rem" }}>
                <div style={{ fontSize: 48, marginBottom: 12 }}>🎓</div>
                <h3 style={{ color: "#818cf8", fontWeight: 900, fontSize: 22, marginBottom: 8 }}>Transformer Master!</h3>
                <p style={{ color: "#64748b", fontSize: 14 }}>You&apos;ve completed all modules. You&apos;re ready for any interview question on transformers.</p>
              </div>
            )}
          </div>
        )}

        {/* Paper Tab */}
        {tab === "paper" && (
          <div>
            <div style={{ background: "rgba(167,139,250,0.08)", border: "1px solid rgba(167,139,250,0.2)", borderRadius: 16, padding: "2rem", marginBottom: "1.5rem" }}>
              <div style={{ display: "flex", gap: "1.5rem", alignItems: "flex-start", flexWrap: "wrap" }}>
                <div style={{ flex: "0 0 auto", background: "rgba(167,139,250,0.15)", border: "1px solid rgba(167,139,250,0.3)", borderRadius: 12, padding: "1rem 1.2rem", textAlign: "center" }}>
                  <div style={{ fontSize: 32 }}>📄</div>
                  <div style={{ color: "#a78bfa", fontWeight: 900, fontSize: 24, marginTop: 4 }}>2017</div>
                  <div style={{ color: "#64748b", fontSize: 11 }}>NeurIPS</div>
                </div>
                <div style={{ flex: 1, minWidth: 200 }}>
                  <h2 style={{ color: "#e2e8f0", fontWeight: 900, fontSize: 22, lineHeight: 1.3, marginBottom: 8 }}>
                    Attention Is All You Need
                  </h2>
                  <p style={{ color: "#64748b", fontSize: 13, lineHeight: 1.6 }}>
                    Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin — Google Brain / Google Research
                  </p>
                  <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer" style={{ display: "inline-block", marginTop: 10, padding: "6px 14px", borderRadius: 6, background: "rgba(167,139,250,0.15)", border: "1px solid rgba(167,139,250,0.3)", color: "#a78bfa", fontSize: 12, fontWeight: 700, textDecoration: "none" }}>
                    arXiv:1706.03762 ↗
                  </a>
                </div>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 12, marginBottom: "1.5rem" }}>
              {[
                ["BLEU EN-DE", "28.4", "WMT 2014, big model"],
                ["BLEU EN-FR", "41.0", "WMT 2014, SOTA"],
                ["Train Cost", "$300", "Base model, 2017 prices"],
                ["Citations", "100,000+", "One of ML's most cited"],
              ].map(([label, val, sub]) => (
                <div key={label} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 12, padding: "1.2rem" }}>
                  <div style={{ color: "#64748b", fontSize: 11, marginBottom: 4 }}>{label}</div>
                  <div style={{ color: "#a78bfa", fontWeight: 900, fontSize: 22 }}>{val}</div>
                  <div style={{ color: "#475569", fontSize: 11, marginTop: 4 }}>{sub}</div>
                </div>
              ))}
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {[
                { heading: "Abstract (paraphrased)", body: "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train." },
                { heading: "Key Contributions", body: "1. Self-attention for sequence modeling (no recurrence)\n2. Multi-head attention for attending to different representation subspaces\n3. Sinusoidal positional encodings\n4. Scaled dot-product attention with √dₖ normalization\n5. Encoder-decoder architecture achieves new SOTA on translation" },
                { heading: "Limitations Noted by Authors", body: "• Quadratic complexity in sequence length O(n²) for attention\n• Positional encoding may not generalize to very long sequences\n• Less inductive bias than CNNs for local structure\n• Requires more data to learn certain patterns that RNNs handle via inductive bias" },
                { heading: "What Changed After This Paper", body: "BERT (2018) — bidirectional pre-training on masked LM\nGPT-1/2/3/4 — scaled decoder-only transformers\nT5 — text-to-text framework, encoder-decoder\nLLaMA, Mistral — efficient open-source variants\nViT — transformers for vision (2020)\nAlphaFold 2 — transformers for protein folding (2021)" },
              ].map(({ heading, body }) => (
                <div key={heading} style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: "1.5rem" }}>
                  <h3 style={{ color: "#a78bfa", fontWeight: 700, fontSize: 13, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "0.8rem" }}>{heading}</h3>
                  <div style={{ color: "#94a3b8", fontSize: 14, lineHeight: 1.8, whiteSpace: "pre-line" }}>{body}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div style={{ height: "3rem" }} />
      </div>

      {activeModule && (
        <ModuleReader
          module={activeModule}
          isCompleted={!!completed[activeModule.id]}
          onClose={() => setActiveModule(null)}
          onMarkComplete={() => setCompleted(c => ({ ...c, [activeModule.id]: true }))}
        />
      )}
    </div>
  );
}
