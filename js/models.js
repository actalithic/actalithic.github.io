// models.js — LocalLLM by Actalithic
// MLC models use WebLLM (Apache 2.0). ACC models use Actalithic ACC Engine.
// To host your own ACC models, edit modelurls.json — no code changes needed.

// ─── Load modelurls.json (non-blocking, applied to ACC_MODELS at runtime) ─────
let _modelUrlOverrides = {};
(async () => {
  try {
    const res = await fetch("./modelurls.json");
    if (res.ok) {
      const data = await res.json();
      for (const entry of (data.models || [])) {
        if (entry.id && entry.hostedBase) {
          _modelUrlOverrides[entry.id] = entry.hostedBase;
        }
      }
      // Re-apply overrides to already-defined ACC_MODELS
      for (const m of ACC_MODELS) {
        if (_modelUrlOverrides[m.id]) {
          m.hostedBase = _modelUrlOverrides[m.id];
        }
      }
    }
  } catch (e) { /* modelurls.json missing or parse error — use defaults */ }
})();

// ─── MLC models (via WebLLM / web-llm CDN) ───────────────────────────────────
export const MODELS = [
  {
    id: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    name: "Llama 3.2 3B",
    fullName: "Llama 3.2 3B Instruct by Meta",
    size: "2 GB", ram: "~3.5 GB", tier: "Light",
    creator: "Meta",
    source: "MLC AI",
    sourceUrl: "https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC",
    modelUrl:  "https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC",
    desc: "Fastest and lightest. Best for phones and low-RAM devices.",
    runability: "easy", ctx: 4096, mobileRecommended: true,
    engine: "mlc",
    tokPerDevice: {
      dedicatedGPU: "40–80 tok/s", steamDeck: "15–22 tok/s",
      laptopIGPU: "8–16 tok/s", phone: "4–10 tok/s",
      cpu: "1–3 tok/s", core: "15–20 tok/s"
    },
    tok_range: "15–40 tok/s"
  },
  {
    id: "Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
    name: "Mistral 7B",
    fullName: "Mistral 7B Instruct v0.3 by Mistral AI",
    size: "4 GB", ram: "~6.5 GB", tier: "Middle",
    creator: "Mistral AI",
    source: "MLC AI",
    sourceUrl: "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
    modelUrl:  "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
    desc: "Well-rounded 7B. Great balance of speed and quality for general tasks.",
    runability: "mid", ctx: 4096,
    engine: "mlc",
    tokPerDevice: {
      dedicatedGPU: "12–25 tok/s", steamDeck: "4–8 tok/s",
      laptopIGPU: "2–6 tok/s", phone: "May not load",
      cpu: "~1 tok/s", core: "7–12 tok/s"
    },
    tok_range: "7–25 tok/s"
  },
  {
    id: "DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC",
    name: "DeepSeek R1 8B",
    fullName: "DeepSeek R1 Distill Llama 8B by DeepSeek",
    size: "5 GB", ram: "~8 GB", tier: "Advanced",
    creator: "DeepSeek",
    source: "MLC AI",
    sourceUrl: "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC",
    modelUrl:  "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC",
    desc: "Reasoning model with step-by-step thinking. Needs dedicated GPU or ActalithicCore.",
    runability: "hard", ctx: 4096,
    engine: "mlc",
    tokPerDevice: {
      dedicatedGPU: "8–18 tok/s", steamDeck: "3–6 tok/s",
      laptopIGPU: "1–4 tok/s", phone: "Will likely crash",
      cpu: "~1 tok/s", core: "~5 tok/s"
    },
    tok_range: "5–18 tok/s"
  },
];

// ─── ACC-native models (Actalithic custom engine) ─────────────────────────────
// hostedBase: pre-compiled .acc shards served from Actalithic CDN (Google Drive / R2).
// Users download the ready-to-run ACC bundle — no on-device compilation needed.
// Admin: replace example.com with real CDN URLs after compiling via acc-converter.html
export const ACC_MODELS = [
  {
    id: "llama-3.2-3b.acc",
    name: "Llama 3.2 3B",
    fullName: "Llama 3.2 3B Instruct by Meta",
    size: "~2 GB", ram: "~2.5 GB", tier: "Light",
    creator: "Meta",
    source: "Actalithic ACC Engine",
    sourceUrl: "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct",
    modelUrl:  "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct",
    hostedBase: "https://example.com/models/llama-3.2-3b.acc",
    // Using Bartowski's public mirror — avoids Meta gated-repo 401 for on-device fallback
    hfRepo: "bartowski/Llama-3.2-3B-Instruct-GGUF",
    hfFile: "model.safetensors",
    hfTokenizerFile: "tokenizer.json",
    hfBase: "https://huggingface.co/unsloth/Llama-3.2-3B-Instruct/resolve/main",
    desc: "Llama 3.2 3B via ACC Engine. Fastest, lightest — great for all devices.",
    runability: "easy", ctx: 4096, mobileRecommended: true,
    engine: "acc", arch: "llama", quant: "q4",
    tokPerDevice: {
      dedicatedGPU: "40–80 tok/s", steamDeck: "15–22 tok/s",
      laptopIGPU: "8–16 tok/s", phone: "4–10 tok/s",
      cpu: "1–3 tok/s", core: "15–20 tok/s"
    },
    tok_range: "15–40 tok/s"
  },
  {
    id: "mistral-7b-instruct.acc",
    name: "Mistral 7B",
    fullName: "Mistral 7B Instruct v0.3 by Mistral AI",
    size: "~4 GB", ram: "~4.5 GB", tier: "Middle",
    creator: "Mistral AI",
    source: "Actalithic ACC Engine",
    sourceUrl: "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
    modelUrl:  "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
    hostedBase: "https://example.com/models/mistral-7b-instruct.acc",
    hfRepo: "mistralai/Mistral-7B-Instruct-v0.3",
    hfFile: "model.safetensors",
    hfTokenizerFile: "tokenizer.json",
    hfBase: "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main",
    desc: "Mistral 7B via ACC Engine. Fast, well-rounded, excellent for general tasks.",
    runability: "mid", ctx: 4096,
    engine: "acc", arch: "mistral", quant: "q4",
    tokPerDevice: {
      dedicatedGPU: "12–28 tok/s", steamDeck: "4–8 tok/s",
      laptopIGPU: "2–6 tok/s", phone: "May not load",
      cpu: "~1 tok/s", core: "7–14 tok/s"
    },
    tok_range: "7–28 tok/s"
  },
  {
    id: "deepseek-r1-8b.acc",
    name: "DeepSeek R1 8B",
    fullName: "DeepSeek R1 Distill Llama 8B by DeepSeek",
    size: "~5 GB", ram: "~6 GB", tier: "Advanced",
    creator: "DeepSeek",
    source: "Actalithic ACC Engine",
    sourceUrl: "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    modelUrl:  "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    hostedBase: "https://example.com/models/deepseek-r1-8b.acc",
    hfRepo: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    hfFile: "model.safetensors",
    hfTokenizerFile: "tokenizer.json",
    hfBase: "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/resolve/main",
    desc: "DeepSeek R1 8B via ACC Engine. Step-by-step reasoning. Needs 8 GB RAM.",
    runability: "hard", ctx: 4096,
    engine: "acc", arch: "llama", quant: "q4",
    tokPerDevice: {
      dedicatedGPU: "8–18 tok/s", steamDeck: "3–6 tok/s",
      laptopIGPU: "1–4 tok/s", phone: "Will likely crash",
      cpu: "~1 tok/s", core: "~5 tok/s"
    },
    tok_range: "5–18 tok/s"
  },
];

// ─── Helpers ──────────────────────────────────────────────────────────────────

export function getAllModels()    { return [...MODELS, ...ACC_MODELS, ...getACCRegistry()]; }
export function getModelById(id) { return getAllModels().find(m => m.id === id) || null; }
export function isACCModel(id)   { return getAllModels().find(m => m.id === id)?.engine === "acc"; }

// ─── User-registered ACC models (from converter page) ────────────────────────

const ACC_REGISTRY_KEY = "acc-model-registry";
export function getACCRegistry() {
  try { return JSON.parse(localStorage.getItem(ACC_REGISTRY_KEY) || "[]"); } catch { return []; }
}
export function registerACCModel(d) {
  const r = getACCRegistry(), i = r.findIndex(m => m.id === d.id);
  if (i >= 0) r[i] = d; else r.push(d);
  localStorage.setItem(ACC_REGISTRY_KEY, JSON.stringify(r));
}
export function unregisterACCModel(id) {
  localStorage.setItem(ACC_REGISTRY_KEY, JSON.stringify(getACCRegistry().filter(m => m.id !== id)));
}

// ─── Display labels ───────────────────────────────────────────────────────────

export const TIER_LABELS = {
  Light:    { color: "var(--green)",  bg: "var(--green-bg)",  border: "var(--green-dim)"  },
  Middle:   { color: "var(--blue)",   bg: "var(--blue-bg)",   border: "var(--blue)"       },
  Advanced: { color: "var(--purple)", bg: "var(--purple-bg)", border: "var(--purple-dim)" },
};

export const RUN_LABELS = {
  easy:    { cls: "run-easy",    text: "Runs on most devices"               },
  mid:     { cls: "run-mid",     text: "Needs ~8 GB RAM"                   },
  hard:    { cls: "run-hard",    text: "Needs dedicated GPU or 8+ GB VRAM" },
  extreme: { cls: "run-extreme", text: "Needs 40+ GB VRAM"                 },
};
