// models.js — model registry for LocalLLM by Actalithic
// MLC models use WebLLM (Apache 2.0). ACC models use Actalithic ACC Engine.

// ─── MLC model (Llama — loads via WebLLM / llm-worker.js) ────────────────────
export const MODELS = [
  {
    id: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    name: "Llama 3.2 3B",
    fullName: "Llama 3.2 3B by Meta",
    size: "2 GB", ram: "~3.5 GB", tier: "Light",
    creator: "Meta",
    source: "Hugging Face / MLC AI",
    sourceUrl: "https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC",
    modelUrl:  "https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC",
    desc: "Fastest and lightest. Best for phones and low-RAM devices.",
    runability: "easy", ctx: 4096, mobileRecommended: true,
    engine: "mlc",
    tokPerDevice: {
      dedicatedGPU: "40-80 tok/s", steamDeck: "15-22 tok/s",
      laptopIGPU: "8-16 tok/s", phone: "4-10 tok/s",
      cpu: "1-3 tok/s", core: "15-20 tok/s"
    },
    tok_range: "15-40 tok/s"
  },
];

// ─── ACC-native models ────────────────────────────────────────────────────────
export const ACC_MODELS = [
  {
    id: "gemma-3-4b-it.acc",
    name: "Gemma 3 4B",
    fullName: "Gemma 3 4B Instruct by Google",
    size: "~3.3 GB", ram: "~5 GB", tier: "Light",
    creator: "Google",
    source: "Actalithic ACC Engine",
    sourceUrl: "https://huggingface.co/google/gemma-3-4b-it",
    hfRepo: "google/gemma-3-4b-it",
    hfFile: "model.safetensors",
    hfTokenizerFile: "tokenizer.json",
    hfBase: "https://huggingface.co/google/gemma-3-4b-it/resolve/main",
    desc: "Google Gemma 3 4B. Excellent instruction following, multilingual, fast for its size.",
    runability: "easy", ctx: 8192,
    engine: "acc", arch: "gemma", quant: "q4",
    tokPerDevice: {
      dedicatedGPU: "15-30 tok/s", steamDeck: "5-9 tok/s",
      laptopIGPU: "2-5 tok/s", phone: "Not recommended",
      cpu: "~1 tok/s", core: "8-14 tok/s"
    },
    tok_range: "8-30 tok/s"
  },
  {
    id: "Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
    name: "Mistral 7B",
    fullName: "Mistral 7B Instruct v0.3 by Mistral AI",
    size: "4 GB", ram: "~6.5 GB", tier: "Middle",
    creator: "Mistral AI",
    source: "Hugging Face / MLC AI",
    sourceUrl: "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
    modelUrl: "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
    desc: "Well-rounded 7B. Great balance of speed and quality for general tasks.",
    runability: "mid", ctx: 4096,
    engine: "mlc",
    tokPerDevice: {
      dedicatedGPU: "12-25 tok/s", steamDeck: "4-8 tok/s",
      laptopIGPU: "2-6 tok/s", phone: "May not load",
      cpu: "~1 tok/s", core: "7-12 tok/s"
    },
    tok_range: "7-25 tok/s"
  },
  {
    id: "qwen2.5-7b-instruct.acc",
    name: "Qwen 2.5 7B",
    fullName: "Qwen 2.5 7B Instruct by Alibaba",
    size: "~4.5 GB", ram: "~7 GB", tier: "Advanced",
    creator: "Alibaba",
    source: "Actalithic ACC Engine",
    sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
    hfRepo: "Qwen/Qwen2.5-7B-Instruct",
    hfFile: "model.safetensors",
    hfTokenizerFile: "tokenizer.json",
    hfBase: "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main",
    desc: "Alibaba Qwen 2.5 7B. Best-in-class coding, math and reasoning.",
    runability: "mid", ctx: 4096,
    engine: "acc", arch: "llama", quant: "q4",
    tokPerDevice: {
      dedicatedGPU: "10-22 tok/s", steamDeck: "4-7 tok/s",
      laptopIGPU: "2-4 tok/s", phone: "Will likely crash",
      cpu: "~1 tok/s", core: "6-12 tok/s"
    },
    tok_range: "6-22 tok/s"
  },
  {
    id: "DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC",
    name: "DeepSeek R1 8B",
    fullName: "DeepSeek R1 Distill Llama 8B by DeepSeek",
    size: "5 GB", ram: "~8 GB", tier: "Advanced",
    creator: "DeepSeek",
    source: "Hugging Face / MLC AI",
    sourceUrl: "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC",
    modelUrl: "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC",
    desc: "Reasoning model with thinking steps. Needs 8 GB RAM or ActalithicCore.",
    runability: "hard", ctx: 4096,
    engine: "mlc",
    tokPerDevice: {
      dedicatedGPU: "8-18 tok/s", steamDeck: "3-6 tok/s",
      laptopIGPU: "1-4 tok/s", phone: "Will likely crash",
      cpu: "~1 tok/s", core: "~5 tok/s"
    },
    tok_range: "5-18 tok/s"
  },
];

export function getAllModels()    { return [...MODELS, ...ACC_MODELS, ...getACCRegistry()]; }
export function getModelById(id) { return getAllModels().find(m => m.id === id) || null; }
export function isACCModel(id)   { return getAllModels().find(m => m.id === id)?.engine === "acc"; }

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

export const TIER_LABELS = {
  Light:    { color: "var(--green)",  bg: "var(--green-bg)",  border: "var(--green-dim)"  },
  Middle:   { color: "var(--blue)",   bg: "var(--blue-bg)",   border: "var(--blue)"       },
  Advanced: { color: "var(--purple)", bg: "var(--purple-bg)", border: "var(--purple-dim)" },
};

export const RUN_LABELS = {
  easy:    { cls: "run-easy",    text: "Runs on most devices"              },
  mid:     { cls: "run-mid",     text: "Needs ~8 GB RAM"                  },
  hard:    { cls: "run-hard",    text: "Needs dedicated GPU or 8+ GB VRAM" },
  extreme: { cls: "run-extreme", text: "Needs 40+ GB VRAM"                },
};
