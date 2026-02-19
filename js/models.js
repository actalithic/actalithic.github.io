// models.js — model registry for LocalLLM by Actalithic

export const MODELS = [
  {
    id: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    name: "Llama 3.2 3B",
    fullName: "Llama 3.2 3B by Meta",
    size: "2 GB",
    ram: "~3.5 GB",
    tier: "Fast",
    creator: "Meta",
    source: "Hugging Face / MLC AI",
    sourceUrl: "https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC",
    modelUrl:  "https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC",
    desc: "Fastest and lightest. Best for phones and low-RAM devices.",
    runability: "easy",
    ctx: 4096,
    mobileRecommended: true,
    tokPerDevice: {
      dedicatedGPU: "30–80 tok/s",
      steamDeck:    "10–18 tok/s",
      laptopIGPU:   "5–12 tok/s",
      phone:        "3–8 tok/s",
      cpu:          "1–3 tok/s",
      core:         "Similar or faster"
    },
    tok_range: "10–35 tok/s"
  },
  {
    id: "Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
    name: "Mistral 7B",
    fullName: "Mistral 7B Instruct v0.3",
    size: "4 GB",
    ram: "~6.5 GB",
    tier: "Balanced",
    creator: "Mistral AI",
    source: "Hugging Face / MLC AI",
    sourceUrl: "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
    modelUrl:  "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
    desc: "High quality general assistant. Needs a modern GPU or ActalithicCore.",
    runability: "mid",
    ctx: 4096,
    tokPerDevice: {
      dedicatedGPU: "30–80 tok/s",
      steamDeck:    "3–6 tok/s",
      laptopIGPU:   "1–4 tok/s",
      phone:        "May not load",
      cpu:          "~1 tok/s",
      core:         "Similar or faster"
    },
    tok_range: "4–18 tok/s"
  },
  {
    id: "DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC",
    name: "DeepSeek R1 8B",
    fullName: "DeepSeek R1 Distill Llama 8B",
    size: "5 GB",
    ram: "~8 GB",
    tier: "Powerful",
    creator: "DeepSeek",
    source: "Hugging Face / MLC AI",
    sourceUrl: "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC",
    modelUrl:  "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC",
    desc: "Reasoning model with visible thinking steps. Great for logic & math.",
    runability: "hard",
    ctx: 4096,
    tokPerDevice: {
      dedicatedGPU: "30–80 tok/s",
      steamDeck:    "3–6 tok/s",
      laptopIGPU:   "1–4 tok/s",
      phone:        "Will likely crash",
      cpu:          "~1 tok/s",
      core:         "Similar or faster"
    },
    tok_range: "2–12 tok/s"
  },
];

export const RUN_LABELS = {
  easy:    { cls: "run-easy",    text: "Runs on most devices" },
  mid:     { cls: "run-mid",     text: "Needs ~8 GB RAM" },
  hard:    { cls: "run-hard",    text: "Needs dedicated GPU or 8+ GB VRAM" },
  extreme: { cls: "run-extreme", text: "Needs 40+ GB VRAM" },
};
