// models.js — model registry for LocalLLM by Actalithic
// To add a model later: copy an object block, fill in the fields, push to the array.

export const MODELS = [
  {
    id: "Llama-3.2-1B-Instruct-q4f32_1-MLC",
    name: "Llama 3.2 1B",
    fullName: "Llama 3.2 1B by Meta",
    size: "0.8 GB",
    ram: "~1.5 GB",
    tier: "Fast",
    creator: "Meta",
    source: "Hugging Face / MLC AI",
    sourceUrl: "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f32_1-MLC",
    modelUrl:  "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f32_1-MLC",
    desc: "Smallest and fastest. Great for quick questions on any device.",
    runability: "easy",
    ctx: 4096,
    tokPerDevice: {
      dedicatedGPU: "60–120 tok/s",
      steamDeck:    "20–40 tok/s",
      laptopIGPU:   "12–28 tok/s",
      phone:        "3–7 tok/s",
      cpu:          "2–5 tok/s",
      core:         "Similar or faster"
    },
    tok_range: "20–60 tok/s"
  },
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
    desc: "Balanced speed and quality. The best starting point for most users.",
    runability: "easy",
    ctx: 4096,
    tokPerDevice: {
      dedicatedGPU: "30–80 tok/s",
      steamDeck:    "10–18 tok/s",
      laptopIGPU:   "5–12 tok/s",
      phone:        "3–7 tok/s",
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
    id: "Llama-3.1-8B-Instruct-q4f16_1-MLC",
    name: "Llama 3.1 8B",
    fullName: "Llama 3.1 8B by Meta",
    size: "5 GB",
    ram: "~8 GB",
    tier: "Powerful",
    creator: "Meta",
    source: "Hugging Face / MLC AI",
    sourceUrl: "https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q4f16_1-MLC",
    modelUrl:  "https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q4f16_1-MLC",
    desc: "Meta's powerful 8B. Needs 8+ GB VRAM or ActalithicCore.",
    runability: "hard",
    ctx: 4096,
    tokPerDevice: {
      dedicatedGPU: "30–80 tok/s",
      steamDeck:    "3–6 tok/s",
      laptopIGPU:   "1–4 tok/s",
      phone:        "May not load",
      cpu:          "~1 tok/s",
      core:         "Similar or faster"
    },
    tok_range: "3–14 tok/s"
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
      phone:        "May not load",
      cpu:          "~1 tok/s",
      core:         "Similar or faster"
    },
    tok_range: "2–12 tok/s"
  },
  {
    id: "Llama-3.3-70B-Instruct-q2f16_1-MLC",
    name: "Llama 3.3 70B",
    fullName: "Llama 3.3 70B by Meta",
    size: "35 GB",
    ram: "~40 GB",
    tier: "Powerful",
    creator: "Meta",
    source: "Hugging Face / MLC AI",
    sourceUrl: "https://huggingface.co/mlc-ai/Llama-3.3-70B-Instruct-q2f16_1-MLC",
    modelUrl:  "https://huggingface.co/mlc-ai/Llama-3.3-70B-Instruct-q2f16_1-MLC",
    desc: "The largest available. Requires a high-end GPU with 40+ GB VRAM.",
    runability: "extreme",
    ctx: 2048,
    tokPerDevice: {
      dedicatedGPU: "1–4 tok/s",
      steamDeck:    "Will not load",
      laptopIGPU:   "Will not load",
      phone:        "Will not load",
      cpu:          "Will not load",
      core:         "Requires 40+ GB RAM"
    },
    tok_range: "1–4 tok/s"
  }
];

export const RUN_LABELS = {
  easy:    { cls: "run-easy",    text: "Runs on most devices" },
  mid:     { cls: "run-mid",     text: "Needs ~8 GB RAM" },
  hard:    { cls: "run-hard",    text: "Needs dedicated GPU or 8+ GB VRAM" },
  extreme: { cls: "run-extreme", text: "Needs 40+ GB VRAM" },
};
