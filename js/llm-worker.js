// llm-worker.js — WebLLM Web Worker handler for LocalLLM by Actalithic
// Offloads all model computation (shader compilation, weight loading,
// token generation) off the main UI thread so the UI stays responsive.
// Apache 2.0 — no additional dependencies beyond web-llm itself.

import * as webllm from "https://esm.run/@mlc-ai/web-llm";

// Register this script as a WebLLM worker handler.
// All messages from CreateWebWorkerMLCEngine are handled here.
const handler = new webllm.WebWorkerMLCEngineHandler();

self.onmessage = (msg) => {
  handler.onmessage(msg);
};
