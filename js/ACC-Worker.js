// ACC-Worker.js — Actalithic ACC Web Worker
// Runs entirely off the main thread:
//   1. Downloads safetensors + tokenizer from HuggingFace
//   2. Converts to .acc format in-memory (quantized)
//   3. Runs forward pass inference via WebGPU
//   4. Streams tokens back to main thread
//
// Message protocol (main → worker):
//   { type: "load",     model: ModelDescriptor }
//   { type: "generate", messages: [...], opts: {...} }
//   { type: "stop" }
//   { type: "unload" }
//
// Message protocol (worker → main):
//   { type: "progress", pct: 0-100, msg: string, phase: string }
//   { type: "ready",    modelId: string }
//   { type: "token",    text: string, id: number }
//   { type: "done",     tokPerSec: number, tokenCount: number }
//   { type: "error",    message: string }

import { convertSafetensors, parseShard, DTYPE, ACC_VERSION } from "./acc-converter.js";

// ─── State ────────────────────────────────────────────────────────────────────

let _device      = null;   // GPUDevice
let _config      = null;   // model config
let _manifest    = null;   // .acc manifest
let _tokenizer   = null;   // ACCTokenizer
let _weights     = new Map(); // tensorName → {buffer, dtype, shape}
let _pipelines   = {};     // compiled compute pipelines
let _kernelSrc   = null;   // WGSL source
let _modelId     = null;
let _loaded      = false;
let _stopFlag    = false;
let _opfsBundle  = null;   // cached bundle reference

// ─── Entry: message handler ───────────────────────────────────────────────────

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case "load":     await handleLoad(msg.model);              break;
      case "generate": await handleGenerate(msg.messages, msg.opts); break;
      case "stop":     _stopFlag = true;                         break;
      case "unload":   await handleUnload();                     break;
    }
  } catch (err) {
    post({ type: "error", message: err.message });
  }
};

function post(msg) { self.postMessage(msg); }
function progress(pct, msg, phase = "load") {
  post({ type: "progress", pct, msg, phase });
}

// ─── Load ─────────────────────────────────────────────────────────────────────

// ─── Download source resolution ───────────────────────────────────────────────
// Priority: model.hostedBase (Actalithic CDN, coming soon) → model.hfBase (HuggingFace)
// When you're ready to ship your own CDN, set hostedBase on models in models.js.
// Users will then download pre-converted .acc shards directly — no HF, no conversion step.
function resolveDownloadUrl(model, filename) {
  if (model.hostedBase) {
    // Future Actalithic-hosted .acc shards — already quantized, no conversion needed
    return `${model.hostedBase}/${filename}`;
  }
  // Fallback: raw safetensors from HuggingFace (converted on-device)
  return `${model.hfBase}/${filename}`;
}

async function handleLoad(model) {
  _modelId = model.id;
  _loaded  = false;

  progress(0, `Starting ${model.name}…`, "load");

  // ── Try OPFS cache first ──────────────────────────────────────────────────
  progress(2, "Checking browser cache…", "cache");
  const cached = await loadFromOPFS(model.id);
  if (cached) {
    progress(8, "Found in cache — loading to GPU…", "cache");
    await initFromBundle(cached, model);
    return;
  }

  // ── Hosted .acc (Actalithic CDN) — future path ────────────────────────────
  if (model.hostedBase) {
    progress(5, `Downloading .acc from Actalithic CDN…`, "download");
    // Future: fetch pre-built manifest + shards directly, skip conversion entirely
    // For now this falls through to HF path until CDN is live
    progress(6, "CDN not yet live — falling back to HuggingFace…", "download");
  }

  // ── Download safetensors from HuggingFace ─────────────────────────────────
  progress(5, `Downloading from HuggingFace… (first time only — cached after this)`, "download");

  const [safetensors, tokenizerJson] = await Promise.all([
    downloadWithProgress(
      `${model.hfBase}/${model.hfFile}`,
      (pct, mb) => progress(5 + pct * 0.45, `Downloading weights… ${mb} MB`, "download")
    ),
    fetchText(`${model.hfBase}/${model.hfTokenizerFile}`).catch(() => null),
  ]);

  progress(50, "Download complete. Converting to .acc (Q4)…", "convert");

  const bundle = await convertSafetensors(safetensors, {
    quantMode:       model.quant || "q4",
    shardSizeBytes:  256 * 1024 * 1024,
    onProgress:      (pct, msg) => progress(50 + pct * 0.3, msg, "convert"),
    configOverrides: { arch: model.arch || "llama" },
    tokenizerJson,
    kernelsSrc:      _kernelSrc,
  });

  progress(80, "Saving to browser cache (OPFS)…", "cache");
  await saveToOPFS(bundle, model.id).catch(e =>
    progress(80, `Cache save skipped: ${e.message}`, "cache")
  );

  progress(84, "Initialising GPU…", "gpu");
  await initFromBundle(bundle, model);
}

async function initFromBundle(bundle, model) {
  _manifest = bundle.manifest;
  _config   = bundle.config;

  progress(86, "Requesting GPU adapter…", "gpu");
  if (!self.navigator?.gpu) throw new Error("WebGPU not available in Worker");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No GPU adapter found");

  _device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize:               2 * 1024 * 1024 * 1024,
      maxStorageBufferBindingSize: 2 * 1024 * 1024 * 1024,
    },
  });

  _device.lost.then(info => {
    post({ type: "error", message: `GPU lost: ${info.reason}` });
  });

  progress(88, "Compiling WGSL shaders…", "gpu");
  await compilePipelines(bundle.kernels || _kernelSrc);

  progress(90, "Uploading weights to GPU…", "gpu");
  const total = bundle.shards.length;
  for (let i = 0; i < total; i++) {
    progress(90 + (i / total) * 7, `GPU upload: shard ${i+1}/${total}`, "gpu");
    uploadShard(bundle.shards[i]);
    await yld();
  }

  if (bundle.tokenizer) {
    _tokenizer = new ACCTokenizer(JSON.parse(bundle.tokenizer));
  }

  progress(98, "Warming up…", "gpu");
  // Small warmup pass to trigger shader JIT
  try { await forwardPass([1], true); } catch {}

  _loaded = true;
  post({ type: "ready", modelId: _modelId });
  progress(100, "Ready ✓", "done");
}

// ─── Generate ────────────────────────────────────────────────────────────────

async function handleGenerate(messages, opts = {}) {
  if (!_loaded) { post({ type: "error", message: "Model not loaded" }); return; }

  const {
    maxNewTokens = 512,
    temperature  = 0.7,
    topP         = 0.9,
    topK         = 50,
  } = opts;

  _stopFlag = false;

  // Build prompt from messages (LLaMA-style chat template)
  const prompt = buildChatPrompt(messages, _config?.arch || "llama");

  const promptIds = _tokenizer
    ? _tokenizer.encode(prompt)
    : [1, 733, 16289, 28793]; // fallback BOS

  const generated  = [];
  const startTime  = performance.now();
  let   inputIds   = promptIds.slice();

  for (let i = 0; i < maxNewTokens; i++) {
    if (_stopFlag) break;

    const logits = await forwardPass(inputIds);
    if (!logits) break;

    const nextId = sampleToken(logits, { temperature, topP, topK });
    if (nextId === (_config?.eos_token_id ?? 2)) break;
    if (nextId === 0) break;

    generated.push(nextId);
    const text = _tokenizer ? _tokenizer.decode([nextId]) : "";
    post({ type: "token", text, id: nextId });

    // Next step: append new token (no KV cache yet — full re-run)
    inputIds = [...promptIds, ...generated];
  }

  const elapsed = (performance.now() - startTime) / 1000;
  post({
    type:       "done",
    tokPerSec:  generated.length / elapsed,
    tokenCount: generated.length,
  });
}

// ─── Unload ───────────────────────────────────────────────────────────────────

async function handleUnload() {
  for (const { buffer } of _weights.values()) {
    try { buffer.destroy(); } catch {}
  }
  _weights.clear();
  try { _device?.destroy(); } catch {}
  _device   = null;
  _loaded   = false;
  _tokenizer = null;
}

// ─── Chat template ────────────────────────────────────────────────────────────

function buildChatPrompt(messages, arch) {
  // LLaMA 3 / Mistral / Qwen / Phi-3 style
  const hasBos = arch === "llama" || arch === "mistral" || arch === "qwen";
  let out = hasBos ? "<|begin_of_text|>" : "";
  for (const m of messages) {
    if (m.role === "system") {
      out += arch === "gemma"
        ? `<start_of_turn>system\n${m.content}<end_of_turn>\n`
        : `<|start_header_id|>system<|end_header_id|>\n\n${m.content}<|eot_id|>\n`;
    } else if (m.role === "user") {
      out += arch === "gemma"
        ? `<start_of_turn>user\n${m.content}<end_of_turn>\n<start_of_turn>model\n`
        : `<|start_header_id|>user<|end_header_id|>\n\n${m.content}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n`;
    } else if (m.role === "assistant") {
      out += arch === "gemma"
        ? `${m.content}<end_of_turn>\n`
        : `${m.content}<|eot_id|>\n`;
    }
  }
  return out;
}

// ─── GPU: compile pipelines ───────────────────────────────────────────────────

async function compilePipelines(src) {
  if (!src) throw new Error("kernels.wgsl source not provided to ACC-Worker");
  _kernelSrc = src;

  const make = (ep) => {
    const mod = _device.createShaderModule({ code: src });
    return _device.createComputePipeline({
      layout: "auto",
      compute: { module: mod, entryPoint: ep },
    });
  };

  _pipelines = {
    token_embed:     make("token_embed"),
    rms_norm:        make("rms_norm"),
    matmul_f32:      make("matmul_f32"),
    matmul_q4:       make("matmul_q4"),
    matmul_q8:       make("matmul_q8"),
    rope_embed:      make("rope_embed"),
    attention_score: make("attention_score"),
    swiglu:          make("swiglu"),
    lm_head:         make("lm_head"),
    residual_add:    make("residual_add"),
  };
}

// ─── GPU: upload shard ────────────────────────────────────────────────────────

function uploadShard(shardBytes) {
  const tensors = parseShard(shardBytes);
  for (const t of tensors) {
    const buf = _device.createBuffer({
      size:  Math.max(t.data.byteLength, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: t.name,
    });
    _device.queue.writeBuffer(buf, 0, t.data);
    _weights.set(t.name, { buffer: buf, dtype: t.dtype, shape: t.shape });
  }
}

// ─── GPU: forward pass ────────────────────────────────────────────────────────

async function forwardPass(tokenIds, dryRun = false) {
  const cfg     = _config;
  const hidden  = cfg.hidden_size;
  const nLayers = cfg.num_hidden_layers;
  const nHeads  = cfg.num_attention_heads;
  const nKV     = cfg.num_key_value_heads || nHeads;
  const headDim = Math.floor(hidden / nHeads);
  const seqLen  = tokenIds.length;

  const enc = _device.createCommandEncoder();

  // ── Embedding ─────────────────────────────────────────────────────────────
  const tokBuf    = mkBuf(seqLen * 4,       new Int32Array(tokenIds));
  const hiddenBuf = mkBuf(seqLen * hidden * 4);
  const embedW    = getW("model.embed_tokens.weight") ||
                    getW("embed_tokens.weight") ||
                    getW("transformer.wte.weight");

  if (!embedW) throw new Error("Missing embed_tokens weight");
  dispatch(enc, _pipelines.token_embed, [tokBuf, embedW.buffer, hiddenBuf,
    uniforms({ seq_len: seqLen, hidden, vocab_size: cfg.vocab_size })], seqLen * hidden / 64);

  let cur = hiddenBuf;

  // ── Transformer layers ────────────────────────────────────────────────────
  for (let l = 0; l < nLayers; l++) {
    const p = `model.layers.${l}`;
    const g = (n) => {
      const w = _weights.get(`${p}.${n}`);
      return w || null;
    };
    const gReq = (n) => {
      const w = g(n);
      if (!w) throw new Error(`Missing weight: ${p}.${n}`);
      return w;
    };

    // Pre-attention RMSNorm
    const normA = mkBuf(seqLen * hidden * 4);
    const anW   = gReq("input_layernorm.weight");
    dispatch(enc, _pipelines.rms_norm, [cur, anW.buffer, normA,
      uniforms({ seq_len: seqLen, hidden, eps: cfg.rms_norm_eps })], seqLen);

    // Q K V projections
    const qDim = nHeads * headDim;
    const kDim = nKV * headDim;
    const qBuf = mkBuf(seqLen * qDim * 4);
    const kBuf = mkBuf(seqLen * kDim * 4);
    const vBuf = mkBuf(seqLen * kDim * 4);

    for (const [out, dim, wn] of [
      [qBuf, qDim, "self_attn.q_proj.weight"],
      [kBuf, kDim, "self_attn.k_proj.weight"],
      [vBuf, kDim, "self_attn.v_proj.weight"],
    ]) {
      const w  = gReq(wn);
      const pl = pickMatmul(w.dtype);
      dispatch(enc, pl, [normA, w.buffer, out,
        uniforms({ M: seqLen, N: dim, K: hidden, quant: w.dtype })],
        Math.ceil(seqLen / 8), Math.ceil(dim / 8));
    }

    // RoPE
    dispatch(enc, _pipelines.rope_embed, [qBuf, kBuf,
      uniforms({ seq_len: seqLen, n_heads: nHeads, n_kv: nKV,
                 head_dim: headDim, theta: cfg.rope_theta || 500000, offset: 0 })],
      Math.ceil(seqLen * nHeads / 64));

    // Attention
    const attnOut = mkBuf(seqLen * qDim * 4);
    const scale   = 1.0 / Math.sqrt(headDim);
    dispatch(enc, _pipelines.attention_score, [qBuf, kBuf, vBuf, attnOut,
      uniforms({ seq_len: seqLen, n_heads: nHeads, n_kv: nKV,
                 head_dim: headDim, scale_attn: scale })],
      Math.ceil(seqLen * nHeads / 64));

    // Output projection + residual
    const oPW    = gReq("self_attn.o_proj.weight");
    const oPl    = pickMatmul(oPW.dtype);
    const attnR  = mkBuf(seqLen * hidden * 4);
    dispatch(enc, oPl, [attnOut, oPW.buffer, attnR,
      uniforms({ M: seqLen, N: hidden, K: qDim, quant: oPW.dtype })],
      Math.ceil(seqLen / 8), Math.ceil(hidden / 8));

    // Residual add: cur += attnR
    dispatch(enc, _pipelines.residual_add, [cur, attnR,
      uniforms({ size: seqLen * hidden })], Math.ceil(seqLen * hidden / 64));

    // Post-attention RMSNorm
    const postNW = g("post_attention_layernorm.weight") || g("post_feedforward_layernorm.weight");
    let ffnIn = cur;
    if (postNW) {
      const normB = mkBuf(seqLen * hidden * 4);
      dispatch(enc, _pipelines.rms_norm, [cur, postNW.buffer, normB,
        uniforms({ seq_len: seqLen, hidden, eps: cfg.rms_norm_eps })], seqLen);
      ffnIn = normB;
    }

    // SwiGLU FFN
    const ffnH  = cfg.intermediate_size;
    const gW    = gReq("mlp.gate_proj.weight");
    const uW    = gReq("mlp.up_proj.weight");
    const dW    = gReq("mlp.down_proj.weight");
    const gBuf  = mkBuf(seqLen * ffnH * 4);
    const uBuf  = mkBuf(seqLen * ffnH * 4);
    const sgBuf = mkBuf(seqLen * ffnH * 4);
    const ffnO  = mkBuf(seqLen * hidden * 4);

    for (const [out, w] of [[gBuf, gW], [uBuf, uW]]) {
      const pl = pickMatmul(w.dtype);
      dispatch(enc, pl, [ffnIn, w.buffer, out,
        uniforms({ M: seqLen, N: ffnH, K: hidden, quant: w.dtype })],
        Math.ceil(seqLen / 8), Math.ceil(ffnH / 8));
    }
    dispatch(enc, _pipelines.swiglu, [gBuf, uBuf, sgBuf,
      uniforms({ size: seqLen * ffnH })], Math.ceil(seqLen * ffnH / 64));

    const dPl = pickMatmul(dW.dtype);
    dispatch(enc, dPl, [sgBuf, dW.buffer, ffnO,
      uniforms({ M: seqLen, N: hidden, K: ffnH, quant: dW.dtype })],
      Math.ceil(seqLen / 8), Math.ceil(hidden / 8));

    // Residual add: cur += ffnO
    dispatch(enc, _pipelines.residual_add, [cur, ffnO,
      uniforms({ size: seqLen * hidden })], Math.ceil(seqLen * hidden / 64));

    cur = ffnO; // updated residual stream
  }

  // ── Final norm ────────────────────────────────────────────────────────────
  const fnW  = _weights.get("model.norm.weight");
  const norm = mkBuf(seqLen * hidden * 4);
  if (fnW) {
    dispatch(enc, _pipelines.rms_norm, [cur, fnW.buffer, norm,
      uniforms({ seq_len: seqLen, hidden, eps: cfg.rms_norm_eps })], seqLen);
  } else { norm; /* skip */ }

  // ── LM head ───────────────────────────────────────────────────────────────
  const lmW     = _weights.get("lm_head.weight") || embedW;
  const logBuf  = mkBuf(cfg.vocab_size * 4);
  dispatch(enc, _pipelines.lm_head, [fnW ? norm : cur, lmW.buffer, logBuf,
    uniforms({ seq_len: seqLen, hidden, vocab_size: cfg.vocab_size, last_only: 1 })],
    Math.ceil(cfg.vocab_size / 64));

  _device.queue.submit([enc.finish()]);
  if (dryRun) return null;

  const raw = await readBuf(logBuf, cfg.vocab_size * 4);
  return new Float32Array(raw);
}

// ─── GPU helpers ──────────────────────────────────────────────────────────────

function mkBuf(size, data = null,
  usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
  const buf = _device.createBuffer({ size: Math.max(size, 4), usage });
  if (data) _device.queue.writeBuffer(buf, 0, data);
  return buf;
}

async function readBuf(buf, bytes) {
  const stage = _device.createBuffer({
    size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = _device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, stage, 0, bytes);
  _device.queue.submit([enc.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const result = stage.getMappedRange().slice(0);
  stage.unmap(); stage.destroy();
  return result;
}

function uniforms(obj) {
  const data = new Float32Array(64);
  Object.values(obj).forEach((v, i) => { data[i] = v; });
  const buf = _device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  _device.queue.writeBuffer(buf, 0, data);
  return buf;
}

function dispatch(enc, pipeline, bindings, x, y = 1, z = 1) {
  const entries = bindings.map((b, i) => ({
    binding: i,
    resource: b instanceof GPUBuffer ? { buffer: b } : { buffer: b },
  }));
  const bg   = _device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(x)), Math.max(1, y), z);
  pass.end();
}

function pickMatmul(dtype) {
  if (dtype === DTYPE.Q4) return _pipelines.matmul_q4;
  if (dtype === DTYPE.Q8) return _pipelines.matmul_q8;
  return _pipelines.matmul_f32;
}

function getW(name) {
  return _weights.get(name) || null;
}

// ─── Sampling ────────────────────────────────────────────────────────────────

function sampleToken(logits, { temperature = 0.7, topP = 0.9, topK = 50 } = {}) {
  for (let i = 0; i < logits.length; i++) logits[i] /= temperature;
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  const p = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) { p[i] = Math.exp(logits[i] - max); sum += p[i]; }
  for (let i = 0; i < p.length; i++) p[i] /= sum;

  const idx = Array.from({ length: p.length }, (_, i) => i).sort((a, b) => p[b] - p[a]).slice(0, topK);
  let cum = 0;
  const nuc = [];
  for (const i of idx) { cum += p[i]; nuc.push(i); if (cum >= topP) break; }

  const ns = nuc.reduce((s, i) => s + p[i], 0);
  let r = Math.random() * ns;
  for (const i of nuc) { r -= p[i]; if (r <= 0) return i; }
  return nuc[0];
}

// ─── OPFS ────────────────────────────────────────────────────────────────────

async function saveToOPFS(bundle, modelId) {
  const root   = await navigator.storage.getDirectory();
  const accDir = await root.getDirectoryHandle(`${modelId}`, { create: true });

  const writeFile = async (dir, name, data) => {
    const fh = await dir.getFileHandle(name, { create: true });
    const wr = await fh.createWritable();
    await wr.write(data instanceof Blob ? data : new Blob([data]));
    await wr.close();
  };

  await writeFile(accDir, "manifest.json", JSON.stringify(bundle.manifest));
  await writeFile(accDir, "config.json",   JSON.stringify(bundle.config));
  if (bundle.tokenizer) await writeFile(accDir, "tokenizer.json", bundle.tokenizer);

  const shardsDir = await accDir.getDirectoryHandle("shards", { create: true });
  for (let i = 0; i < bundle.shards.length; i++) {
    await writeFile(shardsDir, `shard_${String(i).padStart(2,"0")}.bin`, bundle.shards[i]);
  }

  if (bundle.kernels) {
    const wgpuDir = await accDir.getDirectoryHandle("webgpu", { create: true });
    await writeFile(wgpuDir, "kernels.wgsl", bundle.kernels);
  }
}

async function loadFromOPFS(modelId) {
  try {
    const root   = await navigator.storage.getDirectory();
    const accDir = await root.getDirectoryHandle(modelId);
    const readJ  = async (n) => JSON.parse(await (await (await accDir.getFileHandle(n)).getFile()).text());
    const manifest  = await readJ("manifest.json");
    const config    = await readJ("config.json");
    let   tokenizer = null;
    try { tokenizer = await (await (await accDir.getFileHandle("tokenizer.json")).getFile()).text(); } catch {}

    const shardsDir = await accDir.getDirectoryHandle("shards");
    const shards    = [];
    for (let i = 0; i < manifest.num_shards; i++) {
      const fh = await shardsDir.getFileHandle(`shard_${String(i).padStart(2,"0")}.bin`);
      shards.push(new Uint8Array(await (await fh.getFile()).arrayBuffer()));
    }

    let kernels = null;
    try {
      const wd = await accDir.getDirectoryHandle("webgpu");
      kernels  = await (await (await wd.getFileHandle("kernels.wgsl")).getFile()).text();
    } catch {}

    return { manifest, config, tokenizer, shards, kernels };
  } catch { return null; }
}

// ─── Tokenizer ────────────────────────────────────────────────────────────────

class ACCTokenizer {
  constructor(json) {
    this._vocab      = json.model?.vocab || {};
    this._merges     = json.model?.merges || [];
    this._idToToken  = Object.fromEntries(Object.entries(this._vocab).map(([t, id]) => [id, t]));
    for (const t of (json.added_tokens || [])) {
      this._vocab[t.content]      = t.id;
      this._idToToken[t.id]       = t.content;
    }
    this.bosId = this._findSp(["<bos>", "<s>", "<|begin_of_text|>"]) ?? 1;
    this.eosId = this._findSp(["<eos>", "</s>", "<|end_of_text|>", "<|eot_id|>"]) ?? 2;
    this._rank = new Map(this._merges.map((m, i) => [m, i]));
  }

  _findSp(candidates) {
    for (const c of candidates) {
      if (this._vocab[c] !== undefined) return this._vocab[c];
    }
    return null;
  }

  encode(text, bos = true) {
    const ids = bos ? [this.bosId] : [];
    // Simple character-level BPE with Ġ space prefix
    const words = text.replace(/ /g, " Ġ").split(" ").filter(Boolean);
    for (const w of words) {
      ids.push(...this._bpe(w));
    }
    return ids;
  }

  _bpe(word) {
    let syms = [...word].map(c => {
      if (this._vocab[c] !== undefined) return this._vocab[c];
      const b = c.charCodeAt(0);
      const bc = b < 256 ? `Ġ${c}` : c;
      return this._vocab[bc] ?? this._vocab["<unk>"] ?? 0;
    });

    while (syms.length > 1) {
      let best = Infinity, bi = -1;
      for (let i = 0; i < syms.length - 1; i++) {
        const a = this._idToToken[syms[i]]   ?? "";
        const b = this._idToToken[syms[i+1]] ?? "";
        const r = this._rank.get(`${a} ${b}`) ?? Infinity;
        if (r < best) { best = r; bi = i; }
      }
      if (bi < 0 || best === Infinity) break;
      const m = (this._idToToken[syms[bi]] ?? "") + (this._idToToken[syms[bi+1]] ?? "");
      const mid = this._vocab[m] ?? syms[bi];
      syms = [...syms.slice(0, bi), mid, ...syms.slice(bi + 2)];
    }
    return syms;
  }

  decode(ids) {
    return ids.map(id => this._idToToken[id] ?? "").join("").replace(/Ġ/g, " ").replace(/▁/g, " ");
  }
}

// ─── Fetch helpers ────────────────────────────────────────────────────────────

async function downloadWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${url}`);
  const total  = parseInt(res.headers.get("content-length") || "0");
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const pct = total > 0 ? Math.round((received / total) * 100) : 0;
    const mb  = (received / 1e6).toFixed(1);
    onProgress(pct, mb);
  }

  const all = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { all.set(c, offset); offset += c.length; }
  return all.buffer;
}

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${url}`);
  return res.text();
}

function yld() { return new Promise(r => setTimeout(r, 0)); }
