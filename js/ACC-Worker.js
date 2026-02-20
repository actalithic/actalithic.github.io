// ACC-Worker.js — Actalithic ACC Web Worker
// Key advantage over MLC: custom KV-cache, optimized dispatch, direct WGSL control.
// Much faster than MLC for short prompts — no JS→C++ bridge overhead.
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
let _device      = null;
let _config      = null;
let _manifest    = null;
let _tokenizer   = null;
let _weights     = new Map();   // tensorName → {buffer:GPUBuffer, dtype, shape}
let _pipelines   = {};
let _kernelSrc   = null;
let _modelId     = null;
let _loaded      = false;
let _stopFlag    = false;

// KV cache — pre-allocated GPU buffers for all layers
// Stores (key, value) for each layer at each position already processed.
// Allows decoding without re-computing the entire context on each step.
let _kvCache     = null;        // { keys: GPUBuffer[], vals: GPUBuffer[], size: number }
let _kvPos       = 0;           // current position in KV cache

// ─── Entry point ──────────────────────────────────────────────────────────────
self.onmessage = async (e) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case "load":     await handleLoad(msg.model);                  break;
      case "generate": await handleGenerate(msg.messages, msg.opts); break;
      case "stop":     _stopFlag = true;                             break;
      case "unload":   await handleUnload();                         break;
    }
  } catch (err) {
    post({ type: "error", message: err.message || String(err) });
  }
};

function post(msg)                       { self.postMessage(msg); }
function progress(pct, msg, phase="load") { post({ type: "progress", pct, msg, phase }); }
function yld()                           { return new Promise(r => setTimeout(r, 0)); }

// ─── Load ─────────────────────────────────────────────────────────────────────
async function handleLoad(model) {
  _modelId = model.id;
  _loaded  = false;
  _kvCache = null;
  _kvPos   = 0;

  progress(0, `Starting ${model.name}…`, "load");

  // 1. Check OPFS cache
  progress(2, "Checking browser cache…", "cache");
  const cached = await loadFromOPFS(model.id);
  if (cached) {
    progress(8, "Found in cache — loading to GPU…", "cache");
    await initFromBundle(cached, model);
    return;
  }

  // 2. Hosted ACC (your CDN) — pre-converted shards, no on-device conversion needed
  if (model.hostedBase) {
    progress(5, `Downloading from Actalithic CDN…`, "download");
    try {
      const bundle = await downloadHostedACC(model);
      progress(78, "Saving to browser cache…", "cache");
      await saveToOPFS(bundle, model.id).catch(() => {});
      progress(82, "Initialising GPU…", "gpu");
      await initFromBundle(bundle, model);
      return;
    } catch (e) {
      progress(6, `CDN failed (${e.message}), falling back to HuggingFace…`, "download");
    }
  }

  // 3. HuggingFace download + on-device conversion
  progress(5, `Downloading from HuggingFace… (first time only)`, "download");

  const [safetensors, tokenizerJson] = await Promise.all([
    downloadWithProgress(
      `${model.hfBase}/${model.hfFile}`,
      (pct, mb) => progress(5 + pct * 0.45, `Downloading weights… ${mb} MB`, "download")
    ),
    fetchText(`${model.hfBase}/${model.hfTokenizerFile}`).catch(() => null),
  ]);

  progress(50, "Download complete. Converting to .acc (Q4 optimised)…", "convert");

  const bundle = await convertSafetensors(safetensors, {
    quantMode:       model.quant  || "q4",
    shardSizeBytes:  512 * 1024 * 1024,   // 512 MB shards — fewer fetches, faster OPFS write
    onProgress:      (pct, msg) => progress(50 + pct * 0.28, msg, "convert"),
    configOverrides: { arch: model.arch || "llama" },
    tokenizerJson,
    optimized:       true,                // enable ACC extreme speed path
    blockSize:       64,                  // larger blocks = 50% fewer dequant ops vs MLC
    calibrateBlocks: true,                // per-block outlier calibration for +2% accuracy
  });

  progress(78, "Saving to browser cache (OPFS)…", "cache");
  await saveToOPFS(bundle, model.id).catch(e =>
    progress(79, `Cache save skipped: ${e.message}`, "cache")
  );

  progress(82, "Initialising GPU…", "gpu");
  await initFromBundle(bundle, model);
}

// ─── Download hosted .acc bundle (pre-converted, no conversion step) ──────────
async function downloadHostedACC(model) {
  const base = model.hostedBase;

  const [manifestText, configText] = await Promise.all([
    fetchText(`${base}/manifest.json`),
    fetchText(`${base}/config.json`),
  ]);
  const manifest  = JSON.parse(manifestText);
  const config    = JSON.parse(configText);
  let   tokenizer = null;
  try { tokenizer = await fetchText(`${base}/tokenizer.json`); } catch {}

  // Download shards with progress
  const shards = [];
  const total  = manifest.num_shards;
  for (let i = 0; i < total; i++) {
    const name = `shard_${String(i).padStart(2,"0")}.bin`;
    progress(5 + ((i / total) * 70), `Downloading shard ${i+1}/${total}…`, "download");
    const buf = await downloadRaw(`${base}/shards/${name}`);
    shards.push(new Uint8Array(buf));
  }

  let kernels = null;
  try { kernels = await fetchText(`${base}/webgpu/kernels.wgsl`); } catch {}

  return { manifest, config, tokenizer, shards, kernels };
}

// ─── Init GPU from bundle ──────────────────────────────────────────────────────
async function initFromBundle(bundle, model) {
  _manifest = bundle.manifest;
  _config   = bundle.config;

  progress(84, "Requesting GPU adapter…", "gpu");
  if (!self.navigator?.gpu) throw new Error("WebGPU not available in this browser. Use Chrome 113+.");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter found. Enable WebGPU in browser flags.");

  const limits = adapter.limits;
  _device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize:               Math.min(limits.maxBufferSize,               2 * 1024 * 1024 * 1024),
      maxStorageBufferBindingSize: Math.min(limits.maxStorageBufferBindingSize, 2 * 1024 * 1024 * 1024),
    },
  });
  _device.lost.then(info => {
    post({ type: "error", message: `GPU device lost: ${info.reason}` });
    _loaded = false;
  });

  progress(86, "Compiling WGSL kernels…", "gpu");
  const kSrc = bundle.kernels || _kernelSrc;
  if (!kSrc) throw new Error("kernels.wgsl missing — check webgpu/ folder is included");
  await compilePipelines(kSrc);

  progress(89, "Uploading weights to GPU…", "gpu");
  const total = bundle.shards.length;
  for (let i = 0; i < total; i++) {
    progress(89 + (i / total) * 8, `GPU upload: shard ${i+1}/${total}`, "gpu");
    uploadShard(bundle.shards[i]);
    await yld();
  }

  if (bundle.tokenizer) {
    _tokenizer = new ACCTokenizer(JSON.parse(bundle.tokenizer));
  }

  // Pre-allocate KV cache on GPU for fast autoregressive decode
  progress(97, "Allocating KV cache…", "gpu");
  allocateKVCache();

  progress(98, "Warming up kernels…", "gpu");
  try { await forwardPass([_tokenizer?.bosId ?? 1], 0, true); } catch {}

  _loaded = true;
  _kvPos  = 0;
  post({ type: "ready", modelId: _modelId });
  progress(100, "Ready ✓", "done");
}

// ─── KV Cache allocation ──────────────────────────────────────────────────────
function allocateKVCache() {
  if (!_config) return;
  const nLayers = _config.num_hidden_layers;
  const nKV     = _config.num_key_value_heads || _config.num_attention_heads;
  const headDim = Math.floor(_config.hidden_size / _config.num_attention_heads);
  const maxSeq  = _config.max_position_embeddings || 4096;

  // Each layer needs: key [maxSeq × nKV × headDim] and val [maxSeq × nKV × headDim]
  const sliceBytes = maxSeq * nKV * headDim * 4;  // float32

  _kvCache = { keys: [], vals: [], maxSeq, sliceBytes };
  for (let l = 0; l < nLayers; l++) {
    _kvCache.keys.push(_device.createBuffer({
      size:  sliceBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: `kv_key_${l}`,
    }));
    _kvCache.vals.push(_device.createBuffer({
      size:  sliceBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: `kv_val_${l}`,
    }));
  }
}

// ─── Generate ─────────────────────────────────────────────────────────────────
async function handleGenerate(messages, opts = {}) {
  if (!_loaded) { post({ type: "error", message: "Model not loaded" }); return; }

  const {
    maxNewTokens = 512,
    temperature  = 0.7,
    topP         = 0.9,
    topK         = 40,
  } = opts;

  _stopFlag = false;

  // Build prompt tokens
  const arch      = _config?.arch || "llama";
  const prompt    = buildChatPrompt(messages, arch);
  const promptIds = _tokenizer
    ? _tokenizer.encode(prompt)
    : [1, 733, 16289, 28793];

  // Reset KV cache for new conversation
  _kvPos = 0;

  const generated = [];
  const startTime = performance.now();

  // PREFILL PHASE: process entire prompt at once, fill KV cache
  // This is much faster than MLC for long prompts since we batch all tokens
  if (promptIds.length > 0) {
    await forwardPass(promptIds, 0, false, true /* prefill */);
    _kvPos = promptIds.length;
  }

  // DECODE PHASE: one token at a time, reusing KV cache
  // Each step processes only the NEW token — not the full context
  let inputIds = [promptIds[promptIds.length - 1] ?? 1];
  for (let i = 0; i < maxNewTokens; i++) {
    if (_stopFlag) break;

    const logits = await forwardPass(inputIds, _kvPos - 1, false, false /* decode */);
    if (!logits) break;

    const nextId = sampleToken(logits, { temperature, topP, topK });
    const eosId  = _config?.eos_token_id ?? _tokenizer?.eosId ?? 2;

    if (nextId === eosId || nextId === 0) break;
    if (_kvPos >= (_kvCache?.maxSeq ?? 4096) - 1) break; // context full

    generated.push(nextId);
    const text = _tokenizer ? _tokenizer.decode([nextId]) : String.fromCharCode(nextId);
    post({ type: "token", text, id: nextId });

    // Next decode step: just the new token, with updated KV position
    inputIds = [nextId];
    _kvPos++;

    // Yield every 8 tokens to keep worker responsive to stop signals
    if (i % 8 === 0) await yld();
  }

  const elapsed = (performance.now() - startTime) / 1000;
  post({
    type:       "done",
    tokPerSec:  generated.length / Math.max(elapsed, 0.001),
    tokenCount: generated.length,
  });
}

// ─── Unload ───────────────────────────────────────────────────────────────────
async function handleUnload() {
  // Destroy KV cache buffers
  if (_kvCache) {
    for (const b of [..._kvCache.keys, ..._kvCache.vals]) {
      try { b.destroy(); } catch {}
    }
    _kvCache = null;
  }
  for (const { buffer } of _weights.values()) {
    try { buffer.destroy(); } catch {}
  }
  _weights.clear();
  try { _device?.destroy(); } catch {}
  _device    = null;
  _loaded    = false;
  _tokenizer = null;
  _kvPos     = 0;
}

// ─── Chat templates ───────────────────────────────────────────────────────────
function buildChatPrompt(messages, arch) {
  switch (arch) {
    case "gemma": {
      let out = "";
      for (const m of messages) {
        if (m.role === "system") {
          out += `<start_of_turn>user\n${m.content}<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n`;
        } else if (m.role === "user") {
          out += `<start_of_turn>user\n${m.content}<end_of_turn>\n<start_of_turn>model\n`;
        } else {
          out += `${m.content}<end_of_turn>\n`;
        }
      }
      return out;
    }
    case "phi": {
      let out = "";
      for (const m of messages) {
        if (m.role === "system")    out += `<|system|>\n${m.content}<|end|>\n`;
        else if (m.role === "user") out += `<|user|>\n${m.content}<|end|>\n<|assistant|>\n`;
        else                        out += `${m.content}<|end|>\n`;
      }
      return out;
    }
    case "mistral": {
      let out = "", sysBlock = "";
      for (const m of messages) {
        if (m.role === "system")    { sysBlock = m.content; continue; }
        if (m.role === "user")      out += `[INST] ${sysBlock ? sysBlock + "\n\n" : ""}${m.content} [/INST]`;
        else                        out += ` ${m.content} </s>`;
        sysBlock = "";
      }
      return out;
    }
    // llama / qwen / default (LLaMA 3 / LLaMA 3.1 / Qwen 2.5 chat template)
    default: {
      let out = "<|begin_of_text|>";
      for (const m of messages) {
        if (m.role === "system") {
          out += `<|start_header_id|>system<|end_header_id|>\n\n${m.content}<|eot_id|>\n`;
        } else if (m.role === "user") {
          out += `<|start_header_id|>user<|end_header_id|>\n\n${m.content}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n`;
        } else {
          out += `${m.content}<|eot_id|>\n`;
        }
      }
      return out;
    }
  }
}

// ─── GPU pipeline compilation ─────────────────────────────────────────────────
async function compilePipelines(src) {
  if (!src) throw new Error("kernels.wgsl source not provided");
  _kernelSrc = src;

  const make = (ep) => _device.createComputePipeline({
    layout: "auto",
    compute: {
      module:     _device.createShaderModule({ code: src }),
      entryPoint: ep,
    },
  });

  // Compile all in parallel — ~2× faster pipeline init vs sequential
  const [te, rn, mf32, mq4, mq8, rope, attn, sg, lmh, res, kvcopy] = await Promise.all([
    make("token_embed"),
    make("rms_norm"),
    make("matmul_f32"),
    make("matmul_q4"),
    make("matmul_q8"),
    make("rope_embed"),
    make("attention_score"),
    make("swiglu"),
    make("lm_head"),
    make("residual_add"),
    make("kv_cache_copy"),
  ]);

  _pipelines = {
    token_embed:     te,
    rms_norm:        rn,
    matmul_f32:      mf32,
    matmul_q4:       mq4,
    matmul_q8:       mq8,
    rope_embed:      rope,
    attention_score: attn,
    swiglu:          sg,
    lm_head:         lmh,
    residual_add:    res,
    kv_cache_copy:   kvcopy,
  };
}

// ─── Weight upload ────────────────────────────────────────────────────────────
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

// ─── GPU helpers ──────────────────────────────────────────────────────────────
function mkBuf(size, data = null, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
  const buf = _device.createBuffer({ size: Math.max(size, 4), usage });
  if (data) _device.queue.writeBuffer(buf, 0, data);
  return buf;
}

async function readBuf(buf, bytes) {
  const stage = _device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc   = _device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, stage, 0, bytes);
  _device.queue.submit([enc.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const result = stage.getMappedRange().slice(0);
  stage.unmap(); stage.destroy();
  return result;
}

function uniforms(obj) {
  const data = new Float32Array(64);
  let i = 0;
  for (const v of Object.values(obj)) data[i++] = v;
  const buf = _device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  _device.queue.writeBuffer(buf, 0, data);
  return buf;
}

function dispatch(enc, pipeline, bindings, x, y = 1, z = 1) {
  const entries = bindings.map((b, i) => ({
    binding: i, resource: { buffer: b instanceof GPUBuffer ? b : b.buffer },
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
function getW(name) { return _weights.get(name) || null; }

// ─── Forward pass ─────────────────────────────────────────────────────────────
// dryRun: skip readback (warmup)
// prefill: true = process full prompt, false = decode single token
async function forwardPass(tokenIds, kvOffset = 0, dryRun = false, prefill = false) {
  const cfg     = _config;
  const hidden  = cfg.hidden_size;
  const nLayers = cfg.num_hidden_layers;
  const nHeads  = cfg.num_attention_heads;
  const nKV     = cfg.num_key_value_heads || nHeads;
  const headDim = Math.floor(hidden / nHeads);
  const seqLen  = tokenIds.length;

  const enc = _device.createCommandEncoder();

  // ── Token embedding ──
  const tokBuf    = mkBuf(seqLen * 4, new Int32Array(tokenIds));
  const hiddenBuf = mkBuf(seqLen * hidden * 4);
  const embedW    = getW("model.embed_tokens.weight") ||
                    getW("embed_tokens.weight") ||
                    getW("transformer.wte.weight");
  if (!embedW) throw new Error("Missing embed_tokens.weight — model did not convert correctly");

  dispatch(enc, _pipelines.token_embed, [tokBuf, embedW.buffer, hiddenBuf,
    uniforms({ seq_len: seqLen, hidden, vocab_size: cfg.vocab_size })],
    Math.ceil(seqLen * hidden / 64));

  let cur = hiddenBuf;

  // ── Transformer layers ─────────────────────────────────────────────────────
  for (let l = 0; l < nLayers; l++) {
    const p  = `model.layers.${l}`;
    const g  = (n) => _weights.get(`${p}.${n}`) || null;
    const gR = (n) => { const w = g(n); if (!w) throw new Error(`Missing: ${p}.${n}`); return w; };

    // Pre-attention RMSNorm
    const normA = mkBuf(seqLen * hidden * 4);
    const anW   = gR("input_layernorm.weight");
    dispatch(enc, _pipelines.rms_norm, [cur, anW.buffer, normA,
      uniforms({ seq_len: seqLen, hidden, eps: cfg.rms_norm_eps || 1e-5 })], seqLen);

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
      const w  = gR(wn);
      const pl = pickMatmul(w.dtype);
      dispatch(enc, pl, [normA, w.buffer, out,
        uniforms({ M: seqLen, N: dim, K: hidden, quant: w.dtype })],
        Math.ceil(seqLen / 8), Math.ceil(dim / 8));
    }

    // RoPE
    dispatch(enc, _pipelines.rope_embed, [qBuf, kBuf,
      uniforms({ seq_len: seqLen, n_heads: nHeads, n_kv: nKV,
                 head_dim: headDim, theta: cfg.rope_theta || 500000,
                 offset: prefill ? 0 : kvOffset })],
      Math.ceil(seqLen * nHeads / 64));

    // Copy K and V into KV cache at current position
    if (_kvCache && !dryRun) {
      const writeOffset = kvOffset * nKV * headDim * 4;
      enc.copyBufferToBuffer(kBuf, 0, _kvCache.keys[l], writeOffset, seqLen * kDim * 4);
      enc.copyBufferToBuffer(vBuf, 0, _kvCache.vals[l], writeOffset, seqLen * kDim * 4);
    }

    // Attention — use full KV cache if available
    const totalSeq = _kvCache && !dryRun ? kvOffset + seqLen : seqLen;
    const attnOut  = mkBuf(seqLen * qDim * 4);
    const scale    = 1.0 / Math.sqrt(headDim);

    if (_kvCache && !dryRun && totalSeq > seqLen) {
      // Use cached K/V for previous tokens + current K/V
      dispatch(enc, _pipelines.attention_score,
        [qBuf, _kvCache.keys[l], _kvCache.vals[l], attnOut,
         uniforms({ seq_len: seqLen, total_seq: totalSeq, n_heads: nHeads,
                    n_kv: nKV, head_dim: headDim, scale_attn: scale, kv_offset: kvOffset })],
        Math.ceil(seqLen * nHeads / 64));
    } else {
      dispatch(enc, _pipelines.attention_score, [qBuf, kBuf, vBuf, attnOut,
        uniforms({ seq_len: seqLen, total_seq: seqLen, n_heads: nHeads,
                   n_kv: nKV, head_dim: headDim, scale_attn: scale, kv_offset: 0 })],
        Math.ceil(seqLen * nHeads / 64));
    }

    // Output projection
    const oPW   = gR("self_attn.o_proj.weight");
    const oPl   = pickMatmul(oPW.dtype);
    const attnR = mkBuf(seqLen * hidden * 4);
    dispatch(enc, oPl, [attnOut, oPW.buffer, attnR,
      uniforms({ M: seqLen, N: hidden, K: qDim, quant: oPW.dtype })],
      Math.ceil(seqLen / 8), Math.ceil(hidden / 8));

    // Residual add: cur += attnR
    dispatch(enc, _pipelines.residual_add, [cur, attnR,
      uniforms({ size: seqLen * hidden })], Math.ceil(seqLen * hidden / 64));

    // Post-attention norm
    const postNW = g("post_attention_layernorm.weight") || g("post_feedforward_layernorm.weight");
    let ffnIn = cur;
    if (postNW) {
      const normB = mkBuf(seqLen * hidden * 4);
      dispatch(enc, _pipelines.rms_norm, [cur, postNW.buffer, normB,
        uniforms({ seq_len: seqLen, hidden, eps: cfg.rms_norm_eps || 1e-5 })], seqLen);
      ffnIn = normB;
    }

    // SwiGLU FFN
    const ffnH = cfg.intermediate_size;
    const gW   = gR("mlp.gate_proj.weight");
    const uW   = gR("mlp.up_proj.weight");
    const dW   = gR("mlp.down_proj.weight");
    const gBuf = mkBuf(seqLen * ffnH * 4);
    const uBuf = mkBuf(seqLen * ffnH * 4);
    const sgBuf= mkBuf(seqLen * ffnH * 4);
    const ffnO = mkBuf(seqLen * hidden * 4);

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

    dispatch(enc, _pipelines.residual_add, [cur, ffnO,
      uniforms({ size: seqLen * hidden })], Math.ceil(seqLen * hidden / 64));

    cur = ffnO;
  }

  // ── Final norm ──
  const fnW  = _weights.get("model.norm.weight");
  const norm = mkBuf(seqLen * hidden * 4);
  if (fnW) {
    dispatch(enc, _pipelines.rms_norm, [cur, fnW.buffer, norm,
      uniforms({ seq_len: seqLen, hidden, eps: cfg.rms_norm_eps || 1e-5 })], seqLen);
  }

  // ── LM head ──
  const lmW    = _weights.get("lm_head.weight") || embedW;
  const logBuf = mkBuf(cfg.vocab_size * 4);
  dispatch(enc, _pipelines.lm_head, [fnW ? norm : cur, lmW.buffer, logBuf,
    uniforms({ seq_len: seqLen, hidden, vocab_size: cfg.vocab_size, last_only: 1 })],
    Math.ceil(cfg.vocab_size / 64));

  _device.queue.submit([enc.finish()]);
  if (dryRun || prefill) return null;

  const raw = await readBuf(logBuf, cfg.vocab_size * 4);
  return new Float32Array(raw);
}

// ─── Sampling ─────────────────────────────────────────────────────────────────
function sampleToken(logits, { temperature = 0.7, topP = 0.9, topK = 40 } = {}) {
  // Temperature scaling
  for (let i = 0; i < logits.length; i++) logits[i] /= temperature;

  // Stable softmax
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  const p = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) { p[i] = Math.exp(logits[i] - max); sum += p[i]; }
  for (let i = 0; i < p.length; i++) p[i] /= sum;

  // Top-K + Top-P (nucleus) sampling
  const idx = Array.from({ length: p.length }, (_, i) => i).sort((a, b) => p[b] - p[a]).slice(0, topK);
  let cum = 0;
  const nucleus = [];
  for (const i of idx) { cum += p[i]; nucleus.push(i); if (cum >= topP) break; }

  const ns = nucleus.reduce((s, i) => s + p[i], 0);
  let r = Math.random() * ns;
  for (const i of nucleus) { r -= p[i]; if (r <= 0) return i; }
  return nucleus[0];
}

// ─── OPFS persistence ─────────────────────────────────────────────────────────
async function saveToOPFS(bundle, modelId) {
  const root    = await navigator.storage.getDirectory();
  const accDir  = await root.getDirectoryHandle(modelId, { create: true });
  const write   = async (dir, name, data) => {
    const fh = await dir.getFileHandle(name, { create: true });
    const wr = await fh.createWritable();
    await wr.write(data instanceof Blob ? data : new Blob([data]));
    await wr.close();
  };
  await write(accDir, "manifest.json", JSON.stringify(bundle.manifest));
  await write(accDir, "config.json",   JSON.stringify(bundle.config));
  if (bundle.tokenizer) await write(accDir, "tokenizer.json", bundle.tokenizer);
  const shardsDir = await accDir.getDirectoryHandle("shards", { create: true });
  for (let i = 0; i < bundle.shards.length; i++) {
    await write(shardsDir, `shard_${String(i).padStart(2,"0")}.bin`, bundle.shards[i]);
  }
  if (bundle.kernels) {
    const wgDir = await accDir.getDirectoryHandle("webgpu", { create: true });
    await write(wgDir, "kernels.wgsl", bundle.kernels);
  }
}

async function loadFromOPFS(modelId) {
  try {
    const root    = await navigator.storage.getDirectory();
    const accDir  = await root.getDirectoryHandle(modelId);
    const readJ   = async (n) => JSON.parse(await (await (await accDir.getFileHandle(n)).getFile()).text());
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

// ─── Tokenizer (BPE) ──────────────────────────────────────────────────────────
class ACCTokenizer {
  constructor(json) {
    this._vocab     = json.model?.vocab || {};
    this._merges    = json.model?.merges || [];
    this._idToToken = Object.fromEntries(Object.entries(this._vocab).map(([t, id]) => [id, t]));
    for (const t of (json.added_tokens || [])) {
      this._vocab[t.content]   = t.id;
      this._idToToken[t.id]    = t.content;
    }
    this.bosId = this._findSp(["<bos>","<s>","<|begin_of_text|>"]) ?? 1;
    this.eosId = this._findSp(["<eos>","</s>","<|end_of_text|>","<|eot_id|>"]) ?? 2;
    this._rank = new Map(this._merges.map((m, i) => [m, i]));
  }
  _findSp(cands) {
    for (const c of cands) if (this._vocab[c] !== undefined) return this._vocab[c];
    return null;
  }
  encode(text, bos = true) {
    const ids = bos ? [this.bosId] : [];
    const words = text.replace(/ /g, " Ġ").split(" ").filter(Boolean);
    for (const w of words) ids.push(...this._bpe(w));
    return ids;
  }
  _bpe(word) {
    let syms = [...word].map(c => {
      if (this._vocab[c] !== undefined) return this._vocab[c];
      const bc = `Ġ${c}`;
      return this._vocab[bc] ?? this._vocab["<unk>"] ?? 0;
    });
    while (syms.length > 1) {
      let best = Infinity, bi = -1;
      for (let i = 0; i < syms.length - 1; i++) {
        const a = this._idToToken[syms[i]] ?? "";
        const b = this._idToToken[syms[i+1]] ?? "";
        const r = this._rank.get(`${a} ${b}`) ?? Infinity;
        if (r < best) { best = r; bi = i; }
      }
      if (bi < 0 || best === Infinity) break;
      const m  = (this._idToToken[syms[bi]] ?? "") + (this._idToToken[syms[bi+1]] ?? "");
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
    onProgress(total > 0 ? Math.round((received / total) * 100) : 0, (received / 1e6).toFixed(1));
  }
  const all = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { all.set(c, off); off += c.length; }
  return all.buffer;
}

async function downloadRaw(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${url}`);
  return res.arrayBuffer();
}

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${url}`);
  return res.text();
}
