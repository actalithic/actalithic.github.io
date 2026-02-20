// acc-converter.js — Actalithic .acc Format Converter
// Converts safetensors / GGUF model files to .acc format entirely in the browser.
// No backend. No server. Pure WebWorker + FileSystem Access API.
// Apache 2.0 — Actalithic

// ─── .acc Format Spec ────────────────────────────────────────────────────────
//
//  model.acc/
//  ├── manifest.json   → format version, arch, quant, metadata
//  ├── config.json     → layers, heads, vocab_size, rope settings etc
//  ├── tokenizer.json  → vocab + merge rules (BPE / SentencePiece passthrough)
//  ├── shards/
//  │   ├── shard_00.bin  → packed quantized weights
//  │   ├── shard_01.bin
//  │   └── ...
//  └── webgpu/
//      └── kernels.wgsl  → WGSL compute shaders (copied from acc-kernels.wgsl)
//
//  shard_XX.bin layout (per tensor):
//  ┌────────────────────────────────────────────────────────┐
//  │ [4B] name_len  → length of tensor name (UTF-8)        │
//  │ [N B] name     → tensor name string                   │
//  │ [1B] dtype     → 0=f32 1=f16 2=q8 3=q4               │
//  │ [1B] ndim      → number of dimensions (1-4)           │
//  │ [4B×ndim] shape→ each dimension as uint32             │
//  │ [4B] data_len  → byte length of weight data           │
//  │ [M B] data     → raw weight bytes (quantized if q4/q8)│
//  └────────────────────────────────────────────────────────┘

export const ACC_VERSION = "1.0.0";

// Max bytes per shard (256 MB default — fits in browser memory comfortably)
const SHARD_SIZE_BYTES = 256 * 1024 * 1024;

// ─── Quantization ────────────────────────────────────────────────────────────

export const DTYPE = { F32: 0, F16: 1, Q8: 2, Q4: 3 };

/**
 * Quantize a Float32Array to Q8 (symmetric per-block).
 * Block size 32 — one scale per 32 values.
 * Returns { data: Uint8Array, scales: Float32Array }
 */
export function quantizeQ8(f32, blockSize = 32) {
  const BLOCK   = Math.max(16, blockSize);
  const nBlocks = Math.ceil(f32.length / BLOCK);
  const out     = new Int8Array(f32.length);
  const scales  = new Float32Array(nBlocks);

  for (let b = 0; b < nBlocks; b++) {
    const start = b * BLOCK;
    const end   = Math.min(start + BLOCK, f32.length);
    let maxAbs  = 0;
    for (let i = start; i < end; i++) {
      const a = Math.abs(f32[i]);
      if (a > maxAbs) maxAbs = a;
    }
    const scale = maxAbs / 127.0;
    scales[b]   = scale;
    const inv   = scale > 0 ? 1 / scale : 0;
    for (let i = start; i < end; i++) {
      out[i] = Math.round(Math.max(-128, Math.min(127, f32[i] * inv)));
    }
  }
  return { data: new Uint8Array(out.buffer), scales };
}

/**
 * Quantize a Float32Array to Q4 (symmetric per-block, 4-bit packed).
 * Block size 32. Two values packed per byte.
 * Returns { data: Uint8Array, scales: Float32Array }
 */
export function quantizeQ4(f32, blockSize = 32, calibrate = false) {
  const BLOCK    = Math.max(16, blockSize);  // min 16, recommended 32 or 64
  const nBlocks  = Math.ceil(f32.length / BLOCK);
  const packedLen = Math.ceil(f32.length / 2);
  const out    = new Uint8Array(packedLen);
  const scales = new Float32Array(nBlocks);

  for (let b = 0; b < nBlocks; b++) {
    const start = b * BLOCK;
    const end   = Math.min(start + BLOCK, f32.length);

    // Calibration pass: use true max of absolute values (not just first pass)
    // This reduces clamping error significantly for activations with outliers.
    let maxAbs = 0;
    if (calibrate) {
      // Two-pass: find percentile-99 clamp to suppress extreme outliers
      const vals = [];
      for (let i = start; i < end; i++) vals.push(Math.abs(f32[i]));
      vals.sort((a,b) => a - b);
      // Use 99th-percentile to avoid outlier distortion
      const p99idx = Math.min(vals.length - 1, Math.floor(vals.length * 0.99));
      maxAbs = vals[p99idx] || 0;
    } else {
      for (let i = start; i < end; i++) {
        const a = Math.abs(f32[i]);
        if (a > maxAbs) maxAbs = a;
      }
    }

    // 4-bit signed: range -8..7 → scale to fit full dynamic range
    const scale = maxAbs / 7.0;
    scales[b]   = scale;
    const inv   = scale > 0 ? 1 / scale : 0;

    for (let i = start; i < end; i++) {
      const q = Math.round(Math.max(-8, Math.min(7, f32[i] * inv))) & 0xF;
      const byteIdx = Math.floor(i / 2);
      if (i % 2 === 0) out[byteIdx]  = q;
      else             out[byteIdx] |= (q << 4);
    }
  }
  return { data: out, scales };
}

/**
 * Convert Float32Array to Float16 Uint16Array.
 * Uses the standard bit-fiddle conversion.
 */
export function toFloat16(f32) {
  const out = new Uint16Array(f32.length);
  const buf = new DataView(new ArrayBuffer(4));
  for (let i = 0; i < f32.length; i++) {
    buf.setFloat32(0, f32[i], true);
    const x   = buf.getUint32(0, true);
    const s   = (x >>> 31) << 15;
    const exp = ((x >>> 23) & 0xFF) - 127 + 15;
    const mnt = (x >>> 12) & 0x7FF;
    if (exp <= 0)       out[i] = s;
    else if (exp >= 31) out[i] = s | 0x7C00;
    else                out[i] = s | (exp << 10) | mnt;
  }
  return out;
}

// ─── Safetensors Parser ───────────────────────────────────────────────────────

/**
 * Parse a safetensors file (single file format).
 * Returns { header: Object, tensorMap: Map<name, {dtype, shape, dataOffset, dataLen}> }
 *
 * Safetensors format:
 *   [8B little-endian uint64] header_size
 *   [header_size B]            JSON header
 *   [rest]                     raw tensor data
 */
export function parseSafetensors(buffer) {
  const view = new DataView(buffer);

  // Read header length (8 bytes, little-endian uint64)
  // JS DataView doesn't have getBigUint64 in all browsers, so read as two 32s
  const headerLenLo = view.getUint32(0, true);
  const headerLenHi = view.getUint32(4, true);
  if (headerLenHi !== 0) throw new Error("Header too large (>4GB)");
  const headerLen = headerLenLo;

  const headerBytes = new Uint8Array(buffer, 8, headerLen);
  const headerStr   = new TextDecoder().decode(headerBytes);
  const header      = JSON.parse(headerStr);

  const dataStart = 8 + headerLen;
  const tensorMap = new Map();

  for (const [name, meta] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    tensorMap.set(name, {
      dtype:      meta.dtype,          // "F32", "F16", "BF16", etc
      shape:      meta.shape,
      dataOffset: dataStart + meta.data_offsets[0],
      dataLen:    meta.data_offsets[1] - meta.data_offsets[0],
    });
  }

  return { header, tensorMap, dataStart, buffer };
}

/**
 * Extract a tensor's raw bytes from a parsed safetensors file.
 */
export function extractTensor(parsed, name) {
  const meta = parsed.tensorMap.get(name);
  if (!meta) return null;
  return new Uint8Array(parsed.buffer, meta.dataOffset, meta.dataLen);
}

/**
 * Get Float32Array view of a tensor (handles F32, F16, BF16).
 */
export function tensorToFloat32(parsed, name) {
  const meta = parsed.tensorMap.get(name);
  if (!meta) return null;
  const raw = new Uint8Array(parsed.buffer, meta.dataOffset, meta.dataLen);

  if (meta.dtype === "F32") {
    return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  }

  if (meta.dtype === "F16") {
    const u16 = new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2);
    const f32 = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      f32[i] = float16ToFloat32(u16[i]);
    }
    return f32;
  }

  if (meta.dtype === "BF16") {
    const u16 = new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2);
    const f32 = new Float32Array(u16.length);
    const tmp = new DataView(new ArrayBuffer(4));
    for (let i = 0; i < u16.length; i++) {
      tmp.setUint16(2, u16[i], false); // BF16 is just top 16 bits of F32
      f32[i] = tmp.getFloat32(0, false);
    }
    return f32;
  }

  throw new Error(`Unsupported dtype: ${meta.dtype}`);
}

function float16ToFloat32(h) {
  const s = (h >>> 15) & 1;
  const e = (h >>> 10) & 0x1F;
  const m =  h         & 0x3FF;
  if (e === 0)   return (s ? -1 : 1) * Math.pow(2, -14) * (m / 1024);
  if (e === 31)  return m ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + m / 1024);
}

// ─── Manifest / Config Detection ─────────────────────────────────────────────

/**
 * Detect architecture from safetensors header.
 * Returns "llama" | "mistral" | "phi" | "gemma" | "unknown"
 */
export function detectArch(tensorMap) {
  const names = [...tensorMap.keys()].join(" ");
  if (names.includes("model.layers.0.self_attn.q_proj")) {
    if (names.includes("model.embed_tokens")) return "llama";
  }
  if (names.includes("layers.0.attention.wq"))    return "llama_legacy";
  if (names.includes("model.layers.0.self_attn")) return "mistral";
  if (names.includes("transformer.h.0"))          return "phi";
  if (names.includes("model.layers.0.self_attn.q_proj") &&
      names.includes("model.embed_tokens.weight")) return "gemma";
  return "unknown";
}

/**
 * Build a config.json from a parsed safetensors header.
 * Infers layer count, hidden size, heads etc.
 */
export function inferConfig(tensorMap, arch) {
  const names = [...tensorMap.keys()];

  // Count transformer layers
  const layerNums = new Set();
  for (const n of names) {
    const m = n.match(/\.layers?\.(\d+)\./);
    if (m) layerNums.add(parseInt(m[1]));
  }
  const numLayers = layerNums.size || 32;

  // Hidden size from embed
  let hiddenSize = 4096;
  for (const [name, meta] of tensorMap) {
    if (name.includes("embed_tokens") && meta.shape.length === 2) {
      hiddenSize = meta.shape[1];
      break;
    }
  }

  // Vocab size
  let vocabSize = 32000;
  for (const [name, meta] of tensorMap) {
    if (name.includes("embed_tokens") && meta.shape.length === 2) {
      vocabSize = meta.shape[0];
      break;
    }
  }

  // Attention heads from q_proj
  let numHeads = 32;
  for (const [name, meta] of tensorMap) {
    if (name.includes("q_proj.weight") && meta.shape.length === 2) {
      numHeads = meta.shape[0] / 128;  // assume head_dim=128
      break;
    }
  }

  // Intermediate (FFN) size
  let intermediateSize = hiddenSize * 4;
  for (const [name, meta] of tensorMap) {
    if (name.includes("gate_proj.weight") && meta.shape.length === 2) {
      intermediateSize = meta.shape[0];
      break;
    }
  }

  return {
    arch,
    num_hidden_layers:  numLayers,
    hidden_size:        hiddenSize,
    num_attention_heads: Math.round(numHeads),
    num_key_value_heads: Math.round(numHeads),  // GQA override later if needed
    intermediate_size:  intermediateSize,
    vocab_size:         vocabSize,
    max_position_embeddings: 4096,
    rope_theta:         500000.0,
    rms_norm_eps:       1e-5,
    tie_word_embeddings: false,
    bos_token_id:       1,
    eos_token_id:       2,
  };
}

// ─── Shard Writer ─────────────────────────────────────────────────────────────

/**
 * Pack a single tensor into binary shard format.
 * Returns Uint8Array of bytes for this tensor entry.
 */
export function packTensor(name, dtype, shape, data) {
  const nameBytes  = new TextEncoder().encode(name);
  const shapeBuf   = new Uint32Array(shape);
  const totalBytes = 4 + nameBytes.length + 1 + 1 + 4 * shape.length + 4 + data.byteLength;

  const buf  = new ArrayBuffer(totalBytes);
  const view = new DataView(buf);
  const u8   = new Uint8Array(buf);

  let offset = 0;

  // name_len
  view.setUint32(offset, nameBytes.length, true); offset += 4;
  // name
  u8.set(nameBytes, offset); offset += nameBytes.length;
  // dtype
  view.setUint8(offset, dtype); offset += 1;
  // ndim
  view.setUint8(offset, shape.length); offset += 1;
  // shape
  for (const dim of shape) {
    view.setUint32(offset, dim, true); offset += 4;
  }
  // data_len
  view.setUint32(offset, data.byteLength, true); offset += 4;
  // data
  u8.set(new Uint8Array(data instanceof ArrayBuffer ? data : data.buffer,
                         data.byteOffset ?? 0,
                         data.byteLength), offset);

  return u8;
}

// ─── Main Conversion Pipeline ────────────────────────────────────────────────

/**
 * Convert a safetensors ArrayBuffer to a .acc directory structure.
 *
 * Options:
 *   quantMode: "f32" | "f16" | "q8" | "q4"  (default: "q4")
 *   shardSizeBytes: number                   (default: 256MB)
 *   onProgress: (pct: 0-100, msg: string) => void
 *   configOverrides: Partial<Config>         (override inferred config)
 *   tokenizerJson: string | null             (raw tokenizer.json content)
 *   kernelsSrc: string | null               (WGSL source from acc-kernels.wgsl)
 *
 * Returns: ACCBundle { manifest, config, tokenizer, shards: Uint8Array[], kernels }
 */
export async function convertSafetensors(buffer, opts = {}) {
  const {
    quantMode       = "q4",
    shardSizeBytes  = SHARD_SIZE_BYTES,
    onProgress      = () => {},
    configOverrides = {},
    tokenizerJson   = null,
    kernelsSrc      = null,
    // Extreme-speed optimisation flags (used by Actalithic presets)
    optimized       = false,    // enables all speed-oriented passes
    blockSize       = 32,       // Q4: 64, Q8: 32 — larger blocks = fewer dequant ops at inference
    calibrateBlocks = false,    // per-block min/max calibration for better Q4 accuracy
  } = opts;

  onProgress(0, "Parsing safetensors header…");
  const parsed = parseSafetensors(buffer);
  const { tensorMap } = parsed;

  onProgress(5, "Detecting architecture…");
  const arch   = detectArch(tensorMap);
  const config = { ...inferConfig(tensorMap, arch), ...configOverrides };

  onProgress(8, `Detected: ${arch} · ${config.num_hidden_layers} layers · ${config.hidden_size}d`);

  const manifest = {
    acc_version:  ACC_VERSION,
    arch,
    quant:        quantMode,
    num_shards:   0,           // filled in below
    created_at:   new Date().toISOString(),
    source:       "converted-by-actalithic-acc-converter",
    tensor_count: tensorMap.size,
    optimized:    optimized,
    block_size:   optimized ? blockSize : (quantMode === 'q4' ? 32 : 32),
  };

  // ── Quantize + shard tensors ────────────────────────────────────────────
  const shards  = [];
  let   shardBufs = [];   // array of Uint8Array chunks for current shard
  let   shardUsed = 0;

  function flushShard() {
    if (shardBufs.length === 0) return;
    // Concat all chunks for this shard
    const total = shardBufs.reduce((s, b) => s + b.byteLength, 0);
    const shard = new Uint8Array(total);
    let offset = 0;
    for (const chunk of shardBufs) {
      shard.set(chunk, offset);
      offset += chunk.byteLength;
    }
    shards.push(shard);
    shardBufs = [];
    shardUsed = 0;
  }

  const tensorNames = [...tensorMap.keys()];
  const total = tensorNames.length;

  for (let ti = 0; ti < total; ti++) {
    const name = tensorNames[ti];
    const meta = tensorMap.get(name);

    onProgress(
      10 + Math.round((ti / total) * 80),
      `Converting: ${name} [${meta.shape.join("×")}]`
    );

    // Yield to event loop — optimized mode yields less often for speed
    const yieldEvery = optimized ? 20 : 10;
    if (ti % yieldEvery === 0) await new Promise(r => setTimeout(r, 0));

    let packed;

    // Non-weight tensors (e.g. rope freqs) — keep as-is (F32 or F16)
    const isWeight = meta.shape.length >= 2;

    if (!isWeight || quantMode === "f32") {
      // Keep as F32
      const f32 = tensorToFloat32(parsed, name);
      packed = packTensor(name, DTYPE.F32, meta.shape, f32.buffer);

    } else if (quantMode === "f16") {
      const f32 = tensorToFloat32(parsed, name);
      const f16 = toFloat16(f32);
      packed = packTensor(name, DTYPE.F16, meta.shape, f16.buffer);

    } else if (quantMode === "q8") {
      const f32 = tensorToFloat32(parsed, name);
      const effBlock = optimized ? blockSize : 32;
      const { data, scales } = quantizeQ8(f32, effBlock);
      // Pack: scales first (F32), then quantized bytes
      const combined = new Uint8Array(scales.byteLength + data.byteLength);
      combined.set(new Uint8Array(scales.buffer), 0);
      combined.set(data, scales.byteLength);
      packed = packTensor(name, DTYPE.Q8, meta.shape, combined);

    } else { // q4 — use optimized path when enabled
      const f32 = tensorToFloat32(parsed, name);
      const effBlock    = optimized ? blockSize : 32;
      const effCalibrate = optimized && calibrateBlocks && isWeight;
      const { data, scales } = quantizeQ4(f32, effBlock, effCalibrate);
      const combined = new Uint8Array(scales.byteLength + data.byteLength);
      combined.set(new Uint8Array(scales.buffer), 0);
      combined.set(data, scales.byteLength);
      packed = packTensor(name, DTYPE.Q4, meta.shape, combined);
    }

    // Check if this tensor would overflow current shard
    if (shardUsed + packed.byteLength > shardSizeBytes && shardBufs.length > 0) {
      flushShard();
    }

    shardBufs.push(packed);
    shardUsed += packed.byteLength;
  }

  flushShard();  // flush last shard

  manifest.num_shards = shards.length;

  onProgress(95, `Packed ${total} tensors into ${shards.length} shard(s)`);

  const bundle = {
    manifest,
    config,
    tokenizer: tokenizerJson,
    shards,
    kernels: kernelsSrc,
  };

  onProgress(100, "Conversion complete ✓");
  return bundle;
}

// ─── Bundle → Files ──────────────────────────────────────────────────────────

/**
 * Serialize an ACCBundle to downloadable files.
 * Returns an array of { name, data: Blob } for download.
 *
 * Files:
 *   manifest.json, config.json, tokenizer.json (if present),
 *   shards/shard_00.bin, ...,
 *   webgpu/kernels.wgsl (if present)
 */
export function bundleToFiles(bundle) {
  const files = [];

  files.push({
    name: "manifest.json",
    data: new Blob([JSON.stringify(bundle.manifest, null, 2)], { type: "application/json" }),
  });

  files.push({
    name: "config.json",
    data: new Blob([JSON.stringify(bundle.config, null, 2)], { type: "application/json" }),
  });

  if (bundle.tokenizer) {
    files.push({
      name: "tokenizer.json",
      data: new Blob([bundle.tokenizer], { type: "application/json" }),
    });
  }

  for (let i = 0; i < bundle.shards.length; i++) {
    const idx = String(i).padStart(2, "0");
    files.push({
      name:  `shards/shard_${idx}.bin`,
      data:  new Blob([bundle.shards[i]], { type: "application/octet-stream" }),
    });
  }

  if (bundle.kernels) {
    files.push({
      name: "webgpu/kernels.wgsl",
      data: new Blob([bundle.kernels], { type: "text/plain" }),
    });
  }

  return files;
}

/**
 * Trigger browser download for each file in the bundle.
 * Prompts user to choose folder via FileSystem Access API if available,
 * otherwise falls back to individual file downloads.
 */
export async function downloadBundle(bundle, modelName = "model") {
  const files = bundleToFiles(bundle);

  // FileSystem Access API (Chrome 86+) — save to a real folder
  if (window.showDirectoryPicker) {
    try {
      const dirHandle = await window.showDirectoryPicker({
        suggestedName: `${modelName}.acc`,
        mode: "readwrite",
      });

      for (const file of files) {
        const parts   = file.name.split("/");
        let   handle  = dirHandle;

        // Create subdirectories (shards/, webgpu/)
        for (let i = 0; i < parts.length - 1; i++) {
          handle = await handle.getDirectoryHandle(parts[i], { create: true });
        }

        const fileHandle = await handle.getFileHandle(parts[parts.length - 1], { create: true });
        const writable   = await fileHandle.createWritable();
        await writable.write(file.data);
        await writable.close();
      }

      return { method: "filesystem", fileCount: files.length };
    } catch (e) {
      if (e.name !== "AbortError") {
        console.warn("FSA failed, falling back to download:", e);
      } else {
        throw e; // user cancelled
      }
    }
  }

  // Fallback: individual downloads (zip would be better but avoids dependency)
  for (const file of files) {
    const url  = URL.createObjectURL(file.data);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `${modelName}.acc_${file.name.replace("/", "_")}`;
    a.click();
    URL.revokeObjectURL(url);
    await new Promise(r => setTimeout(r, 100)); // small delay between downloads
  }

  return { method: "fallback", fileCount: files.length };
}

// ─── OPFS Cache ──────────────────────────────────────────────────────────────

/**
 * Save an ACCBundle to the browser's Origin Private File System.
 * This persists the model for offline use — no re-download needed.
 *
 * Usage: await saveToOPFS(bundle, "my-model")
 */
export async function saveToOPFS(bundle, modelName) {
  if (!navigator.storage?.getDirectory) {
    throw new Error("OPFS not supported in this browser");
  }

  const root    = await navigator.storage.getDirectory();
  const accDir  = await root.getDirectoryHandle(modelName + ".acc", { create: true });
  const files   = bundleToFiles(bundle);

  for (const file of files) {
    const parts  = file.name.split("/");
    let   handle = accDir;

    for (let i = 0; i < parts.length - 1; i++) {
      handle = await handle.getDirectoryHandle(parts[i], { create: true });
    }

    const fh   = await handle.getFileHandle(parts[parts.length - 1], { create: true });
    const wr   = await fh.createWritable();
    const buf  = await file.data.arrayBuffer();
    await wr.write(buf);
    await wr.close();
  }

  return { modelName, fileCount: files.length };
}

/**
 * Load an ACCBundle from OPFS by model name.
 * Returns null if not found.
 */
export async function loadFromOPFS(modelName) {
  try {
    const root   = await navigator.storage.getDirectory();
    const accDir = await root.getDirectoryHandle(modelName + ".acc");

    async function readJson(dir, name) {
      const fh   = await dir.getFileHandle(name);
      const file = await fh.getFile();
      return JSON.parse(await file.text());
    }

    async function readBin(dir, name) {
      const fh   = await dir.getFileHandle(name);
      const file = await fh.getFile();
      return new Uint8Array(await file.arrayBuffer());
    }

    const manifest  = await readJson(accDir, "manifest.json");
    const config    = await readJson(accDir, "config.json");
    let   tokenizer = null;
    try { tokenizer = await (await (await accDir.getFileHandle("tokenizer.json")).getFile()).text(); } catch {}

    const shardsDir = await accDir.getDirectoryHandle("shards");
    const shards    = [];
    for (let i = 0; i < manifest.num_shards; i++) {
      const idx = String(i).padStart(2, "0");
      shards.push(await readBin(shardsDir, `shard_${idx}.bin`));
    }

    let kernels = null;
    try {
      const wgpuDir = await accDir.getDirectoryHandle("webgpu");
      const kfh     = await wgpuDir.getFileHandle("kernels.wgsl");
      kernels       = await (await kfh.getFile()).text();
    } catch {}

    return { manifest, config, tokenizer, shards, kernels };
  } catch (e) {
    if (e.name === "NotFoundError") return null;
    throw e;
  }
}

/**
 * List all .acc models stored in OPFS.
 */
export async function listOPFSModels() {
  try {
    const root  = await navigator.storage.getDirectory();
    const names = [];
    for await (const [name] of root.entries()) {
      if (name.endsWith(".acc")) names.push(name.slice(0, -4));
    }
    return names;
  } catch { return []; }
}

// ─── Shard Deserializer ───────────────────────────────────────────────────────

/**
 * Parse a shard .bin file back into an array of tensor descriptors.
 * Returns: Array<{ name, dtype, shape, data: Uint8Array }>
 */
export function parseShard(shardBytes) {
  const view    = new DataView(shardBytes.buffer, shardBytes.byteOffset, shardBytes.byteLength);
  const tensors = [];
  let   offset  = 0;
  const u8      = shardBytes;

  while (offset < u8.byteLength) {
    // name_len
    if (offset + 4 > u8.byteLength) break;
    const nameLen = view.getUint32(offset, true); offset += 4;

    // name
    const name = new TextDecoder().decode(u8.slice(offset, offset + nameLen));
    offset += nameLen;

    // dtype + ndim
    const dtype = view.getUint8(offset); offset += 1;
    const ndim  = view.getUint8(offset); offset += 1;

    // shape
    const shape = [];
    for (let i = 0; i < ndim; i++) {
      shape.push(view.getUint32(offset, true)); offset += 4;
    }

    // data_len + data
    const dataLen = view.getUint32(offset, true); offset += 4;
    const data    = u8.slice(offset, offset + dataLen); offset += dataLen;

    tensors.push({ name, dtype, shape, data });
  }

  return tensors;
}
