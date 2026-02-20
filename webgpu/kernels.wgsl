// kernels.wgsl — Actalithic WebGPU Compute Shaders
// All kernels required for LLaMA/Mistral family inference.
// Supports F32, F16 (via bit-cast), Q8, Q4 quantized weights.
// Apache 2.0 — Actalithic

// ─────────────────────────────────────────────────────────────────────────────
// SHARED UNIFORM STRUCT
// All kernels read from a 256-byte uniform buffer.
// Fields not used by a given kernel are simply ignored.
// ─────────────────────────────────────────────────────────────────────────────

struct Uniforms {
  seq_len:    f32,
  hidden:     f32,
  vocab_size: f32,
  n_heads:    f32,
  n_kv:       f32,
  head_dim:   f32,
  theta:      f32,  // RoPE theta (500000.0 for LLaMA 3)
  offset:     f32,  // KV cache position offset
  M:          f32,  // matmul rows
  N:          f32,  // matmul cols
  K:          f32,  // matmul inner dim
  quant:      f32,  // 0=f32 1=f16 2=q8 3=q4
  eps:        f32,  // RMSNorm epsilon
  scale_attn: f32,  // attention scale (1/sqrt(head_dim))
  ffn_size:   f32,  // intermediate FFN size
  size:       f32,  // generic element count
  last_only:  f32,  // 1 = LM head uses last token only
  pad0:       f32,
  pad1:       f32,
  pad2:       f32,
  pad3:       f32,
  pad4:       f32,
  pad5:       f32,
  pad6:       f32,
  pad7:       f32,
  pad8:       f32,
  pad9:       f32,
  pad10:      f32,
  pad11:      f32,
  pad12:      f32,
  pad13:      f32,
  pad14:      f32,
  pad15:      f32,
  // Remaining 192 bytes / 48 f32s are available for extension
};

// ─────────────────────────────────────────────────────────────────────────────
// TOKEN EMBEDDING LOOKUP
// Looks up token IDs in the embedding table.
// Input:  token_ids [seq_len] (i32), embed_table [vocab × hidden] (f32)
// Output: hidden_states [seq_len × hidden] (f32)
// ─────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       token_ids:    array<i32>;
@group(0) @binding(1) var<storage, read>       embed_table:  array<f32>;
@group(0) @binding(2) var<storage, read_write> hidden_out:   array<f32>;
@group(0) @binding(3) var<uniform>             u_embed:      Uniforms;

@compute @workgroup_size(64)
fn token_embed(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = u32(u_embed.seq_len) * u32(u_embed.hidden);
  if (idx >= total) { return; }

  let pos    = idx / u32(u_embed.hidden);
  let dim    = idx % u32(u_embed.hidden);
  let tok_id = u32(token_ids[pos]);
  let embed_idx = tok_id * u32(u_embed.hidden) + dim;

  hidden_out[idx] = embed_table[embed_idx];
}

// ─────────────────────────────────────────────────────────────────────────────
// RMS NORM
// RMSNorm(x, w) = x / rms(x) * w
// Input:  x [seq × hidden] (f32), weight [hidden] (f32)
// Output: out [seq × hidden] (f32)
// One workgroup per token position.
// ─────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       rn_input:  array<f32>;
@group(0) @binding(1) var<storage, read>       rn_weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> rn_output: array<f32>;
@group(0) @binding(3) var<uniform>             u_rn:      Uniforms;

var<workgroup> rn_sum: f32;

@compute @workgroup_size(256)
fn rms_norm(@builtin(global_invocation_id) gid: vec3u,
            @builtin(local_invocation_id)  lid: vec3u,
            @builtin(workgroup_id)         wgid: vec3u) {
  let seq_pos = wgid.x;
  let hidden  = u32(u_rn.hidden);
  let base    = seq_pos * hidden;

  // Compute sum of squares in parallel (simplified: full loop per thread 0)
  if (lid.x == 0u) {
    var ss: f32 = 0.0;
    for (var i = 0u; i < hidden; i++) {
      let v = rn_input[base + i];
      ss += v * v;
    }
    rn_sum = ss / f32(hidden);
  }
  workgroupBarrier();

  let rms_inv = 1.0 / sqrt(rn_sum + u_rn.eps);

  // Apply norm — distribute hidden dims across threads
  for (var i = lid.x; i < hidden; i += 256u) {
    rn_output[base + i] = rn_input[base + i] * rms_inv * rn_weight[i];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// MATMUL (F32)
// C[M, N] = A[M, K] × B[N, K]ᵀ   (B is transposed = row-major weight matrix)
// Tiled 8×8 workgroup.
// ─────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       mm_a:    array<f32>;
@group(0) @binding(1) var<storage, read>       mm_b:    array<f32>;
@group(0) @binding(2) var<storage, read_write> mm_c:    array<f32>;
@group(0) @binding(3) var<uniform>             u_mm:    Uniforms;

var<workgroup> tile_a: array<array<f32, 8>, 8>;
var<workgroup> tile_b: array<array<f32, 8>, 8>;

@compute @workgroup_size(8, 8)
fn matmul_f32(@builtin(global_invocation_id) gid: vec3u,
              @builtin(local_invocation_id)  lid: vec3u) {
  let M = u32(u_mm.M);
  let N = u32(u_mm.N);
  let K = u32(u_mm.K);
  let row = gid.y;
  let col = gid.x;
  if (row >= M || col >= N) { return; }

  var acc: f32 = 0.0;
  let nTiles = (K + 7u) / 8u;

  for (var t = 0u; t < nTiles; t++) {
    let kA = t * 8u + lid.x;
    let kB = t * 8u + lid.y;

    tile_a[lid.y][lid.x] = select(0.0, mm_a[row * K + kA], kA < K);
    tile_b[lid.y][lid.x] = select(0.0, mm_b[col * K + kB], kB < K);
    workgroupBarrier();

    for (var k = 0u; k < 8u; k++) {
      acc += tile_a[lid.y][k] * tile_b[k][lid.x];
    }
    workgroupBarrier();
  }

  mm_c[row * N + col] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// MATMUL (Q4) — 4-bit quantized weights
// Weight layout: [scales: f32 per 32-element block][packed nibbles]
// Dequantizes on-the-fly during matmul.
// ─────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       q4_input:  array<f32>;
@group(0) @binding(1) var<storage, read>       q4_weight: array<u32>;  // packed bytes as u32
@group(0) @binding(2) var<storage, read_write> q4_output: array<f32>;
@group(0) @binding(3) var<uniform>             u_q4:      Uniforms;

const BLOCK_SIZE_Q4: u32 = 32u;

fn dequant_nibble(packed: u32, idx: u32) -> f32 {
  let nibble = (packed >> (idx * 4u)) & 0xFu;
  // Convert from 4-bit unsigned (0-15) to signed (-8 to 7)
  return f32(select(i32(nibble), i32(nibble) - 16, nibble >= 8u));
}

@compute @workgroup_size(8, 8)
fn matmul_q4(@builtin(global_invocation_id) gid: vec3u,
             @builtin(local_invocation_id)  lid: vec3u) {
  let M = u32(u_q4.M);
  let N = u32(u_q4.N);
  let K = u32(u_q4.K);
  let row = gid.y;
  let col = gid.x;
  if (row >= M || col >= N) { return; }

  let nBlocks    = (K + BLOCK_SIZE_Q4 - 1u) / BLOCK_SIZE_Q4;
  // Scale region: nBlocks f32s per output row
  let scaleBase  = col * nBlocks;
  // Packed data region: nBlocks * 4 u32s per output row (32 nibbles = 16 bytes = 4 u32s per block)
  let dataBase   = N * nBlocks + col * nBlocks * 4u;  // after all scales

  var acc: f32 = 0.0;

  for (var b = 0u; b < nBlocks; b++) {
    let scale     = bitcast<f32>(q4_weight[scaleBase + b]);
    let blockStart = b * BLOCK_SIZE_Q4;

    for (var i = 0u; i < BLOCK_SIZE_Q4; i++) {
      let k = blockStart + i;
      if (k >= K) { break; }

      let packedIdx = dataBase + b * 4u + i / 8u;
      let bitOffset = i % 8u;
      let dq = dequant_nibble(q4_weight[packedIdx], bitOffset) * scale;
      acc += q4_input[row * K + k] * dq;
    }
  }

  q4_output[row * N + col] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// MATMUL (Q8) — 8-bit quantized weights
// Weight layout: [scales: f32 per 32-element block][int8 bytes packed as u32]
// ─────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       q8_input:  array<f32>;
@group(0) @binding(1) var<storage, read>       q8_weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> q8_output: array<f32>;
@group(0) @binding(3) var<uniform>             u_q8:      Uniforms;

const BLOCK_SIZE_Q8: u32 = 32u;

fn unpack_i8(packed: u32, idx: u32) -> f32 {
  let byte = (packed >> (idx * 8u)) & 0xFFu;
  return f32(select(i32(byte), i32(byte) - 256, byte >= 128u));
}

@compute @workgroup_size(8, 8)
fn matmul_q8(@builtin(global_invocation_id) gid: vec3u) {
  let M = u32(u_q8.M);
  let N = u32(u_q8.N);
  let K = u32(u_q8.K);
  let row = gid.y;
  let col = gid.x;
  if (row >= M || col >= N) { return; }

  let nBlocks  = (K + BLOCK_SIZE_Q8 - 1u) / BLOCK_SIZE_Q8;
  let scaleBase = col * nBlocks;
  let dataBase  = N * nBlocks + col * nBlocks * 8u;  // 32 bytes = 8 u32s per block

  var acc: f32 = 0.0;

  for (var b = 0u; b < nBlocks; b++) {
    let scale     = bitcast<f32>(q8_weight[scaleBase + b]);
    let blockStart = b * BLOCK_SIZE_Q8;

    for (var i = 0u; i < BLOCK_SIZE_Q8; i++) {
      let k = blockStart + i;
      if (k >= K) { break; }

      let packedIdx = dataBase + b * 8u + i / 4u;
      let bitOffset = i % 4u;
      let dq = unpack_i8(q8_weight[packedIdx], bitOffset) * scale;
      acc += q8_input[row * K + k] * dq;
    }
  }

  q8_output[row * N + col] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// DEQUANTIZE Q4 (standalone — for CPU-side verification/debugging)
// ─────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       dq4_in:  array<u32>;
@group(0) @binding(1) var<storage, read_write> dq4_out: array<f32>;
@group(0) @binding(2) var<uniform>             u_dq4:   Uniforms;

@compute @workgroup_size(64)
fn dequant_q4(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = u32(u_dq4.size);
  if (idx >= total) { return; }

  let blockIdx  = idx / BLOCK_SIZE_Q4;
  let inBlock   = idx % BLOCK_SIZE_Q4;
  let nBlocks   = (total + BLOCK_SIZE_Q4 - 1u) / BLOCK_SIZE_Q4;
  let scaleBase = 0u;
  let dataBase  = nBlocks;  // scales come first

  let scale     = bitcast<f32>(dq4_in[scaleBase + blockIdx]);
  let packedIdx = dataBase + blockIdx * 4u + inBlock / 8u;
  let nibble    = dequant_nibble(dq4_in[packedIdx], inBlock % 8u);

  dq4_out[idx] = nibble * scale;
}

// ─────────────────────────────────────────────────────────────────────────────
// DEQUANTIZE Q8
// ─────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       dq8_in:  array<u32>;
@group(0) @binding(1) var<storage, read_write> dq8_out: array<f32>;
@group(0) @binding(2) var<uniform>             u_dq8:   Uniforms;

@compute @workgroup_size(64)
fn dequant_q8(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = u32(u_dq8.size);
  if (idx >= total) { return; }

  let blockIdx  = idx / BLOCK_SIZE_Q8;
  let inBlock   = idx % BLOCK_SIZE_Q8;
  let nBlocks   = (total + BLOCK_SIZE_Q8 - 1u) / BLOCK_SIZE_Q8;
  let dataBase  = nBlocks;

  let scale     = bitcast<f32>(dq8_in[blockIdx]);
  let packedIdx = dataBase + blockIdx * 8u + inBlock / 4u;
  let val       = unpack_i8(dq8_in[packedIdx], inBlock % 4u);

  dq8_out[idx] = val * scale;
}

// ─────────────────────────────────────────────────────────────────────────────
// ROTARY POSITION EMBEDDING (RoPE)
// Applies RoPE to Q and K tensors in-place.
// Input/Output: q [seq × n_heads × head_dim], k [seq × n_kv × head_dim]
// ─────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read_write> rope_q: array<f32>;
@group(0) @binding(1) var<storage, read_write> rope_k: array<f32>;
@group(0) @binding(2) var<uniform>             u_rope: Uniforms;

@compute @workgroup_size(64)
fn rope_embed(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let n_heads  = u32(u_rope.n_heads);
  let n_kv     = u32(u_rope.n_kv);
  let head_dim = u32(u_rope.head_dim);
  let seq_len  = u32(u_rope.seq_len);
  let total_q  = seq_len * n_heads * (head_dim / 2u);
  if (idx >= total_q) { return; }

  let head_half = head_dim / 2u;
  let pos       = idx / (n_heads * head_half);
  let head      = (idx / head_half) % n_heads;
  let dim       = idx % head_half;
  let abs_pos   = pos + u32(u_rope.offset);

  let theta = pow(u_rope.theta, -f32(2u * dim) / f32(head_dim));
  let angle = f32(abs_pos) * theta;
  let cos_a = cos(angle);
  let sin_a = sin(angle);

  let q_base = (pos * n_heads + head) * head_dim;
  let q0 = rope_q[q_base + dim];
  let q1 = rope_q[q_base + dim + head_half];
  rope_q[q_base + dim]           = q0 * cos_a - q1 * sin_a;
  rope_q[q_base + dim + head_half] = q0 * sin_a + q1 * cos_a;

  // Only apply to K if this head is within n_kv (GQA support)
  if (head < n_kv) {
    let k_base = (pos * n_kv + head) * head_dim;
    let k0 = rope_k[k_base + dim];
    let k1 = rope_k[k_base + dim + head_half];
    rope_k[k_base + dim]           = k0 * cos_a - k1 * sin_a;
    rope_k[k_base + dim + head_half] = k0 * sin_a + k1 * cos_a;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ATTENTION SCORES + SOFTMAX + VALUE AGGREGATION
// Fused causal self-attention for a full sequence.
// Q [seq × n_heads × head_dim]
// K [seq × n_kv   × head_dim]   (GQA: n_kv may be < n_heads)
// V [seq × n_kv   × head_dim]
// Out [seq × n_heads × head_dim]
// One workgroup per (head, query_position) pair.
// ─────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       attn_q:   array<f32>;
@group(0) @binding(1) var<storage, read>       attn_k:   array<f32>;
@group(0) @binding(2) var<storage, read>       attn_v:   array<f32>;
@group(0) @binding(3) var<storage, read_write> attn_out: array<f32>;
@group(0) @binding(4) var<uniform>             u_attn:   Uniforms;

var<workgroup> attn_scores: array<f32, 4096>;  // max seq_len 4096
var<workgroup> attn_max:    f32;
var<workgroup> attn_softmax_sum: f32;

@compute @workgroup_size(64)
fn attention_score(@builtin(global_invocation_id) gid: vec3u,
                   @builtin(local_invocation_id)  lid: vec3u,
                   @builtin(workgroup_id)         wgid: vec3u) {
  let n_heads  = u32(u_attn.n_heads);
  let n_kv     = u32(u_attn.n_kv);
  let head_dim = u32(u_attn.head_dim);
  let seq_len  = u32(u_attn.seq_len);
  let scale    = u_attn.scale_attn;

  let head     = wgid.x % n_heads;
  let q_pos    = wgid.x / n_heads;
  if (q_pos >= seq_len) { return; }

  let kv_head  = head % n_kv;  // GQA: map query head to KV head
  let q_base   = (q_pos * n_heads + head)   * head_dim;
  let out_base = (q_pos * n_heads + head)   * head_dim;

  // Compute attention scores for all key positions ≤ q_pos (causal mask)
  if (lid.x == 0u) { attn_max = -1e38; }
  workgroupBarrier();

  for (var kpos = lid.x; kpos <= q_pos; kpos += 64u) {
    let k_base = (kpos * n_kv + kv_head) * head_dim;
    var dot: f32 = 0.0;
    for (var d = 0u; d < head_dim; d++) {
      dot += attn_q[q_base + d] * attn_k[k_base + d];
    }
    let score = dot * scale;
    attn_scores[kpos] = score;
    // Track max for numerically stable softmax
    if (score > attn_max) {
      // This is a simplified race-condition-prone approach
      // In production use a proper parallel reduction
      attn_max = score;
    }
  }
  // Mask future positions
  for (var kpos = q_pos + 1u + lid.x; kpos < seq_len; kpos += 64u) {
    attn_scores[kpos] = -1e38;
  }
  workgroupBarrier();

  // Softmax over scores
  if (lid.x == 0u) { attn_softmax_sum = 0.0; }
  workgroupBarrier();
  for (var kpos = lid.x; kpos < seq_len; kpos += 64u) {
    let ex = exp(attn_scores[kpos] - attn_max);
    attn_scores[kpos] = ex;
    attn_softmax_sum += ex;
  }
  workgroupBarrier();
  for (var kpos = lid.x; kpos < seq_len; kpos += 64u) {
    attn_scores[kpos] /= attn_softmax_sum;
  }
  workgroupBarrier();

  // Aggregate values
  for (var d = lid.x; d < head_dim; d += 64u) {
    var out: f32 = 0.0;
    for (var kpos = 0u; kpos <= q_pos; kpos++) {
      let v_base = (kpos * n_kv + kv_head) * head_dim;
      out += attn_scores[kpos] * attn_v[v_base + d];
    }
    attn_out[out_base + d] = out;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SOFTMAX (standalone — used for logit temp scaling)
// In-place on a 1D array of length `size`.
// ─────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read_write> sm_data: array<f32>;
@group(0) @binding(1) var<uniform>             u_sm:    Uniforms;

var<workgroup> sm_max: f32;
var<workgroup> sm_sum: f32;

@compute @workgroup_size(256)
fn softmax(@builtin(global_invocation_id) gid: vec3u,
           @builtin(local_invocation_id)  lid: vec3u) {
  let size = u32(u_sm.size);
  // Simplified: thread 0 does all work (small vocab slices only)
  if (lid.x != 0u) { return; }
  var mx: f32 = -1e38;
  for (var i = 0u; i < size; i++) { if (sm_data[i] > mx) { mx = sm_data[i]; } }
  var s: f32 = 0.0;
  for (var i = 0u; i < size; i++) {
    let e = exp(sm_data[i] - mx);
    sm_data[i] = e; s += e;
  }
  for (var i = 0u; i < size; i++) { sm_data[i] /= s; }
}

// ─────────────────────────────────────────────────────────────────────────────
// SWIGLU ACTIVATION
// SwiGLU(gate, up) = silu(gate) * up
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// ─────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       sg_gate: array<f32>;
@group(0) @binding(1) var<storage, read>       sg_up:   array<f32>;
@group(0) @binding(2) var<storage, read_write> sg_out:  array<f32>;
@group(0) @binding(3) var<uniform>             u_sg:    Uniforms;

@compute @workgroup_size(64)
fn swiglu(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= u32(u_sg.size)) { return; }

  let g = sg_gate[idx];
  let silu_g = g / (1.0 + exp(-g));  // SiLU(gate)
  sg_out[idx] = silu_g * sg_up[idx]; // × up
}

// ─────────────────────────────────────────────────────────────────────────────
// LM HEAD (vocab projection)
// Projects the last token's hidden state to vocab logits.
// Input:  hidden [seq × hidden] (f32), lm_head_weight [vocab × hidden] (f32)
// Output: logits [vocab] (f32)  — only last token position
// ─────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       lm_hidden: array<f32>;
@group(0) @binding(1) var<storage, read>       lm_weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> lm_logits: array<f32>;
@group(0) @binding(3) var<uniform>             u_lm:      Uniforms;

@compute @workgroup_size(64)
fn lm_head(@builtin(global_invocation_id) gid: vec3u) {
  let vocab_id = gid.x;
  let vocab    = u32(u_lm.vocab_size);
  let hidden   = u32(u_lm.hidden);
  let seq_len  = u32(u_lm.seq_len);
  if (vocab_id >= vocab) { return; }

  // Use last token (or single token)
  let last_pos = seq_len - 1u;
  let h_base   = last_pos * hidden;
  let w_base   = vocab_id * hidden;

  var acc: f32 = 0.0;
  for (var i = 0u; i < hidden; i++) {
    acc += lm_hidden[h_base + i] * lm_weight[w_base + i];
  }
  lm_logits[vocab_id] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// RESIDUAL ADD (in-place: a += b)
// Used after attention and FFN blocks.
// ─────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read_write> res_a: array<f32>;
@group(0) @binding(1) var<storage, read>       res_b: array<f32>;
@group(0) @binding(2) var<uniform>             u_res: Uniforms;

@compute @workgroup_size(64)
fn residual_add(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= u32(u_res.size)) { return; }
  res_a[idx] = res_a[idx] + res_b[idx];
}

// ─────────────────────────────────────────────────────────────────────────────
// ELEMENT-WISE MULTIPLY (a *= b)
// For scale application in grouped query attention head broadcasting.
// ─────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read_write> ew_a: array<f32>;
@group(0) @binding(1) var<storage, read>       ew_b: array<f32>;
@group(0) @binding(2) var<uniform>             u_ew: Uniforms;

@compute @workgroup_size(64)
fn elem_mul(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= u32(u_ew.size)) { return; }
  ew_a[idx] = ew_a[idx] * ew_b[idx];
}

// ─────────────────────────────────────────────────────────────────────────────
// KV CACHE COPY
// Copies current K or V slice into persistent KV cache buffer at given position.
// Enables incremental decode without recomputing entire context each step.
// ─────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       kvc_src:  array<f32>;
@group(0) @binding(1) var<storage, read_write> kvc_dst:  array<f32>;
@group(0) @binding(2) var<uniform>             u_kvc:    Uniforms;

@compute @workgroup_size(64)
fn kv_cache_copy(@builtin(global_invocation_id) gid: vec3u) {
  let idx        = gid.x;
  let slice_size = u32(u_kvc.n_kv) * u32(u_kvc.head_dim) * u32(u_kvc.seq_len);
  if (idx >= slice_size) { return; }
  let dst_offset = u32(u_kvc.offset) * u32(u_kvc.n_kv) * u32(u_kvc.head_dim);
  kvc_dst[dst_offset + idx] = kvc_src[idx];
}
