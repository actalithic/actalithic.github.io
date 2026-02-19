# LocalLLM by Actalithic

Run LLMs 100% locally in your browser — no server, no API key, no data leaves your device.

## File Structure

```
localllm/
├── index.html        ← Main page (UI markup)
├── sw.js             ← Service worker (offline + logo caching)
├── css/
│   └── style.css     ← All styles (themes, layout, components)
└── js/
    ├── models.js     ← Model registry (IDs, sizes, URLs, per-device tok/s)
    ├── app.js        ← App logic (load/chat/modal/info popup)
    └── theme.js      ← Theme toggle + online/offline banner
```

## Requirements

- **Chrome or Chromium** with WebGPU enabled (chrome://flags/#enable-webgpu-developer-features)  
- OR use the **CPU / WASM fallback** toggle for any browser (slower)

## Running Locally

Native Win/Linux/Mac Port will release in a few weeks,  powered by chromium (electron) with gpu acceleration and fixed settings

## Features

- Dynamic speed table per selected model
- Info button (after load) shows the model's Hugging Face URL
- System prompt is hidden from the chat UI (managed internally)
- Dark/light theme with system preference detection
- Logo and icons cached via service worker after first load
- ActalithicCore: GPU+CPU hybrid for running models larger than your VRAM
