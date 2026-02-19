# LocalLLM by Actalithic

Run LLMs 100% locally in your browser — no server, no API key, no data leaves your device.

## License

© 2026 Actalithic. All rights reserved.  
This project is proprietary and closed source. You may not copy, distribute, modify, resell, or create derivative works of this software without explicit permission.

---

This project uses the following third-party software:

@mlc-ai/web-llm  
Copyright 2026 The Contributors.  
Licensed under the Apache License, Version 2.0  
You may obtain a copy of the License at:  
    https://www.apache.org/licenses/LICENSE-2.0

A copy of the full Apache License, Version 2.0 text is included in this distribution in the file LICENSE‑2.0.txt.
```

## Requirements

- **Chrome or Chromium** with WebGPU enabled (chrome://flags/#enable-webgpu-developer-features)  
- OR use the **CPU / WASM fallback** toggle for any browser (slower)

## Running Locally


## Features

- Dynamic speed table per selected model
- Info button (after load) shows the model's Hugging Face URL
- System prompt is hidden from the chat UI (managed internally)
- Dark/light theme with system preference detection
- Logo and icons cached via service worker after first load
- ActalithicCore: GPU+CPU hybrid for running models larger than your VRAM
