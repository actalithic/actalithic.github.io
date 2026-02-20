// sw.js — Service Worker for LocalLLM by Actalithic
const CACHE = 'localllm-v5';
const STATIC = [
  '/',
  '/index.html',
  '/blog.html',
  '/404.html',
  '/acc-converter.html',
  '/modelurls.json',
  '/css/style.css',
  '/js/app.js',
  '/js/models.js',
  '/js/theme.js',
  '/js/firebase.js',
  '/js/llm-worker.js',
  '/js/ACC-Worker.js',
  '/js/acc-converter.js',
  '/webgpu/kernels.wgsl',
  '/sw.js',
  // Logos & favicon
  'https://i.ibb.co/DfYLtMhQ/favicon.png',
  'https://i.ibb.co/KxCDDsc7/logoico.png',
  'https://i.ibb.co/mV4rQV7B/Chat-GPT-Image-18-Feb-2026-08-42-07.png',
  'https://i.ibb.co/tpSTrg7Z/whitelogo.png',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(STATIC)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = e.request.url;
  // Skip model downloads and external CDNs — let them go direct
  if (url.includes('huggingface.co') || url.includes('mlc-ai') ||
      url.includes('esm.run') || url.includes('gstatic.com/firebasejs')) {
    return;
  }
  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(res => {
        if (res && res.status === 200 && res.type !== 'opaque') {
          caches.open(CACHE).then(c => c.put(e.request, res.clone()));
        }
        return res;
      }).catch(() => {
        if (e.request.mode === 'navigate') return caches.match('/404.html');
      });
    })
  );
});
