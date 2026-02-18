// sw.js — Service Worker for LocalLLM by Actalithic
const CACHE_NAME = 'localllm-v1';
const CACHE_URLS = [
  '/',
  '/index.html',
  '/css/style.css',
  '/js/theme.js',
  '/js/app.js',
  '/js/models.js',
  // Logos — cached on first load
  'https://i.ibb.co/KxCDDsc7/logoico.png',
  'https://i.ibb.co/mV4rQV7B/Chat-GPT-Image-18-Feb-2026-08-42-07.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return Promise.allSettled(
        CACHE_URLS.map(url => cache.add(url).catch(() => null))
      );
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  // Don't intercept MLC model downloads (they need range requests)
  if (event.request.url.includes('huggingface.co') || event.request.url.includes('mlc.ai')) return;

  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        // Cache GET responses from same origin + logos
        if (
          response.ok &&
          event.request.method === 'GET' &&
          (event.request.url.startsWith(self.location.origin) || event.request.url.includes('ibb.co'))
        ) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        }
        return response;
      }).catch(() => cached || new Response('Offline', { status: 503 }));
    })
  );
});
