// firebase.js — Analytics init for LocalLLM by Actalithic
// Only runs when hosted (not file://)

const FIREBASE_CONFIG = {
  apiKey: "AIzaSyBVB56VoK7s0JrTJwuq4mPxfSGmRf_NiHc",
  authDomain: "actalithic.firebaseapp.com",
  projectId: "actalithic",
  storageBucket: "actalithic.firebasestorage.app",
  messagingSenderId: "414840812928",
  appId: "1:414840812928:web:4187304c801d78e4263d65",
  measurementId: "G-4V76FM0XY6"
};

// Don't run analytics when accessed locally via file://
// Chat data is never sent — analytics only tracks page visits
const isLocal = location.protocol === 'file:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';

export async function initAnalytics() {
  if (isLocal) {
    console.log('[Actalithic] Running locally — analytics disabled. No data transferred.');
    return null;
  }

  // Only init if user hasn't opted out
  if (localStorage.getItem('analytics-opt-out') === 'true') {
    console.log('[Actalithic] Analytics opt-out active.');
    return null;
  }

  try {
    const { initializeApp } = await import("https://www.gstatic.com/firebasejs/12.9.0/firebase-app.js");
    const { getAnalytics, isSupported } = await import("https://www.gstatic.com/firebasejs/12.9.0/firebase-analytics.js");

    const supported = await isSupported();
    if (!supported) return null;

    const app = initializeApp(FIREBASE_CONFIG);
    const analytics = getAnalytics(app);
    return analytics;
  } catch (e) {
    console.warn('[Actalithic] Analytics init failed:', e);
    return null;
  }
}

export { isLocal };
