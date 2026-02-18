// theme.js — Theme toggling and service worker registration for LocalLLM

export function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  document.getElementById("themeIcon").textContent = t === "dark" ? "light_mode" : "dark_mode";
  localStorage.setItem("llm-theme", t);
}

export function toggleTheme() {
  applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
}

// Init theme immediately
(function () {
  const s = localStorage.getItem("llm-theme");
  const theme = s || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", theme);
})();

// Online/offline banner
function updateOnline() {
  document.getElementById("offlineBanner").classList.toggle("visible", !navigator.onLine);
}
window.addEventListener("online",  updateOnline);
window.addEventListener("offline", updateOnline);
updateOnline();

// Service worker
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  });
}

window.toggleTheme = toggleTheme;
