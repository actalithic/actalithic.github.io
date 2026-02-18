// theme.js — Theme toggling and service worker registration for LocalLLM by Actalithic

export function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  const icon = document.getElementById("themeIcon");
  if (icon) icon.textContent = t === "dark" ? "light_mode" : "dark_mode";
  localStorage.setItem("llm-theme", t);
}

export function toggleTheme() {
  applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
}

// Apply theme immediately on module load (avoids flash)
(function () {
  const s = localStorage.getItem("llm-theme");
  const t = s || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", t);
})();

// Online/offline banner
function updateOnline() {
  const banner = document.getElementById("offlineBanner");
  if (banner) banner.classList.toggle("visible", !navigator.onLine);
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

// Set correct icon after DOM is ready
window.addEventListener("DOMContentLoaded", () => {
  const icon = document.getElementById("themeIcon");
  if (icon) {
    const t = document.documentElement.getAttribute("data-theme");
    icon.textContent = t === "dark" ? "light_mode" : "dark_mode";
  }
});

window.toggleTheme = toggleTheme;
