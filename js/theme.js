// theme.js — LocalLLM by Actalithic

export function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  localStorage.setItem("llm-theme", t);
  const icon = document.getElementById("themeIcon");
  if (icon) icon.textContent = t === "dark" ? "light_mode" : "dark_mode";
  applyLogos();
}

export function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  applyTheme(current === "dark" ? "light" : "dark");
}

function applyLogos() {
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  document.querySelectorAll(".logo-light").forEach(el => el.style.display = isDark ? "none" : "block");
  document.querySelectorAll(".logo-dark").forEach(el =>  el.style.display = isDark ? "block" : "none");
}

// Apply on load
(function(){
  const s = localStorage.getItem("llm-theme");
  const t = s || "dark";
  document.documentElement.setAttribute("data-theme", t);
  window.addEventListener("DOMContentLoaded", () => {
    applyLogos();
    const icon = document.getElementById("themeIcon");
    if (icon) icon.textContent = t === "dark" ? "light_mode" : "dark_mode";
  });
})();

// Online/offline banner
function updateOnline() {
  const b = document.getElementById("offlineBanner");
  if (b) b.classList.toggle("visible", !navigator.onLine);
}
window.addEventListener("online",  updateOnline);
window.addEventListener("offline", updateOnline);
window.addEventListener("DOMContentLoaded", updateOnline);

// Service Worker
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(e => console.warn("SW:", e));
  });
}

window.toggleTheme = toggleTheme;
window.applyTheme  = applyTheme;
