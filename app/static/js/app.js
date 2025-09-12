// ==================== PieceOfMind Core (App Shell) ====================
window.poom = window.poom || {};

// Bus
window.poom.bus = {
  events: {},
  on(ev, cb) { (this.events[ev] = this.events[ev] || []).push(cb); },
  emit(ev, data) { (this.events[ev] || []).forEach(cb => cb(data)); }
};

// Store
const LS_KEY = "pm:data:v1";
function load() { try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch { return {}; } }
function save(s) { localStorage.setItem(LS_KEY, JSON.stringify(s)); }
function patch(updater) { const s = load(); const n = updater({ ...s }) || s; save(n); return n; }
window.poom.state = { load, save, patch };

// Toast
window.poom.toast = {
  show(msg, kind = "info") {
    const ctn = document.querySelector(".toast-container");
    if (!ctn) return alert(msg);
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.textContent = msg;
    ctn.appendChild(el);
    requestAnimationFrame(() => el.classList.add("show"));
    setTimeout(() => { el.classList.remove("show"); setTimeout(() => el.remove(), 280); }, 2500);
  }
};

// Theme & motion
function applyTheme(theme) { document.documentElement.setAttribute("data-theme", theme); }
function applyMotionPref(pref) { document.documentElement.setAttribute("data-reduced-motion", pref ? "reduce" : "no-preference"); }
function initThemeAndMotion() {
  const s = load();
  applyTheme(s.theme || "dark"); // default to dark; change to "light" if preferred
  applyMotionPref(!!s.reducedMotion);
  const btn = document.querySelector(".theme-toggle");
  btn?.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") || "light";
    const next = cur === "dark" ? "light" : "dark";
    applyTheme(next);
    patch(st => (st.theme = next, st));
  });
}

// Smooth scroll
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener("click", e => {
      const id = a.getAttribute("href");
      if (!id || id === "#") return;
      const el = document.querySelector(id);
      if (el) { e.preventDefault(); el.scrollIntoView({ behavior: "smooth", block: "start" }); }
    });
  });
}

// Active nav
function markActiveNav() {
  const current = location.pathname.split("/").pop() || "/app/templates/index.html";
  document.querySelectorAll("header nav a").forEach(a => {
    const href = a.getAttribute("href").split("/").pop();
    a.classList.toggle("active", href === current);
  });
}

// Shrink header on scroll (thin)
function initHeaderShrink() {
  const hdr = document.querySelector(".sticky-nav");
  if (!hdr) return;
  window.addEventListener("scroll", () => {
    const y = window.scrollY || 0;
    hdr.classList.toggle("compact", y > 12);
  }, { passive: true });
}

// Auth (demo)
function doSignIn() {
  patch(st => { st.user = st.user || { id: "demo-user", name: "You" }; return st; });
  window.poom.toast.show("Signed in (demo)");
  refreshAuthUI();
}
function doSignUp() {
  patch(st => { st.user = { id: "demo-user", name: "You" }; st.needsOnboarding = true; return st; });
  window.poom.toast.show("Welcome! Let’s personalize things.");
  location.href = "/app/templates/onboarding.html";
}
function doSignOut() {
  patch(st => { delete st.user; return st; });
  window.poom.toast.show("Signed out");
  refreshAuthUI();
  location.href = "/app/templates/index.html";
}
function refreshAuthUI() {
  const s = load(); const authed = !!s.user;
  document.querySelectorAll(".auth-signed-out").forEach(el => el.classList.toggle("hidden", authed));
  document.querySelectorAll(".auth-signed-in").forEach(el => el.classList.toggle("hidden", !authed));
  const onb = document.querySelector('header nav a[href="/app/templates/onboarding.html"]');
  onb?.classList.toggle("hidden", !!s.hasOnboarded);
  const settings = document.querySelector('header nav a[href="/app/templates/settings.html"]');
  settings?.classList.toggle("hidden", !authed);
}
function enforceOnboardingGate() {
  const s = load(); const path = location.pathname;
  if (s.needsOnboarding && !/onboarding\.html$/i.test(path)) location.replace("/app/templates/onboarding.html");
}
function initHeaderAuthButtons() {
  document.getElementById("btn-signin")?.addEventListener("click", doSignIn);
  document.getElementById("btn-signup")?.addEventListener("click", doSignUp);
  document.getElementById("btn-logout")?.addEventListener("click", doSignOut);
}

// Carousel
function initCarousel() {
  const track = document.querySelector(".carousel-track");
  if (!track) return;
  const slides = Array.from(document.querySelectorAll(".carousel-slide"));
  const dots = Array.from(document.querySelectorAll(".carousel-dot"));
  const prev = document.querySelector(".carousel-arrow.prev");
  const next = document.querySelector(".carousel-arrow.next");
  let idx = 0;
  function update() { track.style.transform = `translateX(-${idx * 100}%)`; dots.forEach((d,i)=>d.classList.toggle("active", i===idx)); }
  prev?.addEventListener("click", () => { idx = Math.max(0, idx-1); update(); });
  next?.addEventListener("click", () => { idx = Math.min(slides.length-1, idx+1); update(); });
  dots.forEach((d,i)=>d.addEventListener("click", () => { idx=i; update(); }));
  setInterval(() => { idx = (idx + 1) % slides.length; update(); }, 6000);
  update();
}

// Gauges
function renderGauges() {
  document.querySelectorAll(".circular-gauge").forEach(g => {
    const v = Math.max(0, Math.min(1, parseFloat(g.getAttribute("data-value") || "0") || 0));
    const deg = Math.round(v * 360);
    g.style.setProperty("--gauge-value", v);
    g.style.background = `conic-gradient(var(--accent) ${deg}deg, var(--surface-2) ${deg}deg 360deg)`;
    g.title = Math.round(v * 100) + "%";
  });
}
window.poom.renderGauges = renderGauges;

// Demo stream
function startDemoStream() {
  setInterval(() => {
    const strain = Math.random(), drift = Math.random(), fatigue = Math.random() > 0.6;
    window.poom.bus.emit("stream:update", { strain, drift, fatigue });
    if (Math.random() > 0.92) window.poom.bus.emit("stream:stress-spike", { type: "eeg_stress", at: Date.now() });
  }, 1000);
}

// Hero video toggle
function initHeroVideoToggle() {
  const toggle = document.querySelector(".video-toggle");
  const video = document.querySelector(".hero video");
  if (!toggle || !video) return;
  toggle.addEventListener("click", () => {
    if (video.paused) { video.play(); toggle.textContent = "Pause Video"; }
    else { video.pause(); toggle.textContent = "Play Video"; }
  });
}

// FAB
function initFab() {
  document.querySelector(".fab")?.addEventListener("click", () => {
    window.poom.toast.show("Try: inhale 4 — hold 4 — exhale 6", "success");
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initThemeAndMotion();
  initSmoothScroll();
  markActiveNav();
  initHeaderShrink();              // <— thin header shrink
  initHeaderAuthButtons();
  refreshAuthUI();
  enforceOnboardingGate();
  initCarousel();
  initHeroVideoToggle();
  initFab();
  renderGauges();
  startDemoStream();
  const y = document.getElementById("year");
  if (y) y.textContent = new Date().getFullYear();
});

// Expose
window.poom.auth = { doSignIn, doSignUp, doSignOut, refreshAuthUI };
window.poom.ui   = { applyTheme, applyMotionPref };
window.poom.gauges = { set(sel, v01) { const el = document.querySelector(sel); if (el) { el.setAttribute("data-value", String(v01)); renderGauges(); } } };
