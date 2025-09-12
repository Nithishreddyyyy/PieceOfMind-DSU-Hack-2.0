// ==================== PieceOfMind Core (App Shell) ====================
// Event bus • local storage store • toasts • theme/reduced-motion • auth stubs
// header/nav activation • smooth scrolling • carousel • gauges • demo stream
// Onboarding gating (only after first sign-up) • Settings hooks
// ======================================================================

window.poom = window.poom || {};

// ---------- Event Bus ----------
window.poom.bus = {
  events: {},
  on(ev, cb) { (this.events[ev] = this.events[ev] || []).push(cb); },
  emit(ev, data) { (this.events[ev] || []).forEach(cb => cb(data)); }
};

// ---------- Store (localStorage) ----------
const LS_KEY = "pm:data:v1";
function load() { try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch { return {}; } }
function save(s) { localStorage.setItem(LS_KEY, JSON.stringify(s)); }
function patch(updater) { const s = load(); const n = updater({ ...s }) || s; save(n); return n; }
window.poom.state = { load, save, patch };

// ---------- Toaster ----------
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

// ---------- Theme & Reduced Motion ----------
function applyTheme(theme) { document.documentElement.setAttribute("data-theme", theme); }
function applyMotionPref(pref) { document.documentElement.setAttribute("data-reduced-motion", pref ? "reduce" : "no-preference"); }

function initThemeAndMotion() {
  const s = load();
  applyTheme(s.theme || "light");
  applyMotionPref(!!s.reducedMotion);
  // settings toggles (if present)
  const themeToggle = document.querySelector(".theme-toggle");
  themeToggle?.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") || "light";
    const next = (cur === "dark") ? "light" : "dark";
    applyTheme(next);
    patch(st => (st.theme = next, st));
  });
}

// ---------- Smooth Scrolling for in-page anchors ----------
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

// ---------- Header: active link ----------
function markActiveNav() {
  const current = location.pathname.split("/").pop() || "/app/templates/index.html";
  document.querySelectorAll("header nav a").forEach(a => {
    const href = a.getAttribute("href").split("/").pop();
    a.classList.toggle("active", href === current);
  });
}

// ---------- Auth (client-only demo) ----------
/*
  State shape:
  {
    user: { id, name } | undefined,
    hasOnboarded: boolean,
    needsOnboarding: boolean   // set true only when the user signs up first time
  }
*/
function doSignIn() {
  patch(st => {
    st.user = st.user || { id: "demo-user", name: "You" };
    // Keep hasOnboarded if it existed; do NOT set needsOnboarding here.
    if (st.hasOnboarded !== true && st.needsOnboarding !== true) st.needsOnboarding = false;
    return st;
  });
  window.poom.toast.show("Signed in (demo)");
  refreshAuthUI();
}

function doSignUp() {
  patch(st => {
    st.user = { id: "demo-user", name: "You" };
    st.needsOnboarding = true;   // <-- onboarding only after sign-up
    return st;
  });
  window.poom.toast.show("Welcome! Let’s personalize things.");
  // Gate: redirect to onboarding right away
  location.href = "/app/templates/onboarding.html";
}

function doSignOut() {
  patch(st => {
    delete st.user;
    // Keep hasOnboarded persisted — the next sign-in won’t show onboarding again.
    return st;
  });
  window.poom.toast.show("Signed out");
  refreshAuthUI();
  location.href = "/app/templates/index.html";
}

function refreshAuthUI() {
  const s = load();
  const authed = !!s.user;
  // Hide/show auth buttons
  document.querySelectorAll(".auth-signed-out").forEach(el => el.classList.toggle("hidden", authed));
  document.querySelectorAll(".auth-signed-in").forEach(el => el.classList.toggle("hidden", !authed));
  // Hide onboarding nav link if already onboarded
  const onbLink = document.querySelector('header nav a[href="/onboarding.html"]');
  if (onbLink) onbLink.classList.toggle("hidden", !!s.hasOnboarded);
  // Show settings when signed in
  const settingsLink = document.querySelector('header nav a[href="/settings.html"]');
  settingsLink?.classList.toggle("hidden", !authed);
}

// Gate: send users who *need* onboarding to it (but only after sign-up)
function enforceOnboardingGate() {
  const s = load();
  const path = location.pathname;
  const needs = !!s.needsOnboarding;
  if (needs && !/onboarding\.html$/i.test(path)) {
    location.replace("/app/templates/onboarding.html");
  }
}

// Wire header auth buttons (present on all pages)
function initHeaderAuthButtons() {
  document.getElementById("btn-signin")?.addEventListener("click", doSignIn);
  document.getElementById("btn-signup")?.addEventListener("click", doSignUp);
  document.getElementById("btn-logout")?.addEventListener("click", doSignOut);
}

// ---------- Carousel ----------
function initCarousel() {
  const track = document.querySelector(".carousel-track");
  if (!track) return;
  const slides = Array.from(document.querySelectorAll(".carousel-slide"));
  const dots = Array.from(document.querySelectorAll(".carousel-dot"));
  const prev = document.querySelector(".carousel-arrow.prev");
  const next = document.querySelector(".carousel-arrow.next");
  let idx = 0;
  function update() {
    track.style.transform = `translateX(-${idx * 100}%)`;
    dots.forEach((d, i) => d.classList.toggle("active", i === idx));
  }
  prev?.addEventListener("click", () => { idx = Math.max(0, idx - 1); update(); });
  next?.addEventListener("click", () => { idx = Math.min(slides.length - 1, idx + 1); update(); });
  dots.forEach((d, i) => d.addEventListener("click", () => { idx = i; update(); }));
  setInterval(() => { idx = (idx + 1) % slides.length; update(); }, 6000);
  update();
}

// ---------- Circular Gauges ----------
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

// ---------- Demo Stream (strain/drift/fatigue) ----------
function startDemoStream() {
  setInterval(() => {
    const strain = Math.random();
    const drift = Math.random();
    const fatigue = Math.random() > 0.6;
    window.poom.bus.emit("stream:update", { strain, drift, fatigue });
    if (Math.random() > 0.92) window.poom.bus.emit("stream:stress-spike", { type: "eeg_stress", at: Date.now() });
  }, 1000);
}

// ---------- Hero video toggle ----------
function initHeroVideoToggle() {
  const toggle = document.querySelector(".video-toggle");
  const video = document.querySelector(".hero video");
  if (!toggle || !video) return;
  toggle.addEventListener("click", () => {
    if (video.paused) { video.play(); toggle.textContent = "Pause Video"; }
    else { video.pause(); toggle.textContent = "Play Video"; }
  });
}

// ---------- FAB ----------
function initFab() {
  document.querySelector(".fab")?.addEventListener("click", () => {
    window.poom.toast.show("Try: inhale 4 — hold 4 — exhale 6", "success");
  });
}

// ---------- Page bootstrap ----------
document.addEventListener("DOMContentLoaded", () => {
  initThemeAndMotion();
  initSmoothScroll();
  markActiveNav();
  initHeaderAuthButtons();
  refreshAuthUI();
  enforceOnboardingGate(); // must be before other page-specific init
  initCarousel();
  initHeroVideoToggle();
  initFab();
  renderGauges();
  startDemoStream();
});

// Utilities for other scripts
window.poom.auth = { doSignIn, doSignUp, doSignOut, refreshAuthUI };
window.poom.ui = { applyTheme, applyMotionPref };
window.poom.gauges = { set(selector, v01) { const el = document.querySelector(selector); if (el) { el.setAttribute("data-value", String(v01)); renderGauges(); } } };
