// ===================== Focus Page =====================
// Timer presets • progress ring • ambience volume memory
// ======================================================

(function () {
  const $ = (q, ctx=document) => ctx.querySelector(q);

  const presets = document.querySelectorAll(".timer-presets .btn");
  const text = $(".timer .timer-text");
  const ring = $(".timer .visual-ring");
  const toggle = document.querySelector(".timer-toggle");
  const endBtn = document.querySelector(".end-session");
  const micBtn = document.querySelector(".mic-button");
  const audio = document.querySelector(".audio-player audio");

  let total = 25 * 60, left = total, running = false, iv = null;

  function fmt(sec) { const m = String(Math.floor(sec/60)).padStart(2,"0"); const s = String(sec%60).padStart(2,"0"); return `${m}:${s}`; }
  function draw() { text.textContent = fmt(left); const p = 1 - (left/total); ring.style.setProperty("--p", p); }
  function tick() {
    if (!running) return;
    left -= 1;
    if (left <= 0) {
      running = false; clearInterval(iv);
      window.poom.toast.show("Focus complete — take a 5-min break", "success");
      window.poom.state.patch(s => { s.sessions = s.sessions || []; s.sessions.push({ ts: Date.now(), type: "focus", minutes: total/60 }); return s; });
    }
    draw();
  }

  presets.forEach(btn => btn.addEventListener("click", () => {
    const m = parseInt(btn.textContent.trim(), 10);
    total = left = (isNaN(m) ? 25 : m) * 60; draw();
  }));

  toggle?.addEventListener("click", () => {
    if (!running) { running = true; iv = setInterval(tick, 1000); toggle.textContent = "Pause"; }
    else { running = false; clearInterval(iv); toggle.textContent = "Start"; }
  });

  endBtn?.addEventListener("click", () => { running = false; clearInterval(iv); left = total; draw(); window.poom.toast.show("Session ended"); });
  micBtn?.addEventListener("click", () => window.poom.toast.show("Voice check-in (placeholder)"));

  if (audio) {
    const vol = (window.poom.state.load().ambienceVol ?? 0.4);
    audio.volume = vol;
    audio.addEventListener("volumechange", () => window.poom.state.patch(s => (s.ambienceVol = audio.volume, s)));
  }

  draw();
})();
