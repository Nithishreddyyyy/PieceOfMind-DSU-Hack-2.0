// =================== Monitoring Page ===================
// Tabs • timeline • AR overlay (camera) • keyboard cadence metrics
// Hooked to global demo stream for live updates
// =======================================================

(function () {
  const $  = (q, ctx=document) => ctx.querySelector(q);
  const $$ = (q, ctx=document) => Array.from(ctx.querySelectorAll(q));

  // Tabs
  const tabs = $$(".tabs .tab");
  const panes = $$(".tab-content > div");
  function setTab(i) {
    tabs.forEach((t, idx) => t.classList.toggle("active", idx === i));
    panes.forEach((p, idx) => p.style.display = (idx === i ? "block" : "none"));
  }
  tabs.forEach((t, i) => t.addEventListener("click", () => setTab(i)));
  setTab(0);

  // Timeline
  const entries = $(".timeline-entries");
  function addRow(kind, msg) {
    if (!entries) return;
    const el = document.createElement("div");
    el.className = `timeline-row ${kind}`;
    el.innerHTML = `
      <span class="time">${new Date().toLocaleTimeString()}</span>
      <span class="kind">${kind}</span>
      <span class="msg">${msg}</span>
    `;
    entries.prepend(el);
  }

  // Live gauges + timeline feed from demo stream
  const strainG = document.querySelector('.live-tab .gauge:nth-child(1) .circular-gauge');
  const driftG  = document.querySelector('.live-tab .gauge:nth-child(2) .circular-gauge');
  window.poom.bus.on("stream:update", ({ strain, drift, fatigue }) => {
    if (strainG) strainG.setAttribute("data-value", String(strain));
    if (driftG)  driftG.setAttribute("data-value", String(drift));
    window.poom.renderGauges();
    if (fatigue && Math.random() > 0.5) addRow("warn", "Fatigue indicator increased");
  });
  window.poom.bus.on("stream:stress-spike", () => addRow("alert", "Stress spike detected (EEG)"));

  // AR overlay (camera + overlay image)
  const arOverlay = document.querySelector(".ar-overlay");
  const arVideo   = arOverlay?.querySelector("video");
  const arClose   = arOverlay?.querySelector(".btn.btn-primary");

  tabs[1]?.addEventListener("click", async () => {
    if (!arOverlay || !arVideo) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      arVideo.srcObject = stream;
      arOverlay.classList.add("active");
      addRow("info", "AR overlay started (camera on)");
    } catch {
      window.poom.toast.show("Camera permission denied", "error");
    }
  });

  function stopCam() {
    if (arVideo?.srcObject) { arVideo.srcObject.getTracks().forEach(t => t.stop()); arVideo.srcObject = null; }
  }
  arClose?.addEventListener("click", () => { stopCam(); arOverlay.classList.remove("active"); addRow("info", "AR overlay closed"); });
  window.poom.bus.on("ui:escape", () => { if (arOverlay?.classList.contains("active")) { stopCam(); arOverlay.classList.remove("active"); } });

  // Keyboard cadence (mini)
  const kbd = document.createElement("div");
  kbd.className = "kbd-cadence";
  kbd.innerHTML = `
    <h4>Keyboard cadence</h4>
    <textarea class="kbd-area" rows="4" placeholder="Type here to measure…"></textarea>
    <div class="kbd-metrics">
      <span>CPM: <strong class="kbd-cpm">—</strong></span>
      <span>Avg interval: <strong class="kbd-avg">—</strong> ms</span>
      <span>Variance: <strong class="kbd-var">—</strong></span>
    </div>
  `;
  document.querySelector(".live-tab")?.appendChild(kbd);

  let keyTimes = [];
  const area = kbd.querySelector(".kbd-area");
  const mCPM = kbd.querySelector(".kbd-cpm");
  const mAVG = kbd.querySelector(".kbd-avg");
  const mVAR = kbd.querySelector(".kbd-var");
  area?.addEventListener("keydown", () => {
    const now = performance.now();
    keyTimes.push(now);
    if (keyTimes.length > 300) keyTimes.shift();
    if (keyTimes.length > 2) {
      const intervals = keyTimes.slice(1).map((t, i) => t - keyTimes[i]);
      const avg = intervals.reduce((a, b) => a + b, 0) / intervals.length;
      const variance = intervals.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / intervals.length;
      const cpm = Math.round(60000 / avg);
      mCPM.textContent = cpm;
      mAVG.textContent = Math.round(avg);
      mVAR.textContent = Math.round(variance);
    }
  });
})();
