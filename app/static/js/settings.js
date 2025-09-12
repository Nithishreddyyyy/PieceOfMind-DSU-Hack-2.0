// ====================== Settings Page ======================
// Theme • reduced motion • ambience volume • clear data • sign-out
// ===========================================================

(function () {
  const $ = (q, ctx=document) => ctx.querySelector(q);

  // Initialize switches with current state
  const s = window.poom.state.load();
  const themeSel = $("#sel-theme");
  const motionChk = $("#chk-reduced-motion");
  const vol = $("#rng-ambience");

  if (themeSel) {
    themeSel.value = s.theme || "light";
    themeSel.addEventListener("change", () => {
      window.poom.ui.applyTheme(themeSel.value);
      window.poom.state.patch(st => (st.theme = themeSel.value, st));
    });
  }

  if (motionChk) {
    motionChk.checked = !!s.reducedMotion;
    motionChk.addEventListener("change", () => {
      window.poom.ui.applyMotionPref(motionChk.checked);
      window.poom.state.patch(st => (st.reducedMotion = motionChk.checked, st));
    });
  }

  if (vol) {
    const current = (s.ambienceVol ?? 0.4);
    vol.value = current;
    vol.addEventListener("input", () => window.poom.state.patch(st => (st.ambienceVol = +vol.value, st)));
  }

  document.getElementById("btn-clear")?.addEventListener("click", () => {
    localStorage.removeItem("pm:data:v1");
    window.poom.toast.show("Local data cleared", "success");
    setTimeout(() => location.reload(), 600);
  });

  document.getElementById("btn-logout")?.addEventListener("click", () => window.poom.auth.doSignOut());
})();
