// ==================== Onboarding Wizard ====================
// Only accessed right after sign-up. Persists, then routes to dashboard.
// ===========================================================

(function () {
  const $ = (q, ctx=document) => ctx.querySelector(q);
  const $$ = (q, ctx=document) => Array.from(ctx.querySelectorAll(q));

  const steps = $$(".stepper .step");
  const views = $$(".step-content > div");
  const back = document.querySelector(".step-buttons .btn-ghost");
  const next = document.querySelector(".step-buttons .btn-primary");
  const finish = document.querySelector(".step-buttons .btn-success");
  let i = 0;

  function setStep(n) {
    i = n;
    steps.forEach((s, idx) => s.classList.toggle("active", idx === i));
    views.forEach((v, idx) => v.style.display = (idx === i ? "block" : "none"));
    back.disabled = i === 0;
    next.disabled = i === steps.length - 1;
    finish.style.display = i === steps.length - 1 ? "inline-block" : "none";
  }

  function collectSnapshot() {
    const root = document.querySelector(".onboarding");
    const checks = Array.from(root.querySelectorAll("input[type=checkbox]"))
      .map(el => ({ label: el.parentElement.textContent.trim(), checked: el.checked }));
    const profile = {
      age: parseInt(root.querySelector('input[type="number"]')?.value || "0", 10) || null,
      gender: root.querySelector('.step-3 input[type="text"]')?.value || ""
    };
    return { checks, profile, ts: Date.now() };
  }

  function persist() {
    window.poom.state.patch(s => (s.onboarding = collectSnapshot(), s));
  }

  back?.addEventListener("click", () => { if (i > 0) setStep(i - 1); });
  next?.addEventListener("click", () => { persist(); if (i < steps.length - 1) setStep(i + 1); });
  finish?.addEventListener("click", () => {
    persist();
    window.poom.state.patch(s => (s.hasOnboarded = true, s.needsOnboarding = false, s));
    window.poom.toast.show("You're all set!", "success");
    location.href = "/app/templates/dashboard.html";
  });

  // If user somehow lands here without sign-up flag, send to dashboard/home
  const s = window.poom.state.load();
  if (!s.needsOnboarding) {
    location.replace(s.user ? "/app/templates/dashboard.html" : "/app/templates/index.html");
  } else {
    setStep(0);
  }
})();
