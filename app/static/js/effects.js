// Interactions: thin ribbon, reveal-on-scroll, tilt cards, magnetic buttons, parallax.
// All effects gracefully degrade and respect reduced motion.

(function () {
  const prefersReduced = () =>
    (document.documentElement.getAttribute("data-reduced-motion") || "") === "reduce";

  // Thin ribbon
  function initRibbon() {
    const bar = document.querySelector(".thin-ribbon");
    if (!bar) return;
    const KEY = "pm:ribbon:dismissed";
    if (localStorage.getItem(KEY) === "1") { bar.remove(); return; }
    bar.classList.add("mounted");
    bar.querySelector(".ribbon-close")?.addEventListener("click", () => {
      bar.classList.add("hide"); localStorage.setItem(KEY, "1");
      setTimeout(() => bar.remove(), 220);
    });
  }

  // Reveal on scroll
  function initReveal() {
    const els = Array.from(document.querySelectorAll("[data-reveal]"));
    if (!els.length) return;
    if (prefersReduced()) { els.forEach(el => el.classList.add("is-visible")); return; }

    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("is-visible");
          io.unobserve(e.target);
        }
      });
    }, { rootMargin: "0px 0px -10% 0px", threshold: 0.08 });

    els.forEach((el, i) => {
      el.style.setProperty("--reveal-delay", `${Math.min(i * 60, 360)}ms`);
      io.observe(el);
    });
  }

  // Tilt on hover
  function initTilt() {
    if (prefersReduced()) return;
    const els = document.querySelectorAll("[data-tilt]");
    els.forEach(el => {
      const max = parseFloat(el.getAttribute("data-tilt")) || 6;
      const damp = 32;
      function onMove(e) {
        const r = el.getBoundingClientRect();
        const cx = r.left + r.width / 2;
        const cy = r.top + r.height / 2;
        const dx = (e.clientX - cx) / damp;
        const dy = (e.clientY - cy) / damp;
        el.style.transform = `perspective(800px) rotateX(${-dy * max / 10}deg) rotateY(${dx * max / 10}deg) translateZ(0)`;
      }
      function reset() { el.style.transform = ""; }
      el.addEventListener("mousemove", onMove);
      el.addEventListener("mouseleave", reset);
    });
  }

  // Magnetic buttons
  function initMagnetic() {
    if (prefersReduced()) return;
    const mags = document.querySelectorAll("[data-magnetic] .btn, [data-magnetic] .pill");
    mags.forEach(btn => {
      const strength = 16;
      btn.addEventListener("mousemove", (e) => {
        const r = btn.getBoundingClientRect();
        const x = ((e.clientX - r.left) / r.width - 0.5) * strength;
        const y = ((e.clientY - r.top) / r.height - 0.5) * strength;
        btn.style.transform = `translate(${x}px, ${y}px)`;
      });
      btn.addEventListener("mouseleave", () => (btn.style.transform = ""));
    });
  }

  // Parallax
  function initParallax() {
    const els = document.querySelectorAll("[data-parallax]");
    if (!els.length || prefersReduced()) return;

    let ticking = false;
    const update = () => {
      const wh = window.innerHeight;
      els.forEach(el => {
        const speed = parseFloat(el.getAttribute("data-parallax-speed")) || 0.2;
        const rect = el.getBoundingClientRect();
        if (rect.top > wh || rect.bottom < 0) return;
        const offset = (rect.top - wh * 0.5) * speed;
        el.style.transform = `translateY(${offset}px)`;
      });
      ticking = false;
    };
    window.addEventListener("scroll", () => { if (!ticking) { requestAnimationFrame(update); ticking = true; } }, { passive:true });
    update();
  }

  document.addEventListener("DOMContentLoaded", () => {
    initRibbon();
    initReveal();
    initTilt();
    initMagnetic();
    initParallax();
  });
})();
