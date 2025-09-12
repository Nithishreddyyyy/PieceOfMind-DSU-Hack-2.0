/* theme-aware CLOUDS with stronger light-mode contrast */
(function () {
  const el = document.getElementById("animated-bg");
  if (!el) return;

  let effect = null;

  const palettes = {
    light: {
      skyColor: 0x98ccff,          // deeper blue so clouds stand out
      cloudColor: 0xffffff,        // bright clouds
      cloudShadowColor: 0x6ea7ff,  // cool, visible shadow
      sunColor: 0xfff0b0,
      sunGlareColor: 0xffffff,
      sunlightColor: 0xfff4c2
    },
    dark: {
      skyColor: 0x0b1428,          // night navy
      cloudColor: 0x2b3a55,        // blue-gray clouds
      cloudShadowColor: 0x0a0f1a,
      sunColor: 0x86e3ff,
      sunGlareColor: 0x5cc8ff,
      sunlightColor: 0x3aa9ff
    }
  };

  const prefersReduced = () =>
    (document.documentElement.getAttribute("data-reduced-motion") || "") === "reduce";

  function start() {
    if (prefersReduced()) {
      try { effect?.destroy?.(); } catch {}
      el.removeAttribute("data-variant");
      return;
    }
    if (!window.VANTA || !window.VANTA.CLOUDS) return;

    const theme = document.documentElement.getAttribute("data-theme") || "dark";
    const p = palettes[theme];

    try { effect?.destroy?.(); } catch {}
    effect = window.VANTA.CLOUDS({
      el: "#animated-bg",
      mouseControls: true,
      touchControls: true,
      gyroControls: false,
      minHeight: 200.0,
      minWidth: 200.0,
      speed: 1.10,
      backgroundAlpha: 0.0,
      skyColor: p.skyColor,
      cloudColor: p.cloudColor,
      cloudShadowColor: p.cloudShadowColor,
      sunColor: p.sunColor,
      sunGlareColor: p.sunGlareColor,
      sunlightColor: p.sunlightColor
    });

    /* mark which variant so CSS can add a visibility boost in light mode */
    el.setAttribute("data-variant", theme);
  }

  const mo = new MutationObserver(start);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme","data-reduced-motion"] });

  if (document.readyState !== "loading") start();
  else document.addEventListener("DOMContentLoaded", start);

  window.addEventListener("beforeunload", () => { try { effect?.destroy?.(); } catch {} });
})();
