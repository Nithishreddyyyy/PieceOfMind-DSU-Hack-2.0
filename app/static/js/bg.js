// Theme-aware VANTA CLOUDS (daylight for light; night sky for dark)
(function () {
  const host = document.getElementById("animated-bg");
  if (!host) return;

  let effect = null;

  const reduced = () =>
    (document.documentElement.getAttribute("data-reduced-motion") || "") === "reduce";

  function colors() {
    const dark = (document.documentElement.getAttribute("data-theme") || "light") === "dark";
    if (dark) {
      return {
        backgroundColor: 0x0b1020, // night
        skyColor: 0x0b1020,
        cloudColor: 0x3b4252,     // dark clouds
        cloudShadowColor: 0x111827
      };
    }
    return {
      backgroundColor: 0xf6f7fb, // daylight
      skyColor: 0xcfe9ff,
      cloudColor: 0xffffff,
      cloudShadowColor: 0xa0aec0
    };
  }

  function init() {
    if (reduced()) { try { effect?.destroy?.(); } catch{}; return; }
    if (!window.VANTA || !window.VANTA.CLOUDS) return;

    const c = colors();
    try {
      effect?.destroy?.();
      effect = window.VANTA.CLOUDS({
        el: "#animated-bg",
        mouseControls: true,
        touchControls: true,
        gyroControls: false,
        minHeight: 200.0,
        minWidth: 200.0,
        speed: 1.10,
        skyColor: c.skyColor,
        cloudColor: c.cloudColor,
        cloudShadowColor: c.cloudShadowColor,
        backgroundColor: c.backgroundColor
      });
    } catch {}
  }

  // Re-init on theme toggle
  const mo = new MutationObserver(init);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme","data-reduced-motion"] });

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);

  window.addEventListener("beforeunload", () => { try { effect?.destroy?.(); } catch {} });
})();
