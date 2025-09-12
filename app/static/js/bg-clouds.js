// VANTA CLOUDS (theme-aware) – no placement changes
(function () {
  const host = document.getElementById("animated-bg");
  if (!host) return;

  let effect = null;
  const isReduced = () =>
    (document.documentElement.getAttribute("data-reduced-motion") || "") === "reduce";

  function palette() {
    const dark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";
    return dark
      ? { backgroundColor: 0x0b1020, skyColor: 0x0b1020, cloudColor: 0x2d3748, cloudShadowColor: 0x111827 }
      : { backgroundColor: 0xf6f7fb, skyColor: 0xcfe9ff, cloudColor: 0xffffff, cloudShadowColor: 0xa0aec0 };
  }

  function init() {
    // Deactivate when reduced motion is on
    if (isReduced()) { try { effect?.destroy?.(); } catch {} return; }
    if (!window.VANTA || !window.VANTA.CLOUDS) return;

    const c = palette();
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

  const mo = new MutationObserver(init);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme","data-reduced-motion"] });

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);

  window.addEventListener("beforeunload", () => { try { effect?.destroy?.(); } catch {} });
})();
