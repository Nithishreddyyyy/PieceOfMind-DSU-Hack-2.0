// Vanta HALO with theme-aware colors + fallback
(function () {
  const target = document.getElementById("animated-bg");
  if (!target) return;
  let effect = null;
  const colors = () => {
    const theme = document.documentElement.getAttribute("data-theme") || "light";
    return (theme === "dark") ? { base: 0x6d5ef4, bg: 0x0b1020 } : { base: 0x4c5bd5, bg: 0xf6f7fb };
  };
  function init() {
    if (!window.VANTA || !window.VANTA.HALO) return;
    const { base, bg } = colors();
    try {
      effect?.destroy?.();
      effect = window.VANTA.HALO({
        el: "#animated-bg",
        mouseControls: true, touchControls: true, gyroControls: false,
        minHeight: 200.0, minWidth: 200.0,
        size: 1.1, amplitudeFactor: 1.15,
        baseColor: base, backgroundColor: bg
      });
    } catch {}
  }
  const mo = new MutationObserver(init);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  if (document.readyState !== "loading") init(); else document.addEventListener("DOMContentLoaded", init);
  window.addEventListener("beforeunload", () => { try { effect?.destroy?.(); } catch {} });
})();
