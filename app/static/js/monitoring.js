(function(){
  const $ = (q, ctx=document) => ctx.querySelector(q);
  const panel = $("#sensor-panel"); if (!panel) return;

  // Step 2: default message
  panel.innerHTML = `<div class="card"><p class="muted">Choose a source above to begin monitoring.</p></div>`;

  function block(startLabel){return `
    <div style="display:flex;gap:8px;margin-top:8px">
      <button class="btn btn-primary start">${startLabel}</button>
      <button class="btn btn-ghost stop" disabled>Stop</button>
    </div>
    <p class="muted status">Idle.</p>
    <div class="results hidden" style="margin-top:6px">
      <h4>Results</h4>
      <ul>
        <li>Strain: <strong class="r-strain">—</strong></li>
        <li>Drift: <strong class="r-drift">—</strong></li>
        <li>Summary: <span class="r-sum">—</span></li>
      </ul>
      <button class="btn btn-success send-focus">Apply in Focus</button>
    </div>`;}

  function wire(card){
    const start=card.querySelector(".start"), stop=card.querySelector(".stop"),
          status=card.querySelector(".status"), res=card.querySelector(".results"),
          rS=card.querySelector(".r-strain"), rD=card.querySelector(".r-drift"), rSum=card.querySelector(".r-sum");
    let t=null;
    start.addEventListener("click",()=>{ start.disabled=true; stop.disabled=false; res.classList.add("hidden"); status.textContent="Live monitoring in progress…";
      t=setTimeout(()=>{ status.textContent="Processing…"; setTimeout(()=>{ status.textContent="Done."; stop.disabled=true; start.disabled=false;
        const s=0.35+Math.random()*0.4, d=0.2+Math.random()*0.35;
        rS.textContent=Math.round(s*100); rD.textContent=Math.round(d*100);
        rSum.textContent = s>0.65?"High strain — try stress relief." : d>0.5?"Attention drift — micro break." : "Balanced.";
        res.classList.remove("hidden");
      },900); },1500);
    });
    stop.addEventListener("click",()=>{ if(t) clearTimeout(t); status.textContent="Stopped."; stop.disabled=true; start.disabled=false; });
    card.querySelector(".send-focus")?.addEventListener("click",()=>{ location.href="/app/templates/focus.html#recommendations"; });
  }

  document.querySelectorAll(".trimodal .pill").forEach(p=>{
    p.addEventListener("click", async ()=>{
      document.querySelectorAll(".trimodal .pill").forEach(x=>x.classList.remove("active"));
      p.classList.add("active");
      const k = p.textContent.trim().toLowerCase();

      if (k === "eeg") {
        panel.innerHTML = `<div class="card">
          <h4>EEG</h4>
          <p class="muted">Neural strain/relaxation estimates.</p>
          <p><a class="btn btn-primary" target="_blank" href="https://neurosity.co/blog/eeg-headset-placement" rel="noopener">EEG Placement Guide</a></p>
          ${block("Start monitoring")}
        </div>`;
        wire(panel.querySelector(".card"));
      }

      else if (k === "apple watch") {
        panel.innerHTML = `<div class="card">
          <h4>Apple Watch</h4>
          <p class="muted">Upload HRV/export file to analyze.</p>
          <input type="file" accept=".csv,.json" style="margin:.25rem 0">
          ${block("Start analyzing")}
        </div>`;
        wire(panel.querySelector(".card"));
      }

      else if (k === "webcam") {
        panel.innerHTML = `<div class="card">
          <h4>Webcam</h4>
          <p class="muted">Blink/eye openness & fatigue signals.</p>
          <button class="btn btn-ghost ask">Allow webcam access</button>
          <div class="after hidden">${block("Start monitoring")}</div>
        </div>`;
        const card = panel.querySelector(".card");
        card.querySelector(".ask").addEventListener("click", async ()=>{
          if (!confirm("Allow webcam access?")) return;
          try { await navigator.mediaDevices.getUserMedia({video:true}); window.poom?.toast?.show?.("Webcam ready","success");
            card.querySelector(".after").classList.remove("hidden"); wire(card);
          } catch { window.poom?.toast?.show?.("Camera permission denied","error"); }
        });
      }

      else if (k === "keyboard") {
        panel.innerHTML = `<div class="card">
          <h4>Keyboard</h4>
          <p class="muted">Track cadence variability while you type.</p>
          <button class="btn btn-ghost ask">Enable keyboard tracking</button>
          <div class="after hidden">${block("Start monitoring")}</div>
        </div>`;
        const card = panel.querySelector(".card");
        card.querySelector(".ask").addEventListener("click", ()=>{
          if (!confirm("Track keyboard cadence while this tab is open?")) return;
          window.poom?.toast?.show?.("Keyboard tracking enabled","success");
          card.querySelector(".after").classList.remove("hidden"); wire(card);
        });
      }

      else { // Tab/Scroll
        panel.innerHTML = `<div class="card">
          <h4>Tab & Scroll</h4>
          <p class="muted">Monitor tab switches and scroll rhythm.</p>
          <button class="btn btn-ghost ask">Enable attention tracking</button>
          <div class="after hidden">${block("Start monitoring")}</div>
        </div>`;
        const card = panel.querySelector(".card");
        card.querySelector(".ask").addEventListener("click", ()=>{
          if (!confirm("Allow tab/scroll attention tracking for this session?")) return;
          window.poom?.toast?.show?.("Attention tracking enabled","success");
          card.querySelector(".after").classList.remove("hidden"); wire(card);
        });
      }
    });
  });
})();
