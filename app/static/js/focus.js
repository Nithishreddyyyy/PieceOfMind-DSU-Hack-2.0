(function(){
  const rec = (()=>{ try{return JSON.parse(localStorage.getItem("pm:rec"))}catch{return null} })();
  if(rec){
    // highlight a suggested card
    const anchor = document.getElementById("recommendations");
    if(anchor){ setTimeout(()=>anchor.scrollIntoView({behavior:"smooth"}),300); }
    const tag = (rec.strain>0.65) ? "#card-stress" : (rec.drift>0.5) ? "#card-burnout" : "#card-anxiety";
    const el = document.querySelector(tag); if(el){ el.classList.add("pulse"); setTimeout(()=>el.classList.remove("pulse"),2400); }
  }

  // micro break reminder (front-end only)
  let mbIv=null; const btn=document.getElementById("btn-microbreak");
  btn?.addEventListener("click",()=>{
    if(mbIv){ clearInterval(mbIv); mbIv=null; btn.textContent="Enable Micro Break Reminders"; window.poom.toast.show("Reminders off"); return; }
    mbIv=setInterval(()=>window.poom.toast.show("Time for a 30s stretch","success"), 5*60*1000);
    btn.textContent="Disable Micro Break Reminders"; window.poom.toast.show("Reminders every 5 min enabled","success");
  });

  // simple goal tracker
  const input=document.getElementById("goal-input"), add=document.getElementById("goal-add"), list=document.getElementById("goal-list");
  add?.addEventListener("click",()=>{
    const v=(input.value||"").trim(); if(!v) return;
    const li=document.createElement("li"); li.innerHTML=`<label><input type="checkbox"> ${v}</label>`;
    list.appendChild(li); input.value="";
  });

  // binaural beats volume remember
  const audio=document.querySelector(".binaural audio");
  if(audio){ const s=window.poom.state.load(); const v=s.ambienceVol??0.4; audio.volume=v; audio.addEventListener("volumechange",()=>window.poom.state.patch(st=>(st.ambienceVol=audio.volume,st))); }
});
const boxBtn = $('#box-start');
  const ytWrap = $('#box-yt-wrap');
  const yt     = $('#box-yt');
  if (boxBtn){
    boxBtn.addEventListener('click', () => {
      if (yt && !yt.src){
        // Small, clean, autoplay on user gesture
        yt.src = "https://www.youtube-nocookie.com/embed/gUqOnaifsRk?enablejsapi=1&rel=0&modestbranding=1&autoplay=1&playsinline=1";
      } else {
        try { yt.contentWindow.postMessage('{"event":"command","func":"playVideo","args":""}','*'); } catch(e){}
      }
      ytWrap.hidden = false;
    });
  }

  // ----------------- Rain / Waves -----------------
  const rainBtn  = $('#btn-rain');
  const wavesBtn = $('#btn-waves');

  // Try local files; fall back gracefully
  const rainAudio  = new Audio("../static/audio/rain.mp3");
  const wavesAudio = new Audio("../static/audio/waves.mp3");
  rainAudio.preload = "none"; wavesAudio.preload = "none";
  rainAudio.volume = 0.6; wavesAudio.volume = 0.6;

  function effect(kind){
    const d = document.createElement('div');
    d.className = 'fx ' + (kind === 'rain' ? 'fx-rain' : 'fx-waves');
    document.body.appendChild(d);
    setTimeout(() => d.remove(), 3000);
  }
  async function playClip(audio, name){
    try{
      // Stop the other clip
      [rainAudio, wavesAudio].forEach(a => { if (a!==audio){ a.pause(); a.currentTime=0; } });
      await audio.play();
      effect(name);
      setTimeout(()=>{ audio.pause(); audio.currentTime=0; }, 10000); // 10s taste
    }catch(err){
      window.poom?.toast?.show?.(`${name} sound missing — add /static/audio/${name}.mp3`, "warn");
      effect(name);
    }
  }
  rainBtn?.addEventListener('click', () => playClip(rainAudio, 'rain'));
  wavesBtn?.addEventListener('click', () => playClip(wavesAudio, 'waves'));

  // ----------------- Micro Break Reminders -----------------
  const mbBtn = $('#btn-microbreak');
  let mbTimer = null, mbMins = +localStorage.getItem('pm:micro:mins') || 20;

  function notify(msg){
    if ('Notification' in window && Notification.permission === 'granted'){
      try{ new Notification('PieceOfMind', { body: msg }); }catch{}
    }
    window.poom?.toast?.show?.(msg, "success");
  }
  function startMicroBreaks(){
    if ('Notification' in window && Notification.permission !== 'granted'){
      Notification.requestPermission().catch(()=>{});
    }
    stopMicroBreaks();
    const ms = mbMins * 60 * 1000;
    mbTimer = setInterval(() => notify('Micro break: stand, stretch, and relax your gaze.'), ms);
    mbBtn.textContent = `Disable Micro Break Reminders`;
    localStorage.setItem('pm:micro:on', '1');
    notify(`Micro breaks enabled — reminder every ${mbMins} min.`);
  }
  function stopMicroBreaks(){
    if (mbTimer){ clearInterval(mbTimer); mbTimer = null; }
    mbBtn.textContent = `Enable Micro Break Reminders`;
    localStorage.removeItem('pm:micro:on');
  }
  mbBtn?.addEventListener('click', () => {
    if (mbTimer || localStorage.getItem('pm:micro:on') === '1'){ stopMicroBreaks(); return; }
    const v = prompt('Minutes between reminders?', String(mbMins));
    mbMins = Math.max(1, parseInt(v||mbMins,10)||mbMins);
    localStorage.setItem('pm:micro:mins', String(mbMins));
    startMicroBreaks();
  });
  // restore if previously on
  if (localStorage.getItem('pm:micro:on') === '1'){ startMicroBreaks(); }

  // ----------------- Mindfulness mini session (2 minutes) -----------------
  const msOverlay = $('#miniSession'), msRing = $('#msRing'), msTime = $('#msTime'), msEnd = $('#msEnd');
  const mindfulBtn = $('#mindful-play');
  let msTick = null, msLeft = 120, msAudio = null;

  function msDraw(){
    // reuse your .visual-ring CSS via --p
    const p = 1 - (msLeft/120);
    msRing.style.setProperty('--p', p);
    const m = String(Math.floor(msLeft/60)).padStart(2,'0');
    const s = String(msLeft%60).padStart(2,'0');
    msTime.textContent = `${m}:${s}`;
  }
  function msStop(){
    clearInterval(msTick); msTick=null; msLeft=120; msDraw();
    msOverlay.classList.remove('active');
    try{ msAudio?.pause(); msAudio=null; }catch{}
  }
  mindfulBtn?.addEventListener('click', () => {
    msOverlay.classList.add('active'); msLeft=120; msDraw();
    msAudio = new Audio("../static/audio/calm1.mp3"); msAudio.volume=.6;
    msAudio.play().catch(()=>{});
    msTick = setInterval(() => {
      msLeft -= 1; if (msLeft<=0){ notify('Mini session complete ✨'); msStop(); } else { msDraw(); }
    }, 1000);
  });
  msEnd?.addEventListener('click', msStop);

  // ----------------- Tips generator -----------------
  const tips = [
    "Schedule recovery like a meeting.",
    "Switch tasks when attention drifts—don’t force it.",
    "Use a 25–5 focus/break rhythm for deep work.",
    "Stand, stretch, hydrate every 45–60 minutes.",
    "Batch notifications; check messages on your terms.",
    "Finish the day by teeing up one tiny ‘first step’ for tomorrow.",
    "Lower the bar to start—momentum beats perfection."
  ];
  $('#tip-generate')?.addEventListener('click', () => {
    const t = tips[Math.floor(Math.random()*tips.length)];
    $('#tip-text').textContent = t;
  });

  // ----------------- Goal quick-add stays as-is -----------------
  $('#goal-add')?.addEventListener('click', () => {
    const inp = $('#goal-input'); const v = (inp.value||"").trim(); if(!v) return;
    const li = document.createElement('li'); li.textContent = v; $('#goal-list').appendChild(li); inp.value="";
  })();
