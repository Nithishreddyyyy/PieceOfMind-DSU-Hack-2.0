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
})();
