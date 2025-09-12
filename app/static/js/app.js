// Core app shell
window.poom = window.poom || {};
window.poom.bus = { events:{}, on(e,cb){(this.events[e]=this.events[e]||[]).push(cb)}, emit(e,d){(this.events[e]||[]).forEach(cb=>cb(d))} };

const LS_KEY="pm:data:v1";
const load = ()=>{try{return JSON.parse(localStorage.getItem(LS_KEY))||{}}catch{return{}}};
const save = s => localStorage.setItem(LS_KEY, JSON.stringify(s));
const patch = fn => {const s=load(); const n=fn({...s})||s; save(n); return n;};
window.poom.state={load,save,patch};

window.poom.toast={show(msg,kind="info"){const c=document.querySelector(".toast-container"); if(!c) return alert(msg);
  const el=document.createElement("div"); el.className=`toast ${kind}`; el.textContent=msg; c.appendChild(el);
  requestAnimationFrame(()=>el.classList.add("show")); setTimeout(()=>{el.classList.remove("show"); setTimeout(()=>el.remove(),240)},2400);}};

function applyTheme(t){document.documentElement.setAttribute("data-theme",t)}
function applyMotionPref(pref){document.documentElement.setAttribute("data-reduced-motion",pref?"reduce":"no-preference")}
function initThemeAndMotion(){
  const s=load(); applyTheme(s.theme||"dark"); applyMotionPref(!!s.reducedMotion);
  document.querySelector(".theme-toggle")?.addEventListener("click",()=>{
    const cur=document.documentElement.getAttribute("data-theme")||"light";
    const next=cur==="dark"?"light":"dark"; applyTheme(next); patch(st=>(st.theme=next,st));
  });
}

function initSmoothScroll(){
  document.querySelectorAll('a[href^="#"]').forEach(a=>{
    a.addEventListener("click",e=>{
      const id=a.getAttribute("href"); if(!id||id==="#") return;
      const el=document.querySelector(id); if(el){e.preventDefault(); el.scrollIntoView({behavior:"smooth",block:"start"})}
    });
  });
}

function markActiveNav(){
  const file=(location.pathname.split("/").pop()||"index.html").toLowerCase();
  document.querySelectorAll("header nav a").forEach(a=>{
    const href=(a.getAttribute("href")||"").split("/").pop().toLowerCase();
    a.classList.toggle("active", href===file);
  });
}

function initHeaderShrink(){
  const hdr=document.querySelector(".sticky-nav"); if(!hdr) return;
  window.addEventListener("scroll",()=>{ hdr.classList.toggle("compact",(window.scrollY||0)>8); },{passive:true});
}

// demo auth (keeps Settings always visible)
function doSignIn(){ patch(s=>{s.user=s.user||{id:"demo",name:"You"}; return s}); window.poom.toast.show("Signed in (demo)"); }
function doSignUp(){ patch(s=>{s.user={id:"demo",name:"You"}; s.needsOnboarding=true; return s}); location.href="/app/templates/onboarding.html"; }
function doSignOut(){ patch(s=>{delete s.user; return s}); location.href="/app/templates/index.html"; }
function initHeaderAuthButtons(){
  document.getElementById("btn-signin")?.addEventListener("click",doSignIn);
  document.getElementById("btn-signup")?.addEventListener("click",doSignUp);
  document.getElementById("btn-logout")?.addEventListener("click",doSignOut);
}

// Gauges
function renderGauges(){
  document.querySelectorAll(".circular-gauge").forEach(g=>{
    const v=Math.max(0,Math.min(1,parseFloat(g.getAttribute("data-value")||"0")||0));
    const deg=Math.round(v*360);
    g.style.background=`conic-gradient(var(--accent) ${deg}deg, var(--surface-2) ${deg}deg 360deg)`;
    g.title=Math.round(v*100)+"%";
  });
}
window.poom.renderGauges=renderGauges;

// carousel
function initCarousel(){
  const track=document.querySelector(".carousel-track"); if(!track) return;
  const slides=[...document.querySelectorAll(".carousel-slide")];
  const dots=[...document.querySelectorAll(".carousel-dot")];
  let i=0; const prev=document.querySelector(".carousel-arrow.prev"); const next=document.querySelector(".carousel-arrow.next");
  const update=()=>{track.style.transform=`translateX(-${i*100}%)`; dots.forEach((d,di)=>d.classList.toggle("active",di===i));};
  prev?.addEventListener("click",()=>{i=Math.max(0,i-1);update()});
  next?.addEventListener("click",()=>{i=Math.min(slides.length-1,i+1);update()});
  dots.forEach((d,di)=>d.addEventListener("click",()=>{i=di;update()}));
  setInterval(()=>{i=(i+1)%slides.length;update()},6000); update();
}

// FAB demo
function initFab(){ document.querySelector(".fab")?.addEventListener("click",()=>window.poom.toast.show("Inhale 4 — hold 4 — exhale 6","success")); }

// boot
document.addEventListener("DOMContentLoaded",()=>{
  initThemeAndMotion(); initSmoothScroll(); markActiveNav(); initHeaderShrink(); initHeaderAuthButtons();
  initCarousel(); initFab(); renderGauges();
  const y=document.getElementById("year"); if(y) y.textContent=new Date().getFullYear();
});

window.poom.auth={doSignIn,doSignUp,doSignOut}; window.poom.ui={applyTheme,applyMotionPref};
