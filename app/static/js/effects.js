(function(){
  const reduced = ()=> (document.documentElement.getAttribute("data-reduced-motion")||"") === "reduce";

  // thin ribbon
  function initRibbon(){
    const bar=document.querySelector(".thin-ribbon"); if(!bar) return;
    const KEY="pm:ribbon:dismissed"; if(localStorage.getItem(KEY)==="1"){bar.remove();return;}
    bar.querySelector(".ribbon-close")?.addEventListener("click",()=>{bar.classList.add("hide");localStorage.setItem(KEY,"1");setTimeout(()=>bar.remove(),220)});
  }

  // reveal
  function initReveal(){
    const els=[...document.querySelectorAll("[data-reveal]")]; if(!els.length) return;
    if(reduced()){els.forEach(el=>el.classList.add("is-visible")); return;}
    const io=new IntersectionObserver((ents)=>{
      ents.forEach(e=>{ if(e.isIntersecting){ e.target.classList.add("is-visible"); io.unobserve(e.target);} });
    },{rootMargin:"0px 0px -10% 0px",threshold:.08});
    els.forEach((el,i)=>{ el.style.setProperty("--reveal-delay",`${Math.min(i*60,360)}ms`); io.observe(el); });
  }

  // tilt
  function initTilt(){
    if(reduced()) return; document.querySelectorAll("[data-tilt]").forEach(el=>{
      const max=parseFloat(el.getAttribute("data-tilt"))||6, damp=32;
      function onMove(e){const r=el.getBoundingClientRect(), cx=r.left+r.width/2, cy=r.top+r.height/2;
        const dx=(e.clientX-cx)/damp, dy=(e.clientY-cy)/damp;
        el.style.transform=`perspective(800px) rotateX(${-dy*max/10}deg) rotateY(${dx*max/10}deg)`;
      }
      function reset(){el.style.transform=""}
      el.addEventListener("mousemove",onMove); el.addEventListener("mouseleave",reset);
    });
  }

  // magnetic
  function initMagnetic(){
    if(reduced()) return; document.querySelectorAll("[data-magnetic] .btn, [data-magnetic] .pill").forEach(btn=>{
      const s=16;
      btn.addEventListener("mousemove",(e)=>{const r=btn.getBoundingClientRect(); const x=((e.clientX-r.left)/r.width-.5)*s; const y=((e.clientY-r.top)/r.height-.5)*s; btn.style.transform=`translate(${x}px,${y}px)`;});
      btn.addEventListener("mouseleave",()=>btn.style.transform="");
    });
  }

  // parallax
  function initParallax(){
    const els=document.querySelectorAll("[data-parallax]"); if(!els.length || reduced()) return;
    let ticking=false; const update=()=>{const wh=window.innerHeight; els.forEach(el=>{const sp=parseFloat(el.getAttribute("data-parallax-speed"))||.2; const r=el.getBoundingClientRect();
      if(r.top>wh || r.bottom<0) return; const off=(r.top-wh*.5)*sp; el.style.transform=`translateY(${off}px)`;}); ticking=false;};
    window.addEventListener("scroll",()=>{if(!ticking){requestAnimationFrame(update); ticking=true;}},{passive:true}); update();
  }

  document.addEventListener("DOMContentLoaded",()=>{initRibbon();initReveal();initTilt();initMagnetic();initParallax();});
})();
