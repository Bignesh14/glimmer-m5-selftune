"""mascot.py — "Spark", an original little mascot for the demo pages (no images, no external files).

A glowing spark with eyes that rides along the top edge of the panes while text streams, hops to the H8 pane when it
starts, dashes and throws sparkles when a run of confirmed ("free") guesses arrives, and cheers when the faster side
finishes. It never covers the text (it sits on the pane borders and ignores the mouse), can be hidden with the
"hide Spark" link, and stays still for people who ask their system for reduced motion.

Used by demo.py (replay page) and ui.py (web page): both insert MASCOT_CSS, MASCOT_HTML and MASCOT_JS, then call
window.spark.* from their own scripts.
"""

MASCOT_CSS = """
#spark{position:fixed;left:0;top:0;width:60px;height:60px;pointer-events:none;z-index:50;
transition:transform .55s cubic-bezier(.34,1.56,.64,1);will-change:transform}
#spark svg{width:60px;height:60px;overflow:visible;animation:sp-float 2.6s ease-in-out infinite}
#spark .body{transform-origin:23px 26px;transition:transform .25s}
#spark.hop .body{animation:sp-hop .5s ease-out}
#spark.dash .body{animation:sp-dash .35s ease-out}
#spark.cheer .body{animation:sp-cheer .9s ease-in-out 2}
#spark.shrug .body{animation:sp-shrug .9s ease-in-out}
#spark .lid{transform-origin:center;animation:sp-blink 4.2s infinite}
#spark .mouth-smile{opacity:0;transition:opacity .2s}#spark.cheer .mouth-smile{opacity:1}#spark.cheer .mouth-o{opacity:0}
.sp-dot{position:fixed;width:6px;height:6px;border-radius:50%;pointer-events:none;z-index:49;
background:radial-gradient(circle,#ffe066 0,#2fd27a 45%,rgba(47,210,122,0) 72%);animation:sp-fly .8s ease-out forwards}
#sparktoggle{position:fixed;right:14px;bottom:10px;font-size:12px;color:#5d6678;cursor:pointer;z-index:51;
background:none;border:0;padding:4px 6px}
@keyframes sp-float{0%,100%{transform:translateY(0)}50%{transform:translateY(-4px)}}
@keyframes sp-hop{0%{transform:translateY(0) scale(1)}35%{transform:translateY(-16px) scale(.9,1.1)}70%{transform:translateY(0) scale(1.12,.88)}100%{transform:none}}
@keyframes sp-dash{0%{transform:skewX(0)}40%{transform:skewX(-18deg) translateX(4px)}100%{transform:none}}
@keyframes sp-cheer{0%,100%{transform:translateY(0) rotate(0)}25%{transform:translateY(-14px) rotate(-12deg)}50%{transform:translateY(0) rotate(0)}75%{transform:translateY(-14px) rotate(12deg)}}
@keyframes sp-shrug{0%,100%{transform:rotate(0)}30%{transform:rotate(-14deg)}60%{transform:rotate(10deg)}}
@keyframes sp-blink{0%,92%,100%{transform:scaleY(0)}95%{transform:scaleY(1)}}
@keyframes sp-fly{0%{opacity:1;transform:translate(0,0) scale(1)}100%{opacity:0;transform:translate(var(--dx),var(--dy)) scale(.3)}}
@media (prefers-reduced-motion: reduce){#spark,#spark svg,#spark .body,.sp-dot{animation:none!important;transition:none!important}}
"""

MASCOT_HTML = """
<div id="spark" aria-hidden="true"><svg viewBox="0 0 46 46"><defs>
<radialGradient id="spg" cx="40%" cy="35%" r="70%"><stop offset="0" stop-color="#f4ffe8"/><stop offset=".45" stop-color="#b5ec7d"/>
<stop offset="1" stop-color="#16a34a"/></radialGradient>
<filter id="spglow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="2.4" result="b"/>
<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
<g class="body" filter="url(#spglow)">
<path d="M23 3 C26 13 31 16 41 19 C32 23 29 27 27 41 C24 33 21 31 9 29 C17 24 20 17 23 3 Z" fill="url(#spg)" stroke="#0f7a3a" stroke-width=".9" stroke-linejoin="round"/>
<ellipse cx="20" cy="22" rx="2.3" ry="3" fill="#13261a"/><ellipse cx="27" cy="22" rx="2.3" ry="3" fill="#13261a"/>
<circle cx="20.8" cy="21" r=".8" fill="#fff"/><circle cx="27.8" cy="21" r=".8" fill="#fff"/>
<rect class="lid" x="17" y="18.5" width="13" height="7" fill="#b5ec7d"/>
<ellipse class="mouth-o" cx="23.5" cy="28" rx="1.4" ry="1.1" fill="#13261a"/>
<path class="mouth-smile" d="M20.5 27 Q23.5 31 26.5 27" stroke="#13261a" stroke-width="1.4" fill="none" stroke-linecap="round"/>
</g></svg></div>
<button id="sparktoggle" type="button">hide Spark</button>
"""

MASCOT_JS = """
(function(){
const el=document.getElementById("spark"),tg=document.getElementById("sparktoggle");
const still=window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let hidden=false,lastDash=0,x=0,y=0;
function place(nx,ny){x=nx;y=ny;el.style.transform=`translate(${nx}px,${ny}px)`}
function anim(c,ms){if(still||hidden)return;el.classList.remove("hop","dash","cheer","shrug");void el.offsetWidth;el.classList.add(c);
setTimeout(()=>el.classList.remove(c),ms)}
function onPane(pane,frac){if(!pane)return;const r=pane.getBoundingClientRect();
const f=Math.max(0,Math.min(1,frac||0));place(r.left+10+f*(r.width-80),r.top-46)}
function sparkle(n){if(still||hidden)return;for(let i=0;i<(n||6);i++){const d=document.createElement("div");d.className="sp-dot";
d.style.left=(x+28)+"px";d.style.top=(y+28)+"px";const a=Math.random()*Math.PI*2,s=18+Math.random()*26;
d.style.setProperty("--dx",Math.cos(a)*s+"px");d.style.setProperty("--dy",Math.sin(a)*s-8+"px");document.body.appendChild(d);
setTimeout(()=>d.remove(),850)}}
window.spark={
  home(){const h=document.querySelector("h1");if(h){const r=h.getBoundingClientRect();place(Math.min(r.right+14,window.innerWidth-60),r.top-8)}},
  ride(pane,frac){onPane(pane,frac)},
  hopTo(pane,frac){onPane(pane,frac);anim("hop",520)},
  free(){const t=performance.now();if(t-lastDash<350)return;lastDash=t;anim("dash",360);sparkle(4)},
  cheer(){anim("cheer",1850);sparkle(14)},
  shrug(){anim("shrug",950)}
};
tg.onclick=()=>{hidden=!hidden;el.style.display=hidden?"none":"";tg.textContent=hidden?"show Spark":"hide Spark"};
window.addEventListener("resize",()=>window.spark.home());
setTimeout(()=>window.spark.home(),50);
})();
"""


def inject(page, hooks_js=""):
    """Insert the mascot into a page that has </style>, </body> and a <script> block; hooks_js runs after it."""
    page = page.replace("</style>", MASCOT_CSS + "</style>", 1)
    return page.replace("</body>", MASCOT_HTML + "<script>" + MASCOT_JS + hooks_js + "</script></body>", 1)
