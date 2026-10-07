import json
import os
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Flight Price Intelligence Engine",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .price-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        height: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CONSTANTS & CONFIGURATIONS
# ==============================================================================
DEFAULT_MODEL_PATH = Path(r"D:\faeature eng\EDA\flight_price_rf_model.pkl")
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

CITIES: List[str] = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
AIRLINES: List[str] = [
    "Air_India", "AirAsia", "GO FIRST", "Indigo",
    "SpiceJet", "StarAir", "Trujet", "Vistara"
]
AIRLINE_INDEX: Dict[str, int] = {airline: i for i, airline in enumerate(AIRLINES)}

PERIOD_HOURS: Dict[str, int] = {
    "Early morning": 4,
    "Morning": 9,
    "Evening": 17,
    "Night": 22,
}

PERIOD_FEATURES: Dict[str, str] = {
    "Early morning": "Early_morning",
    "Morning": "Morning",
    "Evening": "Evening",
    "Night": "Night",
}

ROUTE_INDEX: Dict[str, int] = {
    f"{src}-{dst}": idx
    for idx, (src, dst) in enumerate(
        (src, dst) for src in CITIES for dst in CITIES if src != dst
    )
}

DAYS_IN_MONTH: Dict[int, int] = {
    1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
}

CITY_COORDS = {  # (longitude, latitude)
    "Bangalore": (77.59, 12.97),
    "Chennai":   (80.27, 13.08),
    "Delhi":     (77.21, 28.61),
    "Hyderabad": (78.48, 17.38),
    "Kolkata":   (88.36, 22.57),
    "Mumbai":    (72.88, 19.08),
}

COUNTRY_NAME = "India"
COUNTRY_FLAG = "in"  # ISO code used for the flag image

# ==============================================================================
# 3. HELPER FUNCTIONS & MODEL LOADER
# ==============================================================================
@st.cache_resource(show_spinner="Loading predictive machine learning model...")
def load_trained_model(path: Path):
    """Loads and caches the trained Random Forest model with fallback checks."""
    search_paths = [
        path,
        Path("flight_price_rf_model.pkl"),
        Path("final_flight_price_rf_model.pkl"),
        Path("../EDA/flight_price_rf_model.pkl")
    ]

    for p in search_paths:
        if p.is_file():
            try:
                return joblib.load(p)
            except Exception as e:
                st.error(f"Error reading model binary at {p}: {e}")
                return None

    return None


ROUTE_ANIMATION_HTML = r"""<!-- __NONCE__ -->
<style>
  *{box-sizing:border-box}
  html,body{margin:0;background:transparent;font-family:Inter,"Segoe UI",system-ui,-apple-system,sans-serif}
  #stage{position:relative;height:520px;border-radius:16px;overflow:hidden;border:1px solid #1e293b;background:#05080f}
  #stage canvas{display:block}
  #ui{position:absolute;inset:0;pointer-events:none;overflow:hidden}
  .pin{position:absolute;left:0;top:0;width:0;height:0;opacity:0;transition:opacity .35s;will-change:transform}
  .pin.on{opacity:1}
  .dot{position:absolute;left:-8px;top:-8px;width:16px;height:16px;border-radius:50%;background:#1da1ff;box-shadow:0 0 16px 3px rgba(29,161,255,.8)}
  .ring{position:absolute;left:-17px;top:-17px;width:34px;height:34px;border-radius:50%;border:3px solid #fff;opacity:0;transition:opacity .4s;box-shadow:0 0 10px rgba(255,255,255,.5)}
  .pin.ringon .ring{opacity:1}
  .wave{position:absolute;left:-17px;top:-17px;width:34px;height:34px;border-radius:50%;border:2px solid #4cc2ff;opacity:0}
  .pin.pulse .wave{animation:wave 1.8s ease-out infinite}
  .pin.pulse .wave.w2{animation-delay:.9s}
  @keyframes wave{0%{transform:scale(.7);opacity:.9}100%{transform:scale(3.6);opacity:0}}
  .tag{position:absolute;left:0;bottom:30px;transform:translate(-50%,8px);white-space:nowrap;text-align:center;
       padding:7px 16px 8px;border-radius:10px;background:rgba(30,38,52,.55);backdrop-filter:blur(8px);
       -webkit-backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.12);color:#fff;opacity:0;
       transition:opacity .5s,transform .5s;text-shadow:0 1px 8px rgba(0,0,0,.6)}
  .pin.show .tag{opacity:1;transform:translate(-50%,0)}
  .city{font-weight:800;font-size:22px;letter-spacing:.04em;text-transform:uppercase;line-height:1.15}
  .country{display:flex;align-items:center;justify-content:center;gap:8px;font-weight:700;font-size:15px;letter-spacing:.12em;
           text-transform:uppercase;margin-top:3px;color:#e2e8f0}
  .flag{width:22px;height:15px;border-radius:2px;object-fit:cover;box-shadow:0 0 0 1px rgba(255,255,255,.25)}
  #km{position:absolute;left:50%;bottom:22px;transform:translateX(-50%);color:#fff;font-weight:800;font-size:26px;
      letter-spacing:.02em;font-variant-numeric:tabular-nums;text-shadow:0 2px 14px rgba(0,0,0,.85);opacity:0;transition:opacity .5s}
  #km.on{opacity:1}
  .chip{position:absolute;top:14px;left:14px;padding:8px 14px;border-radius:999px;background:rgba(15,23,42,.6);
        backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.12);
        color:#e2e8f0;font-weight:700;font-size:13px;letter-spacing:.06em;text-transform:uppercase}
  .chip b{color:#38bdf8}
  #replay{position:absolute;top:14px;right:14px;pointer-events:auto;cursor:pointer;padding:8px 14px;border-radius:999px;
          background:rgba(15,23,42,.6);border:1px solid rgba(255,255,255,.18);color:#fff;font-weight:700;font-size:13px;
          font-family:inherit;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);opacity:0;transition:opacity .4s,background .2s}
  #replay.on{opacity:1}
  #replay:hover{background:rgba(56,189,248,.35)}
  #load{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:14px;
        color:#94a3b8;font-weight:600;font-size:14px;background:#05080f;transition:opacity .6s;z-index:5}
  #load.off{opacity:0;pointer-events:none}
  .spin{width:34px;height:34px;border-radius:50%;border:3px solid #1e293b;border-top-color:#38bdf8;animation:sp 1s linear infinite}
  @keyframes sp{to{transform:rotate(360deg)}}
</style>

<div id="stage">
  <div id="ui">
    <div class="chip" id="chip"></div>
    <button id="replay">&#8635; Replay</button>
    <div id="km">+0 km</div>
  </div>
  <div id="load"><div class="spin"></div><div>Loading satellite imagery&hellip;</div></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/topojson-client@3"></script>
<script>
(function () {
  const SRC = __SRC__, DST = __DST__;
  const FLAG = "__FLAG__", COUNTRY = "__COUNTRY__";

  // ---------- helpers ----------
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  const easeIO = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  const easeSine = t => -(Math.cos(Math.PI * t) - 1) / 2;
  function ll2v(lat, lon) {
    const la = lat * Math.PI / 180, lo = lon * Math.PI / 180;
    return new THREE.Vector3(Math.cos(la) * Math.cos(lo), Math.sin(la), -Math.cos(la) * Math.sin(lo));
  }

  // ---------- renderer / scene ----------
  const stage = document.getElementById("stage");
  const ui = document.getElementById("ui");
  let W = stage.clientWidth || 900;
  const H = 520;

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(W, H);
  renderer.setClearColor(0x05080f, 1);
  renderer.outputEncoding = THREE.sRGBEncoding;
  stage.insertBefore(renderer.domElement, stage.firstChild);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, W / H, 0.05, 150);
  scene.add(camera);
  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(2, 1.2, 3);
  camera.add(sun);
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));

  // globe
  const globeMat = new THREE.MeshPhongMaterial({ color: 0xffffff, shininess: 6, specular: new THREE.Color(0x111827) });
  const globe = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 96), globeMat);
  scene.add(globe);

  // atmosphere glow
  const atmo = new THREE.Mesh(
    new THREE.SphereGeometry(1.13, 64, 64),
    new THREE.ShaderMaterial({
      vertexShader: "varying vec3 vN; void main(){ vN = normalize(normalMatrix * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
      fragmentShader: "varying vec3 vN; void main(){ float a = clamp(-vN.z * 1.9, 0.0, 1.0); float i = pow(a, 2.4); gl_FragColor = vec4(0.25, 0.58, 1.0, 1.0) * i * 1.3; }",
      blending: THREE.AdditiveBlending, side: THREE.BackSide, transparent: true, depthWrite: false
    })
  );
  scene.add(atmo);

  // stars
  (function () {
    const p = [];
    for (let i = 0; i < 1600; i++) {
      const u = Math.random() * 2 - 1, th = Math.random() * Math.PI * 2, s = Math.sqrt(1 - u * u), r = 60;
      p.push(r * s * Math.cos(th), r * u, r * s * Math.sin(th));
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0xffffff, size: 0.14, transparent: true, opacity: 0.75, depthWrite: false })));
  })();

  // ---------- route ----------
  const A = ll2v(SRC.lat, SRC.lon), B = ll2v(DST.lat, DST.lon);
  const omega = Math.max(Math.acos(clamp(A.dot(B), -1, 1)), 0.01);
  const distKm = Math.round(omega * 6371);

  function routePoint(t) {
    const s = Math.sin(omega);
    const a = Math.sin((1 - t) * omega) / s, b = Math.sin(t * omega) / s;
    const p = A.clone().multiplyScalar(a).add(B.clone().multiplyScalar(b)).normalize();
    return p.multiplyScalar(1.004 + Math.sin(Math.PI * t) * Math.min(0.05, omega * 0.14));
  }
  const pts = [];
  for (let i = 0; i <= 240; i++) pts.push(routePoint(i / 240));
  const curve = new THREE.CatmullRomCurve3(pts);

  const SEG = 480, RAD = 8;
  const lineCol = 0x2aa4ff;
  const ghost = new THREE.Mesh(
    new THREE.TubeGeometry(curve, SEG, 0.0009, 6, false),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.22 })
  );
  const glow = new THREE.Mesh(
    new THREE.TubeGeometry(curve, SEG, 0.0078, RAD, false),
    new THREE.MeshBasicMaterial({ color: lineCol, transparent: true, opacity: 0.3, blending: THREE.AdditiveBlending, depthWrite: false })
  );
  const core = new THREE.Mesh(
    new THREE.TubeGeometry(curve, SEG, 0.0024, RAD, false),
    new THREE.MeshBasicMaterial({ color: lineCol })
  );
  scene.add(ghost, glow, core);
  function setProgress(f) {
    const n = Math.floor(clamp(f, 0, 1) * SEG) * RAD * 6;
    core.geometry.setDrawRange(0, n);
    glow.geometry.setDrawRange(0, n);
  }

  // plane
  function makePlaneTexture() {
    const c = document.createElement("canvas");
    c.width = c.height = 256;
    const g = c.getContext("2d");
    g.translate(128, 128); g.scale(1.9, 1.9);
    const half = [[60, 0], [50, -4], [18, -5], [-6, -50], [-16, -50], [-12, -6], [-34, -4], [-42, -20], [-50, -20], [-48, 0]];
    g.beginPath();
    half.forEach((q, i) => i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]));
    for (let i = half.length - 2; i >= 1; i--) g.lineTo(half[i][0], -half[i][1]);
    g.closePath();
    g.shadowColor = "rgba(30,160,255,.95)"; g.shadowBlur = 14;
    g.fillStyle = "#ffffff"; g.fill();
    g.shadowBlur = 0; g.lineWidth = 1.4; g.strokeStyle = "rgba(20,60,110,.55)"; g.stroke();
    const t = new THREE.CanvasTexture(c);
    t.encoding = THREE.sRGBEncoding;
    return t;
  }
  const plane = new THREE.Sprite(new THREE.SpriteMaterial({ map: makePlaneTexture(), transparent: true, depthTest: false, depthWrite: false }));
  plane.renderOrder = 20;
  scene.add(plane);

  // ---------- pins / UI ----------
  function makePin(info, showFlag) {
    const el = document.createElement("div");
    el.className = "pin";
    el.innerHTML =
      '<div class="wave"></div><div class="wave w2"></div><div class="ring"></div><div class="dot"></div>' +
      '<div class="tag"><div class="city"></div><div class="country"><img class="flag" alt=""><span></span></div></div>';
    el.querySelector(".city").textContent = info.name;
    el.querySelector(".country span").textContent = COUNTRY;
    const img = el.querySelector(".flag");
    img.src = "https://flagcdn.com/w40/" + FLAG + ".png";
    img.onerror = () => { img.style.display = "none"; };
    ui.appendChild(el);
    return { el, pos: ll2v(info.lat, info.lon).multiplyScalar(1.002) };
  }
  const pinA = makePin(SRC), pinB = makePin(DST);
  const kmEl = document.getElementById("km");
  const replayBtn = document.getElementById("replay");
  document.getElementById("chip").innerHTML = SRC.name + " &rarr; <b>" + DST.name + "</b>";

  function toScreen(v) {
    const p = v.clone().project(camera);
    return { x: (p.x * .5 + .5) * W, y: (-p.y * .5 + .5) * H };
  }

  // ---------- timeline ----------
  const T_INTRO = 2.4, T_FLIGHT = 5.6, T_ARRIVE = 3.0, T_TOTAL = T_INTRO + T_FLIGHT + T_ARRIVE;
  const LEAD = 0.035;
  const dFollow = 1 + clamp(0.22 + omega * 0.8, 0.3, 0.9);
  const dOverview = 1 + clamp(omega * 3.2, 0.5, 1.6);
  const D_START = 4.4;
  const startDir = ll2v(clamp(SRC.lat + 12, -80, 80), SRC.lon - 95).normalize();
  const followStart = curve.getPointAt(LEAD).clone().normalize();
  const qIntro = new THREE.Quaternion().setFromUnitVectors(startDir, followStart);
  const midDir = curve.getPointAt(0.5).clone().normalize();

  let camDir = startDir.clone(), camDist = D_START, heading = null;
  let clock = 0, last = 0, running = false, raf = 0, ready = false;

  function placeCamera() {
    camera.position.copy(camDir).multiplyScalar(camDist);
    camera.up.set(0, 1, 0);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
  }

  function step(t, dt) {
    let f = 0, u = 0, planeAlpha = 0;

    if (t < T_INTRO) {
      u = easeIO(t / T_INTRO);
      const q = new THREE.Quaternion().slerp(qIntro, u);
      camDir = startDir.clone().applyQuaternion(q).normalize();
      camDist = lerp(D_START, dFollow, u);
      planeAlpha = clamp((t - (T_INTRO - 0.35)) / 0.35, 0, 1);
    } else if (t < T_INTRO + T_FLIGHT) {
      u = (t - T_INTRO) / T_FLIGHT;
      f = easeSine(u);
      const target = curve.getPointAt(Math.min(1, f + LEAD)).clone().normalize();
      camDir.lerp(target, 1 - Math.exp(-dt * 3.5)).normalize();
      camDist = dFollow * (1 + 0.1 * Math.sin(Math.PI * u));
      planeAlpha = 1;
    } else {
      u = (t - T_INTRO - T_FLIGHT) / T_ARRIVE;
      f = 1;
      camDir.lerp(midDir, 1 - Math.exp(-dt * 1.7)).normalize();
      camDist = lerp(camDist, dOverview, 1 - Math.exp(-dt * 1.7));
      planeAlpha = 1 - clamp(u / 0.12, 0, 1);
    }

    placeCamera();
    setProgress(f);

    // plane
    const pp = curve.getPointAt(f);
    plane.position.copy(pp);
    const sz = 0.075 * camera.position.distanceTo(pp) * (H / 520);
    plane.scale.set(sz, sz, 1);
    plane.material.opacity = planeAlpha;
    plane.visible = planeAlpha > 0.01;
    const s0 = toScreen(curve.getPointAt(Math.max(0, f - 0.006)));
    const s1 = toScreen(curve.getPointAt(Math.min(1, f + 0.006)));
    const ang = Math.atan2(-(s1.y - s0.y), s1.x - s0.x);
    if (heading === null) heading = ang;
    let diff = ang - heading;
    diff = Math.atan2(Math.sin(diff), Math.cos(diff));
    heading += diff * (1 - Math.exp(-dt * 10));
    plane.material.rotation = heading;

    // pins
    [pinA, pinB].forEach(p => {
      const s = toScreen(p.pos);
      p.el.style.transform = "translate(" + s.x.toFixed(1) + "px," + s.y.toFixed(1) + "px)";
      const facing = p.pos.clone().normalize().dot(camDir) > 1 / camDist + 0.03;
      p.el.classList.toggle("on", facing && t > 0.4);
    });
    pinA.el.classList.toggle("show", t > 0.9 && f < 0.08);
    pinA.el.classList.toggle("ringon", t > 0.9 && f < 0.02);
    pinB.el.classList.toggle("show", f > 0.55);
    pinB.el.classList.toggle("ringon", f > 0.55);
    pinB.el.classList.toggle("pulse", f >= 0.999);

    // distance counter
    kmEl.textContent = "+" + Math.round(distKm * f).toLocaleString("en-US") + " km";
    kmEl.classList.toggle("on", t > T_INTRO - 0.2);
  }

  function tick(now) {
    if (!running) return;
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    clock += dt;
    step(Math.min(clock, T_TOTAL), dt);
    renderer.render(scene, camera);
    if (clock < T_TOTAL) {
      raf = requestAnimationFrame(tick);
    } else {
      running = false;
      replayBtn.classList.add("on");
    }
  }

  function start() {
    cancelAnimationFrame(raf);
    clock = 0; camDir = startDir.clone(); camDist = D_START; heading = null;
    pinA.el.className = "pin"; pinB.el.className = "pin";
    replayBtn.classList.remove("on");
    running = true;
    last = performance.now();
    raf = requestAnimationFrame(tick);
  }
  replayBtn.addEventListener("click", start);

  window.addEventListener("resize", () => {
    W = stage.clientWidth || W;
    renderer.setSize(W, H);
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    if (ready && !running) { step(T_TOTAL, 0.016); renderer.render(scene, camera); }
  });

  // ---------- textures ----------
  function loadTexture(urls) {
    return new Promise(res => {
      const loader = new THREE.TextureLoader();
      loader.setCrossOrigin("anonymous");
      let i = 0;
      const next = () => {
        if (i >= urls.length) return res(null);
        loader.load(urls[i++], tx => res(tx), undefined, next);
      };
      next();
    });
  }

  async function makeFallbackTexture() {
    const c = document.createElement("canvas");
    c.width = 2048; c.height = 1024;
    const g = c.getContext("2d");
    const grd = g.createLinearGradient(0, 0, 0, 1024);
    grd.addColorStop(0, "#0d3b73"); grd.addColorStop(.5, "#1668b3"); grd.addColorStop(1, "#0d3b73");
    g.fillStyle = grd; g.fillRect(0, 0, 2048, 1024);
    try {
      const w = await (await fetch("https://cdn.jsdelivr.net/npm/world-atlas@2/land-110m.json")).json();
      const land = topojson.feature(w, w.objects.land);
      const geom = land.geometry || land;
      const polys = geom.type === "MultiPolygon" ? geom.coordinates : [geom.coordinates];
      const X = lon => (lon + 180) / 360 * 2048, Y = lat => (90 - lat) / 180 * 1024;
      g.fillStyle = "#4f7d4e"; g.strokeStyle = "rgba(255,255,255,.35)"; g.lineWidth = 1.2;
      polys.forEach(poly => {
        g.beginPath();
        poly.forEach(ring => ring.forEach((q, i) => i ? g.lineTo(X(q[0]), Y(q[1])) : g.moveTo(X(q[0]), Y(q[1]))));
        g.fill("evenodd"); g.stroke();
      });
    } catch (e) { /* plain ocean globe if offline */ }
    return new THREE.CanvasTexture(c);
  }

  (async function init() {
    let tex = await loadTexture([
      "https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg",
      "https://cdn.jsdelivr.net/npm/three-globe/example/img/earth-blue-marble.jpg"
    ]);
    if (!tex) tex = await makeFallbackTexture();
    tex.encoding = THREE.sRGBEncoding;
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    globeMat.map = tex;
    globeMat.needsUpdate = true;
    ready = true;
    placeCamera();
    renderer.render(scene, camera);
    document.getElementById("load").classList.add("off");
    start();
  })();
})();
</script>
"""


def show_route_animation(source_city: str, destination_city: str) -> None:
    """Cinematic 3D satellite globe: camera follows a plane flying the route."""
    def point(city: str) -> dict:
        lon, lat = CITY_COORDS[city]
        return {"name": city, "lat": lat, "lon": lon}

    page = (
        ROUTE_ANIMATION_HTML
        .replace("__SRC__", json.dumps(point(source_city)))
        .replace("__DST__", json.dumps(point(destination_city)))
        .replace("__FLAG__", COUNTRY_FLAG)
        .replace("__COUNTRY__", COUNTRY_NAME)
        .replace("__NONCE__", str(time.time_ns()))
    )
    components.html(page, height=540)


def build_model_input(
    model,
    *,
    airline: str,
    source_city: str,
    destination_city: str,
    ticket_class: str,
    stops: str,
    duration_minutes: int,
    journey_day: int,
    journey_month: int,
    departure_period: str,
    arrival_period: str,
) -> pd.DataFrame:
    """Transforms user inputs into exact feature space expected by Random Forest model."""
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        raise ValueError("Loaded model object lacks 'feature_names_in_' metadata attribute.")

    values = dict.fromkeys(feature_names, 0.0)
    route = f"{source_city}-{destination_city}"

    # Core Numerical Features
    numeric_values = {
        "class": int(ticket_class == "Business"),
        "day": journey_day,
        "month": journey_month,
        "dep_hour": PERIOD_HOURS[departure_period],
        "arr_hour": PERIOD_HOURS[arrival_period],
        "airline_index": AIRLINE_INDEX[airline],
        "route_index": ROUTE_INDEX[route],
        "duration_in_min": duration_minutes,
        "stops": {"Non-stop": 0, "1 stop": 1, "2+ stops": 2}[stops],
    }
    values.update({k: v for k, v in numeric_values.items() if k in values})

    departure_is_daytime = departure_period in {"Morning", "Evening"}
    arrival_is_daytime = arrival_period in {"Morning", "Evening"}

    if "dep_daytime" in values:
        values["dep_daytime"] = int(departure_is_daytime)
    if "arr_daytime" in values:
        values["arr_daytime"] = int(arrival_is_daytime)

    # Categorical One-Hot Flags
    indicators = {
        f"airline_{airline}": 1,
        f"from_{source_city}": 1,
        f"to_{destination_city}": 1,
        f"route_{route}": 1,
        "class_category_Economy": int(ticket_class == "Economy"),
        "stops_category_Non-stop": int(stops == "Non-stop"),
        "stops_category_Multiple-Stops": int(stops == "2+ stops"),
        "arr_daytime_category_Night Arrival": int(not arrival_is_daytime),
        "dep_daytime_category_Night Departure": int(not departure_is_daytime),
        "month_category_March": int(journey_month == 3),
        f"dep_period_{PERIOD_FEATURES[departure_period]}": 1,
        f"arr_period_{PERIOD_FEATURES[arrival_period]}": 1,
    }

    for f_name, val in indicators.items():
        if f_name in values:
            values[f_name] = val

    return pd.DataFrame([values], columns=feature_names)

# ==============================================================================
# 4. APPLICATION LAYOUT & LOGIC
# ==============================================================================
model = load_trained_model(MODEL_PATH)

st.title("✈️ Flight Price Prediction Dashboard")
st.markdown(
    "Production-grade fare estimator driven by a tuned **Random Forest Regressor** ($R^2 \\approx 0.9884$)."
)
st.markdown("---")

# Sidebar - Application Context & Status
with st.sidebar:
    st.header("⚙️ Model Status")
    if model is not None:
        st.success("✅ Model Loaded & Active")
        st.info(f"Features Expectation: **{len(getattr(model, 'feature_names_in_', []))} attributes**")
    else:
        st.error("❌ Model Disconnected")
        st.warning("Please verify `MODEL_PATH` or place `.pkl` in working directory.")

    st.markdown("---")
    st.header("💡 Operational Tips")
    st.caption(
        "- **Business Class** fares generally scale higher regardless of duration.\n"
        "- Fares spike significantly during **March/Festival** seasons.\n"
        "- Non-stop options provide tighter lower-bound price variance."
    )

if model is None:
    st.error("Cannot proceed: Predictive model artifact is unreachable.")
    st.stop()

# Inputs Form
st.subheader("📋 Enter Itinerary Details")

col_left, col_mid, col_right = st.columns(3, gap="medium")

with col_left:
    airline = st.selectbox(
        "Airline Carrier",
        AIRLINES,
        index=AIRLINES.index("SpiceJet"),
        format_func=lambda x: x.replace("_", " ")
    )
    source_city = st.selectbox("Source (From)", CITIES, index=CITIES.index("Delhi"))
    ticket_class = st.selectbox("Class Category", ["Economy", "Business"])

with col_mid:
    dest_options = [c for c in CITIES if c != source_city]
    destination_city = st.selectbox(
        "Destination (To)",
        dest_options,
        index=dest_options.index("Mumbai") if "Mumbai" in dest_options else 0
    )
    stops = st.selectbox("Flight Stops", ["Non-stop", "1 stop", "2+ stops"])
    duration_minutes = st.number_input(
        "Duration (Minutes)",
        min_value=30,
        max_value=3000,
        value=130,
        step=10,
        help="Example: 2 hours = 120 mins | 5 hours = 300 mins"
    )

with col_right:
    journey_month = st.slider("Journey Month", min_value=1, max_value=12, value=2)
    max_days = DAYS_IN_MONTH[journey_month]
    journey_day = st.slider("Journey Day", min_value=1, max_value=max_days, value=11)

    departure_period = st.selectbox(
        "Departure Time Period",
        list(PERIOD_HOURS.keys()),
        index=list(PERIOD_HOURS.keys()).index("Evening")
    )
    arrival_period = st.selectbox(
        "Arrival Time Period",
        list(PERIOD_HOURS.keys()),
        index=list(PERIOD_HOURS.keys()).index("Night")
    )

st.markdown("---")

# Prediction Execution
if st.button("Predict Flight Price 🚀", type="primary", use_container_width=True):
    try:
        input_vector = build_model_input(
            model,
            airline=airline,
            source_city=source_city,
            destination_city=destination_city,
            ticket_class=ticket_class,
            stops=stops,
            duration_minutes=int(duration_minutes),
            journey_day=journey_day,
            journey_month=journey_month,
            departure_period=departure_period,
            arrival_period=arrival_period
        )

        log_prediction = float(model.predict(input_vector)[0])
        predicted_price = float(np.expm1(max(log_prediction, 0.0)))

        # Display Results
        show_route_animation(source_city, destination_city)

        res_col1, res_col2 = st.columns([1.5, 1], gap="medium")

        with res_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span style="color: #94a3b8; font-size: 0.95rem; font-weight: 500;">ESTIMATED FARE</span>
                    <div class="price-value">₹ {predicted_price:,.2f}</div>
                    <span style="color: #64748b; font-size: 0.85rem;">Calculated via Random Forest Pipeline</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        with res_col2:
            st.info(f"""
            **Itinerary Summary:**
            * **Route:** {source_city} ➔ {destination_city}
            * **Carrier:** {airline.replace('_', ' ')} ({ticket_class})
            * **Duration & Stops:** {duration_minutes} mins | {stops}
            * **Schedule:** Day {journey_day}/Month {journey_month} ({departure_period} Dep)
            """)

        with st.expander("🔍 Inspect Generated Feature Vector"):
            st.dataframe(input_vector)

    except Exception as err:
        st.error(f"Prediction Pipeline Failed: {err}")