// Visual effects used by main.js: textures, wind sway, fires/smoke/embers, dust, fading point cloud, SLAM materials,
// scan pulse, FOV cone, trail, ground cover. Everything is GPU-animated from a time uniform.
import * as THREE from 'three';

export function rng(seed) { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
export const hash2 = (x, y) => { const v = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return v - Math.floor(v); };

export function speckleTexture(seed, size, base, dark, light, n, rmin, rmax) {
  const c = document.createElement('canvas'); c.width = c.height = size;
  const g = c.getContext('2d'), r = rng(seed);
  g.fillStyle = base; g.fillRect(0, 0, size, size);
  for (let i = 0; i < n; i++) {
    const x = r() * size, y = r() * size, rad = rmin + r() * (rmax - rmin);
    g.fillStyle = r() < 0.5 ? dark : light; g.globalAlpha = 0.1 + 0.22 * r();
    for (const ox of [0, -size, size]) for (const oy of [0, -size, size]) { g.beginPath(); g.arc(x + ox, y + oy, rad, 0, 6.2832); g.fill(); }
  }
  g.globalAlpha = 1;
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  return t;
}
export function concreteTexture(brick) {
  const t = speckleTexture(5, 256, '#b4b4b4', '#6a6a6a', '#f0f0f0', 1800, 0.6, 2.4), g = t.image.getContext('2d');
  if (brick) {                                         // faint mortar lines (rows of 32 px, staggered joints)
    g.globalAlpha = 0.22; g.strokeStyle = '#444'; g.lineWidth = 1;
    for (let y = 0; y < 256; y += 32) { g.beginPath(); g.moveTo(0, y); g.lineTo(256, y); g.stroke();
      for (let x = ((y / 32) % 2) * 32; x < 256; x += 64) { g.beginPath(); g.moveTo(x, y); g.lineTo(x, y + 32); g.stroke(); } }
    g.globalAlpha = 1; t.needsUpdate = true;
  }
  return t;
}
export function radialTexture(stops, size = 128) {
  const c = document.createElement('canvas'); c.width = c.height = size;
  const g = c.getContext('2d'), gr = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  for (const [o, col] of stops) gr.addColorStop(o, col);
  g.fillStyle = gr; g.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(c);
}
export function labelSprite(text, color = '#59f0ff') {
  const c = document.createElement('canvas'); c.width = 320; c.height = 72;
  const g = c.getContext('2d');
  g.fillStyle = 'rgba(6,10,16,0.78)'; g.strokeStyle = color; g.lineWidth = 3;
  g.beginPath(); g.roundRect(3, 3, 314, 66, 10); g.fill(); g.stroke();
  g.fillStyle = color; g.font = 'bold 30px ui-monospace, Menlo, monospace'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(text, 160, 38);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false, transparent: true, fog: false, toneMapped: false }));
  s.scale.set(4.4, 1.0, 1); s.renderOrder = 20;
  return s;
}

// ---- wind sway injected into standard materials (instanced). Visual only: amplitude is kept small vs the sensed geometry.
export const swayU = { uTime: { value: 0 }, uWind: { value: new THREE.Vector2(0.8, 0.45) } };
export function swayPatch(mat, amp, mode = 'blob') {
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uTime = swayU.uTime; sh.uniforms.uWind = swayU.uWind;
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nuniform float uTime; uniform vec2 uWind;')
      .replace('#include <begin_vertex>', `#include <begin_vertex>
      #ifdef USE_INSTANCING
        vec2 ip = instanceMatrix[3].xy;
        float gust = 0.55 + 0.45 * sin(dot(ip, uWind) * 0.09 - uTime * 0.8);
        float sw = sin(uTime * 1.6 + ip.x * 0.37 + ip.y * 0.29) * 0.7 + sin(uTime * 3.3 + ip.y * 0.5) * 0.3;
        float k = ${mode === 'grass' ? 'clamp(position.z, 0.0, 1.0) * clamp(position.z, 0.0, 1.0)' : 'clamp(position.z * 0.5 + 0.5, 0.0, 1.0)'};
        transformed.xy += uWind * (${amp.toFixed(3)} * gust * sw * k);
      #endif`);
  };
  return mat;
}

// ---- fires: flames + embers (additive) and smoke (alpha), animated entirely in the vertex shader
const FIRE_VS = `attribute vec4 aSeed; attribute float aInt; uniform float uTime, uScale; uniform vec2 uWind;
  varying float vA; varying vec3 vC; varying float vK;
  void main(){
    float kind = aSeed.y;
    float life = kind < 0.5 ? 0.85 : (kind < 1.5 ? 4.5 : 1.7);
    float age = fract(uTime / life + aSeed.x);
    vec3 p = position; float sz; vec3 col; float a;
    if (kind < 0.5) {                         // flame
      float s = aInt;
      p.xy += (aSeed.zw - 0.5) * 0.9 * s * (1.0 - age * 0.7) + vec2(sin(uTime*7.0+aSeed.x*30.0), cos(uTime*6.0+aSeed.z*30.0)) * 0.06;
      p.z += age * 1.7 * s;
      sz = (0.95 - 0.7 * age) * s * 1.5; col = mix(vec3(1.0, 0.85, 0.35), vec3(1.0, 0.25, 0.04), age) * 1.6;
      a = smoothstep(0.0, 0.1, age) * (1.0 - age) * 0.8;
    } else if (kind < 1.5) {                  // smoke
      p.xy += (aSeed.zw - 0.5) * 1.2 + uWind * age * 9.0 * (0.6 + aSeed.z);
      p.z += 0.8 + age * (6.0 + 5.0 * aInt);
      sz = (1.2 + age * 5.5) * aInt; col = mix(vec3(0.20, 0.19, 0.19), vec3(0.42, 0.41, 0.40), age);
      a = smoothstep(0.0, 0.12, age) * (1.0 - age) * 0.38;
    } else {                                  // ember
      p.xy += (aSeed.zw - 0.5) * 1.0 + uWind * age * 4.0 + vec2(sin(uTime*3.0+aSeed.x*40.0), cos(uTime*2.5+aSeed.z*40.0)) * 0.3 * age;
      p.z += 0.3 + age * (3.5 + 3.0 * aSeed.z);
      sz = 0.09 * aInt; col = vec3(1.0, 0.55, 0.15) * 3.0; a = (1.0 - age) * (0.6 + 0.4 * sin(uTime*20.0 + aSeed.x*50.0));
    }
    vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv;
    gl_PointSize = max(sz * uScale / max(-mv.z, 0.5), 1.0);
    vA = a * exp(-max(-mv.z - 25.0, 0.0) * 0.012); vC = col; vK = kind;
  }`;
const FIRE_FS = `varying float vA; varying vec3 vC; varying float vK;
  void main(){ vec2 d = gl_PointCoord - 0.5; float r = length(d) * 2.0; if (r > 1.0) discard;
    float soft = vK > 0.5 && vK < 1.5 ? (1.0 - r) * (1.0 - r) : (1.0 - r * r);
    gl_FragColor = vec4(vC, vA * soft); }`;
export function makeFires(fires) {
  const group = new THREE.Group(), n = fires.length;
  const U = { uTime: { value: 0 }, uScale: { value: 600 }, uWind: swayU.uWind };
  const build = (kinds, blending, depthWrite) => {
    const pos = [], seed = [], inten = [];
    const r = rng(11);
    fires.forEach((f) => { for (const [kind, cnt] of kinds) for (let i = 0; i < cnt; i++) { pos.push(f[0], f[1], f[2]); seed.push(r(), kind, r(), r()); inten.push(f[3]); } });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('aSeed', new THREE.Float32BufferAttribute(seed, 4)); g.setAttribute('aInt', new THREE.Float32BufferAttribute(inten, 1));
    const m = new THREE.Points(g, new THREE.ShaderMaterial({ uniforms: U, vertexShader: FIRE_VS, fragmentShader: FIRE_FS, transparent: true, depthWrite, blending, fog: false }));
    m.frustumCulled = false; return m;
  };
  const glow = build([[0, 16], [2, 7]], THREE.AdditiveBlending, false); glow.renderOrder = 6;
  const smoke = build([[1, 14]], THREE.NormalBlending, false); smoke.renderOrder = 5;
  group.add(smoke, glow);
  const lights = [];
  for (let i = 0; i < 4; i++) { const l = new THREE.PointLight(0xff7a30, 0, 20, 2); lights.push(l); group.add(l); }
  let seedT = 0;
  return {
    group, U, lights,
    update(t, camera, H, focus) {
      U.uTime.value = t; U.uScale.value = H * 0.5 / Math.tan(camera.fov * Math.PI / 360);
      const near = fires.map((f, i) => [Math.hypot(f[0] - focus.x, f[1] - focus.y, f[2] - focus.z), i]).sort((a, b) => a[0] - b[0]);
      lights.forEach((l, k) => {
        const c = near[k]; if (!c || c[0] > 45) { l.intensity = 0; return; }
        const f = fires[c[1]]; l.position.set(f[0], f[1], f[2] + 1.2);
        l.intensity = 70 * f[3] * (0.8 + 0.2 * Math.sin(t * 17 + c[1] * 3) * Math.sin(t * 5.3 + c[1])) * (1 - c[0] / 60);
      });
    },
  };
}

// ---- dust motes drifting in the sun, wrapped around a moving centre
export function makeDust(n = 700, box = 34) {
  const pos = new Float32Array(n * 3), r = rng(8), ph = new Float32Array(n);
  for (let i = 0; i < n; i++) { pos.set([(r() - .5) * box, (r() - .5) * box, r() * 12], i * 3); ph[i] = r() * 6.28; }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('aPh', new THREE.BufferAttribute(ph, 1));
  const U = { uTime: { value: 0 }, uC: { value: new THREE.Vector3() }, uBox: { value: box }, uStr: { value: 1 }, uScale: { value: 600 } };
  const m = new THREE.Points(g, new THREE.ShaderMaterial({ uniforms: U, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    vertexShader: `attribute float aPh; uniform float uTime, uBox, uScale; uniform vec3 uC; varying float vA;
      void main(){ vec3 p = position + vec3(0.25 * uTime, 0.12 * uTime + sin(uTime * 0.3 + aPh) * 0.6, sin(uTime * 0.2 + aPh) * 0.4);
        p.xy = uC.xy + mod(p.xy - uC.xy + uBox * 0.5, uBox) - uBox * 0.5; p.z = uC.z - 3.0 + mod(p.z - uC.z + 3.0, 12.0);
        vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv; gl_PointSize = 0.035 * uScale / max(-mv.z, 0.5);
        vA = (0.35 + 0.65 * pow(0.5 + 0.5 * sin(uTime * 1.3 + aPh * 5.0), 3.0)) * exp(-max(-mv.z - 18.0, 0.0) * 0.06); }`,
    fragmentShader: `uniform float uStr; varying float vA; void main(){ float r = length(gl_PointCoord - 0.5) * 2.0; if (r > 1.0) discard; gl_FragColor = vec4(vec3(1.0, 0.92, 0.75) * 1.4, vA * (1.0 - r) * 0.55 * uStr); }`,
  }));
  m.frustumCulled = false; m.renderOrder = 7;
  return { points: m, U };
}

// ---- SLAM surface materials (per-vertex birth time -> fade-in + flash) and the fading point cloud
export const slamU = { uNow: { value: 0 }, uFlash: { value: 1 } };
const SLAM_VS = `attribute vec3 color; attribute float aBorn; uniform float uNow; varying vec3 vC; varying float vAge;
  void main(){ vC = color; vAge = uNow - aBorn; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
export const slamLineMat = new THREE.ShaderMaterial({ uniforms: slamU, transparent: true, depthWrite: false, fog: false, vertexShader: SLAM_VS,
  fragmentShader: `uniform float uFlash; varying vec3 vC; varying float vAge;
    void main(){ float fade = smoothstep(0.0, 0.45, vAge); float fl = vAge < 0.0 ? 0.0 : exp(-vAge * 1.8) * 3.0 * uFlash;
      gl_FragColor = vec4(vC * 1.5 + vec3(0.35, 0.8, 1.0) * fl, 0.88 * fade); }` });
export const slamFillMat = new THREE.ShaderMaterial({ uniforms: slamU, transparent: true, depthWrite: false, side: THREE.DoubleSide, fog: false, vertexShader: SLAM_VS,
  fragmentShader: `uniform float uFlash; varying vec3 vC; varying float vAge;
    void main(){ float fade = smoothstep(0.0, 0.45, vAge); float fl = vAge < 0.0 ? 0.0 : exp(-vAge * 1.8) * 1.6 * uFlash;
      gl_FragColor = vec4(vC * 1.1 + vec3(0.3, 0.7, 1.0) * fl, (0.17 + 0.25 * fl) * fade); }` });

export function makeCloud(N = 9000) {
  const g = new THREE.BufferGeometry(), pos = new Float32Array(N * 3), col = new Float32Array(N * 3), born = new Float32Array(N).fill(-1e9);
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3)); g.setAttribute('aBorn', new THREE.BufferAttribute(born, 1));
  const U = { uNow: slamU.uNow, uSize: { value: 2.4 }, uLife: { value: 1.3 } };
  const m = new THREE.Points(g, new THREE.ShaderMaterial({ uniforms: U, transparent: true, depthWrite: false, depthTest: false, fog: false,
    vertexShader: `attribute vec3 color; attribute float aBorn; uniform float uNow, uSize, uLife; varying vec3 vC; varying float vA;
      void main(){ float a = (uNow - aBorn) / uLife; vA = a < 0.0 || a > 1.0 ? 0.0 : (1.0 - a) * (1.0 - a); vC = color;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_PointSize = uSize * (1.0 + 1.5 * (1.0 - clamp(a, 0.0, 1.0))); }`,
    fragmentShader: `varying vec3 vC; varying float vA; void main(){ float r = length(gl_PointCoord - 0.5) * 2.0; if (r > 1.0 || vA < 0.01) discard; gl_FragColor = vec4(vC * 2.2, vA * (1.0 - r * 0.5)); }` }));
  m.frustumCulled = false; m.renderOrder = 9;
  let head = 0;
  return {
    points: m, U,
    push(x, y, z, r, gc, b, t) { pos[head * 3] = x; pos[head * 3 + 1] = y; pos[head * 3 + 2] = z; col[head * 3] = r; col[head * 3 + 1] = gc; col[head * 3 + 2] = b; born[head] = t; head = (head + 1) % N; },
    commit() { g.attributes.position.needsUpdate = g.attributes.color.needsUpdate = g.attributes.aBorn.needsUpdate = true; },
    clear() { born.fill(-1e9); head = 0; g.attributes.aBorn.needsUpdate = true; },
  };
}

// ---- expanding scan pulse + FOV cone
export function makePulse() {
  const m = new THREE.Mesh(new THREE.SphereGeometry(1, 40, 20), new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false, side: THREE.FrontSide,
    uniforms: { uP: { value: 0 } },
    vertexShader: 'varying vec3 vN, vV; void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = -mv.xyz; gl_Position = projectionMatrix * mv; }',
    fragmentShader: `varying vec3 vN, vV; uniform float uP; void main(){ float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.2);
      gl_FragColor = vec4(vec3(0.25, 0.85, 1.0) * f * (1.0 - uP) * (1.0 - uP) * 1.4, 1.0); }` }));
  m.frustumCulled = false; m.renderOrder = 8; return m;
}
export function makeFov(hfov, vfov, pitchDeg, len) {
  const th = Math.tan(hfov * Math.PI / 360), tv = Math.tan(vfov * Math.PI / 360), p = pitchDeg * Math.PI / 180, c = [];
  for (const [u, v] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
    let x = 1, y = -u * th, z = -v * tv; const l = Math.hypot(x, y, z); x /= l; y /= l; z /= l;
    c.push(new THREE.Vector3(x * Math.cos(p) + z * Math.sin(p), y, -x * Math.sin(p) + z * Math.cos(p)).multiplyScalar(len));
  }
  const o = new THREE.Vector3(), lp = [];
  for (let i = 0; i < 4; i++) { lp.push(o, c[i], c[i], c[(i + 1) % 4]); }
  const g = new THREE.BufferGeometry().setFromPoints(lp);
  const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x4cd8ff, transparent: true, opacity: 0.55, fog: false, depthWrite: false }));
  const tri = []; for (let i = 0; i < 4; i++) tri.push(o, c[i], c[(i + 1) % 4]);
  const fill = new THREE.Mesh(new THREE.BufferGeometry().setFromPoints(tri), new THREE.MeshBasicMaterial({ color: 0x4cd8ff, transparent: true, opacity: 0.05, side: THREE.DoubleSide, depthWrite: false, fog: false, blending: THREE.AdditiveBlending }));
  const grp = new THREE.Group(); grp.add(lines, fill); return grp;
}

// ---- fading trail
export function makeTrail(N = 420) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3)); g.setAttribute('aT', new THREE.BufferAttribute(new Float32Array(N), 1));
  g.setDrawRange(0, 0);
  const l = new THREE.Line(g, new THREE.ShaderMaterial({ transparent: true, depthWrite: false, fog: false,
    vertexShader: 'attribute float aT; varying float vT; void main(){ vT = aT; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'varying float vT; void main(){ gl_FragColor = vec4(vec3(1.0, 0.62, 0.22) * (0.6 + 1.8 * vT), pow(vT, 1.6) * 0.9); }' }));
  l.frustumCulled = false; l.renderOrder = 4;
  return { line: l, N };
}

// ---- ground cover (grass tufts, bushes, rocks) re-centred around the drone from hashed cells
export function crossQuads(n = 3) {
  const P = [], UV = [], NR = [], I = [];
  for (let k = 0; k < n; k++) {
    const a = k * Math.PI / n, c = Math.cos(a) * 0.5, s = Math.sin(a) * 0.5, b = P.length / 3;
    P.push(-c, -s, 0, c, s, 0, c, s, 1, -c, -s, 1); UV.push(0, 0, 1, 0, 1, 1, 0, 1); NR.push(0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1);
    I.push(b, b + 1, b + 2, b, b + 2, b + 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3)); g.setAttribute('uv', new THREE.Float32BufferAttribute(UV, 2)); g.setAttribute('normal', new THREE.Float32BufferAttribute(NR, 3)); g.setIndex(I);
  return g;
}
export function grassTexture() {
  const c = document.createElement('canvas'); c.width = 128; c.height = 128;
  const g = c.getContext('2d'), r = rng(3);
  for (let i = 0; i < 22; i++) {
    const x = 8 + r() * 112, h = 50 + r() * 70, lean = (r() - 0.5) * 26;
    g.fillStyle = `hsl(${88 + r() * 30},${40 + r() * 20}%,${28 + r() * 26}%)`;
    g.beginPath(); g.moveTo(x - 3, 128); g.quadraticCurveTo(x - 1 + lean * 0.3, 128 - h * 0.6, x + lean, 128 - h); g.quadraticCurveTo(x + 1 + lean * 0.3, 128 - h * 0.6, x + 3, 128); g.fill();
  }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
