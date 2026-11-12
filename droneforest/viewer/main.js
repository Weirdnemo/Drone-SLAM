// DroneForest viewer v3: replays a flight (rollout.json / mission.json) through a foggy forest and a collapsed-city
// disaster zone, with a simulated depth camera, a chunked voxel SLAM map + surface mesh, survivor detection,
// post-processing (bloom / god rays / grade), a cinematic director camera and WebM recording.
// World is Z-up and uses the SAME analytic terrain / tree / box primitives as the training env.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import GUI from './vendor/lil-gui.esm.js';
import { patchHeightFog, makePost } from './post.js';
import * as FX from './fx.js';

patchHeightFog(2.2);

// ------------------------------------------------------------------ constants / state
let VOX = 0.4;                    // SLAM voxel size (m); 0.5 for long missions
const OCC_HITS = 2, DEPTH_W = 48, DEPTH_H = 28, DEPTH_HFOV = 95, DEPTH_VFOV = 60, CAM_PITCH = -14;
const N_AZ = 16, N_EL = 5, SENSE_DT = 0.05, CH = 16, ARENA = { x: 26, y: 18 };
const FOG_FOREST = new THREE.Color(0xa3bccb), FOG_RUIN = new THREE.Color(0xa59a90);
const q0 = new URLSearchParams(location.search);

const S = {
  pos: '', play: true, speed: 1, time: 0, episode: 0, loop: true, follow: true, cine: false,
  noise: 0.0, pointSize: 0.08, showCloud: true, showScan: true, pulse: true, fovCone: true, flash: true,
  voxels: 0, showMesh: true, useEst: false, fogOfWar: true, showPanels: true, mmTruth: true,
  coverage: '-', precision: '-', explored: '-', drift: '-', found: '-',
  fog: 0.027, shafts: true, g: 0.6, shadows: true, droneScale: 2.6,
  post: !q0.has('nopost'), bloom: 0.5, godrays: 0.55, vignette: 0.45, grain: 0.03, dust: true, spot: true,
};

// ------------------------------------------------------------------ three.js setup
const renderer = new THREE.WebGLRenderer({ antialias: !S.post, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(FOG_FOREST.getHex(), S.fog);
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 600);
camera.up.set(0, 0, 1);
camera.position.set(-30, -8, 8);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.maxPolarAngle = Math.PI * 0.49;

const skyU = { top: { value: new THREE.Color(0x4a78a8) }, mid: { value: new THREE.Color(0x86a8c2) }, hor: { value: new THREE.Color(FOG_FOREST) } };
const sky = new THREE.Mesh(new THREE.SphereGeometry(450, 32, 16), new THREE.ShaderMaterial({
  side: THREE.BackSide, depthWrite: false, fog: false, uniforms: skyU,
  vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
  fragmentShader: `varying vec3 vP; uniform vec3 top, mid, hor;
    void main(){ float h = clamp(normalize(vP).z, 0.0, 1.0);
      vec3 c = mix(hor, mid, smoothstep(0.0, 0.18, h)); c = mix(c, top, smoothstep(0.15, 0.9, h)); gl_FragColor = vec4(c, 1.0); }` }));
scene.add(sky);

const hemi = new THREE.HemisphereLight(0xc4dbee, 0x46603a, 1.5); scene.add(hemi);
const sunDir = new THREE.Vector3(-0.55, -0.35, 0.75).normalize();
const sun = new THREE.DirectionalLight(0xfff0d0, 3.4);
sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -38, right: 38, top: 38, bottom: -38, near: 1, far: 220 });
sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.05;
scene.add(sun, sun.target);
const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: FX.radialTexture([[0, 'rgba(255,244,214,1)'], [0.18, 'rgba(255,230,180,0.55)'], [1, 'rgba(255,220,160,0)']]),
  blending: THREE.AdditiveBlending, depthWrite: false, fog: false, transparent: true }));
glow.scale.set(150, 150, 1); scene.add(glow);

const beams = [];
const beamMat = (phase) => new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false, side: THREE.DoubleSide,
  uniforms: { k: { value: 0.1 }, t: { value: 0 }, ph: { value: phase } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
  fragmentShader: `varying vec2 vUv; uniform float k, t, ph;
    void main(){ float w = 1.0 - abs(vUv.x*2.0 - 1.0); w = w*w*(3.0-2.0*w);
      float l = smoothstep(0.0, 0.25, vUv.y) * (1.0 - smoothstep(0.55, 1.0, vUv.y)); float fl = 0.65 + 0.35*sin(t*0.5 + ph);
      gl_FragColor = vec4(vec3(1.0, 0.92, 0.7) * w * l * fl * k, 1.0); }` });
{ const r = FX.rng(7);
  for (let i = 0; i < 16; i++) { const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), beamMat(r() * 6.28)); m.userData = { ox: (r() - 0.5) * 60, oy: (r() - 0.5) * 60, w: 1.4 + r() * 2.8 }; m.frustumCulled = false; m.renderOrder = 5; beams.push(m); scene.add(m); } }

const post = S.post ? makePost(renderer, scene, camera) : null;

// ------------------------------------------------------------------ shared geometry / materials
const hAt = (T, x, y) => { let h = 0; for (let i = 0; i < T.length; i++) { const p = T[i]; h += p[0] * Math.sin(p[1] * x + p[2] * y + p[3]); } return h; };
const leafTex = FX.speckleTexture(21, 256, '#a6a6a6', '#585858', '#e8e8e8', 2200, 1, 3.5);
const grassTex = FX.speckleTexture(33, 512, '#b0b0b0', '#666666', '#ececec', 6500, 1, 4);
const brickTex = FX.concreteTexture(true), concTex = FX.concreteTexture(false);

const canopyGeo = (() => {
  const g = new THREE.IcosahedronGeometry(1, 2), p = g.attributes.position, cache = new Map();
  const hash = (x, y, z) => { const v = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453; return v - Math.floor(v); };
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i), key = x.toFixed(3) + y.toFixed(3) + z.toFixed(3);
    if (!cache.has(key)) cache.set(key, 1 + Math.sin(x * 3.1 + y * 2.3) * Math.cos(z * 3.7 + x * 1.9) * 0.08 + (hash(x, y, z) - 0.5) * 0.10);
    const sc = cache.get(key); p.setXYZ(i, x * sc, y * sc, z * sc);
  }
  const nrm = g.attributes.normal;
  for (let i = 0; i < p.count; i++) { const x = p.getX(i), y = p.getY(i), z = p.getZ(i), l = Math.hypot(x, y, z); nrm.setXYZ(i, x / l, y / l, z / l); }
  return g;
})();
const trunkGeo = (() => { const g = new THREE.CylinderGeometry(0.65, 1, 1, 8, 1); g.rotateX(Math.PI / 2); g.translate(0, 0, 0.5); return g; })();
const coneGeo = (() => { const g = new THREE.ConeGeometry(1, 1, 9, 1); g.rotateX(Math.PI / 2); g.translate(0, 0, 0.5); return g; })();
const rockGeo = new THREE.ConeGeometry(1, 1, 5).rotateX(Math.PI / 2).translate(0, 0, 0.5);
const grassGeo = FX.crossQuads(3);
const leafMat = FX.swayPatch(new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.92, map: leafTex }), 0.07);
const conifMat = FX.swayPatch(new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, map: leafTex, flatShading: false }), 0.05);
const barkMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1 });
const grassMat = FX.swayPatch(new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, map: FX.grassTexture(), alphaTest: 0.45, side: THREE.DoubleSide }), 0.22, 'grass');
const rockMat = new THREE.MeshStandardMaterial({ color: 0x4a4d48, roughness: 1, flatShading: true });
const boxMats = [
  new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, map: brickTex }),                       // 0 wall
  new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, map: concTex, flatShading: true }),        // 1 rubble
  new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9, map: concTex }),                          // 2 slab
  [0x9a2b2b, 0x2b4d9a, 0xcfcfcf, 0x2d2f33, 0xb79a2b, 0x3a6a4a].map((c) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.5, metalness: 0.45 })), // 3 vehicles
  new THREE.MeshStandardMaterial({ color: 0x5a4330, roughness: 1 }),                                          // 4 log
  new THREE.MeshStandardMaterial({ color: 0x35373a, roughness: 0.8 }),                                        // 5 pole
];
const glassMat = new THREE.MeshStandardMaterial({ color: 0x10161c, roughness: 0.15, metalness: 0.6 });
const wallTints = [0xd8d0c0, 0xc9c1b2, 0xb7a995, 0xa89a88, 0xcfc6b8], rubbleTints = [0x9a9288, 0x7c766d, 0x8a7f72, 0x6f6a63];
const shadowTex = FX.radialTexture([[0, 'rgba(0,0,0,0.9)'], [0.55, 'rgba(0,0,0,0.45)'], [1, 'rgba(0,0,0,0)']]);
const pinTex = FX.radialTexture([[0, 'rgba(255,255,255,1)'], [0.4, 'rgba(255,255,255,0.4)'], [1, 'rgba(255,255,255,0)']]);

// ------------------------------------------------------------------ world
let world = null, DATA = null, DT = 0.02;
let observed = new Uint8Array(1), obsMapped = new Uint8Array(1), exploredCells = 0, fowDirty = true;

function makeGrid(items, xi = 0, yi = 1, cell = 8) {
  const m = new Map();
  items.forEach((it) => { const k = Math.floor(it[xi] / cell) * 4096 + Math.floor(it[yi] / cell); (m.get(k) || m.set(k, []).get(k)).push(it); });
  return { m, cell, query(px, py, R, out = []) {
    const c = this.cell, x0 = Math.floor((px - R) / c), x1 = Math.floor((px + R) / c), y0 = Math.floor((py - R) / c), y1 = Math.floor((py + R) / c);
    for (let a = x0; a <= x1; a++) for (let b = y0; b <= y1; b++) { const l = this.m.get(a * 4096 + b); if (l) for (const it of l) out.push(it); }
    return out; } };
}
const nearTrees = (px, py, R) => world.gT.query(px, py, R + 4);
const nearBoxes = (px, py, R) => world.gB.query(px, py, R + 9);
const pathY = (x) => { const P = world.path; if (!P) return 0; const f = Math.min(Math.max(x / 2, 0), P.length - 1.001), i = Math.floor(f); return P[i][1] + (P[i + 1][1] - P[i][1]) * (f - i); };
function disW(x) {                                                         // 0 = forest, 1 = disaster zone (smooth across 10 m)
  const w = world; if (!w.biomes.length) return w.boxes.length ? 1 : 0;
  let s = 0; for (const b of w.biomes) if (b.kind === 'disaster') s += THREE.MathUtils.smoothstep(x, b.x0 - 5, b.x0 + 5) * (1 - THREE.MathUtils.smoothstep(x, b.x1 - 5, b.x1 + 5));
  return Math.min(s, 1);
}

const TSZ = 260;
function buildTerrain() {
  const g = new THREE.PlaneGeometry(TSZ, TSZ, TSZ, TSZ), n = g.attributes.position.count;
  const lx = new Float32Array(n), ly = new Float32Array(n);
  for (let i = 0; i < n; i++) { lx[i] = g.attributes.position.getX(i); ly[i] = g.attributes.position.getY(i); }
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, map: grassTex }));
  mesh.receiveShadow = true; mesh.frustumCulled = false;
  return { mesh, g, lx, ly, cx: 1e9, cy: 1e9, baseCol: new Float32Array(n * 3) };
}
const cA = new THREE.Color(0x4c7230), cB = new THREE.Color(0x6e8438), cD = new THREE.Color(0x2c4a20), cR = new THREE.Color(0x55534a);
const rA = new THREE.Color(0x7a6d5d), rB = new THREE.Color(0x5c554b), rAsh = new THREE.Color(0x3b3935), rRoad = new THREE.Color(0x26282c);
function rebuildTerrain(cx, cy) {
  const t = world.terr, g = t.g, pos = g.attributes.position, uv = g.attributes.uv, col = g.attributes.color, T = world.T, n = pos.count;
  t.cx = cx; t.cy = cy;
  for (let i = 0; i < n; i++) { const x = t.lx[i] + cx, y = t.ly[i] + cy; pos.setXYZ(i, x, y, hAt(T, x, y)); uv.setXY(i, x * 0.5, y * 0.5); }
  g.computeVertexNormals();
  const nrm = g.attributes.normal, c = new THREE.Color();
  const fires = world.fires.filter((f) => Math.abs(f[0] - cx) < 150 && Math.abs(f[1] - cy) < 150);
  for (let i = 0; i < n; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i), slope = 1 - nrm.getZ(i), dw = disW(x), nz = FX.hash2(Math.round(x), Math.round(y));
    c.copy(cD).lerp(cA, THREE.MathUtils.smoothstep(z, -4, 1)).lerp(cB, THREE.MathUtils.smoothstep(z, 1, 5));
    c.lerp(cR, THREE.MathUtils.smoothstep(slope, 0.12, 0.35));
    if (dw > 0) {
      const dust = new THREE.Color().copy(rB).lerp(rA, nz);
      c.lerp(dust, dw * 0.85);
      if (world.path) { const d = Math.abs(y - pathY(x)); c.lerp(rRoad, dw * (1 - THREE.MathUtils.smoothstep(d, 2.4, 3.6)) * (0.8 + 0.2 * nz)); }
      let sc = 0; for (const f of fires) { const d = Math.hypot(f[0] - x, f[1] - y); if (d < 8) sc = Math.max(sc, 1 - d / 8); }
      c.lerp(rAsh, Math.min(1, sc * 0.9 * dw + 0.12 * nz * dw));
    }
    c.multiplyScalar(0.85 + 0.3 * nz + 0.12 * Math.sin(x * 0.31) * Math.cos(y * 0.27));
    t.baseCol[i * 3] = c.r; t.baseCol[i * 3 + 1] = c.g; t.baseCol[i * 3 + 2] = c.b;
  }
  pos.needsUpdate = uv.needsUpdate = true; fowDirty = true;
}

function disposeWorld() {
  world.group.traverse((o) => { if (o.geometry && ![canopyGeo, trunkGeo, coneGeo, rockGeo, grassGeo].includes(o.geometry)) o.geometry.dispose(); });
  scene.remove(world.group); scene.remove(world.fx.group);
}

function buildWorld(ep) {
  if (world) disposeWorld();
  const isM = ep.scene === 'mission';
  VOX = isM ? 0.5 : 0.4;
  const group = new THREE.Group(), T = ep.terrain;
  const trees = ep.trees.map((t) => (t.length < 7 ? t.concat([0]) : t)), boxes = ep.boxes || [], fires = ep.fires || [], pois = (ep.pois || []).map((p) => ({ ...p }));
  const maxX = isM ? ep.stats.length : 0;
  world = { ep, group, T, trees, boxes, fires, pois, isM, biomes: ep.biomes || [], path: ep.path || null,
    bounds: isM ? { x0: -30, x1: maxX + 30, y0: -75, y1: 75 } : { x0: -ARENA.x - 4, x1: ARENA.x + 4, y0: -ARENA.y - 4, y1: ARENA.y + 4 } };
  world.gT = makeGrid(trees); world.gB = makeGrid(boxes);
  world.terr = buildTerrain(); group.add(world.terr.mesh);
  scene.add(group);

  // ---- trees (real, collidable) + decorative backdrop
  const r = FX.rng(99), decor = [];
  if (isM) {
    for (let i = 0; i < 1300; i++) {
      const x = -30 + r() * (maxX + 90), side = r() < 0.5 ? -1 : 1, y = pathY(Math.min(x, maxX)) + side * (30 + r() * 80), dw = disW(x);
      if (dw > 0.5 && r() < 0.75) continue;
      decor.push([x, y, hAt(T, x, y), 6 + r() * 8, 0.25 + r() * 0.3, 2.0 + r() * 2.2, r() < 0.4 ? 1 : 0]);
    }
  } else {
    for (let i = 0; i < 520; i++) {
      const a = r() * Math.PI * 2, d = 26 + Math.pow(r(), 0.7) * 80, x = Math.cos(a) * d * 1.15, y = Math.sin(a) * d;
      if (Math.abs(x) < 27 && Math.abs(y) < 19) continue;
      decor.push([x, y, hAt(T, x, y), 6 + r() * 7, 0.25 + r() * 0.3, 2.0 + r() * 2.2, r() < 0.3 ? 1 : 0]);
    }
  }
  const all = trees.concat(decor), n = all.length;
  const nConif = all.filter((t) => t[6] === 1).length, nBroad = all.filter((t) => t[6] !== 1).length;
  const trunks = new THREE.InstancedMesh(trunkGeo, barkMat, n), canopy = new THREE.InstancedMesh(canopyGeo, leafMat, nBroad * 5), conif = new THREE.InstancedMesh(coneGeo, conifMat, nConif * 3);
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3(), p3 = new THREE.Vector3(), colr = new THREE.Color(), zax = new THREE.Vector3(0, 0, 1), rr = FX.rng(5);
  let ib = 0, ic = 0;
  all.forEach(([x, y, zg, h, rt, rc, sp], i) => {
    const charred = sp === 2 || (sp !== 1 && isM && disW(x) > 0.6);
    m4.compose(p3.set(x, y, zg - 0.3), q.identity(), sc.set(rt, rt, h + 0.3)); trunks.setMatrixAt(i, m4);
    trunks.setColorAt(i, charred ? colr.setHSL(0.06, 0.1, 0.05 + rr() * 0.04) : colr.setHSL(0.07, 0.3, 0.12 + rr() * 0.07));
    if (sp === 1) {                                              // conifer: three stacked cones, all inside the sensed sphere
      const hue = 0.38 + rr() * 0.05, lit = 0.12 + rr() * 0.06;
      [[0.70, -0.75, 0.95], [0.55, -0.18, 0.85], [0.38, 0.30, 0.70]].forEach(([rad, dz, hh]) => {
        m4.compose(p3.set(x, y, zg + h + dz * rc), q.setFromAxisAngle(zax, rr() * 6), sc.set(rc * rad, rc * rad, rc * hh)); conif.setMatrixAt(ic, m4);
        conif.setColorAt(ic++, colr.setHSL(hue, 0.4, lit + 0.07 * (dz + 0.75)));
      });
      return;
    }
    const hue = charred ? 0.08 : 0.25 + rr() * 0.07, lit = charred ? 0.06 : 0.19 + rr() * 0.08, nbl = sp === 2 ? 2 : 5;
    const blobs = [[0, 0, 0, sp === 2 ? 0.7 : 0.9, sp === 2 ? 0.5 : 0.9]];
    for (let k = 1; k < 5; k++) { const a = rr() * 6.28, d = (0.2 + 0.2 * rr()) * rc, s2 = 0.42 + 0.1 * rr(); blobs.push([Math.cos(a) * d, Math.sin(a) * d, (rr() - 0.5) * 0.5 * rc, s2, s2 * 0.92]); }
    blobs.forEach((b, k) => {
      const keep = k < nbl;
      m4.compose(p3.set(x + b[0], y + b[1], zg + h + b[2]), q.setFromAxisAngle(zax, rr() * 6), keep ? sc.set(rc * b[3], rc * b[3], rc * b[4]) : sc.set(0, 0, 0));
      canopy.setMatrixAt(ib, m4); canopy.setColorAt(ib++, colr.setHSL(hue, charred ? 0.2 : 0.42, lit + 0.09 * (b[2] / rc + 0.5)));
    });
  });
  for (const m of [trunks, canopy, conif]) { m.castShadow = true; m.receiveShadow = true; m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; group.add(m); }

  // ---- boxes: walls / rubble / slabs / vehicles / logs / poles (exact AABBs of the sensing model, world-scaled UVs)
  const mkBox = (hx, hy, hz, mat, cx, cy, cz, tint) => {
    const g = new THREE.BoxGeometry(hx * 2, hy * 2, hz * 2), uv = g.attributes.uv, sz = [[hy, hz], [hy, hz], [hx, hz], [hx, hz], [hx, hy], [hx, hy]];
    for (let f = 0; f < 6; f++) for (let v = 0; v < 4; v++) { const i = f * 4 + v; uv.setXY(i, uv.getX(i) * sz[f][0] * 2 / 3, uv.getY(i) * sz[f][1] * 2 / 3); }
    const m = new THREE.Mesh(g, tint === undefined ? mat : mat.clone()); if (tint !== undefined) m.material.color.setHex(tint);
    m.position.set(cx, cy, cz); m.castShadow = m.receiveShadow = true; group.add(m); return m;
  };
  const rb = FX.rng(31);
  for (const [cx, cy, cz, hx, hy, hz, k] of boxes) {
    if (k === 0) mkBox(hx, hy, hz, boxMats[0], cx, cy, cz, wallTints[Math.floor(rb() * wallTints.length)]);
    else if (k === 1) mkBox(hx, hy, hz, boxMats[1], cx, cy, cz, rubbleTints[Math.floor(rb() * rubbleTints.length)]);
    else if (k === 2) mkBox(hx, hy, hz, boxMats[2], cx, cy, cz, 0xcdc9c0);
    else if (k === 3) {                                          // body (lower 60 %) + darker cabin (upper 40 %, narrower) -> stays inside the AABB
      const pm = boxMats[3][Math.floor(rb() * 6)], bodyH = hz * 1.2, cabH = hz * 0.8, bz = cz - hz + bodyH / 2;
      const body = new THREE.Mesh(new THREE.BoxGeometry(hx * 2, hy * 2, bodyH), pm); body.position.set(cx, cy, bz); body.castShadow = body.receiveShadow = true; group.add(body);
      const along = hx > hy, cab = new THREE.Mesh(new THREE.BoxGeometry(along ? hx * 1.0 : hx * 1.7, along ? hy * 1.7 : hy * 1.0, cabH), glassMat);
      cab.position.set(cx - (along ? hx * 0.15 : 0), cy - (along ? 0 : hy * 0.15), cz - hz + bodyH + cabH / 2); cab.castShadow = true; group.add(cab);
    } else if (k === 4) mkBox(hx, hy, hz, boxMats[4], cx, cy, cz);
    else mkBox(hx, hy, hz, boxMats[5], cx, cy, cz);
  }

  // ---- contact-shadow decals under trees and boxes
  {
    const decals = [...trees.map((t) => [t[0], t[1], t[2], t[5] * 2.4, t[5] * 2.4, 0.45]), ...boxes.filter((b) => b[6] !== 2).map((b) => [b[0], b[1], b[2] - b[5], b[3] * 2 + 1.8, b[4] * 2 + 1.8, 0.55])];
    const im = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, opacity: 0.5, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2, fog: true }), decals.length);
    decals.forEach((d, i) => { m4.compose(p3.set(d[0], d[1], hAt(T, d[0], d[1]) + 0.06), q.identity(), sc.set(d[3], d[4], 1)); im.setMatrixAt(i, m4); });
    im.renderOrder = 1; group.add(im);
  }

  // ---- ground cover (grass tufts / bushes / rocks), regenerated around the drone from hashed cells
  const GC = 6000, BC = 700, RC = 500;
  world.cover = { grass: new THREE.InstancedMesh(grassGeo, grassMat, GC), bush: new THREE.InstancedMesh(canopyGeo, leafMat, BC), rock: new THREE.InstancedMesh(rockGeo, rockMat, RC), cx: 1e9, cy: 1e9 };
  for (const m of Object.values(world.cover)) if (m.isInstancedMesh) { m.frustumCulled = false; m.receiveShadow = true; group.add(m); }
  world.cover.grass.castShadow = false; world.cover.bush.castShadow = world.cover.rock.castShadow = true;
  world.cover.grass.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(GC * 3), 3);
  world.cover.bush.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(BC * 3), 3);

  // ---- survivors / hikers (visible in the world, detected by the drone later) + pins
  if (pois.length) {
    const bodies = new THREE.InstancedMesh(new THREE.CapsuleGeometry(0.17, 0.7, 4, 8).rotateX(Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8 }), pois.length);
    const heads = new THREE.InstancedMesh(new THREE.SphereGeometry(0.15, 10, 8), new THREE.MeshStandardMaterial({ color: 0xd9a37e, roughness: 0.8 }), pois.length);
    pois.forEach((p, i) => {
      const lying = p.type !== 'hiker', gz = hAt(T, p.x, p.y), ang = FX.hash2(i, 7) * 6.28;
      if (lying) { q.setFromAxisAngle(zax, ang); m4.compose(p3.set(p.x, p.y, gz + 0.2), q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2)), sc.set(1, 1, 1)); }
      else m4.compose(p3.set(p.x, p.y, gz + 0.62), q.identity(), sc.set(1, 1, 1));
      bodies.setMatrixAt(i, m4); bodies.setColorAt(i, colr.setHex(lying ? 0xff8a1f : 0xd33a2c));
      const hx = lying ? Math.cos(ang) * 0.62 : 0, hy = lying ? Math.sin(ang) * 0.62 : 0;
      m4.compose(p3.set(p.x + hx, p.y + hy, gz + (lying ? 0.22 : 1.28)), q.identity(), sc.set(1, 1, 1)); heads.setMatrixAt(i, m4);
      const pin = new THREE.Group(), col = p.type === 'hiker' ? '#7dffb0' : '#ffb23c', hex = p.type === 'hiker' ? 0x7dffb0 : 0xffb23c;
      const ring = new THREE.Mesh(new THREE.TorusGeometry(1.1, 0.045, 8, 40), new THREE.MeshBasicMaterial({ color: new THREE.Color(hex).multiplyScalar(2.2), transparent: true, opacity: 0.9, fog: false, toneMapped: false }));
      const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 9, 6).rotateX(Math.PI / 2).translate(0, 0, 4.5), new THREE.MeshBasicMaterial({ color: new THREE.Color(hex).multiplyScalar(1.6), transparent: true, opacity: 0.5, fog: false, depthWrite: false }));
      const lab = FX.labelSprite(`${p.type.toUpperCase()} #${i + 1}`, col); lab.position.set(0, 0, 10.2);
      pin.add(ring, beam, lab); pin.position.set(p.x, p.y, gz + 0.1); pin.visible = false; group.add(pin); p.pin = pin; p.ring = ring; p.tDet = Infinity;
    });
    bodies.castShadow = heads.castShadow = true; bodies.instanceMatrix.needsUpdate = heads.instanceMatrix.needsUpdate = true; bodies.instanceColor.needsUpdate = true;
    group.add(bodies, heads);
  }

  // ---- goal beacon
  const goal = new THREE.Group();
  goal.add(new THREE.Mesh(new THREE.TorusGeometry(1.5, 0.06, 8, 48), new THREE.MeshBasicMaterial({ color: 0x59ffa0, transparent: true, opacity: 0.55, fog: false, depthWrite: false })),
    new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 40, 6).rotateX(Math.PI / 2).translate(0, 0, 20), new THREE.MeshBasicMaterial({ color: 0x59ffa0, transparent: true, opacity: 0.25, fog: false, depthWrite: false })));
  goal.position.set(...ep.goal); group.add(goal); world.goal = goal;

  world.fx = { group: new THREE.Group() };
  world.fires_ = FX.makeFires(fires); world.fx.group.add(world.fires_.group, dust.points); scene.add(world.fx.group);

  buildObservedGrid(); world.gt = buildGT();
  rebuildTerrain(Math.round(ep.frames[0][0] / 20) * 20, Math.round(ep.frames[0][1] / 20) * 20);
  computeDetections(ep);
  trailSet = false;
}

function buildObservedGrid() {
  const b = world.bounds; world.nx = Math.ceil(b.x1 - b.x0); world.ny = Math.ceil(b.y1 - b.y0);
  observed = new Uint8Array(world.nx * world.ny); obsMapped = new Uint8Array(world.nx * world.ny); exploredCells = 0; fowDirty = true;
}
const obsIdx = (x, y) => { const ix = Math.floor(x - world.bounds.x0), iy = Math.floor(y - world.bounds.y0); return ix < 0 || iy < 0 || ix >= world.nx || iy >= world.ny ? -1 : iy * world.nx + ix; };
function markObserved(x, y) { const k = obsIdx(x, y); if (k >= 0 && !observed[k]) { observed[k] = 1; exploredCells++; fowDirty = true; } }

// ------------------------------------------------------------------ ground cover regeneration
function updateCover(px, py) {
  const cv = world.cover; if (Math.hypot(px - cv.cx, py - cv.cy) < 7) return; cv.cx = px; cv.cy = py;
  const T = world.T, m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3(), p3 = new THREE.Vector3(), c = new THREE.Color(), zax = new THREE.Vector3(0, 0, 1);
  let ng = 0, nb = 0, nr = 0; const CELL = 2.0, R = 36, ix0 = Math.floor((px - R) / CELL), ix1 = Math.floor((px + R) / CELL), iy0 = Math.floor((py - R) / CELL), iy1 = Math.floor((py + R) / CELL);
  const blocked = (x, y, pad) => { for (const t of world.gT.query(x, y, 1)) if (Math.hypot(t[0] - x, t[1] - y) < t[4] + pad) return true;
    for (const b of world.gB.query(x, y, 1)) if (Math.abs(x - b[0]) < b[3] + pad && Math.abs(y - b[1]) < b[4] + pad) return true; return false; };
  for (let a = ix0; a <= ix1; a++) for (let b = iy0; b <= iy1; b++) {
    const x0 = a * CELL, y0 = b * CELL; if (Math.hypot(x0 - px, y0 - py) > R) continue;
    const dw = disW(x0), h = (s) => FX.hash2(a * 3.17 + s, b * 5.31 + s * 2.1);
    const nG = Math.round((1 - dw * 0.75) * 5 * (0.5 + h(1)));
    for (let k = 0; k < nG && ng < 6000; k++) { const x = x0 + h(10 + k) * CELL, y = y0 + h(20 + k) * CELL; if (blocked(x, y, 0.3)) continue;
      const s = 0.45 + h(30 + k) * 0.55 + (1 - dw) * 0.2;
      m4.compose(p3.set(x, y, hAt(T, x, y) - 0.02), q.setFromAxisAngle(zax, h(40 + k) * 6.28), sc.set(s * 1.3, s * 1.3, s)); world.cover.grass.setMatrixAt(ng, m4);
      world.cover.grass.setColorAt(ng++, c.setHSL(0.2 + h(50 + k) * 0.08 - dw * 0.07, 0.4 - dw * 0.25, 0.32 + h(60 + k) * 0.18)); }
    if (nb < 700 && h(2) < 0.12 * (1 - dw * 0.6)) { const x = x0 + h(3) * CELL, y = y0 + h(4) * CELL, k = 0.35 + h(5) * 0.7;
      if (!blocked(x, y, 0.8)) { m4.compose(p3.set(x, y, hAt(T, x, y) + k * 0.25), q.setFromAxisAngle(zax, h(6) * 6), sc.set(k * 1.3, k * 1.3, k * 0.7)); world.cover.bush.setMatrixAt(nb, m4); world.cover.bush.setColorAt(nb++, c.setHSL(0.22 + h(7) * 0.08 - dw * 0.1, 0.4, 0.1 + h(8) * 0.07)); } }
    if (nr < 500 && h(9) < 0.08 + dw * 0.22) { const x = x0 + h(11) * CELL, y = y0 + h(12) * CELL, k = 0.2 + h(13) * 0.45;
      if (!blocked(x, y, 0.6)) { m4.compose(p3.set(x, y, hAt(T, x, y) - 0.1), q.setFromAxisAngle(zax, h(14) * 6), sc.set(k, k * 0.8, k * (0.8 + h(15) * 1.6))); world.cover.rock.setMatrixAt(nr++, m4); } }
  }
  const z = new THREE.Matrix4().makeScale(0, 0, 0);
  for (let i = ng; i < 6000; i++) world.cover.grass.setMatrixAt(i, z); for (let i = nb; i < 700; i++) world.cover.bush.setMatrixAt(i, z); for (let i = nr; i < 500; i++) world.cover.rock.setMatrixAt(i, z);
  for (const m of [world.cover.grass, world.cover.bush, world.cover.rock]) { m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; }
}

// ------------------------------------------------------------------ drone model, trail, FX objects
const drone = new THREE.Group();
{
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.16, 0.07), new THREE.MeshStandardMaterial({ color: 0x1b1f27, roughness: 0.5 }));
  const nose = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.12, 0.05), new THREE.MeshStandardMaterial({ color: 0xff7a1a, emissive: 0x331500 })); nose.position.x = 0.13;
  const armM = new THREE.MeshStandardMaterial({ color: 0xe2b45a, roughness: 0.6 });
  for (const s of [1, -1]) { const arm = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.028, 0.02), armM); arm.rotation.z = s * Math.PI / 4; drone.add(arm); }
  drone.add(body, nose); drone.userData.rotors = [];
  for (const [x, y, c] of [[0.22, 0.22, 0xff7a1a], [0.22, -0.22, 0xff7a1a], [-0.22, 0.22, 0xdddddd], [-0.22, -0.22, 0xdddddd]]) {
    const disc = new THREE.Mesh(new THREE.CircleGeometry(0.15, 20), new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.2, side: THREE.DoubleSide, depthWrite: false })); disc.position.set(x, y, 0.035);
    const blade = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.016, 0.006), new THREE.MeshBasicMaterial({ color: 0x20242a, transparent: true, opacity: 0.55 })); blade.position.set(x, y, 0.04);
    drone.add(disc, blade); drone.userData.rotors.push(blade);
  }
  const led = (x, y, z, r, g, b) => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.032, 8, 6), new THREE.MeshBasicMaterial({ color: new THREE.Color(r, g, b), fog: false, toneMapped: false })); m.position.set(x, y, z); drone.add(m); return m; };
  led(0.22, 0.22, -0.02, 5, 0.2, 0.15); led(0.22, -0.22, -0.02, 0.2, 5, 0.4); drone.userData.strobe = led(-0.22, 0, 0.05, 6, 6, 6);
  drone.traverse((o) => { if (o.isMesh && o.geometry.type !== 'SphereGeometry') o.castShadow = true; });
  scene.add(drone);
}
const trail = FX.makeTrail(420); scene.add(trail.line);
const cloud = FX.makeCloud(9000); scene.add(cloud.points);
const pulse = FX.makePulse(); scene.add(pulse);
const fov = FX.makeFov(DEPTH_HFOV, DEPTH_VFOV, CAM_PITCH, 9); scene.add(fov);
const dust = FX.makeDust();
const spot = new THREE.SpotLight(0xcfe8ff, 90, 22, 0.5, 0.85, 1.3); scene.add(spot, spot.target);
const spotDecal = new THREE.Mesh(new THREE.CircleGeometry(2.6, 32), new THREE.MeshBasicMaterial({ map: FX.radialTexture([[0, 'rgba(190,225,255,1)'], [0.5, 'rgba(160,200,255,0.35)'], [1, 'rgba(160,200,255,0)']]), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3, fog: false }));
const blob = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, opacity: 0.55, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 }));
scene.add(spotDecal, blob);
let trailSet = false;

// ------------------------------------------------------------------ sensors (JS mirror of world.py raycast incl. AABB slab test)
function castRay(ox, oy, oz, dx, dy, dz, maxR, nt, nb, T) {
  let best = maxR;
  for (let i = 0; i < nt.length; i++) {
    const [cx, cy, zg, h, rt, rc] = nt[i];
    const ocx = ox - cx, ocy = oy - cy, ocz = oz - (zg + h), b = dx * ocx + dy * ocy + dz * ocz, cc = ocx * ocx + ocy * ocy + ocz * ocz - rc * rc, disc = b * b - cc;
    if (disc > 0) { const t = -b - Math.sqrt(disc); if (t > 0 && t < best) best = t; }
    const a2 = dx * dx + dy * dy;
    if (a2 > 1e-6) { const b2 = dx * ocx + dy * ocy, c2 = ocx * ocx + ocy * ocy - rt * rt, d2 = b2 * b2 - a2 * c2;
      if (d2 > 0) { const t = (-b2 - Math.sqrt(d2)) / a2; if (t > 0 && t < best) { const zh = oz + t * dz; if (zh > zg && zh < zg + h) best = t; } } }
  }
  for (let i = 0; i < nb.length; i++) {
    const [cx, cy, cz, hx, hy, hz] = nb[i]; let tn = -1e9, tf = 1e9, miss = false;
    const ax = (o, d, c, h) => { if (Math.abs(d) < 1e-9) { if (o < c - h || o > c + h) miss = true; return; } let t1 = (c - h - o) / d, t2 = (c + h - o) / d; if (t1 > t2) { const s = t1; t1 = t2; t2 = s; } if (t1 > tn) tn = t1; if (t2 < tf) tf = t2; };
    ax(ox, dx, cx, hx); ax(oy, dy, cy, hy); ax(oz, dz, cz, hz);
    if (!miss && tf > tn && tn > 0 && tn < best) best = tn;
  }
  const gz0 = hAt(T, ox, oy); if (oz - gz0 < 0) return 0;
  if (dz > 0.05 && oz - gz0 > 4.5) return best;                          // climbing away from the ground: cannot hit the terrain
  const STEPS = 30; let pt = 0, pf = oz - gz0;
  for (let i = 1; i <= STEPS; i++) { const t = best * i / STEPS, f = oz + dz * t - hAt(T, ox + dx * t, oy + dy * t); if (f < 0) { best = pt + (t - pt) * pf / (pf - f); break; } pt = t; pf = f; }
  return best;
}
const rotMat = (w, x, y, z) => [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
const gauss = () => { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const depthRays = (() => {
  const out = [], th = Math.tan(DEPTH_HFOV * Math.PI / 360), tv = Math.tan(DEPTH_VFOV * Math.PI / 360), p = CAM_PITCH * Math.PI / 180;
  for (let j = 0; j < DEPTH_H; j++) for (let i = 0; i < DEPTH_W; i++) { const u = (i + 0.5) / DEPTH_W * 2 - 1, v = (j + 0.5) / DEPTH_H * 2 - 1;
    let x = 1, y = -u * th, z = -v * tv; const l = Math.hypot(x, y, z); x /= l; y /= l; z /= l; out.push([x * Math.cos(p) + z * Math.sin(p), y, -x * Math.sin(p) + z * Math.cos(p)]); }
  return out;
})();
const scanRays = (() => { const out = []; for (let a = 0; a < N_AZ; a++) for (let e = 0; e < N_EL; e++) { const az = (-70 + 140 * a / (N_AZ - 1)) * Math.PI / 180, el = [-35, -15, 0, 15, 35][e] * Math.PI / 180; out.push([Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)]); } return out; })();
const scanPos = new Float32Array(N_AZ * N_EL * 6), scanGeo = new THREE.BufferGeometry();
scanGeo.setAttribute('position', new THREE.BufferAttribute(scanPos, 3));
const scanLines = new THREE.LineSegments(scanGeo, new THREE.LineBasicMaterial({ color: 0x66e0ff, transparent: true, opacity: 0.3, fog: false })); scanLines.frustumCulled = false; scene.add(scanLines);

// ------------------------------------------------------------------ voxel SLAM map (chunked) + metrics
let voxels = new Map(); const chunks = new Map(), dirty = new Set(), slamGroup = new THREE.Group(); scene.add(slamGroup);
const vkey = (ix, iy, iz) => (ix + 4096) + 8192 * ((iy + 4096) + 8192 * (iz + 4096));
const unkey = (k) => [(k % 8192) - 4096, (Math.floor(k / 8192) % 8192) - 4096, Math.floor(k / 67108864) - 4096];
const ckeyOf = (ix, iy, iz) => ((Math.floor(ix / CH) + 512) * 1024 + (Math.floor(iy / CH) + 512)) * 1024 + (Math.floor(iz / CH) + 512);
let inside = 0, good = 0, covered = new Set(), quiet = false;

function getChunk(ix, iy, iz) {
  const k = ckeyOf(ix, iy, iz); let c = chunks.get(k);
  if (!c) { c = { k, vox: [], vb: new Map(), lines: null, fill: null }; chunks.set(k, c); }
  return c;
}
function addVoxel(x, y, z) {
  const ix = Math.floor(x / VOX), iy = Math.floor(y / VOX), iz = Math.floor(z / VOX), k = vkey(ix, iy, iz), n = (voxels.get(k) || 0) + 1;
  voxels.set(k, n);
  if (n !== OCC_HITS) return;
  getChunk(ix, iy, iz).vox.push(k);
  for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) { const ck = ckeyOf(ix + a, iy + b, iz + c); if (chunks.has(ck)) dirty.add(ck); }
  // incremental scorecard
  const gx = (ix + 0.5) * VOX, gy = (iy + 0.5) * VOX, gt = world.gt;
  if (gt.region(gx, gy)) { inside++; if (nearGT(gt.set, ix, iy, iz)) good++; }
  for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) { const kk = vkey(ix + a, iy + b, iz + c); if (gt.set.has(kk)) covered.add(kk); }
  if (z - hAt(world.T, x, y) > 1.0) { const o = obsIdx(x, y); if (o >= 0) obsMapped[o] = 1; }
}
const occ = (ix, iy, iz) => (voxels.get(vkey(ix, iy, iz)) || 0) >= OCC_HITS;
const FACES = [[[1, 0, 0], [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]], [[-1, 0, 0], [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]]],
  [[0, 1, 0], [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]]], [[0, -1, 0], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]],
  [[0, 0, 1], [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]], [[0, 0, -1], [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]]];
const hCol = [new THREE.Color(0x2a62ff), new THREE.Color(0x2fb4ff), new THREE.Color(0x6fe3ff)];

function rebuildChunk(ch, stamp) {
  // Exposed voxel faces -> quads over shared corner cells; vertices placed surface-nets style then Laplacian-relaxed (per chunk).
  const vid = new Map(), C = [], quads = [];
  const getV = (cx, cy, cz) => { const k = vkey(cx, cy, cz); let i = vid.get(k); if (i === undefined) { i = C.length / 3; vid.set(k, i); C.push(cx, cy, cz); } return i; };
  for (const k of ch.vox) {
    const [ix, iy, iz] = unkey(k);
    if (!(occ(ix + 1, iy, iz) || occ(ix - 1, iy, iz) || occ(ix, iy + 1, iz) || occ(ix, iy - 1, iz) || occ(ix, iy, iz + 1) || occ(ix, iy, iz - 1))) continue;
    for (const [d, cs] of FACES) { if (d[2] === -1 || occ(ix + d[0], iy + d[1], iz + d[2])) continue; quads.push(cs.map((c) => getV(ix + c[0], iy + c[1], iz + c[2]))); }
  }
  const nV = C.length / 3, orig = new Float32Array(nV * 3);
  for (let v = 0; v < nV; v++) {
    const cx = C[v * 3], cy = C[v * 3 + 1], cz = C[v * 3 + 2]; let sx = 0, sy = 0, sz = 0, m = 0; const o = (a, b, c) => occ(cx + a, cy + b, cz + c);
    for (let a = -1; a <= 0; a++) for (let b = -1; b <= 0; b++) for (let c = -1; c <= 0; c++) {
      const me = o(a, b, c);
      if (a === -1 && o(0, b, c) !== me) { sx += cx; sy += cy + b + 0.5; sz += cz + c + 0.5; m++; }
      if (b === -1 && o(a, 0, c) !== me) { sx += cx + a + 0.5; sy += cy; sz += cz + c + 0.5; m++; }
      if (c === -1 && o(a, b, 0) !== me) { sx += cx + a + 0.5; sy += cy + b + 0.5; sz += cz; m++; }
    }
    if (m) { orig[v * 3] = sx / m * VOX; orig[v * 3 + 1] = sy / m * VOX; orig[v * 3 + 2] = sz / m * VOX; } else { orig[v * 3] = cx * VOX; orig[v * 3 + 1] = cy * VOX; orig[v * 3 + 2] = cz * VOX; }
  }
  const cur = Float32Array.from(orig), nb = Array.from({ length: nV }, () => []);
  for (const q of quads) for (let i = 0; i < 4; i++) nb[q[i]].push(q[(i + 1) % 4], q[(i + 3) % 4]);
  const lim = 0.6 * VOX;
  for (let it = 0; it < 3; it++) { const nxt = Float32Array.from(cur);
    for (let v = 0; v < nV; v++) { const L = nb[v]; if (!L.length) continue; let x = 0, y = 0, z = 0; for (const u of L) { x += cur[u * 3]; y += cur[u * 3 + 1]; z += cur[u * 3 + 2]; } x /= L.length; y /= L.length; z /= L.length;
      for (let k = 0; k < 3; k++) { const t = [x, y, z][k], c0 = cur[v * 3 + k] + 0.5 * (t - cur[v * 3 + k]); nxt[v * 3 + k] = Math.min(Math.max(c0, orig[v * 3 + k] - lim), orig[v * 3 + k] + lim); } }
    cur.set(nxt); }
  const vcol = new Float32Array(nV * 3), born = new Float32Array(nV), tmp = new THREE.Color();
  for (let v = 0; v < nV; v++) {
    const h = Math.min(Math.max((cur[v * 3 + 2] - hAt(world.T, cur[v * 3], cur[v * 3 + 1])) / 9, 0), 1);
    h < 0.35 ? tmp.copy(hCol[0]).lerp(hCol[1], h / 0.35) : tmp.copy(hCol[1]).lerp(hCol[2], (h - 0.35) / 0.65); vcol.set([tmp.r, tmp.g, tmp.b], v * 3);
    const vk = vkey(C[v * 3], C[v * 3 + 1], C[v * 3 + 2]); let b = ch.vb.get(vk); if (b === undefined) { b = stamp; ch.vb.set(vk, b); } born[v] = b;
  }
  const seen = new Set(), lp = [], lc = [], lb = [], idx = [];
  const edge = (a, b) => { const lo = Math.min(a, b), hi = Math.max(a, b), k = lo * nV + hi; if (seen.has(k)) return; seen.add(k);
    lp.push(cur[a * 3], cur[a * 3 + 1], cur[a * 3 + 2], cur[b * 3], cur[b * 3 + 1], cur[b * 3 + 2]); lc.push(vcol[a * 3], vcol[a * 3 + 1], vcol[a * 3 + 2], vcol[b * 3], vcol[b * 3 + 1], vcol[b * 3 + 2]); lb.push(born[a], born[b]); };
  for (const q of quads) { edge(q[0], q[1]); edge(q[1], q[2]); edge(q[2], q[3]); edge(q[3], q[0]); edge(q[0], q[2]); idx.push(q[0], q[1], q[2], q[0], q[2], q[3]); }
  if (!ch.lines) { ch.lines = new THREE.LineSegments(new THREE.BufferGeometry(), FX.slamLineMat); ch.fill = new THREE.Mesh(new THREE.BufferGeometry(), FX.slamFillMat); ch.fill.renderOrder = 2; ch.lines.renderOrder = 3; slamGroup.add(ch.lines, ch.fill); }
  ch.lines.geometry.dispose(); ch.fill.geometry.dispose();
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(lp, 3)); g.setAttribute('color', new THREE.Float32BufferAttribute(lc, 3)); g.setAttribute('aBorn', new THREE.Float32BufferAttribute(lb, 1)); ch.lines.geometry = g;
  const f = new THREE.BufferGeometry(); f.setAttribute('position', new THREE.BufferAttribute(cur, 3)); f.setAttribute('color', new THREE.BufferAttribute(vcol, 3)); f.setAttribute('aBorn', new THREE.BufferAttribute(born, 1)); f.setIndex(idx); ch.fill.geometry = f;
}
function processDirty(budgetMs, stamp) {
  const t0 = performance.now();
  for (const k of dirty) { dirty.delete(k); const c = chunks.get(k); if (c) rebuildChunk(c, stamp); if (performance.now() - t0 > budgetMs) break; }
}
function clearMap() {
  voxels = new Map(); for (const c of chunks.values()) { if (c.lines) { c.lines.geometry.dispose(); c.fill.geometry.dispose(); } } chunks.clear(); dirty.clear(); slamGroup.clear();
  covered = new Set(); inside = good = 0; cloud.clear(); scanGeo.setDrawRange(0, 0);
  if (world) { observed.fill(0); obsMapped.fill(0); exploredCells = 0; fowDirty = true; }
  clrHist.length = 0; depthImg.fill(14);
}

function buildGT() {
  const set = new Set(), T = world.T, b = world.bounds;
  const region = world.isM ? (x, y) => x >= 0 && x <= world.ep.stats.length && Math.abs(y - pathY(x)) < 18 : (x, y) => Math.abs(x) <= ARENA.x && Math.abs(y) <= ARENA.y;
  const add = (x, y, z) => set.add(vkey(Math.floor(x / VOX), Math.floor(y / VOX), Math.floor(z / VOX)));
  const st = VOX * 0.5, x0 = world.isM ? 0 : -ARENA.x, x1 = world.isM ? world.ep.stats.length : ARENA.x, y0 = world.isM ? -60 : -ARENA.y, y1 = world.isM ? 60 : ARENA.y;
  for (let x = x0; x <= x1; x += st) for (let y = y0; y <= y1; y += st) if (region(x, y)) add(x, y, hAt(T, x, y));
  const N = 500, ga = Math.PI * (3 - Math.sqrt(5));
  for (const [cx, cy, zg, h, rt, rc] of world.trees) {
    if (!region(cx, cy)) continue;
    for (let z = zg; z <= zg + h; z += 0.25) for (let a = 0; a < 16; a++) { const th = a / 16 * 6.2832; add(cx + rt * Math.cos(th), cy + rt * Math.sin(th), z); }
    for (let i = 0; i < N; i++) { const zz = 1 - 2 * (i + 0.5) / N, rr = Math.sqrt(1 - zz * zz), th = ga * i; add(cx + rc * rr * Math.cos(th), cy + rc * rr * Math.sin(th), zg + h + rc * zz); }
  }
  for (const [cx, cy, cz, hx, hy, hz] of world.boxes) {
    if (!region(cx, cy)) continue;
    for (let u = -hx; u <= hx + 1e-6; u += st) for (let v = -hy; v <= hy + 1e-6; v += st) { add(cx + u, cy + v, cz + hz); }
    for (let u = -hx; u <= hx + 1e-6; u += st) for (let w = -hz; w <= hz + 1e-6; w += st) { add(cx + u, cy - hy, cz + w); add(cx + u, cy + hy, cz + w); }
    for (let v = -hy; v <= hy + 1e-6; v += st) for (let w = -hz; w <= hz + 1e-6; w += st) { add(cx - hx, cy + v, cz + w); add(cx + hx, cy + v, cz + w); }
  }
  return { set, region };
}
const nearGT = (set, ix, iy, iz) => { for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) if (set.has(vkey(ix + a, iy + b, iz + c))) return true; return false; };
function updateMetrics() {
  if (!world) return;
  S.coverage = world.gt.set.size ? (100 * covered.size / world.gt.set.size).toFixed(1) + ' %' : '-';
  S.precision = inside ? (100 * good / inside).toFixed(1) + ' %' : '-';
  S.explored = exploredCells + ' m²';
}
function updateFogOfWar() {
  if (!world) return;
  const t = world.terr, col = t.g.attributes.color, pos = t.g.attributes.position, base = t.baseCol, b = world.bounds;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i); let f = 1;
    if (S.fogOfWar && x > b.x0 && x < b.x1 && y > b.y0 && y < b.y1) { const k = obsIdx(x, y); f = k >= 0 && observed[k] ? 1.25 : 0.5; }
    col.setXYZ(i, base[i * 3] * f, base[i * 3 + 1] * f, base[i * 3 + 2] * f);
  }
  col.needsUpdate = true; fowDirty = false;
}
function exportPLY() {
  const pts = [];
  for (const [k, n] of voxels) { if (n < OCC_HITS) continue; const [ix, iy, iz] = unkey(k), x = (ix + 0.5) * VOX, y = (iy + 0.5) * VOX, z = (iz + 0.5) * VOX, h = Math.min(Math.max((z - hAt(world.T, x, y)) / 10, 0), 1);
    pts.push(`${x.toFixed(3)} ${y.toFixed(3)} ${z.toFixed(3)} ${Math.round(40 + 200 * h)} ${Math.round(200 - 60 * h)} ${Math.round(255 - 200 * h)}`); }
  const txt = `ply\nformat ascii 1.0\ncomment drone SLAM map, voxel ${VOX} m, z up\nelement vertex ${pts.length}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n${pts.join('\n')}\n`;
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([txt], { type: 'text/plain' })); a.download = `map_episode${S.episode + 1}_t${S.time.toFixed(1)}s.ply`; a.click();
}

// ------------------------------------------------------------------ playback / sensing
const clrHist = [], depthImg = new Float32Array(DEPTH_W * DEPTH_H).fill(14);
function samplePose(ep, t) {
  const F = ep.frames, f = Math.min(Math.max(t / DT, 0), F.length - 1), i = Math.floor(f), j = Math.min(i + 1, F.length - 1), a = f - i;
  const A = F[i], B = F[j], p = [0, 1, 2].map((k) => A[k] + (B[k] - A[k]) * a);
  const qa = new THREE.Quaternion(A[4], A[5], A[6], A[3]), qb = new THREE.Quaternion(B[4], B[5], B[6], B[3]); qa.slerp(qb, a);
  let e = [0, 0, 0]; if (ep.est && ep.est.length) { const EA = ep.est[Math.min(i, ep.est.length - 1)], EB = ep.est[Math.min(j, ep.est.length - 1)]; e = [0, 1, 2].map((k) => EA[k] + (EB[k] - EA[k]) * a); }
  return { p, q: [qa.w, qa.x, qa.y, qa.z], thrust: A[7] + (B[7] - A[7]) * a, e, f };
}
let lastSense = -1, sensePending = true, doneAt = 0, lastMetricT = 0, wall = 0;

function sense(ps, tm, withCloud = true) {
  const [ox, oy, oz] = ps.p, [qw, qx, qy, qz] = ps.q, M = rotMat(qw, qx, qy, qz), off = S.useEst ? ps.e : [0, 0, 0];
  const maxR = DATA.meta.max_range + 2, nt = nearTrees(ox, oy, maxR), nbx = nearBoxes(ox, oy, maxR), T = world.T;
  let minT = maxR;
  for (let k = 0; k < depthRays.length; k++) {
    const r = depthRays[k], dx = M[0] * r[0] + M[1] * r[1] + M[2] * r[2], dy = M[3] * r[0] + M[4] * r[1] + M[5] * r[2], dz = M[6] * r[0] + M[7] * r[1] + M[8] * r[2];
    let t = castRay(ox, oy, oz, dx, dy, dz, maxR, nt, nbx, T);
    if (t >= maxR - 1e-3) { depthImg[k] = maxR; continue; }
    if (S.noise > 0) t = Math.max(0.05, t + S.noise * gauss());
    depthImg[k] = t; if (t < minT) minT = t;
    const hx = ox + dx * t, hy = oy + dy * t, hz = oz + dz * t, bx = hx + off[0], by = hy + off[1], bz = hz + off[2];
    if (withCloud) cloud.push(bx, by, bz, 1, 0.04 + 0.22 * Math.min(t / maxR, 1), 0.03, tm);
    addVoxel(bx, by, bz);
    if (Math.abs(hz - hAt(T, hx, hy)) < 0.35) markObserved(bx, by);
  }
  clrHist.push([tm, minT]); if (clrHist.length > 4000) clrHist.shift();
  if (withCloud) cloud.commit();
  const yaw = Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)), c = Math.cos(yaw), s = Math.sin(yaw); let m = 0;
  for (const r of scanRays) { const dx = c * r[0] - s * r[1], dy = s * r[0] + c * r[1], dz = r[2], t = castRay(ox, oy, oz, dx, dy, dz, DATA.meta.max_range, nt, nbx, T); scanPos.set([ox, oy, oz, ox + dx * t, oy + dy * t, oz + dz * t], m); m += 6; }
  scanGeo.attributes.position.needsUpdate = true; scanGeo.setDrawRange(0, N_AZ * N_EL * 2);
}
function seek(t, step = SENSE_DT) {
  clearMap();
  const ep = DATA.episodes[S.episode], tEnd = (ep.frames.length - 1) * DT; t = Math.min(Math.max(t, 0), tEnd);
  for (let u = 0; u < t; u += step) sense(samplePose(ep, u), u, u > t - 1.4);
  processDirty(1e9, -1e3);
  S.time = t; lastSense = t; sensePending = true; lastEvt = t;
}
function computeDetections(ep) {
  const F = ep.frames, step = Math.max(1, Math.round(0.2 / DT)), T = world.T;
  for (let i = 0; i < F.length; i += step) {
    const f = F[i], ox = f[0], oy = f[1], oz = f[2], yaw = Math.atan2(2 * (f[3] * f[6] + f[4] * f[5]), 1 - 2 * (f[5] * f[5] + f[6] * f[6])); let nt = null, nbx = null;
    for (const p of world.pois) {
      if (p.tDet < Infinity) continue; const dx = p.x - ox, dy = p.y - oy, d = Math.hypot(dx, dy); if (d > 10) continue;
      const br = Math.atan2(dy, dx) - yaw, bw = Math.atan2(Math.sin(br), Math.cos(br)); if (Math.abs(bw) > 0.95) continue;
      nt = nt || nearTrees(ox, oy, 12); nbx = nbx || nearBoxes(ox, oy, 12);
      const tz = hAt(T, p.x, p.y) + 0.5, dz = tz - oz, L = Math.hypot(d, dz), t = castRay(ox, oy, oz, dx / L, dy / L, dz / L, L, nt, nbx, T);
      if (t >= L - 0.7) p.tDet = i * DT;
    }
  }
}

function loadEpisode(i) {
  S.episode = (i + DATA.episodes.length) % DATA.episodes.length;
  const ep = DATA.episodes[S.episode];
  if (!ep.speed) { const F = ep.frames; ep.speed = F.map((f, i) => { const g = F[Math.max(i - 3, 0)], h = F[Math.min(i + 3, F.length - 1)], n = Math.min(i + 3, F.length - 1) - Math.max(i - 3, 0) || 1; return Math.hypot(h[0] - g[0], h[1] - g[1], h[2] - g[2]) / (n * DT); }); }
  buildWorld(ep); seek(0); doneAt = 0;
  const p = samplePose(ep, 0); camera.position.set(p.p[0] - 10, p.p[1] - 7, p.p[2] + 3.5); controls.target.set(...p.p); controls.update(); prevDrone.set(...p.p);
  dirSt.shotT = 99; world.cover.cx = 1e9;
}
const prevDrone = new THREE.Vector3();
function updateOutcome(ep) {
  const o = document.getElementById('outcome'), st = ep.stats ? (world.isM ? ` &nbsp;·&nbsp; ${ep.stats.length} m · ${ep.stats.time}s` : ` &nbsp;·&nbsp; ${ep.stats.time}s · ${ep.stats.path}m · min clearance ${ep.stats.min_clearance}m`) : '';
  const dw = disW(world.curX || 0), zone = world.isM ? ` &nbsp;·&nbsp; <b style="color:${dw > 0.5 ? '#ff9a5e' : '#8fe0a0'}">${dw > 0.5 ? 'DISASTER ZONE' : 'FOREST'}</b>` : '';
  o.innerHTML = `${world.isM ? 'MISSION' : `episode ${S.episode + 1}/${DATA.episodes.length}`} &nbsp;·&nbsp; <b class="${ep.outcome}">${ep.outcome.toUpperCase()}</b>${st}${zone}` + (DATA.meta.policy ? ` &nbsp;·&nbsp; ${DATA.meta.policy}${DATA.meta.realism ? ' (realism)' : ''}` : '');
}

// ------------------------------------------------------------------ side panels
const cTel = document.getElementById('telem'), cSpark = document.getElementById('spark'), cDepth = document.getElementById('depthc'), cMini = document.getElementById('mini'), cStrip = document.getElementById('strip');
const gSpark = cSpark.getContext('2d'), gDepth = cDepth.getContext('2d'), gMini = cMini.getContext('2d'), gStrip = cStrip.getContext('2d'), dImg = gDepth.createImageData(DEPTH_W, DEPTH_H);
function turbo(x) { x = Math.min(Math.max(x, 0), 1); return [Math.round(255 * Math.min(Math.max(1.5 - Math.abs(4 * x - 3), 0) + Math.max(1 - 4 * x, 0) * 0.2, 1)), Math.round(255 * Math.min(Math.max(1.5 - Math.abs(4 * x - 2), 0), 1)), Math.round(255 * Math.min(Math.max(1.5 - Math.abs(4 * x - 1), 0), 1))]; }
function drawDepth(maxR) { for (let i = 0; i < depthImg.length; i++) { const t = depthImg[i], c = t >= maxR - 1e-3 ? [14, 18, 26] : turbo(1 - Math.min(t / (maxR - 2), 1)); dImg.data.set([c[0], c[1], c[2], 255], i * 4); } gDepth.putImageData(dImg, 0, 0); }
function drawSpark(ep) {
  const W = cSpark.width, H = cSpark.height, g = gSpark, tEnd = (ep.frames.length - 1) * DT, X = (t) => 4 + (W - 8) * t / tEnd;
  g.clearRect(0, 0, W, H); g.fillStyle = 'rgba(8,12,18,0.6)'; g.fillRect(0, 0, W, H);
  const line = (pts, col, ymax, wd) => { g.beginPath(); pts.forEach(([t, v], i) => { const x = X(t), y = H - 4 - (H - 8) * Math.min(v / ymax, 1); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.strokeStyle = col; g.lineWidth = wd; g.stroke(); };
  const sp = ep.speed.filter((_, i) => i % 4 === 0).map((v, i) => [i * 4 * DT, v]);
  line(sp, 'rgba(255,154,60,0.28)', 7, 1); line(sp.filter(([t]) => t <= S.time), '#ff9a3c', 7, 1.6); line(clrHist.filter(([t]) => t <= S.time), '#5ad0ff', 14, 1.6);
  g.strokeStyle = 'rgba(255,255,255,0.5)'; g.beginPath(); g.moveTo(X(S.time), 0); g.lineTo(X(S.time), H); g.stroke();
  g.fillStyle = '#ff9a3c'; g.font = '10px ui-monospace,monospace'; g.fillText('speed (0-7 m/s)', 6, 11); g.fillStyle = '#5ad0ff'; g.fillText('nearest return (0-14 m)', 120, 11);
}
function drawStrip(ep, ps) {
  const W = cStrip.width, H = cStrip.height, g = gStrip, b = world.bounds; g.fillStyle = 'rgba(8,12,18,0.7)'; g.fillRect(0, 0, W, H);
  if (!world.isM) { g.fillStyle = 'rgba(200,215,230,0.6)'; g.font = '10px ui-monospace,monospace'; g.fillText('single-arena episode', 6, 26); return; }
  const L = ep.stats.length, X = (x) => 6 + (W - 12) * x / L;
  for (const s of world.biomes) { g.fillStyle = s.kind === 'disaster' ? 'rgba(190,100,60,0.55)' : 'rgba(60,140,80,0.45)'; g.fillRect(X(s.x0), 14, X(s.x1) - X(s.x0), 14); }
  g.strokeStyle = 'rgba(255,255,255,0.35)'; g.beginPath(); world.path.forEach((p, i) => { const x = X(p[0]), y = 21 + p[1] * 0.12; i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
  g.strokeStyle = '#ff9a3c'; g.lineWidth = 2; g.beginPath(); let on = false; world.path.forEach((p) => { if (p[0] <= ps.p[0]) { const x = X(p[0]), y = 21 + p[1] * 0.12; on ? g.lineTo(x, y) : g.moveTo(x, y); on = true; } }); g.stroke(); g.lineWidth = 1;
  for (const p of world.pois) { g.fillStyle = p.tDet <= S.time ? '#ffd24a' : 'rgba(255,255,255,0.18)'; g.fillRect(X(p.x) - 1, 33, 2, p.tDet <= S.time ? 6 : 3); }
  for (const f of world.fires) { g.fillStyle = 'rgba(255,110,40,0.9)'; g.fillRect(X(f[0]) - 0.5, 10, 1.5, 3); }
  g.fillStyle = '#fff'; g.beginPath(); g.arc(X(ps.p[0]), 21 + ps.p[1] * 0.12, 3, 0, 6.28); g.fill();
  g.fillStyle = 'rgba(200,215,230,0.7)'; g.font = '9px ui-monospace,monospace'; g.fillText(`${Math.round(ps.p[0])} / ${L} m   green=forest  rust=disaster  tick=survivor  orange=fire`, 4, 8);
}
function drawMini(ep, ps) {
  const W = cMini.width, H = cMini.height, g = gMini, b = world.bounds;
  let cx, cy, wx, wy;
  if (world.isM) { cx = ps.p[0]; cy = ps.p[1]; wx = 84; wy = 84 * H / W; } else { cx = 0; cy = 0; wx = 2 * ARENA.x; wy = 2 * ARENA.y; }
  const X0 = cx - wx / 2, Y0 = cy - wy / 2, sx = W / wx, sy = H / wy, px = (x) => (x - X0) * sx, py = (y) => H - (y - Y0) * sy;
  g.fillStyle = '#0a0f15'; g.fillRect(0, 0, W, H);
  g.fillStyle = 'rgba(70,140,80,0.45)';
  for (let yy = Math.floor(Y0); yy < Y0 + wy; yy++) for (let xx = Math.floor(X0); xx < X0 + wx; xx++) { const k = obsIdx(xx + 0.5, yy + 0.5); if (k >= 0 && observed[k]) g.fillRect(px(xx), py(yy + 1), sx + 0.5, sy + 0.5); }
  if (S.mmTruth) {
    g.strokeStyle = 'rgba(190,190,190,0.4)'; g.lineWidth = 1; for (const t of world.gT.query(cx, cy, wx * 0.6)) { g.beginPath(); g.arc(px(t[0]), py(t[1]), t[5] * sx, 0, 6.283); g.stroke(); }
    g.strokeStyle = 'rgba(225,170,120,0.55)'; for (const bx of world.gB.query(cx, cy, wx * 0.6)) g.strokeRect(px(bx[0] - bx[3]), py(bx[1] + bx[4]), bx[3] * 2 * sx, bx[4] * 2 * sy);
  }
  g.fillStyle = 'rgba(58,168,255,0.85)';
  for (let yy = Math.floor(Y0); yy < Y0 + wy; yy++) for (let xx = Math.floor(X0); xx < X0 + wx; xx++) { const k = obsIdx(xx + 0.5, yy + 0.5); if (k >= 0 && obsMapped[k]) g.fillRect(px(xx), py(yy + 1), sx + 0.3, sy + 0.3); }
  for (const f of world.fires) if (Math.abs(f[0] - cx) < wx / 2 && Math.abs(f[1] - cy) < wy / 2) { g.fillStyle = 'rgba(255,120,40,0.95)'; g.beginPath(); g.arc(px(f[0]), py(f[1]), 2.4, 0, 6.28); g.fill(); }
  const F = ep.frames, last = Math.min(Math.floor(S.time / DT), F.length - 1);
  g.lineWidth = 1.5; g.strokeStyle = '#ff9a3c'; g.beginPath(); for (let i = Math.max(0, last - 1500); i <= last; i += 3) { const f = F[i]; i === Math.max(0, last - 1500) ? g.moveTo(px(f[0]), py(f[1])) : g.lineTo(px(f[0]), py(f[1])); } g.stroke();
  if (S.useEst && ep.est) { g.strokeStyle = '#ff5ad0'; g.setLineDash([3, 3]); g.beginPath(); for (let i = Math.max(0, last - 1500); i <= last; i += 3) { const f = F[i], e = ep.est[Math.min(i, ep.est.length - 1)]; i === Math.max(0, last - 1500) ? g.moveTo(px(f[0] + e[0]), py(f[1] + e[1])) : g.lineTo(px(f[0] + e[0]), py(f[1] + e[1])); } g.stroke(); g.setLineDash([]); }
  for (const p of world.pois) if (Math.abs(p.x - cx) < wx / 2 && Math.abs(p.y - cy) < wy / 2) { const f = p.tDet <= S.time; g.fillStyle = f ? '#ffd24a' : 'rgba(255,255,255,0.22)'; g.beginPath(); g.arc(px(p.x), py(p.y), f ? 3.2 : 1.6, 0, 6.28); g.fill(); if (f) { g.strokeStyle = '#ffd24a'; g.beginPath(); g.arc(px(p.x), py(p.y), 6, 0, 6.28); g.stroke(); } }
  const gl = ep.goal; g.strokeStyle = '#59ffa0'; g.lineWidth = 2; g.beginPath(); g.arc(px(gl[0]), py(gl[1]), 1.5 * sx + 2, 0, 6.283); g.stroke();
  const [qw, qx, qy, qz] = ps.q, yaw = Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)), dx = px(ps.p[0]), dy = py(ps.p[1]), R = DATA.meta.max_range * sx, hf = DEPTH_HFOV * Math.PI / 360;
  g.fillStyle = 'rgba(255,255,255,0.12)'; g.beginPath(); g.moveTo(dx, dy); g.arc(dx, dy, R, -(yaw + hf), -(yaw - hf)); g.closePath(); g.fill();
  g.fillStyle = '#fff'; g.beginPath(); g.moveTo(dx + 6 * Math.cos(-yaw), dy + 6 * Math.sin(-yaw)); g.lineTo(dx + 4 * Math.cos(-yaw + 2.5), dy + 4 * Math.sin(-yaw + 2.5)); g.lineTo(dx + 4 * Math.cos(-yaw - 2.5), dy + 4 * Math.sin(-yaw - 2.5)); g.fill();
  g.fillStyle = 'rgba(200,215,230,0.6)'; g.font = '10px ui-monospace,monospace'; g.fillText(`${Math.round(wx)} x ${Math.round(wy)} m  cyan=mapped  grey/tan=true  yellow=found`, 6, H - 6);
}
function drawPanels(ep, ps) {
  const g = ps.thrust / 9.81, v = ep.speed[Math.min(Math.round(S.time / DT), ep.speed.length - 1)] || 0, gz = hAt(world.T, ps.p[0], ps.p[1]);
  const toGoal = Math.hypot(ep.goal[0] - ps.p[0], ep.goal[1] - ps.p[1], ep.goal[2] - ps.p[2]), nr = clrHist.length ? clrHist[clrHist.length - 1][1] : NaN, maxR = DATA.meta.max_range + 2;
  const nF = world.pois.filter((p) => p.tDet <= S.time).length; S.found = `${nF} / ${world.pois.length}`;
  cTel.innerHTML = `speed <b>${v.toFixed(1)}</b> m/s &nbsp; AGL <b>${(ps.p[2] - gz).toFixed(1)}</b> m &nbsp; thrust <b>${g.toFixed(2)}g</b><br>` +
    `to goal <b>${toGoal.toFixed(0)}</b> m &nbsp; nearest return <b>${nr >= maxR - 0.01 ? '—' : nr.toFixed(1)}</b> m<br>` +
    `map <b>${(voxels.size / 1000).toFixed(0)}k</b> vox &nbsp; coverage <b>${S.coverage}</b> &nbsp; precision <b>${S.precision}</b><br>` +
    `ground explored <b>${S.explored}</b> &nbsp; INS drift <b>${S.drift}</b>` + (world.pois.length ? `<br>survivors / hikers located <b style="color:#ffd24a">${S.found}</b>` : '');
  drawSpark(ep); drawDepth(maxR); drawMini(ep, ps); drawStrip(ep, ps);
}

// ------------------------------------------------------------------ camera: orbit-follow + cinematic director + recording
const _v = new THREE.Vector3(), _x = new THREE.Vector3(), _z = new THREE.Vector3(), _m = new THREE.Matrix4();
const dirSt = { shot: 0, shotT: 99, dur: 8, pos: new THREE.Vector3(), tgt: new THREE.Vector3(), poi: null, order: [0, 3, 1, 4, 2, 3, 0, 1], k: 0, heading: new THREE.Vector3(1, 0, 0), init: false };
function liftCamera(drone, follow) {
  let cg = hAt(world.T, camera.position.x, camera.position.y) + 1.8;
  if (follow) { cg = Math.max(cg, drone.z + 2.0); for (let k = 1; k <= 12; k++) { const f = k / 12, px = drone.x + (camera.position.x - drone.x) * f, py = drone.y + (camera.position.y - drone.y) * f; cg = Math.max(cg, drone.z + (hAt(world.T, px, py) + 1.0 - drone.z) / f); } }
  if (camera.position.z < cg) camera.position.z = cg;
}
function director(dt, ps, ep) {
  const a = samplePose(ep, S.time + 0.5).p, b = samplePose(ep, S.time - 0.5).p;
  const h = new THREE.Vector3(a[0] - b[0], a[1] - b[1], 0); if (h.lengthSq() > 1e-4) dirSt.heading.lerp(h.normalize(), 1 - Math.exp(-dt * 2.5)).normalize();
  const hd = dirSt.heading, left = new THREE.Vector3(-hd.y, hd.x, 0), p = _v.set(...ps.p), t = dirSt.shotT;
  dirSt.shotT += dt;
  if (dirSt.shotT > dirSt.dur && !dirSt.poi) { dirSt.shot = dirSt.order[dirSt.k++ % dirSt.order.length]; dirSt.shotT = 0; dirSt.dur = 6 + 3 * Math.abs(Math.sin(dirSt.k * 2.3)); }
  if (dirSt.poi && dirSt.shotT > 5) dirSt.poi = null, dirSt.shotT = 99;
  const des = new THREE.Vector3(), tg = p.clone(), tt = dirSt.shotT;
  if (dirSt.poi) { const q = dirSt.poi, ang = tt * 0.45 + 1; des.set(q.x + Math.cos(ang) * 7, q.y + Math.sin(ang) * 7, hAt(world.T, q.x, q.y) + 3.2); tg.set(q.x, q.y, hAt(world.T, q.x, q.y) + 1.0).lerp(p, Math.min(tt / 5, 1) * 0.35); }
  else switch (dirSt.shot) {
    case 0: des.copy(p).addScaledVector(hd, -9).addScaledVector(left, 2.5 * Math.sin(tt * 0.35)).add(new THREE.Vector3(0, 0, 3.4)); tg.addScaledVector(hd, 5); break;   // chase
    case 1: des.copy(p).addScaledVector(left, 10).addScaledVector(hd, -2 + tt * 0.4).add(new THREE.Vector3(0, 0, 2.2)); break;                                          // side track
    case 2: des.copy(p).addScaledVector(hd, -3).add(new THREE.Vector3(0, 0, 30)); tg.addScaledVector(hd, 6); break;                                                      // top-down
    case 3: { const an = tt * 0.32 + 0.6; des.set(p.x + Math.cos(an) * 12, p.y + Math.sin(an) * 12, p.z + 2.5); break; }                                                  // orbit
    default: des.copy(p).addScaledVector(hd, 11).addScaledVector(left, -2).add(new THREE.Vector3(0, 0, 2.6)); break;                                                      // oncoming
  }
  if (!dirSt.init || dirSt.shotT < 0.02) { if (!dirSt.init) { dirSt.pos.copy(camera.position); dirSt.tgt.copy(p); dirSt.init = true; } }
  const k = 1 - Math.exp(-dt * (dirSt.shotT < 0.6 ? 1.2 : 2.6));
  dirSt.pos.lerp(des, k); dirSt.tgt.lerp(tg, 1 - Math.exp(-dt * 4));
  camera.position.copy(dirSt.pos); liftCamera(p, dirSt.shot !== 2 || !!dirSt.poi); dirSt.pos.copy(camera.position);
  camera.lookAt(dirSt.tgt); camera.fov += ((dirSt.shot === 2 ? 58 : 46 + Math.min(((ep.speed[Math.round(S.time / DT)] || 0)), 6) * 1.2) - camera.fov) * Math.min(1, dt * 2); camera.updateProjectionMatrix();
}
let rec = null, recChunks = [], recAuto = false;
function toggleRecord(auto = false) {
  const el = document.getElementById('rec');
  if (rec) { rec.stop(); return; }
  if (!window.MediaRecorder) { toast('MediaRecorder not supported in this browser'); return; }
  const stream = renderer.domElement.captureStream(30), mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find((m) => MediaRecorder.isTypeSupported(m));
  rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 14e6 }); recChunks = []; recAuto = auto;
  rec.ondataavailable = (e) => e.data.size && recChunks.push(e.data);
  rec.onstop = () => { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob(recChunks, { type: 'video/webm' })); a.download = `drone_slam_${world.isM ? 'mission' : 'episode'}.webm`; a.click(); rec = null; el.style.display = 'none'; };
  rec.start(500); el.style.display = 'block';
}
function recordRun() { S.cine = true; controls.enabled = false; S.follow = true; S.loop = false; seek(0); S.play = true; toggleRecord(true); toast('Recording the full run (cinematic camera). The file downloads when the flight ends.'); }
let toastT = 0; function toast(msg) { const e = document.getElementById('toast'); e.textContent = msg; e.style.opacity = 1; toastT = 4; }
let lastEvt = 0;

// ------------------------------------------------------------------ GUI
const gui = new GUI({ title: 'Drone SLAM', width: 300, autoPlace: false });
{ const host = document.createElement('div'); host.style.cssText = 'position:fixed;left:12px;top:12px;z-index:5;max-height:96vh;overflow:auto'; host.appendChild(gui.domElement); document.body.appendChild(host); }
gui.add(S, 'pos').listen().disable().name('pos');
gui.add(S, 'play').name('play (SPACE)').listen();
gui.add({ prev: () => loadEpisode(S.episode - 1) }, 'prev').name('◀ prev episode (P)');
gui.add({ next: () => loadEpisode(S.episode + 1) }, 'next').name('next episode ▶ (N)');
gui.add(S, 'speed', 0.1, 4, 0.05).name('speed');
const timeCtl = gui.add(S, 'time', 0, 30, 0.01).name('time (s)').listen().onFinishChange((v) => seek(v, 0.15));
gui.add(S, 'loop').name('loop');
gui.add(S, 'follow').name('follow drone (F)').listen();
const fc = gui.addFolder('Camera / recording');
fc.add(S, 'cine').name('cinematic auto-camera (C)').listen().onChange((v) => { controls.enabled = !v; if (!v) { controls.target.copy(_v.set(...samplePose(DATA.episodes[S.episode], S.time).p)); } });
fc.add({ rec: () => toggleRecord() }, 'rec').name('● record / stop (V)');
fc.add({ run: recordRun }, 'run').name('● record whole run (cinematic)');
const fd = gui.addFolder('Depth camera + mapping FX');
fd.add(S, 'noise', 0, 0.5, 0.01).name('range noise (m)');
fd.add(S, 'pointSize', 0.01, 0.15, 0.005).name('point size').onChange((v) => { cloud.U.uSize.value = v * 30; });
fd.add(S, 'showCloud').name('show live cloud').onChange((v) => { cloud.points.visible = v; });
fd.add(S, 'showScan').name('show scan grid').onChange((v) => { scanLines.visible = v; });
fd.add(S, 'pulse').name('scan pulse').onChange((v) => { pulse.visible = v; });
fd.add(S, 'fovCone').name('FOV cone').onChange((v) => { fov.visible = v; });
fd.add(S, 'flash').name('new-geometry flash').onChange((v) => { FX.slamU.uFlash.value = v ? 1 : 0; });
const fm = gui.addFolder('Mapping scorecard');
for (const [k, n] of [['coverage', 'surface coverage'], ['precision', 'map precision'], ['explored', 'ground explored'], ['drift', 'INS drift now'], ['found', 'survivors located']]) fm.add(S, k).listen().disable().name(n);
fm.add(S, 'useEst').name('map with drifting pose (INS)').onChange(() => seek(S.time, 0.15));
fm.add(S, 'fogOfWar').name('dim unmapped ground').onChange(() => { fowDirty = true; });
fm.add(S, 'showPanels').name('show side panels').onChange((v) => { document.getElementById('right').style.display = v ? 'block' : 'none'; });
fm.add(S, 'mmTruth').name('minimap: show true obstacles');
fm.add({ ply: exportPLY }, 'ply').name('export map (.ply)');
const fs = gui.addFolder('SLAM');
fs.add(S, 'voxels').listen().disable().name('voxels');
fs.add(S, 'showMesh').name('show SLAM mesh').onChange((v) => { slamGroup.visible = v; });
fs.add({ rebuild: () => { for (const k of chunks.keys()) dirty.add(k); } }, 'rebuild').name('rebuild surface now');
fs.add({ reset: () => seek(0) }, 'reset').name('reset (R)');
const fx = gui.addFolder('Look (post-processing, atmosphere)');
if (post) { fx.add(S, 'post').name('post-processing'); fx.add(S, 'bloom', 0, 1.5, 0.01).name('bloom').onChange((v) => { post.bloom.strength = v; }); fx.add(S, 'godrays', 0, 1.5, 0.01).name('god rays').onChange((v) => { post.rays.uniforms.uStr.value = v; });
  fx.add(S, 'vignette', 0, 1, 0.01).name('vignette').onChange((v) => { post.grade.uniforms.uVig.value = v; }); fx.add(S, 'grain', 0, 0.12, 0.002).name('film grain').onChange((v) => { post.grade.uniforms.uGrain.value = v; }); }
fx.add(S, 'fog', 0, 0.06, 0.001).name('fog density').onChange((v) => { scene.fog.density = v; });
fx.add(S, 'shafts').name('sun shafts'); fx.add(S, 'g', 0, 0.95, 0.01).name('sun glow');
fx.add(S, 'dust').name('dust motes').onChange((v) => { dust.points.visible = v; });
fx.add(S, 'spot').name('drone ground light');
fx.add(S, 'shadows').name('shadows').onChange((v) => { renderer.shadowMap.enabled = v; sun.castShadow = v; world && world.group.traverse((o) => { if (o.material) o.material.needsUpdate = true; }); });
fx.add(S, 'droneScale', 1, 5, 0.1).name('drone scale');
fs.close(); fx.close(); fd.close();

addEventListener('keydown', (e) => {
  if (e.code === 'Space') { S.play = !S.play; e.preventDefault(); } else if (e.code === 'KeyR') seek(0);
  else if (e.code === 'KeyN') loadEpisode(S.episode + 1); else if (e.code === 'KeyP') loadEpisode(S.episode - 1);
  else if (e.code === 'KeyF') S.follow = !S.follow;
  else if (e.code === 'KeyC') { S.cine = !S.cine; controls.enabled = !S.cine; } else if (e.code === 'KeyV') toggleRecord();
});
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); post && post.resize(innerWidth, innerHeight); });
addEventListener('dragover', (e) => e.preventDefault());
addEventListener('drop', (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) f.text().then((t) => start(JSON.parse(t))); });

// ------------------------------------------------------------------ main loop
const clock = new THREE.Clock(); let fpsT = 0, fpsN = 0, fps = 0, frameNo = 0, dwS = 0;
const hud = document.getElementById('hud'), _c = new THREE.Color();
const trailPos = trail.line.geometry.attributes.position, trailT = trail.line.geometry.attributes.aT;

function frame() {
  requestAnimationFrame(frame);
  const dt = Math.min(clock.getDelta(), 0.1); wall += dt; frameNo++;
  if (!DATA) return;
  const ep = DATA.episodes[S.episode], tEnd = (ep.frames.length - 1) * DT, prevT = S.time;

  if (S.play) {
    if (S.time < tEnd) S.time = Math.min(tEnd, S.time + dt * S.speed);
    else { doneAt += dt; if (rec && recAuto && doneAt > 2.5) { toggleRecord(); recAuto = false; } if (S.loop && doneAt > 2.0) { seek(0); doneAt = 0; } }
  }
  timeCtl.max(Math.max(tEnd, 0.1));
  const ps = samplePose(ep, S.time); world.curX = ps.p[0];
  if (S.time - lastSense >= SENSE_DT || S.time < lastSense || sensePending) { sense(ps, S.time); lastSense = S.time; sensePending = false; }
  FX.slamU.uNow.value = S.time;
  processDirty(5, S.time);
  S.voxels = voxels.size;
  if (wall - lastMetricT > 0.5) { updateMetrics(); if (fowDirty) updateFogOfWar(); lastMetricT = wall; if (frameNo % 6 < 3) updateOutcome(ep); }
  S.drift = (S.useEst ? Math.hypot(...ps.e) : 0).toFixed(2) + ' m';
  if (S.showPanels && frameNo % 3 === 0) drawPanels(ep, ps);

  // detection events
  if (S.time > prevT && S.time - prevT < 1.0) for (const p of world.pois) if (p.tDet > prevT && p.tDet <= S.time) {
    toast(`${p.type.toUpperCase()} #${p.id + 1} located · t = ${p.tDet.toFixed(1)} s`);
    if (S.cine) { dirSt.poi = p; dirSt.shotT = 0; }
  }
  for (const p of world.pois) { const on = p.tDet <= S.time; p.pin.visible = on; if (on) { const a = S.time - p.tDet, s = 1 + 0.15 * Math.sin(wall * 4) + Math.max(0, 1 - a) * 1.2; p.ring.scale.set(s, s, 1); p.ring.rotation.z = wall; } }
  if (toastT > 0) { toastT -= dt; if (toastT <= 0) document.getElementById('toast').style.opacity = 0; }

  // world follows the drone: terrain patch, ground cover, dust
  if (Math.hypot(ps.p[0] - world.terr.cx, ps.p[1] - world.terr.cy) > 45) rebuildTerrain(Math.round(ps.p[0] / 20) * 20, Math.round(ps.p[1] / 20) * 20);
  updateCover(ps.p[0], ps.p[1]);

  // drone + FX
  drone.position.set(...ps.p); drone.quaternion.set(ps.q[1], ps.q[2], ps.q[3], ps.q[0]); drone.scale.setScalar(S.droneScale);
  const spin = (ps.thrust / 9.81) * 55 * dt; drone.userData.rotors.forEach((b, i) => { b.rotation.z += spin * (i % 2 ? -1 : 1); });
  drone.userData.strobe.visible = Math.sin(wall * 9) > 0.55;
  const gzD = hAt(world.T, ps.p[0], ps.p[1]), agl = ps.p[2] - gzD;
  spot.visible = S.spot; spot.position.set(...ps.p); spot.target.position.set(ps.p[0], ps.p[1], gzD); spot.target.updateMatrixWorld();
  spotDecal.visible = S.spot; spotDecal.position.set(ps.p[0], ps.p[1], gzD + 0.07); const ss = 0.8 + agl * 0.55; spotDecal.scale.set(ss, ss, 1); spotDecal.material.opacity = Math.max(0, 0.7 - agl * 0.05);
  blob.position.set(ps.p[0], ps.p[1], gzD + 0.06); blob.scale.setScalar(1.2 + agl * 0.2); blob.material.opacity = Math.max(0.1, 0.6 - agl * 0.06);
  fov.visible = S.fovCone; fov.position.set(...ps.p); fov.quaternion.copy(drone.quaternion);
  pulse.visible = S.pulse; { const per = 1.8, ph = (S.time % per) / per, R = (DATA.meta.max_range + 2) * ph; pulse.position.set(...ps.p); pulse.scale.setScalar(Math.max(R, 0.01)); pulse.material.uniforms.uP.value = ph; }
  { const i1 = Math.floor(S.time / DT), n0 = Math.max(0, i1 - trail.N + 1), cnt = i1 - n0 + 1;
    for (let k = 0; k < cnt; k++) { const f = ep.frames[n0 + k]; trailPos.setXYZ(k, f[0], f[1], f[2]); trailT.setX(k, k / Math.max(cnt - 1, 1)); }
    trail.line.geometry.setDrawRange(0, cnt); trailPos.needsUpdate = trailT.needsUpdate = true; }
  FX.swayU.uTime.value = wall;
  _v.set(...ps.p);
  world.fires_.update(wall, camera, innerHeight * renderer.getPixelRatio(), _v);
  dust.U.uTime.value = wall; dust.U.uC.value.copy(camera.position); dust.U.uScale.value = innerHeight * renderer.getPixelRatio() * 0.5 / Math.tan(camera.fov * Math.PI / 360);
  world.goal.children[0].rotation.z += dt;
  S.pos = `x=${ps.p[0].toFixed(1).padStart(5)}  y=${ps.p[1].toFixed(1).padStart(5)}  z=${ps.p[2].toFixed(2)} m`;

  // chunk culling
  for (const c of chunks.values()) if (c.lines) { const [ix, iy] = unkey(c.vox[0] || 0), d = Math.hypot((ix + 0.5) * VOX - ps.p[0], (iy + 0.5) * VOX - ps.p[1]); c.lines.visible = c.fill.visible = d < 130; }

  // camera
  if (S.cine) { director(dt, ps, ep); controls.target.copy(dirSt.tgt); }
  else {
    camera.fov += (50 - camera.fov) * Math.min(1, dt * 2); camera.updateProjectionMatrix();
    if (S.follow) { camera.position.add(_v.clone().sub(prevDrone)); controls.target.copy(_v); }
    liftCamera(_v, S.follow); controls.update();
  }
  prevDrone.copy(_v);

  // biome-dependent atmosphere (eased)
  dwS += (disW(ps.p[0]) - dwS) * Math.min(1, dt * 0.8);
  _c.copy(FOG_FOREST).lerp(FOG_RUIN, dwS); scene.fog.color.copy(_c); skyU.hor.value.copy(_c); scene.fog.density = S.fog * (1 + 0.45 * dwS);
  hemi.color.setHex(0xc4dbee).lerp(_c.set(0xd2c2b0), dwS * 0.5); sun.color.setHex(0xfff0d0).lerp(_c.set(0xffd2a0), dwS * 0.4); hemi.intensity = 1.5;

  sun.position.copy(_v).addScaledVector(sunDir, 120); sun.target.position.copy(_v); sun.target.updateMatrixWorld();
  sky.position.copy(camera.position); glow.position.copy(camera.position).addScaledVector(sunDir, 330); glow.material.opacity = 0.25 + 0.75 * S.g;
  for (const b of beams) {
    b.visible = S.shafts; if (!S.shafts) continue;
    const u = b.userData, pos = _x.set(_v.x + u.ox, _v.y + u.oy, _v.z + 6); b.position.copy(pos);
    const toCam = _z.copy(camera.position).sub(b.position).normalize(), xd = new THREE.Vector3().crossVectors(sunDir, toCam).normalize(), zd = new THREE.Vector3().crossVectors(xd, sunDir);
    _m.makeBasis(xd, sunDir.clone(), zd); b.quaternion.setFromRotationMatrix(_m); b.scale.set(u.w, 70, 1); b.position.addScaledVector(sunDir, 25);
    b.material.uniforms.t.value = wall; b.material.uniforms.k.value = 0.05 + 0.15 * S.g;
  }

  if (post && S.post) { post.update(wall, sunDir, dwS); post.composer.render(); } else renderer.render(scene, camera);
  fpsT += dt; fpsN++; if (fpsT > 0.5) { fps = Math.round(fpsN / fpsT); fpsT = 0; fpsN = 0; }
  hud.textContent = `${fps} fps  |  SPACE play/pause · R reset · F follow · C cinematic · V record · mouse=orbit/zoom`;
}

// ------------------------------------------------------------------ boot
function start(data) {
  DATA = data; DT = data.meta.dt; window.__ready = false;
  if (data.episodes[0].scene === 'mission') S.droneScale = 3.0;
  loadEpisode(0);
  if (q0.has('ep')) loadEpisode(+q0.get('ep'));
  if (q0.has('cine')) { S.cine = true; controls.enabled = false; }
  if (q0.has('t')) { S.play = false; seek(+q0.get('t'), 0.1); }
  if (q0.has('noshafts')) S.shafts = false;
  if (q0.has('noshadows')) { S.shadows = false; renderer.shadowMap.enabled = false; sun.castShadow = false; }
  setTimeout(() => { window.__ready = true; }, 600);
}
const which = q0.get('data') || 'mission';
fetch(`data/${which}.json`).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }).catch((e) => (which === 'mission' ? fetch('data/rollout.json').then((r) => r.json()) : Promise.reject(e))).then(start).catch((e) => {
  const m = document.getElementById('msg'); m.style.display = 'flex';
  m.innerHTML = `<div>No flight data loaded (${e.message}).<br>Serve this folder with <code>python -m http.server</code> and run<br><code>python mission.py</code> or <code>python rollout.py --policy heuristic</code> first,<br>or drag a json onto the window.</div>`;
});
frame();
