// Lidar drone flying inside a Gaussian-splat scene. The splat is rendered as-is; the drone's lidar ranges come from the torch sim, which collided
// and ray-marched against an SDF built from the SAME (transformed) splats, so scans land on the visible surfaces.
import * as THREE from 'three';
import { OrbitControls } from '../viewer/vendor/OrbitControls.js';
import GUI from '../viewer/vendor/lil-gui.esm.js';
import * as GS from './vendor/gaussian-splats-3d.module.js';

const S = { play: true, speed: 1, time: 0, episode: 0, scene: '', cam: 'orbit', showSplat: true, showMap: true, mapPose: 'true', showRays: false, showCloud: true, showVoxels: true, voxel: 0.5, voxOpacity: 0.35, groundCut: 0.7, pickAGL: 3, showTrail: true, showPath: true, pointSize: 4, droneScale: 2.0 };
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setSize(innerWidth, innerHeight); document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0a0e14);
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.05, 500); camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
const msg = document.getElementById('msg'), panel = document.getElementById('panel'), heat = document.getElementById('heat'), gH = heat.getContext('2d'), rng = document.getElementById('range'), gR = rng.getContext('2d'), pickEl = document.getElementById('pick');

let D = null, ep = null, splatViewer = null, NR = 80, DT = 0.04, MAXR = 12, built = null;
// ---- drone model (z up, x forward)
const drone = new THREE.Group(), rotors = [];
{ const body = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.12, 0.05), new THREE.MeshBasicMaterial({ color: 0x20262e })); drone.add(body);
  for (const [sx, sy] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) { const arm = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.02, 0.015), new THREE.MeshBasicMaterial({ color: 0x3a4350 })); arm.position.set(sx * 0.1, sy * 0.1, 0); arm.rotation.z = Math.atan2(sy, sx) * 1; arm.rotation.z = (sx * sy > 0) ? Math.PI / 4 : -Math.PI / 4; drone.add(arm);
    const r = new THREE.Mesh(new THREE.CircleGeometry(0.1, 20), new THREE.MeshBasicMaterial({ color: 0xbfe9ff, transparent: true, opacity: 0.28, side: THREE.DoubleSide, depthWrite: false })); r.position.set(sx * 0.2, sy * 0.2, 0.02); drone.add(r); rotors.push(r); }
  const led = (c, x) => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.02, 8, 6), new THREE.MeshBasicMaterial({ color: c })); m.position.set(x, 0, 0.03); drone.add(m); };
  led(0x30ff70, -0.09); led(0xff3040, 0.09);
  const lid = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.03, 12), new THREE.MeshBasicMaterial({ color: 0x59e0ff })); lid.position.set(0, 0, 0.05); drone.add(lid); scene.add(drone); }
// ---- trail, rays, path, goal, map
const TN = 500; const trailG = new THREE.BufferGeometry(); trailG.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TN * 3), 3)); trailG.setAttribute('color', new THREE.BufferAttribute(new Float32Array(TN * 3), 3));
const trail = new THREE.Line(trailG, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); trail.frustumCulled = false; scene.add(trail);
const rayG = new THREE.BufferGeometry(); rayG.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NR * 6), 3));
const rays = new THREE.LineSegments(rayG, new THREE.LineBasicMaterial({ color: 0x59e0ff, transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false })); rays.frustumCulled = false; scene.add(rays);
const pathLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: 0xffd27a, dashSize: 0.3, gapSize: 0.2, transparent: true, opacity: 0.6 })); scene.add(pathLine);
const goal = new THREE.Mesh(new THREE.TorusGeometry(0.5, 0.03, 8, 32), new THREE.MeshBasicMaterial({ color: 0x40ff90 })); scene.add(goal);
const mapMat = new THREE.ShaderMaterial({ uniforms: { uFrame: { value: 0 }, uSize: { value: 4 }, uMax: { value: 12 }, uDt: { value: 0.04 }, uZ0: { value: 0 }, uZ1: { value: 4 } }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexShader: `attribute float aR; attribute float aF; uniform float uFrame, uSize, uMax, uDt, uZ0, uZ1; varying vec3 vC; varying float vA;
    void main(){ float age=(uFrame-aF)*uDt; bool off = aF>uFrame || aR>=uMax-0.08; vec4 mv=modelViewMatrix*vec4(position,1.0); gl_Position = off ? vec4(2.0,2.0,2.0,1.0) : projectionMatrix*mv;
      float h=clamp((position.z-uZ0)/(uZ1-uZ0),0.0,1.0); vec3 c=mix(vec3(0.1,0.5,1.0),vec3(1.0,0.85,0.25),h); float fl=exp(-max(age,0.0)*3.0); vC=c*0.9+vec3(0.6,1.0,1.0)*fl*1.4; vA=0.85;
      gl_PointSize = off ? 0.0 : uSize*(1.0+1.2*fl); }`,
  fragmentShader: `varying vec3 vC; varying float vA; void main(){ vec2 d=gl_PointCoord-0.5; float a=smoothstep(0.5,0.1,length(d)); gl_FragColor=vec4(vC*a, a*vA); }` });
let mapPts = null;

// turbo colormap (Google), t in 0..1 -> [r,g,b] 0..1
function turbo(t) { t = Math.min(Math.max(t, 0), 1); const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.54893924 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const bl = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973)))); return [Math.min(Math.max(r, 0), 1), Math.min(Math.max(g, 0), 1), Math.min(Math.max(bl, 0), 1)]; }

let SRC = null, vox = null, ZMAX = 10;
const voxGeo = new THREE.BoxGeometry(1, 1, 1), voxMat = new THREE.MeshLambertMaterial({ transparent: true, opacity: 0.35, depthWrite: false });
scene.add(new THREE.AmbientLight(0xffffff, 0.75)); { const dl = new THREE.DirectionalLight(0xffffff, 1.4); dl.position.set(0.4, -0.6, 1); scene.add(dl); }

// scans -> world points for both pose models. src = {n scans, nr rays, dirs (nr*3), r (n*nr, metres), fr (frame idx per scan), maxR}
function makeSrc() {
  const L = D.lidar, lid = ep.lidar_scan, F = ep.frames;
  if (L && lid && lid.length) { const el = L.el_deg.map((d) => d * Math.PI / 180), nz = el.length, nr = L.az * nz, dirs = new Float32Array(nr * 3);
    for (let ia = 0; ia < L.az; ia++) for (let ie = 0; ie < nz; ie++) { const a = ia / L.az * 2 * Math.PI, k = (ia * nz + ie) * 3; dirs[k] = Math.cos(el[ie]) * Math.cos(a); dirs[k + 1] = Math.cos(el[ie]) * Math.sin(a); dirs[k + 2] = Math.sin(el[ie]); }
    return { n: lid.length, nr, dirs, r: lid.map((s) => s.map((v) => v / 100)), fr: lid.map((_, k) => k * L.stride), maxR: L.max_range, dense: true, az: L.az, nel: nz }; }
  const nr = D.ray_dirs.length, dirs = new Float32Array(nr * 3); D.ray_dirs.forEach((d, k) => { dirs[k * 3] = d[0]; dirs[k * 3 + 1] = d[1]; dirs[k * 3 + 2] = d[2]; });
  return { n: F.length, nr, dirs, r: ep.scan.map((s) => s.map((v) => v / 100)), fr: F.map((_, i) => i), maxR: D.max_range, dense: false };
}
function worldPts(pose) {            // pose 0 = true, 1 = dead-reckoned
  const { n, nr, dirs, r, fr, maxR } = SRC, F = ep.frames, P = new Float32Array(n * nr * 3), H = new Uint8Array(n * nr);
  for (let s = 0; s < n; s++) { const i = fr[s], f = F[i], b = pose ? ep.est[i] : [0, 0, 0], yaw = f[8], cy = Math.cos(yaw), sy = Math.sin(yaw), ox = f[0] + b[0], oy = f[1] + b[1], oz = f[2] + b[2];
    for (let k = 0; k < nr; k++) { const t = r[s][k], d = k * 3, o = (s * nr + k) * 3; P[o] = ox + t * (cy * dirs[d] - sy * dirs[d + 1]); P[o + 1] = oy + t * (sy * dirs[d] + cy * dirs[d + 1]); P[o + 2] = oz + t * dirs[d + 2]; H[s * nr + k] = t < maxR - 0.08 ? 1 : 0; } }
  return { P, H };
}
let WP = null;
function build() {
  ep = D.episodes[S.episode]; if (!ep) { msg.textContent = 'no successful flights in this rollout file'; return; }
  SRC = makeSrc(); WP = [worldPts(0), worldPts(1)]; const { n, nr, fr, r } = SRC;
  const aR = new Float32Array(n * nr), aF = new Float32Array(n * nr); for (let s = 0; s < n; s++) for (let k = 0; k < nr; k++) { aR[s * nr + k] = r[s][k]; aF[s * nr + k] = fr[s]; }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(WP[0].P, 3)); g.setAttribute('aR', new THREE.BufferAttribute(aR, 1)); g.setAttribute('aF', new THREE.BufferAttribute(aF, 1));
  if (mapPts) { scene.remove(mapPts); mapPts.geometry.dispose(); } mapPts = new THREE.Points(g, mapMat); mapPts.frustumCulled = false; scene.add(mapPts); built = { g };
  mapMat.uniforms.uMax.value = SRC.maxR; mapMat.uniforms.uDt.value = DT;
  const pp = ep.path.map((p) => new THREE.Vector3(p[0], p[1], p[2])); pathLine.geometry.setFromPoints(pp); pathLine.computeLineDistances(); goal.position.set(...ep.goal);
  ZMAX = Math.max(D.scene.hi[2] * 0.7, 4); mapMat.uniforms.uZ0.value = 0; mapMat.uniforms.uZ1.value = ZMAX; buildHeatBase();
  S.time = 0; tcCtl.max((F_len() - 1) * DT); setMapPose(); buildVox(); hm = null;
}
const F_len = () => ep.frames.length;
function setMapPose() { if (!built) return; built.g.setAttribute('position', new THREE.BufferAttribute(WP[S.mapPose === 'true' ? 0 : 1].P, 3)); buildVox(); hm = null; }

// ---- voxel mesh: one cube per occupied cell of the accumulated lidar hits, born when first hit
function buildVox() {
  if (!WP) return; if (vox) { scene.remove(vox); vox.dispose(); vox = null; }
  const { P, H } = WP[S.mapPose === 'true' ? 0 : 1], { n, nr, fr } = SRC, v = S.voxel, cells = new Map();
  for (let s = 0; s < n; s++) for (let k = 0; k < nr; k++) { if (!H[s * nr + k]) continue; const o = (s * nr + k) * 3; if (P[o + 2] < S.groundCut) continue; const ix = Math.floor(P[o] / v), iy = Math.floor(P[o + 1] / v), iz = Math.floor(P[o + 2] / v), key = (ix + 4096) * 16777216 + (iy + 4096) * 4096 + (iz + 4096);
    if (!cells.has(key)) { if (cells.size > 220000) continue; cells.set(key, [ix, iy, iz, fr[s]]); } }
  const arr = [...cells.values()].sort((a, b) => a[3] - b[3]); vox = new THREE.InstancedMesh(voxGeo, voxMat, Math.max(arr.length, 1)); vox.frustumCulled = false; const M = new THREE.Matrix4(), C = new THREE.Color();
  arr.forEach((c, k) => { M.compose(new THREE.Vector3((c[0] + 0.5) * v, (c[1] + 0.5) * v, (c[2] + 0.5) * v), new THREE.Quaternion(), new THREE.Vector3(v * 0.96, v * 0.96, v * 0.96)); vox.setMatrixAt(k, M); const t = turbo(((c[2] + 0.5) * v) / ZMAX); C.setRGB(t[0], t[1], t[2]); vox.setColorAt(k, C); });
  vox.userData.births = arr.map((c) => c[3]); vox.count = 0; scene.add(vox);
}
function voxCount(fl) { const b = vox.userData.births; let lo = 0, hi = b.length; while (lo < hi) { const m = (lo + hi) >> 1; if (b[m] <= fl) lo = m + 1; else hi = m; } return lo; }

// ---- heat maps
let hm = null, hmK = 0, baseImg = null; const hmCv = document.createElement('canvas'), hmG = hmCv.getContext('2d');
function buildHeatBase() { const H = D.heightmap; if (!H) return; hmCv.width = H.nx; hmCv.height = H.ny; baseImg = new ImageData(H.nx, H.ny); hm = null; }
function hmCell(x, y) { const H = D.heightmap, i = Math.floor((x - H.x0) / H.cell), j = Math.floor((y - H.y0) / H.cell); return (i < 0 || j < 0 || i >= H.nx || j >= H.ny) ? -1 : i * H.ny + j; }
const hmGeom = () => { const H = D.heightmap, W = heat.width, Hh = heat.height, s = Math.min((W - 8) / (H.nx * H.cell), (Hh - 8) / (H.ny * H.cell)); return { H, s, X: (x) => 4 + (x - H.x0) * s, Y: (y) => Hh - 4 - (y - H.y0) * s }; };
function drawHeat(fl) {
  if (!D.heightmap || !baseImg) return; const k = heat.width / 600, { H, s, X, Y } = hmGeom(), pose = S.mapPose === 'true' ? 0 : 1, { P, H: HIT } = WP[pose], { n, nr, fr } = SRC;
  if (!hm || hm.fl > fl) hm = { z: new Float32Array(H.nx * H.ny).fill(-1), k: 0, fl: 0 }; hm.fl = fl;
  while (hm.k < n && fr[hm.k] <= fl) { const sc = hm.k++; for (let k = 0; k < nr; k++) { if (!HIT[sc * nr + k]) continue; const o = (sc * nr + k) * 3, c = hmCell(P[o], P[o + 1]); if (c >= 0 && P[o + 2] > hm.z[c]) hm.z[c] = P[o + 2]; } }
  const d = baseImg.data; for (let i = 0; i < H.nx; i++) for (let j = 0; j < H.ny; j++) { const c = i * H.ny + j, o = ((H.ny - 1 - j) * H.nx + i) * 4, zt = H.z[c] / 100, zm = hm.z[c];
    if (zm >= 0) { const t = turbo(zm / ZMAX); d[o] = t[0] * 255; d[o + 1] = t[1] * 255; d[o + 2] = t[2] * 255; d[o + 3] = 255; } else { const g = 18 + 45 * Math.min(zt / ZMAX, 1); d[o] = g; d[o + 1] = g + 4; d[o + 2] = g + 12; d[o + 3] = 255; } }
  hmG.putImageData(baseImg, 0, 0); gH.fillStyle = '#080c12'; gH.fillRect(0, 0, heat.width, heat.height); gH.imageSmoothingEnabled = false; gH.drawImage(hmCv, 4, heat.height - 4 - H.ny * H.cell * s, H.nx * H.cell * s, H.ny * H.cell * s);
  gH.strokeStyle = 'rgba(255,210,122,.8)'; gH.lineWidth = 2 * k; gH.beginPath(); ep.path.forEach((p, i) => (i ? gH.lineTo(X(p[0]), Y(p[1])) : gH.moveTo(X(p[0]), Y(p[1])))); gH.stroke();
  const mk = (p, col, rad) => { gH.fillStyle = col; gH.beginPath(); gH.arc(X(p[0]), Y(p[1]), rad, 0, 7); gH.fill(); gH.strokeStyle = '#000'; gH.lineWidth = 1.5; gH.stroke(); }; mk(ep.start, '#ffffff', 7 * k); mk(ep.goal, '#40ff90', 8 * k);
  if (PL.route) { gH.strokeStyle = '#59e0ff'; gH.lineWidth = 2 * k; gH.beginPath(); PL.route.path.forEach((p, i) => (i ? gH.lineTo(X(p[0]), Y(p[1])) : gH.moveTo(X(p[0]), Y(p[1])))); gH.stroke(); } if (PL.st) mk(PL.st, '#ffd27a', 7 * k); if (PL.go) mk(PL.go, '#ff6ad5', 8 * k);
  const f = ep.frames[Math.min(Math.floor(fl), ep.frames.length - 1)], yaw = f[8]; const LR = D.lidar ? D.lidar.max_range : 12; gH.strokeStyle = 'rgba(255,255,255,.35)'; gH.lineWidth = 1.5 * k; gH.setLineDash([6 * k, 6 * k]); gH.beginPath(); gH.arc(X(f[0]), Y(f[1]), LR * s, 0, 7); gH.stroke(); gH.setLineDash([]);
  { let sc = 0; while (sc + 1 < n && fr[sc + 1] <= fl) sc++; gH.fillStyle = '#fff'; for (let q = 0; q < nr; q++) { if (!HIT[sc * nr + q]) continue; const o = (sc * nr + q) * 3; gH.fillRect(X(P[o]) - 1.5 * k, Y(P[o + 1]) - 1.5 * k, 3 * k, 3 * k); } }
  gH.save(); gH.translate(X(f[0]), Y(f[1])); gH.rotate(-yaw); gH.scale(k, k); gH.fillStyle = '#fff'; gH.strokeStyle = '#000'; gH.beginPath(); gH.moveTo(13, 0); gH.lineTo(-8, 8); gH.lineTo(-4, 0); gH.lineTo(-8, -8); gH.closePath(); gH.fill(); gH.stroke(); gH.restore();
  gH.fillStyle = '#cfe3f5'; gH.font = (22 * k) + 'px ui-monospace,monospace'; gH.fillText('LIDAR MAP · height', 10 * k, 24 * k);
  for (let j = 0; j < 60; j++) { const t = turbo(j / 59); gH.fillStyle = `rgb(${t[0] * 255 | 0},${t[1] * 255 | 0},${t[2] * 255 | 0})`; gH.fillRect((12 + j * 3) * k, heat.height - 22 * k, 3 * k + 1, 10 * k); } gH.fillStyle = '#cfe3f5'; gH.font = (16 * k) + 'px ui-monospace,monospace'; gH.fillText('0 m', 12 * k, heat.height - 26 * k); gH.fillText(ZMAX.toFixed(0) + ' m', 160 * k, heat.height - 26 * k);
}
// click = pick a start (click) / end (shift+click) point on the heat map; z = local surface + pickAGL
const picks = { s: null, g: null };
heat.addEventListener('click', (e) => { if (!D.heightmap) return; const { H, s } = hmGeom(), r = heat.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * heat.width, py = (e.clientY - r.top) / r.height * heat.height, x = H.x0 + (px - 4) / s, y = H.y0 + (heat.height - 4 - py) / s;
  if (hmCell(x, y) < 0) return; setPoint(e.shiftKey ? 'goal' : 'start', x, y); picks.s = PL.st; picks.g = PL.go; hm = null; });
function drawRange(fl) {
  if (!SRC || !SRC.dense) { gR.clearRect(0, 0, rng.width, rng.height); gR.fillStyle = 'rgba(8,12,18,.8)'; gR.fillRect(0, 0, rng.width, rng.height); gR.fillStyle = '#8fa6bb'; gR.font = '20px ui-monospace,monospace'; gR.fillText('re-run run_splat.py for the 360° range image', 14, 70); return; }
  const { r, az, nel, maxR, fr } = SRC; let s = 0; while (s + 1 < SRC.n && fr[s + 1] <= fl) s++; const img = new ImageData(az, nel);
  for (let ia = 0; ia < az; ia++) for (let ie = 0; ie < nel; ie++) { const v = r[s][ia * nel + ie], col = (ia + az / 2) % az, row = nel - 1 - ie, o = (row * az + col) * 4;
    if (v >= maxR - 0.08) { img.data[o] = 14; img.data[o + 1] = 18; img.data[o + 2] = 26; } else { const t = turbo(1 - Math.min(v / 14, 1)); img.data[o] = t[0] * 255; img.data[o + 1] = t[1] * 255; img.data[o + 2] = t[2] * 255; } img.data[o + 3] = 255; }
  const c2 = rngCv; c2.width = az; c2.height = nel; c2g.putImageData(img, 0, 0); gR.fillStyle = '#080c12'; gR.fillRect(0, 0, rng.width, rng.height); gR.imageSmoothingEnabled = true; gR.imageSmoothingQuality = 'high'; gR.drawImage(c2, 0, 30, rng.width, rng.height - 30);
  gR.fillStyle = '#cfe3f5'; gR.font = '20px ui-monospace,monospace'; gR.fillText('RANGE IMAGE 360° (ahead = centre) · hot = near', 10, 22);
}
const rngCv = document.createElement('canvas'), c2g = rngCv.getContext('2d');

function sizeHeat() { const asp = D && D.heightmap ? D.heightmap.ny / D.heightmap.nx : 1, wmax = { small: 300, large: 480, huge: 760 }[S.mapSize], hmax = innerHeight - (S.mapSize === 'huge' ? 80 : 190); let w = wmax, h = w * asp; if (h > hmax) { h = hmax; w = h / asp; } w = Math.round(w); h = Math.round(h);
  heat.style.width = w + 'px'; heat.style.height = h + 'px'; heat.width = w * 2; heat.height = h * 2; pickEl.style.bottom = (h + 20) + 'px'; pickEl.style.width = Math.min(w, 420) + 'px'; hm = null; }
addEventListener('resize', sizeHeat);

// ---- cameras
const sm = { p: new THREE.Vector3(), t: new THREE.Vector3(), init: false }, tp = new THREE.Vector3(), tt = new THREE.Vector3(), fwd = new THREE.Vector3(), prevD = new THREE.Vector3(); let shot = 0, shotT = 0;
addEventListener('wheel', (e) => { if (S.cam === 'birdseye') S.birdH = Math.min(200, Math.max(8, S.birdH * Math.exp(e.deltaY * 0.001))); }, { passive: true });
function placeCamera(dt, dp, yaw) {
  const bird = S.cam === 'birdseye' || S.cam === 'overview'; camera.up.set(bird ? 0 : 0, bird ? 1 : 0, bird ? 0 : 1);
  if (S.cam === 'orbit') { controls.enabled = true; controls.update(); return; } controls.enabled = false; fwd.set(Math.cos(yaw), Math.sin(yaw), 0);
  let mode = S.cam; if (mode === 'cinematic') { shotT += dt; if (shotT > 7) { shotT = 0; shot = (shot + 1) % 3; } mode = ['chase', 'side', 'high'][shot]; }
  if (mode === 'birdseye') { tp.set(dp.x, dp.y, dp.z + S.birdH); tt.copy(dp); }
  else if (mode === 'overview') { const lo = D.scene.lo, hi = D.scene.hi, span = Math.max(hi[0] - lo[0], (hi[1] - lo[1]) * camera.aspect / 1.0); tp.set((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, span * 0.62 + 8); tt.set(tp.x, tp.y, 0); }
  else if (mode === 'follow') { tp.copy(dp).add(new THREE.Vector3(-3.2, -3.2, 2.2)); tt.copy(dp); }
  else if (mode === 'chase') { tp.copy(dp).addScaledVector(fwd, -2.2).add(new THREE.Vector3(0, 0, 0.9)); tt.copy(dp).addScaledVector(fwd, 3); }
  else if (mode === 'side') { tp.copy(dp).add(new THREE.Vector3(-fwd.y * 3.2, fwd.x * 3.2, 0.6)).addScaledVector(fwd, 0.8); tt.copy(dp).addScaledVector(fwd, 1.2); }
  else { tp.copy(dp).add(new THREE.Vector3(-1.5, -1.5, 8)); tt.copy(dp); }
  const k = sm.init ? 1 - Math.exp(-dt * 3) : 1; sm.init = true; sm.p.lerp(tp, k); sm.t.lerp(tt, k); camera.position.copy(sm.p); camera.lookAt(sm.t);
}

// ---- recording
let mr = null, chunks = [];
function rec() { if (mr) { mr.stop(); return; } mr = new MediaRecorder(renderer.domElement.captureStream(30), { mimeType: 'video/webm;codecs=vp9', videoBitsPerSecond: 10e6 }); chunks = []; mr.ondataavailable = (e) => chunks.push(e.data);
  mr.onstop = () => { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob(chunks, { type: 'video/webm' })); a.download = 'drone_in_splat.webm'; a.click(); mr = null; document.getElementById('rec').style.display = 'none'; }; mr.start(); document.getElementById('rec').style.display = 'block'; }

// ---- GUI
const CAMS = ['orbit', 'follow', 'chase', 'side', 'birdseye', 'overview', 'cinematic']; S.birdH = 45; S.show = 'splat + lidar'; S.mapSize = 'large';
const gui = new GUI({ title: 'Lidar drone' }); gui.add(S, 'play').listen(); const tcCtl = gui.add(S, 'time', 0, 1, 0.04).name('time (s)').listen();
const fView = gui.addFolder('View'); fView.add(S, 'cam', CAMS).name('camera (C)').listen(); fView.add(S, 'show', ['splat + lidar', 'splat only', 'lidar only']).name('show (L)').listen(); let epCtl = null;
const fPlan = gui.addFolder('Plan a flight'), fMap = gui.addFolder('Map & lidar').close(), fMore = gui.addFolder('More').close();
fMap.add(S, 'mapSize', { small: 'small', large: 'large', huge: 'huge' }).name('lidar map size (M)').onChange(sizeHeat); fMap.add(S, 'showCloud').name('point cloud'); fMap.add(S, 'showVoxels').name('voxel mesh'); fMap.add(S, 'voxel', 0.25, 2, 0.05).name('voxel size (m)').onFinishChange(buildVox); fMap.add(S, 'voxOpacity', 0.1, 1, 0.05).name('voxel opacity'); fMap.add(S, 'groundCut', 0, 3, 0.1).name('hide voxels below (m)').onFinishChange(buildVox);
fMap.add(S, 'mapPose', { 'true pose (corrected)': 'true', 'dead-reckoned (drifting)': 'est' }).name('map built from').onChange(setMapPose); 
fMore.add(S, 'speed', 0.1, 4, 0.1).name('playback speed'); fMore.add(S, 'birdH', 8, 200, 1).name('bird\'s-eye height (m)'); fMore.add(S, 'showRays').name('scan rays'); fMore.add(S, 'showTrail').name('trail'); fMore.add(S, 'showPath').name('planned path'); fMore.add(S, 'pointSize', 1, 10, 0.5).name('point size'); fMore.add(S, 'droneScale', 1, 6, 0.1).name('drone scale'); fMore.add({ rec }, 'rec').name('record on/off (V)');
addEventListener('keydown', (e) => { if (e.code === 'Space') { S.play = !S.play; e.preventDefault(); } else if (e.code === 'KeyR') S.time = 0; else if (e.code === 'KeyV') rec(); else if (e.code === 'KeyL') { const l = ['splat + lidar', 'splat only', 'lidar only']; S.show = l[(l.indexOf(S.show) + 1) % 3]; } else if (e.code === 'KeyM') { S.mapSize = { small: 'large', large: 'huge', huge: 'small' }[S.mapSize]; sizeHeat(); gui.controllersRecursive().forEach((c) => c.updateDisplay()); } else if (e.code === 'KeyC') { const l = CAMS; S.cam = l[(l.indexOf(S.cam) + 1) % l.length]; gui.controllersRecursive().forEach((c) => c.updateDisplay()); } });
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });

// ---- loop
const clock = new THREE.Clock(), q = new THREE.Quaternion(), dp = new THREE.Vector3(); let fN = 0;
function frame() {
  requestAnimationFrame(frame); const dt = Math.min(clock.getDelta(), 0.1); if (!ep) return;
  const F = ep.frames, tEnd = (F.length - 1) * DT; if (S.play) { S.time += dt * S.speed; if (S.time > tEnd + 1.5) { S.time = 0; } }
  const tt_ = Math.min(Math.max(S.time, 0), tEnd), fl = tt_ / DT, i = Math.min(Math.floor(fl), F.length - 1), j = Math.min(i + 1, F.length - 1), a = fl - i, A = F[i], B = F[j], L = (o) => A[o] + (B[o] - A[o]) * a;
  dp.set(L(0), L(1), L(2)); drone.position.copy(dp); q.set(A[4], A[5], A[6], A[3]).slerp(new THREE.Quaternion(B[4], B[5], B[6], B[3]), a); drone.quaternion.copy(q); drone.scale.setScalar(S.droneScale);
  rotors.forEach((r, k) => { r.rotation.z += dt * 40 * (k % 2 ? 1 : -1); });
  mapMat.uniforms.uFrame.value = fl; const sh = S.show; if (splatViewer) splatViewer.visible = sh !== 'lidar only'; mapPts.visible = S.showMap && S.showCloud && sh !== 'splat only'; if (vox) { vox.count = voxCount(fl); vox.visible = S.showVoxels && sh !== 'splat only'; voxMat.opacity = S.voxOpacity; } mapMat.uniforms.uSize.value = S.pointSize * renderer.getPixelRatio();
  // rays
  const R = rayG.attributes.position.array, dirs = D.ray_dirs, yaw = A[8], cy = Math.cos(yaw), sy = Math.sin(yaw);
  for (let r = 0; r < NR; r++) { const t = ep.scan[i][r] / 100, d = dirs[r]; R[r * 6] = dp.x; R[r * 6 + 1] = dp.y; R[r * 6 + 2] = dp.z; R[r * 6 + 3] = dp.x + t * (cy * d[0] - sy * d[1]); R[r * 6 + 4] = dp.y + t * (sy * d[0] + cy * d[1]); R[r * 6 + 5] = dp.z + t * d[2]; }
  rayG.attributes.position.needsUpdate = true; rays.visible = S.showRays;
  // trail
  const n0 = Math.max(0, i - TN + 1), cnt = i - n0 + 1, TP = trailG.attributes.position.array, TC = trailG.attributes.color.array; for (let k = 0; k < cnt; k++) { const f = F[n0 + k], w = Math.pow(k / Math.max(cnt - 1, 1), 1.5); TP[k * 3] = f[0]; TP[k * 3 + 1] = f[1]; TP[k * 3 + 2] = f[2]; TC[k * 3] = 1.2 * w; TC[k * 3 + 1] = 0.8 * w; TC[k * 3 + 2] = 0.25 * w; }
  trailG.setDrawRange(0, cnt); trailG.attributes.position.needsUpdate = trailG.attributes.color.needsUpdate = true; trail.visible = S.showTrail; pathLine.visible = S.showPath; goal.rotation.z += dt; goal.scale.setScalar(1 + 0.08 * Math.sin(performance.now() / 300));
  placeCamera(dt, dp, yaw); if (S.cam === 'orbit' && fN === 0) { controls.target.copy(dp); }
  const minR = Math.min(...ep.scan[i]) / 100, bias = ep.est[i], dr = Math.hypot(...bias), sp = Math.hypot(B[0] - A[0], B[1] - A[1], B[2] - A[2]) / DT;
  panel.innerHTML = `flight <b>${S.episode + 1}/${D.episodes.length}</b> · t <b>${tt_.toFixed(1)}</b> s · <b>${sp.toFixed(1)}</b> m/s · alt <b>${A[2].toFixed(1)}</b> m<br>nearest <b>${minR.toFixed(1)}</b> m · drift <b style="color:#ff7a8a">${dr.toFixed(2)}</b> m`;
  if (fN++ % 2 === 0) { drawHeat(fl); drawRange(fl); }
  renderer.render(scene, camera);
}

// ---- load

// ---------------- interactive flight planner (needs plan_server.py; otherwise falls back to printing the command)
const PL = { edit: 'off', startH: 3, goalH: 3, margin: 1.2, ceiling: 0, server: false, busy: false, st: null, go: null, route: null }, plMsg = document.createElement('div');
plMsg.className = 'card'; plMsg.style.cssText = 'position:fixed;left:50%;transform:translateX(-50%);top:12px;max-width:560px;padding:6px 10px;color:#dff9ff;font:12px ui-monospace,monospace;display:none;text-align:center'; document.body.appendChild(plMsg);
const say = (t, col) => { plMsg.style.display = t ? 'block' : 'none'; plMsg.innerHTML = t; plMsg.style.borderLeft = '3px solid ' + (col || '#59e0ff'); };
const mkMarker = (col) => { const g = new THREE.Group(), s = new THREE.Mesh(new THREE.SphereGeometry(0.55, 20, 14), new THREE.MeshBasicMaterial({ color: col })), ring = new THREE.Mesh(new THREE.TorusGeometry(0.9, 0.05, 8, 32), new THREE.MeshBasicMaterial({ color: col })),
  pole = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 0, -1)]), new THREE.LineBasicMaterial({ color: col })); g.add(s, ring, pole); g.userData = { pole, ring }; g.visible = false; scene.add(g); return g; };
const mkS = mkMarker(0xffd27a), mkG = mkMarker(0xff6ad5);
const routeLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x59e0ff })); routeLine.visible = false; routeLine.renderOrder = 5; scene.add(routeLine);
const topAt = (x, y) => { const H = D.heightmap; if (!H) return 0; const i = Math.floor((x - H.x0) / H.cell), j = Math.floor((y - H.y0) / H.cell); return (i < 0 || j < 0 || i >= H.nx || j >= H.ny) ? NaN : H.z[i * H.ny + j] / 100; };
function placeMarker(mk, p) { if (!p) { mk.visible = false; return; } mk.visible = true; mk.position.set(p[0], p[1], p[2]); const g = topAt(p[0], p[1]); const h = Math.max(p[2] - (isNaN(g) ? 0 : g), 0.1); mk.userData.pole.scale.set(1, 1, h); mk.userData.ring.rotation.set(0, 0, 0); }
function refreshMarkers() { placeMarker(mkS, PL.st); placeMarker(mkG, PL.go); if (!PL.st || !PL.go) { routeLine.visible = false; } }
function setPoint(which, x, y) { const g = topAt(x, y); if (isNaN(g)) return; const p = [+x.toFixed(1), +y.toFixed(1), +(g + (which === 'start' ? PL.startH : PL.goalH)).toFixed(1)]; if (which === 'start') PL.st = p; else PL.go = p; refreshMarkers(); schedulePreview(); showCmd(); }
function reheight() { if (PL.st) PL.st[2] = +(topAt(PL.st[0], PL.st[1]) + PL.startH).toFixed(1); if (PL.go) PL.go[2] = +(topAt(PL.go[0], PL.go[1]) + PL.goalH).toFixed(1); refreshMarkers(); schedulePreview(); showCmd(); }
function showCmd() { const sx = PL.st ? `--start ${PL.st.join(' ')}` : '', gx = PL.go ? ` --goal ${PL.go.join(' ')}` : ''; pickEl.style.display = 'block'; pickEl.innerHTML = `${sx}${gx || ' <i style="color:#8fa6bb">(set the end point)</i>'}<br><span style="color:#8fa6bb">z = surface + start ${PL.startH} m / goal ${PL.goalH} m</span>`; }
let pvT = null; function schedulePreview() { clearTimeout(pvT); if (PL.server && PL.st && PL.go) pvT = setTimeout(preview, 250); else if (!PL.server && PL.st && PL.go) say('no planner server: run <b>python plan_server.py</b> to preview and fly here, or paste the command shown bottom-left', '#ffd27a'); }
const body = () => ({ scene: S.scene, start: PL.st, goal: PL.go, margin: PL.margin, ceiling: PL.ceiling, radius: D.radius || 0.35 });
async function api(path, b) { const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) }); return r.json(); }
async function preview() { if (PL.busy) return; say('planning…'); try { const r = await api('/api/plan', body()); if (!r.ok) { routeLine.visible = false; PL.route = null; say('✗ ' + r.error, '#ff6a6a'); return; }
  PL.route = r; routeLine.geometry.setFromPoints(r.path.map((p) => new THREE.Vector3(...p))); routeLine.visible = true; if (r.start) { PL.st = r.start; PL.go = r.goal; refreshMarkers(); showCmd(); }
  say(`✓ collision-free route ${r.length} m` + (r.msgs.length ? '<br>' + r.msgs.join('<br>') : '') + '<br>press <b>Fly it</b> (F) to simulate it', '#40ff90'); } catch (e) { say('planner server not reachable', '#ff6a6a'); } }
async function flyIt() { if (PL.busy || !PL.st || !PL.go) return; PL.busy = true; say('flying the full simulation… (10–60 s)'); try { const r = await api('/api/fly', body()); PL.busy = false;
  if (!r.ok) { say('✗ ' + r.error, '#ff6a6a'); return; } D.episodes.push(r.episode); S.episode = D.episodes.length - 1; epCtl.options(Object.fromEntries(D.episodes.map((_, k) => ['flight ' + (k + 1), k]))); epCtl.updateDisplay(); build(); S.time = 0; S.play = true;
  say(`${r.outcome === 'success' ? '✓ flight succeeded' : '✗ ' + r.outcome} · min clearance ${r.episode.min_clearance} m · nearest splat ${r.episode.min_splat_dist} m (${r.wall}s)`, r.outcome === 'success' ? '#40ff90' : '#ff6a6a'); routeLine.visible = false; } catch (e) { PL.busy = false; say('fly failed: ' + e.message, '#ff6a6a'); } }
// click in the 3D view: march the camera ray down onto the heightfield
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(); let down = null;
renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => { if (!down || PL.edit === 'off' || !D.heightmap) return; if (Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
  ndc.set(e.clientX / innerWidth * 2 - 1, -(e.clientY / innerHeight) * 2 + 1); ray.setFromCamera(ndc, camera); const o = ray.ray.origin, d = ray.ray.direction;
  for (let t = 0.5; t < 600; t += 0.25) { const x = o.x + d.x * t, y = o.y + d.y * t, z = o.z + d.z * t, g = topAt(x, y); if (!isNaN(g) && z <= g + 0.05) { setPoint(PL.edit, x, y); if (PL.edit === 'start') { PL.edit = 'goal'; plCtl.updateDisplay(); } return; } } });
let plCtl = null;
function setupPlanner() {
  const f = fPlan; plCtl = f.add(PL, 'edit', ['off', 'start', 'goal']).name('click scene to place'); f.add(PL, 'startH', 0.5, 60, 0.5).name('start height (m)').onChange(reheight); f.add(PL, 'goalH', 0.5, 60, 0.5).name('goal height (m)').onChange(reheight);
  f.add({ fly: flyIt }, 'fly').name('▶ Fly it (F)'); f.add({ clr: () => { PL.st = PL.go = null; PL.route = null; routeLine.visible = false; refreshMarkers(); say(''); pickEl.style.display = 'none'; PL.edit = 'start'; plCtl.updateDisplay(); } }, 'clr').name('clear points'); const adv = f.addFolder('route limits').close(); adv.add(PL, 'ceiling', 0, 80, 1).name('max altitude (0 = auto)').onFinishChange(schedulePreview); adv.add(PL, 'margin', 0.6, 3, 0.1).name('keep-out margin (m)').onFinishChange(schedulePreview);
  fetch('/api/ping').then((r) => r.json()).then(() => { PL.server = true; say('planner ready: choose <b>click scene to place → start</b>, click the scene, then the goal. Q/E = start/goal height ±0.5 m', '#40ff90'); setTimeout(() => { if (!PL.st) say(''); }, 9000); }).catch(() => { PL.server = false; });
  addEventListener('keydown', (e) => { if (e.code === 'KeyF') flyIt(); else if (e.code === 'KeyQ' && PL.st) { PL.startH = Math.max(0.5, PL.startH + (e.shiftKey ? -0.5 : 0.5)); reheight(); f.controllersRecursive().forEach((c) => c.updateDisplay()); } else if (e.code === 'KeyE' && PL.go) { PL.goalH = Math.max(0.5, PL.goalH + (e.shiftKey ? -0.5 : 0.5)); reheight(); f.controllersRecursive().forEach((c) => c.updateDisplay()); } });
}

async function main() {
  let idx = []; try { idx = await (await fetch('data/index.json?t=' + Date.now(), { cache: 'no-store' })).json(); } catch (e) { /* none */ }
  const q = new URLSearchParams(location.search).get('scene'); S.scene = q || (idx[0] && idx[0].id) || 'demo'; const ent = idx.find((e) => e.id === S.scene);
  D = await (await fetch((ent ? ent.file : `data/${S.scene}_rollout.json`) + '?t=' + Date.now(), { cache: 'no-store' })).json(); NR = D.ray_dirs.length; DT = D.dt; MAXR = D.max_range;
  rayG.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NR * 6), 3));
  if (idx.length > 1) fView.add(S, 'scene', Object.fromEntries(idx.map((e) => [e.name, e.id]))).onChange((v) => { location.search = '?scene=' + v; });
  epCtl = fView.add(S, 'episode', Object.fromEntries(D.episodes.map((_, k) => ['flight ' + (k + 1), k]))).name('flight').onChange(() => { build(); });
  splatViewer = new GS.DropInViewer({ gpuAcceleratedSort: false, sharedMemoryForWorkers: false, sceneRevealMode: GS.SceneRevealMode.Instant, antialiased: false, logLevel: GS.LogLevel.None });
  await splatViewer.addSplatScene(D.scene.splat + '?t=' + Date.now(), { format: GS.SceneFormat.Splat, showLoadingUI: false, splatAlphaRemovalThreshold: 5 }); scene.add(splatViewer);
  setupPlanner(); sizeHeat(); build(); const lo = D.scene.lo, hi = D.scene.hi, c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2], span = Math.max(hi[0] - lo[0], hi[1] - lo[1]);
  camera.position.set(c[0] - span * 0.45, c[1] - span * 0.55, hi[2] * 0.9 + 4); controls.target.set(...ep.start); controls.update(); msg.style.display = 'none'; window.__ready = true;
}
window.__S = S; window.__PL = PL; window.__dbg = () => ({ inst: scene.children.filter((c) => c.isInstancedMesh).length, pts: scene.children.filter((c) => c.isPoints).length, voxN: vox ? vox.userData.births.length : 0, ep: S.episode, nF: ep.frames.length }); window.__rebuild = () => build(); window.__cam = () => ({ p: camera.position.toArray() }); main().catch((e) => { msg.textContent = 'failed: ' + e.message; console.error(e); }); frame();
