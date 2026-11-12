// Asteroid SLAM viewer v2.
// The spacecraft orbits a rotating asteroid in an inertial frame; the surface is "revealed" (rock detail, flash) where the lidar map has reached it,
// and the reconstructed shape model is drawn as a holographic mesh coloured by its error against the true shape.
import * as THREE from 'three';
import { OrbitControls } from '../viewer/vendor/OrbitControls.js';
import GUI from '../viewer/vendor/lil-gui.esm.js';
import { makePost } from '../viewer/post.js';

const S = { play: true, speed: 400, time: 0, view: 'both', post: true, bloom: 0.6, cine: false, follow: false, showScan: true, showEst: true, showDR: true, labels: true, kind: '' };
const KM = 1000, NLAT = 45, NLON = 90, FOVR = 30 * Math.PI / 180, NR = 21;
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap; document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x000000);
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 1, 60000); camera.up.set(0, 0, 1); camera.position.set(1400, -1300, 700);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.maxDistance = 12000;
const post = makePost(renderer, scene, camera); post.rays.uniforms.uStr.value = 0.5; post.rays.uniforms.uThr.value = 1.5; post.grade.uniforms.uVig.value = 0.5; post.grade.uniforms.uCA.value = 0.002;
post.grade.uniforms.uSat.value = 1.05;
const sunDir = new THREE.Vector3(1, -0.4, 0.5).normalize();

// ---- lighting + sky
const sun = new THREE.DirectionalLight(0xfff1dc, 5.0); sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.6;
sun.shadow.camera.near = 1000; sun.shadow.camera.far = 6000; scene.add(sun, sun.target);
scene.add(new THREE.HemisphereLight(0x3a4a70, 0x08080c, 0.28));
const fill = new THREE.DirectionalLight(0x5a78c0, 0.18); fill.position.copy(sunDir).multiplyScalar(-1); scene.add(fill);
const sunDisc = new THREE.Mesh(new THREE.SphereGeometry(260, 24, 12), new THREE.MeshBasicMaterial({ color: new THREE.Color(40, 34, 24), fog: false })); scene.add(sunDisc);
{ const N = 5200, p = new Float32Array(N * 3), c = new Float32Array(N * 3); let s = 11; const r = () => (s = (s * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < N; i++) {
    let a = r() * 6.2832, z = r() * 2 - 1;
    if (i > 2600) { a = r() * 6.2832; z = (r() + r() + r() - 1.5) * 0.28; const t = 0.35; const x0 = Math.cos(a), y0 = Math.sin(a) * Math.cos(t) - z * Math.sin(t), z0 = Math.sin(a) * Math.sin(t) + z * Math.cos(t); const q = Math.hypot(x0, y0, z0); p.set([x0 / q * 30000, y0 / q * 30000, z0 / q * 30000], i * 3); }
    else { const q = Math.sqrt(1 - z * z); p.set([30000 * q * Math.cos(a), 30000 * q * Math.sin(a), 30000 * z], i * 3); }
    const m = Math.pow(r(), 3) * (i > 2600 ? 0.5 : 1) + 0.12, tint = r(); c.set([m * (0.8 + 0.2 * tint), m * 0.9, m * (1.1 - 0.3 * tint)], i * 3); }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(p, 3)); g.setAttribute('color', new THREE.BufferAttribute(c, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ size: 1.8, sizeAttenuation: false, vertexColors: true, fog: false, depthWrite: false }))); }

// ---- world state
let D = null, AST = null, F = null, OMEGA = 0, birthH = null, obsTex = null, BM = null, bumpArr = null;
const body = new THREE.Group(); scene.add(body);
const uT = { value: 0 }, uReveal = { value: 1 }, uObsTex = { value: null };
function setAst(A) {
  AST = A; const n = A.a.length; bumpArr = { n, cx: new Float64Array(n), cy: new Float64Array(n), cz: new Float64Array(n), a: Float64Array.from(A.a), k: Float64Array.from(A.k) };
  A.c.forEach((c, i) => { bumpArr.cx[i] = c[0]; bumpArr.cy[i] = c[1]; bumpArr.cz[i] = c[2]; });
}
function radius(ux, uy, uz) {                                      // identical to asteroid/shape.py (km)
  const s = AST.s, e = 1 / Math.sqrt((ux / s[0]) ** 2 + (uy / s[1]) ** 2 + (uz / s[2]) ** 2); let b = 0; const B = bumpArr;
  for (let i = 0; i < B.n; i++) b += B.a[i] * Math.exp(B.k[i] * (ux * B.cx[i] + uy * B.cy[i] + uz * B.cz[i] - 1));
  const w = 1 - AST.pinch[0] * Math.exp(-((ux / AST.pinch[1]) ** 2)), g = 1 + AST.ridge[0] * Math.exp(-((uz / AST.ridge[1]) ** 2));
  return AST.R0 * e * w * g * Math.max(1 + b, 0.55);
}
const fField = (x, y, z) => { const r = Math.hypot(x, y, z) || 1e-9; return r - radius(x / r, y / r, z / r); };

// ---- rocky shader (shared by the asteroid and the loose boulders): procedural albedo + bump, reveal mask, flash, graticule
const GLSL = `
float hash31(vec3 p){ p = fract(p*0.3183099+0.1); p *= 17.0; return fract(p.x*p.y*p.z*(p.x+p.y+p.z)); }
float vnoise(vec3 x){ vec3 i=floor(x), f=fract(x); f=f*f*(3.0-2.0*f);
  return mix(mix(mix(hash31(i),hash31(i+vec3(1,0,0)),f.x),mix(hash31(i+vec3(0,1,0)),hash31(i+vec3(1,1,0)),f.x),f.y),
             mix(mix(hash31(i+vec3(0,0,1)),hash31(i+vec3(1,0,1)),f.x),mix(hash31(i+vec3(0,1,1)),hash31(i+vec3(1,1,1)),f.x),f.y),f.z); }
float fbm(vec3 p){ float a=0.5,s=0.0; for(int i=0;i<5;i++){ s+=a*vnoise(p); p=p*2.03+vec3(1.7,9.2,3.1); a*=0.5; } return s; }
float cells(vec3 p){ vec3 i=floor(p), f=fract(p); float d=1.0; for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++)for(int z=-1;z<=1;z++){ vec3 g=vec3(float(x),float(y),float(z)); vec3 o=vec3(hash31(i+g),hash31(i+g+7.1),hash31(i+g+13.7)); d=min(d,length(g+o-f)); } return d; }
vec3 perturbN(vec3 sp, vec3 N, vec2 dH, float fd){ vec3 sx=dFdx(sp), sy=dFdy(sp); vec3 R1=cross(sy,N), R2=cross(N,sx); float det=dot(sx,R1)*fd; vec3 g=sign(det)*(dH.x*R1+dH.y*R2); return normalize(abs(det)*N-g); }
`;
function rockify(mat, instanced) {
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uT = uT; sh.uniforms.uReveal = uReveal; sh.uniforms.tObs = uObsTex;
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vObj;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvObj = ' + (instanced ? '(instanceMatrix * vec4(position,1.0)).xyz;' : 'position;'));
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', `#include <common>\nvarying vec3 vObj; uniform float uT, uReveal; uniform sampler2D tObs;\n${GLSL}
      float gSeen, gFlash, gH;`)
      .replace('#include <color_fragment>', `#include <color_fragment>
      { vec3 u = normalize(vObj); float lat = asin(clamp(u.z,-1.0,1.0)), lon = atan(u.y,u.x);
        vec4 o = texture2D(tObs, vec2((lon+3.14159265)/6.2831853, (lat+1.5707963)/3.14159265)); float birth = o.g / max(o.r, 1e-3);
        gSeen = mix(1.0, smoothstep(0.35, 0.65, o.r) * step(birth, uT), uReveal); float age = uT - birth; gFlash = (o.r > 0.4 && age > 0.0 && uReveal > 0.5) ? exp(-age * 5.0) : 0.0;
        float broad = fbm(vObj*0.025), mid = fbm(vObj*0.11), cl = 1.0 - smoothstep(0.0, 0.8, cells(vObj*0.12));
        gH = (0.9*mid + 0.3*cl + 0.1*fbm(vObj*0.5) + 0.6*broad) * 1.0;
        vec3 rock = mix(vec3(0.16,0.145,0.13), vec3(0.115,0.13,0.16), smoothstep(0.3,0.7,broad)) * (0.55 + 1.0*mid) * (0.85 + 0.2*cl);
        diffuseColor.rgb = mix(vec3(0.05,0.065,0.09), rock, gSeen); }`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
      normal = perturbN(-vViewPosition, normal, vec2(dFdx(gH), dFdy(gH)) * (2.2 * gSeen), faceDirection);`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
      { vec3 u = normalize(vObj); float lat = asin(clamp(u.z,-1.0,1.0)) * 57.29578, lon = atan(u.y,u.x) * 57.29578; vec2 q = vec2(lat, lon) / 15.0;
        vec2 w = fwidth(q) + 1e-4; vec2 gl = 1.0 - smoothstep(vec2(0.0), w * 1.2, abs(fract(q + 0.5) - 0.5)); float grid = max(gl.x, gl.y) * (1.0 - gSeen) * 0.32 * smoothstep(0.5, 0.0, max(w.x, w.y));
        totalEmissiveRadiance += vec3(0.1, 0.85, 1.0) * (grid + gFlash * 1.6); }`);
  };
  mat.customProgramCacheKey = () => 'rock' + instanced;
  return mat;
}

// ---- build everything that depends on the asteroid
let truthMesh, rocks, recon, reconWire, reconGeo, fpPts, fpLine, features = [], cent = [];
function disposeTree(o) { o.traverse((c) => { c.geometry && c.geometry.dispose(); }); }
function build() {
  while (body.children.length) { const c = body.children[0]; disposeTree(c); body.remove(c); }
  cent = []; for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) { const la = -Math.PI / 2 + (i + 0.5) * Math.PI / NLAT, lo = -Math.PI + (j + 0.5) * 2 * Math.PI / NLON; cent.push([Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo), Math.sin(la)]); }
  const R = AST.Req * KM; sun.shadow.camera.left = sun.shadow.camera.bottom = -R * 1.35; sun.shadow.camera.right = sun.shadow.camera.top = R * 1.35; sun.shadow.camera.updateProjectionMatrix();
  // observation texture: R = observed, G = birth hour (premultiplied), linear filtered
  const dat = new Uint16Array(NLAT * NLON * 4), H = THREE.DataUtils.toHalfFloat; birthH = D.birth;
  for (let k = 0; k < NLAT * NLON; k++) { const b = birthH[k], ob = b >= 0 ? 1 : 0; dat[k * 4] = H(ob); dat[k * 4 + 1] = H(ob ? b : 0); dat[k * 4 + 2] = 0; dat[k * 4 + 3] = H(1); }
  obsTex = new THREE.DataTexture(dat, NLON, NLAT, THREE.RGBAFormat, THREE.HalfFloatType); obsTex.magFilter = obsTex.minFilter = THREE.LinearFilter; obsTex.wrapS = THREE.RepeatWrapping; obsTex.wrapT = THREE.ClampToEdgeWrapping; obsTex.needsUpdate = true; uObsTex.value = obsTex;
  // truth mesh with analytic normals (no seams)
  const NT = 120, MT = 240, pos = new Float32Array((NT + 1) * MT * 3), nor = new Float32Array((NT + 1) * MT * 3), idx = [], e = 0.0004;
  for (let i = 0; i <= NT; i++) for (let j = 0; j < MT; j++) {
    const la = -Math.PI / 2 + i * Math.PI / NT, lo = -Math.PI + j * 2 * Math.PI / MT, ux = Math.cos(la) * Math.cos(lo), uy = Math.cos(la) * Math.sin(lo), uz = Math.sin(la), rk = radius(ux, uy, uz), x = ux * rk, y = uy * rk, z = uz * rk, k = (i * MT + j) * 3;
    pos[k] = x * KM; pos[k + 1] = y * KM; pos[k + 2] = z * KM; const gx = fField(x + e, y, z) - fField(x - e, y, z), gy = fField(x, y + e, z) - fField(x, y - e, z), gz = fField(x, y, z + e) - fField(x, y, z - e), gl = Math.hypot(gx, gy, gz) || 1;
    nor[k] = gx / gl; nor[k + 1] = gy / gl; nor[k + 2] = gz / gl; }
  for (let i = 0; i < NT; i++) for (let j = 0; j < MT; j++) { const a = i * MT + j, b = i * MT + (j + 1) % MT, c = (i + 1) * MT + j, d = (i + 1) * MT + (j + 1) % MT; idx.push(a, b, c, b, d, c); }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('normal', new THREE.BufferAttribute(nor, 3)); g.setIndex(idx);
  truthMesh = new THREE.Mesh(g, rockify(new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, metalness: 0 }), false)); truthMesh.castShadow = truthMesh.receiveShadow = true; body.add(truthMesh);
  // loose boulders
  let sd = 7; const rnd = () => (sd = (sd * 16807) % 2147483647) / 2147483647, NRK = 650, ig = new THREE.IcosahedronGeometry(1, 2), pa = ig.attributes.position;
  for (let i = 0; i < pa.count; i++) { const x = pa.getX(i), y = pa.getY(i), z = pa.getZ(i), n = 1 + 0.28 * Math.sin(x * 4.1 + y * 2.3) * Math.cos(z * 3.7 + 1.0) + 0.12 * Math.sin(7 * y + 3 * z); pa.setXYZ(i, x * n, y * n, z * n); } ig.computeVertexNormals();
  rocks = new THREE.InstancedMesh(ig, rockify(new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1 }), true), NRK); const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), eu = new THREE.Euler(), sc = new THREE.Vector3(), ps = new THREE.Vector3();
  const bo = AST.features.filter((f) => f.kind === 'boulder');
  for (let i = 0; i < NRK; i++) {
    let u; if (bo.length && i < NRK * 0.3) { const b = bo[i % bo.length].u; u = [b[0] + (rnd() - 0.5) * 0.35, b[1] + (rnd() - 0.5) * 0.35, b[2] + (rnd() - 0.5) * 0.35]; } else u = [rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1];
    const l = Math.hypot(...u) || 1; u = u.map((v) => v / l); const s = 1.1 + 6.5 * Math.pow(rnd(), 3.2), r = radius(u[0], u[1], u[2]) * KM - 0.3 * s;
    ps.set(u[0] * r, u[1] * r, u[2] * r); q.setFromEuler(eu.set(rnd() * 6.28, rnd() * 6.28, rnd() * 6.28)); sc.set(s, s * (0.65 + 0.35 * rnd()), s * (0.7 + 0.3 * rnd())); m4.compose(ps, q, sc); rocks.setMatrixAt(i, m4); }
  rocks.castShadow = rocks.receiveShadow = true; body.add(rocks);
  // reconstruction mesh: persistent buffers, per-snapshot positions/colours/index; per-vertex birth for the flash
  const NV = NLAT * NLON + 2; reconGeo = new THREE.BufferGeometry();
  reconGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NV * 3), 3)); reconGeo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(NV * 3), 3));
  reconGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NV * 3), 3)); const ab = new Float32Array(NV); for (let k = 0; k < NLAT * NLON; k++) ab[k] = birthH[k] >= 0 ? birthH[k] : 1e9; ab[NV - 2] = ab[NV - 1] = 0;
  reconGeo.setAttribute('aBorn', new THREE.BufferAttribute(ab, 1)); reconGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(NLAT * NLON * 6 + NLON * 6), 1)); reconGeo.setDrawRange(0, 0);
  const wireGeo = new THREE.BufferGeometry(); for (const k of ['position', 'color', 'aBorn']) wireGeo.setAttribute(k, reconGeo.attributes[k]); wireGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(NLAT * NLON * 4 + NLON * 4), 1)); wireGeo.setDrawRange(0, 0);
  const uni = { uT, uSunV: { value: new THREE.Vector3() } };
  const vs = 'attribute float aBorn; attribute vec3 color; varying vec3 vC; varying float vB; varying vec3 vN; varying vec3 vV; void main(){ vC=color; vB=aBorn; vN=normalize(normalMatrix*normal); vec4 mv=modelViewMatrix*vec4(position,1.0); vV=-mv.xyz; gl_Position=projectionMatrix*mv; }';
  recon = new THREE.Mesh(reconGeo, new THREE.ShaderMaterial({ uniforms: uni, vertexShader: vs, side: THREE.DoubleSide, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
    fragmentShader: 'uniform float uT; uniform vec3 uSunV; varying vec3 vC; varying float vB; varying vec3 vN; varying vec3 vV; void main(){ float age=uT-vB; if(age<0.0) discard; vec3 N=normalize(vN)*(gl_FrontFacing?1.0:-1.0); vec3 V=normalize(vV); float ndl=max(dot(N,uSunV),0.0); float rim=pow(1.0-abs(dot(N,V)),2.5); float fl=exp(-age*5.0);' +
    ' gl_FragColor=vec4(vC*(0.15+0.35*ndl)+vC*rim*0.7+vec3(0.2,0.9,1.0)*fl*3.0, 0.07+0.32*rim+0.6*fl); }' }));
  recon.scale.setScalar(1.004);
  reconWire = new THREE.LineSegments(wireGeo, new THREE.ShaderMaterial({ uniforms: uni, vertexShader: 'attribute float aBorn; attribute vec3 color; varying vec3 vC; varying float vB; void main(){ vC=color; vB=aBorn; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }', transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    fragmentShader: 'uniform float uT; varying vec3 vC; varying float vB; void main(){ float age=uT-vB; if(age<0.0) discard; float fl=exp(-age*5.0); gl_FragColor=vec4(vC*0.2+vec3(0.3,1.0,1.0)*fl*2.0, 0.4); }' })); reconWire.scale.setScalar(1.004);
  body.add(recon, reconWire);
  // lidar footprint: the 21x21 beam hits on the surface, plus an outline
  const fg = new THREE.BufferGeometry(); fg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NR * NR * 3), 3)); fg.setAttribute('aOn', new THREE.BufferAttribute(new Float32Array(NR * NR), 1)); fg.setAttribute('aIdx', new THREE.BufferAttribute(Float32Array.from({ length: NR * NR }, (_, i) => i), 1));
  fpPts = new THREE.Points(fg, new THREE.ShaderMaterial({ uniforms: { uTm: { value: 0 } }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: 'attribute float aOn; attribute float aIdx; uniform float uTm; varying float vA; void main(){ float row=floor(aIdx/21.0); vA=aOn*(0.55+0.45*sin(row*0.6-uTm*9.0)); gl_PointSize=4.5; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
    fragmentShader: 'varying float vA; void main(){ vec2 d=gl_PointCoord-0.5; float a=smoothstep(0.5,0.0,length(d))*vA; gl_FragColor=vec4(vec3(0.3,1.5,2.0)*a, a); }' })); fpPts.frustumCulled = false;
  const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array((NR * 4 - 3) * 3), 3));
  fpLine = new THREE.Line(lg, new THREE.LineBasicMaterial({ color: new THREE.Color(0.4, 2.0, 2.6), transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false })); fpLine.frustumCulled = false; body.add(fpPts, fpLine);
  // feature pins
  const box = document.getElementById('labels'); box.innerHTML = ''; features = AST.features.map((f) => { const r = radius(...f.u) * KM, el = document.createElement('div'); el.className = 'pin'; el.textContent = f.name; el.style.opacity = 0; box.appendChild(el);
    const i = Math.min(NLAT - 1, Math.floor((Math.asin(f.u[2]) + Math.PI / 2) / (Math.PI / NLAT))), j = Math.floor(((Math.atan2(f.u[1], f.u[0]) + Math.PI) / (2 * Math.PI)) * NLON) % NLON;
    return { ...f, p: new THREE.Vector3(f.u[0] * r, f.u[1] * r, f.u[2] * r), birth: birthH[i * NLON + j], el, said: false }; });
}
let lastSnap = -1;
function updateRecon(si) {
  const snap = D.snaps[si], P = reconGeo.attributes.position.array, C = reconGeo.attributes.color.array, tmp = new THREE.Color(), idx = reconGeo.index.array, widx = reconWire.geometry.index.array; let ni = 0, nw = 0;
  for (let k = 0; k < NLAT * NLON; k++) { const r = snap[k] || 0, c = cent[k]; P[k * 3] = c[0] * r; P[k * 3 + 1] = c[1] * r; P[k * 3 + 2] = c[2] * r;
    if (r) { const err = Math.min(Math.abs(r / KM - radius(c[0], c[1], c[2])) * 1000 / 8, 1); tmp.setRGB(0.15 + 1.0 * err, 0.8 - 0.55 * err, 1.1 - 0.9 * err); C[k * 3] = tmp.r; C[k * 3 + 1] = tmp.g; C[k * 3 + 2] = tmp.b; } }
  for (let i = 0; i < NLAT - 1; i++) for (let j = 0; j < NLON; j++) { const j2 = (j + 1) % NLON, a = i * NLON + j, b = i * NLON + j2, c = (i + 1) * NLON + j, d = (i + 1) * NLON + j2;
    if (snap[a] && snap[b] && snap[c] && snap[d]) { idx[ni++] = a; idx[ni++] = b; idx[ni++] = c; idx[ni++] = b; idx[ni++] = d; idx[ni++] = c; widx[nw++] = a; widx[nw++] = b; widx[nw++] = a; widx[nw++] = c; } }
  for (const [ring, pole, ord] of [[0, NLAT * NLON, 1], [NLAT - 1, NLAT * NLON + 1, -1]]) {       // polar fans when the whole ring is mapped
    let all = true, mean = 0; for (let j = 0; j < NLON; j++) { const r = snap[ring * NLON + j]; if (!r) all = false; mean += r / NLON; }
    P[pole * 3] = 0; P[pole * 3 + 1] = 0; P[pole * 3 + 2] = (ring ? 1 : -1) * mean; C[pole * 3] = 0.2; C[pole * 3 + 1] = 0.8; C[pole * 3 + 2] = 1;
    if (all) for (let j = 0; j < NLON; j++) { const a = ring * NLON + j, b = ring * NLON + (j + 1) % NLON; if (ord > 0) { idx[ni++] = pole; idx[ni++] = a; idx[ni++] = b; } else { idx[ni++] = pole; idx[ni++] = b; idx[ni++] = a; } } }
  reconGeo.attributes.position.needsUpdate = reconGeo.attributes.color.needsUpdate = true; reconGeo.index.needsUpdate = true; reconWire.geometry.index.needsUpdate = true; reconGeo.setDrawRange(0, ni); reconWire.geometry.setDrawRange(0, nw);
  reconGeo.computeVertexNormals(); lastSnap = si;
}

// ---- spacecraft (local +Z = nadir, wings along X)
function gridTex() { const c = document.createElement('canvas'); c.width = c.height = 128; const g = c.getContext('2d'); g.fillStyle = '#12285f'; g.fillRect(0, 0, 128, 128); g.strokeStyle = '#6f8fd8'; g.lineWidth = 2; for (let i = 0; i <= 128; i += 32) { g.beginPath(); g.moveTo(i, 0); g.lineTo(i, 128); g.moveTo(0, i); g.lineTo(128, i); g.stroke(); } const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t; }
const craft = new THREE.Group(), pivot = new THREE.Group(); craft.up.set(0, 0, 1);
{ const foil = new THREE.MeshStandardMaterial({ color: 0xc9a24a, metalness: 0.9, roughness: 0.35 }), white = new THREE.MeshStandardMaterial({ color: 0xe8e8ee, metalness: 0.4, roughness: 0.5 }), dark = new THREE.MeshStandardMaterial({ color: 0x222833, metalness: 0.6, roughness: 0.4 });
  const panel = new THREE.MeshStandardMaterial({ map: gridTex(), metalness: 0.6, roughness: 0.3, emissive: 0x0a1a40, emissiveIntensity: 0.6 });
  craft.add(new THREE.Mesh(new THREE.BoxGeometry(22, 16, 18), foil));
  const head = new THREE.Mesh(new THREE.CylinderGeometry(5, 6.5, 6, 20), dark); head.rotation.x = Math.PI / 2; head.position.z = 12; craft.add(head);
  const lens = new THREE.Mesh(new THREE.CircleGeometry(3.6, 20), new THREE.MeshBasicMaterial({ color: new THREE.Color(0.3, 1.8, 2.4) })); lens.position.z = 15.05; craft.add(lens);
  const dish = new THREE.Mesh(new THREE.SphereGeometry(9, 24, 10, 0, 6.2832, 0, Math.PI / 2), white); dish.rotation.x = -Math.PI / 2; dish.position.set(0, 0, -14); dish.material.side = THREE.DoubleSide; craft.add(dish);
  const boom = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 8, 8), dark); boom.rotation.x = Math.PI / 2; boom.position.z = -11; craft.add(boom);
  for (const sx of [-1, 1]) { const arm = new THREE.Mesh(new THREE.BoxGeometry(14, 1, 1), dark); arm.position.x = sx * 18; pivot.add(arm);
    for (const k of [0, 1]) { const w = new THREE.Mesh(new THREE.BoxGeometry(24, 16, 0.5), panel); w.position.set(sx * (31 + 25 * k), 0, 0); pivot.add(w); } }
  craft.add(pivot);
  for (const [x, c] of [[-9, [2.5, 0.2, 0.2]], [9, [0.2, 2.5, 0.2]]]) { const l = new THREE.Mesh(new THREE.SphereGeometry(1.1, 8, 6), new THREE.MeshBasicMaterial({ color: new THREE.Color(...c) })); l.position.set(x, 8.4, 6); craft.add(l); l.userData.led = true; }
  scene.add(craft); }
const ghost = (col) => { const m = new THREE.Mesh(new THREE.OctahedronGeometry(10), new THREE.MeshBasicMaterial({ color: col, wireframe: true, transparent: true, opacity: 0.9 })); scene.add(m); return m; };
const gEst = ghost(new THREE.Color(2.2, 0.4, 1.8)), gDR = ghost(new THREE.Color(2.4, 0.3, 0.3));
const TN = 420; function trail(col) { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TN * 3), 3)); g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(TN * 3), 3));
  const l = new THREE.Line(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); l.frustumCulled = false; l.userData.col = col; scene.add(l); return l; }
const trTrue = trail([1.6, 0.9, 0.25]), trEst = trail([1.4, 0.3, 1.2]), trDR = trail([1.6, 0.2, 0.2]);
function setTrail(l, F, f, o) { const P = l.geometry.attributes.position.array, C = l.geometry.attributes.color.array, n0 = Math.max(0, f - TN + 1), n = f - n0 + 1;
  for (let i = 0; i < n; i++) { const r = F[n0 + i], k = i * 3, a = Math.pow(i / Math.max(n - 1, 1), 1.6); P[k] = r[o] * KM; P[k + 1] = r[o + 1] * KM; P[k + 2] = r[o + 2] * KM; C[k] = l.userData.col[0] * a; C[k + 1] = l.userData.col[1] * a; C[k + 2] = l.userData.col[2] * a; }
  l.geometry.setDrawRange(0, n); l.geometry.attributes.position.needsUpdate = l.geometry.attributes.color.needsUpdate = true; }
const cone = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: new THREE.Color(0.3, 1.2, 1.6), transparent: true, opacity: 0.3, blending: THREE.AdditiveBlending, depthWrite: false })); scene.add(cone);
const rings = [0, 1, 2].map(() => { const l = new THREE.LineLoop(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: new THREE.Color(0.4, 1.6, 2.2), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); scene.add(l); return l; });

// ---- lidar footprint on the surface (same geometry as asteroid/sim.py: lidar_dirs + ray march against the true field)
const gr = Array.from({ length: NR }, (_, i) => Math.tan(-FOVR / 2 + FOVR * i / (NR - 1)));
const tmpP = new THREE.Vector3(), tmpB = new THREE.Vector3(), e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), dd = new THREE.Vector3(), dpos = new THREE.Vector3();
function footprint(pb) {                                              // pb: spacecraft position in body frame (km, THREE.Vector3)
  const b = tmpB.copy(pb).negate().normalize(); e1.set(0, 0, 1).cross(b); if (e1.length() < 1e-3) e1.set(1, 0, 0).cross(b); e1.normalize(); e2.copy(b).cross(e1);
  const P = fpPts.geometry.attributes.position.array, On = fpPts.geometry.attributes.aOn.array, Rb = AST.Req * 1.7 * 1.2, L = fpLine.geometry.attributes.position.array; let li = 0; const hit = new Float32Array(NR * NR * 3), ok = new Uint8Array(NR * NR);
  const o = pb, ob = o.dot(o);
  for (let i = 0; i < NR; i++) for (let j = 0; j < NR; j++) {
    dd.copy(b).addScaledVector(e1, gr[j]).addScaledVector(e2, gr[i]).normalize(); const bb = dd.dot(o), disc = bb * bb - (ob - Rb * Rb), k = i * NR + j; ok[k] = 0; On[k] = 0;
    if (disc <= 0) continue; const sq = Math.sqrt(disc), t0 = Math.max(-bb - sq, 0), t1 = -bb + sq, M = 30; let lo = t0, hi = -1, prev = t0;
    for (let s = 0; s <= M; s++) { const t = t0 + (t1 - t0) * s / M; if (fField(o.x + t * dd.x, o.y + t * dd.y, o.z + t * dd.z) < 0) { hi = t; lo = prev; break; } prev = t; }
    if (hi < 0) continue; for (let s = 0; s < 7; s++) { const m = 0.5 * (lo + hi); if (fField(o.x + m * dd.x, o.y + m * dd.y, o.z + m * dd.z) < 0) hi = m; else lo = m; }
    const t = 0.5 * (lo + hi); hit[k * 3] = (o.x + t * dd.x) * KM; hit[k * 3 + 1] = (o.y + t * dd.y) * KM; hit[k * 3 + 2] = (o.z + t * dd.z) * KM; ok[k] = 1; On[k] = 1; P.set(hit.subarray(k * 3, k * 3 + 3), k * 3); }
  const border = []; for (let j = 0; j < NR; j++) border.push(j); for (let i = 1; i < NR; i++) border.push(i * NR + NR - 1); for (let j = NR - 2; j >= 0; j--) border.push((NR - 1) * NR + j); for (let i = NR - 2; i >= 1; i--) border.push(i * NR);
  let all = true; for (const k of border) if (!ok[k]) all = false; fpLine.visible = all && S.showScan; if (all) for (let n = 0; n < Math.min(border.length, NR * 4 - 4); n++) { const k = border[n]; L[n * 3] = hit[k * 3]; L[n * 3 + 1] = hit[k * 3 + 1]; L[n * 3 + 2] = hit[k * 3 + 2]; li = n; }
  if (all) { L.set(L.subarray(0, 3), (li + 1) * 3); fpLine.geometry.setDrawRange(0, li + 2); fpLine.geometry.attributes.position.needsUpdate = true; } fpPts.geometry.attributes.position.needsUpdate = fpPts.geometry.attributes.aOn.needsUpdate = true; fpPts.visible = S.showScan;
}

// ---- 2D HUD: ground-track map, error plot, mission strip
const cMap = document.getElementById('map'), gM = cMap.getContext('2d'), cPlot = document.getElementById('plot'), gP = cPlot.getContext('2d'), cStrip = document.getElementById('strip'), gS = cStrip.getContext('2d'), panel = document.getElementById('panel');
const off = document.createElement('canvas'); off.width = NLON; off.height = NLAT; const gO = off.getContext('2d'), img = gO.createImageData(NLON, NLAT);
const bodyFrame = (r, theta) => { const c = Math.cos(theta), s = Math.sin(theta); return [c * r[0] + s * r[1], -s * r[0] + c * r[1], r[2]]; };
const thetaAt = (t) => 0.3 + OMEGA * t;
function drawMap(t, f) {
  const W = 300, H = 150, th = t / 3600; for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) { const b = birthH[i * NLON + j], k = ((NLAT - 1 - i) * NLON + j) * 4, seen = b >= 0 && b <= th, fl = seen ? Math.exp(-(th - b) * 5) : 0;
    img.data[k] = 12 + 220 * fl; img.data[k + 1] = seen ? 70 + 130 * (1 - Math.min(1, fl)) + 120 * fl : 14; img.data[k + 2] = seen ? 120 + 100 * (1 - Math.min(1, fl)) + 100 * fl : 26; img.data[k + 3] = 255; }
  gO.putImageData(img, 0, 0); gM.imageSmoothingEnabled = true; gM.drawImage(off, 0, 0, W, H); gM.fillStyle = 'rgba(8,12,18,.35)'; gM.fillRect(0, 0, W, H);
  const X = (lon) => (lon + Math.PI) / (2 * Math.PI) * W, Y = (lat) => H / 2 - lat / Math.PI * H;
  gM.strokeStyle = 'rgba(255,255,255,.12)'; gM.lineWidth = 1; for (let a = -60; a <= 60; a += 30) { gM.beginPath(); gM.moveTo(0, Y(a * Math.PI / 180)); gM.lineTo(W, Y(a * Math.PI / 180)); gM.stroke(); }
  for (const ft of features) { if (!(ft.birth >= 0 && ft.birth <= th)) continue; gM.fillStyle = ft.kind === 'crater' ? '#ffd27a' : '#b6ffb0'; gM.beginPath(); gM.arc(X(Math.atan2(ft.u[1], ft.u[0])), Y(Math.asin(ft.u[2])), 2.6, 0, 7); gM.fill(); }
  gM.strokeStyle = 'rgba(255,170,70,.85)'; gM.beginPath(); let pl = null; for (let i = Math.max(0, f - 260); i <= f; i += 2) { const q = bodyFrame(F[i].slice(1, 4), thetaAt(F[i][0])), lat = Math.asin(q[2] / Math.hypot(...q)), lon = Math.atan2(q[1], q[0]);
    if (pl !== null && Math.abs(lon - pl) > Math.PI) gM.moveTo(X(lon), Y(lat)); else if (pl === null) gM.moveTo(X(lon), Y(lat)); else gM.lineTo(X(lon), Y(lat)); pl = lon; } gM.stroke();
  const q = bodyFrame(F[f].slice(1, 4), thetaAt(F[f][0])); gM.fillStyle = '#fff'; gM.beginPath(); gM.arc(X(Math.atan2(q[1], q[0])), Y(Math.asin(q[2] / Math.hypot(...q))), 3, 0, 7); gM.fill();
  gM.fillStyle = '#8fa6bb'; gM.font = '10px ui-monospace,monospace'; gM.fillText('mapped surface (lat/lon) · ground track', 6, 11);
}
function drawPlot(t) {
  const W = 300, H = 150; gP.clearRect(0, 0, W, H); gP.fillStyle = 'rgba(8,12,18,0.72)'; gP.fillRect(0, 0, W, H);
  const tmax = D.compare['active/slam'].slice(-1)[0][0], X = (h) => 6 + (W - 12) * h / tmax, Y = (v, m) => H - 18 - (H - 34) * Math.min(v / m, 1), col = { perfect: '#3fd16b', drift: '#ff5a6e', slam: '#4d8cff' };
  for (const [k, cv] of Object.entries(D.compare)) { const [pol, mode] = k.split('/'); gP.beginPath(); gP.setLineDash(pol === 'active' ? [] : [4, 3]); cv.forEach((c, i) => { const x = X(c[0]), y = Y(c[2], 30); i ? gP.lineTo(x, y) : gP.moveTo(x, y); }); gP.strokeStyle = col[mode]; gP.lineWidth = 1.4; gP.stroke(); }
  gP.setLineDash([]); gP.strokeStyle = 'rgba(255,255,255,.6)'; gP.beginPath(); gP.moveTo(X(t / 3600), 0); gP.lineTo(X(t / 3600), H); gP.stroke();
  gP.fillStyle = '#8fa6bb'; gP.font = '10px ui-monospace,monospace'; gP.fillText('RMS shape error (0-30 m)   solid=active dashed=scripted', 6, 11);
  gP.fillStyle = '#3fd16b'; gP.fillText('perfect pose', 6, H - 5); gP.fillStyle = '#ff5a6e'; gP.fillText('dead-reckoned', 80, H - 5); gP.fillStyle = '#4d8cff'; gP.fillText('SLAM-corrected', 176, H - 5);
}
const PHC = ['#2a4a7a', '#2f6a8a', '#2f8a8a', '#3a8a6a', '#7a8a3a', '#8a6a3a'];
function drawStrip(t) {
  const W = cStrip.width = Math.floor(cStrip.clientWidth), H = 54, tEnd = F[F.length - 1][0]; gS.clearRect(0, 0, W, H); gS.fillStyle = 'rgba(8,12,18,0.72)'; gS.fillRect(0, 0, W, H);
  const X = (s) => 8 + (W - 16) * s / tEnd; D.phases.forEach((p, i) => { gS.fillStyle = PHC[i % 6]; gS.globalAlpha = 0.55; gS.fillRect(X(p.t0), 4, X(p.t1) - X(p.t0) - 1, 12); gS.globalAlpha = 1; gS.fillStyle = '#dff'; gS.font = '9px ui-monospace,monospace'; gS.save(); gS.beginPath(); gS.rect(X(p.t0), 4, X(p.t1) - X(p.t0) - 2, 12); gS.clip(); gS.fillText(p.name.replace('Detailed mapping', 'Mapping').replace('Close proximity', 'Close'), X(p.t0) + 3, 13); gS.restore(); });
  const cv = D.compare['active/slam']; gS.beginPath(); gS.moveTo(X(0), H - 6); cv.forEach((c) => gS.lineTo(X(c[0] * 3600), H - 6 - 26 * c[1])); gS.lineTo(X(tEnd), H - 6); gS.fillStyle = 'rgba(89,224,255,.28)'; gS.fill(); gS.strokeStyle = '#59e0ff'; gS.stroke();
  for (const ft of features) if (ft.birth >= 0) { gS.fillStyle = ft.kind === 'crater' ? '#ffd27a' : '#b6ffb0'; gS.fillRect(X(ft.birth * 3600) - 1, 18, 2, 5); }
  gS.fillStyle = '#fff'; gS.fillRect(X(Math.min(t, tEnd)) - 1, 2, 2, H - 4); gS.fillStyle = '#8fa6bb'; gS.fillText('coverage', W - 52, H - 8);
}
cStrip.addEventListener('pointerdown', (e) => { const drag = (ev) => { const r = cStrip.getBoundingClientRect(), tEnd = F[F.length - 1][0]; S.time = Math.min(Math.max((ev.clientX - r.left - 8) / (r.width - 16), 0), 1) * tEnd; }; drag(e); const up = () => { removeEventListener('pointermove', drag); removeEventListener('pointerup', up); }; addEventListener('pointermove', drag); addEventListener('pointerup', up); });
function toast(msg) { const d = document.createElement('div'); d.className = 'toast'; d.textContent = msg; document.getElementById('toasts').appendChild(d); setTimeout(() => d.remove(), 4200); }

// ---- cinematic director + recording
const shots = ['wide', 'chase', 'skim', 'nadir'], dirP = new THREE.Vector3(), dirT = new THREE.Vector3(), curT = new THREE.Vector3(); let shotI = 0, shotT = 0, shotFov = 45;
function director(dt, p, v, Req) {
  shotT += dt; if (shotT > 9) { shotT = 0; shotI = (shotI + 1) % shots.length; } const sh = shots[shotI], n = p.clone().normalize(), vh = v.clone().normalize();
  if (sh === 'wide') { const a = performance.now() * 0.00006; dirP.set(Math.cos(a) * 3.4 * Req, Math.sin(a) * 3.4 * Req, 1.2 * Req); dirT.set(0, 0, 0); shotFov = 42; }
  else if (sh === 'chase') { dirP.copy(p).addScaledVector(vh, -170).addScaledVector(n, 70); dirT.copy(p).multiplyScalar(0.6); shotFov = 50; }
  else if (sh === 'skim') { const r = radius(...bodyDir(p)) * KM; dirP.copy(bodyWorld(p)).multiplyScalar(r + 70).addScaledVector(vh, -60); dirT.copy(p); shotFov = 55; }
  else { dirP.copy(p).addScaledVector(n, 90).addScaledVector(vh, -30); dirT.copy(p).multiplyScalar(0.2); shotFov = 32; }
  const k = 1 - Math.exp(-dt * 2.2); camera.position.lerp(dirP, k); curT.lerp(dirT, k); camera.lookAt(curT); camera.fov += (shotFov - camera.fov) * k; camera.updateProjectionMatrix();
}
const bodyDir = (p) => { const q = p.clone().applyAxisAngle(new THREE.Vector3(0, 0, 1), -body.rotation.z).normalize(); return [q.x, q.y, q.z]; };
const bodyWorld = (p) => p.clone().normalize();
let mr = null, chunks = [], runRec = false;
function startRec() { if (mr) return; const st = renderer.domElement.captureStream(30); mr = new MediaRecorder(st, { mimeType: 'video/webm;codecs=vp9', videoBitsPerSecond: 12e6 }); chunks = []; mr.ondataavailable = (e) => chunks.push(e.data);
  mr.onstop = () => { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob(chunks, { type: 'video/webm' })); a.download = 'asteroid_slam.webm'; a.click(); mr = null; document.getElementById('rec').style.display = 'none'; }; mr.start(); document.getElementById('rec').style.display = 'block'; }
function stopRec() { if (mr) mr.stop(); runRec = false; }
function recordRun() { S.time = 0; S.cine = true; S.play = true; S.speed = F[F.length - 1][0] / 75; runRec = true; startRec(); }

// ---- GUI
const gui = new GUI({ title: 'Asteroid SLAM' });
const kindCtl = { c: null }; gui.add(S, 'play').listen(); gui.add(S, 'speed', 20, 2000, 10).name('sim speed (x)').listen(); const tc = gui.add(S, 'time', 0, 1, 1).name('time (s)').listen();
gui.add(S, 'view', { 'true surface + model': 'both', 'model only': 'model', 'true surface only': 'truth' }).onChange(applyView);
gui.add(S, 'cine').name('cinematic camera (C)').listen(); gui.add(S, 'follow').name('follow spacecraft (F)').listen(); gui.add(S, 'post').name('post-FX (P)').listen(); gui.add(S, 'bloom', 0, 1.5, 0.05).onChange((v) => { post.bloom.strength = v; });
gui.add(S, 'showScan').name('lidar footprint'); gui.add(S, 'showEst').name('believed track (pink)'); gui.add(S, 'showDR').name('dead-reckoned ghost (red)'); gui.add(S, 'labels').name('feature labels');
gui.add({ rec: () => (mr ? stopRec() : startRec()) }, 'rec').name('record on/off (V)'); gui.add({ run: recordRun }, 'run').name('record whole run');
function applyView() { if (!truthMesh) return; truthMesh.visible = rocks.visible = S.view !== 'model'; recon.visible = reconWire.visible = S.view !== 'truth'; uReveal.value = S.view === 'truth' ? 0 : 1; }
addEventListener('keydown', (e) => { if (e.code === 'Space') { S.play = !S.play; e.preventDefault(); } else if (e.code === 'KeyR') S.time = 0; else if (e.code === 'KeyC') S.cine = !S.cine; else if (e.code === 'KeyF') S.follow = !S.follow; else if (e.code === 'KeyP') S.post = !S.post; else if (e.code === 'KeyV') (mr ? stopRec() : startRec()); });
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); post.resize(innerWidth, innerHeight); });

// ---- main loop
const clock = new THREE.Clock(), prev = new THREE.Vector3(), pv = new THREE.Vector3(), pw = new THREE.Vector3(), vel = new THREE.Vector3(), up = new THREE.Vector3(0, 0, 1), proj = new THREE.Vector3(); let fN = 0, hudT = 0;
const lerp = (a, b, k) => a + (b - a) * k;
function frame() {
  requestAnimationFrame(frame); const dt = Math.min(clock.getDelta(), 0.1); if (!D) return;
  const tEnd = F[F.length - 1][0], dtF = F[1][0] - F[0][0]; if (S.play) { S.time += dt * S.speed; if (S.time > tEnd + 300) { S.time = 0; features.forEach((f) => (f.said = false)); if (runRec) stopRec(); } } tc.max(tEnd);
  const tt = Math.min(Math.max(S.time, 0), tEnd), fl = tt / dtF, f = Math.min(Math.floor(fl), F.length - 1), f2 = Math.min(f + 1, F.length - 1), a = fl - f, A = F[f], B = F[f2], iv = (o) => lerp(A[o], B[o], a);
  uT.value = tt / 3600; body.rotation.z = thetaAt(tt);
  pv.set(iv(1), iv(2), iv(3)).multiplyScalar(KM); vel.set(B[1] - A[1], B[2] - A[2], B[3] - A[3]).normalize();
  craft.position.copy(pv); craft.lookAt(0, 0, 0);
  { const sl = sunDir.clone().applyQuaternion(craft.quaternion.clone().invert()); pivot.rotation.x = Math.atan2(-sl.y, sl.z); }
  craft.children.forEach((c) => { if (c.userData.led) c.visible = Math.floor(performance.now() / 600) % 2 === 0; });
  gEst.position.set(iv(4), iv(5), iv(6)).multiplyScalar(KM); gEst.visible = S.showEst; gEst.rotation.y += dt;
  gDR.position.set(iv(11), iv(12), iv(13)).multiplyScalar(KM); gDR.visible = S.showDR; gDR.rotation.y -= dt;
  setTrail(trTrue, F, f, 1); setTrail(trEst, F, f, 4); setTrail(trDR, F, f, 11); trEst.visible = S.showEst; trDR.visible = S.showDR;
  const si = Math.max(0, D.snap_t.findIndex((t) => t > tt) < 0 ? D.snaps.length - 1 : D.snap_t.findIndex((t) => t > tt) - 1); if (si !== lastSnap) updateRecon(si);
  // sun, shadows, god rays
  sun.position.copy(sunDir).multiplyScalar(3000); sun.target.position.set(0, 0, 0); sunDisc.position.copy(sunDir).multiplyScalar(20000); recon.material.uniforms.uSunV.value.copy(sunDir).transformDirection(camera.matrixWorldInverse);
  // lidar footprint (30 Hz) + cone + pulses
  if (S.showScan && fN % 2 === 0) { footprint(pw.set(...bodyFrame([iv(1), iv(2), iv(3)], thetaAt(tt))).clone()); } fpPts.material.uniforms.uTm.value = performance.now() / 1000;
  { const p = pv, dir = p.clone().negate().normalize(), half = Math.tan(FOVR / 2), e1c = new THREE.Vector3(0, 0, 1).cross(dir).normalize(), e2c = dir.clone().cross(e1c), Lc = Math.max(p.length() - AST.Req * KM * 0.9, 50), pts = [], corner = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
    for (const [u, v] of corner) pts.push(p, p.clone().addScaledVector(dir, Lc).addScaledVector(e1c, u * half * Lc).addScaledVector(e2c, v * half * Lc)); cone.geometry.setFromPoints(pts); cone.visible = S.showScan;
    rings.forEach((r, i) => { const s = ((performance.now() / 1400 + i / 3) % 1), d = s * Lc; r.geometry.setFromPoints(corner.map(([u, v]) => p.clone().addScaledVector(dir, d).addScaledVector(e1c, u * half * d).addScaledVector(e2c, v * half * d))); r.material.opacity = (1 - s) * 0.7; r.visible = S.showScan; }); }
  // camera
  const Req = AST.Req * KM;
  if (S.cine) { controls.enabled = false; director(dt, pv, vel, Req); } else { controls.enabled = true; camera.fov += (45 - camera.fov) * 0.1; camera.updateProjectionMatrix(); if (S.follow) { camera.position.add(pv.clone().sub(prev)); controls.target.copy(pv); } controls.update(); } prev.copy(pv);
  // features: pins + toasts
  body.updateMatrixWorld(); const th = tt / 3600;
  for (const ft of features) { const mapped = ft.birth >= 0 && ft.birth <= th; if (mapped && !ft.said) { ft.said = true; if (S.play) toast(`${ft.kind === 'crater' ? '◎' : '◆'} ${ft.name} mapped`); } if (!mapped && ft.said) ft.said = false;
    pw.copy(ft.p).applyMatrix4(body.matrixWorld); const nrm = new THREE.Vector3(...ft.u).transformDirection(body.matrixWorld); proj.copy(pw).project(camera);
    const vis = S.labels && mapped && proj.z < 1 && Math.abs(proj.x) < 1 && Math.abs(proj.y) < 1 && nrm.dot(camera.position.clone().sub(pw)) > 0 && S.view !== 'model';
    ft.el.style.opacity = vis ? 1 : 0; if (vis) { ft.el.style.left = (proj.x * 0.5 + 0.5) * innerWidth + 'px'; ft.el.style.top = (-proj.y * 0.5 + 0.5) * innerHeight + 'px'; } }
  // HUD
  const cv = D.compare['active/slam'], c = cv[Math.min(Math.floor(f / 10), cv.length - 1)], ph = D.phases[A[10]] || D.phases[0];
  panel.innerHTML = `<div class="ttl">${AST.name.toUpperCase()}</div>phase <b>${ph.name}</b> &nbsp; orbit r <b>${(ph.r_km * 1000).toFixed(0)}</b> m<br>time <b>${(A[0] / 3600).toFixed(1)}</b> h &nbsp; alt <b>${((Math.hypot(A[1], A[2], A[3]) - AST.R0) * 1000).toFixed(0)}</b> m<br>coverage <b>${(c[1] * 100).toFixed(1)}</b> % &nbsp; RMS <b>${c[2].toFixed(1)}</b> m &nbsp; vol <b>${c[3] >= 0 ? '+' : ''}${c[3].toFixed(1)}</b> %<br>` +
    `position err: SLAM <b style="color:#7fb0ff">${A[8].toFixed(1)}</b> m · dead-reck. <b style="color:#ff7a8a">${A[9].toFixed(1)}</b> m`;
  if (fN % 3 === 0) { drawPlot(tt); drawMap(tt, f); drawStrip(tt); }
  if (S.post) { post.bloom.strength = S.bloom; post.update(performance.now() / 1000, sunDir, 0); post.composer.render(); } else renderer.render(scene, camera);
  if (fN++ === 0) document.getElementById('hud').textContent = 'SPACE play · R restart · C cinematic · F follow · P post-FX · V record · drag the strip to scrub';
}

// ---- load
async function load(id) {
  const ent = (window.__idx || []).find((e) => e.id === id), r = await fetch(ent ? ent.file : 'data/mission.json'); D = await r.json(); F = D.frames; setAst(D.asteroid); OMEGA = 2 * Math.PI / (AST.period_h * 3600);
  build(); lastSnap = -1; applyView(); S.time = 0; const R = AST.Req * KM; camera.position.set(2.6 * R, -2.4 * R, 1.3 * R); controls.target.set(0, 0, 0); curT.set(0, 0, 0); window.__ready = true;
}
(async () => {
  try { window.__idx = await (await fetch('data/index.json')).json(); } catch (e) { window.__idx = []; }
  const q = new URLSearchParams(location.search).get('kind'); S.kind = q || (window.__idx[0] && window.__idx[0].id) || 'rubble';
  if (window.__idx.length > 1) gui.add(S, 'kind', Object.fromEntries(window.__idx.map((e) => [e.name, e.id]))).name('asteroid').onChange((v) => load(v));
  try { await load(S.kind); } catch (e) { document.body.insertAdjacentHTML('beforeend', `<div style="position:fixed;inset:0;display:flex;align-items:center;justify-content:center">No data (${e.message}). Run python run_asteroid.py --kind all, then serve this repo with python -m http.server.</div>`); }
})();
window.__dbg = () => ({ on: fpPts.geometry.attributes.aOn.array.reduce((a, b) => a + b, 0), line: fpLine.visible, rocks: rocks.count, snap: lastSnap }); window.__shot = (i) => { shotI = i; shotT = 0; }; window.__S = S; window.__applyView = () => applyView(); frame();
