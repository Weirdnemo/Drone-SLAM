// Post-processing + atmosphere plumbing: height-aware fog (patched ShaderChunk), bloom, screen-space god rays,
// colour grade / vignette / grain / chromatic aberration. All HDR, tone-mapped last by OutputPass.
import * as THREE from 'three';
import { EffectComposer } from './vendor/postprocessing/EffectComposer.js';
import { RenderPass } from './vendor/postprocessing/RenderPass.js';
import { ShaderPass } from './vendor/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from './vendor/postprocessing/UnrealBloomPass.js';
import { OutputPass } from './vendor/postprocessing/OutputPass.js';

// ---- ground-hugging mist: FogExp2 density is multiplied by 1 + MIST*exp(-(worldZ+1)/2.5)  (must run before any material compiles)
export function patchHeightFog(mist = 2.2) {
  const C = THREE.ShaderChunk;
  C.fog_pars_vertex = C.fog_pars_vertex.replace('varying float vFogDepth;', 'varying float vFogDepth;\n\tvarying float vFogWZ;');
  C.fog_vertex = C.fog_vertex.replace('vFogDepth = - mvPosition.z;',
    'vFogDepth = - mvPosition.z;\n\tvFogWZ = dot( viewMatrix[2].xyz, mvPosition.xyz ) + cameraPosition.z;');
  C.fog_pars_fragment = C.fog_pars_fragment.replace('varying float vFogDepth;', 'varying float vFogDepth;\n\tvarying float vFogWZ;');
  C.fog_fragment = C.fog_fragment.replace('float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );',
    `float hm = 1.0 + ${mist.toFixed(2)} * exp( - ( vFogWZ + 1.0 ) / 2.5 );
\t\tfloat fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth * hm );`);
}

const GodRays = {
  uniforms: { tDiffuse: { value: null }, uSun: { value: new THREE.Vector2(0.5, 0.8) }, uStr: { value: 0.55 }, uThr: { value: 0.55 }, uFace: { value: 1 } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
  fragmentShader: `varying vec2 vUv; uniform sampler2D tDiffuse; uniform vec2 uSun; uniform float uStr, uThr, uFace;
    void main(){
      vec3 base = texture2D(tDiffuse, vUv).rgb; vec2 d = (uSun - vUv) / 40.0 * 0.95; vec2 uv = vUv; float w = 1.0; vec3 acc = vec3(0.0);
      for (int i = 0; i < 40; i++) { uv += d; vec3 s = texture2D(tDiffuse, uv).rgb; float l = max(dot(s, vec3(0.3333)) - uThr, 0.0); acc += s * l * w; w *= 0.955; }
      gl_FragColor = vec4(base + acc / 40.0 * uStr * 3.0 * uFace * vec3(1.0, 0.93, 0.78), 1.0); }`,
};

const Grade = {
  uniforms: { tDiffuse: { value: null }, uTime: { value: 0 }, uVig: { value: 0.45 }, uGrain: { value: 0.03 }, uSat: { value: 1.12 },
              uContrast: { value: 1.08 }, uCA: { value: 0.0035 }, uWarm: { value: 0 } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
  fragmentShader: `varying vec2 vUv; uniform sampler2D tDiffuse; uniform float uTime, uVig, uGrain, uSat, uContrast, uCA, uWarm;
    float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main(){
      vec2 d = vUv - 0.5; float r2 = dot(d, d);
      vec3 c = vec3(texture2D(tDiffuse, vUv + d * uCA * r2 * 8.0).r, texture2D(tDiffuse, vUv).g, texture2D(tDiffuse, vUv - d * uCA * r2 * 8.0).b);
      float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
      c = mix(vec3(l), c, uSat);
      c *= mix(vec3(0.90, 1.0, 1.10), vec3(1.10, 1.0, 0.88), smoothstep(0.0, 1.2, l));          // split tone: cool shadows, warm highlights
      c *= mix(vec3(1.0), vec3(1.10, 0.96, 0.86), uWarm);                                          // smoky tint in the disaster zone
      c = max((c - 0.18) * uContrast + 0.18, 0.0);
      c *= mix(1.0, smoothstep(0.95, 0.22, length(d) * 1.3), uVig);
      c += (hash(vUv * 1000.0 + uTime) - 0.5) * uGrain;
      gl_FragColor = vec4(c, 1.0); }`,
};

export function makePost(renderer, scene, camera) {
  const pr = renderer.getPixelRatio(), w = innerWidth, h = innerHeight;
  const rt = new THREE.WebGLRenderTarget(w * pr, h * pr, { type: THREE.HalfFloatType, samples: 4 });
  const composer = new EffectComposer(renderer, rt);
  composer.setPixelRatio(pr); composer.setSize(w, h);
  composer.addPass(new RenderPass(scene, camera));
  const rays = new ShaderPass(GodRays); composer.addPass(rays);
  const bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.5, 0.7, 0.9); composer.addPass(bloom);
  const grade = new ShaderPass(Grade); composer.addPass(grade);
  composer.addPass(new OutputPass());
  return {
    composer, bloom, rays, grade,
    resize(w2, h2) { composer.setSize(w2, h2); },
    update(t, sunDir, warm) {
      grade.uniforms.uTime.value = t % 100; grade.uniforms.uWarm.value = warm;
      const p = camera.position.clone().addScaledVector(sunDir, 1000).project(camera);
      rays.uniforms.uSun.value.set(p.x * 0.5 + 0.5, p.y * 0.5 + 0.5);
      const f = new THREE.Vector3(); camera.getWorldDirection(f);
      rays.uniforms.uFace.value = THREE.MathUtils.smoothstep(f.dot(sunDir), -0.1, 0.5) * (p.z < 1 ? 1 : 0);
    },
  };
}
