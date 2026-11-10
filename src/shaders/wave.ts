import {
  COMMON_WGSL,
  MAX_ACTIVE_DROPS,
  WAVE_GRID_SIZE,
  WAVE_HALF_EXTENT,
} from './common';

const WAVE_DOMAIN_LEN = (WAVE_HALF_EXTENT * 2.0).toFixed(4);
const WAVE_HALF_GRID = WAVE_GRID_SIZE / 2;
const WAVE_FFT_STAGES = Math.round(Math.log2(WAVE_GRID_SIZE));
const WAVE_BIT_REV_SHIFT = 32 - WAVE_FFT_STAGES;
const WAVE_UNITARY_SCALE = (1.0 / Math.sqrt(WAVE_GRID_SIZE)).toFixed(11);
const WAVE_LAP_SCALE = Math.pow(WAVE_GRID_SIZE / 256.0, 2.0).toFixed(4);

export const WAVE_CLEAR_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var waveOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${WAVE_GRID_SIZE}u || gid.y >= ${WAVE_GRID_SIZE}u) {
    return;
  }
  let zeroTouch = u.surfaceParams.w * 0.0;
  textureStore(waveOut, vec2<i32>(gid.xy), vec4<f32>(zeroTouch, 0.0, 0.0, 0.0));
}
`;

export const WAVE_CLEAR_SPEC_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var specOut: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${WAVE_GRID_SIZE}u || gid.y >= ${WAVE_GRID_SIZE}u) {
    return;
  }
  let zeroTouch = u.surfaceParams.w * 0.0;
  textureStore(specOut, vec2<i32>(gid.xy), vec4<f32>(zeroTouch, 0.0, 0.0, 0.0));
}
`;

/**
 * Spatial pre-FFT pass:
 *  1. Evaluates multi-drop zero-integral polar Ricker stagnation/coalescence impulses across all 8 drops
 *  2. Evaluates blood surfactant Marangoni surface-tension pull (Delta sigma ~ -17 mN/m -> outward traction)
 *  3. Couples 3D MAC top free-surface vertical velocity upward into meniscus kinematics
 *  4. Applies radial sponge prior to 2D shared-memory FFT
 * Outputs complex (eta, dEta/dt) in .rg and updated surfactant film in .b of rgba32float specOut.
 */
export const WAVE_SPATIAL_FORCE_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var waveIn: texture_2d<f32>;
@group(0) @binding(2) var velMacIn: texture_3d<f32>;
@group(0) @binding(3) var linSampler: sampler;
@group(0) @binding(4) var specOut: texture_storage_2d<rgba32float, write>;

fn loadWaveClamped(x: i32, y: i32) -> vec4<f32> {
  let cx = clamp(x, 0, ${WAVE_GRID_SIZE - 1});
  let cy = clamp(y, 0, ${WAVE_GRID_SIZE - 1});
  return textureLoad(waveIn, vec2<i32>(cx, cy), 0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${WAVE_GRID_SIZE}u || gid.y >= ${WAVE_GRID_SIZE}u) {
    return;
  }
  let ix = i32(gid.x);
  let iy = i32(gid.y);

  let c = loadWaveClamped(ix, iy);
  var eta = c.r;
  var vel = c.g;
  var crimsonFilm = c.a;

  let dt = clamp(u.viewportAndDt.z, 0.0, 0.025);
  let timeScale = dt / 0.0166667;

  // Isotropic 9-point stencil of blood surfactant film for Marangoni tangential divergence
  let fn0 = loadWaveClamped(ix, iy + 1).a;
  let fs0 = loadWaveClamped(ix, iy - 1).a;
  let fe0 = loadWaveClamped(ix + 1, iy).a;
  let fw0 = loadWaveClamped(ix - 1, iy).a;
  let fne = loadWaveClamped(ix + 1, iy + 1).a;
  let fnw = loadWaveClamped(ix - 1, iy + 1).a;
  let fse = loadWaveClamped(ix + 1, iy - 1).a;
  let fsw = loadWaveClamped(ix - 1, iy - 1).a;
  let rawLapFilm = 0.20 * (fn0 + fs0 + fe0 + fw0) + 0.05 * (fne + fnw + fse + fsw) - crimsonFilm;
  let lapFilm = rawLapFilm * ${WAVE_LAP_SCALE};

  let uv = (vec2<f32>(f32(ix), f32(iy)) + 0.5) / ${WAVE_GRID_SIZE.toFixed(1)};
  let worldXZ = waveUVToWorldXZ(uv);

  // 1. Multi-drop stagnation & wetting-neck crater excavation across all 8 concurrent drops
  var forceDrops = 0.0;
  var activeExcav = 0.0;
  var maxDrain = 0.0;
  for (var d = 0u; d < ${MAX_ACTIVE_DROPS}u; d = d + 1u) {
    let slot = u.drops[d];
    let signedSeed = slot.velOscAndDrain.w;
    let drainRate = slot.velOscAndDrain.z;
    if (signedSeed > 0.0 && drainRate > 0.0005) {
      maxDrain = max(maxDrain, drainRate);
      let dropXZ = slot.posAndRadius.xz;
      let rDrop = max(slot.posAndRadius.w, 0.018);
      let speedY = clamp(-slot.velOscAndDrain.x, 0.06, 0.65);
      let r = length(worldXZ - dropXZ);

      // Exact 2D polar zero-integral Ricker wavelet: int_0^inf r * exp(-r^2) * (1 - r^2) dr == 0
      // Single momentum-coupled entry crater + wetting-line capillary impulse
      let rFootprint = clamp(rDrop * 1.05, 0.024, 0.082);
      let rNorm = r / rFootprint;
      let emergNorm = (slot.posAndRadius.y - u.surfaceParams.x) / max(rDrop, 0.018);
      let excavGate = smoothstep(-0.78, -0.04, emergNorm);
      let impactBell = sin(clamp((1.05 - emergNorm) / 1.83, 0.0, 1.0) * 3.14159265) * excavGate;
      let impactShape = exp(-rNorm * rNorm) * (1.0 - rNorm * rNorm);
      let dynImpulse = (6.8 * impactBell * (rDrop / 0.056)) * (0.72 + 2.65 * speedY);

      // Wetting-neck capillary meniscus contact impulse exciting dispersive precursor wavelets (resolved at dx = 6.25 mm)
      let rCap = clamp(rDrop * 0.44, 0.013, 0.030);
      let rNormCap = r / rCap;
      let capWindow = smoothstep(-0.35, 0.28, emergNorm) * (1.0 - smoothstep(0.68, 1.06, emergNorm));
      let capShape = exp(-rNormCap * rNormCap) * (1.0 - rNormCap * rNormCap);
      let capImpulse = (4.6 * capWindow * (0.60 + 0.40 * (rDrop / 0.056))) * (0.65 + 2.20 * speedY);

      forceDrops = forceDrops - dynImpulse * impactShape - capImpulse * capShape;
      activeExcav = max(activeExcav, excavGate * exp(-0.5 * rNorm * rNorm));

      // Deposit localized blood surfactant film (sigma_blood ~ 55 mN/m vs water 72 mN/m) inside immediate collar
      let rStain = max(0.018, rDrop * 0.60);
      let stain = exp(-(r * r) / (rStain * rStain)) * clamp(drainRate * 0.42, 0.0, 1.0);
      crimsonFilm = max(crimsonFilm, stain * 0.65);
    }
  }

  // 2. Marangoni surfactant pull (Delta sigma ~ -17 mN/m pulls outward along contact perimeter without pinning crater floor)
  let marangoniStr = u.surfaceParams.y;
  let effLapFilm = mix(max(lapFilm, -0.0015), lapFilm, activeExcav);
  let forceMarangoni = marangoniStr * 0.28 * effLapFilm;
  crimsonFilm = clamp(crimsonFilm + marangoniStr * 0.065 * rawLapFilm * timeScale, 0.0, 1.0);
  crimsonFilm = crimsonFilm * pow(0.88, timeScale);

  // 3. Two-way 3D MAC subsurface upward feedback gated to active coalescence (preventing perpetual deep return-flow sonar rings!)
  var forceSubsurface = 0.0;
  let halfXZ = u.domainBounds.x;
  let subsurfGate = clamp(activeExcav * 1.4 + smoothstep(0.0, 0.06, maxDrain), 0.0, 1.0);
  if (subsurfGate > 0.001 && length(worldXZ) < halfXZ * 0.94) {
    let tankUV = (worldXZ + vec2<f32>(halfXZ)) / (2.0 * halfXZ);
    let topUVW = vec3<f32>(tankUV.x, 0.988, tankUV.y);
    let vSubY = textureSampleLevel(velMacIn, linSampler, topUVW, 0.0).y;
    let wallTaper = 1.0 - smoothstep(halfXZ * 0.62, halfXZ * 0.92, length(worldXZ));
    let nonSuctionV = max(vSubY, 0.0);
    forceSubsurface = clamp(nonSuctionV, 0.0, 0.48) * 0.42 * wallTaper * subsurfGate;
  }

  vel = vel + (forceDrops + forceMarangoni + forceSubsurface) * dt;

  // Non-reflecting radial Rayleigh velocity-damping absorber beyond the viewport horizon (r > 0.90 m),
  // eliminating any square grid-wall checkerboard interference or inward drumhead reflections!
  let rDist = length(worldXZ);
  let absorbZone = smoothstep(0.90, 1.54, rDist);
  let rayleighDamp = exp(-dt * (24.0 * absorbZone * absorbZone));
  let borderZero = 1.0 - smoothstep(1.22, 1.56, rDist);
  eta = clamp(eta * rayleighDamp * borderZero, -0.14, 0.14);
  vel = clamp(vel * rayleighDamp * borderZero, -1.8, 1.8);
  crimsonFilm = crimsonFilm * borderZero;

  textureStore(specOut, vec2<i32>(ix, iy), vec4<f32>(eta, vel, crimsonFilm, 0.0));
}
`;

function createRadix2Fft1DShader(direction: 'row' | 'col', inverse: boolean): string {
  const fftSign = inverse ? '1.0' : '-1.0';
  const loadCoord0 =
    direction === 'row'
      ? 'vec2<i32>(i32(rev0), i32(lineIdx))'
      : 'vec2<i32>(i32(lineIdx), i32(rev0))';
  const loadCoord1 =
    direction === 'row'
      ? 'vec2<i32>(i32(rev1), i32(lineIdx))'
      : 'vec2<i32>(i32(lineIdx), i32(rev1))';
  const storeCoord0 =
    direction === 'row'
      ? 'vec2<i32>(i32(idx0), i32(lineIdx))'
      : 'vec2<i32>(i32(lineIdx), i32(idx0))';
  const storeCoord1 =
    direction === 'row'
      ? 'vec2<i32>(i32(idx1), i32(lineIdx))'
      : 'vec2<i32>(i32(lineIdx), i32(idx1))';

  return /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var specIn: texture_2d<f32>;
@group(0) @binding(2) var specOut: texture_storage_2d<rgba32float, write>;

var<workgroup> smem: array<vec2<f32>, ${WAVE_GRID_SIZE}>;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let lineIdx = wid.x;
  if (lineIdx >= ${WAVE_GRID_SIZE}u) {
    return;
  }

  // Exact 9-bit bit-reversal for N = 512 using 256 threads (1 butterfly/thread/stage, 4 KB smem)
  let idx0 = tid;
  let idx1 = tid + 256u;
  let rev0 = reverseBits(idx0) >> ${WAVE_BIT_REV_SHIFT}u;
  let rev1 = reverseBits(idx1) >> ${WAVE_BIT_REV_SHIFT}u;
  smem[idx0] = textureLoad(specIn, ${loadCoord0}, 0).xy;
  smem[idx1] = textureLoad(specIn, ${loadCoord1}, 0).xy;
  workgroupBarrier();

  for (var stage = 0u; stage < ${WAVE_FFT_STAGES}u; stage = stage + 1u) {
    let halfSpan = 1u << stage;
    let span = halfSpan << 1u;
    let k = tid & (halfSpan - 1u);
    let blockBase = (tid >> stage) << (stage + 1u);
    let i0 = blockBase + k;
    let i1 = i0 + halfSpan;

    let a = smem[i0];
    let b = smem[i1];
    let angle = ${fftSign} * (6.28318530718 * f32(k)) / f32(span);
    let c = cos(angle);
    let s = sin(angle);
    let twB = vec2<f32>(b.x * c - b.y * s, b.x * s + b.y * c);
    workgroupBarrier();
    smem[i0] = a + twB;
    smem[i1] = a - twB;
    workgroupBarrier();
  }

  // Exact unitary 1/sqrt(N) scaling on every 1D transform
  let zeroBias = vec2<f32>(u.surfaceParams.x * 0.0);
  let scaled0 = smem[idx0] * ${WAVE_UNITARY_SCALE} + zeroBias;
  let scaled1 = smem[idx1] * ${WAVE_UNITARY_SCALE} + zeroBias;
  textureStore(specOut, ${storeCoord0}, vec4<f32>(scaled0.x, scaled0.y, 0.0, 0.0));
  textureStore(specOut, ${storeCoord1}, vec4<f32>(scaled1.x, scaled1.y, 0.0, 0.0));
}
`;
}

export const WAVE_FFT_ROW_WGSL = createRadix2Fft1DShader('row', false);
export const WAVE_FFT_COL_WGSL = createRadix2Fft1DShader('col', false);
export const WAVE_IFFT_COL_WGSL = createRadix2Fft1DShader('col', true);
export const WAVE_IFFT_ROW_WGSL = createRadix2Fft1DShader('row', true);

/**
 * Exact Fourier-space deep/finite-depth gravity-capillary propagator:
 *   omega(k) = sqrt((g_eff * k + (sigma/rho)_eff * k^3) * tanh(k * H))
 * Unpacks packed real spectra (eta_hat, w_hat) via Hermitian symmetry, applies exact
 * unitary dispersive phase rotation + physical viscous attenuation exp(-2 * nu * k^2 * dt),
 * enforces exact zero DC mean (volume conservation), and repacks eta_hat' + i * w_hat'.
 */
export const WAVE_SPECTRAL_PROPAGATE_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var specIn: texture_2d<f32>;
@group(0) @binding(2) var specOut: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${WAVE_GRID_SIZE}u || gid.y >= ${WAVE_GRID_SIZE}u) {
    return;
  }
  let ux = i32(gid.x);
  let uy = i32(gid.y);

  // Enforce exact zero DC mean (int eta dx dz == 0) and suppress Nyquist 2-texel border
  if ((ux == 0 && uy == 0) || ux == ${WAVE_HALF_GRID} || uy == ${WAVE_HALF_GRID}) {
    textureStore(specOut, vec2<i32>(ux, uy), vec4<f32>(0.0));
    return;
  }

  let nx = (${WAVE_GRID_SIZE} - ux) & ${WAVE_GRID_SIZE - 1};
  let ny = (${WAVE_GRID_SIZE} - uy) & ${WAVE_GRID_SIZE - 1};

  let fPos = textureLoad(specIn, vec2<i32>(ux, uy), 0).xy;
  let fNeg = textureLoad(specIn, vec2<i32>(nx, ny), 0).xy;

  // Exact Hermitian unpacking of two real fields packed as Z = eta + i * w
  let etaHat = 0.5 * vec2<f32>(fPos.x + fNeg.x, fPos.y - fNeg.y);
  let velHat = 0.5 * vec2<f32>(fPos.y + fNeg.y, -fPos.x + fNeg.x);

  let mx = f32(select(ux, ux - ${WAVE_GRID_SIZE}, ux > ${WAVE_HALF_GRID}));
  let mz = f32(select(uy, uy - ${WAVE_GRID_SIZE}, uy > ${WAVE_HALF_GRID}));
  let twoPiOverL = 6.28318530718 / ${WAVE_DOMAIN_LEN};
  let kx = mx * twoPiOverL;
  let kz = mz * twoPiOverL;
  let k2 = kx * kx + kz * kz;
  let k = sqrt(k2);

  let dt = clamp(u.viewportAndDt.z, 0.0, 0.025);
  let depthH = u.domainBounds.z - u.domainBounds.y; // 1.28 tank depth
  let tanhKH = tanh(min(k * depthH, 12.0));

  // Physical elastocapillary dispersion in cuvette world units (resolving fine capillary precursor wavelets at dx = 6.25 mm):
  // omega(k_crater=34) ~ 11.1 rad/s (half-period T/2 ~ 0.28s -> crisp central swell rebound + dispersive capillary train)
  let gEff = 0.84;
  let sigmaOverRho = 0.0024;
  let omega = sqrt(max((gEff * k + sigmaOverRho * k * k2) * tanhKH, 1e-8));

  // Physical slow-motion viscous capillary-gravity damping + smooth high-k spectral de-aliasing roll-off:
  // Preserves dispersive k = 110..330 capillary precursor rings ahead of the main swell while settling cleanly!
  let viscRate = 0.36 + 0.00018 * k2;
  let nyquistRoll = 1.0 - smoothstep(335.0, 465.0, k);
  let damp = exp(-viscRate * dt) * nyquistRoll;

  let phase = omega * dt;
  let cosP = cos(phase);
  let sinP = sin(phase);

  let etaNext = damp * (etaHat * cosP + velHat * (sinP / omega));
  let velNext = damp * (-etaHat * (omega * sinP) + velHat * cosP);

  // Re-pack Z' = etaNext + i * velNext
  let packed = vec2<f32>(etaNext.x - velNext.y, etaNext.y + velNext.x);
  textureStore(specOut, vec2<i32>(ux, uy), vec4<f32>(packed.x, packed.y, 0.0, 0.0));
}
`;

/**
 * Spatial post-IFFT finalize pass:
 * Combines propagated real (eta, w) from specWaveIn with surfactant film from specAuxIn,
 * evaluates resolution-normalized 9-point isotropic Laplacian into .b (for refractive caustics & Young-Laplace top pressure),
 * and writes filterable rgba16float texWave.
 */
export const WAVE_FINALIZE_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var specWaveIn: texture_2d<f32>;
@group(0) @binding(2) var specAuxIn: texture_2d<f32>;
@group(0) @binding(3) var waveOut: texture_storage_2d<rgba16float, write>;

fn loadEta(x: i32, y: i32) -> f32 {
  let cx = clamp(x, 0, ${WAVE_GRID_SIZE - 1});
  let cy = clamp(y, 0, ${WAVE_GRID_SIZE - 1});
  return textureLoad(specWaveIn, vec2<i32>(cx, cy), 0).x;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${WAVE_GRID_SIZE}u || gid.y >= ${WAVE_GRID_SIZE}u) {
    return;
  }
  let ix = i32(gid.x);
  let iy = i32(gid.y);

  let prop = textureLoad(specWaveIn, vec2<i32>(ix, iy), 0);
  let film = textureLoad(specAuxIn, vec2<i32>(ix, iy), 0).z;

  let n = loadEta(ix, iy + 1);
  let s = loadEta(ix, iy - 1);
  let e = loadEta(ix + 1, iy);
  let w = loadEta(ix - 1, iy);
  let ne = loadEta(ix + 1, iy + 1);
  let nw = loadEta(ix - 1, iy + 1);
  let se = loadEta(ix + 1, iy - 1);
  let sw = loadEta(ix - 1, iy - 1);

  let lap = (0.20 * (n + s + e + w) + 0.05 * (ne + nw + se + sw) - prop.x) * ${WAVE_LAP_SCALE};

  let uv = (vec2<f32>(f32(ix), f32(iy)) + 0.5) / ${WAVE_GRID_SIZE.toFixed(1)};
  let worldXZ = waveUVToWorldXZ(uv);
  let rDist = length(worldXZ);
  let borderZero = 1.0 - smoothstep(1.22, 1.56, rDist) + u.surfaceParams.x * 0.0;

  let etaOut = clamp(prop.x * borderZero, -0.14, 0.14);
  let velOut = clamp(prop.y * borderZero, -1.8, 1.8);
  textureStore(waveOut, vec2<i32>(ix, iy), vec4<f32>(etaOut, velOut, lap * borderZero, film * borderZero));
}
`;
