import {
  COMMON_WGSL,
  MAX_ACTIVE_DROPS,
  WAVE_GRID_SIZE,
  WAVE_HALF_EXTENT,
} from './common';

const WAVE_DOMAIN_LEN = (WAVE_HALF_EXTENT * 2.0).toFixed(4);
const WAVE_SPONGE_INNER = (WAVE_HALF_EXTENT * 0.72).toFixed(4);
const WAVE_SPONGE_OUTER = (WAVE_HALF_EXTENT * 0.96).toFixed(4);

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
  let lapFilm = 0.20 * (fn0 + fs0 + fe0 + fw0) + 0.05 * (fne + fnw + fse + fsw) - crimsonFilm;

  let uv = (vec2<f32>(f32(ix), f32(iy)) + 0.5) / ${WAVE_GRID_SIZE.toFixed(1)};
  let worldXZ = waveUVToWorldXZ(uv);

  // 1. Multi-drop stagnation & wetting-neck crater excavation across all 8 concurrent drops
  var forceDrops = 0.0;
  var activeExcav = 0.0;
  for (var d = 0u; d < ${MAX_ACTIVE_DROPS}u; d = d + 1u) {
    let slot = u.drops[d];
    let signedSeed = slot.velOscAndDrain.w;
    let drainRate = slot.velOscAndDrain.z;
    if (signedSeed > 0.0 && drainRate > 0.0005) {
      let dropXZ = slot.posAndRadius.xz;
      let rDrop = max(slot.posAndRadius.w, 0.018);
      let speedY = clamp(-slot.velOscAndDrain.x, 0.06, 0.65);
      let r = length(worldXZ - dropXZ);

      // Exact 2D polar zero-integral Ricker wavelet: int_0^inf r * exp(-r^2) * (1 - r^2) dr == 0
      // Single momentum-coupled entry crater + sharp wetting-line capillary impulse (zero delayed rebound piston;
      // central Worthington swell & dispersive capillary precursor train emerge purely from the unitary FFT PDE!)
      let rFootprint = clamp(rDrop * 0.68, 0.016, 0.052);
      let rNorm = r / rFootprint;
      let emergNorm = (slot.posAndRadius.y - u.surfaceParams.x) / max(rDrop, 0.018);
      let excavGate = smoothstep(-0.78, -0.04, emergNorm);
      let impactBell = sin(clamp((1.05 - emergNorm) / 1.83, 0.0, 1.0) * 3.14159265) * excavGate;
      let impactShape = exp(-rNorm * rNorm) * (1.0 - rNorm * rNorm);
      let dynImpulse = (14.2 * impactBell * (rDrop / 0.056)) * (0.72 + 2.65 * speedY);

      // Wetting-neck capillary meniscus contact impulse exciting high-k (2-8 mm) dispersive precursor wavelets
      let rCap = clamp(rDrop * 0.25, 0.0105, 0.0185);
      let rNormCap = r / rCap;
      let capWindow = smoothstep(-0.35, 0.28, emergNorm) * (1.0 - smoothstep(0.68, 1.06, emergNorm));
      let capShape = exp(-rNormCap * rNormCap) * (1.0 - rNormCap * rNormCap);
      let capImpulse = (8.8 * capWindow * (0.60 + 0.40 * (rDrop / 0.056))) * (0.65 + 2.20 * speedY);

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
  crimsonFilm = clamp(crimsonFilm + marangoniStr * 0.065 * lapFilm * timeScale, 0.0, 1.0);
  crimsonFilm = crimsonFilm * pow(0.945, timeScale);

  // 3. Two-way 3D MAC subsurface upward feedback onto the free surface (decoupled from submerged stem wake drainage so craters rebound!)
  var forceSubsurface = 0.0;
  let halfXZ = u.domainBounds.x;
  if (abs(worldXZ.x) < halfXZ * 0.94 && abs(worldXZ.y) < halfXZ * 0.94) {
    let tankUV = (worldXZ + vec2<f32>(halfXZ)) / (2.0 * halfXZ);
    let topUVW = vec3<f32>(tankUV.x, 0.988, tankUV.y);
    let vSubY = textureSampleLevel(velMacIn, linSampler, topUVW, 0.0).y;
    let wallTaper = 1.0 - smoothstep(halfXZ * 0.68, halfXZ * 0.94, max(abs(worldXZ.x), abs(worldXZ.y)));
    let nonSuctionV = max(vSubY, 0.0);
    forceSubsurface = clamp(nonSuctionV, 0.0, 0.48) * 0.46 * wallTaper;
  }

  vel = vel + (forceDrops + forceMarangoni + forceSubsurface) * dt;

  // Radial absorbing sponge prevents periodic wrap-around across the spectral box while keeping [-0.48, +0.48] tank unattenuated
  let distOrigin = length(worldXZ);
  let sponge = 1.0 - smoothstep(${WAVE_SPONGE_INNER}, ${WAVE_SPONGE_OUTER}, distOrigin);
  eta = clamp(eta * sponge, -0.14, 0.14);
  vel = clamp(vel * sponge, -1.8, 1.8);
  crimsonFilm = crimsonFilm * sponge;

  textureStore(specOut, vec2<i32>(ix, iy), vec4<f32>(eta, vel, crimsonFilm, 0.0));
}
`;

function createRadix2Fft1DShader(direction: 'row' | 'col', inverse: boolean): string {
  const fftSign = inverse ? '1.0' : '-1.0';
  const loadCoord =
    direction === 'row'
      ? 'vec2<i32>(i32(rev), i32(lineIdx))'
      : 'vec2<i32>(i32(lineIdx), i32(rev))';
  const storeCoord =
    direction === 'row'
      ? 'vec2<i32>(i32(tid), i32(lineIdx))'
      : 'vec2<i32>(i32(lineIdx), i32(tid))';

  return /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var specIn: texture_2d<f32>;
@group(0) @binding(2) var specOut: texture_storage_2d<rgba32float, write>;

var<workgroup> smem: array<vec2<f32>, ${WAVE_GRID_SIZE}>;

@compute @workgroup_size(${WAVE_GRID_SIZE}, 1, 1)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let lineIdx = wid.x;
  if (lineIdx >= ${WAVE_GRID_SIZE}u) {
    return;
  }

  // Exact 8-bit bit-reversal for N = 256
  let rev = reverseBits(tid) >> 24u;
  let src = textureLoad(specIn, ${loadCoord}, 0);
  smem[tid] = src.xy;
  workgroupBarrier();

  for (var stage = 0u; stage < 8u; stage = stage + 1u) {
    let halfSpan = 1u << stage;
    let span = halfSpan << 1u;
    let blockBase = (tid / span) * span;
    let offset = tid & (span - 1u);
    let isTop = offset < halfSpan;
    let k = select(offset - halfSpan, offset, isTop);
    let i0 = blockBase + k;
    let i1 = i0 + halfSpan;

    let a = smem[i0];
    let b = smem[i1];
    let angle = ${fftSign} * (6.28318530718 * f32(k)) / f32(span);
    let c = cos(angle);
    let s = sin(angle);
    let twB = vec2<f32>(b.x * c - b.y * s, b.x * s + b.y * c);
    let nextVal = select(a - twB, a + twB, isTop);
    workgroupBarrier();
    smem[tid] = nextVal;
    workgroupBarrier();
  }

  // Exact unitary 1/sqrt(256) = 1/16 scaling on every 1D transform
  let scaled = smem[tid] * 0.0625 + vec2<f32>(u.surfaceParams.x * 0.0);
  textureStore(specOut, ${storeCoord}, vec4<f32>(scaled.x, scaled.y, 0.0, 0.0));
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
  if ((ux == 0 && uy == 0) || ux == 128 || uy == 128) {
    textureStore(specOut, vec2<i32>(ux, uy), vec4<f32>(0.0));
    return;
  }

  let nx = (256 - ux) & 255;
  let ny = (256 - uy) & 255;

  let fPos = textureLoad(specIn, vec2<i32>(ux, uy), 0).xy;
  let fNeg = textureLoad(specIn, vec2<i32>(nx, ny), 0).xy;

  // Exact Hermitian unpacking of two real fields packed as Z = eta + i * w
  let etaHat = 0.5 * vec2<f32>(fPos.x + fNeg.x, fPos.y - fNeg.y);
  let velHat = 0.5 * vec2<f32>(fPos.y + fNeg.y, -fPos.x + fNeg.x);

  let mx = f32(select(ux, ux - 256, ux > 128));
  let mz = f32(select(uy, uy - 256, uy > 128));
  let twoPiOverL = 6.28318530718 / ${WAVE_DOMAIN_LEN};
  let kx = mx * twoPiOverL;
  let kz = mz * twoPiOverL;
  let k2 = kx * kx + kz * kz;
  let k = sqrt(k2);

  let dt = clamp(u.viewportAndDt.z, 0.0, 0.025);
  let depthH = u.domainBounds.z - u.domainBounds.y; // 1.28 tank depth
  let tanhKH = tanh(min(k * depthH, 12.0));

  // Physical elastocapillary dispersion in cuvette world units (resolving fine 2-8 mm capillary precursor wavelets):
  // omega(k_crater=34) ~ 11.1 rad/s (half-period T/2 ~ 0.28s -> crisp central swell rebound + dispersive capillary train)
  let gEff = 0.84;
  let sigmaOverRho = 0.0024;
  let omega = sqrt(max((gEff * k + sigmaOverRho * k * k2) * tanhKH, 1e-8));

  // Physical slow-motion viscous capillary damping + smooth high-k spectral de-aliasing roll-off near k_Nyquist = 574
  let viscRate = 0.030 + 0.000028 * k2;
  let nyquistRoll = 1.0 - smoothstep(400.0, 555.0, k);
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
 * evaluates dimensionless 9-point isotropic Laplacian into .b (for refractive caustics & Young-Laplace top pressure),
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

  let lap = 0.20 * (n + s + e + w) + 0.05 * (ne + nw + se + sw) - prop.x;

  let uv = (vec2<f32>(f32(ix), f32(iy)) + 0.5) / ${WAVE_GRID_SIZE.toFixed(1)};
  let worldXZ = waveUVToWorldXZ(uv);
  let sponge = 1.0 - smoothstep(${WAVE_SPONGE_INNER}, ${WAVE_SPONGE_OUTER}, length(worldXZ)) + u.surfaceParams.x * 0.0;

  let etaOut = clamp(prop.x * sponge, -0.14, 0.14);
  let velOut = clamp(prop.y * sponge, -1.8, 1.8);
  textureStore(waveOut, vec2<i32>(ix, iy), vec4<f32>(etaOut, velOut, lap * sponge, film * sponge));
}
`;
