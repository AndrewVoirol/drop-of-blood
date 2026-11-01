import {
  COMMON_WGSL,
  MAX_ACTIVE_DROPS,
  VEL_GRID_X,
  VEL_GRID_Y,
  VEL_GRID_Z,
} from './common';

/**
 * Shared two-way free-surface Dirichlet top pressure helper (used identically in MAC RHS and MAC Project):
 * Evaluates dimensionless pressure p_tilde = (dt / h) * (p / rho) on the top ghost plane j = VEL_GRID_Y
 * (y = WATER_SURFACE_Y = +0.30) from:
 *   1. Live dispersive wave hydrostatic + Young-Laplace capillary curvature pressure
 *   2. Active multi-drop wetting-neck capillary & dynamic stagnation overpressure
 */
const TOP_DIRICHLET_PRESSURE_WGSL = /* wgsl */ `
fn evalTopDirichletPressure(
  ix: i32,
  iz: i32,
  u: SimUniforms,
  waveTex: texture_2d<f32>,
  linSampler: sampler
) -> f32 {
  let halfXZ = u.domainBounds.x;
  let fx = (f32(ix) + 0.5) / ${VEL_GRID_X}.0;
  let fz = (f32(iz) + 0.5) / ${VEL_GRID_Z}.0;
  let worldXZ = vec2<f32>(
    mix(-halfXZ, halfXZ, fx),
    mix(-halfXZ, halfXZ, fz)
  );

  let dt = clamp(u.viewportAndDt.z, 0.0, 0.025);
  let cellH = (2.0 * halfXZ) / ${VEL_GRID_X}.0; // 0.01 exact cubic cell
  let dtOverH = dt / cellH;

  let waveUV = clamp(worldXZToWaveUV(worldXZ), vec2<f32>(0.002), vec2<f32>(0.998));
  let wSample = textureSampleLevel(waveTex, linSampler, waveUV, 0.0);
  let eta = wSample.r;
  let lapDimless = wSample.b; // ~ 0.25 * dx_wave^2 * nabla^2(eta)

  // Hydrostatic g_eff * eta - Young-Laplace (sigma/rho) * nabla^2(eta)
  // Depressed crater (eta < 0) creates low pressure that hydrodynamically drives cavity collapse & Worthington rebound!
  let gEff = 0.155;
  let capTerm = -18.4 * lapDimless;
  var pKinematic = 2.45 * (gEff * eta + 0.00095 * capTerm);

  // Active wetting-neck coalescence capillary + stagnation overpressure across all 8 drops
  for (var d = 0u; d < ${MAX_ACTIVE_DROPS}u; d = d + 1u) {
    let slot = u.drops[d];
    let signedSeed = slot.velOscAndDrain.w;
    let drainRate = slot.velOscAndDrain.z;
    if (signedSeed > 0.0 && drainRate > 0.0005) {
      let dropXZ = slot.posAndRadius.xz;
      let rel = worldXZ - dropXZ;
      let r = length(rel);
      let theta = atan2(rel.y, rel.x);
      let seed = signedSeed;

      // Even Widnall m=4,6,8 azimuthal modes break axisymmetry with identically zero m=1 lateral dipole drift
      let azim = 1.0
        + 0.022 * cos(4.0 * theta + seed * 1.73)
        + 0.016 * cos(6.0 * theta - seed * 2.31)
        + 0.010 * sin(8.0 * theta + seed * 0.89);

      let slugPhase = smoothstep(0.32, 0.92, drainRate);
      let rOrifice = clamp(slot.posAndRadius.w * 1.12 + mix(0.018, 0.032, slugPhase), 0.030, 0.090) * azim;
      let rNorm = r / max(rOrifice, 1e-4);
      let speedY = clamp(-slot.velOscAndDrain.x, 0.08, 0.65);
      // Annular-peaked crater overpressure + outer rim suction rolls up an oblate toroidal vortex umbrella (zero axial V-spike!)
      let neckProfile = (0.28 + 1.05 * rNorm * rNorm) * exp(-0.92 * rNorm * rNorm) * (1.0 - 0.38 * rNorm * rNorm);
      pKinematic = pKinematic + drainRate * (0.014 + 0.048 * speedY) * neckProfile;
    }
  }

  let wallTaper = 1.0 - smoothstep(halfXZ * 0.72, halfXZ * 0.96, max(abs(worldXZ.x), abs(worldXZ.y)));
  return pKinematic * dtOverH * wallTaper;
}
`;

export const FLUID_INIT_VEL_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var velOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${VEL_GRID_X}u || gid.y >= ${VEL_GRID_Y}u || gid.z >= ${VEL_GRID_Z}u) {
    return;
  }
  let dims = vec3<f32>(${VEL_GRID_X}.0, ${VEL_GRID_Y}.0, ${VEL_GRID_Z}.0);
  let seed = u.surfaceParams.w;
  let surfY = u.surfaceParams.x;
  let halfXZ = u.domainBounds.x;

  // Staggered face positions: X-face at i+1, Y-face at j+1, Z-face at k+1
  let pX = uvwToWorld((vec3<f32>(f32(gid.x) + 1.0, f32(gid.y) + 0.5, f32(gid.z) + 0.5)) / dims, u);
  let pY = uvwToWorld((vec3<f32>(f32(gid.x) + 0.5, f32(gid.y) + 1.0, f32(gid.z) + 0.5)) / dims, u);
  let pZ = uvwToWorld((vec3<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5, f32(gid.z) + 1.0)) / dims, u);

  let fadeX = (1.0 - smoothstep(halfXZ * 0.55, halfXZ * 0.96, max(abs(pX.x), abs(pX.z))))
            * (1.0 - smoothstep(surfY - 0.12, surfY, pX.y))
            * smoothstep(0.06, 0.24, length(pX.xz));
  let fadeY = (1.0 - smoothstep(halfXZ * 0.55, halfXZ * 0.96, max(abs(pY.x), abs(pY.z))))
            * (1.0 - smoothstep(surfY - 0.12, surfY, pY.y))
            * smoothstep(0.06, 0.24, length(pY.xz));
  let fadeZ = (1.0 - smoothstep(halfXZ * 0.55, halfXZ * 0.96, max(abs(pZ.x), abs(pZ.z))))
            * (1.0 - smoothstep(surfY - 0.12, surfY, pZ.y))
            * smoothstep(0.06, 0.24, length(pZ.xz));

  // Strictly odd under (x, z) -> (-x, -z): zero net horizontal crossflow dipole at every depth
  var vx =  0.00045 * sin(pX.x * 6.5) * cos(pX.z * 6.5) * sin(pX.y * 5.2 + seed * 1.3) * fadeX;
  var vy =  0.00030 * cos(pY.x * 6.5) * sin(pY.z * 6.5) * cos(pY.y * 5.2 + seed * 0.7) * fadeY;
  var vz = -0.00045 * cos(pZ.x * 6.5) * sin(pZ.z * 6.5) * sin(pZ.y * 5.2 + seed * 1.3) * fadeZ;

  if (gid.x >= ${VEL_GRID_X - 1}u) { vx = 0.0; }
  if (gid.z >= ${VEL_GRID_Z - 1}u) { vz = 0.0; }

  textureStore(velOut, vec3<i32>(gid), vec4<f32>(vx, vy, vz, 0.0));
}
`;

export const FLUID_CLEAR_R32_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var scalarOut: texture_storage_3d<r32float, write>;

@compute @workgroup_size(8, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(scalarOut);
  if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
    return;
  }
  let zeroVal = u.surfaceParams.x * 0.0;
  textureStore(scalarOut, vec3<i32>(gid), vec4<f32>(zeroVal, 0.0, 0.0, 0.0));
}
`;

/**
 * Staggered MAC Semi-Lagrangian RK2 velocity advection + wetting-neck liquid entry momentum continuity
 * + Marangoni surfactant tangential stress on the top free-surface layer.
 */
export const FLUID_ADVECT_MAC_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var velMacIn: texture_3d<f32>;
@group(0) @binding(2) var velCenterIn: texture_3d<f32>;
@group(0) @binding(3) var waveIn: texture_2d<f32>;
@group(0) @binding(4) var linSampler: sampler;
@group(0) @binding(5) var velMacOut: texture_storage_3d<rgba16float, write>;

fn sampleMacU(uvw: vec3<f32>) -> f32 {
  let shifted = clamp(uvw - vec3<f32>(0.5 / ${VEL_GRID_X}.0, 0.0, 0.0), vec3<f32>(0.001), vec3<f32>(0.999));
  return textureSampleLevel(velMacIn, linSampler, shifted, 0.0).x;
}

fn sampleMacV(uvw: vec3<f32>) -> f32 {
  let shifted = clamp(uvw - vec3<f32>(0.0, 0.5 / ${VEL_GRID_Y}.0, 0.0), vec3<f32>(0.001), vec3<f32>(0.999));
  return textureSampleLevel(velMacIn, linSampler, shifted, 0.0).y;
}

fn sampleMacW(uvw: vec3<f32>) -> f32 {
  let shifted = clamp(uvw - vec3<f32>(0.0, 0.0, 0.5 / ${VEL_GRID_Z}.0), vec3<f32>(0.001), vec3<f32>(0.999));
  return textureSampleLevel(velMacIn, linSampler, shifted, 0.0).z;
}

fn traceBackUVW(uvwFace: vec3<f32>, dt: f32, domSize: vec3<f32>) -> vec3<f32> {
  let v0 = textureSampleLevel(velCenterIn, linSampler, clamp(uvwFace, vec3<f32>(0.002), vec3<f32>(0.998)), 0.0).xyz;
  let uvwMid = clamp(uvwFace - 0.5 * (v0 * dt) / domSize, vec3<f32>(0.002), vec3<f32>(0.998));
  let vMid = textureSampleLevel(velCenterIn, linSampler, uvwMid, 0.0).xyz;
  return clamp(uvwFace - (vMid * dt) / domSize, vec3<f32>(0.002), vec3<f32>(0.998));
}

@compute @workgroup_size(8, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${VEL_GRID_X}u || gid.y >= ${VEL_GRID_Y}u || gid.z >= ${VEL_GRID_Z}u) {
    return;
  }
  let dims = vec3<f32>(${VEL_GRID_X}.0, ${VEL_GRID_Y}.0, ${VEL_GRID_Z}.0);
  let dt = clamp(u.viewportAndDt.z, 0.0, 0.025);
  let domSize = domainSize(u);
  let damp = u.fluidParams.z;

  let uvwX = vec3<f32>(f32(gid.x) + 1.0, f32(gid.y) + 0.5, f32(gid.z) + 0.5) / dims;
  let uvwY = vec3<f32>(f32(gid.x) + 0.5, f32(gid.y) + 1.0, f32(gid.z) + 0.5) / dims;
  let uvwZ = vec3<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5, f32(gid.z) + 1.0) / dims;

  var vx = sampleMacU(traceBackUVW(uvwX, dt, domSize)) * damp;
  var vy = sampleMacV(traceBackUVW(uvwY, dt, domSize)) * damp;
  var vz = sampleMacW(traceBackUVW(uvwZ, dt, domSize)) * damp;

  let pCell = uvwToWorld((vec3<f32>(gid) + 0.5) / dims, u);
  let pFaceY = uvwToWorld(uvwY, u);
  let surfY = u.surfaceParams.x;

  // Wetting-neck momentum continuity across all 8 concurrent drops:
  // Primary slug uses an annular-boosted vortex-sheet piston profile (forming an oblate rolled toroidal umbrella),
  // while the late capillary wake transitions smoothly to a gentle non-overtaking stem flow (zero centerline V-tip!).
  var activeNeckMask = 0.0;
  for (var d = 0u; d < ${MAX_ACTIVE_DROPS}u; d = d + 1u) {
    let slot = u.drops[d];
    let signedSeed = slot.velOscAndDrain.w;
    let drainRate = slot.velOscAndDrain.z;
    if (signedSeed > 0.0 && drainRate > 0.0005) {
      let dropXZ = slot.posAndRadius.xz;
      let relXZ = pFaceY.xz - dropXZ;
      let r = length(relXZ);
      let theta = atan2(relXZ.y, relXZ.x);
      let seed = signedSeed;

      let azim = 1.0
        + 0.036 * cos(4.0 * theta + seed * 1.73)
        + 0.026 * cos(6.0 * theta - seed * 2.31)
        + 0.018 * sin(8.0 * theta + seed * 0.89);

      let slugPhase = smoothstep(0.14, 0.76, drainRate);
      let rNeckBase = clamp(slot.posAndRadius.w * mix(0.80, 1.08, slugPhase) + mix(0.012, 0.030, slugPhase), 0.018, 0.088);
      let rNeck = clamp(slot.posAndRadius.w * mix(0.80, 1.08, slugPhase) + mix(0.012, 0.030, slugPhase), 0.018, 0.088) * azim;
      let rRatio = r / max(rNeck, 1e-4);

      let slugRadial = (0.14 + 1.00 * min(rRatio * rRatio, 1.20)) * (1.0 - smoothstep(0.78, 1.15, rRatio));
      let stemRadial = (0.68 + 0.32 * min(rRatio * rRatio, 1.0)) * (1.0 - smoothstep(0.74, 1.10, rRatio));
      let radialProfile = mix(stemRadial, slugRadial, slugPhase);

      let vertMask = smoothstep(surfY - 0.058, surfY - 0.014, pFaceY.y);
      let meniscusShear = mix(1.0, 0.36, smoothstep(surfY - 0.024, surfY - 0.002, pFaceY.y));
      let mask = radialProfile * vertMask;
      activeNeckMask = max(activeNeckMask, radialProfile);

      if (mask > 0.001) {
        let lobeMod = 1.0 + 0.036 * cos(4.0 * theta + seed * 1.73) + 0.026 * cos(6.0 * theta - seed * 2.31);
        let impactSpeed = clamp(-slot.velOscAndDrain.x, 0.10, 0.65);
        // Collar wake draft feeds upper stem into the vortex ring's Biot-Savart draft without shooting a centerline jet through the torus floor
        let slugDrive = clamp(pow(max(drainRate, 0.0), 0.66) * 0.84, 0.0, 1.24);
        let stemDrive = clamp(0.045 + drainRate * 0.22, 0.0, 0.20) * smoothstep(0.0, 0.05, drainRate);
        let fluxDrive = mix(stemDrive, slugDrive, slugPhase);
        let targetVy = -clamp(0.14 + 0.64 * impactSpeed, 0.15, 0.39) * fluxDrive * lobeMod * meniscusShear;
        let blend = clamp(dt * mix(14.0, 22.0, slugPhase) * mask, 0.0, 1.0);
        vy = mix(vy, targetVy, blend);
      }

      // Outward crater-base radial roll-out widens the toroidal umbrella cap; gentle collar pull near meniscus.
      // Evaluated at each staggered X/Z face's true world coordinate (uvwX, uvwZ) to preserve exact XZ symmetry (zero +X/+Z dipole!).
      let vertMaskXZ = smoothstep(surfY - 0.058, surfY - 0.014, pCell.y);
      if (vertMaskXZ > 0.001) {
        let flare = mix(0.104 * (0.35 + 0.65 * slugPhase), -0.020, smoothstep(surfY - 0.066, surfY - 0.012, pCell.y)) * clamp(drainRate, 0.0, 1.3);
        let relXFace = uvwToWorld(uvwX, u).xz - dropXZ;
        let rXF = length(relXFace);
        if (rXF > 1e-4 && rXF < rNeckBase * 1.34) {
          let thetaX = atan2(relXFace.y, relXFace.x);
          let azimX = 1.0
            + 0.036 * cos(4.0 * thetaX + seed * 1.73)
            + 0.026 * cos(6.0 * thetaX - seed * 2.31)
            + 0.018 * sin(8.0 * thetaX + seed * 0.89);
          let rRatioX = rXF / max(rNeckBase * azimX, 1e-4);
          if (rRatioX < 1.22) {
            let slugRadX = (0.14 + 1.00 * min(rRatioX * rRatioX, 1.20)) * (1.0 - smoothstep(0.78, 1.15, rRatioX));
            let stemRadX = (0.68 + 0.32 * min(rRatioX * rRatioX, 1.0)) * (1.0 - smoothstep(0.74, 1.10, rRatioX));
            let maskX = mix(stemRadX, slugRadX, slugPhase) * vertMaskXZ;
            let blendX = clamp(dt * mix(14.0, 22.0, slugPhase) * maskX, 0.0, 1.0);
            vx = mix(vx, (relXFace.x / rXF) * flare, blendX * 0.65);
          }
        }
        let relZFace = uvwToWorld(uvwZ, u).xz - dropXZ;
        let rZF = length(relZFace);
        if (rZF > 1e-4 && rZF < rNeckBase * 1.34) {
          let thetaZ = atan2(relZFace.y, relZFace.x);
          let azimZ = 1.0
            + 0.036 * cos(4.0 * thetaZ + seed * 1.73)
            + 0.026 * cos(6.0 * thetaZ - seed * 2.31)
            + 0.018 * sin(8.0 * thetaZ + seed * 0.89);
          let rRatioZ = rZF / max(rNeckBase * azimZ, 1e-4);
          if (rRatioZ < 1.22) {
            let slugRadZ = (0.14 + 1.00 * min(rRatioZ * rRatioZ, 1.20)) * (1.0 - smoothstep(0.78, 1.15, rRatioZ));
            let stemRadZ = (0.68 + 0.32 * min(rRatioZ * rRatioZ, 1.0)) * (1.0 - smoothstep(0.74, 1.10, rRatioZ));
            let maskZ = mix(stemRadZ, slugRadZ, slugPhase) * vertMaskXZ;
            let blendZ = clamp(dt * mix(14.0, 22.0, slugPhase) * maskZ, 0.0, 1.0);
            vz = mix(vz, (relZFace.y / rZF) * flare, blendZ * 0.65);
          }
        }
      }
    }
  }

  // Marangoni surface-tension tangential pull + kinematic free-surface boundary condition (v_y|_{y=eta} -> d_eta/dt)
  // Once wetting-neck drainage finishes, the calm meniscus acts as a physical hyperbolic stagnation boundary so the
  // central crimson stem stretches continuously from y = +0.30 into the descending vortex ring without detaching!
  if (gid.y >= ${VEL_GRID_Y - 3}u) {
    let topFade = f32(gid.y - ${VEL_GRID_Y - 3}u + 1u) / 3.0;
    let epsW = 0.015;
    let wCenter = textureSampleLevel(waveIn, linSampler, clamp(worldXZToWaveUV(pCell.xz), vec2<f32>(0.002), vec2<f32>(0.998)), 0.0);
    let wC  = wCenter.a;
    let wXp = textureSampleLevel(waveIn, linSampler, clamp(worldXZToWaveUV(pCell.xz + vec2<f32>(epsW, 0.0)), vec2<f32>(0.002), vec2<f32>(0.998)), 0.0).a;
    let wXn = textureSampleLevel(waveIn, linSampler, clamp(worldXZToWaveUV(pCell.xz - vec2<f32>(epsW, 0.0)), vec2<f32>(0.002), vec2<f32>(0.998)), 0.0).a;
    let wZp = textureSampleLevel(waveIn, linSampler, clamp(worldXZToWaveUV(pCell.xz + vec2<f32>(0.0, epsW)), vec2<f32>(0.002), vec2<f32>(0.998)), 0.0).a;
    let wZn = textureSampleLevel(waveIn, linSampler, clamp(worldXZToWaveUV(pCell.xz - vec2<f32>(0.0, epsW)), vec2<f32>(0.002), vec2<f32>(0.998)), 0.0).a;
    let cmcWindow = 1.0 - smoothstep(0.45, 0.88, wC);
    let gradFilm = vec2<f32>(wXp - wXn, wZp - wZn) / (2.0 * epsW);
    let marangoniPull = -u.surfaceParams.y * 0.026 * cmcWindow * gradFilm * topFade;
    vx = vx + marangoniPull.x * dt;
    vz = vz + marangoniPull.y * dt;

    if (gid.y >= ${VEL_GRID_Y - 2}u && activeNeckMask < 0.02) {
      let calmGate = 1.0 - smoothstep(0.020, 0.14, abs(wCenter.g));
      let layerWeight = select(0.26, 0.72, gid.y == ${VEL_GRID_Y - 1}u);
      vy = mix(vy, clamp(wCenter.g * 0.35, -0.045, 0.045), calmGate * layerWeight);
    }
  }

  // Soft outer tank wall & thin viscous floor boundary layer (allowing RBC plumes to reach and spread on y = -0.98)
  let halfXZ = u.domainBounds.x;
  let yMin = u.domainBounds.y;
  let wallFade = (1.0 - smoothstep(halfXZ * 0.85, halfXZ * 0.99, max(abs(pCell.x), abs(pCell.z))))
               * smoothstep(yMin + 0.004, yMin + 0.052, pCell.y);
  let spongeMul = mix(0.88, 1.0, wallFade);
  vx = vx * spongeMul;
  vy = vy * spongeMul;
  vz = vz * spongeMul;

  // Enforce impenetrable normal velocity on the 5 solid tank walls (top j=127 remains a FREE surface!)
  if (gid.x >= ${VEL_GRID_X - 1}u) { vx = 0.0; }
  if (gid.z >= ${VEL_GRID_Z - 1}u) { vz = 0.0; }

  textureStore(velMacOut, vec3<i32>(gid), vec4<f32>(vx, vy, vz, 0.0));
}
`;

export const FLUID_CURL_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var velCenterIn: texture_3d<f32>;
@group(0) @binding(2) var curlOut: texture_storage_3d<rgba16float, write>;

fn loadVelC(x: i32, y: i32, z: i32) -> vec3<f32> {
  let cx = clamp(x, 0, ${VEL_GRID_X - 1});
  let cy = clamp(y, 0, ${VEL_GRID_Y - 1});
  let cz = clamp(z, 0, ${VEL_GRID_Z - 1});
  return textureLoad(velCenterIn, vec3<i32>(cx, cy, cz), 0).xyz;
}

@compute @workgroup_size(8, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${VEL_GRID_X}u || gid.y >= ${VEL_GRID_Y}u || gid.z >= ${VEL_GRID_Z}u) {
    return;
  }
  let ix = i32(gid.x);
  let iy = i32(gid.y);
  let iz = i32(gid.z);

  let vR = loadVelC(ix + 1, iy, iz);
  let vL = loadVelC(ix - 1, iy, iz);
  let vU = loadVelC(ix, iy + 1, iz);
  let vD = loadVelC(ix, iy - 1, iz);
  let vF = loadVelC(ix, iy, iz + 1);
  let vB = loadVelC(ix, iy, iz - 1);

  let omega = 0.5 * vec3<f32>(
    (vU.z - vD.z) - (vF.y - vB.y),
    (vF.x - vB.x) - (vR.z - vL.z),
    (vR.y - vL.y) - (vU.x - vD.x)
  );
  let mag = length(omega) + u.fluidParams.x * 0.0;
  textureStore(curlOut, vec3<i32>(ix, iy, iz), vec4<f32>(omega, mag));
}
`;

/**
 * Applies:
 *  1. Regularized Casson viscoplastic shear-thinning stress divergence mu(c_RBC, ||D||) (rouleaux cohesion at low shear!)
 *  2. Differential Boussinesq negative buoyancy (dense intact RBCs vs diluted lysed Hb -> crisp RT tendril bifurcations!)
 *  3. Sub-grid vorticity confinement preserving thin rolled vortex cores without beading calm upper stems
 */
export const FLUID_CASSON_FORCES_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var velMacIn: texture_3d<f32>;
@group(0) @binding(2) var curlIn: texture_3d<f32>;
@group(0) @binding(3) var concIn: texture_3d<f32>;
@group(0) @binding(4) var linSampler: sampler;
@group(0) @binding(5) var velMacOut: texture_storage_3d<rgba16float, write>;

fn loadMac(x: i32, y: i32, z: i32) -> vec3<f32> {
  let cx = clamp(x, 0, ${VEL_GRID_X - 1});
  let cy = clamp(y, 0, ${VEL_GRID_Y - 1});
  let cz = clamp(z, 0, ${VEL_GRID_Z - 1});
  return textureLoad(velMacIn, vec3<i32>(cx, cy, cz), 0).xyz;
}

fn loadCurl(x: i32, y: i32, z: i32) -> vec4<f32> {
  let cx = clamp(x, 0, ${VEL_GRID_X - 1});
  let cy = clamp(y, 0, ${VEL_GRID_Y - 1});
  let cz = clamp(z, 0, ${VEL_GRID_Z - 1});
  return textureLoad(curlIn, vec3<i32>(cx, cy, cz), 0);
}

fn evalVortConf(ix: i32, iy: i32, iz: i32, vcScale: f32) -> vec3<f32> {
  if (ix <= 1 || ix >= ${VEL_GRID_X - 2} || iy <= 1 || iy >= ${VEL_GRID_Y - 3} || iz <= 1 || iz >= ${VEL_GRID_Z - 2}) {
    return vec3<f32>(0.0);
  }
  let curlC = loadCurl(ix, iy, iz);
  let omega = curlC.xyz;
  let omegaMag = curlC.w;
  if (omegaMag <= 0.18) {
    return vec3<f32>(0.0);
  }
  // Suppress numerical KH beading on calm stems and saturate multi-drop high-omega sheets for smooth laminar curtains!
  let ringGate = smoothstep(0.18, 0.95, omegaMag) / (1.0 + 0.28 * omegaMag);
  let eta = 0.5 * vec3<f32>(
    loadCurl(ix + 1, iy, iz).w - loadCurl(ix - 1, iy, iz).w,
    loadCurl(ix, iy + 1, iz).w - loadCurl(ix, iy - 1, iz).w,
    loadCurl(ix, iy, iz + 1).w - loadCurl(ix, iy, iz - 1).w
  );
  let etaLen = length(eta);
  if (etaLen <= 1e-5) {
    return vec3<f32>(0.0);
  }
  return (vcScale * ringGate) * cross(eta / etaLen, omega);
}

@compute @workgroup_size(8, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${VEL_GRID_X}u || gid.y >= ${VEL_GRID_Y}u || gid.z >= ${VEL_GRID_Z}u) {
    return;
  }
  let ix = i32(gid.x);
  let iy = i32(gid.y);
  let iz = i32(gid.z);
  let dims = vec3<f32>(${VEL_GRID_X}.0, ${VEL_GRID_Y}.0, ${VEL_GRID_Z}.0);
  let dt = clamp(u.viewportAndDt.z, 0.0, 0.025);

  var mac = loadMac(ix, iy, iz);
  let mXp = loadMac(ix + 1, iy, iz);
  let mXn = loadMac(ix - 1, iy, iz);
  let mYp = loadMac(ix, iy + 1, iz);
  let mYn = loadMac(ix, iy - 1, iz);
  let mZp = loadMac(ix, iy, iz + 1);
  let mZn = loadMac(ix, iy, iz - 1);

  // 1. Casson viscoplastic shear-thinning viscosity mu_Casson(c_RBC, gamma_dot):
  //    mu_eff = (sqrt(mu_inf(H)) + sqrt(tau_y(H) / sqrt(gamma_dot^2 + eps^2)))^2
  let uvwCell = (vec3<f32>(gid) + 0.5) / dims;
  let concCell = textureSampleLevel(concIn, linSampler, uvwCell, 0.0);
  let cRbc = clamp(concCell.r, 0.0, 1.5); // Intact erythrocytes carry rouleaux yield stress!

  // Local velocity-difference strain-rate proxy ||D|| * h -> physical shear rate ||D|| = diff / 0.01
  let du = mXp - mXn;
  let dv = mYp - mYn;
  let dw = mZp - mZn;
  let shearRate = 50.0 * sqrt(dot(du, du) + dot(dv, dv) + dot(dw, dw) + 1e-6);

  // Calibrated Casson rouleaux cohesion stabilizes low-shear hairline threads while allowing
  // shear-thinned secondary Rayleigh-Taylor tendrils to bifurcate cleanly at late times!
  let cassonScale = u.lensAndRheology.z;
  let sqrtMuInf = sqrt(0.00010 * (1.0 + 1.35 * cRbc));
  let tauYield = 0.0036 * cassonScale * cRbc * cRbc;
  let sqrtYieldTerm = sqrt(tauYield / max(shearRate + 0.40, 0.40));
  let cassonDiffAlpha = clamp((sqrtMuInf + sqrtYieldTerm) * (sqrtMuInf + sqrtYieldTerm), 0.00010, 0.0044);

  let lapMac = (mXp + mXn + mYp + mYn + mZp + mZn) - 6.0 * mac;
  mac = mac + cassonDiffAlpha * lapMac;

  // 2. Staggered face vorticity confinement (averaged onto X, Y, Z faces)
  let vc0 = evalVortConf(ix, iy, iz, u.fluidParams.x);
  let vcX = 0.5 * (vc0.x + evalVortConf(min(ix + 1, ${VEL_GRID_X - 1}), iy, iz, u.fluidParams.x).x);
  let vcY = 0.5 * (vc0.y + evalVortConf(ix, min(iy + 1, ${VEL_GRID_Y - 1}), iz, u.fluidParams.x).y);
  let vcZ = 0.5 * (vc0.z + evalVortConf(ix, iy, min(iz + 1, ${VEL_GRID_Z - 1}), u.fluidParams.x).z);
  mac = mac + vec3<f32>(vcX, vcY, vcZ) * dt;

  // 3. Differential Boussinesq negative buoyancy on staggered Y-faces (zero synthetic sine lattice!):
  //    Intact erythrocytes (rho ~ 1.11 g/cm^3) drive sharp baroclinic torque nabla(rho) x g into bifurcating
  //    Rayleigh-Taylor tendrils while diluted lysed hemoglobin trails gently as a translucent rose-ruby halo!
  let uvwFaceY = vec3<f32>(f32(ix) + 0.5, f32(iy) + 1.0, f32(iz) + 0.5) / dims;
  let concFaceY = textureSampleLevel(concIn, linSampler, clamp(uvwFaceY, vec3<f32>(0.001), vec3<f32>(0.999)), 0.0);
  let buoyDensity = 1.24 * concFaceY.r + 0.26 * concFaceY.g;
  if (buoyDensity > 0.0008) {
    let pFaceY = uvwToWorld(uvwFaceY, u);
    let surfBuoyGate = smoothstep(0.002, 0.024, max(0.0, u.surfaceParams.x - pFaceY.y));
    let effBuoyConc = 1.24 * tanh(buoyDensity / 1.24);
    let buoyAccel = -u.fluidParams.y * 0.345 * effBuoyConc * surfBuoyGate;
    mac.y = mac.y + buoyAccel * dt;
  }

  mac = clamp(mac, vec3<f32>(-1.8), vec3<f32>(1.8));
  if (ix >= ${VEL_GRID_X - 1}) { mac.x = 0.0; }
  if (iz >= ${VEL_GRID_Z - 1}) { mac.z = 0.0; }

  textureStore(velMacOut, vec3<i32>(ix, iy, iz), vec4<f32>(mac, 0.0));
}
`;

/**
 * Exact 1-cell staggered MAC divergence D_{i,j,k} minus inhomogeneous top free-surface Dirichlet pressure p_top:
 *   L_0(p) = D - delta_{j, Ny-1} * p_top
 */
export const FLUID_MAC_RHS_WGSL = /* wgsl */ `
${COMMON_WGSL}
${TOP_DIRICHLET_PRESSURE_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var velMacIn: texture_3d<f32>;
@group(0) @binding(2) var waveIn: texture_2d<f32>;
@group(0) @binding(3) var linSampler: sampler;
@group(0) @binding(4) var rhsOut: texture_storage_3d<r32float, write>;

@compute @workgroup_size(8, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${VEL_GRID_X}u || gid.y >= ${VEL_GRID_Y}u || gid.z >= ${VEL_GRID_Z}u) {
    return;
  }
  let ix = i32(gid.x);
  let iy = i32(gid.y);
  let iz = i32(gid.z);

  let mC = textureLoad(velMacIn, vec3<i32>(ix, iy, iz), 0).xyz;
  let uRight = select(mC.x, 0.0, ix >= ${VEL_GRID_X - 1});
  let uLeft  = select(textureLoad(velMacIn, vec3<i32>(max(ix - 1, 0), iy, iz), 0).x, 0.0, ix == 0);

  let vTop   = mC.y; // Free surface at iy == VEL_GRID_Y - 1!
  let vBot   = select(textureLoad(velMacIn, vec3<i32>(ix, max(iy - 1, 0), iz), 0).y, 0.0, iy == 0);

  let wFront = select(mC.z, 0.0, iz >= ${VEL_GRID_Z - 1});
  let wBack  = select(textureLoad(velMacIn, vec3<i32>(ix, iy, max(iz - 1, 0)), 0).z, 0.0, iz == 0);

  var rhs = (uRight - uLeft) + (vTop - vBot) + (wFront - wBack);

  // Move inhomogeneous Dirichlet free-surface ghost pressure p_{Ny} = p_top to RHS!
  if (iy == ${VEL_GRID_Y - 1}) {
    rhs = rhs - evalTopDirichletPressure(ix, iz, u, waveIn, linSampler);
  }

  textureStore(rhsOut, vec3<i32>(ix, iy, iz), vec4<f32>(rhs, 0.0, 0.0, 0.0));
}
`;

/**
 * Shared Multigrid homogeneous Poisson stencil helper (5 solid Neumann walls + 1 top Dirichlet free surface):
 * Works identically across L0 (96x128x96), L1 (48x64x48), L2 (24x32x24), and L3 (12x16x12)!
 */
const MG_STENCIL_HELPER_WGSL = /* wgsl */ `
struct MgStencil {
  sumNeighbors: f32,
  diagCount: f32,
  h2: f32,
};

fn evalMgStencil(pTex: texture_3d<f32>, ix: i32, iy: i32, iz: i32, dims: vec3<i32>) -> MgStencil {
  var sumN = 0.0;
  var diag = 1.0; // Top face is ALWAYS active (internal for iy < dims.y - 1, homogeneous Dirichlet 0 at iy == dims.y - 1)

  if (ix > 0) {
    sumN = sumN + textureLoad(pTex, vec3<i32>(ix - 1, iy, iz), 0).r;
    diag = diag + 1.0;
  }
  if (ix < dims.x - 1) {
    sumN = sumN + textureLoad(pTex, vec3<i32>(ix + 1, iy, iz), 0).r;
    diag = diag + 1.0;
  }
  if (iy > 0) {
    sumN = sumN + textureLoad(pTex, vec3<i32>(ix, iy - 1, iz), 0).r;
    diag = diag + 1.0;
  }
  if (iy < dims.y - 1) {
    sumN = sumN + textureLoad(pTex, vec3<i32>(ix, iy + 1, iz), 0).r;
  }
  if (iz > 0) {
    sumN = sumN + textureLoad(pTex, vec3<i32>(ix, iy, iz - 1), 0).r;
    diag = diag + 1.0;
  }
  if (iz < dims.z - 1) {
    sumN = sumN + textureLoad(pTex, vec3<i32>(ix, iy, iz + 1), 0).r;
    diag = diag + 1.0;
  }

  let scaleH = ${VEL_GRID_X}.0 / f32(dims.x); // 1, 2, 4, 8
  var res: MgStencil;
  res.sumNeighbors = sumN;
  res.diagCount = diag;
  res.h2 = scaleH * scaleH;
  return res;
}
`;

function createMgRbgsShader(parity: 0 | 1, assumeZeroInput: boolean): string {
  if (assumeZeroInput) {
    return /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var rhsIn: texture_3d<f32>;
@group(0) @binding(2) var pOut: texture_storage_3d<r32float, write>;

@compute @workgroup_size(8, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dimsU = textureDimensions(rhsIn);
  if (gid.x >= dimsU.x || gid.y >= dimsU.y || gid.z >= dimsU.z) {
    return;
  }
  let ix = i32(gid.x);
  let iy = i32(gid.y);
  let iz = i32(gid.z);
  let dims = vec3<i32>(dimsU);

  if (((ix + iy + iz) & 1) != ${parity}) {
    textureStore(pOut, vec3<i32>(ix, iy, iz), vec4<f32>(0.0));
    return;
  }

  var diag = 1.0;
  if (ix > 0) { diag = diag + 1.0; }
  if (ix < dims.x - 1) { diag = diag + 1.0; }
  if (iy > 0) { diag = diag + 1.0; }
  if (iz > 0) { diag = diag + 1.0; }
  if (iz < dims.z - 1) { diag = diag + 1.0; }

  let scaleH = ${VEL_GRID_X}.0 / f32(dims.x);
  let h2 = scaleH * scaleH;
  let rhs = textureLoad(rhsIn, vec3<i32>(ix, iy, iz), 0).r;
  let val = (-h2 * rhs) / diag + u.domainBounds.x * 0.0;
  textureStore(pOut, vec3<i32>(ix, iy, iz), vec4<f32>(val, 0.0, 0.0, 0.0));
}
`;
  }

  return /* wgsl */ `
${COMMON_WGSL}
${MG_STENCIL_HELPER_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var pIn: texture_3d<f32>;
@group(0) @binding(2) var rhsIn: texture_3d<f32>;
@group(0) @binding(3) var pOut: texture_storage_3d<r32float, write>;

@compute @workgroup_size(8, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dimsU = textureDimensions(rhsIn);
  if (gid.x >= dimsU.x || gid.y >= dimsU.y || gid.z >= dimsU.z) {
    return;
  }
  let ix = i32(gid.x);
  let iy = i32(gid.y);
  let iz = i32(gid.z);
  let pCurr = textureLoad(pIn, vec3<i32>(ix, iy, iz), 0).r;

  if (((ix + iy + iz) & 1) != ${parity}) {
    textureStore(pOut, vec3<i32>(ix, iy, iz), vec4<f32>(pCurr, 0.0, 0.0, 0.0));
    return;
  }

  let st = evalMgStencil(pIn, ix, iy, iz, vec3<i32>(dimsU));
  let rhs = textureLoad(rhsIn, vec3<i32>(ix, iy, iz), 0).r;
  let gsVal = (st.sumNeighbors - st.h2 * rhs) / st.diagCount + u.domainBounds.x * 0.0;
  textureStore(pOut, vec3<i32>(ix, iy, iz), vec4<f32>(gsVal, 0.0, 0.0, 0.0));
}
`;
}

export const MG_RBGS_RED_ZERO_WGSL = createMgRbgsShader(0, true);
export const MG_RBGS_RED_WGSL = createMgRbgsShader(0, false);
export const MG_RBGS_BLACK_WGSL = createMgRbgsShader(1, false);

/**
 * Single-workgroup shared-memory coarse-grid solver for Level 3 (12x16x12 = 2,304 cells = 9,216 B < 16,384 B):
 * Executes 6 full Red-Black Gauss-Seidel sweeps in L1 workgroup shared memory in ONE dispatch instead of 8 global barriers!
 */
export const MG_COARSE_SOLVE_L3_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var rhsIn: texture_3d<f32>;
@group(0) @binding(2) var pOut: texture_storage_3d<r32float, write>;

var<workgroup> sErr: array<f32, 2304>;

fn l3Idx(ix: i32, iy: i32, iz: i32) -> i32 {
  return (iz * 16 + iy) * 12 + ix;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = i32(lid.x);
  for (var idx = tid; idx < 2304; idx = idx + 256) {
    sErr[idx] = 0.0;
  }
  workgroupBarrier();

  let h2 = 64.0; // (96 / 12)^2
  for (var sweep = 0; sweep < 6; sweep = sweep + 1) {
    for (var parity = 0; parity < 2; parity = parity + 1) {
      for (var idx = tid; idx < 2304; idx = idx + 256) {
        let ix = idx % 12;
        let iy = (idx / 12) % 16;
        let iz = idx / 192;
        if (((ix + iy + iz) & 1) == parity) {
          var sumN = 0.0;
          var diag = 1.0;
          if (ix > 0)  { sumN = sumN + sErr[l3Idx(ix - 1, iy, iz)]; diag = diag + 1.0; }
          if (ix < 11) { sumN = sumN + sErr[l3Idx(ix + 1, iy, iz)]; diag = diag + 1.0; }
          if (iy > 0)  { sumN = sumN + sErr[l3Idx(ix, iy - 1, iz)]; diag = diag + 1.0; }
          if (iy < 15) { sumN = sumN + sErr[l3Idx(ix, iy + 1, iz)]; }
          if (iz > 0)  { sumN = sumN + sErr[l3Idx(ix, iy, iz - 1)]; diag = diag + 1.0; }
          if (iz < 11) { sumN = sumN + sErr[l3Idx(ix, iy, iz + 1)]; diag = diag + 1.0; }
          let rhs = textureLoad(rhsIn, vec3<i32>(ix, iy, iz), 0).r;
          sErr[idx] = (sumN - h2 * rhs) / diag;
        }
      }
      workgroupBarrier();
    }
  }

  for (var idx = tid; idx < 2304; idx = idx + 256) {
    let ix = idx % 12;
    let iy = (idx / 12) % 16;
    let iz = idx / 192;
    textureStore(pOut, vec3<i32>(ix, iy, iz), vec4<f32>(sErr[idx] + u.domainBounds.x * 0.0, 0.0, 0.0, 0.0));
  }
}
`;

/**
 * Evaluates residual r_l = b_l - L_l(p_l) on level l
 */
export const MG_RESIDUAL_WGSL = /* wgsl */ `
${COMMON_WGSL}
${MG_STENCIL_HELPER_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var pIn: texture_3d<f32>;
@group(0) @binding(2) var rhsIn: texture_3d<f32>;
@group(0) @binding(3) var resOut: texture_storage_3d<r32float, write>;

@compute @workgroup_size(8, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dimsU = textureDimensions(rhsIn);
  if (gid.x >= dimsU.x || gid.y >= dimsU.y || gid.z >= dimsU.z) {
    return;
  }
  let ix = i32(gid.x);
  let iy = i32(gid.y);
  let iz = i32(gid.z);

  let pC = textureLoad(pIn, vec3<i32>(ix, iy, iz), 0).r;
  let rhs = textureLoad(rhsIn, vec3<i32>(ix, iy, iz), 0).r;
  let st = evalMgStencil(pIn, ix, iy, iz, vec3<i32>(dimsU));
  let lapScaled = (st.sumNeighbors - st.diagCount * pC) / st.h2;
  let residual = rhs - lapScaled + u.domainBounds.x * 0.0;
  textureStore(resOut, vec3<i32>(ix, iy, iz), vec4<f32>(residual, 0.0, 0.0, 0.0));
}
`;

/**
 * Full-weighting 2x2x2 cell-centered restriction R_l^{l+1}(resFine) -> rhsCoarse
 */
export const MG_RESTRICT_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var fineResIn: texture_3d<f32>;
@group(0) @binding(2) var coarseRhsOut: texture_storage_3d<r32float, write>;

@compute @workgroup_size(8, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let coarseDims = textureDimensions(coarseRhsOut);
  if (gid.x >= coarseDims.x || gid.y >= coarseDims.y || gid.z >= coarseDims.z) {
    return;
  }
  let base = vec3<i32>(gid) * 2;
  var sum = 0.0;
  for (var dz = 0; dz < 2; dz = dz + 1) {
    for (var dy = 0; dy < 2; dy = dy + 1) {
      for (var dx = 0; dx < 2; dx = dx + 1) {
        sum = sum + textureLoad(fineResIn, base + vec3<i32>(dx, dy, dz), 0).r;
      }
    }
  }
  let avg = sum * 0.125 + u.domainBounds.x * 0.0;
  textureStore(coarseRhsOut, vec3<i32>(gid), vec4<f32>(avg, 0.0, 0.0, 0.0));
}
`;

/**
 * Trilinear prolongation + coarse-grid correction p_l <- p_l + I_{l+1}^l(e_{l+1})
 * Uses manual 8-tap textureLoad interpolation on unfilterable r32float coarse error textures.
 */
export const MG_PROLONG_ADD_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var finePIn: texture_3d<f32>;
@group(0) @binding(2) var coarseErrIn: texture_3d<f32>;
@group(0) @binding(3) var finePOut: texture_storage_3d<r32float, write>;

fn loadCoarseErr(c: vec3<i32>, cDims: vec3<i32>) -> f32 {
  if (c.y >= cDims.y) {
    // Homogeneous Dirichlet error = 0 above top free surface!
    return 0.0;
  }
  let clamped = clamp(c, vec3<i32>(0), cDims - vec3<i32>(1));
  return textureLoad(coarseErrIn, clamped, 0).r;
}

@compute @workgroup_size(8, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let fineDims = textureDimensions(finePIn);
  if (gid.x >= fineDims.x || gid.y >= fineDims.y || gid.z >= fineDims.z) {
    return;
  }
  let cDims = vec3<i32>(textureDimensions(coarseErrIn));
  let uvw = (vec3<f32>(gid) + 0.5) / vec3<f32>(fineDims);
  let coord = uvw * vec3<f32>(cDims) - 0.5;
  let base = vec3<i32>(floor(coord));
  let f = coord - vec3<f32>(base);

  let c000 = loadCoarseErr(base + vec3<i32>(0, 0, 0), cDims);
  let c100 = loadCoarseErr(base + vec3<i32>(1, 0, 0), cDims);
  let c010 = loadCoarseErr(base + vec3<i32>(0, 1, 0), cDims);
  let c110 = loadCoarseErr(base + vec3<i32>(1, 1, 0), cDims);
  let c001 = loadCoarseErr(base + vec3<i32>(0, 0, 1), cDims);
  let c101 = loadCoarseErr(base + vec3<i32>(1, 0, 1), cDims);
  let c011 = loadCoarseErr(base + vec3<i32>(0, 1, 1), cDims);
  let c111 = loadCoarseErr(base + vec3<i32>(1, 1, 1), cDims);

  let x00 = mix(c000, c100, f.x);
  let x10 = mix(c010, c110, f.x);
  let x01 = mix(c001, c101, f.x);
  let x11 = mix(c011, c111, f.x);
  let y0 = mix(x00, x10, f.y);
  let y1 = mix(x01, x11, f.y);
  let corr = mix(y0, y1, f.z);

  let pOrig = textureLoad(finePIn, vec3<i32>(gid), 0).r;
  textureStore(finePOut, vec3<i32>(gid), vec4<f32>(pOrig + corr + u.domainBounds.x * 0.0, 0.0, 0.0, 0.0));
}
`;

/**
 * Exact staggered MAC pressure gradient projection:
 *   u_{i+1/2} <- u_{i+1/2} - (p_{i+1} - p_i)
 *   v_{j+1/2} <- v_{j+1/2} - (p_{j+1} - p_j)   (with p_{Ny} = p_top(eta) at j = Ny - 1!)
 *   w_{k+1/2} <- w_{k+1/2} - (p_{k+1} - p_k)
 */
export const FLUID_MAC_PROJECT_WGSL = /* wgsl */ `
${COMMON_WGSL}
${TOP_DIRICHLET_PRESSURE_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var velMacIn: texture_3d<f32>;
@group(0) @binding(2) var pressIn: texture_3d<f32>;
@group(0) @binding(3) var waveIn: texture_2d<f32>;
@group(0) @binding(4) var linSampler: sampler;
@group(0) @binding(5) var velMacOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${VEL_GRID_X}u || gid.y >= ${VEL_GRID_Y}u || gid.z >= ${VEL_GRID_Z}u) {
    return;
  }
  let ix = i32(gid.x);
  let iy = i32(gid.y);
  let iz = i32(gid.z);

  var mac = textureLoad(velMacIn, vec3<i32>(ix, iy, iz), 0).xyz;
  let pC = textureLoad(pressIn, vec3<i32>(ix, iy, iz), 0).r;

  if (ix < ${VEL_GRID_X - 1}) {
    let pXp = textureLoad(pressIn, vec3<i32>(ix + 1, iy, iz), 0).r;
    mac.x = mac.x - (pXp - pC);
  } else {
    mac.x = 0.0;
  }

  if (iy < ${VEL_GRID_Y - 1}) {
    let pYp = textureLoad(pressIn, vec3<i32>(ix, iy + 1, iz), 0).r;
    mac.y = mac.y - (pYp - pC);
  } else {
    let pTop = evalTopDirichletPressure(ix, iz, u, waveIn, linSampler);
    mac.y = mac.y - (pTop - pC);
  }

  if (iz < ${VEL_GRID_Z - 1}) {
    let pZp = textureLoad(pressIn, vec3<i32>(ix, iy, iz + 1), 0).r;
    mac.z = mac.z - (pZp - pC);
  } else {
    mac.z = 0.0;
  }

  mac = clamp(mac, vec3<f32>(-2.0), vec3<f32>(2.0));
  textureStore(velMacOut, vec3<i32>(ix, iy, iz), vec4<f32>(mac, 0.0));
}
`;

/**
 * Reconstructs cell-centered divergence-free velocity u_center(i, j, k) = 0.5 * (u_{i-1/2} + u_{i+1/2})
 * from face-staggered MAC velocities for hardware-trilinear sampling in Lagrangian RBC & Eulerian Hb advection.
 */
export const FLUID_MAC_TO_CENTER_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var velMacIn: texture_3d<f32>;
@group(0) @binding(2) var velCenterOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${VEL_GRID_X}u || gid.y >= ${VEL_GRID_Y}u || gid.z >= ${VEL_GRID_Z}u) {
    return;
  }
  let ix = i32(gid.x);
  let iy = i32(gid.y);
  let iz = i32(gid.z);

  let mC = textureLoad(velMacIn, vec3<i32>(ix, iy, iz), 0).xyz;
  let uR = select(mC.x, 0.0, ix >= ${VEL_GRID_X - 1});
  let uL = select(textureLoad(velMacIn, vec3<i32>(max(ix - 1, 0), iy, iz), 0).x, 0.0, ix == 0);

  let vU = mC.y;
  let vD = select(textureLoad(velMacIn, vec3<i32>(ix, max(iy - 1, 0), iz), 0).y, 0.0, iy == 0);

  let wF = select(mC.z, 0.0, iz >= ${VEL_GRID_Z - 1});
  let wB = select(textureLoad(velMacIn, vec3<i32>(ix, iy, max(iz - 1, 0)), 0).z, 0.0, iz == 0);

  var vCenter = 0.5 * vec3<f32>(uL + uR, vD + vU, wB + wF) + vec3<f32>(u.domainBounds.x * 0.0);
  if (iy >= ${VEL_GRID_Y - 9} && vCenter.y < 0.0) {
    let dims = vec3<f32>(${VEL_GRID_X}.0, ${VEL_GRID_Y}.0, ${VEL_GRID_Z}.0);
    let pCell = uvwToWorld((vec3<f32>(gid) + 0.5) / dims, u);
    var activeNeck = 0.0;
    for (var d = 0u; d < ${MAX_ACTIVE_DROPS}u; d = d + 1u) {
      let slot = u.drops[d];
      if (slot.velOscAndDrain.w > 0.0 && slot.velOscAndDrain.z > 0.015) {
        let rActive = clamp(slot.posAndRadius.w * 0.85, 0.012, 0.036);
        let rDist = length(pCell.xz - slot.posAndRadius.xz);
        let drainGate = smoothstep(0.015, 0.095, slot.velOscAndDrain.z);
        activeNeck = max(activeNeck, (1.0 - smoothstep(rActive * 0.72, rActive * 1.15, rDist)) * drainGate);
      }
    }
    let stagGate = smoothstep(0.006, 0.072, max(0.0, u.surfaceParams.x - pCell.y));
    vCenter.y = vCenter.y * mix(stagGate, 1.0, activeNeck);
  }
  textureStore(velCenterOut, vec3<i32>(ix, iy, iz), vec4<f32>(vCenter, 0.0));
}
`;
