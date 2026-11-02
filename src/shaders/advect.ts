import {
  COMMON_WGSL,
  CONC_GRID_X,
  CONC_GRID_Y,
  CONC_GRID_Z,
  SPLAT_FP_SCALE,
} from './common';

export const CONC_CLEAR_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var concOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${CONC_GRID_X}u || gid.y >= ${CONC_GRID_Y}u || gid.z >= ${CONC_GRID_Z}u) {
    return;
  }
  let zeroVal = u.surfaceParams.x * 0.0;
  textureStore(concOut, vec3<i32>(gid), vec4<f32>(zeroVal, 0.0, 0.0, 0.0));
}
`;

/**
 * RK2 semi-Lagrangian predictor for the Eulerian dissolved (lysed) free-hemoglobin channel (.g).
 */
export const CONC_PREDICT_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var concIn: texture_3d<f32>;
@group(0) @binding(2) var velCenterIn: texture_3d<f32>;
@group(0) @binding(3) var linSampler: sampler;
@group(0) @binding(4) var concPredOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${CONC_GRID_X}u || gid.y >= ${CONC_GRID_Y}u || gid.z >= ${CONC_GRID_Z}u) {
    return;
  }
  // Keep outer lateral walls and top air boundary zeroed; allow gid.y == 0u floor slice to hold settled coagulum
  if (gid.x == 0u || gid.x >= ${CONC_GRID_X - 1}u || gid.z == 0u || gid.z >= ${CONC_GRID_Z - 1}u || gid.y >= ${CONC_GRID_Y - 1}u) {
    textureStore(concPredOut, vec3<i32>(gid), vec4<f32>(0.0));
    return;
  }
  let dims = vec3<f32>(${CONC_GRID_X}.0, ${CONC_GRID_Y}.0, ${CONC_GRID_Z}.0);
  let uvw = (vec3<f32>(gid) + 0.5) / dims;
  let dt = clamp(u.viewportAndDt.z, 0.0, 0.025);
  let domSize = domainSize(u);

  let v0 = textureSampleLevel(velCenterIn, linSampler, uvw, 0.0).xyz;
  let speed2 = dot(v0, v0);
  if (speed2 < 2.5e-5) {
    let gStatic = textureLoad(concIn, vec3<i32>(gid), 0).g;
    textureStore(concPredOut, vec3<i32>(gid), vec4<f32>(0.0, gStatic, 0.0, gStatic));
    return;
  }
  var uvwBack: vec3<f32>;
  if (speed2 < 0.012) {
    // Below 0.11 m/s (displacement < 0.18 voxels), midpoint velocity differs by < 0.002 voxels: save 2nd 3D fetch!
    uvwBack = clamp(uvw - (v0 * dt) / domSize, vec3<f32>(0.001), vec3<f32>(0.999));
  } else {
    let uvwMid = clamp(uvw - 0.5 * (v0 * dt) / domSize, vec3<f32>(0.001), vec3<f32>(0.999));
    let vMid = textureSampleLevel(velCenterIn, linSampler, uvwMid, 0.0).xyz;
    uvwBack = clamp(uvw - (vMid * dt) / domSize, vec3<f32>(0.001), vec3<f32>(0.999));
  }

  let cBack = textureSampleLevel(concIn, linSampler, uvwBack, 0.0);
  textureStore(concPredOut, vec3<i32>(gid), vec4<f32>(0.0, cBack.g, 0.0, cBack.g));
}
`;

/**
 * Hybrid Lagrangian-Eulerian Concentration & Rouleaux Sedimentation Resolver (192x256x192 rgba16float):
 *  - .r = Intact RBC + settled floor rouleaux coagulum density c_RBC = c_parcel + c_bed
 *  - .g = Dissolved free hemoglobin c_Hb,lysed (MacCormack advection + molecular diffusion + slow ambient dilution)
 *  - .b = Physical oxygenation & shear dispersion state v_oxShear in [0, 1] (0 = venous rouleaux SO2~0.56, 1 = sheared arterial HbO2 SO2~0.99)
 *  - .a = Persistent Eulerian settled floor rouleaux coagulum bed c_bed (conserving settled floor mass across >8-drop ring wraps)
 */
export const CONC_RESOLVE_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var concOrigIn: texture_3d<f32>;
@group(0) @binding(2) var concPredIn: texture_3d<f32>;
@group(0) @binding(3) var velCenterIn: texture_3d<f32>;
@group(0) @binding(4) var linSampler: sampler;
@group(0) @binding(5) var<storage, read> splatIntact: array<u32>;
@group(0) @binding(6) var<storage, read> splatLysed: array<u32>;
@group(0) @binding(7) var concOut: texture_storage_3d<rgba16float, write>;

fn loadOrigLysed(c: vec3<i32>) -> f32 {
  let cc = clamp(
    c,
    vec3<i32>(0, 0, 0),
    vec3<i32>(${CONC_GRID_X - 1}, ${CONC_GRID_Y - 1}, ${CONC_GRID_Z - 1})
  );
  return textureLoad(concOrigIn, cc, 0).g;
}

fn loadOrigBed(c: vec3<i32>) -> f32 {
  let cc = clamp(
    c,
    vec3<i32>(0, 0, 0),
    vec3<i32>(${CONC_GRID_X - 1}, ${CONC_GRID_Y - 1}, ${CONC_GRID_Z - 1})
  );
  return textureLoad(concOrigIn, cc, 0).a;
}

fn loadPredLysed(c: vec3<i32>) -> f32 {
  let cc = clamp(
    c,
    vec3<i32>(0, 0, 0),
    vec3<i32>(${CONC_GRID_X - 1}, ${CONC_GRID_Y - 1}, ${CONC_GRID_Z - 1})
  );
  return textureLoad(concPredIn, cc, 0).g;
}

fn loadIntactRaw(ix: i32, iy: i32, iz: i32) -> f32 {
  let cx = clamp(ix, 0, ${CONC_GRID_X - 1});
  let cy = clamp(iy, 0, ${CONC_GRID_Y - 1});
  let cz = clamp(iz, 0, ${CONC_GRID_Z - 1});
  let flatIdx = u32((cz * ${CONC_GRID_Y} + cy) * ${CONC_GRID_X} + cx);
  return f32(splatIntact[flatIdx]) * ${(1.0 / SPLAT_FP_SCALE).toExponential(8)};
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${CONC_GRID_X}u || gid.y >= ${CONC_GRID_Y}u || gid.z >= ${CONC_GRID_Z}u) {
    return;
  }
  let ix = i32(gid.x);
  let iy = i32(gid.y);
  let iz = i32(gid.z);
  if (ix == 0 || ix >= ${CONC_GRID_X - 1} || iz == 0 || iz >= ${CONC_GRID_Z - 1} || iy >= ${CONC_GRID_Y - 1}) {
    textureStore(concOut, vec3<i32>(ix, iy, iz), vec4<f32>(0.0));
    return;
  }

  let dims = vec3<f32>(${CONC_GRID_X}.0, ${CONC_GRID_Y}.0, ${CONC_GRID_Z}.0);
  let flatIdx = u32((iz * ${CONC_GRID_Y} + iy) * ${CONC_GRID_X} + ix);
  let rawIntact = splatIntact[flatIdx];
  let rawLysed = splatLysed[flatIdx];
  let inFloorZone = iy <= 7;

  // Exact 2-tap hardware-trilinear octant positivity filter (extended to preserve thin floor rouleaux bed in iy <= 7)
  if (rawIntact == 0u && rawLysed == 0u) {
    let oct0 = textureSampleLevel(concPredIn, linSampler, vec3<f32>(gid) / dims, 0.0).g;
    let oct1 = textureSampleLevel(concPredIn, linSampler, (vec3<f32>(gid) + 1.0) / dims, 0.0).g;
    var bedActive = 0.0;
    if (inFloorZone) {
      bedActive = textureSampleLevel(concOrigIn, linSampler, (vec3<f32>(gid) + 0.5) / dims, 0.0).a;
    }
    if (oct0 + oct1 <= 1.0e-7 && bedActive <= 1.0e-6) {
      textureStore(concOut, vec3<i32>(ix, iy, iz), vec4<f32>(0.0));
      return;
    }
  }

  let gPred = textureLoad(concPredIn, vec3<i32>(ix, iy, iz), 0).g;
  let lapNeighborsG = (
    loadPredLysed(vec3<i32>(ix + 1, iy, iz)) +
    loadPredLysed(vec3<i32>(ix - 1, iy, iz)) +
    loadPredLysed(vec3<i32>(ix, iy + 1, iz)) +
    loadPredLysed(vec3<i32>(ix, iy - 1, iz)) +
    loadPredLysed(vec3<i32>(ix, iy, iz + 1)) +
    loadPredLysed(vec3<i32>(ix, iy, iz - 1))
  ) * (1.0 / 6.0);

  if (!inFloorZone && rawIntact == 0u && rawLysed == 0u && gPred <= 1.2e-6 && lapNeighborsG <= 1.2e-6) {
    textureStore(concOut, vec3<i32>(ix, iy, iz), vec4<f32>(0.0));
    return;
  }

  var lapIntact = 0.0;
  let loadNeighborhood = rawIntact > 0u || gPred > 8.0e-5 || lapNeighborsG > 8.0e-5;
  if (loadNeighborhood) {
    lapIntact = (
      loadIntactRaw(ix + 1, iy, iz) +
      loadIntactRaw(ix - 1, iy, iz) +
      loadIntactRaw(ix, iy + 1, iz) +
      loadIntactRaw(ix, iy - 1, iz) +
      loadIntactRaw(ix, iy, iz + 1) +
      loadIntactRaw(ix, iy, iz - 1)
    ) * (1.0 / 6.0);
  }

  let uvw = (vec3<f32>(gid) + 0.5) / dims;
  let dt = clamp(u.viewportAndDt.z, 0.0, 0.025);
  let domSize = domainSize(u);
  let p = uvwToWorld(uvw, u);
  let halfXZ = u.domainBounds.x;
  let yMin = u.domainBounds.y;

  // 1. Advect & molecularly diffuse Eulerian lysed free hemoglobin (.g)
  let v0 = textureSampleLevel(velCenterIn, linSampler, uvw, 0.0).xyz;
  let uvwMid = clamp(uvw - 0.5 * (v0 * dt) / domSize, vec3<f32>(0.001), vec3<f32>(0.999));
  let vMid = textureSampleLevel(velCenterIn, linSampler, uvwMid, 0.0).xyz;
  let uvwBack = clamp(uvw - (vMid * dt) / domSize, vec3<f32>(0.001), vec3<f32>(0.999));

  let uvwFwd = clamp(uvw + (v0 * dt) / domSize, vec3<f32>(0.001), vec3<f32>(0.999));
  let gBackEst = textureSampleLevel(concPredIn, linSampler, uvwFwd, 0.0).g;
  let gOrig = textureLoad(concOrigIn, vec3<i32>(ix, iy, iz), 0).g;

  let sharpness = u.fluidParams.w;
  let gCorr = gPred + (0.45 * sharpness) * (gOrig - gBackEst);

  let texCoord = uvwBack * dims - 0.5;
  let baseIdx = vec3<i32>(floor(texCoord));
  let s000 = loadOrigLysed(baseIdx + vec3<i32>(0, 0, 0));
  let s100 = loadOrigLysed(baseIdx + vec3<i32>(1, 0, 0));
  let s010 = loadOrigLysed(baseIdx + vec3<i32>(0, 1, 0));
  let s110 = loadOrigLysed(baseIdx + vec3<i32>(1, 1, 0));
  let s001 = loadOrigLysed(baseIdx + vec3<i32>(0, 0, 1));
  let s101 = loadOrigLysed(baseIdx + vec3<i32>(1, 0, 1));
  let s011 = loadOrigLysed(baseIdx + vec3<i32>(0, 1, 1));
  let s111 = loadOrigLysed(baseIdx + vec3<i32>(1, 1, 1));

  let minG = min(min(min(s000, s100), min(s010, s110)), min(min(s001, s101), min(s011, s111)));
  let maxG = max(max(max(s000, s100), max(s010, s110)), max(max(s001, s101), max(s011, s111)));
  var cLysed = clamp(gCorr, minG, maxG);

  // Co-moving molecular diffusion + ambient dilution so suspended free hemoglobin forms a soft dispersing cloud around tendrils
  let diffBlend = clamp(dt * 4.80, 0.0, 0.088);
  cLysed = mix(cLysed, lapNeighborsG, diffBlend);
  let upperDilute = exp(-dt * mix(0.11, 0.22, smoothstep(yMin + 0.045, yMin + 0.26, p.y)));
  cLysed = cLysed * upperDilute;

  let dmTransfer = f32(rawLysed) * ${(1.0 / SPLAT_FP_SCALE).toExponential(8)};
  let floorBedCapture = 1.0 - smoothstep(yMin + 0.018, yMin + 0.058, p.y);
  cLysed = cLysed + dmTransfer * (1.0 - floorBedCapture * 0.72);
  if (cLysed < 2.5e-6) {
    cLysed = 0.0;
  }

  // 2. Softly spreading & gradually dissipating circular bottom pool (.a):
  //    Spreads radially across the bottom boundary layer and slowly dissolves into the surrounding water
  //    so settled blood never accumulates into a hard-edged square/diamond floor tile!
  let rHoriz = length(p.xz);
  var cRouleauxBed = 0.0;
  if (inFloorZone) {
    let bCenter = loadOrigBed(vec3<i32>(ix, iy, iz));
    let bHoriz = 0.25 * (
      loadOrigBed(vec3<i32>(ix + 1, iy, iz)) +
      loadOrigBed(vec3<i32>(ix - 1, iy, iz)) +
      loadOrigBed(vec3<i32>(ix, iy, iz + 1)) +
      loadOrigBed(vec3<i32>(ix, iy, iz - 1))
    );
    let bAbove = loadOrigBed(vec3<i32>(ix, iy + 1, iz));
    let bBelow = loadOrigBed(vec3<i32>(ix, max(0, iy - 1), iz));
    let spreadAlpha = clamp(dt * (3.20 + 6.50 * min(bCenter, 0.65)), 0.0, 0.165);
    let vertDiffuse = clamp(dt * 0.75, 0.0, 0.026);
    let settleAlpha = clamp(dt * 2.80, 0.0, 0.090);
    let bedDecay = exp(-dt * mix(0.095, 0.28, smoothstep(halfXZ * 0.35, halfXZ * 0.78, rHoriz)));
    let drainedSelf = select(bCenter * (1.0 - settleAlpha), bCenter, iy == 0) * bedDecay;
    let settledFromAbove = select(bAbove * settleAlpha, 0.0, iy >= 7);
    let advectedBed = drainedSelf + settledFromAbove;
    let smoothedBed = mix(mix(advectedBed, bHoriz, spreadAlpha), 0.5 * (bAbove + bBelow), vertDiffuse);
    let topBedFade = 1.0 - 0.38 * smoothstep(3.5, 7.0, f32(iy));
    cRouleauxBed = (smoothedBed + dmTransfer * floorBedCapture * 0.68) * topBedFade;
    if (cRouleauxBed < 1.0e-6) {
      cRouleauxBed = 0.0;
    }
  }

  // 3. Reconstruct total intact erythrocyte + bottom pool density (.r) with circular radial domain feathering (zero square corners!)
  let wallSponge = 1.0 - smoothstep(halfXZ * 0.68, halfXZ * 0.94, rHoriz);
  let floorRadialTaper = 1.0 - smoothstep(halfXZ * 0.52, halfXZ * 0.88, rHoriz);
  let maxBedProfile = 0.54 * exp(-0.56 * f32(iy)) * floorRadialTaper;
  cRouleauxBed = min(cRouleauxBed * wallSponge, maxBedProfile);
  let cParcelRaw = f32(rawIntact) * ${(1.0 / SPLAT_FP_SCALE).toExponential(8)};
  let shoulderFeather = select(0.18, 0.36, rawIntact > 0u);
  let cParcelIntact = mix(cParcelRaw, lapIntact, shoulderFeather) * wallSponge;
  let cIntact = cParcelIntact + cRouleauxBed;
  cLysed = cLysed * wallSponge;

  // 4. Physical oxygenation + shear/stretch dispersion state (.b in [0, 1]):
  //    v -> 0: Low-shear dense cores, diffused lysed halos & settled floor rouleaux (SO2 ~ 0.56 + Duysens sieve -> venous burgundy)
  //    v -> 1: High-shear stretched intact-RBC vortex sheets & thin margins (SO2 ~ 0.99 -> amber/vermilion thin sheets -> ruby)
  let cTotal = cIntact + cLysed;
  var vOxShear = 0.0;
  if (cTotal > 1.0e-5) {
    let fLysis = clamp(cLysed / max(cTotal, 1.0e-5), 0.0, 1.0);
    let intactFrac = 1.0 - fLysis;
    let bedFrac = smoothstep(0.0, 0.85, cRouleauxBed / max(cTotal, 1.0e-5));
    let flowSpeed = length(v0);
    let convShear = length(v0 - vMid) / max(0.5 * dt, 1.0e-3);
    let shearDispersion = smoothstep(0.014, 0.135, flowSpeed + 0.38 * convShear);
    let thinIntactSheet = (1.0 - smoothstep(0.024, 0.24, cIntact)) * smoothstep(0.0006, 0.0065, cIntact);
    let activeSheet = thinIntactSheet * (0.38 + 0.62 * shearDispersion) * intactFrac;
    let rawOx = clamp(0.08 + 0.62 * activeSheet + 0.36 * shearDispersion * intactFrac, 0.0, 1.0);
    vOxShear = rawOx * (1.0 - 0.82 * bedFrac);
  }

  textureStore(concOut, vec3<i32>(ix, iy, iz), vec4<f32>(cIntact, cLysed, vOxShear, cRouleauxBed));
}
`;
