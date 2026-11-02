import {
  COMMON_WGSL,
  CONC_GRID_X,
  CONC_GRID_Y,
  CONC_GRID_Z,
  MAX_ACTIVE_DROPS,
  PARTICLE_COUNT,
  PARTICLE_WORKGROUP_SIZE,
  SPLAT_FP_SCALE,
} from './common';

export const PARCELS_PER_REF_DROP = 49152; // Exact 393,216 / 8 capacity -> all 8 concurrent drops fit with zero ring-buffer overwrite
// Physical voxel-normalized volume per parcel: V_ref / (V_voxel * PARCELS_PER_REF_DROP) = 6538.24 / 49152 ~ 0.1330
export const INITIAL_PARCEL_VOXEL_MASS = 0.134;

const PARCEL_STRUCT_WGSL = /* wgsl */ `
struct RbcParcel {
  posAndMass: vec4<f32>,      // .xyz = worldPos, .w = intactRbcMass (0 if inactive)
  stretchAndLysis: vec4<f32>, // .xyz = flowStretchVec, .w = osmoticLysisState [0, 1]
};

fn flatConcIndex(ix: i32, iy: i32, iz: i32) -> u32 {
  return u32((iz * ${CONC_GRID_Y} + iy) * ${CONC_GRID_X} + ix);
}
`;

export const PARTICLE_RESET_WGSL = /* wgsl */ `
${COMMON_WGSL}
${PARCEL_STRUCT_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<RbcParcel>;

@compute @workgroup_size(${PARTICLE_WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${PARTICLE_COUNT}u) {
    return;
  }
  let zeroTouch = u.surfaceParams.x * 0.0;
  var empty: RbcParcel;
  empty.posAndMass = vec4<f32>(0.0, zeroTouch, 0.0, 0.0);
  empty.stretchAndLysis = vec4<f32>(0.0, -0.003, 0.0, 0.0);
  particles[gid.x] = empty;
}
`;

/**
 * Emits `u.particleParams.z` new high-Schmidt intact RBC parcels into the persistent ring buffer
 * starting at `u.particleParams.y`, partitioned across all 8 concurrent drops by exact wetting-neck
 * volume drainage rate (`velOscAndDrain.z`). If >8 drops in a single cycle wrap onto a live parcel,
 * its surviving hemoglobin mass is atomically transferred into `splatLysed` so zero mass is lost.
 */
export const PARTICLE_EMIT_WGSL = /* wgsl */ `
${COMMON_WGSL}
${PARCEL_STRUCT_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var waveIn: texture_2d<f32>;
@group(0) @binding(2) var linSampler: sampler;
@group(0) @binding(3) var<storage, read_write> particles: array<RbcParcel>;
@group(0) @binding(4) var<storage, read_write> splatLysed: array<atomic<u32>>;

@compute @workgroup_size(${PARTICLE_WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let spawnCount = u32(max(0.0, u.particleParams.z));
  if (gid.x >= spawnCount) {
    return;
  }

  // Sum active volume drainage rates across all 8 drop slots
  var totalDrain = 0.0;
  for (var d = 0u; d < ${MAX_ACTIVE_DROPS}u; d = d + 1u) {
    let slot = u.drops[d];
    if (slot.velOscAndDrain.w > 0.0 && slot.velOscAndDrain.z > 0.0001) {
      totalDrain = totalDrain + slot.velOscAndDrain.z;
    }
  }
  if (totalDrain <= 1e-6) {
    return;
  }

  // Select target drop slot proportionally to instantaneous volume drainage flux
  let targetFrac = (f32(gid.x) + 0.5) / f32(max(spawnCount, 1u)) * totalDrain;
  var accum = 0.0;
  var chosenIdx = 0u;
  for (var d = 0u; d < ${MAX_ACTIVE_DROPS}u; d = d + 1u) {
    let slot = u.drops[d];
    if (slot.velOscAndDrain.w > 0.0 && slot.velOscAndDrain.z > 0.0001) {
      accum = accum + slot.velOscAndDrain.z;
      chosenIdx = d;
      if (accum >= targetFrac) {
        break;
      }
    }
  }

  let drop = u.drops[chosenIdx];
  let seed = max(drop.velOscAndDrain.w, 0.5);
  let cursor = u32(max(0.0, u.particleParams.y));
  let ringIdx = (cursor + gid.x) % ${PARTICLE_COUNT}u;

  // Conservative wraparound transfer: if >8 drops wrap onto an active parcel, consolidate deep settling
  // parcels directly into the floor rouleaux capture zone rather than mid-water lysed smoke
  let oldP = particles[ringIdx];
  if (oldP.posAndMass.w > 1e-6) {
    let settleDepositPos = select(
      oldP.posAndMass.xyz,
      vec3<f32>(oldP.posAndMass.x, u.domainBounds.y + 0.012, oldP.posAndMass.z),
      oldP.posAndMass.y < u.surfaceParams.x - 0.32
    );
    let oldUvw = worldToUVW(settleDepositPos, u);
    if (all(oldUvw > vec3<f32>(0.002)) && all(oldUvw < vec3<f32>(0.998))) {
      let dims = vec3<f32>(${CONC_GRID_X}.0, ${CONC_GRID_Y}.0, ${CONC_GRID_Z}.0);
      let maxIdx = vec3<i32>(${CONC_GRID_X - 1}, ${CONC_GRID_Y - 1}, ${CONC_GRID_Z - 1});
      let c = clamp(vec3<i32>(floor(oldUvw * dims)), vec3<i32>(0), maxIdx);
      let fpOld = u32(round(oldP.posAndMass.w * ${SPLAT_FP_SCALE.toFixed(1)}));
      if (fpOld > 0u) {
        atomicAdd(&splatLysed[flatConcIndex(c.x, c.y, c.z)], fpOld);
      }
    }
  }

  // Low-discrepancy Martin Roberts R3 plastic-constant sequence + gentle sub-voxel jitter for zero-banding emission
  let seqIdx = f32(ringIdx) + seed * 137.0;
  let r3 = fract(vec3<f32>(
    seqIdx * 0.8191725134,
    seqIdx * 0.6710436067,
    seqIdx * 0.5497004779
  ) + hash33(vec3<f32>(f32(gid.x), u.cameraPosAndTime.w * 60.0, seed)) * 0.032);

  let dropXZ = drop.posAndRadius.xz;
  let waveUV = clamp(worldXZToWaveUV(dropXZ), vec2<f32>(0.002), vec2<f32>(0.998));
  let craterEta = clamp(textureSampleLevel(waveIn, linSampler, waveUV, 0.0).r, -0.065, 0.025);
  let neckTopY = u.surfaceParams.x + min(0.0, craterEta * 0.32);

  let theta = 6.28318530718 * r3.x;
  // Even Widnall m=4,6,8 azimuthal modes eliminate m=1 beat dipole and early 5-clump blotching
  let azim = 1.0
    + 0.036 * cos(4.0 * theta + seed * 1.73)
    + 0.026 * cos(6.0 * theta - seed * 2.31)
    + 0.018 * sin(8.0 * theta + seed * 0.89);

  let rDrop = max(drop.posAndRadius.w, 0.010);
  let slugWeight = smoothstep(0.26, 0.85, drop.velOscAndDrain.z);
  let wakeThin = mix(0.46, 1.0, smoothstep(0.018, 0.26, drop.velOscAndDrain.z));
  let rNeck = mix(0.013 * wakeThin, clamp(rDrop * 0.84 + 0.014, 0.020, 0.056), slugWeight) * azim;
  // Hollow-centered toroidal vortex sheet (84% annular ring + 16% continuous dome membrane overlapping innerRing)
  // keeps heavy RBC buoyancy in the outer rolled torus while preventing pale holes across the upper umbrella cap!
  let innerRing = 0.37 * slugWeight;
  let isCapVeil = slugWeight > 0.08 && r3.z < 0.16;
  let radFracAnnular = mix(innerRing, 1.0, pow(r3.y, mix(0.50, 0.44, slugWeight)));
  let radFracCap = mix(0.14 * slugWeight, max(innerRing * 1.32, 0.08), sqrt(r3.y));
  let radFrac = select(radFracAnnular, radFracCap, isCapVeil);
  let r = rNeck * radFrac;
  let wakeDriftAngle = u.cameraPosAndTime.w * 1.85 + seed * 3.17;
  let wakeDriftXZ = vec2<f32>(cos(wakeDriftAngle), sin(wakeDriftAngle * 0.83 + seed))
                  * (0.0038 * (1.0 - slugWeight) * (1.0 - wakeThin * 0.45));
  let posXZ = dropXZ + wakeDriftXZ + vec2<f32>(cos(theta), sin(theta)) * r;
  let zStrat = select((r3.z - 0.16 * slugWeight) / max(1.0 - 0.16 * slugWeight, 0.5), r3.z * 3.6, isCapVeil);
  let posY = neckTopY - mix(0.0012, 0.013, clamp(zStrat, 0.0, 1.0));

  // Outer water-sheared contact perimeter AND smaller satellite droplets start with higher osmotic pre-swelling
  // so small drops and outer veils disperse rapidly into soft diffusing clouds while large drops penetrate deeper
  let smallDropFragility = 1.0 - smoothstep(0.022, 0.056, rDrop);
  let rimExposure = smoothstep(0.58, 1.0, radFrac);
  let initLysis = clamp(0.14 * rimExposure + 0.28 * smallDropFragility * (0.45 + 0.55 * radFrac), 0.0, 0.44);
  let tangentDir = vec3<f32>(-sin(theta) * max(radFrac, 0.25), -0.80, cos(theta) * max(radFrac, 0.25));
  let initStretch = normalize(tangentDir) * 0.0036;

  var p: RbcParcel;
  p.posAndMass = vec4<f32>(posXZ.x, posY, posXZ.y, ${INITIAL_PARCEL_VOXEL_MASS.toFixed(5)});
  p.stretchAndLysis = vec4<f32>(initStretch, initLysis);
  particles[ringIdx] = p;
}
`;

/**
 * Advances all active Lagrangian RBC parcels (Sc ~ 10^7) via 3rd-order Ralston RK3 on the
 * divergence-free MAC velocity field, evolves velocity-gradient stretch anisotropy, models
 * hypotonic osmotic hemolysis transfer into `splatLysed`, and splats continuous 4-point
 * RBC filament ribbons onto `splatIntact`.
 */
export const PARTICLE_ADVECT_SPLAT_WGSL = /* wgsl */ `
${COMMON_WGSL}
${PARCEL_STRUCT_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var velCenterIn: texture_3d<f32>;
@group(0) @binding(2) var concIn: texture_3d<f32>;
@group(0) @binding(3) var linSampler: sampler;
@group(0) @binding(4) var<storage, read_write> particles: array<RbcParcel>;
@group(0) @binding(5) var<storage, read_write> splatIntact: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> splatLysed: array<atomic<u32>>;

fn sampleVelWorld(p: vec3<f32>, rouleauxMaturity: f32) -> vec3<f32> {
  let depthBelowSurf = u.surfaceParams.x - p.y;
  let uvw = clamp(worldToUVW(p, u), vec3<f32>(0.002), vec3<f32>(0.998));
  // Smooth continuous Stokes sedimentation: eliminates both the frozen 0.2mm/s top bead and the 3x 0.165m/s
  // age cliff that ripped a detached rubbery O-ring out the bottom of the vortex veil!
  let settleGate = mix(0.38, 1.0, smoothstep(0.002, 0.055, depthBelowSurf));
  let rouleauxFall = mix(0.020, 0.046, smoothstep(0.04, 0.42, depthBelowSurf));
  let monoSettle = rouleauxFall * settleGate;
  let stackSettle = mix(0.032, 0.062, smoothstep(0.04, 0.48, depthBelowSurf));
  let effSettle = mix(monoSettle, stackSettle, rouleauxMaturity);

  // Sub-grid divergence-free curl unfurling active below the initial vortex roll-up (depth > 3.5 cm),
  // opening descending rings into delicate, coalescing 3D tendrils and dispersing curtains as they settle!
  let deepUnfurl = smoothstep(0.035, 0.22, depthBelowSurf);
  var subGridCurl = vec3<f32>(0.0);
  if (deepUnfurl > 0.001) {
    let tPhase = u.cameraPosAndTime.w * 0.46;
    let a1 = p.x * 18.5 + p.z * 14.2 + p.y * 11.3 + tPhase;
    let a2 = p.z * 21.3 - p.x * 16.7 + p.y * 13.9 - tPhase * 0.82;
    let a3 = p.x * 31.0 + p.z * 27.0 - p.y * 23.0 + tPhase * 1.15;
    let vxCurl = (14.2 * cos(a1) + 21.3 * cos(a2) + 27.0 * sin(a3) * 0.42) * 0.00068 * deepUnfurl;
    let vzCurl = (-18.5 * cos(a1) + 16.7 * cos(a2) - 31.0 * sin(a3) * 0.42) * 0.00068 * deepUnfurl;
    let vyCurl = sin(a1 * 0.75 - a2 * 0.65) * 0.0036 * deepUnfurl * smoothstep(0.08, 0.30, depthBelowSurf);
    subGridCurl = vec3<f32>(vxCurl, vyCurl, vzCurl);
  }

  return textureSampleLevel(velCenterIn, linSampler, uvw, 0.0).xyz + vec3<f32>(0.0, -effSettle, 0.0) + subGridCurl;
}

fn splatTrilinearIntact(uvw: vec3<f32>, mass: f32) {
  if (mass <= 1e-6) {
    return;
  }
  let dims = vec3<f32>(${CONC_GRID_X}.0, ${CONC_GRID_Y}.0, ${CONC_GRID_Z}.0);
  let coord = uvw * dims - 0.5;
  let base = vec3<i32>(floor(coord));
  let f = clamp(coord - vec3<f32>(base), vec3<f32>(0.0), vec3<f32>(1.0));
  let omf = vec3<f32>(1.0) - f;
  let maxIdx = vec3<i32>(${CONC_GRID_X - 1}, ${CONC_GRID_Y - 1}, ${CONC_GRID_Z - 1});

  let b0 = clamp(base, vec3<i32>(0), maxIdx);
  let b1 = clamp(base + vec3<i32>(1), vec3<i32>(0), maxIdx);

  let scale = mass * ${SPLAT_FP_SCALE.toFixed(1)};

  let w000 = u32(round(scale * omf.x * omf.y * omf.z));
  let w100 = u32(round(scale * f.x   * omf.y * omf.z));
  let w010 = u32(round(scale * omf.x * f.y   * omf.z));
  let w110 = u32(round(scale * f.x   * f.y   * omf.z));
  let w001 = u32(round(scale * omf.x * omf.y * f.z));
  let w101 = u32(round(scale * f.x   * omf.y * f.z));
  let w011 = u32(round(scale * omf.x * f.y   * f.z));
  let w111 = u32(round(scale * f.x   * f.y   * f.z));

  if (w000 > 0u) { atomicAdd(&splatIntact[flatConcIndex(b0.x, b0.y, b0.z)], w000); }
  if (w100 > 0u) { atomicAdd(&splatIntact[flatConcIndex(b1.x, b0.y, b0.z)], w100); }
  if (w010 > 0u) { atomicAdd(&splatIntact[flatConcIndex(b0.x, b1.y, b0.z)], w010); }
  if (w110 > 0u) { atomicAdd(&splatIntact[flatConcIndex(b1.x, b1.y, b0.z)], w110); }
  if (w001 > 0u) { atomicAdd(&splatIntact[flatConcIndex(b0.x, b0.y, b1.z)], w001); }
  if (w101 > 0u) { atomicAdd(&splatIntact[flatConcIndex(b1.x, b0.y, b1.z)], w101); }
  if (w011 > 0u) { atomicAdd(&splatIntact[flatConcIndex(b0.x, b1.y, b1.z)], w011); }
  if (w111 > 0u) { atomicAdd(&splatIntact[flatConcIndex(b1.x, b1.y, b1.z)], w111); }
}

fn splatTrilinearLysed(uvw: vec3<f32>, lysedMass: f32) {
  if (lysedMass <= 1e-7) {
    return;
  }
  let dims = vec3<f32>(${CONC_GRID_X}.0, ${CONC_GRID_Y}.0, ${CONC_GRID_Z}.0);
  let coord = uvw * dims - 0.5;
  let base = vec3<i32>(floor(coord));
  let f = clamp(coord - vec3<f32>(base), vec3<f32>(0.0), vec3<f32>(1.0));
  let omf = vec3<f32>(1.0) - f;
  let maxIdx = vec3<i32>(${CONC_GRID_X - 1}, ${CONC_GRID_Y - 1}, ${CONC_GRID_Z - 1});

  let b0 = clamp(base, vec3<i32>(0), maxIdx);
  let b1 = clamp(base + vec3<i32>(1), vec3<i32>(0), maxIdx);
  let scale = lysedMass * ${SPLAT_FP_SCALE.toFixed(1)};

  let w000 = u32(round(scale * omf.x * omf.y * omf.z));
  let w100 = u32(round(scale * f.x   * omf.y * omf.z));
  let w010 = u32(round(scale * omf.x * f.y   * omf.z));
  let w110 = u32(round(scale * f.x   * f.y   * omf.z));
  let w001 = u32(round(scale * omf.x * omf.y * f.z));
  let w101 = u32(round(scale * f.x   * omf.y * f.z));
  let w011 = u32(round(scale * omf.x * f.y   * f.z));
  let w111 = u32(round(scale * f.x   * f.y   * f.z));

  if (w000 > 0u) { atomicAdd(&splatLysed[flatConcIndex(b0.x, b0.y, b0.z)], w000); }
  if (w100 > 0u) { atomicAdd(&splatLysed[flatConcIndex(b1.x, b0.y, b0.z)], w100); }
  if (w010 > 0u) { atomicAdd(&splatLysed[flatConcIndex(b0.x, b1.y, b0.z)], w010); }
  if (w110 > 0u) { atomicAdd(&splatLysed[flatConcIndex(b1.x, b1.y, b0.z)], w110); }
  if (w001 > 0u) { atomicAdd(&splatLysed[flatConcIndex(b0.x, b0.y, b1.z)], w001); }
  if (w101 > 0u) { atomicAdd(&splatLysed[flatConcIndex(b1.x, b0.y, b1.z)], w101); }
  if (w011 > 0u) { atomicAdd(&splatLysed[flatConcIndex(b0.x, b1.y, b1.z)], w011); }
  if (w111 > 0u) { atomicAdd(&splatLysed[flatConcIndex(b1.x, b1.y, b1.z)], w111); }
}

@compute @workgroup_size(${PARTICLE_WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${PARTICLE_COUNT}u) {
    return;
  }

  var p = particles[gid.x];
  var mass = p.posAndMass.w;
  if (mass <= 1e-6) {
    return;
  }

  let dt = clamp(u.viewportAndDt.z, 0.0, 0.025);
  let halfXZ = u.domainBounds.x;
  let yMin = u.domainBounds.y;
  var pos = p.posAndMass.xyz;
  var stretch = p.stretchAndLysis.xyz;
  let rawW = max(0.0, p.stretchAndLysis.w);
  var ageTicks = floor(rawW);
  var lysis = clamp(rawW - ageTicks, 0.0, 0.98);
  let parcelAge = ageTicks * (1.0 / 240.0);
  let rouleauxMaturity = smoothstep(8.5, 18.0, parcelAge);

  // 1. 3rd-order Ralston RK3 advection (RK2 midpoint below 5 cm/s where displacement < 0.04 voxels -> < 0.3 um delta)
  let v1 = sampleVelWorld(pos, rouleauxMaturity);
  let speed2 = dot(v1, v1);
  if (dt > 0.0) {
    ageTicks = min(8191.0, ageTicks + max(1.0, round(dt * 240.0)));
    let v2 = sampleVelWorld(pos + 0.5 * v1 * dt, rouleauxMaturity);
    var vAdv = v2;
    if (speed2 > 0.0025) {
      let v3 = sampleVelWorld(pos + 0.75 * v2 * dt, rouleauxMaturity);
      vAdv = (2.0 / 9.0) * v1 + (3.0 / 9.0) * v2 + (4.0 / 9.0) * v3;
    }
    // Viscous floor contact damping + radial outward boundary-layer cushion glide on y = -0.98
    let floorSlip = smoothstep(yMin + 0.005, yMin + 0.042, pos.y);
    let rXZFloor = length(pos.xz);
    let floorOutward = select(vec2<f32>(0.0), (pos.xz / max(rXZFloor, 1e-4)) * (0.016 * (1.0 - floorSlip) * smoothstep(0.01, 0.22, rXZFloor)), rXZFloor > 1e-4);
    vAdv = vec3<f32>(
      vAdv.x * mix(0.45, 1.0, floorSlip) + floorOutward.x,
      vAdv.y * mix(0.15, 1.0, floorSlip),
      vAdv.z * mix(0.45, 1.0, floorSlip) + floorOutward.y
    );
    pos = pos + vAdv * dt;
    pos.y = clamp(pos.y, yMin + 0.0055, u.surfaceParams.x - 0.0018);

    // 2. Symmetric 2nd-order material line-element evolution ds/dt = (s . nabla) u up to 5.4 voxels:
    //    Active down to 1.4 mm/s (speed2 > 2.0e-6) so all upper-stem wake parcels stretch into continuous threads!
    if (speed2 > 2.0e-6 && floorSlip > 0.15) {
      let sLen = max(length(stretch), 1e-5);
      let sDir = stretch / sLen;
      let probeStep = 0.0065;
      let vPlus  = sampleVelWorld(pos + sDir * probeStep, rouleauxMaturity);
      let vMinus = sampleVelWorld(pos - sDir * probeStep, rouleauxMaturity);
      let dVelAlongS = (vPlus - vMinus) * (sLen / (2.0 * probeStep));
      stretch = stretch + dVelAlongS * dt;
      let vMag = sqrt(speed2);
      if (vMag > 0.003 && sLen < 0.006) {
        stretch = stretch + (v1 / vMag) * (0.0022 * dt);
      }
      let nextLen = clamp(length(stretch), 0.0026, 0.0270);
      stretch = (stretch / max(length(stretch), 1e-5)) * nextLen;
    } else if (floorSlip <= 0.15) {
      stretch.y = stretch.y * exp(-dt * 5.0);
    }
  }

  let uvw = worldToUVW(pos, u);
  // Circular radial outer-domain sponge (zero square/diamond floor tile boundary!)
  let rHoriz = length(pos.xz);
  let wallSponge = 1.0 - smoothstep(halfXZ * 0.68, halfXZ * 0.94, rHoriz);
  mass = mass * mix(1.0, wallSponge, clamp(dt * 6.0, 0.0, 1.0));
  if (mass <= 1e-5 || rHoriz >= halfXZ * 0.95 || uvw.x <= 0.002 || uvw.x >= 0.998 || uvw.z <= 0.002 || uvw.z >= 0.998 || uvw.y >= 0.998) {
    p.posAndMass.w = 0.0;
    particles[gid.x] = p;
    return;
  }

  // Gradual mass-conserving consolidation of floor-settled RBC parcels into the Eulerian bottom pool (conc.a)
  // with slow osmotic dissolution so settled pools spread radially and softly clear rather than forming a permanent slab
  if (dt > 0.0 && pos.y < yMin + 0.026) {
    let settleWeight = 1.0 - smoothstep(yMin + 0.005, yMin + 0.026, pos.y);
    let dmBed = mass * clamp(settleWeight * dt * 1.65, 0.0, 0.095);
    if (dmBed > 1e-7) {
      mass = max(0.0, mass - dmBed);
      splatTrilinearLysed(clamp(uvw, vec3<f32>(0.002), vec3<f32>(0.998)), dmBed * 0.68);
    }
    if (mass <= 1e-5) {
      p.posAndMass.w = 0.0;
      particles[gid.x] = p;
      return;
    }
  }

  // 3. Hypotonic Osmotic Hemolysis with Dilution- & Size-Coupled Dispersion:
  //    Small drops, outer skirts, and dilute tendrils disperse rapidly into translucent ruby hemoglobin,
  //    while dense vortex-ring cores retain cohesive structure during initial descent before unfurling!
  let cellRes = fract(f32(gid.x) * 0.6180339887 + 0.17);
  let cSample = textureSampleLevel(concIn, linSampler, clamp(uvw, vec3<f32>(0.002), vec3<f32>(0.998)), 0.0);
  let cNeighborhood = cSample.r + cSample.g;
  let diluteExposure = 1.0 - smoothstep(0.012, 0.22, cNeighborhood);
  let ageDispersion = smoothstep(1.1, 7.8, parcelAge + diluteExposure * 2.2 + lysis * 2.5);
  let minIntactMass = ${INITIAL_PARCEL_VOXEL_MASS.toFixed(5)} * mix(mix(0.46, 0.64, cellRes), mix(0.07, 0.20, cellRes), ageDispersion);
  let lysableHeadroom = mass - minIntactMass;
  let floorShelter = smoothstep(yMin + 0.024, yMin + 0.062, pos.y);
  if (dt > 0.0 && lysableHeadroom > 1e-7 && floorShelter > 0.02) {
    let shearActivity = mix(0.22, 1.0, smoothstep(0.0005, 0.0120, speed2));
    let waterExposure = (1.0 - 0.72 * smoothstep(0.016, 0.30, cNeighborhood)) * floorShelter * shearActivity;
    let kLysis = u.lensAndRheology.w;
    lysis = clamp(lysis + kLysis * (1.85 * waterExposure) * dt, 0.0, 0.98);

    let fragility = smoothstep(0.04, 0.48, lysis) * waterExposure;
    let dmLysed = min(lysableHeadroom, mass * clamp(fragility * kLysis * 1.95 * dt, 0.0, 0.16));
    if (dmLysed > 1e-7) {
      mass = max(0.0, mass - dmLysed);
      splatTrilinearLysed(clamp(uvw, vec3<f32>(0.002), vec3<f32>(0.998)), dmLysed);
    }
  }

  p.posAndMass = vec4<f32>(pos, mass);
  p.stretchAndLysis = vec4<f32>(stretch, ageTicks + clamp(lysis, 0.0, 0.98));
  particles[gid.x] = p;

  // 4. Continuous 4-point tapered filament & 2D sheet-tangent splatting (weights 0.21 + 0.29 + 0.29 + 0.21 == 1.00):
  //    Unstretched slug-core parcels (<0.75 voxels) coalesce into 1 trilinear splat (4x less atomic contention),
  //    while stretched umbrella shoulders & curtains splat origin-independent 2D tangent sheets!
  let sLenNow = length(stretch);
  if (sLenNow < 0.0038) {
    splatTrilinearIntact(clamp(uvw, vec3<f32>(0.002), vec3<f32>(0.998)), mass);
  } else {
    let stretchXZLen = length(stretch.xz);
    let horizTangent = select(
      vec3<f32>(1.0, 0.0, 0.0),
      vec3<f32>(-stretch.z, 0.0, stretch.x) / max(stretchXZLen, 1e-5),
      stretchXZLen > 1e-4
    );
    let sheetCross = cross(stretch, horizTangent);
    let crossChoice = select(horizTangent, sheetCross / max(length(sheetCross), 1e-5), length(sheetCross) > 1e-4);
    let horizMix = smoothstep(0.35, 0.85, abs(stretch.y) / max(sLenNow, 1e-5));
    let sheetNorm = normalize(mix(crossChoice, horizTangent, horizMix));
    let sheetSpan = clamp(sLenNow * 0.62, 0.0032, 0.0085);
    let sheetVec = sheetNorm * sheetSpan;

    let uvw0 = clamp(worldToUVW(pos - 0.42 * stretch, u), vec3<f32>(0.002), vec3<f32>(0.998));
    let uvw1 = clamp(worldToUVW(pos - 0.14 * stretch + sheetVec, u), vec3<f32>(0.002), vec3<f32>(0.998));
    let uvw2 = clamp(worldToUVW(pos + 0.14 * stretch - sheetVec, u), vec3<f32>(0.002), vec3<f32>(0.998));
    let uvw3 = clamp(worldToUVW(pos + 0.42 * stretch, u), vec3<f32>(0.002), vec3<f32>(0.998));
    splatTrilinearIntact(uvw0, 0.21 * mass);
    splatTrilinearIntact(uvw1, 0.29 * mass);
    splatTrilinearIntact(uvw2, 0.29 * mass);
    splatTrilinearIntact(uvw3, 0.21 * mass);
  }
}
`;
