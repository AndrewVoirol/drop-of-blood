import {
  BRICK_GRID_X,
  BRICK_GRID_Y,
  BRICK_GRID_Z,
  COMMON_WGSL,
  MAX_ACTIVE_DROPS,
  VEL_GRID_X,
  VEL_GRID_Y,
  VEL_GRID_Z,
} from './common';

/**
 * Computes a 96x128x96 3D Volumetric Optical Cache (texLightVol, rgba16float):
 *   .r = Overhead key light transmittance * dispersive wave caustics * multi-drop ball-lens caustics
 *   .g = View-coupled studio backlight transmittance (intact cellular extinction vs clear lysed dye)
 *   .b = Local multi-scattered cellular RBC radiance neighborhood (strictly from intact conc.r)
 *   .a = Smooth 4-tap tetrahedral total hemoglobin c_total for Schlieren grad_perp(n) & macro DOF footprinting
 */
export const LIGHTING_VOLUME_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var concIn: texture_3d<f32>;
@group(0) @binding(2) var waveIn: texture_2d<f32>;
@group(0) @binding(3) var linSampler: sampler;
@group(0) @binding(4) var lightOut: texture_storage_3d<rgba16float, write>;

fn evalOpticalDensity(c: vec4<f32>) -> f32 {
  // Intact RBCs (c.r) attenuate collimated beams via both absorption + cellular Mie scattering,
  // whereas lysed free hemoglobin (c.g) exhibits pure molecular absorption without scattering haze.
  return 1.16 * c.r + 0.78 * c.g;
}

@compute @workgroup_size(8, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${VEL_GRID_X}u || gid.y >= ${VEL_GRID_Y}u || gid.z >= ${VEL_GRID_Z}u) {
    return;
  }
  let dims = vec3<f32>(${VEL_GRID_X}.0, ${VEL_GRID_Y}.0, ${VEL_GRID_Z}.0);
  let uvw = (vec3<f32>(gid) + 0.5) / dims;
  let p = uvwToWorld(uvw, u);
  let domSize = domainSize(u);

  // Outer sponge margin where concOut is attenuated to zero: skip all 75.5MB 3D texture reads
  if (max(abs(p.x), abs(p.z)) > 0.462) {
    textureStore(lightOut, vec3<i32>(gid), vec4<f32>(1.0, 1.0, 0.0, 0.0));
    return;
  }

  // 5-tap center + isotropic tetrahedral neighborhood of total hemoglobin (.r + .g) & intact RBCs (.r):
  // Center tap ctC guarantees <10mm hairline stems along voxel axes are never straddled by diagonal corners (r = 14.4mm),
  // while the 4 tetrahedral corners preserve the 28mm dilated bokeh support (> 22mm max rWorldCone)!
  let ctC = textureSampleLevel(concIn, linSampler, uvw, 0.0);
  let epsT = vec3<f32>(1.02) / dims;
  let ct0 = textureSampleLevel(concIn, linSampler, clamp(uvw + vec3<f32>( epsT.x, -epsT.y, -epsT.z), vec3<f32>(0.0), vec3<f32>(1.0)), 0.0);
  let ct1 = textureSampleLevel(concIn, linSampler, clamp(uvw + vec3<f32>(-epsT.x, -epsT.y,  epsT.z), vec3<f32>(0.0), vec3<f32>(1.0)), 0.0);
  let ct2 = textureSampleLevel(concIn, linSampler, clamp(uvw + vec3<f32>(-epsT.x,  epsT.y, -epsT.z), vec3<f32>(0.0), vec3<f32>(1.0)), 0.0);
  let ct3 = textureSampleLevel(concIn, linSampler, clamp(uvw + vec3<f32>( epsT.x,  epsT.y,  epsT.z), vec3<f32>(0.0), vec3<f32>(1.0)), 0.0);

  let totC          = ctC.r + ctC.g;
  let tetTotal      = 0.25 * ((ct0.r + ct0.g) + (ct1.r + ct1.g) + (ct2.r + ct2.g) + (ct3.r + ct3.g));
  let filteredTotal = max(0.50 * totC + 0.50 * tetTotal, tetTotal);
  let occDensity    = max(totC, filteredTotal);
  let smoothIntact  = 0.28 * ctC.r + 0.18 * (ct0.r + ct1.r + ct2.r + ct3.r);

  // Exact empty-water early-out: marchPlume masks all voxels <= 4.0e-5 to zero opacity;
  // since occDensity = max(totC, filteredTotal), flushing < 2.8e-5 culls invisible diffusion dust while keeping 100% of thin stems!
  if (occDensity < 0.000028) {
    textureStore(lightOut, vec3<i32>(gid), vec4<f32>(1.0, 1.0, 0.0, 0.0));
    return;
  }

  let darkMode = u.opticalParams.w > 0.5;

  // 1. Key Light + Dispersive Free-Surface Caustics + Multi-Drop Ball-Lens Attenuation
  //    In Dark-Field mode, keyDir shifts to an oblique upper-right chiaroscuro side-key beam!
  let keyDir = select(
    normalize(vec3<f32>(0.22, 0.95, 0.20)),
    normalize(vec3<f32>(0.64, 0.70, 0.32)),
    darkMode
  );
  let distToSurf = max(0.0, (u.surfaceParams.x - p.y) / keyDir.y);
  let surfHitXZ = p.xz + keyDir.xz * distToSurf;
  let waveUV = worldXZToWaveUV(surfHitXZ);
  var caustic = 1.0;
  if (all(waveUV >= vec2<f32>(0.01)) && all(waveUV <= vec2<f32>(0.99))) {
    let wSample = textureSampleLevel(waveIn, linSampler, waveUV, 0.0);
    let depthFactor = smoothstep(0.0, 0.30, u.surfaceParams.x - p.y);
    caustic = clamp(1.0 - wSample.b * 28.0 * depthFactor * u.opticalParams.z, 0.72, 1.38);
  }

  // Account for overhead shadow/ball-lens caustic of any active aerial/pendant drops above the meniscus
  for (var d = 0u; d < ${MAX_ACTIVE_DROPS}u; d = d + 1u) {
    let slot = u.drops[d];
    let rDrop = slot.posAndRadius.w;
    if (rDrop > 0.002 && slot.posAndRadius.y > p.y && slot.posAndRadius.y > u.surfaceParams.x - 0.55 * rDrop) {
      let tDrop = (slot.posAndRadius.y - p.y) / keyDir.y;
      let rayAtDropXZ = p.xz + keyDir.xz * tDrop;
      let rNorm = length(rayAtDropXZ - slot.posAndRadius.xz) / rDrop;
      if (rNorm < 1.25) {
        // Ball-lens absorbs outer chord while focusing a subtle central ruby caustic spot
        let shadowRing = 1.0 - 0.55 * (1.0 - smoothstep(0.55, 1.15, rNorm));
        let focusCore = 1.0 + 0.32 * (1.0 - smoothstep(0.0, 0.42, rNorm));
        caustic = caustic * shadowRing * focusCore;
      }
    }
  }

  let keyShadowScale = select(22.0, 38.0, darkMode) * u.opticalParams.x;
  var tauKey = 0.0;
  let stepKey = 0.042;
  let dUVWKey = (keyDir * stepKey) / domSize;
  var posUVW = uvw + dUVWKey * 0.5;
  for (var i = 0; i < 9; i = i + 1) {
    if (any(posUVW < vec3<f32>(0.0)) || any(posUVW > vec3<f32>(1.0)) || tauKey * keyShadowScale > 4.8) {
      break;
    }
    let cs = textureSampleLevel(concIn, linSampler, posUVW, 0.0);
    tauKey = tauKey + evalOpticalDensity(cs) * stepKey;
    posUVW = posUVW + dUVWKey;
  }
  let keyLight = caustic * exp(-tauKey * keyShadowScale);

  // 2. View-Coupled Backlight Softbox (Studio mode) vs Opposing Grazing Rim Beam (Dark-Field Chiaroscuro)
  let camPos = u.cameraPosAndTime.xyz;
  let toOrigin = vec3<f32>(0.0, -0.05, 0.0) - camPos;
  let camRightXZ = normalize(vec2<f32>(-toOrigin.z, toOrigin.x) + vec2<f32>(1e-6, 0.0));
  let studioBackDir = normalize(vec3<f32>(toOrigin.x, 0.22, toOrigin.z));
  let chiaroRimDir  = normalize(vec3<f32>(-0.78 * camRightXZ.x + 0.38 * toOrigin.x, 0.24, -0.78 * camRightXZ.y + 0.38 * toOrigin.z));
  let backDir = select(studioBackDir, chiaroRimDir, darkMode);

  let backShadowScale = select(23.0, 42.0, darkMode) * u.opticalParams.x;
  var tauBack = 0.0;
  let stepBack = 0.040;
  let dUVWBack = (backDir * stepBack) / domSize;
  var backUVW = uvw + dUVWBack * 0.5;
  for (var j = 0; j < 9; j = j + 1) {
    if (any(backUVW < vec3<f32>(0.0)) || any(backUVW > vec3<f32>(1.0)) || tauBack * backShadowScale > 4.8) {
      break;
    }
    let cs = textureSampleLevel(concIn, linSampler, backUVW, 0.0);
    tauBack = tauBack + evalOpticalDensity(cs) * stepBack;
    backUVW = backUVW + dUVWBack;
  }
  let backLight = exp(-tauBack * backShadowScale);

  // 3. Multi-scattered 650-680 nm diffusive red photon transport in intact erythrocytes (smoothIntact):
  //    Long-wavelength red photons penetrate 4-6x deeper than Q-band green/blue, preventing thick multi-drop
  //    cores and settled floor rouleaux from crushing to neutral charcoal black!
  let midScatter = exp(-tauKey * select(15.0, 26.0, darkMode)) * 0.44 + exp(-tauBack * select(14.0, 28.0, darkMode)) * 0.36;
  let deepRedDiff = exp(-tauKey * select(5.8, 10.5, darkMode)) * 0.30 + exp(-tauBack * select(5.2, 10.0, darkMode)) * 0.28 + select(0.14, 0.07, darkMode);
  let multiScatter = smoothIntact * (midScatter + 0.62 * deepRedDiff);

  textureStore(
    lightOut,
    vec3<i32>(gid),
    vec4<f32>(keyLight, backLight, multiScatter, filteredTotal)
  );
}
`;

/**
 * Pass 1 of 3D Empty-Space Brick Hierarchy (24x32x24 r32float, 40mm cubic bricks):
 * Reduces each 4x4x4 block of lightTex.a to 1.0 if any texel has filteredTotal >= 2.8e-5.
 */
export const PLUME_OCCUPANCY_RAW_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var lightIn: texture_3d<f32>;
@group(0) @binding(2) var brickRawOut: texture_storage_3d<r32float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${BRICK_GRID_X}u || gid.y >= ${BRICK_GRID_Y}u || gid.z >= ${BRICK_GRID_Z}u) {
    return;
  }
  let base = vec3<i32>(gid) * 4;
  var maxA = 0.0;
  for (var dz = 0; dz < 4; dz = dz + 1) {
    for (var dy = 0; dy < 4; dy = dy + 1) {
      for (var dx = 0; dx < 4; dx = dx + 1) {
        maxA = max(maxA, textureLoad(lightIn, base + vec3<i32>(dx, dy, dz), 0).a);
      }
    }
  }
  let occFlag = select(0.0, 1.0, maxA >= 0.000028) + u.domainBounds.x * 0.0;
  textureStore(brickRawOut, vec3<i32>(gid), vec4<f32>(occFlag, 0.0, 0.0, 0.0));
}
`;

/**
 * Pass 2 of 3D Empty-Space Brick Hierarchy (24x32x24 r32float):
 * Computes exact 2-shell Chebyshev dilation over brickRawIn:
 *   - 2.0: within +-1 brick (+-40mm + 28mm tetrahedral halo = 68mm > max 22mm thin-lens bokeh cone!)
 *   - 1.0: within +-2 bricks (+-80mm clearance -> safe 34mm mid-skip)
 *   - 0.0: >80mm from any active plume voxel -> safe 72mm macro-skip across empty tank water
 */
export const PLUME_OCCUPANCY_DILATE_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var brickRawIn: texture_3d<f32>;
@group(0) @binding(2) var brickDilatedOut: texture_storage_3d<r32float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${BRICK_GRID_X}u || gid.y >= ${BRICK_GRID_Y}u || gid.z >= ${BRICK_GRID_Z}u) {
    return;
  }
  let c = vec3<i32>(gid);
  let maxCoord = vec3<i32>(${BRICK_GRID_X - 1}, ${BRICK_GRID_Y - 1}, ${BRICK_GRID_Z - 1});
  var near1 = false;
  var near2 = false;

  for (var dz = -2; dz <= 2; dz = dz + 1) {
    for (var dy = -2; dy <= 2; dy = dy + 1) {
      for (var dx = -2; dx <= 2; dx = dx + 1) {
        let nc = clamp(c + vec3<i32>(dx, dy, dz), vec3<i32>(0), maxCoord);
        if (textureLoad(brickRawIn, nc, 0).r > 0.5) {
          near2 = true;
          if (max(abs(dx), max(abs(dy), abs(dz))) <= 1) {
            near1 = true;
          }
        }
      }
    }
  }

  let shellVal = select(select(0.0, 1.0, near2), 2.0, near1) + u.domainBounds.x * 0.0;
  textureStore(brickDilatedOut, c, vec4<f32>(shellVal, 0.0, 0.0, 0.0));
}
`;

