export const WAVE_GRID_SIZE = 256;
export const WAVE_HALF_EXTENT = 1.60; // X, Z in [-1.60, +1.60] (3.20 m full-canvas open meniscus horizon)

export const VEL_GRID_X = 96;
export const VEL_GRID_Y = 128;
export const VEL_GRID_Z = 96;

export const BRICK_GRID_X = 24; // 4x4x4 reduction of 96x128x96 lightTex (40mm cubic bricks)
export const BRICK_GRID_Y = 32;
export const BRICK_GRID_Z = 24;

export const CONC_GRID_X = 192;
export const CONC_GRID_Y = 256;
export const CONC_GRID_Z = 192;
export const CONC_TOTAL_VOXELS = CONC_GRID_X * CONC_GRID_Y * CONC_GRID_Z; // 9,437,184

export const MAX_ACTIVE_DROPS = 8;
export const PARTICLE_COUNT = 393216; // 384 * 1024 persistent Lagrangian RBC parcels (12 MB)
export const PARTICLE_WORKGROUP_SIZE = 256;
export const SPLAT_FP_SCALE = 65536.0;

export const SPECTRAL_LUT_WIDTH = 256;
export const SPECTRAL_LUT_HEIGHT = 64;

export const COMMON_WGSL = /* wgsl */ `
struct DropGpuSlot {
  posAndRadius: vec4<f32>,   // .xyz = center, .w = aerialRadius (0 if inactive/submerged)
  velOscAndDrain: vec4<f32>, // .x = velY, .y = l2Oscillation, .z = volDrainRate, .w = signedSeed (<0 pendant, >0 free)
};

struct SimUniforms {
  invViewProj: mat4x4<f32>,      // [0..15]
  viewProj: mat4x4<f32>,         // [16..31]
  cameraPosAndTime: vec4<f32>,   // [32..35] .xyz = cameraPos, .w = simTime
  viewportAndDt: vec4<f32>,      // [36..39] .x = width, .y = height, .z = dt, .w = frameIndex
  surfaceParams: vec4<f32>,      // [40..43] .x = surfaceY, .y = marangoniStr, .z = activeDropCount, .w = globalSeed
  fluidParams: vec4<f32>,        // [44..47] .x = vorticityConf, .y = buoyancy, .z = velDamp, .w = concSharpness
  opticalParams: vec4<f32>,      // [48..51] .x = extinctionScale, .y = scatterAlbedo, .z = causticStr, .w = lightingMode
  domainBounds: vec4<f32>,       // [52..55] .x = halfXZ (0.48), .y = yMin (-0.98), .z = yMax (+0.30), .w = frontPaneDist (1.10)
  lensAndRheology: vec4<f32>,    // [56..59] .x = focalDistance, .y = apertureRadius, .z = cassonYield, .w = hemolysisRate
  particleParams: vec4<f32>,     // [60..63] .x = schlierenStr, .y = spawnCursor, .z = spawnCount, .w = neckCapillaryLen
  drops: array<DropGpuSlot, ${MAX_ACTIVE_DROPS}>, // [64..127] 8 * 8 floats = 256 bytes (total 512 bytes)
};

fn domainMin(u: SimUniforms) -> vec3<f32> {
  return vec3<f32>(-u.domainBounds.x, u.domainBounds.y, -u.domainBounds.x);
}

fn domainMax(u: SimUniforms) -> vec3<f32> {
  return vec3<f32>(u.domainBounds.x, u.domainBounds.z, u.domainBounds.x);
}

fn domainSize(u: SimUniforms) -> vec3<f32> {
  return domainMax(u) - domainMin(u);
}

fn worldToUVW(p: vec3<f32>, u: SimUniforms) -> vec3<f32> {
  return (p - domainMin(u)) / domainSize(u);
}

fn uvwToWorld(uvw: vec3<f32>, u: SimUniforms) -> vec3<f32> {
  return domainMin(u) + uvw * domainSize(u);
}

fn worldXZToWaveUV(xz: vec2<f32>) -> vec2<f32> {
  return (xz + vec2<f32>(${WAVE_HALF_EXTENT.toFixed(2)})) / ${(WAVE_HALF_EXTENT * 2.0).toFixed(2)};
}

fn waveUVToWorldXZ(uv: vec2<f32>) -> vec2<f32> {
  return uv * ${(WAVE_HALF_EXTENT * 2.0).toFixed(2)} - vec2<f32>(${WAVE_HALF_EXTENT.toFixed(2)});
}

fn hash13(p3_in: vec3<f32>) -> f32 {
  var p3 = fract(p3_in * 0.1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}

fn hash33(p3_in: vec3<f32>) -> vec3<f32> {
  var p3 = fract(p3_in * vec3<f32>(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx);
}
`;
