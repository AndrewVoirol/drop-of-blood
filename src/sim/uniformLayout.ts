import type { Mat4, Vec3 } from '../math/camera';
import { MAX_ACTIVE_DROPS } from '../shaders/common';
import type { MultiDropSnapshot } from './dropKinematics';
import { WATER_SURFACE_Y } from './dropKinematics';

export const SIM_UNIFORM_FLOATS = 128;
export const SIM_UNIFORM_BYTES = SIM_UNIFORM_FLOATS * 4; // 512 bytes (exact 256-byte WebGPU alignment)

export const DOMAIN_HALF_XZ = 0.48; // 0.96 width across 96 vel / 192 conc voxels (exact cubic voxels)
export const DOMAIN_Y_MIN = -0.98;  // 1.28 height [-0.98, +0.30] across 128 vel / 256 conc voxels
export const DOMAIN_Y_MAX = WATER_SURFACE_Y; // +0.30
export const FRONT_PANE_DIST = 1.10;

export interface FluidOpticalConfig {
  vorticityConfinement: number;
  buoyancyStrength: number;
  velocityDamping: number;
  concSharpness: number;
  extinctionScale: number;
  scatterAlbedo: number;
  causticStrength: number;
  lightingMode: number; // 0 = Studio Backlit Crystal, 1 = Chiaroscuro Dark Field
  marangoniStrength: number;
  apertureRadius: number;
  cassonYield: number;
  hemolysisRate: number;
  schlierenStrength: number;
  neckCapillaryLength: number;
}

export const DEFAULT_FLUID_OPTICAL_CONFIG: FluidOpticalConfig = {
  vorticityConfinement: 1.02,
  buoyancyStrength: 0.48,
  velocityDamping: 0.9993,
  concSharpness: 0.995,
  extinctionScale: 1.0,
  scatterAlbedo: 1.0,
  causticStrength: 1.0,
  lightingMode: 0.0,
  marangoniStrength: 1.0,
  apertureRadius: 0.0032,
  cassonYield: 1.0,
  hemolysisRate: 0.088,
  schlierenStrength: 0.032,
  neckCapillaryLength: 0.028,
};

/**
 * Zero-allocation packer for the 512-byte WGSL SimUniforms block (128 floats).
 * Every field aligns to 16-byte vec4/mat4 boundaries for 100% WGSL uniform parity.
 */
export class SimUniformPacker {
  public readonly buffer = new ArrayBuffer(SIM_UNIFORM_BYTES);
  public readonly f32 = new Float32Array(this.buffer);

  public pack(params: {
    invViewProj: Mat4;
    viewProj: Mat4;
    cameraPos: Vec3;
    cameraTarget: Vec3;
    viewportWidth: number;
    viewportHeight: number;
    dt: number;
    frameIndex: number;
    globalSeed: number;
    drops: MultiDropSnapshot;
    spawnCursor: number;
    spawnCount: number;
    config: FluidOpticalConfig;
  }): Float32Array {
    const out = this.f32;
    const { invViewProj, viewProj, cameraPos, cameraTarget, drops, config } = params;

    // [0..15] invViewProj
    out.set(invViewProj, 0);
    // [16..31] viewProj
    out.set(viewProj, 16);

    // [32..35] cameraPosAndTime
    out[32] = cameraPos[0];
    out[33] = cameraPos[1];
    out[34] = cameraPos[2];
    out[35] = drops.simTime;

    // [36..39] viewportAndDt
    out[36] = Math.max(1, params.viewportWidth);
    out[37] = Math.max(1, params.viewportHeight);
    out[38] = params.dt;
    out[39] = params.frameIndex;

    // [40..43] surfaceParams (.x = surfaceY, .y = marangoniStr, .z = activeDropCount, .w = globalSeed)
    out[40] = WATER_SURFACE_Y;
    out[41] = config.marangoniStrength;
    out[42] = drops.activeCount;
    out[43] = params.globalSeed;

    // [44..47] fluidParams
    out[44] = config.vorticityConfinement;
    out[45] = config.buoyancyStrength;
    out[46] = config.velocityDamping;
    out[47] = config.concSharpness;

    // [48..51] opticalParams
    out[48] = config.extinctionScale;
    out[49] = config.scatterAlbedo;
    out[50] = config.causticStrength;
    out[51] = config.lightingMode;

    // [52..55] domainBounds
    out[52] = DOMAIN_HALF_XZ;
    out[53] = DOMAIN_Y_MIN;
    out[54] = DOMAIN_Y_MAX;
    out[55] = FRONT_PANE_DIST;

    // Compute thin-lens focal distance along the camera optical axis: anchor to aerial/pendant drop
    // when viewing near/above the meniscus from topside, blending smoothly into the 3D orbit target
    // when submerged or when manually focused/panned onto a deep underwater plume.
    const dxT = cameraTarget[0] - cameraPos[0];
    const dyT = cameraTarget[1] - cameraPos[1];
    const dzT = cameraTarget[2] - cameraPos[2];
    const baseTargetDist = Math.hypot(dxT, dyT, dzT);
    let focalDistance = Math.max(0.20, baseTargetDist);
    if (drops.primaryAerialFocus && cameraPos[1] > WATER_SURFACE_Y + 0.004) {
      const targetTopsideGate = Math.min(
        1.0,
        Math.max(0.0, (cameraTarget[1] - (WATER_SURFACE_Y - 0.24)) / 0.12)
      );
      if (targetTopsideGate > 0.001) {
        const f = drops.primaryAerialFocus;
        const dfx = f[0] - cameraPos[0];
        const dfy = f[1] - cameraPos[1];
        const dfz = f[2] - cameraPos[2];
        const dropAxialDist =
          baseTargetDist > 1e-6
            ? (dfx * dxT + dfy * dyT + dfz * dzT) / baseTargetDist
            : Math.hypot(dfx, dfy, dfz);
        const airBlend =
          Math.min(1.0, Math.max(0.0, (cameraPos[1] - WATER_SURFACE_Y) / 0.025)) *
          targetTopsideGate;
        focalDistance = focalDistance * (1.0 - airBlend) + Math.max(0.20, dropAxialDist) * airBlend;
      }
    }

    // Automatic macro aperture stop-down inside < 0.82 m (the newly unlocked macro zoom range) keeps both
    // near and far vortex-ring filaments sharp during close-up inspection while leaving the entire legacy
    // [0.82, 2.85] m camera range and all default presets 100% bit-exact (scale === 1.0).
    const macroStopScale = Math.min(1.0, Math.pow(Math.max(focalDistance, 0.20) / 0.82, 1.25));

    // [56..59] lensAndRheology (.x = focalDistance, .y = apertureRadius, .z = cassonYield, .w = hemolysisRate)
    out[56] = focalDistance;
    out[57] = Math.max(0.0, config.apertureRadius) * macroStopScale;
    out[58] = config.cassonYield;
    out[59] = config.hemolysisRate;

    // [60..63] particleParams (.x = schlierenStr, .y = spawnCursor, .z = spawnCount, .w = neckCapillaryLen)
    out[60] = config.schlierenStrength;
    out[61] = params.spawnCursor;
    out[62] = params.spawnCount;
    out[63] = config.neckCapillaryLength;

    // [64..127] drops: array<DropGpuSlot, 8> (8 floats per slot = 64 floats)
    const slots = drops.slots;
    for (let i = 0; i < MAX_ACTIVE_DROPS; i++) {
      const base = 64 + i * 8;
      const s = slots[i];
      if (s && s.active) {
        out[base + 0] = s.center[0];
        out[base + 1] = s.center[1];
        out[base + 2] = s.center[2];
        out[base + 3] = s.radius;
        out[base + 4] = s.vel[1];
        out[base + 5] = s.oscillation;
        // Positive .z = active meniscus volume drain rate; negative .z = -pipettePresence for glass capillary tip
        out[base + 6] = s.volDrainRate > 0.0 ? s.volDrainRate : -Math.max(0.0, s.pipettePresence);
        // Negative signedSeed marks a suspended pendant drop; positive marks free/coalescing drop
        out[base + 7] = s.mode === 'pendant' ? -Math.abs(s.seed) : Math.abs(s.seed);
      } else {
        out[base + 0] = 0.0;
        out[base + 1] = 0.0;
        out[base + 2] = 0.0;
        out[base + 3] = 0.0;
        out[base + 4] = 0.0;
        out[base + 5] = 0.0;
        out[base + 6] = 0.0;
        out[base + 7] = 0.0;
      }
    }

    return out;
  }
}
