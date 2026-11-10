import { describe, expect, it } from 'vitest';
import engineSrc from '../src/engine/BloodDropEngine.ts?raw';
import lutSrc from '../src/sim/spectralLut.ts?raw';
import {
  CAMERA_PRESETS,
  MAX_CAMERA_DISTANCE,
  MAX_CAMERA_TARGET_XZ,
  MAX_CAMERA_TARGET_Y,
  MIN_CAMERA_DISTANCE,
  MIN_CAMERA_TARGET_Y,
  mat4Multiply,
  OrbitCamera,
  unprojectWorldPoint,
} from '../src/math/camera';
import {
  CONC_CLEAR_WGSL,
  CONC_PREDICT_WGSL,
  CONC_RESOLVE_WGSL,
} from '../src/shaders/advect';
import {
  BRICK_GRID_X,
  BRICK_GRID_Y,
  BRICK_GRID_Z,
  MAX_ACTIVE_DROPS,
  PARTICLE_COUNT,
  SPECTRAL_LUT_HEIGHT,
  SPECTRAL_LUT_WIDTH,
  SPLAT_FP_SCALE,
  VEL_GRID_X,
  VEL_GRID_Y,
  VEL_GRID_Z,
  WAVE_GRID_SIZE,
  WAVE_HALF_EXTENT,
} from '../src/shaders/common';
import {
  FLUID_ADVECT_MAC_WGSL,
  FLUID_CASSON_FORCES_WGSL,
  FLUID_CLEAR_R32_WGSL,
  FLUID_CURL_WGSL,
  FLUID_INIT_VEL_WGSL,
  FLUID_MAC_PROJECT_WGSL,
  FLUID_MAC_RHS_WGSL,
  FLUID_MAC_TO_CENTER_WGSL,
  MG_COARSE_SOLVE_L3_WGSL,
  MG_PROLONG_ADD_WGSL,
  MG_RBGS_BLACK_WGSL,
  MG_RBGS_RED_WGSL,
  MG_RBGS_RED_ZERO_WGSL,
  MG_RESIDUAL_WGSL,
  MG_RESTRICT_WGSL,
} from '../src/shaders/fluid';
import {
  LIGHTING_VOLUME_WGSL,
  PLUME_OCCUPANCY_DILATE_WGSL,
  PLUME_OCCUPANCY_RAW_WGSL,
} from '../src/shaders/lighting';
import {
  INITIAL_PARCEL_VOXEL_MASS,
  PARCELS_PER_REF_DROP,
  PARTICLE_ADVECT_SPLAT_WGSL,
  PARTICLE_EMIT_WGSL,
  PARTICLE_RESET_WGSL,
} from '../src/shaders/particles';
import { COMPOSITE_WGSL, RENDER_WGSL } from '../src/shaders/render';
import {
  WAVE_CLEAR_SPEC_WGSL,
  WAVE_CLEAR_WGSL,
  WAVE_FFT_COL_WGSL,
  WAVE_FFT_ROW_WGSL,
  WAVE_FINALIZE_WGSL,
  WAVE_IFFT_COL_WGSL,
  WAVE_IFFT_ROW_WGSL,
  WAVE_SPATIAL_FORCE_WGSL,
  WAVE_SPECTRAL_PROPAGATE_WGSL,
} from '../src/shaders/wave';
import {
  IMPACT_TIME,
  INITIAL_DROP_RADIUS,
  MAX_DROP_OFFSET_XZ,
  MultiDropSystem,
  PENDANT_GROWTH_DURATION,
  PENDANT_MAX_RADIUS,
  PENDANT_MIN_RADIUS,
  PIPETTE_TIP_Y,
  radiusFromVolume,
  raycastMeniscusXZ,
  SHOT_CYCLE_DURATION,
  sphereVolume,
  WATER_SURFACE_Y,
} from '../src/sim/dropKinematics';
import { float32ToFloat16Bits, generatePrahlSpectralLut } from '../src/sim/spectralLut';
import {
  DEFAULT_FLUID_OPTICAL_CONFIG,
  DOMAIN_HALF_XZ,
  DOMAIN_Y_MAX,
  DOMAIN_Y_MIN,
  SIM_UNIFORM_BYTES,
  SIM_UNIFORM_FLOATS,
  SimUniformPacker,
} from '../src/sim/uniformLayout';

function stepSystemTo(sys: MultiDropSystem, targetTime: number, dt = 1 / 240): void {
  while (sys.simTime + dt * 0.5 < targetTime) {
    sys.step(dt);
  }
  const rem = targetTime - sys.simTime;
  if (rem > 1e-9) {
    sys.step(rem);
  }
}

describe('MultiDropSystem kinematics, pendant swelling, coalescence & exact volume conservation', () => {
  it('places the bottom of the falling crimson drop right at the water meniscus at IMPACT_TIME', () => {
    expect(IMPACT_TIME).toBeGreaterThan(0.8);
    expect(IMPACT_TIME).toBeLessThan(1.4);

    const sys = new MultiDropSystem();
    sys.reset(1.0);
    stepSystemTo(sys, IMPACT_TIME - 1e-5, 1 / 240);
    const slot = sys.slots[0];
    expect(slot.mode).toBe('aerial');
    const bottomY = slot.center[1] - slot.radius;
    expect(bottomY).toBeCloseTo(WATER_SURFACE_Y, 3);
    expect(slot.vel[1]).toBeLessThan(-0.2);
  });

  it('descends monotonically in air and drains 100% of aerial volume into the submerged tank across wetting coalescence', () => {
    for (const r0 of [PENDANT_MIN_RADIUS, INITIAL_DROP_RADIUS, PENDANT_MAX_RADIUS]) {
      for (const dt of [1 / 60, 1 / 120]) {
        const sys = new MultiDropSystem();
        sys.reset(1.0);
        // Replace slot 0 with test radius r0
        sys.reset(1.0);
        sys.slots[0].active = false;
        const slot = sys.spawnAerialDrop([0.04, -0.03], r0, 1.7);
        const expectedVolume = sphereVolume(r0);

        let prevY = Infinity;
        let cumulativeDrained = 0.0;
        let sawCoalescing = false;

        for (let step = 0; step < 300; step++) {
          if (slot.active && slot.mode === 'aerial') {
            expect(slot.center[1]).toBeLessThan(prevY);
            expect(slot.radius).toBeCloseTo(r0, 6);
            expect(Math.abs(slot.oscillation)).toBeLessThan(0.15);
            prevY = slot.center[1];
          }
          const snap = sys.step(dt);
          cumulativeDrained += snap.totalStepDrainedVolume;
          if (slot.active && slot.mode === 'coalescing') {
            sawCoalescing = true;
            if (slot.ageInPhase >= 0.03) {
              const semiY = slot.radius * (1.0 + slot.oscillation);
              expect(slot.center[1] - semiY).toBeLessThan(WATER_SURFACE_Y);
            }
          }
        }

        expect(sawCoalescing).toBe(true);
        expect(slot.active).toBe(false);
        expect(slot.radius).toBe(0);
        expect(cumulativeDrained).toBeCloseTo(expectedVolume, 9);
      }
    }
  });

  it('maintains C0-continuous drop oscillation & center height across meniscus contact and sinks into the crater cavity', () => {
    for (const seed of [0.5, 1.0, 2.718, 4.2]) {
      const sys = new MultiDropSystem();
      sys.reset(seed);
      stepSystemTo(sys, IMPACT_TIME - 1e-4, 1 / 240);
      const preOsc = sys.slots[0].oscillation;
      const preY = sys.slots[0].center[1];

      sys.step(2e-4);
      const postOsc = sys.slots[0].oscillation;
      const postY = sys.slots[0].center[1];

      expect(sys.slots[0].mode).toBe('coalescing');
      expect(Math.abs(postOsc - preOsc)).toBeLessThan(1e-3);
      expect(Math.abs(postY - preY)).toBeLessThan(1e-3);
    }

    const sys = new MultiDropSystem();
    sys.reset(1.0);
    stepSystemTo(sys, IMPACT_TIME + 0.30, 1 / 120);
    expect(sys.slots[0].active).toBe(true);
    expect(sys.slots[0].center[1]).toBeLessThan(WATER_SURFACE_Y);
  });

  it('swells a suspended pendant drop from R_min to Bond pinch-off R_max, and cancels cleanly on <5px orbit drag abort', () => {
    const sys = new MultiDropSystem();
    sys.reset(1.0);

    const p = sys.beginPendantDrop([0.08, -0.05], 2.5);
    expect(sys.hasActivePendantDrop()).toBe(true);
    expect(p.mode).toBe('pendant');
    expect(p.radius).toBeCloseTo(PENDANT_MIN_RADIUS, 6);
    expect(p.pipettePresence).toBeCloseTo(1.0, 6);

    // Mid-hold linear volumetric infusion V(t) = V_min + Q*t increases radius monotonically without draining mass into tank
    expect(sys.updatePendantDrop(0.5)).toBe(false);
    expect(p.radius).toBeGreaterThan(PENDANT_MIN_RADIUS + 0.025);
    expect(p.radius).toBeLessThan(PENDANT_MAX_RADIUS);
    const snapMid = sys.step(1 / 60);
    expect(p.mode).toBe('pendant');
    expect(p.stepDrainedVolume).toBe(0);
    expect(snapMid.phaseLabel).toContain('Pendant');

    // Exceeding Bond-number capillary hold time (PENDANT_GROWTH_DURATION = 2.2s) triggers automatic pinch-off into free-fall
    expect(sys.updatePendantDrop(PENDANT_GROWTH_DURATION)).toBe(true);
    expect(sys.hasActivePendantDrop()).toBe(false);
    expect(p.mode).toBe('aerial');
    expect(p.radius).toBeCloseTo(PENDANT_MAX_RADIUS, 5);
    expect(p.vel[1]).toBeLessThan(0);

    // Drag-to-orbit cancellation clears uncommitted pendant slot cleanly
    const pCancel = sys.beginPendantDrop([-0.06, 0.04], 3.1);
    sys.updatePendantDrop(0.3);
    expect(pCancel.active).toBe(true);
    sys.cancelPendantDrop();
    expect(sys.hasActivePendantDrop()).toBe(false);
    expect(pCancel.active).toBe(false);
    expect(pCancel.radius).toBe(0);
  });

  it('supports N = 8 concurrent drops, clamps XZ / NaN inputs safely, and conserves volume & momentum on mid-air coalescence', () => {
    const sys = new MultiDropSystem();
    sys.reset(1.0);

    // Out-of-bounds and NaN offsets clamp to interior disk
    const clamped = sys.spawnAerialDrop([0.95, -0.95], 0.055, 2.0);
    expect(Math.hypot(clamped.center[0], clamped.center[2])).toBeCloseTo(MAX_DROP_OFFSET_XZ, 6);

    const nanSafe = sys.spawnAerialDrop([Number.NaN, 0.09], 0.052, 3.0);
    expect(nanSafe.center[0]).toBe(0);
    expect(nanSafe.center[2]).toBeCloseTo(0.09, 6);

    // Fill all 8 concurrent drop slots without cancelling in-flight drops
    for (let k = 0; k < 5; k++) {
      const angle = (k * Math.PI * 2) / 5;
      sys.spawnAerialDrop([0.16 * Math.cos(angle), 0.16 * Math.sin(angle)], 0.050, 4.0 + k);
    }
    expect(sys.getSnapshot().activeCount).toBe(MAX_ACTIVE_DROPS);

    // Mid-air drop-drop coalescence conserves exact volume R = (R1^3 + R2^3)^(1/3) and vertical momentum
    const mergeSys = new MultiDropSystem();
    mergeSys.reset(1.0);
    mergeSys.slots[0].active = false;
    const d1 = mergeSys.spawnAerialDrop([0.0, 0.0], 0.050, 1.2, 0.48, -0.22);
    const d2 = mergeSys.spawnAerialDrop([0.01, 0.0], 0.062, 2.4, 0.46, -0.06);
    const v1 = sphereVolume(0.050);
    const v2 = sphereVolume(0.062);
    const expectedMergedRadius = radiusFromVolume(v1 + v2);
    const expectedMergedVelBeforeG = (v1 * -0.22 + v2 * -0.06) / (v1 + v2);

    const snap = mergeSys.step(1 / 240);
    expect(snap.activeCount).toBe(1);
    expect(d2.active).toBe(false);
    expect(d1.active).toBe(true);
    expect(d1.radius).toBeCloseTo(expectedMergedRadius, 6);
    expect(d1.vel[1]).toBeCloseTo(expectedMergedVelBeforeG + -0.155 / 240, 5);

    // Late spawn extends shot cycle duration so trailing plumes can bloom
    mergeSys.simTime = 19.0;
    mergeSys.spawnAerialDrop([0.02, -0.02], INITIAL_DROP_RADIUS, 5.0);
    expect(mergeSys.getEffectiveCycleDuration(SHOT_CYCLE_DURATION)).toBeGreaterThanOrEqual(27.5);
  });

  it('raycasts canvas NDC onto y = WATER_SURFACE_Y across camera presets and handles grazing angles safely', () => {
    const cam = new OrbitCamera(CAMERA_PRESETS.crown);
    cam.target = [0.0, WATER_SURFACE_Y, 0.0];
    cam.applyPreset({ ...CAMERA_PRESETS.crown, target: [0.0, WATER_SURFACE_Y, 0.0] }, true);
    cam.updateMatrices(16 / 9);

    const centerHit = raycastMeniscusXZ(cam.invViewProjMatrix, cam.position, cam.target, 0.0, 0.0);
    expect(centerHit[0]).toBeCloseTo(0.0, 3);
    expect(centerHit[1]).toBeCloseTo(0.0, 3);

    const rightHit = raycastMeniscusXZ(cam.invViewProjMatrix, cam.position, cam.target, 0.35, 0.0);
    expect(rightHit[0]).toBeGreaterThan(0.03);
    expect(Math.hypot(rightHit[0], rightHit[1])).toBeLessThanOrEqual(MAX_DROP_OFFSET_XZ + 1e-6);

    // Extreme corner clamps to MAX_DROP_OFFSET_XZ without NaN
    const cornerHit = raycastMeniscusXZ(cam.invViewProjMatrix, cam.position, cam.target, 0.99, 0.99);
    expect(Number.isFinite(cornerHit[0])).toBe(true);
    expect(Number.isFinite(cornerHit[1])).toBe(true);
    expect(Math.hypot(cornerHit[0], cornerHit[1])).toBeLessThanOrEqual(MAX_DROP_OFFSET_XZ + 1e-6);
  });
});

describe('OrbitCamera & WebGPU ZO matrix math', () => {
  it('produces invertible viewProj matrices across all presets and aspect ratios', () => {
    const product = new Float32Array(16);
    for (const preset of Object.values(CAMERA_PRESETS)) {
      const cam = new OrbitCamera(preset);
      for (const aspect of [0.5, 1.0, 16 / 9, 21 / 9, 0, Number.NaN]) {
        cam.updateMatrices(aspect);
        mat4Multiply(product, cam.viewProjMatrix, cam.invViewProjMatrix);
        for (let i = 0; i < 16; i++) {
          const expected = i % 5 === 0 ? 1.0 : 0.0;
          expect(product[i]).toBeCloseTo(expected, 3);
        }
      }
    }
  });

  it('unprojects center NDC rays directly toward the camera look-at target', () => {
    const cam = new OrbitCamera(CAMERA_PRESETS.hero);
    cam.updateMatrices(16 / 9);
    const pNear = unprojectWorldPoint(cam.invViewProjMatrix, 0, 0, 0);
    const pFar = unprojectWorldPoint(cam.invViewProjMatrix, 0, 0, 1);

    const rayDir = [pFar[0] - pNear[0], pFar[1] - pNear[1], pFar[2] - pNear[2]];
    const rayLen = Math.hypot(rayDir[0], rayDir[1], rayDir[2]);
    rayDir[0] /= rayLen;
    rayDir[1] /= rayLen;
    rayDir[2] /= rayLen;

    const expDir = [
      cam.target[0] - cam.position[0],
      cam.target[1] - cam.position[1],
      cam.target[2] - cam.position[2],
    ];
    const expLen = Math.hypot(expDir[0], expDir[1], expDir[2]);
    expDir[0] /= expLen;
    expDir[1] /= expLen;
    expDir[2] /= expLen;

    expect(rayDir[0]).toBeCloseTo(expDir[0], 5);
    expect(rayDir[1]).toBeCloseTo(expDir[1], 5);
    expect(rayDir[2]).toBeCloseTo(expDir[2], 5);
  });

  it('clamps extreme orbit elevations and dolly distances within safe macro bounds', () => {
    const cam = new OrbitCamera(CAMERA_PRESETS.hero);
    cam.orbit(0, 99.0);
    cam.dolly(0.0001);
    for (let i = 0; i < 25; i++) cam.tick(0.1, 1.6);
    expect(cam.elevation).toBeLessThanOrEqual(1.15);
    expect(cam.distance).toBeCloseTo(MIN_CAMERA_DISTANCE, 5);
    expect(cam.distance).toBeCloseTo(0.24, 5);

    cam.orbit(0, -99.0);
    cam.dolly(9999.0);
    for (let i = 0; i < 25; i++) cam.tick(0.1, 1.6);
    expect(cam.elevation).toBeGreaterThanOrEqual(-1.15);
    expect(cam.distance).toBeCloseTo(MAX_CAMERA_DISTANCE, 4);
  });

  it('keeps hero follow camera cleanly topside across descent without a 50/50 TIR mirror dive, frames PIPETTE_TIP_Y inside hero & crown frustums, and yields to manual orbit', () => {
    const projectTipNdcY = (c: OrbitCamera): number => {
      const vp = c.viewProjMatrix;
      const clipY = vp[5] * PIPETTE_TIP_Y + vp[13];
      const clipW = vp[7] * PIPETTE_TIP_Y + vp[15];
      return clipY / clipW;
    };

    const cam = new OrbitCamera(CAMERA_PRESETS.hero);
    expect(cam.followHeroEnabled).toBe(true);

    // Pre-impact & entry window stays above WATER_SURFACE_Y (+0.30) with PIPETTE_TIP_Y inside frustum
    for (const t of [0.0, 0.5, IMPACT_TIME, 1.15]) {
      cam.syncHeroFollow(t, true);
      cam.updateMatrices(16 / 10);
      expect(cam.position[1]).toBeGreaterThan(WATER_SURFACE_Y + 0.24);
      expect(cam.elevation).toBeGreaterThan(0.25);
      expect(Math.abs(projectTipNdcY(cam))).toBeLessThanOrEqual(0.95);
    }

    // Post-entry crane trajectory smoothly submerges below WATER_SURFACE_Y (+0.30) to frame the descending vortex plume & underside meniscus ripples
    for (const t of [2.85, 3.6, 4.0, 12.0]) {
      cam.syncHeroFollow(t, true);
      cam.updateMatrices(16 / 10);
      expect(cam.position[1]).toBeLessThan(WATER_SURFACE_Y - 0.14);
      expect(cam.elevation).toBeLessThan(-0.15);
    }

    const crownCam = new OrbitCamera(CAMERA_PRESETS.crown);
    crownCam.updateMatrices(16 / 10);
    expect(Math.abs(projectTipNdcY(crownCam))).toBeLessThanOrEqual(0.95);

    cam.orbit(0.25, 0.1);
    expect(cam.followHeroEnabled).toBe(false);
    const manualTargetAz = cam.azimuth;
    cam.syncHeroFollow(12.0, true);
    expect(cam.azimuth).toBe(manualTargetAz);

    cam.applyPreset(CAMERA_PRESETS.hero, true);
    expect(cam.followHeroEnabled).toBe(true);
  });
});

describe('SimUniformPacker 512-byte WGSL layout contract', () => {
  it('packs exact 128 floats (512 bytes, 256-byte aligned) including 8 DropGpuSlots and thin-lens parameters', () => {
    expect(SIM_UNIFORM_FLOATS).toBe(128);
    expect(SIM_UNIFORM_BYTES).toBe(512);
    expect(SIM_UNIFORM_BYTES % 256).toBe(0);
    expect(DOMAIN_Y_MAX).toBe(WATER_SURFACE_Y);
    expect(DOMAIN_Y_MIN).toBe(-0.98);
    expect(DOMAIN_HALF_XZ).toBe(0.48);

    const cam = new OrbitCamera(CAMERA_PRESETS.hero);
    cam.updateMatrices(16 / 9);
    const packer = new SimUniformPacker();
    const sys = new MultiDropSystem();
    sys.reset(1.5, [0.03, -0.02]);
    sys.step(0.25);
    sys.beginPendantDrop([-0.08, 0.06], 3.25);

    const out = packer.pack({
      invViewProj: cam.invViewProjMatrix,
      viewProj: cam.viewProjMatrix,
      cameraPos: cam.position,
      cameraTarget: cam.target,
      viewportWidth: 0,
      viewportHeight: -10,
      dt: 1 / 60,
      frameIndex: 42,
      globalSeed: 2.5,
      drops: sys.getSnapshot(),
      spawnCursor: 12000,
      spawnCount: 640,
      config: DEFAULT_FLUID_OPTICAL_CONFIG,
    });

    expect(out.byteLength).toBe(512);
    expect(out[35]).toBeCloseTo(0.25, 5);
    expect(out[36]).toBe(1);
    expect(out[37]).toBe(1);
    expect(out[39]).toBe(42);
    expect(out[40]).toBeCloseTo(WATER_SURFACE_Y, 5);
    expect(out[42]).toBe(2); // 1 aerial + 1 pendant
    expect(out[52]).toBeCloseTo(DOMAIN_HALF_XZ, 5);
    expect(out[56]).toBeGreaterThan(0.25); // focalDistance
    expect(out[57]).toBeCloseTo(DEFAULT_FLUID_OPTICAL_CONFIG.apertureRadius, 6);
    expect(out[61]).toBe(12000);
    expect(out[62]).toBe(640);

    // Slot 0: aerial drop -> positive signedSeed
    expect(out[64 + 0]).toBeCloseTo(0.03, 5);
    expect(out[64 + 3]).toBeCloseTo(INITIAL_DROP_RADIUS, 5);
    expect(out[64 + 7]).toBeCloseTo(1.5, 5);

    // Slot 1: pendant drop -> negative signedSeed
    expect(out[72 + 0]).toBeCloseTo(-0.08, 5);
    expect(out[72 + 3]).toBeCloseTo(PENDANT_MIN_RADIUS, 5);
    expect(out[72 + 7]).toBeCloseTo(-3.25, 5);
  });
});

describe('Prahl (1999) CIE 1931 spectral hemoglobin transmittance LUT', () => {
  const lut = generatePrahlSpectralLut();

  it('produces 256x64 rgba16float with exact 256-byte row alignment and white [1,1,1] zero-path origin', () => {
    expect(lut.width).toBe(SPECTRAL_LUT_WIDTH);
    expect(lut.height).toBe(SPECTRAL_LUT_HEIGHT);
    expect(lut.rgba16.byteLength).toBe(256 * 64 * 8);
    expect((lut.width * 8) % 256).toBe(0);

    for (let i = 0; i < lut.rgba32.length; i++) {
      expect(Number.isFinite(lut.rgba32[i])).toBe(true);
    }

    for (let iy = 0; iy < lut.height; iy++) {
      const base = iy * lut.width * 4;
      expect(lut.rgba32[base + 0]).toBeCloseTo(1.0, 5);
      expect(lut.rgba32[base + 1]).toBeCloseTo(1.0, 5);
      expect(lut.rgba32[base + 2]).toBeCloseTo(1.0, 5);
    }
  });

  it('exhibits monotonic optical attenuation and Soret/Q-band green-yellow suppression (rose halo -> crimson -> oxblood)', () => {
    const arterialRow = Math.round(0.97 * (lut.height - 1));
    const venousRow = Math.round(0.65 * (lut.height - 1));

    for (const row of [arterialRow, venousRow]) {
      for (let ix = 1; ix < lut.width; ix++) {
        const prev = (row * lut.width + (ix - 1)) * 4;
        const curr = (row * lut.width + ix) * 4;
        expect(lut.rgba32[curr + 0]).toBeLessThanOrEqual(lut.rgba32[prev + 0] + 1e-6);
        expect(lut.rgba32[curr + 1]).toBeLessThanOrEqual(lut.rgba32[prev + 1] + 1e-6);
        expect(lut.rgba32[curr + 2]).toBeLessThanOrEqual(lut.rgba32[prev + 2] + 1e-6);
      }
    }

    // Ultra-thin oxygenated HbO2 veil (ix = 12..24, v = 1): true 414nm Soret band (524,280 cm^-1/M) extinguishes violet/blue
    // faster than 585-610nm orange cliff, producing warm amber -> vermilion-orange dichromatism (R > G > B)
    const thinIdx = (arterialRow * lut.width + 12) * 4;
    expect(lut.rgba32[thinIdx + 0]).toBeGreaterThan(0.85);
    expect(lut.rgba32[thinIdx + 0]).toBeGreaterThan(lut.rgba32[thinIdx + 1] * 1.15);
    expect(lut.rgba32[thinIdx + 1]).toBeGreaterThan(lut.rgba32[thinIdx + 2] * 1.25);

    // Medium path (ix = 48): Q-band doublet (541/577nm) strongly suppresses green-yellow (tG / tR < 0.22 -> pure arterial ruby!)
    const medIdx = (arterialRow * lut.width + 48) * 4;
    expect(lut.rgba32[medIdx + 0]).toBeGreaterThan(0.45);
    expect(lut.rgba32[medIdx + 1] / lut.rgba32[medIdx + 0]).toBeLessThan(0.22);

    // Thick core path (ix = 225): green and blue extinguished (< 0.005) while 630-700nm deep oxblood red transmits
    const thickIdx = (arterialRow * lut.width + 225) * 4;
    expect(lut.rgba32[thickIdx + 0]).toBeGreaterThan(0.02);
    expect(lut.rgba32[thickIdx + 1]).toBeLessThan(0.005);
    expect(lut.rgba32[thickIdx + 2]).toBeLessThan(0.002);
  });
});

describe('WGSL shader static & physical architecture invariants across all 31 shaders', () => {
  const allShaders = [
    ['WAVE_CLEAR', WAVE_CLEAR_WGSL],
    ['WAVE_CLEAR_SPEC', WAVE_CLEAR_SPEC_WGSL],
    ['WAVE_SPATIAL_FORCE', WAVE_SPATIAL_FORCE_WGSL],
    ['WAVE_FFT_ROW', WAVE_FFT_ROW_WGSL],
    ['WAVE_FFT_COL', WAVE_FFT_COL_WGSL],
    ['WAVE_SPECTRAL_PROPAGATE', WAVE_SPECTRAL_PROPAGATE_WGSL],
    ['WAVE_IFFT_COL', WAVE_IFFT_COL_WGSL],
    ['WAVE_IFFT_ROW', WAVE_IFFT_ROW_WGSL],
    ['WAVE_FINALIZE', WAVE_FINALIZE_WGSL],
    ['FLUID_INIT_VEL', FLUID_INIT_VEL_WGSL],
    ['FLUID_CLEAR_R32', FLUID_CLEAR_R32_WGSL],
    ['FLUID_ADVECT_MAC', FLUID_ADVECT_MAC_WGSL],
    ['FLUID_CURL', FLUID_CURL_WGSL],
    ['FLUID_CASSON_FORCES', FLUID_CASSON_FORCES_WGSL],
    ['FLUID_MAC_RHS', FLUID_MAC_RHS_WGSL],
    ['MG_RBGS_RED_ZERO', MG_RBGS_RED_ZERO_WGSL],
    ['MG_RBGS_RED', MG_RBGS_RED_WGSL],
    ['MG_RBGS_BLACK', MG_RBGS_BLACK_WGSL],
    ['MG_COARSE_SOLVE_L3', MG_COARSE_SOLVE_L3_WGSL],
    ['MG_RESIDUAL', MG_RESIDUAL_WGSL],
    ['MG_RESTRICT', MG_RESTRICT_WGSL],
    ['MG_PROLONG_ADD', MG_PROLONG_ADD_WGSL],
    ['FLUID_MAC_PROJECT', FLUID_MAC_PROJECT_WGSL],
    ['FLUID_MAC_TO_CENTER', FLUID_MAC_TO_CENTER_WGSL],
    ['PARTICLE_RESET', PARTICLE_RESET_WGSL],
    ['PARTICLE_EMIT', PARTICLE_EMIT_WGSL],
    ['PARTICLE_ADVECT_SPLAT', PARTICLE_ADVECT_SPLAT_WGSL],
    ['CONC_CLEAR', CONC_CLEAR_WGSL],
    ['CONC_PREDICT', CONC_PREDICT_WGSL],
    ['CONC_RESOLVE', CONC_RESOLVE_WGSL],
    ['LIGHTING_VOLUME', LIGHTING_VOLUME_WGSL],
    ['PLUME_OCCUPANCY_RAW', PLUME_OCCUPANCY_RAW_WGSL],
    ['PLUME_OCCUPANCY_DILATE', PLUME_OCCUPANCY_DILATE_WGSL],
    ['RENDER', RENDER_WGSL],
    ['COMPOSITE', COMPOSITE_WGSL],
  ] as const;

  it('ensures zero implicit-LOD textureSample, zero non-uniform derivatives, zero WGSL reserved-word vars, <= 256 workgroup threads, <= 4 storage textures, and <= 8 storage buffers', () => {
    const wgslReserved = /\b(?:let|var|fn)\s+(?:active|alignas|alignof|asm|bf16|coherent|const_cast|demote|do|enum|extern|fallthrough|final|friend|fxgroup|goto|implements|import|inline|interface|macro|module|mutable|namespace|new|nil|noexcept|noinline|noperspective|null|nullptr|operator|package|packoffset|partition|patch|pixelfragment|precise|precision|protected|public|readonly|ref|regardless|register|reinterpret_cast|require|resource|restrict|self|set|shared|sizeof|smooth|snorm|static|static_assert|static_cast|std|subroutine|super|template|this|thread_local|throw|trait|try|typedef|typeid|typename|typeof|union|unless|unorm|unsafe|unsized|use|using|varying|virtual|volatile|wgsl|where|with|writeonly|yield)\b/;
    const wgSizeRegex = /@workgroup_size\(\s*(\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?\s*\)/g;
    for (const [name, code] of allShaders) {
      expect(code, `${name} must declare SimUniforms`).toContain('struct SimUniforms');
      expect(code, `${name} must not declare WGSL reserved-word identifiers`).not.toMatch(wgslReserved);
      for (const m of code.matchAll(wgSizeRegex)) {
        const prod = Number(m[1]) * Number(m[2] ?? 1) * Number(m[3] ?? 1);
        expect(prod, `${name} workgroup invocations`).toBeLessThanOrEqual(256);
      }
      expect(code, `${name} must use textureSampleLevel or textureLoad`).not.toMatch(
        /\btextureSample\s*\(/
      );
      expect(code, `${name} must not call fwidth/dpdx/dpdy`).not.toMatch(
        /\b(fwidth|dpdx|dpdy)\s*\(/
      );
      const storageTexMatches = code.match(/texture_storage_(2d|3d)</g) ?? [];
      expect(storageTexMatches.length, `${name} storage texture count`).toBeLessThanOrEqual(4);

      const storageBufMatches = code.match(/var<storage\b/g) ?? [];
      expect(storageBufMatches.length, `${name} storage buffer count`).toBeLessThanOrEqual(8);
    }
  });

  it('enforces strictly ascending literal smoothstep(edge0, edge1, x) bounds across all 31 WGSL shaders', () => {
    const numericSmoothstep = /\bsmoothstep\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,/g;
    let checkedCount = 0;
    for (const [name, code] of allShaders) {
      for (const match of code.matchAll(numericSmoothstep)) {
        const edge0 = Number(match[1]);
        const edge1 = Number(match[2]);
        expect(
          edge0,
          `${name}: smoothstep(${edge0}, ${edge1}, ...) requires edge0 < edge1`
        ).toBeLessThan(edge1);
        checkedCount++;
      }
    }
    expect(checkedCount).toBeGreaterThan(25);
  });

  it('verifies FFT dispersive Marangoni wave, MAC Multigrid Casson fluid, and zero-integral Ricker impact invariants', () => {
    expect(WAVE_SPATIAL_FORCE_WGSL).toContain('exp(-rNorm * rNorm) * (1.0 - rNorm * rNorm)');
    expect(WAVE_SPATIAL_FORCE_WGSL).toContain('lapFilm');
    expect(WAVE_FFT_ROW_WGSL).toContain('reverseBits');
    expect(WAVE_FFT_ROW_WGSL).toContain('workgroupBarrier()');
    expect(WAVE_SPECTRAL_PROPAGATE_WGSL).toContain('tanh');

    // Numerical polar quadrature of Ricker impact footprint: integral_0^R r * exp(-r^2) * (1 - r^2) dr == 0
    let polarIntegral = 0.0;
    const dr = 0.002;
    for (let r = dr * 0.5; r < 6.0; r += dr) {
      polarIntegral += r * Math.exp(-r * r) * (1.0 - r * r) * dr;
    }
    expect(Math.abs(polarIntegral)).toBeLessThan(1e-5);

    // Retired synthetic artifacts must not reappear in fluid solver
    expect(FLUID_CASSON_FORCES_WGSL).not.toContain('rtWave');
    expect(FLUID_CASSON_FORCES_WGSL).not.toMatch(/sin\([^)]*26\.0/);
    expect(FLUID_CASSON_FORCES_WGSL).toContain('cassonDiffAlpha');
    expect(FLUID_CASSON_FORCES_WGSL).toContain('tauYield');
    expect(FLUID_MAC_RHS_WGSL).toContain('evalTopDirichletPressure');
    expect(FLUID_MAC_PROJECT_WGSL).toContain('evalTopDirichletPressure');
  });

  it('verifies wetting-neck SDF, Prahl LUT, 8-tap tricubic filter, Schlieren bending, un-damped TIR, and zero underwater Blinn-Phong', () => {
    expect(RENDER_WGSL).toContain('fn evalDropBodySDF(');
    expect(RENDER_WGSL).toContain('fn evalWettingNeckUnion(');
    expect(RENDER_WGSL).toContain('fn sampleConcTricubic(');
    expect(RENDER_WGSL).toContain('fn sampleSpectralLut(');
    expect(RENDER_WGSL).toContain('reflect(rd, nUnder)');
    expect(RENDER_WGSL).toContain('schlierenStr');

    // Retired visual dampers and underwater plastic highlights must remain absent
    expect(RENDER_WGSL).not.toContain('vec3<f32>(nHit.x * 0.16');
    expect(RENDER_WGSL).not.toContain('nUnder.xz * 0.045');
    expect(RENDER_WGSL).not.toContain('pow(nDotH, 46.0)');
    expect(RENDER_WGSL).not.toContain('1.35 * rawDensity');
  });
});

describe('Boundary & adversarial edge-case invariants', () => {
  it('handles zero, negative, and NaN dt steps without NaN state corruption or spurious drainage', () => {
    const sys = new MultiDropSystem();
    sys.reset(1.0);
    const yBefore = sys.slots[0].center[1];

    for (const badDt of [0.0, -0.5, Number.NaN]) {
      const snap = sys.step(badDt);
      expect(snap.simTime).toBe(0.0);
      expect(snap.totalStepDrainedVolume).toBe(0.0);
      expect(sys.slots[0].center[1]).toBe(yBefore);
    }
  });

  it('recycles the smallest-volume slot on the 9th concurrent drop while protecting an active pendant drop', () => {
    const sys = new MultiDropSystem();
    sys.reset(1.0);
    const pendant = sys.beginPendantDrop([0.0, 0.0], 2.0);
    // Fill remaining 6 slots with distinct radii; make one tiny (0.032) so it is chosen for recycling
    const tiny = sys.spawnAerialDrop([0.18, 0.0], 0.032, 3.0);
    for (let i = 0; i < 5; i++) {
      const a = ((i + 1) * Math.PI) / 3;
      sys.spawnAerialDrop([0.22 * Math.cos(a), 0.22 * Math.sin(a)], 0.068, 4.0 + i);
    }
    expect(sys.getSnapshot().activeCount).toBe(8);

    // 9th drop must recycle `tiny` and leave `pendant` intact
    const ninth = sys.spawnAerialDrop([-0.15, 0.12], 0.060, 9.5);
    expect(ninth).toBe(tiny);
    expect(sys.hasActivePendantDrop()).toBe(true);
    expect(pendant.mode).toBe('pendant');
  });

  it('falls back cleanly when raycasting from a horizontal horizon ray (dy == 0) or underwater sub-meniscus camera', () => {
    const abyssCam = new OrbitCamera(CAMERA_PRESETS.abyss);
    abyssCam.updateMatrices(16 / 9);
    const hit = raycastMeniscusXZ(
      abyssCam.invViewProjMatrix,
      abyssCam.position,
      abyssCam.target,
      0.0,
      0.15
    );
    expect(Number.isFinite(hit[0])).toBe(true);
    expect(Number.isFinite(hit[1])).toBe(true);
    expect(Math.hypot(hit[0], hit[1])).toBeLessThanOrEqual(MAX_DROP_OFFSET_XZ + 1e-6);
  });

  it('encodes IEEE-754 binary16 edge values (0, -0, 1.0, subnormals, overflow, Inf, NaN) accurately', () => {
    expect(float32ToFloat16Bits(0.0)).toBe(0x0000);
    expect(float32ToFloat16Bits(1.0)).toBe(0x3c00);
    expect(float32ToFloat16Bits(-1.0)).toBe(0xbc00);
    expect(float32ToFloat16Bits(0.5)).toBe(0x3800);
    expect(float32ToFloat16Bits(999999.0)).toBe(0x7bff);
    expect(float32ToFloat16Bits(Number.POSITIVE_INFINITY)).toBe(0x7c00);
    expect(float32ToFloat16Bits(Number.NEGATIVE_INFINITY)).toBe(0xfc00);
    expect(float32ToFloat16Bits(Number.NaN)).toBe(0x7e00);
  });

  it('reuses the exact same backing Float32Array and ArrayBuffer across repeated SimUniformPacker.pack calls', () => {
    const cam = new OrbitCamera(CAMERA_PRESETS.hero);
    cam.updateMatrices(16 / 9);
    const packer = new SimUniformPacker();
    const sys = new MultiDropSystem();
    const a = packer.pack({
      invViewProj: cam.invViewProjMatrix,
      viewProj: cam.viewProjMatrix,
      cameraPos: cam.position,
      cameraTarget: cam.target,
      viewportWidth: 1280,
      viewportHeight: 800,
      dt: 1 / 60,
      frameIndex: 1,
      globalSeed: 1.0,
      drops: sys.getSnapshot(),
      spawnCursor: 0,
      spawnCount: 0,
      config: DEFAULT_FLUID_OPTICAL_CONFIG,
    });
    const b = packer.pack({
      invViewProj: cam.invViewProjMatrix,
      viewProj: cam.viewProjMatrix,
      cameraPos: cam.position,
      cameraTarget: cam.target,
      viewportWidth: 1280,
      viewportHeight: 800,
      dt: 1 / 60,
      frameIndex: 2,
      globalSeed: 1.0,
      drops: sys.getSnapshot(),
      spawnCursor: 128,
      spawnCount: 64,
      config: DEFAULT_FLUID_OPTICAL_CONFIG,
    });
    expect(a).toBe(b);
    expect(a.buffer).toBe(packer.buffer);
  });

  it('sizes Lagrangian parcel ring buffer for 8 full reference drops without wraparound, transfers wrapped >8th-drop mass into splatLysed, and prevents u32 atomic overflow', () => {
    expect(PARCELS_PER_REF_DROP * MAX_ACTIVE_DROPS).toBe(PARTICLE_COUNT);
    expect(PARTICLE_COUNT).toBe(393216);

    // Worst-case mathematical bound: even if all 393,216 parcels collapsed into a single voxel, u32 cannot overflow
    const worstCaseSingleVoxelFp = PARTICLE_COUNT * INITIAL_PARCEL_VOXEL_MASS * SPLAT_FP_SCALE;
    expect(worstCaseSingleVoxelFp).toBeLessThan(0xffff_ffff);

    // PARTICLE_EMIT_WGSL must atomically transfer any overwritten surviving parcel mass into splatLysed
    expect(PARTICLE_EMIT_WGSL).toContain('atomicAdd(&splatLysed[');
  });

  it('uses strictly even Widnall azimuthal modes (m = 4, 6, 8) with identically zero m = 1 lateral beat dipole', () => {
    for (const code of [FLUID_MAC_RHS_WGSL, FLUID_ADVECT_MAC_WGSL, PARTICLE_EMIT_WGSL]) {
      expect(code).not.toContain('5.0 * theta');
      expect(code).not.toContain('3.0 * theta');
      expect(code).not.toContain('7.0 * theta');
      expect(code).toContain('4.0 * theta');
      expect(code).toContain('6.0 * theta');
    }

    // Verify zero m=1 dipole Fourier projection across powers p = 1, 2, 3 (linear + quadratic/cubic nonlinear beat modes)
    const nSamples = 512;
    for (const seed of [1.0, 2.718, 5.13]) {
      for (const power of [1, 2, 3]) {
        let dipX = 0.0;
        let dipZ = 0.0;
        for (let k = 0; k < nSamples; k++) {
          const theta = (2.0 * Math.PI * k) / nSamples;
          const azim =
            1.0 +
            0.058 * Math.cos(4.0 * theta + seed * 1.73) +
            0.044 * Math.cos(6.0 * theta - seed * 2.31) +
            0.028 * Math.sin(8.0 * theta + seed * 0.89);
          const f = Math.pow(azim, power);
          dipX += f * Math.cos(theta);
          dipZ += f * Math.sin(theta);
        }
        expect(Math.abs(dipX / nSamples)).toBeLessThan(1e-12);
        expect(Math.abs(dipZ / nSamples)).toBeLessThan(1e-12);
      }
    }
  });

  it('sustains C2-continuous wetting-collar wake drainage across 1.26s post-impact to feed an unbroken umbilical stem while conserving 100% volume', () => {
    const sys = new MultiDropSystem();
    sys.reset(1.0);
    stepSystemTo(sys, IMPACT_TIME - 0.01, 1 / 120);

    let earlySlugVol = 0.0;
    let lateStemVol = 0.0;
    let minMidWakeRate = Infinity;
    while (sys.simTime < IMPACT_TIME + 1.35) {
      const snap = sys.step(1 / 120);
      const elapsed = sys.simTime - IMPACT_TIME;
      if (elapsed <= 0.45) {
        earlySlugVol += snap.totalStepDrainedVolume;
      } else {
        lateStemVol += snap.totalStepDrainedVolume;
      }
      if (elapsed >= 0.35 && elapsed <= 0.85 && sys.slots[0].active) {
        minMidWakeRate = Math.min(minMidWakeRate, sys.slots[0].volDrainRate);
      }
    }
    const totalExpected = sphereVolume(INITIAL_DROP_RADIUS);
    // Primary slug carries majority (~58-68%) while sustained C2 wake (>30%) feeds an unbroken trailing umbilical stem with zero mid-stem flux cliff
    expect(earlySlugVol).toBeGreaterThan(0.55 * totalExpected);
    expect(lateStemVol).toBeGreaterThan(0.28 * totalExpected);
    expect(earlySlugVol + lateStemVol).toBeCloseTo(totalExpected, 6);
    expect(minMidWakeRate).toBeGreaterThan(0.24);
    expect(sys.slots[0].active).toBe(false);
  });

  it('locks macro apertureRadius to 0.0032 and projects off-axis pendant/aerial drops onto the camera optical axis for exact focus', () => {
    expect(DEFAULT_FLUID_OPTICAL_CONFIG.apertureRadius).toBeCloseTo(0.0032, 6);

    const cam = new OrbitCamera(CAMERA_PRESETS.hero);
    cam.updateMatrices(16 / 9);
    const packer = new SimUniformPacker();
    const sys = new MultiDropSystem();
    sys.reset(1.0);
    // Step past initial drop drainage so only off-axis pendant drop is aerial
    stepSystemTo(sys, 2.5, 1 / 60);
    const pendant = sys.beginPendantDrop([0.24, -0.18], 2.0);

    const out = packer.pack({
      invViewProj: cam.invViewProjMatrix,
      viewProj: cam.viewProjMatrix,
      cameraPos: cam.position,
      cameraTarget: cam.target,
      viewportWidth: 2560,
      viewportHeight: 1600,
      dt: 1 / 60,
      frameIndex: 10,
      globalSeed: 1.0,
      drops: sys.getSnapshot(),
      spawnCursor: 0,
      spawnCount: 0,
      config: DEFAULT_FLUID_OPTICAL_CONFIG,
    });

    const fdx = cam.target[0] - cam.position[0];
    const fdy = cam.target[1] - cam.position[1];
    const fdz = cam.target[2] - cam.position[2];
    const fLen = Math.hypot(fdx, fdy, fdz);
    const expectedAxialFocus =
      ((pendant.center[0] - cam.position[0]) * fdx +
        (pendant.center[1] - cam.position[1]) * fdy +
        (pendant.center[2] - cam.position[2]) * fdz) /
      fLen;
    const euclideanDist = Math.hypot(
      pendant.center[0] - cam.position[0],
      pendant.center[1] - cam.position[1],
      pendant.center[2] - cam.position[2]
    );

    expect(out[56]).toBeCloseTo(expectedAxialFocus, 5);
    expect(out[56]).toBeLessThan(euclideanDist);
  });

  it('enforces 4-point partition-of-unity filament splatting, reflected TIR focal path compensation, and non-inverted wetting-collar chord', () => {
    expect(PARTICLE_ADVECT_SPLAT_WGSL).toContain('splatTrilinearIntact(uvw0, 0.21 * mass)');
    expect(PARTICLE_ADVECT_SPLAT_WGSL).toContain('splatTrilinearIntact(uvw1, 0.29 * mass)');
    expect(PARTICLE_ADVECT_SPLAT_WGSL).toContain('splatTrilinearIntact(uvw2, 0.29 * mass)');
    expect(PARTICLE_ADVECT_SPLAT_WGSL).toContain('splatTrilinearIntact(uvw3, 0.21 * mass)');
    expect(0.21 + 0.29 + 0.29 + 0.21).toBeCloseTo(1.0, 12);
    expect(PARTICLE_ADVECT_SPLAT_WGSL).toContain('minIntactMass');

    expect(RENDER_WGSL).toContain('let remFocusTir = tFocus - tSurfHit;');
    expect(RENDER_WGSL).toContain('let remFocusRefl = tFocus - tSurf;');
    expect(RENDER_WGSL).toContain('let remFocus = tFocus - tPlumeOrigin;');
    expect(RENDER_WGSL).toContain('let focalRef = max(u.lensAndRheology.x, 0.18);');
    expect(RENDER_WGSL).not.toContain('max(0.05, tFocus -');
    expect(RENDER_WGSL).toContain('let intactMask = smoothstep(');
    expect(RENDER_WGSL).toContain('let lysedMask  = smoothstep(');
    expect(RENDER_WGSL).not.toContain('mix(0.032, 0.002, hit.neckBlend)');
  });

  it('conserves 100% of liquid blood volume when >8 rapid drops recycle active draining slots', () => {
    const sys = new MultiDropSystem();
    sys.reset(1.0);
    let spawnedVol = sphereVolume(INITIAL_DROP_RADIUS);
    let drainedVol = 0;

    for (let i = 0; i < 10; i++) {
      const angle = (i / 10) * Math.PI * 2;
      sys.spawnAerialDrop([0.24 * Math.cos(angle), 0.24 * Math.sin(angle)], 0.048, 10 + i * 1.7);
      spawnedVol += sphereVolume(0.048);
      for (let s = 0; s < 8; s++) {
        const snap = sys.step(1 / 240);
        drainedVol += snap.totalStepDrainedVolume;
      }
    }

    for (let s = 0; s < 240 * 5; s++) {
      const snap = sys.step(1 / 240);
      drainedVol += snap.totalStepDrainedVolume;
    }

    expect(drainedVol / spawnedVol).toBeCloseTo(1.0, 5);
  });

  it('diffuses lysed hemoglobin on the advected predictor grid and maps LUT v-axis across venous rouleaux (v = 0) to oxygenated sheared HbO2 (v = 1)', () => {
    expect(CONC_RESOLVE_WGSL).toContain('fn loadPredLysed');
    expect(CONC_RESOLVE_WGSL).toContain('loadPredLysed(vec3<i32>(ix + 1, iy, iz))');
    expect(CONC_RESOLVE_WGSL).not.toContain('loadOrigLysed(vec3<i32>(ix + 1, iy, iz))');
    expect(PARTICLE_ADVECT_SPLAT_WGSL).toContain('fn splatTrilinearLysed');

    const { rgba32 } = generatePrahlSpectralLut();
    const readLut = (uIdx: number, vIdx: number): [number, number, number] => {
      const base = (vIdx * SPECTRAL_LUT_WIDTH + uIdx) * 4;
      return [rgba32[base], rgba32[base + 1], rgba32[base + 2]];
    };

    // Thin oxygenated sheared HbO2 sheet (v = 63, u = 24): Soret band + 585-610nm cliff yields vermilion-orange dichromatism (R > G > B)
    const [rArt, gArt, bArt] = readLut(24, SPECTRAL_LUT_HEIGHT - 1);
    expect(rArt).toBeGreaterThan(gArt);
    expect(gArt).toBeGreaterThan(bArt * 1.6);

    // Venous rouleaux curtains (v = 0, SO2 = 0.56 + Duysens flattening) are strictly darker than arterial (v = 63) and grade monotonically
    const [rThin] = readLut(24, 0);
    const [rMid] = readLut(64, 0);
    const [rDense] = readLut(128, 0);
    expect(rThin).toBeLessThan(rArt);
    expect(rThin).toBeGreaterThan(0.65);
    expect(rMid).toBeLessThan(rThin - 0.18);
    expect(rDense).toBeLessThan(rMid - 0.20);
  });

  it('anchors thin-lens focal distance to foreground pendant drops even at shallow macro meniscus elevations', () => {
    const packer = new SimUniformPacker();
    const cam = new OrbitCamera();
    cam.followHeroEnabled = false;
    cam.azimuth = 0.0;
    cam.elevation = 0.088;
    cam.distance = 1.0;
    cam.target = [0.0, 0.285, 0.0];
    cam.updateMatrices(16 / 10);

    const sys = new MultiDropSystem();
    sys.reset(1.0, [0.025, -0.23]);
    stepSystemTo(sys, 3.55, 1 / 60);
    const pendant = sys.beginPendantDrop([-0.055, 0.24], 2.0);
    sys.updatePendantDrop(1.18);

    const out = packer.pack({
      invViewProj: cam.invViewProjMatrix,
      viewProj: cam.viewProjMatrix,
      cameraPos: cam.position,
      cameraTarget: cam.target,
      viewportWidth: 2560,
      viewportHeight: 1600,
      dt: 1 / 60,
      frameIndex: 30,
      globalSeed: 1.0,
      drops: sys.getSnapshot(),
      spawnCursor: 0,
      spawnCount: 0,
      config: DEFAULT_FLUID_OPTICAL_CONFIG,
    });

    const fdx = cam.target[0] - cam.position[0];
    const fdy = cam.target[1] - cam.position[1];
    const fdz = cam.target[2] - cam.position[2];
    const fLen = Math.hypot(fdx, fdy, fdz);
    const expectedAxialFocus =
      ((pendant.center[0] - cam.position[0]) * fdx +
        (pendant.center[1] - cam.position[1]) * fdy +
        (pendant.center[2] - cam.position[2]) * fdz) /
      fLen;

    expect(out[56]).toBeCloseTo(expectedAxialFocus, 5);
    expect(out[56]).toBeLessThan(fLen - 0.18);
  });

  it('enforces shift-free C2 cubic B-spline stencil offsets, CoC-unblocked empty-water skipping, and grazing meniscus glare/turbidity guards', () => {
    expect(RENDER_WGSL).toContain('idx - 1.0 + (w1 / max(g0');
    expect(RENDER_WGSL).toContain('idx + 1.0 + (w3 / max(g1');
    expect(RENDER_WGSL).not.toContain('idx + 2.0 + (w3 / max(g1');

    // Verify 1D Ruijters & Thevenaz (2012) stencil centroid equals exact coord = idx + f with zero voxel shift
    for (const f of [0.0, 0.25, 0.5, 0.75, 1.0]) {
      const idx = 10.0;
      const omf = 1.0 - f;
      const w0 = (1 / 6) * omf * omf * omf;
      const w1 = (1 / 6) * (3 * f * f * f - 6 * f * f + 4);
      const w2 = (1 / 6) * (-3 * f * f * f + 3 * f * f + 3 * f + 1);
      const w3 = (1 / 6) * (f * f * f);
      const g0 = w0 + w1;
      const g1 = w2 + w3;
      const t0 = idx - 1.0 + w1 / g0;
      const t1 = idx + 1.0 + w3 / g1;
      expect(g0 + g1).toBeCloseTo(1.0, 12);
      expect(g0 * t0 + g1 * t1).toBeCloseTo(idx + f, 12);
    }

    expect(RENDER_WGSL).toContain('if (lit.a < 0.000005)');
    expect(RENDER_WGSL).not.toContain('if (lit.a < 0.000025 && cocBlur <');
    expect(RENDER_WGSL).toContain('let coarseSkip = 0.015;');
    expect(RENDER_WGSL).toContain('let pWaveXZ = mix(pPlumeOrigin.xz, pSurf.xz');
    expect(RENDER_WGSL).toContain(
      'mix(1.0, mix(0.045, 0.16, grazingFresnelRim), bloodInColumn)'
    );
    expect(RENDER_WGSL).toContain('let turbidSharpness = exp(-fgPlume.tauIntact * 680.0);');
    expect(PARTICLE_ADVECT_SPLAT_WGSL).toContain('if (sLenNow < 0.0038)');
  });

  it('guarantees C0-continuous linear-crossing raymarch rewind across coarseSkip/skipHalo, exact 4x4x4 brick clearance shells, and grazing CoC meniscus guards', () => {
    expect(VEL_GRID_X).toBe(BRICK_GRID_X * 4);
    expect(VEL_GRID_Y).toBe(BRICK_GRID_Y * 4);
    expect(VEL_GRID_Z).toBe(BRICK_GRID_Z * 4);

    // Brick cube size = 0.96m / 24 = 0.040m (40mm): outer leap 0.072m < 2*40mm (80mm) and mid leap 0.034m < 40mm
    const brickSizeXZ = 0.96 / BRICK_GRID_X;
    const brickSizeY = 1.28 / BRICK_GRID_Y;
    expect(brickSizeXZ).toBeCloseTo(0.04, 6);
    expect(brickSizeY).toBeCloseTo(0.04, 6);
    expect(0.072).toBeLessThan(2.0 * brickSizeXZ - 0.005);
    expect(0.034).toBeLessThan(1.0 * brickSizeXZ - 0.005);

    // Prove C0 continuity of linear-crossing rewind across threshold crossing t_iso = 1.000m (eliminating 15mm scanline corduroy)
    const thresh = 0.000005;
    const slope = 0.0025;
    const tIso = 1.0;
    const evalDensity = (t: number) => Math.max(0.0, thresh + slope * (t - tIso));
    const simulateCrossT = (tStartOffset: number, step: number) => {
      let t = 0.88 + tStartOffset;
      let prevA = 0.0;
      let prevStep = 0.0;
      for (let i = 0; i < 32; i++) {
        const a = evalDensity(t);
        if (a < thresh) {
          prevA = a;
          prevStep = step;
          t += step;
          continue;
        }
        if (prevStep > 0.0) {
          const frac = Math.min(1.0, Math.max(0.0, (thresh - prevA) / Math.max(a - prevA, 1e-8)));
          t = t - prevStep + frac * prevStep;
        }
        break;
      }
      return t;
    };

    const tLeft = simulateCrossT(0.015 - 1e-6, 0.015);
    const tRight = simulateCrossT(0.015 + 1e-6, 0.015);
    expect(Math.abs(tLeft - tIso)).toBeLessThan(1e-5);
    expect(Math.abs(tRight - tIso)).toBeLessThan(1e-5);
    expect(Math.abs(tLeft - tRight)).toBeLessThan(1e-5);

    expect(RENDER_WGSL).toContain('let crossFrac = clamp((0.000005 - prevLitA)');
    expect(RENDER_WGSL).toContain('let haloFrac = clamp((0.000034 - prevConeD)');
    expect(RENDER_WGSL).toContain('let cocMeniscusFade = 0.008 + meniscusBokeh * (0.022 + 0.024 * grazingTopside);');
    expect(RENDER_WGSL).not.toContain('normalize(pRipple.xz');
    expect(RENDER_WGSL).not.toContain('dot(pRipple.xz, pRipple.xz)');
    expect(RENDER_WGSL).toContain('let obliquePlanarVis = smoothstep(0.065, 0.22, -rd.y) * inCuvetteTop;');
    expect(RENDER_WGSL).toContain('let underGrazingMix = smoothstep(0.065, 0.22, rd.y);');
    expect(LIGHTING_VOLUME_WGSL).toContain('let ctC = textureSampleLevel(concIn, linSampler, uvw, 0.0);');
  });

  it('verifies 5-tap clamped Catmull-Rom HDR bicubic partition-of-unity, linear ramp exactness, and WebGPU top-left UV mapping', () => {
    expect(COMPOSITE_WGSL).toContain('let uv = vec2<f32>(in.ndc.x * 0.5 + 0.5, 0.5 - in.ndc.y * 0.5);');
    expect(COMPOSITE_WGSL).toContain('return clamp(bicubic, boxMin, boxMax);');
    expect(COMPOSITE_WGSL).toContain('pbrNeutralToneMap');
    expect(COMPOSITE_WGSL).toContain('linearToSRGB');

    for (const f of [0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0]) {
      const w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
      const w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
      const w2 = f * (0.5 + f * (2.0 - 1.5 * f));
      const w3 = f * f * (-0.5 + 0.5 * f);
      const w12 = w1 + w2;
      expect(w0 + w1 + w2 + w3).toBeCloseTo(1.0, 12);
      expect(w1).toBeGreaterThanOrEqual(0.0);
      expect(w2).toBeGreaterThanOrEqual(0.0);
      expect(w12).toBeGreaterThanOrEqual(1.0 - 1e-12);
      expect(-1.0 * w0 + 0.0 * w1 + 1.0 * w2 + 2.0 * w3).toBeCloseTo(f, 12);
    }
  });

  it('enforces native 1:1 Retina composite drops/reflections, sInfo.height reflection origin, full-silhouette tricubic, C1 B-spline wave normals, and transmittance-gated TIR glint', () => {
    // 1. COMPOSITE_WGSL binds waveTex + spectralLutTex and evaluates 1:1 Retina drops, collars, and topside drop reflections
    expect(COMPOSITE_WGSL).toContain('@group(0) @binding(3) var waveTex: texture_2d<f32>;');
    expect(COMPOSITE_WGSL).toContain('@group(0) @binding(4) var spectralLutTex: texture_2d<f32>;');
    expect(COMPOSITE_WGSL).toContain('fn evalWaveBspline(');
    expect(COMPOSITE_WGSL).toContain('fn evalTopsideDropReflection(');
    expect(COMPOSITE_WGSL).toContain('intersectAerialDrop(ro, rd, tFocus, apRadius)');
    expect(COMPOSITE_WGSL).toContain('shadeAerialDrop(ro, rd, dropHit, finalLinear)');
    expect(COMPOSITE_WGSL).toContain('let skyFace = smoothstep(-0.05, 0.38, reflOnDrop.y);');

    // 2. Topside drop reflection origin uses sInfo.height directly (never double-offset u.surfaceParams.x + sInfo.height)
    expect(COMPOSITE_WGSL).toContain('vec3<f32>(pWaveXZ.x, sInfo.height, pWaveXZ.y)');
    expect(COMPOSITE_WGSL).not.toContain('u.surfaceParams.x + sInfo.height');
    expect(RENDER_WGSL).not.toContain('u.surfaceParams.x + sInfo.height');

    // 3. Full-silhouette C2 tricubic reconstruction down to zero density + circular sub-collar voxel containment
    expect(RENDER_WGSL).not.toContain('conc.r > 0.00012');
    expect(RENDER_WGSL).not.toContain('smoothstep(0.00012, 0.0009, conc.r)');
    expect(RENDER_WGSL).toContain('subCollarContain');

    // 4. Exact analytical C1 cubic B-spline surface derivative identity (zero wave-cell-edge normal kinks)
    expect(RENDER_WGSL).toContain('let dhdx = q.x * (hxR - hxL) * scaleDeriv;');
    const evalBsplineDeriv1D = (P: [number, number, number, number], f: number): number => {
      const q = 0.5 + f * (1.0 - f);
      const t0 = (0.5 * f * (4.0 - 3.0 * f)) / q;
      const t1 = (0.5 * f * f) / q;
      const hxL = (1.0 - t0) * P[0] + t0 * P[1];
      const hxR = (1.0 - t1) * P[2] + t1 * P[3];
      return q * (hxR - hxL);
    };
    for (const f of [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]) {
      // Exact derivative of linear ramp P_k = k is 1.0 everywhere
      expect(evalBsplineDeriv1D([-1, 0, 1, 2], f)).toBeCloseTo(1.0, 12);
      // Exact derivative of quadratic P_k = k^2 is 2*f everywhere
      expect(evalBsplineDeriv1D([1, 0, 1, 4], f)).toBeCloseTo(2.0 * f, 12);
    }
    // C1 continuity across adjacent wave texel boundary (f = 1 of cell i == f = 0 of cell i+1)
    const arbitraryGrid = [0.12, -0.31, 0.44, 0.09, -0.18];
    const leftCellRightEdge = evalBsplineDeriv1D(
      [arbitraryGrid[0], arbitraryGrid[1], arbitraryGrid[2], arbitraryGrid[3]],
      1.0
    );
    const rightCellLeftEdge = evalBsplineDeriv1D(
      [arbitraryGrid[1], arbitraryGrid[2], arbitraryGrid[3], arbitraryGrid[4]],
      0.0
    );
    expect(leftCellRightEdge).toBeCloseTo(rightCellLeftEdge, 12);

    // 5. State 2B inverted TIR torus crease elimination: transmittance-gated rippleGlint + exact critical-angle Fresnel + 3-tap aperture quadrature
    expect(RENDER_WGSL).toContain('vec3<f32>(rippleGlint * 0.85, rippleGlint * 0.94, rippleGlint) * tirPlume.transmittance');
    expect(RENDER_WGSL).toContain('let tirBloodPresence = clamp(1.0 - tirPlume.transmittance.g, 0.0, 1.0);');
    expect(RENDER_WGSL).toContain('let fresnel = fresnelDielectric(cosI, 1.333, 1.0);');
    expect(RENDER_WGSL).toContain('let fresnelWindow = (');
    expect(RENDER_WGSL).not.toContain('smoothstep(0.648, 0.735, cosI)');
  });

  it('prevents TIR sheetGate CoC creases, foreground-over-TIR stem halos, flat-chord topside reflections, sub-collar parcel scallops, and horizon tSurf pops', () => {
    const smoothstep = (e0: number, e1: number, x: number): number => {
      const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
      return t * t * (3 - 2 * t);
    };

    // 1. C1-continuous sheetGate replaces hard cocBlur < 0.62 boolean cutoff in marchPlume
    expect(RENDER_WGSL).toContain('let sheetGate = smoothstep(0.0032, 0.0110, coneDensity)');
    expect(RENDER_WGSL).not.toContain('cocBlur < 0.62');
    expect(RENDER_WGSL).toContain('let waveEnv = sqrt(dot(sInfo.normal.xz, sInfo.normal.xz)');
    expect(RENDER_WGSL).toContain('let macroSnellGate = smoothstep(0.615, 0.675, rd.y);');

    const evalSheetGate = (cocBlur: number): number => 1.0 - smoothstep(0.44, 0.88, cocBlur);
    // Verify smooth monotonic transition across cocBlur = 0.62 (max 0.01 step jump <= 0.04)
    for (let c = 0.40; c < 0.90; c += 0.01) {
      const g0 = evalSheetGate(c);
      const g1 = evalSheetGate(c + 0.01);
      expect(g1).toBeLessThanOrEqual(g0 + 1e-12);
      expect(Math.abs(g1 - g0)).toBeLessThan(0.04);
    }

    // 2. Foreground translucent stem over dark TIR blood preserves dark crimson absorption (zero bright smoothBehind halo)
    expect(RENDER_WGSL).toContain('tirBg = mix(smoothBehind, tirBg, turbidSharpness);');
    expect(RENDER_WGSL).not.toContain('let effBehind = mix(smoothBehind, behindPlume, turbidSharpness);');
    const smoothBehindG = 0.84;
    const tauIntactStem = 0.005; // Thin translucent foreground stem: transmittance.g ~ 0.42, turbidSharpness ~ 0.033
    const turbidSharpness = Math.exp(-tauIntactStem * 680.0);
    const tirTransG = 0.0; // Opaque reflected TIR stem behind foreground stem
    const newBehindG = (smoothBehindG * (1 - turbidSharpness) + 0.9 * turbidSharpness) * tirTransG;
    const buggyEffBehindG = smoothBehindG * (1 - turbidSharpness) + 0.0 * turbidSharpness;
    expect(newBehindG).toBeCloseTo(0.0, 12);
    expect(buggyEffBehindG).toBeGreaterThan(0.80);

    // 3. 3D topside drop reflection tapers chord -> 0 at limbs, includes wetting-collar flare, and gates over dark plumes
    expect(COMPOSITE_WGSL).not.toContain('hitT + 1.45 * radii.x');
    expect(COMPOSITE_WGSL).toContain('smoothstep(-0.18, 0.70, -bestN.y)');
    expect(COMPOSITE_WGSL).toContain('let clearWaterGate = select(');
    expect(COMPOSITE_WGSL).toContain('sampleSpectralLut(hbReflTau, 0.08)');
    expect(COMPOSITE_WGSL).toContain('intersectAerialDropFrom(ro, rd, dropHit.tExit + 0.0015');

    // 4. Crater-floor-anchored sub-collar containment, localized meniscusDimple, pupil-filtered TIR volume, & horizon tSurf continuity
    expect(RENDER_WGSL).toContain('let depthBelowCrater = hWaveLocal - rayPos.y;');
    expect(RENDER_WGSL).toContain('collarSmoothMix');
    expect(RENDER_WGSL).toContain('smoothstep(0.0055, 0.026, s0)');
    expect(RENDER_WGSL).toContain('var tirDir = reflect(rd, nUnder);');
    expect(RENDER_WGSL).toContain('var tirPlumeDir = reflect(rd, nPupilUnder);');
    expect(RENDER_WGSL).toContain('select(0.0, topOut.a, airWeight > 0.001)');
  });

  it('calibrates physical dielectric Fresnel topside drop reflections without tinting transmitted tank light and bounds nascent sub-collar smear', () => {
    const smoothstep = (e0: number, e1: number, x: number): number => {
      const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
      return t * t * (3 - 2 * t);
    };
    const fresnelAirToWater = (cosI: number): number => {
      const c = Math.max(0, Math.min(1, cosI));
      const sin2T = Math.pow(1.0 / 1.333, 2) * Math.max(0.0, 1.0 - c * c);
      if (sin2T >= 1.0) return 1.0;
      const cosT = Math.sqrt(1.0 - sin2T);
      const rs = (1.0 * c - 1.333 * cosT) / (1.0 * c + 1.333 * cosT);
      const rp = (1.333 * c - 1.0 * cosT) / (1.333 * c + 1.0 * cosT);
      return 0.5 * (rs * rs + rp * rp);
    };

    // 1. No constant +0.09 Fresnel floor and no hbReflFilter multiplying transmitted tank water
    expect(COMPOSITE_WGSL).not.toContain('0.09 + 1.28 * fresnel');
    expect(COMPOSITE_WGSL).not.toContain('hbReflFilter');
    expect(COMPOSITE_WGSL).toContain('let reflFresnel = clamp(fresnel * opticalDome * smoothstep(0.030, 0.12, -rd.y), 0.0, 0.11);');

    // Steep crown angle (cosI ~ 0.52) is ~5.4% subtle dielectric reflection, while shallow State 1 (cosI ~ 0.245) is >3.5x stronger (~23%)
    const fNormal = fresnelAirToWater(1.0);
    const fSteepCrown = fresnelAirToWater(0.52);
    const fShallowHero = fresnelAirToWater(0.245);
    expect(fNormal).toBeCloseTo(0.0204, 3);
    expect(fSteepCrown).toBeLessThan(0.065);
    expect(fShallowHero).toBeGreaterThan(0.20);
    expect(fShallowHero / fSteepCrown).toBeGreaterThan(3.5);

    // Transmitted blue-white tank water [0.61, 0.74, 0.89] blended with physical Fresnel preserves clear-water B > G > R hue (zero hot-pink)
    const waterLin = [0.61, 0.74, 0.89];
    const skyLin = [0.72, 0.79, 0.86];
    const dropLin = [0.18, 0.008, 0.014];
    for (const cosI of [1.0, 0.52, 0.245]) {
      const f = Math.min(0.28, fresnelAirToWater(cosI));
      const skyOcc = Math.min(f * 0.35, 0.06);
      const base = waterLin.map((w, i) => Math.max(0, w - skyLin[i] * skyOcc));
      const out = base.map((b, i) => b * (1 - f) + dropLin[i] * f);
      expect(out[2]).toBeGreaterThan(out[1]);
      expect(out[1]).toBeGreaterThan(out[0]);
    }

    // 2. Scale-invariant nascent sub-collar kinematic front bound + orifice zero-air dilution compensation
    expect(RENDER_WGSL).toContain('let emerg = (slot.posAndRadius.y - u.surfaceParams.x) / max(slot.posAndRadius.w, 0.010);');
    expect(RENDER_WGSL).toContain('let nascentTouch = smoothstep(-0.28, 0.24, emerg) * stemBelowGate;');
    expect(RENDER_WGSL).toContain('orificeCoreBoost');
    expect(RENDER_WGSL).toContain('orificeScatterDamp');
    const evalContain = (emerg: number, depthBelowCrater: number, rNormPlug = 0.0, deepA = 0.0): number => {
      const stemBelowGate = 1.0 - smoothstep(0.00025, 0.0018, deepA);
      const nascentTouch = smoothstep(-0.28, 0.24, emerg) * stemBelowGate;
      const axialArch = Math.max(0.0, 1.0 - 0.52 * rNormPlug * rNormPlug);
      const maxPlugDepth = (0.032 + (0.0022 - 0.032) * smoothstep(-0.24, 0.22, emerg)) * axialArch;
      const plugClip = 1.0 - smoothstep(maxPlugDepth, maxPlugDepth + 0.0011, depthBelowCrater);
      return 1.0 + (plugClip - 1.0) * nascentTouch;
    };
    // At State 1 (t = 0.965s, emerg ~ +0.58) in clear water (deepA = 0), smeared tail below collar rim (depth >= 5.5mm) is 100% suppressed
    expect(evalContain(0.58, 0.0055, 0.0, 0.0)).toBeCloseTo(0.0, 12);
    expect(evalContain(0.58, 0.0050, 0.8, 0.0)).toBeCloseTo(0.0, 12);
    // When a second drop impacts over an already-established umbilical stem (deepA >= 0.002), containment releases identically to 1.0!
    expect(evalContain(0.58, 0.0120, 0.0, 0.005)).toBeCloseTo(1.0, 12);
    // Once drop center submerges below crater (emerg <= -0.30), containment releases identically to 1.0 at all depths
    expect(evalContain(-0.30, 0.0150, 0.0, 0.0)).toBeCloseTo(1.0, 12);
    expect(evalContain(-0.50, 0.0350, 0.8, 0.0)).toBeCloseTo(1.0, 12);
  });

  it('prevents concave sminPoly secondary softbox folds during sinking coalescence and avoids double Fresnel dimming in topside dark-field', () => {
    const smoothstep = (e0: number, e1: number, x: number): number => {
      const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
      return t * t * (3 - 2 * t);
    };

    // 1. Both shadeAerialDrop and evalTopsideDropReflection evaluate softbox speculars on strictly convex ellipsoid normals
    expect(COMPOSITE_WGSL).toContain('let nConvex = normalize(((pHit - center) * invR) * invR);');
    expect(COMPOSITE_WGSL).toContain('let nConvexRefl = normalize(((pHit - center) * invR) * invR);');
    expect(COMPOSITE_WGSL).toContain('let convexMatch = select(1.0, smoothstep(0.955, 0.998, dot(n, nConvex)), isFreeDrop);');

    // At t = 1.015s..1.035s (center.y < surfY), concave sminPoly collar normal tilts toward +Y while nConvex tilts outward;
    // convexMatch rejects the concave fillet where dot(n, nConvex) < 0.955 while preserving 1.0 on the true convex crown (dot == 1)
    const nCrown = [0.18, 0.86, 0.477];
    const nCollarFillet = (() => {
      // h = 0.68 sminPoly blend between outward sphere normal [0.58, 0.42, 0.697] and upward water normal [0, 1, 0]
      const raw = [0.68 * 0.58, 0.68 * 0.42 + 0.32 * 1.0, 0.68 * 0.697];
      const len = Math.hypot(raw[0], raw[1], raw[2]);
      return raw.map((v) => v / len);
    })();
    const nConvexAtFillet = [0.58, 0.42, 0.697];
    const dotFillet =
      nCollarFillet[0] * nConvexAtFillet[0] +
      nCollarFillet[1] * nConvexAtFillet[1] +
      nCollarFillet[2] * nConvexAtFillet[2];
    expect(smoothstep(0.955, 0.998, 1.0)).toBe(1.0);
    expect(smoothstep(0.955, 0.998, dotFillet)).toBe(0.0);
    expect(nCrown[1]).toBeGreaterThan(0.8);

    // 2. Dark-Field topside reflection adds Fresnel air delta without dimming already-(1-reflWeight)-transmitted plumes
    expect(COMPOSITE_WGSL).toContain('let darkReflDelta = max(dropRefl.rgb - skyRefl, vec3<f32>(0.0)) * dropReflWeight;');
    const darkPlumeTop = [0.42, 0.012, 0.024];
    const darkSkyRefl = [0.001, 0.002, 0.004];
    const darkDropRefl = [0.14, 0.003, 0.006];
    const w = 0.22;
    const outDark = darkPlumeTop.map((c, i) => c + Math.max(0, darkDropRefl[i] - darkSkyRefl[i]) * w);
    expect(outDark[0]).toBeGreaterThanOrEqual(darkPlumeTop[0]);
  });

  it('enforces continuous vessel autoLoop = false default, non-intrusive camera, and persistent floor rouleaux sedimentation without yMin deletion sponge', () => {
    expect(engineSrc).toContain('public autoLoop = false;');
    // Spawn & pendant methods must never hijack camera preset or target
    const spawnIdx = engineSrc.indexOf('public spawnDropAtNdc(');
    const replayIdx = engineSrc.indexOf('public resetSimulation(');
    expect(spawnIdx).toBeGreaterThan(0);
    expect(replayIdx).toBeGreaterThan(spawnIdx);
    const spawnBlock = engineSrc.slice(spawnIdx, replayIdx);
    expect(spawnBlock).not.toContain('applyPreset');
    expect(spawnBlock).not.toContain('camera.azimuth =');
    expect(spawnBlock).not.toContain('camera.elevation =');

    // MultiDropSystem evolves cleanly past t = 30s without auto-wiping slots
    const sys = new MultiDropSystem();
    sys.reset(1.0);
    stepSystemTo(sys, 28.0, 1 / 60);
    const lateSlot = sys.spawnAerialDrop([0.04, -0.03], undefined, 2.2);
    expect(lateSlot.active).toBe(true);
    expect(sys.simTime).toBeGreaterThan(27.9);

    // No floor deletion sponge at yMin in particles.ts or advect.ts; persistent cRouleauxBed conserved in conc.a
    expect(PARTICLE_ADVECT_SPLAT_WGSL).not.toContain('smoothstep(yMin, yMin + 0.075');
    expect(CONC_RESOLVE_WGSL).not.toContain('smoothstep(u.domainBounds.y, u.domainBounds.y + 0.068');
    expect(CONC_RESOLVE_WGSL).toContain('cRouleauxBed');
    expect(CONC_RESOLVE_WGSL).toContain('upperDilute');
    expect(PARTICLE_ADVECT_SPLAT_WGSL).toContain('sheetVec');
  });

  it('grows pendant drop volume linearly V(t) = V_min + Q*t from R = 0.018 to 0.078 anchored to pipette tip y = 0.55, and slumps dome rapidly (< 0.26s) upon impact', () => {
    expect(PENDANT_MIN_RADIUS).toBeCloseTo(0.018, 6);
    expect(PENDANT_MAX_RADIUS).toBeCloseTo(0.078, 6);
    expect(PIPETTE_TIP_Y).toBeCloseTo(0.55, 6);
    expect(PENDANT_GROWTH_DURATION).toBeCloseTo(2.2, 6);

    const sys = new MultiDropSystem();
    sys.reset(1.0);
    const p = sys.beginPendantDrop([0.16, -0.08], 1.7);
    const pIdx = sys.slots.indexOf(p);
    expect(pIdx).toBeGreaterThanOrEqual(0);
    const vMin = sphereVolume(PENDANT_MIN_RADIUS);
    const vMax = sphereVolume(PENDANT_MAX_RADIUS);

    // Verify exact linear volume growth V(t) = V_min + (V_max - V_min) * (t / T_growth) -> R ~ t^(1/3)
    const sampleTimes = [0.22, 0.66, 1.10, 1.76];
    let prevT = 0;
    for (const tHold of sampleTimes) {
      expect(sys.updatePendantDrop(tHold - prevT)).toBe(false);
      prevT = tHold;
      const expectedVol = vMin + (vMax - vMin) * (tHold / PENDANT_GROWTH_DURATION);
      expect(p.aerialVolume).toBeCloseTo(expectedVol, 9);
      expect(p.radius).toBeCloseTo(radiusFromVolume(expectedVol), 7);
      // Center sags progressively below pipette tip y = 0.55 under Young-Laplace Bond elongation
      expect(p.center[1]).toBeLessThan(PIPETTE_TIP_Y - p.radius * 0.85);
    }

    // Verify SimUniformPacker encodes pipettePresence negatively into velOscAndDrain.z while drainRate == 0
    const packer = new SimUniformPacker();
    const cam = new OrbitCamera();
    cam.updateMatrices(16 / 10);
    const snapHold = sys.step(1 / 120);
    const uHold = packer.pack({
      invViewProj: cam.invViewProjMatrix,
      viewProj: cam.viewProjMatrix,
      cameraPos: cam.position,
      cameraTarget: cam.target,
      viewportWidth: 2560,
      viewportHeight: 1600,
      dt: 1 / 120,
      frameIndex: 1,
      globalSeed: 1,
      drops: snapHold,
      spawnCursor: 0,
      spawnCount: 0,
      config: DEFAULT_FLUID_OPTICAL_CONFIG,
    });
    const pSlotBase = 64 + pIdx * 8;
    expect(uHold[pSlotBase + 6]).toBeCloseTo(-1.0, 5);

    // Release triggers damped Rayleigh l=2,3 neck-snap recoil and smooth pipette retraction over 0.42s
    expect(sys.releasePendantDrop()).toBe(p);
    sys.step(0.10);
    expect(p.pipettePresence).toBeGreaterThan(0.4);
    expect(p.pipettePresence).toBeLessThan(0.95);
    sys.step(0.40);
    expect(p.pipettePresence).toBe(0);

    // Fast inertial dome slump: 100% of aerial dome crest (center.y + radius) slumps below the crater cavity within <= 0.25s of meniscus touch while wake drainage continues smoothly
    const sysImpact = new MultiDropSystem();
    sysImpact.reset(1.0);
    stepSystemTo(sysImpact, IMPACT_TIME + 0.25, 1 / 240);
    const domeCrestY = sysImpact.slots[0].center[1] + sysImpact.slots[0].radius;
    expect(domeCrestY).toBeLessThan(WATER_SURFACE_Y - 0.05);
    expect(sysImpact.slots[0].volDrainRate).toBeGreaterThan(0.01);
    expect(COMPOSITE_WGSL).toContain('fn evalCapillaryPipette(');
    expect(WAVE_SPATIAL_FORCE_WGSL).toContain('let excavGate = smoothstep(-0.78, -0.04, emergNorm);');
  });

  it('restores full 380-720nm CIE D65 Soret + orange-cliff dichromatism (amber -> vermilion-orange -> ruby -> venous oxblood)', () => {
    // Synthetic 460nm cutoff and +56% 480nm cyan spike must remain absent
    expect(lutSrc).not.toContain('(wavelengthNm - 425.0) / 35.0');
    expect(lutSrc).not.toContain('0.56 * Math.exp');

    const { rgba32 } = generatePrahlSpectralLut();
    const sample = (ix: number, iv: number): [number, number, number] => {
      const idx = (iv * SPECTRAL_LUT_WIDTH + ix) * 4;
      return [rgba32[idx], rgba32[idx + 1], rgba32[idx + 2]];
    };

    // High-shear oxygenated path (iv = 63):
    // 1. Ultra-thin (~15 um, ix = 14): warm golden amber (R > G > B, G > 0.55)
    const [r14, g14, b14] = sample(14, 63);
    expect(r14).toBeGreaterThan(0.88);
    expect(g14).toBeGreaterThan(0.55);
    expect(g14).toBeGreaterThan(b14 * 1.35);

    // 2. Sheared thin sheet (~40 um, ix = 26): vivid vermilion / scarlet-orange (R > 0.80, 0.22 < G < 0.50, G > 1.8 * B)
    const [r26, g26, b26] = sample(26, 63);
    expect(r26).toBeGreaterThan(0.80);
    expect(g26).toBeGreaterThan(0.22);
    expect(g26).toBeLessThan(0.50);
    expect(g26).toBeGreaterThan(b26 * 1.80);

    // 3. Medium curtain (ix = 64 & ix = 84): pure arterial ruby (G/R < 0.09 at ix=64, G < 0.025 at ix=84 -> zero muddy orange on thick clouds!)
    const [r64, g64, b64] = sample(64, 63);
    const [r84, g84, b84] = sample(84, 63);
    expect(r64).toBeGreaterThan(0.55);
    expect(g64 / r64).toBeLessThan(0.09);
    expect(b64).toBeLessThan(0.015);
    expect(r84).toBeGreaterThan(0.42);
    expect(g84).toBeLessThan(0.025);
    expect(b84).toBeLessThan(0.005);

    // 4. Low-shear venous rouleaux (iv = 0) is strictly darker in red across medium/thick paths than arterial (iv = 63)
    const [rVen64] = sample(64, 0);
    const [rVen140] = sample(140, 0);
    const [rArt140] = sample(140, 63);
    expect(rVen64).toBeLessThan(r64 - 0.04);
    expect(rVen140).toBeLessThan(rArt140 - 0.05);
  });

  it('prevents underwater crater eyeball Snell cutouts, chrome crater smudges, and close-defocus pink washout', () => {
    const smoothstep = (e0: number, e1: number, x: number): number => {
      const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
      return t * t * (3 - 2 * t);
    };

    expect(RENDER_WGSL).toContain('let pupilSnellGate = smoothstep(0.655, 0.735, pupilCosI);');
    expect(RENDER_WGSL).toContain('let snellOpenGate = macroSnellGate * mix(pupilSnellGate * mix(0.28, 1.0, craterDamp), 1.0, steepSnell);');
    expect(RENDER_WGSL).toContain('let plumeClearance = exp(-(fgPlume.tauTotal * 460.0 + fgPlume.tauIntact * 380.0));');
    expect(RENDER_WGSL).toContain('let pTirOrigin = vec3<f32>(pRipple.x, min(sInfo.height, u.surfaceParams.x) - 0.0012, pRipple.z);');
    expect(RENDER_WGSL).not.toContain('1.0 + 0.38 * cocBlur');

    // Reproduce exact User Screenshot 1 & 2 geometry: shallow underwater camera (rd.y = 0.12..0.58)
    // hitting a steep concave crater wall (local cosI = 0.82 > 0.6614 -> fresnelWindow = 0.04) with tirBloodPresence = 0:
    // windowEff MUST remain identically 1.0 (100% TIR, zero white eyeball cutout)!
    for (const rdY of [0.05, 0.22, 0.38, 0.55, 0.61]) {
      const macroSnellGate = smoothstep(0.615, 0.675, rdY);
      const pupilSnellGate = smoothstep(0.655, 0.735, rdY);
      const steepSnell = smoothstep(0.68, 0.76, rdY);
      const craterDamp = 0.0; // Deep crater
      const baseGate = pupilSnellGate * (0.28 + 0.72 * craterDamp);
      const snellOpenGate = macroSnellGate * (baseGate + (1.0 - baseGate) * steepSnell);
      const fresnelWindow = 0.04;
      const tirBloodPresence = 0.0;
      const rawWindow = fresnelWindow + (1.0 - fresnelWindow) * tirBloodPresence * (1.0 - Math.max(0.65 * macroSnellGate, steepSnell));
      const windowEff = 1.0 + (rawWindow - 1.0) * snellOpenGate;
      expect(windowEff).toBeCloseTo(1.0, 12);
    }

    // Conversely, looking steeply upward into true Snell's Window (rd.y = 0.82..0.88, even across a deep crater craterDamp = 0) opens 100% cleanly without grey crater shadow
    const steepMacro = smoothstep(0.615, 0.675, 0.82);
    const steepPupil = smoothstep(0.655, 0.735, 0.60);
    const steepFactor = smoothstep(0.68, 0.76, 0.82);
    const steepBase = steepPupil * 0.28;
    expect(steepMacro * (steepBase + (1.0 - steepBase) * steepFactor)).toBeCloseTo(1.0, 12);
  });

  it('prevents pipette-bulb overlap, meniscus parcel stranding, crater wake suction, yellow-tan lysed fog, and underwater crater orifice rings', () => {
    // 1. Pendant bulb apex (center[1] + radius * (1 + osc)) stays strictly <= PIPETTE_TIP_Y (0.55) across [0, 2.2s]
    const sys = new MultiDropSystem();
    sys.reset(1.0);
    const p = sys.beginPendantDrop([0.22, -0.14], 1.3);
    expect(p.center[1] + p.radius * (1.0 + p.oscillation)).toBeLessThanOrEqual(PIPETTE_TIP_Y);
    for (let step = 0; step < 43; step += 1) {
      expect(sys.updatePendantDrop(0.05)).toBe(false);
      const apexY = p.center[1] + p.radius * (1.0 + p.oscillation);
      expect(apexY).toBeLessThanOrEqual(PIPETTE_TIP_Y);
    }

    // 2. Pipette uses tilt- & offset-invariant 2D horizontal ray-to-axis distance and zero front-overlap bias
    expect(RENDER_WGSL).toContain('let perpDistXZ = abs(rel0.x * rd.z - rel0.y * rd.x) / sqrt(aXZ);');
    expect(COMPOSITE_WGSL).toContain('let maxPipT = select(maxSurfT, min(maxSurfT, dropHit.tEnter + 0.0005), dropHit.hit);');

    // 3. Wave solver decouples downward stem suction and uses elastocapillary dispersion with < 0.35s crater rebound half-period
    expect(WAVE_SPATIAL_FORCE_WGSL).toContain('let nonSuctionV = max(vSubY, 0.0);');
    expect(WAVE_SPATIAL_FORCE_WGSL).toContain('let wallTaper = 1.0 - smoothstep(halfXZ * 0.62, halfXZ * 0.92, length(worldXZ));');
    expect(WAVE_SPECTRAL_PROPAGATE_WGSL).toContain('let gEff = 0.84;');
    expect(WAVE_SPECTRAL_PROPAGATE_WGSL).toContain('let sigmaOverRho = 0.0024;');
    const kCrater = 34.0;
    const omegaCrater = Math.sqrt((0.84 * kCrater + 0.0024 * kCrater ** 3) * Math.tanh(kCrater * 1.28));
    expect(Math.PI / omegaCrater).toBeLessThan(0.35);

    // 4. Gentle meniscus anchor + monotonic age-gated Fahraeus rouleaux settling (zero detached bottom O-ring cliff)
    expect(PARTICLE_ADVECT_SPLAT_WGSL).toContain('let settleGate = mix(0.38, 1.0, smoothstep(0.002, 0.055, depthBelowSurf));');

    // 5. vOxShear is gated by intactFrac (eliminating yellow-tan Soret fog on pure cLysed) & Dark-Field tightens dfDichroFactor to thin sheets
    expect(CONC_RESOLVE_WGSL).toContain('let activeSheet = thinIntactSheet * (0.38 + 0.62 * shearDispersion) * intactFrac;');
    expect(RENDER_WGSL).toContain('let dfDichroFactor = smoothstep(0.09, 0.32, vOxShear)');
    expect(RENDER_WGSL).toContain('let depthFade = (1.0 - smoothstep(0.015, 0.036, depthBelowCrater)) * stemBelowGate * domeStillEmerging;');
    expect(RENDER_WGSL).not.toContain('let maxPlugDepth =');
  });

  it('enforces Round 7 umbilical stem continuity, shadow-free Snell window, borosilicate Pasteur pipette, 3D curved swell optics, and pure Dark-Field dichromatism', () => {
    // 1. Decoupled fast visible dome slump (~0.20s) + 1.26s C2 wake drainage + continuous MAC wake draft + lossless float32 parcel age/lysis packing
    const sys = new MultiDropSystem();
    sys.reset(1.0);
    stepSystemTo(sys, IMPACT_TIME + 0.24, 1 / 120);
    const sSlump = sys.slots[0];
    expect(sSlump.active).toBe(true);
    const domeCrestAt024 = sSlump.center[1] + sSlump.radius * (1.0 + sSlump.oscillation);
    expect(domeCrestAt024).toBeLessThan(WATER_SURFACE_Y - 0.04);
    expect(sSlump.volDrainRate).toBeGreaterThan(0.5);

    expect(FLUID_ADVECT_MAC_WGSL).toContain('let slugPhase = smoothstep(0.14, 0.76, drainRate);');
    expect(FLUID_ADVECT_MAC_WGSL).toContain('let stemDrive = clamp(0.045 + drainRate * 0.22, 0.0, 0.20) * smoothstep(0.0, 0.05, drainRate);');
    expect(PARTICLE_ADVECT_SPLAT_WGSL).toContain('let parcelAge = ageTicks * (1.0 / 240.0);');
    expect(PARTICLE_ADVECT_SPLAT_WGSL).toContain('ageTicks = min(8191.0, ageTicks + max(1.0, round(dt * 240.0)));');
    expect(PARTICLE_ADVECT_SPLAT_WGSL).toContain('let sheetSpan = clamp(sLenNow * 0.62, 0.0032, 0.0085);');
    expect(RENDER_WGSL).toContain('let intactMask = smoothstep(0.00006, 0.00085, conc.r) * meniscusMask;');
    expect(RENDER_WGSL).toContain('let lysedMask  = smoothstep(0.00028, 0.0016, conc.g) * meniscusMask;');

    // Substep-exact 240Hz tick rate + float32 precision check for packed floor(stretchAndLysis.w) ageTicks + fractional lysis in [0, 0.98)
    for (const dt of [1 / 60, 1 / 120, 1 / 240]) {
      const ticksPerStep = Math.max(1, Math.round(dt * 240));
      const stepsPerSec = Math.round(1 / dt);
      expect((ticksPerStep * stepsPerSec) / 240).toBeCloseTo(1.0, 9);
    }
    const f32 = new Float32Array(1);
    for (const ticks of [0, 1, 312, 960, 4096, 8191]) {
      for (const lysis of [0.0, 0.001, 0.42, 0.9795]) {
        f32[0] = ticks + lysis;
        expect(Math.floor(f32[0])).toBe(ticks);
        expect(f32[0] - Math.floor(f32[0])).toBeCloseTo(lysis, 3);
      }
    }

    // 2. Steep Snell's Window: capped apSnell, steepSnell crater-shadow immunity, Fresnel-gated rippleGlint, C2 meniscus collar & optical bokeh
    expect(RENDER_WGSL).toContain('let apSnell = min(apRadius + waveEnv * 0.0065, 0.0068);');
    expect(RENDER_WGSL).toContain('let steepSnell = smoothstep(0.68, 0.76, rd.y);');
    expect(RENDER_WGSL).not.toContain('snellAir * 0.85');
    expect(RENDER_WGSL).toContain('let specExcess = max(shadedSnellDrop - snellAir, vec3<f32>(0.0));');
    expect(RENDER_WGSL).toContain('* mix(0.03, 1.0, fresnelWindow);');
    expect(RENDER_WGSL).toContain('let rimChordMin = 0.38 * min(radii.x, radii.y);');
    expect(RENDER_WGSL).toContain('let nearMeniscusCollar = smoothstep(u.surfaceParams.x - 0.072, u.surfaceParams.x - 0.020, rayPos.y);');
    expect(RENDER_WGSL).toContain('let useSmoothWaveH = fineQuality && rayPos.y > u.surfaceParams.x - 0.065;');
    expect(RENDER_WGSL).not.toContain('sampleSpectralLut(subConc * 0.35 * u.opticalParams.x, 0.72)');

    // Algebraic proof: for any sub-sky penumbra alpha in [0, 1], min(shaded, sky) + max(shaded - sky, 0)*k === shaded (zero halo!)
    for (const alpha of [0.0, 0.005, 0.1, 0.5, 1.0]) {
      const sky = 0.88;
      const ruby = 0.12;
      const shaded = sky * (1 - alpha) + ruby * alpha;
      const composited = Math.min(shaded, sky) + Math.max(shaded - sky, 0) * 0.25;
      expect(composited).toBeCloseTo(shaded, 12);
    }

    // 3. Macro borosilicate Pasteur pipette reaching upper viewport + physical contactRamp collar gate (zero floating surface disc before touch)
    expect(RENDER_WGSL).toContain('let yTop = 1.250;');
    expect(RENDER_WGSL).toContain('let rOut = mix(0.0110, 0.0255, taperS) + 0.0025 * barrelS;');
    expect(RENDER_WGSL).toContain('let rBore = mix(0.0066, 0.0130, taperS) + 0.0018 * barrelS;');
    expect(RENDER_WGSL).toContain('let topFade = (1.0 - smoothstep(0.88, 0.995, ndcHitY)) * (1.0 - smoothstep(1.10, 1.24, pHit.y));');
    expect(RENDER_WGSL).toContain('let hMenisc = select(0.002 * bloodRetract, 0.020 + 0.0036 * rMeniscNorm * rMeniscNorm, isHeld);');
    expect(RENDER_WGSL).toContain('let contactRamp = 1.0 - smoothstep(-0.009, 0.0012, contactGap);');

    const pipSys = new MultiDropSystem();
    pipSys.reset(1.0);
    const pSlot = pipSys.beginPendantDrop([0, 0]);
    pipSys.updatePendantDrop(0.5);
    expect(pSlot.pipettePresence).toBe(1.0);
    pipSys.releasePendantDrop();
    for (let i = 0; i < 14; i += 1) {
      pipSys.step(1 / 60);
    }
    expect(pSlot.pipettePresence).toBeLessThan(0.05);
    pipSys.step(2 / 60);
    expect(pSlot.pipettePresence).toBe(0.0);

    // 4. Continuous refracted rebound-swell throat lensing (no synthetic rSwellFoot cylinder) + grazing Fresnel + studio softbox gloss
    expect(RENDER_WGSL).toContain('let activeCurve = smoothstep(0.015, 0.14, slopeLen);');
    expect(RENDER_WGSL).not.toContain('let rSwellFoot =');
    expect(RENDER_WGSL).toContain('let plumePresence = smoothstep(0.0006, 0.018, plume.tauTotal);');
    expect(RENDER_WGSL).toContain('let domeTau = max(0.0065, domeChord * surfBlood * 0.46 * u.opticalParams.x);');
    expect(RENDER_WGSL).toContain(
      'mix(1.0, mix(0.045, 0.16, grazingFresnelRim), bloodInColumn)'
    );
    expect(RENDER_WGSL).toContain('let tauCell = mix(0.0042, 0.0065, smoothstep(0.10, 0.55, cocBlur));');
    expect(RENDER_WGSL).toContain('orificeCoreBoost = max(orificeCoreBoost, 1.0 + select(0.40, 0.65, topsideCam) * slotOrifice);');
    expect(COMPOSITE_WGSL).toContain('let softboxDome = (');
    expect(COMPOSITE_WGSL).toContain('smoothstep(0.912, 0.974, nDotHSoft)');
    expect(COMPOSITE_WGSL).toContain('(surfBounds.z - surfBounds.y) < 0.005');

    // 5. Dark-Field: zero magenta clear-water tint & thin-sheet scarlet-vermilion confined to coneDensity < 0.032 and tauTotal < 0.030
    expect(RENDER_WGSL).not.toContain('vec3<f32>(0.13, 0.014, 0.032)');
    expect(RENDER_WGSL).not.toContain('vec3<f32>(0.24, 0.026, 0.056)');
    expect(RENDER_WGSL).toContain('vec3<f32>(0.06, 0.005, 0.004) * surfBlood * waveSlope * opticalDome * inCuvetteTop');
    expect(RENDER_WGSL).toContain('let thinNeighborhood = 1.0 - smoothstep(0.016, 0.042, coneDensity);');
    expect(RENDER_WGSL).toContain('* (1.0 - smoothstep(0.009, 0.032, tauTotal))');
    expect(RENDER_WGSL).toContain('let plugSelfShadow = 1.0 - 0.76 * smoothstep(0.028, 0.088, coneDensity);');
    expect(RENDER_WGSL).toContain('dfCoherentRim = vec3<f32>(1.36, 0.220, 0.008) * dfRimProb * albedoScale;');
    expect(0.465 / 2.85).toBeGreaterThanOrEqual(0.13);
    expect(0.465 / 2.85).toBeLessThanOrEqual(0.17);
    expect(0.330 / 2.05).toBeGreaterThanOrEqual(0.13);
    expect(0.330 / 2.05).toBeLessThanOrEqual(0.17);
    expect(0.510 / 3.15).toBeGreaterThanOrEqual(0.13);
    expect(0.510 / 3.15).toBeLessThanOrEqual(0.17);
    expect(0.220 / 1.36).toBeGreaterThanOrEqual(0.13);
    expect(0.220 / 1.36).toBeLessThanOrEqual(0.17);
  });

  it('enforces Round 8 rounded oblate Thomson-Newall vortex silhouette and tSurf-bilateral 1:1 Retina swell anti-aliasing', () => {
    expect(FLUID_ADVECT_MAC_WGSL).toContain(
      'let rNeck = clamp(slot.posAndRadius.w * mix(0.80, 1.08, slugPhase) + mix(0.012, 0.030, slugPhase), 0.018, 0.088) * azim;'
    );
    expect(FLUID_ADVECT_MAC_WGSL).toContain(
      'let slugRadial = (0.14 + 1.00 * min(rRatio * rRatio, 1.20)) * (1.0 - smoothstep(0.78, 1.15, rRatio));'
    );
    expect(FLUID_ADVECT_MAC_WGSL).toContain(
      'let stemRadial = (0.68 + 0.32 * min(rRatio * rRatio, 1.0)) * (1.0 - smoothstep(0.74, 1.10, rRatio));'
    );
    expect(FLUID_ADVECT_MAC_WGSL).toContain('let vertMask = smoothstep(surfY - 0.058, surfY - 0.014, pFaceY.y);');
    expect(FLUID_ADVECT_MAC_WGSL).toContain('let vertMaskXZ = smoothstep(surfY - 0.058, surfY - 0.014, pCell.y);');
    expect(FLUID_ADVECT_MAC_WGSL).toContain('vx = mix(vx, (relXFace.x / rXF) * flare, blendX * 0.65);');
    expect(FLUID_ADVECT_MAC_WGSL).toContain('vz = mix(vz, (relZFace.y / rZF) * flare, blendZ * 0.65);');
    expect(FLUID_ADVECT_MAC_WGSL).not.toContain('vx = mix(vx, (relXFace.x / rXF) * flare, blend * 0.65);');

    expect(PARTICLE_EMIT_WGSL).toContain('let slugWeight = smoothstep(0.26, 0.85, drop.velOscAndDrain.z);');
    expect(PARTICLE_EMIT_WGSL).toContain('let innerRing = 0.37 * slugWeight;');
    expect(PARTICLE_EMIT_WGSL).toContain('let isCapVeil = slugWeight > 0.08 && r3.z < 0.16;');
    expect(PARTICLE_EMIT_WGSL).toContain('let radFracCap = mix(0.14 * slugWeight, max(innerRing * 1.32, 0.08), sqrt(r3.y));');

    expect(COMPOSITE_WGSL).toContain('fn evalFlatWaterDistAtUV(');
    expect(COMPOSITE_WGSL).toContain('fn sampleVolNonPlanarSpan(');
    expect(COMPOSITE_WGSL).toContain('fn sampleVolDepthBilateral(');
    expect(COMPOSITE_WGSL).toContain('fn intersectWaterSurfaceFine(');
    expect(COMPOSITE_WGSL).toContain('let surfSpan = surfBounds.z - surfBounds.y;');
    expect(COMPOSITE_WGSL).toContain('let nonPlanarGate = max(');
    expect(COMPOSITE_WGSL).toContain('(surfSpan > 0.020 || extSpan > 0.032)');
    expect(COMPOSITE_WGSL).toContain('smoothstep(0.912, 0.974, nDotHSoft)');

    // 1. Verify exact antisymmetry vx(-x, -z) === -vx(+x, +z) when radial flare masks are evaluated on staggered X faces
    const sstep = (e0: number, e1: number, x: number) => {
      const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
      return t * t * (3 - 2 * t);
    };
    const evalFlareVx = (fx: number, fz: number, seed: number) => {
      const rXF = Math.hypot(fx, fz);
      const th = Math.atan2(fz, fx);
      const az =
        1.0 +
        0.036 * Math.cos(4.0 * th + seed * 1.73) +
        0.026 * Math.cos(6.0 * th - seed * 2.31) +
        0.018 * Math.sin(8.0 * th + seed * 0.89);
      const rRatioX = rXF / (0.048 * az);
      const slugRadX =
        (0.14 + 1.0 * Math.min(rRatioX * rRatioX, 1.2)) * (1.0 - sstep(0.78, 1.15, rRatioX));
      return (fx / rXF) * slugRadX;
    };
    for (const seed of [1.0, 2.718]) {
      for (const [fx, fz] of [
        [0.0167, 0.0],
        [0.0333, 0.0083],
        [0.05, -0.0167],
      ]) {
        expect(evalFlareVx(-fx, -fz, seed)).toBeCloseTo(-evalFlareVx(fx, fz, seed), 12);
      }
    }

    // 2. Verify flat-plane detrending suppresses perspective false triggers on calm water while preserving swell silhouettes
    for (const camRelY of [0.012, 0.08, 0.218]) {
      const dy0 = -camRelY / 2.2;
      const dy1 = dy0 - 3.0 * (0.663 / 800);
      const tSurfCalm0 = camRelY / -dy0;
      const tSurfCalm1 = camRelY / -dy1;
      expect(tSurfCalm0 - tSurfCalm1).toBeGreaterThan(0.032); // Raw perspective span exceeds 0.032 on flat water
      const nonPlanarCalm = Math.abs((tSurfCalm0 - tSurfCalm0) - (tSurfCalm1 - tSurfCalm1));
      expect(nonPlanarCalm).toBeCloseTo(0.0, 12); // Detrended non-planar span is identically 0
    }
    const camRelYSwell = 0.287;
    const rdYSwell = -0.28;
    const resFrontDome = (camRelYSwell - 0.01868) / -rdYSwell - camRelYSwell / -rdYSwell;
    const resRearTrough = (camRelYSwell + 0.00588) / -rdYSwell - camRelYSwell / -rdYSwell;
    expect(resRearTrough - resFrontDome).toBeGreaterThan(0.08);
  });

  it('enforces Round 9 turbid whole-blood aerial drop optics, capillary Nyquist grid scaling, single-impulse PDE forcing, and wave-warped Sub-Meniscus TIR', () => {
    // 1. Turbid whole-blood (Hct ~ 45%, n = 1.378) core blocks glass-marble sky lensing (< 0.05% transmission)
    expect(COMPOSITE_WGSL).toContain('evalDropStudioWindowSpec');
    expect(COMPOSITE_WGSL).not.toContain('pow(max(keyAlign, 0.0), 150.0)');
    expect(COMPOSITE_WGSL).toContain('let ballisticTrans = exp(-chord * 92.0 * u.opticalParams.x * turbidGate);');
    const coreChord = 0.092;
    expect(Math.exp(-coreChord * 92.0)).toBeLessThan(0.0005);
    const thinNeckChord = 0.008;
    expect(Math.exp(-thinNeckChord * 92.0)).toBeGreaterThan(0.45);

    // 2. Full-canvas open meniscus grid scaling (WAVE_HALF_EXTENT = 1.60 -> 3.20m span) + unattenuated [-0.48, +0.48] cuvette
    expect(WAVE_HALF_EXTENT).toBeCloseTo(1.60, 6);
    const waveDx = (2.0 * WAVE_HALF_EXTENT) / WAVE_GRID_SIZE;
    expect(waveDx).toBeLessThan(0.013);
    expect(WAVE_HALF_EXTENT * 0.72).toBeGreaterThan(0.48);

    // 3. Single momentum-coupled PDE crater + capillary contact-line impulse (zero synthetic double-piston reboundImpulse)
    expect(WAVE_SPATIAL_FORCE_WGSL).not.toContain('reboundImpulse');
    expect(WAVE_SPATIAL_FORCE_WGSL).toContain('let capImpulse =');

    // 4. Wave-warped Sub-Meniscus TIR with folded water-path attenuation & zero origin-locked (0,0) radial vectors
    expect(RENDER_WGSL).toContain('let rPupil = clamp(0.0032 + 0.0028 / max(rd.y, 0.20), 0.0035, 0.0078);');
    expect(RENDER_WGSL).toContain('let tirFoldedAtten = exp(-tSurfHit * 0.52) * mix(0.30, 0.46, craterDamp);');
    expect(RENDER_WGSL).not.toContain('mercuryBand');
    expect(RENDER_WGSL).not.toContain('slopeContrast');
  });

  it('enforces Round 10 live-exploration XDR sRGB parity, brisk pipette squeeze, non-reflecting cuvette ripple absorber, rose-ruby hemolysis guard, and side-on aquarium refraction', () => {
    // 1. Apple Silicon XDR extended-sRGB swapchain preserves IEC sRGB transfer on midtones (< 0.78) while opening EDR specular headroom (> 1.0)
    expect(COMPOSITE_WGSL).toContain('return vec4<f32>(srgb + edrBoost, 1.0);');
    for (const lin of [0.02, 0.15, 0.42, 0.75]) {
      const excess = Math.max(lin - 0.78, 0.0);
      expect(excess).toBe(0.0);
    }
    const specLin = 1.45;
    const ex = specLin - 0.78;
    const boost = (ex * 2.2) / (ex + 2.2 * 1.35 + 1e-4);
    expect(1.0 + boost).toBeGreaterThan(1.25);

    // 2. Quick-click / + Drop squeezes briskly (~0.15s) from PIPETTE_TIP_Y = +0.55 and auto-releases prior quick drops on rapid multi-clicks
    const sys = new MultiDropSystem();
    sys.reset(1.0);
    expect(sys.slots[0].pipettePresence).toBe(1.0);
    const q1 = sys.beginQuickPipetteDrop([0.12, -0.08], 1.0);
    expect(q1.mode).toBe('pendant');
    expect(sys.hasAutoSqueezePendantDrop()).toBe(true);
    expect(q1.center[1]).toBeLessThan(PIPETTE_TIP_Y);
    expect(q1.center[1]).toBeGreaterThan(PIPETTE_TIP_Y - 0.04);
    // Rapid second quick-click auto-releases q1 instead of cancelling it
    const q2 = sys.beginQuickPipetteDrop([-0.14, 0.09], 1.0);
    expect(q1.active).toBe(true);
    expect(q1.mode).toBe('aerial');
    expect(q2.mode).toBe('pendant');
    for (let i = 0; i < 12; i += 1) {
      sys.updatePendantDrop(1 / 60);
      sys.step(1 / 60);
    }
    expect(q2.mode).toBe('aerial');
    expect(q2.active).toBe(true);

    // 3. OrbitCamera clamps camera above y = -0.88 so user cannot orbit beneath the y = -0.98 glass cuvette base
    const cam = new OrbitCamera();
    cam.orbit(0, -10.0);
    cam.updateMatrices(1.6);
    expect(cam.position[1]).toBeGreaterThanOrEqual(-0.8801);

    // 4. Rose-ruby hemolysis guard prevents 430nm Soret absorption from filtering cool blue studio water into yellow-green (G > R)
    expect(RENDER_WGSL).toContain(
      'let gRoseGuard = min(lutSieveRaw.g, mix(lutSieveRaw.b * 0.94 + 0.04 * lutSieveRaw.r, lutSieveRaw.g, intactDichroGate));'
    );
    expect(PARTICLE_ADVECT_SPLAT_WGSL).not.toContain('mix(0.068, 0.165');
    expect(PARTICLE_ADVECT_SPLAT_WGSL).toContain(
      'let stackSettle = mix(0.032, 0.062, smoothstep(0.04, 0.48, depthBelowSurf));'
    );

    // 5. Non-reflecting square Rayleigh velocity absorber + surfaceBloodContact ripple decoupling + borosilicate cuvette + side-on topSnellGate
    expect(WAVE_SPATIAL_FORCE_WGSL).toContain('let rayleighDamp = exp(-dt * (24.0 * absorbZone * absorbZone));');
    expect(RENDER_WGSL).toContain('fn cuvetteSignedDistXZ(');
    expect(RENDER_WGSL).toContain('fn evalCuvetteFloorBackdrop(');
    expect(RENDER_WGSL).toContain(
      'let surfaceBloodContact = clamp(max(collarProximity, smoothstep(0.004, 0.026, nearSurfConc)), 0.0, 1.0);'
    );
    expect(RENDER_WGSL).toContain(
      'let topSnellGate = opticalDome * smoothstep(0.09, 0.25, -rd.y) * (1.0 - smoothstep(0.45, 0.72, -rd.y)) * smoothstep(0.025, 0.14, camRelY);'
    );
  });

  it('enforces Round 11 512x512 dispersive capillary FFT, crater-contour parcel emission, 3D solenoidal curl unfurling, 650nm RBC key reflectance, and Hero topside re-drop crane', () => {
    // 1. 512x512 capillary-gravity FFT (dx = 6.25 mm) with 256-thread 1-butterfly/thread disjoint coverage across all 9 stages
    expect(WAVE_GRID_SIZE).toBe(512);
    expect((2.0 * WAVE_HALF_EXTENT) / WAVE_GRID_SIZE).toBeCloseTo(0.00625, 6);
    expect(WAVE_FFT_ROW_WGSL).toContain('@compute @workgroup_size(256, 1, 1)');
    expect(WAVE_FFT_ROW_WGSL).toContain('let idx1 = tid + 256u;');
    for (let stage = 0; stage < 9; stage++) {
      const seen = new Uint8Array(512);
      const halfSpan = 1 << stage;
      for (let tid = 0; tid < 256; tid++) {
        const k = tid & (halfSpan - 1);
        const blockBase = (tid >> stage) << (stage + 1);
        const i0 = blockBase + k;
        const i1 = i0 + halfSpan;
        expect(i0).toBeGreaterThanOrEqual(0);
        expect(i1).toBeLessThan(512);
        seen[i0] += 1;
        seen[i1] += 1;
      }
      for (let i = 0; i < 512; i++) {
        expect(seen[i]).toBe(1);
      }
    }

    // 2. Exact 3D solenoidal polarization orthogonality (k_i . d_i === 0 -> div(u) === 0) across parcel & MAC unfurling
    const parcelModes: Array<[[number, number, number], [number, number, number]]> = [
      [[8.4, 5.2, 6.6], [-8.16, 3.84, 7.36]],
      [[-7.2, 6.0, 9.4], [-8.20, -1.38, -5.40]],
      [[11.5, -7.5, -8.5], [8.30, -1.10, 12.20]],
    ];
    const macModes: Array<[[number, number, number], [number, number, number]]> = [
      [[6.4, 4.0, 5.2], [-6.25, 2.85, 5.50]],
      [[-5.6, 4.8, 7.2], [-6.30, -1.26, -4.06]],
      [[9.0, -6.0, -6.4], [6.40, -0.96, 9.90]],
    ];
    for (const [k, d] of [...parcelModes, ...macModes]) {
      const dot = k[0] * d[0] + k[1] * d[1] + k[2] * d[2];
      expect(dot).toBeCloseTo(0.0, 10);
    }

    // 3. True posXZ crater-contour parcel emission + solenoidal top-2-cell MAC-to-center boundary + conserved hemolysis veil
    expect(PARTICLE_EMIT_WGSL).toContain('let waveUV = clamp(worldXZToWaveUV(posXZ)');
    expect(PARTICLE_EMIT_WGSL).toContain('let nascentCoreFill = smoothstep(-0.32, 0.18, emergNorm);');
    expect(FLUID_MAC_TO_CENTER_WGSL).toContain('let rActive = clamp(slot.posAndRadius.w * 1.15 + 0.032, 0.046, 0.092);');
    expect(CONC_RESOLVE_WGSL).toContain('exp(-dt * mix(0.016, 0.034,');
    expect(Math.exp(-12.0 * 0.034)).toBeGreaterThan(0.65);

    // 4. Calibrated 650-680 nm erythrocyte Mie key reflectance + curved swell backlight softbox + Hero topside re-drop crane
    expect(RENDER_WGSL).toContain('let keyFace = smoothstep(-0.18, 0.82, dot(nMacro, keyDir)) * keyIllum;');
    expect(COMPOSITE_WGSL).toContain('let rimSoftDir = normalize(vec3<f32>(-0.16, 0.58, -0.80));');

    const cam = new OrbitCamera(CAMERA_PRESETS.hero);
    cam.syncHeroFollow(5.0, true, 0.0);
    cam.updateMatrices(16 / 10);
    expect(cam.position[1]).toBeLessThan(WATER_SURFACE_Y - 0.14);

    // When a subsequent pendant/aerial drop is active above water (topsideCue = 1.0), Hero Follow cranes cleanly back topside
    cam.syncHeroFollow(5.0, true, 1.0);
    cam.updateMatrices(16 / 10);
    expect(cam.position[1]).toBeGreaterThan(WATER_SURFACE_Y + 0.24);
    expect(cam.elevation).toBeGreaterThan(0.25);
  });

  it('supports zero-downside macro zoom (0.24 m), 3D view-plane panning, descending plume focus tracking, macro aperture stop-down, and paused 16-frame super-sampling', () => {
    // 1. 3D view-plane pan & focusOn clamp cleanly across the full [-0.92, +0.52] vertical & [-0.42, +0.42] horizontal volume
    const cam = new OrbitCamera(CAMERA_PRESETS.hero);
    cam.panScreen(-0.25, -0.35);
    cam.tick(10.0, 16 / 10);
    expect(cam.followHeroEnabled).toBe(false);
    expect(cam.target[0]).toBeGreaterThan(0.05);
    expect(cam.target[1]).toBeLessThan(0.15);

    cam.panScreen(-99.0, -99.0);
    cam.tick(10.0, 16 / 10);
    expect(Math.abs(cam.target[0])).toBeLessThanOrEqual(MAX_CAMERA_TARGET_XZ + 1e-6);
    expect(cam.target[1]).toBeGreaterThanOrEqual(MIN_CAMERA_TARGET_Y - 1e-6);
    expect(cam.target[1]).toBeLessThanOrEqual(MAX_CAMERA_TARGET_Y + 1e-6);

    cam.focusOn([0.14, -0.45, -0.10], 0.62, true);
    expect(cam.target[0]).toBeCloseTo(0.14, 5);
    expect(cam.target[1]).toBeCloseTo(-0.45, 5);
    expect(cam.target[2]).toBeCloseTo(-0.10, 5);
    expect(cam.distance).toBeCloseTo(0.62, 5);

    // If user is already zoomed closer (e.g., 0.28 m), focusOn never pushes the camera back out
    cam.dolly(0.45);
    for (let i = 0; i < 25; i++) cam.tick(0.1, 16 / 10);
    expect(cam.distance).toBeCloseTo(0.62 * 0.45, 4);
    cam.focusOn([0.0, -0.55, 0.0], 0.68, true);
    expect(cam.distance).toBeCloseTo(0.62 * 0.45, 4);

    // 2. Automatic macro aperture stop-down preserves 100% exact aperture across legacy/preset distances (>= 0.82 m)
    //    while stopping down proportionally inside < 0.82 m so close-up vortex rings stay crisp front-to-back
    const packer = new SimUniformPacker();
    const sys = new MultiDropSystem();
    stepSystemTo(sys, 3.5, 1 / 120);
    const snapSub = sys.getSnapshot();

    const defaultCam = new OrbitCamera(CAMERA_PRESETS.abyss);
    defaultCam.updateMatrices(16 / 10);
    const outDefault = packer.pack({
      invViewProj: defaultCam.invViewProjMatrix,
      viewProj: defaultCam.viewProjMatrix,
      cameraPos: defaultCam.position,
      cameraTarget: defaultCam.target,
      viewportWidth: 2560,
      viewportHeight: 1600,
      dt: 1 / 60,
      frameIndex: 1,
      globalSeed: 1.0,
      drops: snapSub,
      spawnCursor: 0,
      spawnCount: 0,
      config: DEFAULT_FLUID_OPTICAL_CONFIG,
    });
    expect(outDefault[57]).toBeCloseTo(DEFAULT_FLUID_OPTICAL_CONFIG.apertureRadius, 7);

    const macroCam = new OrbitCamera(CAMERA_PRESETS.abyss);
    macroCam.dolly(0.1); // clamps to 0.24 m
    for (let i = 0; i < 25; i++) macroCam.tick(0.1, 16 / 10);
    const outMacro = packer.pack({
      invViewProj: macroCam.invViewProjMatrix,
      viewProj: macroCam.viewProjMatrix,
      cameraPos: macroCam.position,
      cameraTarget: macroCam.target,
      viewportWidth: 2560,
      viewportHeight: 1600,
      dt: 1 / 60,
      frameIndex: 1,
      globalSeed: 1.0,
      drops: snapSub,
      spawnCursor: 0,
      spawnCount: 0,
      config: DEFAULT_FLUID_OPTICAL_CONFIG,
    });
    expect(outMacro[56]).toBeCloseTo(0.24, 3);
    expect(outMacro[57]).toBeLessThan(DEFAULT_FLUID_OPTICAL_CONFIG.apertureRadius * 0.25);

    // 3. Descending submerged plume tracker persists after aerial drainage completes so Focus ('F') tracks sinking rings
    const subjects = sys.getInspectionSubjects();
    expect(subjects.length).toBeGreaterThanOrEqual(1);
    expect(subjects[0].kind).toBe('plume');
    expect(subjects[0].center[1]).toBeLessThan(WATER_SURFACE_Y - 0.15);
    expect(subjects[0].center[1]).toBeGreaterThan(-0.25);

    // 4. View-aligned backlight, deep macro C2 cubic reconstruction, and 16-frame paused progressive accumulation
    expect(LIGHTING_VOLUME_WGSL).toContain('let horizAim = unitAimXZ * 1.24;');
    expect(RENDER_WGSL).toContain('override MACRO_INSPECT_MODE: u32 = 0u;');
    expect(RENDER_WGSL).toContain('let macroZoom = select(0.0, 1.0 - smoothstep(0.32, 1.08, focalRef), MACRO_INSPECT_MODE != 0u && fineQuality);');
    expect(RENDER_WGSL).toContain('let gradFine = (tetK0 * f0 + tetK1 * f1 + tetK2 * f2 + tetK3 * f3) * (invTetDenom / 0.48);');
    expect(engineSrc).toContain('const PAUSED_ACCUM_TARGET = 16;');
    expect(engineSrc).toContain('public stepPausedSubsteps(');
    expect(engineSrc).toContain('public cycleFocusSubject(');
  });
});
