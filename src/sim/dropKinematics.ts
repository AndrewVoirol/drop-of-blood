import { unprojectWorldPoint } from '../math/camera';
import type { Mat4, Vec3 } from '../math/camera';
import { MAX_ACTIVE_DROPS } from '../shaders/common';

export const WATER_SURFACE_Y = 0.30;
export const PIPETTE_TIP_Y = 0.55;
export const INITIAL_DROP_RADIUS = 0.058;
export const PENDANT_MIN_RADIUS = 0.018;
export const PENDANT_MAX_RADIUS = 0.078; // Bond-number capillary pinch-off limit
export const PENDANT_GROWTH_DURATION = 2.2;
export const SHOT_CYCLE_DURATION = 22.0;
export const MAX_DROP_OFFSET_XZ = 0.26;

export const Y_START = 0.495;
export const V_START = -0.075;
export const G_SLOW = -0.155;

export const REFERENCE_DROP_VOLUME = (4.0 / 3.0) * Math.PI * Math.pow(INITIAL_DROP_RADIUS, 3);
export const BASE_COALESCE_DURATION = 1.26;

// Solve exact contact time where y(t) - R_0 = WATER_SURFACE_Y for initial hero drop:
export const IMPACT_TIME = (() => {
  const a = 0.5 * G_SLOW;
  const b = V_START;
  const c = Y_START - INITIAL_DROP_RADIUS - WATER_SURFACE_Y;
  const disc = Math.max(0, b * b - 4 * a * c);
  return (-b - Math.sqrt(disc)) / (2 * a); // ~0.9203s
})();

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

export function sphereVolume(radius: number): number {
  const r = Math.max(0.0, radius);
  return (4.0 / 3.0) * Math.PI * r * r * r;
}

export function radiusFromVolume(volume: number): number {
  if (volume <= 0.0) return 0.0;
  return Math.cbrt((3.0 * volume) / (4.0 * Math.PI));
}

export function clampDropDiskXZ(x: number, z: number, maxR = MAX_DROP_OFFSET_XZ): [number, number] {
  const sx = Number.isFinite(x) ? x : 0.0;
  const sz = Number.isFinite(z) ? z : 0.0;
  const r = Math.hypot(sx, sz);
  if (r <= maxR || r < 1e-8) {
    return [sx, sz];
  }
  const scale = maxR / r;
  return [sx * scale, sz * scale];
}

/**
 * Raycasts normalized device coordinates (ndcX, ndcY in [-1, 1]) onto the horizontal
 * water meniscus plane y = WATER_SURFACE_Y (falling back to camera-target plane at grazing angles),
 * clamping radially to safe interior tank coordinates.
 */
export function raycastMeniscusXZ(
  invViewProj: Mat4,
  cameraPos: Vec3,
  fallbackTarget: Vec3,
  ndcX: number,
  ndcY: number
): [number, number] {
  const pFar = unprojectWorldPoint(invViewProj, ndcX, ndcY, 1.0);
  let dx = pFar[0] - cameraPos[0];
  let dy = pFar[1] - cameraPos[1];
  let dz = pFar[2] - cameraPos[2];
  const len = Math.hypot(dx, dy, dz);
  if (len < 1e-8) {
    return [0.0, 0.0];
  }
  dx /= len;
  dy /= len;
  dz /= len;

  if (Math.abs(dy) > 1e-3) {
    const t = (WATER_SURFACE_Y - cameraPos[1]) / dy;
    if (t > 0.02 && t < 12.0) {
      return clampDropDiskXZ(cameraPos[0] + dx * t, cameraPos[2] + dz * t);
    }
  }

  // Grazing horizon fallback: intersect vertical plane facing camera through fallbackTarget
  const vx = fallbackTarget[0] - cameraPos[0];
  const vz = fallbackTarget[2] - cameraPos[2];
  const denom = dx * vx + dz * vz;
  if (Math.abs(denom) > 1e-5) {
    const tPlane = (vx * vx + vz * vz) / denom;
    if (tPlane > 0.02 && tPlane < 12.0) {
      return clampDropDiskXZ(cameraPos[0] + dx * tPlane, cameraPos[2] + dz * tPlane);
    }
  }
  return [0.0, 0.0];
}

export type DropLifecycleMode = 'inactive' | 'pendant' | 'aerial' | 'coalescing';

export interface DropSlotState {
  active: boolean;
  mode: DropLifecycleMode;
  center: [number, number, number];
  vel: [number, number, number];
  radius: number;
  impactRadius: number;
  impactVelY: number;
  aerialVolume: number;
  initialCoalesceVolume: number;
  ageInPhase: number;
  coalesceDuration: number;
  oscillation: number;
  oscBoost: number;
  oscAtContact: number;
  /** Exact instantaneous volume drainage rate -dV_aerial/dt normalized by REFERENCE_DROP_VOLUME */
  volDrainRate: number;
  /** Exact physical volume DeltaV transferred into the submerged tank during the last step */
  stepDrainedVolume: number;
  /** [0, 1] capillary pipette presence above this drop (1 while pendant, smoothly retracting after snap) */
  pipettePresence: number;
  /** Target auto-pinch volume (> 0 when quick-clicked so the drop squeezes briskly from the glass pipette tip) */
  autoReleaseVolume: number;
  seed: number;
}

export interface MultiDropSnapshot {
  simTime: number;
  activeCount: number;
  totalStepDrainedVolume: number;
  lastSpawnSimTime: number;
  phaseLabel: string;
  /** Primary visual focus point for thin-lens macro focal plane anchoring */
  primaryAerialFocus: [number, number, number] | null;
  slots: readonly DropSlotState[];
}

function createEmptySlot(): DropSlotState {
  return {
    active: false,
    mode: 'inactive',
    center: [0.0, Y_START, 0.0],
    vel: [0.0, 0.0, 0.0],
    radius: 0.0,
    impactRadius: 0.0,
    impactVelY: 0.0,
    aerialVolume: 0.0,
    initialCoalesceVolume: 0.0,
    ageInPhase: 0.0,
    coalesceDuration: BASE_COALESCE_DURATION,
    oscillation: 0.0,
    oscBoost: 0.0,
    oscAtContact: 0.0,
    volDrainRate: 0.0,
    stepDrainedVolume: 0.0,
    pipettePresence: 0.0,
    autoReleaseVolume: 0.0,
    seed: 1.0,
  };
}

function evalRayleighOscillation(age: number, radius: number, seed: number, boost: number): number {
  const rRatio = Math.pow(INITIAL_DROP_RADIUS / Math.max(radius, 0.020), 1.5);
  const omega2 = 9.6 * rRatio;
  const omega3 = 18.4 * rRatio; // l=3 capillary pear harmonic
  const damp = Math.exp(-age * 0.36);
  const amp2 = 0.092 + boost;
  const amp3 = 0.022 + 0.35 * boost;
  // Subtle aerodynamic oblate flattening (Weber number ~ v^2) during late free-fall
  const aeroOblate = -0.020 * smoothstep(0.25, 0.90, age);
  return (amp2 * Math.cos(age * omega2 + seed) + amp3 * Math.sin(age * omega3 + seed * 1.7)) * damp + aeroOblate;
}

/**
 * Concurrent N = 8 multi-drop kinematic, pendant-drop capillary swelling/pinch-off,
 * mid-air/crater drop-drop coalescence (R = (R1^3 + R2^3)^(1/3)), and exact
 * volume-conserving wetting-neck drainage (-dV_aerial/dt) solver.
 *Zero per-frame heap allocations; zero scripted primarySlug/secondaryStem/worthingtonJet pulses.
 */
export class MultiDropSystem {
  public readonly slots: DropSlotState[] = Array.from({ length: MAX_ACTIVE_DROPS }, () =>
    createEmptySlot()
  );
  public simTime = 0.0;
  public lastSpawnSimTime = 0.0;
  private pendantSlotIndex = -1;
  private pendingRecycledVolume = 0.0;

  private readonly snapshotObj: MultiDropSnapshot = {
    simTime: 0.0,
    activeCount: 0,
    totalStepDrainedVolume: 0.0,
    lastSpawnSimTime: 0.0,
    phaseLabel: 'Aerial Descent',
    primaryAerialFocus: null,
    slots: this.slots,
  };

  constructor() {
    this.reset(1.0, [0.0, 0.0]);
  }

  private clearSlot(slot: DropSlotState): void {
    slot.active = false;
    slot.mode = 'inactive';
    slot.radius = 0.0;
    slot.aerialVolume = 0.0;
    slot.initialCoalesceVolume = 0.0;
    slot.volDrainRate = 0.0;
    slot.stepDrainedVolume = 0.0;
    slot.pipettePresence = 0.0;
    slot.autoReleaseVolume = 0.0;
    slot.oscillation = 0.0;
    slot.oscBoost = 0.0;
  }

  private acquireSlotIndex(): number {
    for (let i = 0; i < MAX_ACTIVE_DROPS; i++) {
      if (!this.slots[i].active) return i;
    }
    // All 8 slots active: recycle the most completely drained non-pendant slot and conserve 100% of its remaining volume
    let bestIdx = 0;
    let smallestVol = Infinity;
    for (let i = 0; i < MAX_ACTIVE_DROPS; i++) {
      if (i === this.pendantSlotIndex) continue;
      if (this.slots[i].aerialVolume < smallestVol) {
        smallestVol = this.slots[i].aerialVolume;
        bestIdx = i;
      }
    }
    const evicted = this.slots[bestIdx];
    if (evicted.active && evicted.mode !== 'pendant' && evicted.aerialVolume > 0.0) {
      this.pendingRecycledVolume += evicted.aerialVolume;
      evicted.aerialVolume = 0.0;
    }
    return bestIdx;
  }

  public reset(seed = 1.0, offsetXZ: [number, number] = [0.0, 0.0]): MultiDropSnapshot {
    this.simTime = 0.0;
    this.lastSpawnSimTime = 0.0;
    this.pendantSlotIndex = -1;
    this.pendingRecycledVolume = 0.0;
    for (let i = 0; i < MAX_ACTIVE_DROPS; i++) {
      this.clearSlot(this.slots[i]);
    }
    const hero = this.spawnAerialDrop(offsetXZ, INITIAL_DROP_RADIUS, seed, Y_START, V_START);
    hero.pipettePresence = 1.0;
    return this.buildSnapshot(0.0);
  }

  public spawnAerialDrop(
    offsetXZ: [number, number],
    radius = INITIAL_DROP_RADIUS,
    seed = 1.0,
    startY = Y_START,
    velY = V_START
  ): DropSlotState {
    const idx = this.acquireSlotIndex();
    const s = this.slots[idx];
    const [cx, cz] = clampDropDiskXZ(offsetXZ[0], offsetXZ[1]);
    const rClamped = clamp(radius, PENDANT_MIN_RADIUS, 0.095);
    const vol = sphereVolume(rClamped);

    s.active = true;
    s.mode = 'aerial';
    s.center[0] = cx;
    s.center[1] = startY;
    s.center[2] = cz;
    s.vel[0] = 0.0;
    s.vel[1] = velY;
    s.vel[2] = 0.0;
    s.radius = rClamped;
    s.impactRadius = rClamped;
    s.impactVelY = velY;
    s.aerialVolume = vol;
    s.initialCoalesceVolume = vol;
    s.ageInPhase = 0.0;
    s.coalesceDuration = clamp(BASE_COALESCE_DURATION * (rClamped / INITIAL_DROP_RADIUS), 0.48, 1.54);
    s.oscBoost = 0.024;
    s.oscillation = evalRayleighOscillation(0.0, rClamped, seed, s.oscBoost);
    s.oscAtContact = s.oscillation;
    s.volDrainRate = 0.0;
    s.stepDrainedVolume = 0.0;
    s.pipettePresence = 0.0;
    s.autoReleaseVolume = 0.0;
    s.seed = Math.max(0.5, Math.abs(seed));
    this.lastSpawnSimTime = this.simTime;
    return s;
  }

  /**
   * Begins suspending a growing Young-Laplace pendant drop anchored to a capillary pipette tip
   * at y = PIPETTE_TIP_Y (0.55) above (x, z) during stationary press-and-hold.
   */
  public beginPendantDrop(offsetXZ: [number, number], seed = 1.0): DropSlotState {
    if (this.pendantSlotIndex >= 0 && this.slots[this.pendantSlotIndex].active) {
      if (this.slots[this.pendantSlotIndex].autoReleaseVolume > 0.0) {
        this.releasePendantDrop();
      } else {
        this.cancelPendantDrop();
      }
    }
    const idx = this.acquireSlotIndex();
    this.pendantSlotIndex = idx;
    const s = this.slots[idx];
    const [cx, cz] = clampDropDiskXZ(offsetXZ[0], offsetXZ[1]);
    const vol = sphereVolume(PENDANT_MIN_RADIUS);

    s.active = true;
    s.mode = 'pendant';
    s.center[0] = cx;
    s.center[1] = PIPETTE_TIP_Y - 1.04 * PENDANT_MIN_RADIUS;
    s.center[2] = cz;
    s.vel[0] = 0.0;
    s.vel[1] = 0.0;
    s.vel[2] = 0.0;
    s.radius = PENDANT_MIN_RADIUS;
    s.impactRadius = PENDANT_MIN_RADIUS;
    s.impactVelY = 0.0;
    s.aerialVolume = vol;
    s.initialCoalesceVolume = vol;
    s.ageInPhase = 0.0;
    s.coalesceDuration = clamp(
      BASE_COALESCE_DURATION * (PENDANT_MIN_RADIUS / INITIAL_DROP_RADIUS),
      0.48,
      1.54
    );
    s.oscBoost = 0.0;
    s.oscillation = 0.03; // Non-dimensional Young-Laplace Bond deformation parameter while pendant
    s.oscAtContact = 0.03;
    s.volDrainRate = 0.0;
    s.stepDrainedVolume = 0.0;
    s.pipettePresence = 1.0;
    s.autoReleaseVolume = 0.0;
    s.seed = Math.max(0.5, Math.abs(seed));
    return s;
  }

  /**
   * Begins a brisk (~0.15s) auto-squeezed pendant drop from the glass Pasteur pipette tip
   * for quick clicks and '+ Drop' triggers so drops never pop out of thin air.
   */
  public beginQuickPipetteDrop(
    offsetXZ: [number, number],
    targetRadius = INITIAL_DROP_RADIUS,
    seed = 1.0
  ): DropSlotState {
    const s = this.beginPendantDrop(offsetXZ, seed);
    const rClamped = clamp(targetRadius, 0.030, PENDANT_MAX_RADIUS);
    s.autoReleaseVolume = sphereVolume(rClamped);
    return s;
  }

  /**
   * Swells the suspended pendant drop at constant volumetric infusion rate V(t) = V_min + Q * t
   * over PENDANT_GROWTH_DURATION (~2.2s, R(t) = 0.018 -> 0.078 via R ~ t^(1/3)); automatically
   * snaps off into free-fall when the Bond-number capillary pinch-off limit (R_max = 0.078) is reached.
   * Returns true if capillary pinch-off occurred on this update.
   */
  public updatePendantDrop(dtRealSec: number): boolean {
    if (this.pendantSlotIndex < 0) return false;
    const s = this.slots[this.pendantSlotIndex];
    if (!s.active || s.mode !== 'pendant') {
      this.pendantSlotIndex = -1;
      return false;
    }
    const dt = Math.max(0.0, Number.isFinite(dtRealSec) ? dtRealSec : 0.0);
    s.ageInPhase += dt;

    const vMin = sphereVolume(PENDANT_MIN_RADIUS);
    const isQuickSqueeze = s.autoReleaseVolume > 0.0;
    const vTarget = isQuickSqueeze
      ? Math.max(vMin + 1e-8, s.autoReleaseVolume)
      : sphereVolume(PENDANT_MAX_RADIUS);
    const squeezeDuration = isQuickSqueeze ? 0.15 : PENDANT_GROWTH_DURATION;
    const volFrac = Math.min(1.0, s.ageInPhase / squeezeDuration);
    const nextVol = vMin + (vTarget - vMin) * volFrac;
    const nextR = radiusFromVolume(nextVol);

    s.radius = nextR;
    s.impactRadius = nextR;
    s.aerialVolume = nextVol;
    s.initialCoalesceVolume = nextVol;
    s.pipettePresence = 1.0;

    // Young-Laplace bulb sag & concave capillary neck elongation strictly below PIPETTE_TIP_Y (0.55)
    const bondFrac = smoothstep(0.12, 1.0, volFrac);
    const osc = 0.03 + (isQuickSqueeze ? 0.14 : 0.24) * bondFrac * bondFrac;
    s.oscillation = osc;
    s.center[1] = PIPETTE_TIP_Y - nextR * (1.01 + osc) - 0.036 * bondFrac * bondFrac;

    if (volFrac >= 1.0 - 1e-6) {
      this.releasePendantDrop();
      return true;
    }
    return false;
  }

  /**
   * Detaches the active pendant drop into free-fall (capillary neck snap excites damped Rayleigh l=2,3 recoil).
   */
  public releasePendantDrop(): DropSlotState | null {
    if (this.pendantSlotIndex < 0) return null;
    const s = this.slots[this.pendantSlotIndex];
    this.pendantSlotIndex = -1;
    if (!s.active || s.mode !== 'pendant') return null;

    const wasQuick = s.autoReleaseVolume > 0.0;
    s.autoReleaseVolume = 0.0;
    s.mode = 'aerial';
    s.vel[1] = wasQuick ? -0.068 : -0.042; // Gentle downward detachment momentum after capillary neck snap
    s.ageInPhase = 0.0;
    s.oscBoost = 0.062; // Capillary neck-snap recoil exciting l=2,3 oscillations
    s.oscillation = evalRayleighOscillation(0.0, s.radius, s.seed, s.oscBoost);
    s.coalesceDuration = clamp(
      BASE_COALESCE_DURATION * (s.radius / INITIAL_DROP_RADIUS),
      0.48,
      1.54
    );
    s.pipettePresence = 1.0;
    this.lastSpawnSimTime = this.simTime;
    return s;
  }

  /**
   * Cancels an uncommitted pendant drop cleanly when pointer drag exceeds 5 px into camera orbit.
   */
  public cancelPendantDrop(): void {
    if (this.pendantSlotIndex < 0) return;
    const s = this.slots[this.pendantSlotIndex];
    this.clearSlot(s);
    this.pendantSlotIndex = -1;
  }

  public hasActivePendantDrop(): boolean {
    return this.pendantSlotIndex >= 0 && this.slots[this.pendantSlotIndex].active;
  }

  public hasAutoSqueezePendantDrop(): boolean {
    return (
      this.pendantSlotIndex >= 0 &&
      this.slots[this.pendantSlotIndex].active &&
      this.slots[this.pendantSlotIndex].autoReleaseVolume > 0.0
    );
  }

  private advanceCoalescingSlot(s: DropSlotState, dtSub: number): void {
    if (dtSub <= 0.0) return;
    const prevVol = s.aerialVolume;
    s.ageInPhase += dtSub;
    s.pipettePresence = 0.0;
    const u = clamp(s.ageInPhase / Math.max(s.coalesceDuration, 0.08), 0.0, 1.0);
    // Smooth C2-blended volume-conserving drainage decoupled from fast aerial dome slump:
    // 62% drains across u in [0, 0.62] (~0.78s, peaking in the first 0.26s to roll a compact toroidal vortex ring)
    // while 38% drains continuously across u in [0, 1] (~1.26s) as a slender capillary stem wake without any flux cliff.
    const uCore = clamp(u / 0.62, 0.0, 1.0);
    const omCore = 1.0 - uCore;
    const fCore = omCore * omCore * omCore * (1.0 + 3.0 * uCore);
    const omWake = 1.0 - u;
    const fWake = omWake * omWake * (1.0 + 0.65 * u);
    const fVol = 0.62 * fCore + 0.38 * fWake;
    const nextVol = u >= 0.9999 ? 0.0 : s.initialCoalesceVolume * fVol;
    const dVol = Math.max(0.0, prevVol - nextVol);

    s.aerialVolume = nextVol;
    s.stepDrainedVolume += dVol;
    s.radius = nextVol > 0.0 ? radiusFromVolume(nextVol) : 0.0;

    // Crisp (~0.11-0.15s) C0/C1-continuous inertial dome plunge directly into the live FFT crater floor,
    // eliminating the slow plastic gumdrop sit on the surface while releasing the cavity to rebound naturally.
    const tauSlump = clamp(0.125 * (s.impactRadius / INITIAL_DROP_RADIUS), 0.09, 0.155);
    const sNorm = Math.min(1.0, s.ageInPhase / tauSlump);
    const slumpSmooth = smoothstep(0.0, 1.0, sNorm);
    const vImpact = Math.min(s.impactVelY, -0.18);
    s.vel[1] = vImpact * (1.0 - 0.78 * smoothstep(0.0, 1.0, u));
    s.center[1] =
      WATER_SURFACE_Y +
      s.impactRadius +
      vImpact * Math.min(s.ageInPhase, tauSlump) -
      2.22 * s.impactRadius * slumpSmooth * slumpSmooth -
      0.115 * Math.max(0.0, s.ageInPhase - tauSlump);

    s.oscillation =
      (s.oscAtContact * (1.0 - smoothstep(0.0, 0.35, sNorm)) -
        0.14 * Math.sin(sNorm * Math.PI)) *
      (1.0 - slumpSmooth);

    if (u >= 0.9999 || s.radius <= 1e-5) {
      s.active = false;
      s.mode = 'inactive';
      s.radius = 0.0;
      s.aerialVolume = 0.0;
      s.oscillation = 0.0;
    }
  }

  /**
   * Advances all active released drops by `dt` slow-motion seconds, resolving mid-air/crater
   * drop-drop coalescence and exact volume-conserving wetting-neck drainage.
   */
  public step(dt: number): MultiDropSnapshot {
    const safeDt = Number.isFinite(dt) ? Math.max(0.0, dt) : 0.0;
    this.simTime += safeDt;

    for (let i = 0; i < MAX_ACTIVE_DROPS; i++) {
      this.slots[i].stepDrainedVolume = 0.0;
      if (!this.slots[i].active) {
        this.slots[i].volDrainRate = 0.0;
      }
    }

    if (safeDt <= 0.0) {
      return this.buildSnapshot(0.0);
    }

    // 1. Mid-air & crater drop-drop coalescence: R = (R1^3 + R2^3)^(1/3), conserving volume & momentum
    for (let i = 0; i < MAX_ACTIVE_DROPS; i++) {
      const a = this.slots[i];
      if (!a.active || a.mode === 'pendant') continue;
      for (let j = i + 1; j < MAX_ACTIVE_DROPS; j++) {
        const b = this.slots[j];
        if (!b.active || b.mode === 'pendant') continue;

        const dx = b.center[0] - a.center[0];
        const dy = b.center[1] - a.center[1];
        const dz = b.center[2] - a.center[2];
        const dist = Math.hypot(dx, dy, dz);
        if (dist < 0.88 * (a.radius + b.radius)) {
          const volA = a.aerialVolume;
          const volB = b.aerialVolume;
          const volSum = volA + volB;
          if (volSum > 1e-12) {
            const wA = volA / volSum;
            const wB = volB / volSum;
            a.center[0] = a.center[0] * wA + b.center[0] * wB;
            a.center[1] = a.center[1] * wA + b.center[1] * wB;
            a.center[2] = a.center[2] * wA + b.center[2] * wB;
            a.vel[0] = a.vel[0] * wA + b.vel[0] * wB;
            a.vel[1] = a.vel[1] * wA + b.vel[1] * wB;
            a.vel[2] = a.vel[2] * wA + b.vel[2] * wB;
            a.aerialVolume = volSum;
            a.radius = radiusFromVolume(volSum);
            a.oscBoost = Math.min(0.11, a.oscBoost + 0.065);
            a.pipettePresence = Math.max(a.pipettePresence, b.pipettePresence);

            if (a.mode === 'coalescing' || b.mode === 'coalescing') {
              a.mode = 'coalescing';
              a.impactRadius = a.radius;
              a.impactVelY = Math.min(a.vel[1], -0.12);
              a.initialCoalesceVolume = volSum;
              a.ageInPhase = 0.0;
              a.coalesceDuration = clamp(
                BASE_COALESCE_DURATION * (a.radius / INITIAL_DROP_RADIUS),
                0.48,
                1.54
              );
              a.oscAtContact = a.oscillation;
              a.pipettePresence = 0.0;
            }
          }
          this.clearSlot(b);
        }
      }
    }

    // 2. Advance free-fall and exact meniscus wetting-neck drainage
    let totalStepDrained = 0.0;

    for (let i = 0; i < MAX_ACTIVE_DROPS; i++) {
      const s = this.slots[i];
      if (!s.active || s.mode === 'pendant') {
        s.volDrainRate = 0.0;
        continue;
      }

      if (s.mode === 'aerial') {
        const y0 = s.center[1];
        const v0 = s.vel[1];
        const r0 = s.radius;
        const yNext = y0 + v0 * safeDt + 0.5 * G_SLOW * safeDt * safeDt;
        const vNext = v0 + G_SLOW * safeDt;

        // Check exact sub-frame contact with meniscus y - R <= WATER_SURFACE_Y
        if (yNext - r0 <= WATER_SURFACE_Y) {
          let dtAir = 0.0;
          if (y0 - r0 > WATER_SURFACE_Y) {
            const qa = 0.5 * G_SLOW;
            const qb = v0;
            const qc = y0 - r0 - WATER_SURFACE_Y;
            const disc = Math.max(0.0, qb * qb - 4.0 * qa * qc);
            dtAir = clamp((-qb - Math.sqrt(disc)) / (2.0 * qa), 0.0, safeDt);
          }
          const ageAtContact = s.ageInPhase + dtAir;
          const vContact = v0 + G_SLOW * dtAir;
          const oscContact = evalRayleighOscillation(ageAtContact, r0, s.seed, s.oscBoost);

          s.center[0] += s.vel[0] * dtAir;
          s.center[1] = WATER_SURFACE_Y + r0;
          s.center[2] += s.vel[2] * dtAir;
          s.vel[1] = vContact;
          s.mode = 'coalescing';
          s.impactRadius = r0;
          s.impactVelY = vContact;
          s.initialCoalesceVolume = s.aerialVolume;
          s.ageInPhase = 0.0;
          s.coalesceDuration = clamp(
            BASE_COALESCE_DURATION * (r0 / INITIAL_DROP_RADIUS),
            0.48,
            1.54
          );
          s.oscillation = oscContact;
          s.oscAtContact = oscContact;
          s.pipettePresence = 0.0;

          const dtSub = safeDt - dtAir;
          this.advanceCoalescingSlot(s, dtSub);
        } else {
          s.center[0] += s.vel[0] * safeDt;
          s.center[1] = yNext;
          s.center[2] += s.vel[2] * safeDt;
          s.vel[1] = vNext;
          s.ageInPhase += safeDt;
          s.oscillation = evalRayleighOscillation(s.ageInPhase, r0, s.seed, s.oscBoost);
          s.volDrainRate = 0.0;
          if (s.pipettePresence > 0.0) {
            s.pipettePresence = Math.max(0.0, s.pipettePresence - safeDt / 0.24);
          }
        }
      } else if (s.mode === 'coalescing') {
        this.advanceCoalescingSlot(s, safeDt);
      }

      s.volDrainRate =
        safeDt > 1e-8 ? s.stepDrainedVolume / (safeDt * REFERENCE_DROP_VOLUME) : 0.0;
      totalStepDrained += s.stepDrainedVolume;
    }

    if (this.pendingRecycledVolume > 0.0 && totalStepDrained > 1e-9) {
      totalStepDrained += this.pendingRecycledVolume;
      this.pendingRecycledVolume = 0.0;
    }

    return this.buildSnapshot(totalStepDrained);
  }

  public getSnapshot(): MultiDropSnapshot {
    return this.buildSnapshot(0.0);
  }

  public getEffectiveCycleDuration(baseDuration = SHOT_CYCLE_DURATION): number {
    if (this.lastSpawnSimTime <= 0.0) {
      return baseDuration;
    }
    return Math.max(baseDuration, this.lastSpawnSimTime + 9.0);
  }

  private buildSnapshot(totalStepDrained: number): MultiDropSnapshot {
    let activeCount = 0;
    let hasPendant = false;
    let hasAerial = false;
    let hasCoalescing = false;
    let focusSlot: DropSlotState | null = null;

    for (let i = 0; i < MAX_ACTIVE_DROPS; i++) {
      const s = this.slots[i];
      if (!s.active) continue;
      activeCount++;
      if (s.mode === 'pendant') {
        hasPendant = true;
        focusSlot = s;
      } else if (s.mode === 'aerial') {
        hasAerial = true;
        if (!focusSlot) focusSlot = s;
      } else if (s.mode === 'coalescing') {
        hasCoalescing = true;
        if (!focusSlot) focusSlot = s;
      }
    }

    const dtSinceLast = this.simTime - this.lastSpawnSimTime;
    let phaseLabel = 'Aerial Descent';
    if (hasPendant) {
      phaseLabel = 'Pendant Drop Capillary Swelling';
    } else if (hasCoalescing) {
      phaseLabel = 'Capillary Wetting & Crater Excavation';
    } else if (hasAerial && this.simTime < IMPACT_TIME + 0.1) {
      phaseLabel = 'Aerial Descent';
    } else if (dtSinceLast < 3.4) {
      phaseLabel =
        this.lastSpawnSimTime > 0.2
          ? 'Multi-Drop Vortex Leapfrogging'
          : 'Hydrodynamic Vortex Roll-Up & Rebound';
    } else if (dtSinceLast < 9.0) {
      phaseLabel = 'Laminar Toroidal Plume & Casson Veil';
    } else {
      phaseLabel = 'High-Sc Tendrils & Osmotic Hemolysis';
    }

    this.snapshotObj.simTime = this.simTime;
    this.snapshotObj.activeCount = activeCount;
    this.snapshotObj.totalStepDrainedVolume = totalStepDrained;
    this.snapshotObj.lastSpawnSimTime = this.lastSpawnSimTime;
    this.snapshotObj.phaseLabel = phaseLabel;
    this.snapshotObj.primaryAerialFocus = focusSlot ? focusSlot.center : null;
    return this.snapshotObj;
  }
}
