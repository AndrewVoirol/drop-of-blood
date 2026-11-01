export type Vec3 = [number, number, number];
export type Mat4 = Float32Array; // 16 floats, column-major (WGSL standard)

export interface CameraPreset {
  id: 'hero' | 'crown' | 'abyss';
  label: string;
  azimuth: number;
  elevation: number;
  distance: number;
  target: Vec3;
  fovY: number;
}

export const CAMERA_PRESETS: Record<CameraPreset['id'], CameraPreset> = {
  hero: {
    id: 'hero',
    label: 'Follow Shot',
    azimuth: 0.0,
    elevation: 0.29, // Above-meniscus macro view tracking aerial drop, capillary impact & refracted plume without TIR ceiling clone
    distance: 1.28,
    target: [0.0, 0.22, 0.0],
    fovY: (40 * Math.PI) / 180,
  },
  crown: {
    id: 'crown',
    label: 'Surface Crown',
    azimuth: 0.32,
    elevation: 0.44, // High-oblique view keeping PIPETTE_TIP_Y = 0.55 inside frustum + concentric capillary ripples
    distance: 1.48,
    target: [0.0, 0.11, 0.0],
    fovY: (40 * Math.PI) / 180,
  },
  abyss: {
    id: 'abyss',
    label: 'Sub-Meniscus',
    azimuth: -0.16,
    elevation: -0.21, // Submerged macro view framing underside liquid-mercury TIR mirror + unbroken crimson plume
    distance: 1.32,
    target: [0.0, 0.06, 0.0],
    fovY: (43 * Math.PI) / 180,
  },
};

function quinticSmoothstep(e0: number, e1: number, x: number): number {
  const t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

export class OrbitCamera {
  public azimuth: number;
  public elevation: number;
  public distance: number;
  public target: Vec3;
  public fovY: number;
  public followHeroEnabled = true;

  private targetAzimuth: number;
  private targetElevation: number;
  private targetDistance: number;
  private targetCenter: Vec3;
  private targetFovY: number;

  public readonly viewMatrix = new Float32Array(16);
  public readonly projMatrix = new Float32Array(16);
  public readonly viewProjMatrix = new Float32Array(16);
  public readonly invViewProjMatrix = new Float32Array(16);
  public readonly position: Vec3 = [0, 0, 1.28];

  constructor(initial: CameraPreset = CAMERA_PRESETS.hero) {
    this.azimuth = this.targetAzimuth = initial.azimuth;
    this.elevation = this.targetElevation = initial.elevation;
    this.distance = this.targetDistance = initial.distance;
    this.target = [...initial.target];
    this.targetCenter = [...initial.target];
    this.fovY = this.targetFovY = initial.fovY;
    this.followHeroEnabled = initial.id === 'hero';
    this.updateMatrices(16 / 9);
  }

  public applyPreset(preset: CameraPreset, instant = false): void {
    this.followHeroEnabled = preset.id === 'hero';
    this.targetAzimuth = preset.azimuth;
    this.targetElevation = preset.elevation;
    this.targetDistance = preset.distance;
    this.targetCenter = [...preset.target];
    this.targetFovY = preset.fovY;
    if (instant) {
      this.azimuth = preset.azimuth;
      this.elevation = preset.elevation;
      this.distance = preset.distance;
      this.target = [...preset.target];
      this.fovY = preset.fovY;
    }
  }

  /**
   * Evaluates smooth motion-control macro follow trajectory when in Hero Follow mode:
   *  - t in [0, 1.12s]: Close above-water macro framing of falling drop, reflection & capillary crater impact
   *  - t in [1.12s, 3.20s]: Smooth topside tilt & pullback framing capillary wavelets + descending vortex ring
   *    without dipping below WATER_SURFACE_Y (eliminating 50/50 TIR ceiling mirror & keeping PIPETTE_TIP_Y in frame)
   *  - t in [3.20s, 14.0s]: Gentle topside follow keeping both pipette orifice and deep tank plume visible
   */
  public syncHeroFollow(simTime: number, instant = false): void {
    if (!this.followHeroEnabled) return;
    const followPlume = quinticSmoothstep(1.12, 3.20, simTime);
    const deepTrack = quinticSmoothstep(3.20, 14.0, simTime);

    const elev = 0.29 * (1.0 - followPlume) + 0.35 * followPlume + 0.015 * deepTrack;
    const targY = 0.22 * (1.0 - followPlume) + 0.095 * followPlume - 0.020 * deepTrack;
    const dist = 1.28 * (1.0 - followPlume) + 1.44 * followPlume + 0.05 * deepTrack;
    const azim = 0.045 * Math.sin(Math.max(0.0, simTime - 1.12) * 0.16) * followPlume;
    const fov = ((40 + 1.0 * followPlume) * Math.PI) / 180;

    this.targetAzimuth = azim;
    this.targetElevation = elev;
    this.targetDistance = dist;
    this.targetCenter[0] = 0.0;
    this.targetCenter[1] = targY;
    this.targetCenter[2] = 0.0;
    this.targetFovY = fov;

    if (instant) {
      this.azimuth = azim;
      this.elevation = elev;
      this.distance = dist;
      this.target[0] = 0.0;
      this.target[1] = targY;
      this.target[2] = 0.0;
      this.fovY = fov;
    }
  }

  private minElevationForTarget(centerY: number, dist: number): number {
    // Keep camera above the y = -0.98 borosilicate cuvette floor (minCamY = -0.88)
    const minCamY = -0.88;
    const minSin = clamp((minCamY - centerY) / Math.max(dist, 0.25), -0.90, 0.90);
    return Math.max(-1.15, Math.asin(minSin));
  }

  public orbit(deltaAzimuth: number, deltaElevation: number): void {
    this.followHeroEnabled = false;
    this.targetAzimuth += deltaAzimuth;
    const minEl = this.minElevationForTarget(this.targetCenter[1], this.targetDistance);
    this.targetElevation = clamp(this.targetElevation + deltaElevation, minEl, 1.15);
  }

  public dolly(factor: number): void {
    this.followHeroEnabled = false;
    this.targetDistance = clamp(this.targetDistance * factor, 0.82, 2.85);
    const minEl = this.minElevationForTarget(this.targetCenter[1], this.targetDistance);
    this.targetElevation = Math.max(this.targetElevation, minEl);
  }

  public panVertical(deltaY: number): void {
    this.followHeroEnabled = false;
    this.targetCenter[1] = clamp(this.targetCenter[1] + deltaY, -0.65, 0.35);
    const minEl = this.minElevationForTarget(this.targetCenter[1], this.targetDistance);
    this.targetElevation = Math.max(this.targetElevation, minEl);
  }

  public tick(dtRealSec: number, aspect: number): void {
    const alpha = 1.0 - Math.exp(-Math.min(dtRealSec, 0.1) * 10.0);
    this.azimuth += (this.targetAzimuth - this.azimuth) * alpha;
    this.elevation += (this.targetElevation - this.elevation) * alpha;
    this.distance += (this.targetDistance - this.distance) * alpha;
    this.target[0] += (this.targetCenter[0] - this.target[0]) * alpha;
    this.target[1] += (this.targetCenter[1] - this.target[1]) * alpha;
    this.target[2] += (this.targetCenter[2] - this.target[2]) * alpha;
    this.fovY += (this.targetFovY - this.fovY) * alpha;

    this.updateMatrices(aspect);
  }

  public updateMatrices(aspect: number): void {
    const safeAspect = Number.isFinite(aspect) && aspect > 0.05 ? aspect : 16 / 9;
    const minEl = this.minElevationForTarget(this.target[1], this.distance);
    if (this.elevation < minEl) {
      this.elevation = minEl;
    }
    const cosEl = Math.cos(this.elevation);
    this.position[0] = this.target[0] + this.distance * cosEl * Math.sin(this.azimuth);
    this.position[1] = this.target[1] + this.distance * Math.sin(this.elevation);
    this.position[2] = this.target[2] + this.distance * cosEl * Math.cos(this.azimuth);

    mat4LookAt(this.viewMatrix, this.position, this.target, [0, 1, 0]);
    mat4PerspectiveZO(this.projMatrix, this.fovY, safeAspect, 0.08, 12.0);
    mat4Multiply(this.viewProjMatrix, this.projMatrix, this.viewMatrix);
    mat4Invert(this.invViewProjMatrix, this.viewProjMatrix);
  }
}

function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v;
}

/** Column-major right-handed lookAt matrix */
export function mat4LookAt(out: Mat4, eye: Vec3, center: Vec3, up: Vec3): Mat4 {
  let zx = eye[0] - center[0];
  let zy = eye[1] - center[1];
  let zz = eye[2] - center[2];
  let len = Math.hypot(zx, zy, zz);
  if (len < 1e-8) {
    zx = 0;
    zy = 0;
    zz = 1;
  } else {
    zx /= len;
    zy /= len;
    zz /= len;
  }

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  len = Math.hypot(xx, xy, xz);
  if (len < 1e-8) {
    xx = 1;
    xy = 0;
    xz = 0;
  } else {
    xx /= len;
    xy /= len;
    xz /= len;
  }

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  out[0] = xx;
  out[1] = yx;
  out[2] = zx;
  out[3] = 0;

  out[4] = xy;
  out[5] = yy;
  out[6] = zy;
  out[7] = 0;

  out[8] = xz;
  out[9] = yz;
  out[10] = zz;
  out[11] = 0;

  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  out[15] = 1;
  return out;
}

/** WebGPU [0, 1] depth range right-handed perspective matrix (column-major) */
export function mat4PerspectiveZO(
  out: Mat4,
  fovY: number,
  aspect: number,
  near: number,
  far: number
): Mat4 {
  const f = 1.0 / Math.tan(fovY * 0.5);
  const nf = 1.0 / (near - far);

  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far * nf;
  out[11] = -1.0;
  out[14] = far * near * nf;
  return out;
}

export function mat4Multiply(out: Mat4, a: Mat4, b: Mat4): Mat4 {
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

  for (let c = 0; c < 4; c++) {
    const b0 = b[c * 4 + 0];
    const b1 = b[c * 4 + 1];
    const b2 = b[c * 4 + 2];
    const b3 = b[c * 4 + 3];
    out[c * 4 + 0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[c * 4 + 1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[c * 4 + 2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[c * 4 + 3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
  }
  return out;
}

export function mat4Invert(out: Mat4, m: Mat4): Mat4 {
  const m00 = m[0], m01 = m[1], m02 = m[2], m03 = m[3];
  const m10 = m[4], m11 = m[5], m12 = m[6], m13 = m[7];
  const m20 = m[8], m21 = m[9], m22 = m[10], m23 = m[11];
  const m30 = m[12], m31 = m[13], m32 = m[14], m33 = m[15];

  const b00 = m00 * m11 - m01 * m10;
  const b01 = m00 * m12 - m02 * m10;
  const b02 = m00 * m13 - m03 * m10;
  const b03 = m01 * m12 - m02 * m11;
  const b04 = m01 * m13 - m03 * m11;
  const b05 = m02 * m13 - m03 * m12;
  const b06 = m20 * m31 - m21 * m30;
  const b07 = m20 * m32 - m22 * m30;
  const b08 = m20 * m33 - m23 * m30;
  const b09 = m21 * m32 - m22 * m31;
  const b10 = m21 * m33 - m23 * m31;
  const b11 = m22 * m33 - m23 * m32;

  let det =
    b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-12) {
    out.fill(0);
    out[0] = out[5] = out[10] = out[15] = 1;
    return out;
  }
  det = 1.0 / det;

  out[0] = (m11 * b11 - m12 * b10 + m13 * b09) * det;
  out[1] = (m02 * b10 - m01 * b11 - m03 * b09) * det;
  out[2] = (m31 * b05 - m32 * b04 + m33 * b03) * det;
  out[3] = (m22 * b04 - m21 * b05 - m23 * b03) * det;
  out[4] = (m12 * b08 - m10 * b11 - m13 * b07) * det;
  out[5] = (m00 * b11 - m02 * b08 + m03 * b07) * det;
  out[6] = (m32 * b02 - m30 * b05 - m33 * b01) * det;
  out[7] = (m20 * b05 - m22 * b02 + m23 * b01) * det;
  out[8] = (m10 * b10 - m11 * b08 + m13 * b06) * det;
  out[9] = (m01 * b08 - m00 * b10 - m03 * b06) * det;
  out[10] = (m30 * b04 - m31 * b02 + m33 * b00) * det;
  out[11] = (m21 * b02 - m20 * b04 - m23 * b00) * det;
  out[12] = (m11 * b07 - m10 * b09 - m12 * b06) * det;
  out[13] = (m00 * b09 - m01 * b07 + m02 * b06) * det;
  out[14] = (m31 * b01 - m30 * b03 - m32 * b00) * det;
  out[15] = (m20 * b03 - m21 * b01 + m22 * b00) * det;
  return out;
}

export function unprojectWorldPoint(
  invViewProj: Mat4,
  ndcX: number,
  ndcY: number,
  ndcZ: number
): Vec3 {
  const x =
    invViewProj[0] * ndcX +
    invViewProj[4] * ndcY +
    invViewProj[8] * ndcZ +
    invViewProj[12];
  const y =
    invViewProj[1] * ndcX +
    invViewProj[5] * ndcY +
    invViewProj[9] * ndcZ +
    invViewProj[13];
  const z =
    invViewProj[2] * ndcX +
    invViewProj[6] * ndcY +
    invViewProj[10] * ndcZ +
    invViewProj[14];
  const w =
    invViewProj[3] * ndcX +
    invViewProj[7] * ndcY +
    invViewProj[11] * ndcZ +
    invViewProj[15];
  const invW = Math.abs(w) > 1e-8 ? 1.0 / w : 1.0;
  return [x * invW, y * invW, z * invW];
}
