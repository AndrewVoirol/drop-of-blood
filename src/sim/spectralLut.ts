import { SPECTRAL_LUT_HEIGHT, SPECTRAL_LUT_WIDTH } from '../shaders/common';

/**
 * Optical path scale tau_0 mapping normalized LUT coordinate u in [0, 1] -> optical mass path tau:
 *   tau(u) = SPECTRAL_TAU_HALF * u / max(1.0 - 0.996 * u, 0.004)
 * Inverse on GPU:
 *   u(tau) = tau / (SPECTRAL_TAU_HALF + 0.996 * tau)
 */
export const SPECTRAL_TAU_HALF = 0.012;

/**
 * Tabulated Scott Prahl (1999) Oregon Medical Laser Center molar extinction coefficients
 * of human Oxyhemoglobin (HbO2) and Deoxyhemoglobin (Hb) in [cm^-1 / M] across 380-720 nm.
 * Captures the Soret peak (414 nm HbO2 / 430 nm Hb), the 485 nm blue-cyan rose trough,
 * the characteristic Q-band doublet (541 nm beta, 577 nm alpha for HbO2; 555 nm single peak for Hb),
 * and the steep 600-720 nm deep crimson red window (~10x variation across the red band alone).
 */
const PRAHL_CONTROL_POINTS: ReadonlyArray<readonly [number, number, number]> = [
  // [wavelength_nm, eps_HbO2 (cm^-1/M), eps_Hb (cm^-1/M)]
  [380, 113876, 111016],
  [390, 146800, 167780],
  [400, 266232, 303924],
  [410, 466840, 390364],
  [414, 524280, 421400], // HbO2 Soret maximum
  [420, 480360, 496200],
  [430, 246072, 552160], // Deoxy-Hb Soret maximum
  [440, 102540, 412520],
  [450, 62816, 103292],
  [460, 44480, 42360],
  [470, 33209, 29156],
  [480, 26337, 25040],
  [486, 24410, 24680],  // Blue-cyan Soret/Q trough -> cool rose-carmine thin halo
  [490, 24190, 25660],
  [500, 20932, 29268],
  [510, 20035, 34770],
  [520, 24202, 41840],
  [530, 39956, 49576],
  [541, 53336, 53650],  // HbO2 Q-beta absorption peak
  [550, 43016, 55200],
  [555, 36920, 55540],  // Deoxy-Hb Q peak
  [560, 32613, 53788],  // HbO2 Q inter-peak valley
  [570, 44496, 45072],
  [577, 55540, 38120],  // HbO2 Q-alpha absorption peak (quenches yellow-orange!)
  [585, 33200, 30512],
  [590, 14400, 25473],
  [600, 3200, 14677],   // Steep red cut-on
  [610, 1506, 9443],
  [620, 942, 6509],
  [630, 610, 5148],
  [640, 445, 4345],
  [650, 368, 3750],
  [660, 319, 3226],     // Pulse-oximetry deep crimson transmission window
  [680, 277, 2407],
  [700, 290, 1794],
  [720, 348, 1326],
];

function evalPrahlExtinction(lambdaNm: number): { hbO2: number; hb: number } {
  const pts = PRAHL_CONTROL_POINTS;
  if (lambdaNm <= pts[0][0]) {
    return { hbO2: pts[0][1], hb: pts[0][2] };
  }
  const last = pts[pts.length - 1];
  if (lambdaNm >= last[0]) {
    return { hbO2: last[1], hb: last[2] };
  }
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i];
    const b = pts[i + 1];
    if (lambdaNm >= a[0] && lambdaNm <= b[0]) {
      const t = (lambdaNm - a[0]) / (b[0] - a[0]);
      // Log-linear interpolation preserves steep exponential band flanks across 585-620 nm
      const hbO2 = Math.exp(Math.log(a[1]) * (1.0 - t) + Math.log(b[1]) * t);
      const hb = Math.exp(Math.log(a[2]) * (1.0 - t) + Math.log(b[2]) * t);
      return { hbO2, hb };
    }
  }
  return { hbO2: last[1], hb: last[2] };
}

/**
 * Wyman, Sloan & Shirley (2013) piecewise-Gaussian analytic fit to the
 * CIE 1931 2-degree Standard Colorimetric Observer (RMS error < 0.0028).
 */
function asymmetricGaussian(x: number, mu: number, sigmaLeft: number, sigmaRight: number): number {
  const s = x < mu ? sigmaLeft : sigmaRight;
  const u = (x - mu) * s;
  return Math.exp(-0.5 * u * u);
}

function evalCIE1931XYZ(lambdaNm: number): [number, number, number] {
  const xBar =
    1.056 * asymmetricGaussian(lambdaNm, 599.8, 0.0264, 0.0323) +
    0.362 * asymmetricGaussian(lambdaNm, 442.0, 0.0624, 0.0374) -
    0.065 * asymmetricGaussian(lambdaNm, 501.1, 0.0490, 0.0382);
  const yBar =
    0.821 * asymmetricGaussian(lambdaNm, 568.8, 0.0213, 0.0247) +
    0.286 * asymmetricGaussian(lambdaNm, 530.9, 0.0613, 0.0322);
  const zBar =
    1.217 * asymmetricGaussian(lambdaNm, 437.0, 0.0845, 0.0278) +
    0.681 * asymmetricGaussian(lambdaNm, 459.0, 0.0385, 0.0725);
  return [Math.max(0, xBar), Math.max(0, yBar), Math.max(0, zBar)];
}

/**
 * Studio 6504K D65 daylight/xenon softbox + physical ~400 nm optical crown-glass UV roll-off:
 * Preserves hemoglobin's 414 nm Soret band (524,280 cm^-1/M) and steep 585-610 nm orange cut-on
 * so thin oxygenated paths exhibit authentic Beer-Lambert dichromatism (warm amber ~15 um ->
 * vivid vermilion/scarlet-orange ~35-60 um -> arterial ruby -> deep crimson -> venous oxblood).
 */
function evalStudioD65(lambdaNm: number): number {
  const wlMeters = lambdaNm * 1e-9;
  const c2 = 1.4388e-2;
  const T = 6504.0;
  const planck = (1.0 / (Math.pow(wlMeters, 5) * (Math.exp(c2 / (wlMeters * T)) - 1.0))) * 1e-13;
  // Physical borosilicate / optical crown-glass UV cut-on around ~400 nm
  const glassUvRollOff = 1.0 / (1.0 + Math.exp(-(lambdaNm - 400.0) / 5.0));
  return planck * glassUvRollOff;
}

/**
 * Exact IEEE-754 binary32 -> binary16 (half-float) bit packer for rgba16float GPU upload.
 */
export function float32ToFloat16Bits(val: number): number {
  if (!Number.isFinite(val)) {
    return Number.isNaN(val) ? 0x7e00 : val > 0 ? 0x7c00 : 0xfc00;
  }
  const sign = val < 0 ? 0x8000 : 0;
  const abs = Math.abs(val);
  if (abs === 0) return sign;
  if (abs >= 65504.0) return sign | 0x7bff;
  if (abs < 6.103515625e-5) {
    // Subnormal half-float
    return sign | Math.round(abs / 5.960464477539063e-8);
  }
  const exp = Math.floor(Math.log2(abs));
  const mant = abs / Math.pow(2, exp) - 1.0;
  const biasedExp = exp + 15;
  let roundedMant = Math.round(mant * 1024.0);
  let finalExp = biasedExp;
  if (roundedMant >= 1024) {
    roundedMant = 0;
    finalExp += 1;
    if (finalExp >= 31) return sign | 0x7bff;
  }
  return sign | (finalExp << 10) | roundedMant;
}

export interface SpectralHemoglobinLut {
  readonly width: number;
  readonly height: number;
  /** Packed IEEE-754 binary16 texels (width * height * 4 uint16s, 2048 bytes/row -> exact 256B WebGPU pitch) */
  readonly rgba16: Uint16Array;
  /** Exact Float32 reference values (width * height * 4 floats: [T_r, T_g, T_b, T_photopicY]) */
  readonly rgba32: Float32Array;
}

/**
 * Generates a 256 x 64 CIE-integrated Prahl (1999) HbO2/Hb spectral transmittance LUT
 * across [380 nm, 720 nm] at 2 nm quadrature resolution, mapped to linear Rec.709 / sRGB.
 *  - X axis u in [0, 1]: Non-linear hemoglobin optical path tau(u)
 *  - Y axis v in [0, 1]: Oxygenation + shear dispersion state
 *      v = 0: Low-shear venous core & settled floor rouleaux (SO2 ~ 0.56 + Duysens cellular sieve darkening)
 *      v = 1: High-shear stretched/aerated arterial HbO2 sheets (SO2 ~ 0.99, dispersed single-cell/free-heme dichromatism)
 */
export function generatePrahlSpectralLut(): SpectralHemoglobinLut {
  const width = SPECTRAL_LUT_WIDTH;
  const height = SPECTRAL_LUT_HEIGHT;
  const rgba16 = new Uint16Array(width * height * 4);
  const rgba32 = new Float32Array(width * height * 4);

  const wlStart = 380;
  const wlEnd = 720;
  const wlStep = 2;
  const numBands = Math.floor((wlEnd - wlStart) / wlStep) + 1; // 171 spectral bands

  const epsHbO2 = new Float64Array(numBands);
  const epsHb = new Float64Array(numBands);
  const weightX = new Float64Array(numBands);
  const weightY = new Float64Array(numBands);
  const weightZ = new Float64Array(numBands);

  let whiteX = 0.0;
  let whiteY = 0.0;
  let whiteZ = 0.0;

  for (let b = 0; b < numBands; b++) {
    const wl = wlStart + b * wlStep;
    const prahl = evalPrahlExtinction(wl);
    // Normalize molar extinction (1/M/cm) by whole-blood heme molarity (~0.0093 M -> 2.303 * C_heme)
    // so optical path tau ~ O(0.01..0.25) matches physical world chord integrals
    epsHbO2[b] = prahl.hbO2 * 0.0214;
    epsHb[b] = prahl.hb * 0.0214;

    const spd = evalStudioD65(wl) * wlStep;
    const [xb, yb, zb] = evalCIE1931XYZ(wl);
    weightX[b] = spd * xb;
    weightY[b] = spd * yb;
    weightZ[b] = spd * zb;
    whiteX += weightX[b];
    whiteY += weightY[b];
    whiteZ += weightZ[b];
  }

  // Exact white-point normalization to D65 (X_n, Y_n, Z_n) -> Linear sRGB (1.0, 1.0, 1.0) at tau = 0
  // Standard IEC 61966-2-1 XYZ (D65) -> Linear sRGB matrix with white-point scale adaptation
  const d65RefX = 0.95047;
  const d65RefY = 1.00000;
  const d65RefZ = 1.08883;
  const scaleX = d65RefX / Math.max(whiteX, 1e-12);
  const scaleY = d65RefY / Math.max(whiteY, 1e-12);
  const scaleZ = d65RefZ / Math.max(whiteZ, 1e-12);

  for (let iy = 0; iy < height; iy++) {
    const vOxShear = iy / (height - 1);
    // Low-shear / settled venous rouleaux (v=0): SO2 = 0.56 + Duysens cellular sieve & forward-scatter loss -> deep burgundy/oxblood
    // High-shear stretched vortex lips & water-aerated outer sheets (v=1): SO2 = 0.99 oxygenated HbO2 -> amber/vermilion thin sheets -> arterial ruby
    const sOx = 0.56 + 0.43 * vOxShear;
    const rouleauxExtinction = (1.0 - vOxShear) * 32.0;

    let prevR = 1.0;
    let prevG = 1.0;
    let prevB = 1.0;
    let prevY = 1.0;

    for (let ix = 0; ix < width; ix++) {
      const u = ix / (width - 1);
      const tau =
        ix === 0
          ? 0.0
          : ix === width - 1
            ? SPECTRAL_TAU_HALF * 260.0
            : (SPECTRAL_TAU_HALF * u) / Math.max(1.0 - 0.996 * u, 0.004);

      let sumX = 0.0;
      let sumY = 0.0;
      let sumZ = 0.0;

      for (let b = 0; b < numBands; b++) {
        const wl = wlStart + b * wlStep;
        const cellTurbidity = rouleauxExtinction * Math.pow(610.0 / wl, 0.58);
        const muA = sOx * epsHbO2[b] + (1.0 - sOx) * epsHb[b] + cellTurbidity;
        const transLambda = Math.exp(-muA * tau);
        sumX += weightX[b] * transLambda;
        sumY += weightY[b] * transLambda;
        sumZ += weightZ[b] * transLambda;
      }

      const X = sumX * scaleX;
      const Y = sumY * scaleY;
      const Z = sumZ * scaleZ;

      let linR = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
      let linG = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
      let linB = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;

      // Gamut-preserving non-negative clamp + strict monotonic transmittance with optical depth
      linR = Math.min(prevR, Math.max(0.0, Math.min(1.0, linR)));
      linG = Math.min(prevG, Math.max(0.0, Math.min(1.0, linG)));
      linB = Math.min(prevB, Math.max(0.0, Math.min(1.0, linB)));
      const lumY = Math.min(prevY, Math.max(0.0, Math.min(1.0, Y)));

      prevR = linR;
      prevG = linG;
      prevB = linB;
      prevY = lumY;

      const base = (iy * width + ix) * 4;
      rgba32[base + 0] = linR;
      rgba32[base + 1] = linG;
      rgba32[base + 2] = linB;
      rgba32[base + 3] = lumY;

      rgba16[base + 0] = float32ToFloat16Bits(linR);
      rgba16[base + 1] = float32ToFloat16Bits(linG);
      rgba16[base + 2] = float32ToFloat16Bits(linB);
      rgba16[base + 3] = float32ToFloat16Bits(lumY);
    }
  }

  return { width, height, rgba16, rgba32 };
}
