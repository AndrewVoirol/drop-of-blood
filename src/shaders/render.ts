import {
  BRICK_GRID_X,
  BRICK_GRID_Y,
  BRICK_GRID_Z,
  COMMON_WGSL,
  CONC_GRID_X,
  CONC_GRID_Y,
  CONC_GRID_Z,
  MAX_ACTIVE_DROPS,
  SPECTRAL_LUT_HEIGHT,
  SPECTRAL_LUT_WIDTH,
  WAVE_GRID_SIZE,
  WAVE_HALF_EXTENT,
} from './common';

const WAVE_WIN_INNER = (WAVE_HALF_EXTENT * 0.72).toFixed(4);
const WAVE_WIN_OUTER = (WAVE_HALF_EXTENT * 0.96).toFixed(4);
const WAVE_RAY_RADIUS = (WAVE_HALF_EXTENT * 0.98).toFixed(4);

const SHARED_OPTICAL_DROP_WGSL = /* wgsl */ `
struct SurfaceInfo {
  height: f32,
  normal: vec3<f32>,
  stain: f32,
  lap: f32,
};

struct DropHit {
  hit: bool,
  slotIdx: u32,
  tEnter: f32,
  tExit: f32,
  normal: vec3<f32>,
  neckBlend: f32, // 1.0 = pure drop body, (0..1) = concave capillary wetting collar into crater
  edgeCov: f32,   // Analytic thin-lens Circle-of-Confusion silhouette coverage in [0, 1]
};

/**
 * Samples the precomputed 256x64 CIE 1931 + Prahl (1999) HbO2/Hb spectral transmittance LUT.
 * Maps optical mass integral tau >= 0 and hemolysis ratio lysisFrac in [0, 1] to exact linear sRGB transmittance.
 */
fn sampleSpectralLut(tau: f32, lysisFrac: f32) -> vec3<f32> {
  let safeTau = max(0.0, tau);
  let uNorm = safeTau / (0.012 + 0.996 * safeTau);
  let uTex = (0.5 + clamp(uNorm, 0.0, 1.0) * ${(SPECTRAL_LUT_WIDTH - 1).toFixed(1)}) / ${SPECTRAL_LUT_WIDTH.toFixed(1)};
  let vTex = (0.5 + clamp(lysisFrac, 0.0, 1.0) * ${(SPECTRAL_LUT_HEIGHT - 1).toFixed(1)}) / ${SPECTRAL_LUT_HEIGHT.toFixed(1)};
  return textureSampleLevel(spectralLutTex, linSampler, vec2<f32>(uTex, vTex), 0.0).rgb;
}

fn evalWaveHeightRaw(xz: vec2<f32>) -> vec4<f32> {
  let rDist = length(xz);
  let window = 1.0 - smoothstep(${WAVE_WIN_INNER}, ${WAVE_WIN_OUTER}, rDist);
  if (window <= 0.0001) {
    return vec4<f32>(0.0);
  }
  let uv = clamp(worldXZToWaveUV(xz), vec2<f32>(0.002), vec2<f32>(0.998));
  let w = textureSampleLevel(waveTex, linSampler, uv, 0.0);
  return w * window;
}

/**
 * 4-tap Ruijters & Thevenaz (2012) hardware-bilinear C2 cubic B-spline 2D wave reconstruction.
 * Smoothly reconstructs fine capillary wavelets without bilinear grid creases.
 */
fn evalWaveBspline(xz: vec2<f32>) -> vec4<f32> {
  let rDist = length(xz);
  let window = 1.0 - smoothstep(${WAVE_WIN_INNER}, ${WAVE_WIN_OUTER}, rDist);
  if (window <= 0.0001) {
    return vec4<f32>(0.0);
  }
  let uv = clamp(worldXZToWaveUV(xz), vec2<f32>(0.002), vec2<f32>(0.998));
  let dims = vec2<f32>(${WAVE_GRID_SIZE.toFixed(1)}, ${WAVE_GRID_SIZE.toFixed(1)});
  let invDims = 1.0 / dims;
  let coord = uv * dims - 0.5;
  let idx = floor(coord);
  let f = coord - idx;
  let f2 = f * f;
  let f3 = f2 * f;
  let omf = vec2<f32>(1.0) - f;

  let w0 = (1.0 / 6.0) * omf * omf * omf;
  let w1 = (1.0 / 6.0) * (3.0 * f3 - 6.0 * f2 + vec2<f32>(4.0));
  let w2 = (1.0 / 6.0) * (-3.0 * f3 + 3.0 * f2 + 3.0 * f + vec2<f32>(1.0));
  let w3 = (1.0 / 6.0) * f3;

  let g0 = w0 + w1;
  let g1 = w2 + w3;
  let h0 = clamp((idx - 1.0 + (w1 / max(g0, vec2<f32>(1e-5))) + 0.5) * invDims, vec2<f32>(0.002), vec2<f32>(0.998));
  let h1 = clamp((idx + 1.0 + (w3 / max(g1, vec2<f32>(1e-5))) + 0.5) * invDims, vec2<f32>(0.002), vec2<f32>(0.998));

  let s00 = textureSampleLevel(waveTex, linSampler, vec2<f32>(h0.x, h0.y), 0.0);
  let s10 = textureSampleLevel(waveTex, linSampler, vec2<f32>(h1.x, h0.y), 0.0);
  let s0  = mix(s10, s00, g0.x);

  let s01 = textureSampleLevel(waveTex, linSampler, vec2<f32>(h0.x, h1.y), 0.0);
  let s11 = textureSampleLevel(waveTex, linSampler, vec2<f32>(h1.x, h1.y), 0.0);
  let s1  = mix(s11, s01, g0.x);

  return mix(s1, s0, g0.y) * window;
}

fn evalSurfaceHeightOnly(xz: vec2<f32>) -> f32 {
  return u.surfaceParams.x + evalWaveHeightRaw(xz).r;
}

fn evalSurfaceHeightSmooth(xz: vec2<f32>) -> f32 {
  return u.surfaceParams.x + evalWaveBspline(xz).r;
}

fn evalSurface(xz: vec2<f32>) -> SurfaceInfo {
  let baseY = u.surfaceParams.x;
  let rDist = length(xz);
  let window = 1.0 - smoothstep(${WAVE_WIN_INNER}, ${WAVE_WIN_OUTER}, rDist);
  if (window <= 0.0001) {
    var flatInfo: SurfaceInfo;
    flatInfo.height = baseY;
    flatInfo.normal = vec3<f32>(0.0, 1.0, 0.0);
    flatInfo.stain = 0.0;
    flatInfo.lap = 0.0;
    return flatInfo;
  }

  let uv = clamp(worldXZToWaveUV(xz), vec2<f32>(0.002), vec2<f32>(0.998));
  let dims = vec2<f32>(${WAVE_GRID_SIZE.toFixed(1)}, ${WAVE_GRID_SIZE.toFixed(1)});
  let invDims = 1.0 / dims;
  let coord = uv * dims - 0.5;
  let idx = floor(coord);
  let f = coord - idx;
  let f2 = f * f;
  let f3 = f2 * f;
  let omf = vec2<f32>(1.0) - f;

  // Exact C2 cubic B-spline value weights
  let w0 = (1.0 / 6.0) * omf * omf * omf;
  let w1 = (1.0 / 6.0) * (3.0 * f3 - 6.0 * f2 + vec2<f32>(4.0));
  let w2 = (1.0 / 6.0) * (-3.0 * f3 + 3.0 * f2 + 3.0 * f + vec2<f32>(1.0));
  let w3 = (1.0 / 6.0) * f3;

  let g0 = w0 + w1;
  let g1 = w2 + w3;
  let h0 = clamp((idx - 1.0 + (w1 / max(g0, vec2<f32>(1e-5))) + 0.5) * invDims, vec2<f32>(0.002), vec2<f32>(0.998));
  let h1 = clamp((idx + 1.0 + (w3 / max(g1, vec2<f32>(1e-5))) + 0.5) * invDims, vec2<f32>(0.002), vec2<f32>(0.998));

  let s00 = textureSampleLevel(waveTex, linSampler, vec2<f32>(h0.x, h0.y), 0.0);
  let s10 = textureSampleLevel(waveTex, linSampler, vec2<f32>(h1.x, h0.y), 0.0);
  let s0  = mix(s10, s00, g0.x);
  let s01 = textureSampleLevel(waveTex, linSampler, vec2<f32>(h0.x, h1.y), 0.0);
  let s11 = textureSampleLevel(waveTex, linSampler, vec2<f32>(h1.x, h1.y), 0.0);
  let wC  = mix(mix(s11, s01, g0.x), s0, g0.y) * window;

  // Exact C1 analytical cubic B-spline partial derivatives via non-negative pair grouping q(f) = 0.5 + f*(1 - f)
  let q = vec2<f32>(0.5) + f * omf;
  let dw1Abs = 0.5 * f * (vec2<f32>(4.0) - 3.0 * f);
  let dw3Pos = 0.5 * f2;
  let d0 = clamp((idx - 1.0 + (dw1Abs / q) + 0.5) * invDims, vec2<f32>(0.002), vec2<f32>(0.998));
  let d1 = clamp((idx + 1.0 + (dw3Pos / q) + 0.5) * invDims, vec2<f32>(0.002), vec2<f32>(0.998));

  let hxL = mix(
    textureSampleLevel(waveTex, linSampler, vec2<f32>(d0.x, h1.y), 0.0).r,
    textureSampleLevel(waveTex, linSampler, vec2<f32>(d0.x, h0.y), 0.0).r,
    g0.y
  );
  let hxR = mix(
    textureSampleLevel(waveTex, linSampler, vec2<f32>(d1.x, h1.y), 0.0).r,
    textureSampleLevel(waveTex, linSampler, vec2<f32>(d1.x, h0.y), 0.0).r,
    g0.y
  );
  let hzL = mix(
    textureSampleLevel(waveTex, linSampler, vec2<f32>(h1.x, d0.y), 0.0).r,
    textureSampleLevel(waveTex, linSampler, vec2<f32>(h0.x, d0.y), 0.0).r,
    g0.x
  );
  let hzR = mix(
    textureSampleLevel(waveTex, linSampler, vec2<f32>(h1.x, d1.y), 0.0).r,
    textureSampleLevel(waveTex, linSampler, vec2<f32>(h0.x, d1.y), 0.0).r,
    g0.x
  );

  let scaleDeriv = (dims.x / ${(WAVE_HALF_EXTENT * 2.0).toFixed(4)}) * window;
  let dhdx = q.x * (hxR - hxL) * scaleDeriv;
  let dhdz = q.y * (hzR - hzL) * scaleDeriv;

  var info: SurfaceInfo;
  info.height = baseY + wC.r;
  info.normal = normalize(vec3<f32>(-dhdx, 1.0, -dhdz));
  info.stain = wC.a;
  info.lap = wC.b;
  return info;
}

fn fresnelDielectric(cosThetaI_in: f32, etaI: f32, etaT: f32) -> f32 {
  let cosThetaI = clamp(cosThetaI_in, 0.0, 1.0);
  let sinThetaI2 = max(0.0, 1.0 - cosThetaI * cosThetaI);
  let eta = etaI / etaT;
  let sinThetaT2 = eta * eta * sinThetaI2;
  if (sinThetaT2 >= 1.0) {
    return 1.0; // Exact Total Internal Reflection
  }
  let cosThetaT = sqrt(max(0.0, 1.0 - sinThetaT2));
  let rs = (etaI * cosThetaI - etaT * cosThetaT) / max(etaI * cosThetaI + etaT * cosThetaT, 1e-5);
  let rp = (etaT * cosThetaI - etaI * cosThetaT) / max(etaT * cosThetaI + etaI * cosThetaT, 1e-5);
  return 0.5 * (rs * rs + rp * rp);
}

fn henyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
  let g2 = g * g;
  let denom = pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5);
  return (1.0 - g2) / (4.0 * 3.14159265 * denom);
}

fn evalStudioBackdrop(rd: vec3<f32>, inWater: bool) -> vec3<f32> {
  let darkMode = u.opticalParams.w > 0.5;
  let camPos = u.cameraPosAndTime.xyz;
  let viewAxis = normalize(vec3<f32>(0.0, -0.04, 0.0) - camPos);
  let centerAlign = clamp(dot(rd, viewAxis), 0.0, 1.0);

  let keyDir = normalize(vec3<f32>(0.24, 0.84, 0.44));

  if (darkMode) {
    let baseDark = vec3<f32>(0.0004, 0.0007, 0.0013);
    let centerGlow = vec3<f32>(0.0022, 0.0036, 0.0062) * pow(centerAlign, 4.2);
    let vertGrad = smoothstep(-0.65, 0.45, rd.y) * vec3<f32>(0.0008, 0.0014, 0.0024);
    return baseDark + centerGlow + vertGrad;
  }

  let brightSoftbox = vec3<f32>(0.865, 0.916, 0.954);
  let outerCyclorama = vec3<f32>(0.36, 0.465, 0.555);
  let vignette = pow(centerAlign, 2.1);
  var studio = mix(outerCyclorama, brightSoftbox, vignette);

  if (inWater) {
    let depthTint = mix(vec3<f32>(0.72, 0.83, 0.91), vec3<f32>(0.985, 0.996, 1.00), smoothstep(-0.52, 0.18, rd.y));
    studio = studio * depthTint;
  } else {
    let airGrad = mix(vec3<f32>(0.64, 0.735, 0.82), vec3<f32>(0.965, 0.986, 1.005), smoothstep(-0.16, 0.38, rd.y));
    let overheadGlow = vec3<f32>(0.135, 0.145, 0.16) * smoothstep(0.18, 0.68, rd.y) * pow(max(dot(rd, keyDir), 0.0), 3.8);
    studio = mix(studio, airGrad, 0.58) + overheadGlow;
  }

  return studio;
}

fn evalDropCenterSlot(slot: DropGpuSlot) -> vec3<f32> {
  var center = slot.posAndRadius.xyz;
  let rBase = max(slot.posAndRadius.w, 1e-4);
  if (slot.velOscAndDrain.w > 0.0 && slot.velOscAndDrain.z > 0.0005) {
    let nearMeniscus = 1.0 - smoothstep(-rBase * 1.15, rBase * 1.05, center.y - u.surfaceParams.x);
    let drainWeight = smoothstep(0.0, 0.18, slot.velOscAndDrain.z) * nearMeniscus;
    if (drainWeight > 0.001) {
      let localCraterDelta = clamp(evalWaveHeightRaw(center.xz).r, -0.095, 0.045);
      center.y = center.y + 0.92 * localCraterDelta * drainWeight;
    }
  }
  return center;
}

fn evalDropRadiiSlot(slot: DropGpuSlot) -> vec3<f32> {
  let rBase = max(slot.posAndRadius.w, 1e-4);
  let osc = slot.velOscAndDrain.y;
  return max(
    vec3<f32>(
      rBase * (1.0 - 0.5 * osc),
      rBase * (1.0 + osc),
      rBase * (1.0 - 0.5 * osc)
    ),
    vec3<f32>(1e-4)
  );
}

fn evalDropBodySDF(p: vec3<f32>, center: vec3<f32>, radii: vec3<f32>, isPendant: bool) -> f32 {
  let rel = p - center;
  if (isPendant) {
    // Young-Laplace capillary pendant drop: sagging pear-shaped bulb + concave narrowing neck anchored to pipette tip y = 0.546
    let yTip = 0.546;
    let hToTip = max(0.014, yTip - center.y);
    let yNorm = clamp(rel.y / radii.y, -1.35, 1.85);
    let bellyBulge = 1.0 + 0.06 * (1.0 - smoothstep(-0.75, 0.15, yNorm)) - 0.24 * smoothstep(-0.10, 0.96, yNorm);
    let qBulb = vec3<f32>(rel.x / (radii.x * bellyBulge), rel.y / radii.y, rel.z / (radii.z * bellyBulge));
    let dBulb = (length(qBulb) - 1.0) * min(radii.x, radii.y) * 0.84;

    let neckTop = yTip + 0.0014 - center.y;
    let neckBot = 0.14 * radii.y;
    let neckSpan = max(0.008, neckTop - neckBot);
    let uNeck = clamp((rel.y - neckBot) / neckSpan, 0.0, 1.0);
    let pinchStage = smoothstep(0.045, 0.132, hToTip);
    let rOrifice = 0.0068;
    let rWaist = mix(min(rOrifice * 0.90, radii.x * 0.62), 0.0028, pinchStage);
    let waistShape = 4.0 * uNeck * (1.0 - uNeck);
    let rProfile = mix(
      mix(radii.x * 0.64, rOrifice, smoothstep(0.0, 1.0, uNeck)),
      rWaist,
      waistShape * (0.48 + 0.52 * pinchStage)
    );
    let rXZ = length(rel.xz);
    let dNeck = max(rXZ - rProfile, max(rel.y - neckTop, neckBot - rel.y));

    let kFillet = clamp(radii.x * 0.34, 0.005, 0.022);
    let hU = clamp(0.5 + 0.5 * (dNeck - dBulb) / kFillet, 0.0, 1.0);
    return (mix(dNeck, dBulb, hU) - kFillet * hU * (1.0 - hU)) * 0.85;
  }
  let q = rel / radii;
  let rLen = length(q);
  // Axisymmetric Legendre P_3(cos theta) capillary pinch-off recoil asymmetry (zero polar harmonics!)
  let uy = clamp(q.y / max(rLen, 1e-4), -1.0, 1.0);
  let p3 = 0.5 * (5.0 * uy * uy * uy - 3.0 * uy);
  let oscShape = clamp((radii.y - radii.x) / max(radii.x, 1e-4), -0.35, 0.35);
  return (rLen - 1.0 - 0.085 * oscShape * p3) * min(radii.x, radii.y);
}

/**
 * Evaluates smooth concave capillary wetting-neck union SDF between a coalescing drop and live crater eta(x, z):
 *   returns vec2(dUnion, hDropFraction) where hDropFraction in [0, 1] is 1 on drop dome and 0 on outer water.
 */
fn evalWettingNeckUnion(p: vec3<f32>, center: vec3<f32>, radii: vec3<f32>, kNeck: f32) -> vec2<f32> {
  let dDrop = evalDropBodySDF(p, center, radii, false);
  let dWater = p.y - evalSurfaceHeightOnly(p.xz);
  let h = clamp(0.5 + 0.5 * (dWater - dDrop) / max(kNeck, 1e-4), 0.0, 1.0);
  let dUnion = mix(dWater, dDrop, h) - kNeck * h * (1.0 - h);
  return vec2<f32>(dUnion, h);
}

fn evalWettingNeckSmooth(p: vec3<f32>, center: vec3<f32>, radii: vec3<f32>, kNeck: f32) -> vec2<f32> {
  let dDrop = evalDropBodySDF(p, center, radii, false);
  let dWater = p.y - evalSurfaceHeightSmooth(p.xz);
  let h = clamp(0.5 + 0.5 * (dWater - dDrop) / max(kNeck, 1e-4), 0.0, 1.0);
  let dUnion = mix(dWater, dDrop, h) - kNeck * h * (1.0 - h);
  return vec2<f32>(dUnion, h);
}

/**
 * Raycasts all 8 concurrent drop slots with native Retina pixel footprint & thin-lens Circle-of-Confusion coverage:
 *  - High free-falling drops use exact closed-form ellipsoid intersection + optical CoC penumbra
 *  - Suspended pendant drops use capillary teardrop SDF intersection + optical CoC penumbra
 *  - Meniscus-contacting drops use conservative concave capillary wetting-neck SDF coalescence
 */
fn intersectAerialDropFrom(ro: vec3<f32>, rd: vec3<f32>, tMinRay: f32, tFocus: f32, apRadius: f32) -> DropHit {
  var best: DropHit;
  best.hit = false;
  best.slotIdx = 0u;
  best.tEnter = 1e9;
  best.tExit = 0.0;
  best.normal = vec3<f32>(0.0, 1.0, 0.0);
  best.neckBlend = 1.0;
  best.edgeCov = 1.0;

  if (u.surfaceParams.z < 0.5) {
    return best;
  }

  let kNeckBase = max(0.008, u.particleParams.w);
  let maxCoCPad = apRadius * 2.2;
  let pixRad = 0.72 / max(u.viewportAndDt.y, 600.0);

  for (var d = 0u; d < ${MAX_ACTIVE_DROPS}u; d = d + 1u) {
    let slot = u.drops[d];
    let rBase = slot.posAndRadius.w;
    if (rBase <= 0.0005) {
      continue;
    }

    let isPendant = slot.velOscAndDrain.w < 0.0;
    let pendantNeckPad = select(0.0, max(0.025, 0.558 - slot.posAndRadius.y), isPendant);

    // Fast conservative bounding-sphere cull skips 98.5%+ of viewport rays before any wave texture fetch
    let fastR = max(rBase * 1.38 + kNeckBase * 1.45, pendantNeckPad) + maxCoCPad;
    let ocFast = ro - slot.posAndRadius.xyz;
    let bFast = dot(ocFast, rd);
    let cFast = dot(ocFast, ocFast) - fastR * fastR;
    if (bFast * bFast - cFast <= 0.0) {
      continue;
    }

    let center = evalDropCenterSlot(slot);
    let radii = evalDropRadiiSlot(slot);
    let rMax = max(radii.x, radii.y);
    let surfCenter = evalSurfaceHeightSmooth(center.xz);
    let domeAboveFade = select(smoothstep(-0.004, 0.008, center.y + radii.y - surfCenter), 1.0, isPendant);
    if (domeAboveFade <= 0.001) {
      continue;
    }
    let contactGap = (center.y - radii.y) - surfCenter;
    let contactRamp = 1.0 - smoothstep(-0.009, 0.0012, contactGap);
    let kNeck = kNeckBase * clamp(rBase / 0.048, 0.35, 1.15) * contactRamp;
    let nearMeniscus = !isPendant && (contactRamp > 0.002);

    let tClose = max(0.02, dot(center - ro, rd));
    let focalRef = max(u.lensAndRheology.x, 0.25);
    let rPixDrop = tClose * pixRad;
    let rCoCDrop = max(rPixDrop, apRadius * abs(tClose - tFocus) / focalRef);

    if (!isPendant && !nearMeniscus) {
      // Closed-form ellipsoid intersection + analytic thin-lens CoC silhouette penumbra
      let radiiPen = radii + vec3<f32>(rCoCDrop * 1.15);
      let invRPen = 1.0 / radiiPen;
      let ocPen = (ro - center) * invRPen;
      let dirPen = rd * invRPen;
      let aPen = dot(dirPen, dirPen);
      let bPen = dot(ocPen, dirPen);
      let cPen = dot(ocPen, ocPen) - 1.0;
      if (bPen * bPen - aPen * cPen <= 0.0) {
        continue;
      }

      let invR = 1.0 / radii;
      let oc = (ro - center) * invR;
      let dir = rd * invR;
      let a = dot(dir, dir);
      let b = dot(oc, dir);
      let c = dot(oc, oc) - 1.0;
      let disc = b * b - a * c;

      let tClosest = max(tMinRay, -b / max(a, 1e-6));
      if (-b / max(a, 1e-6) < tMinRay) {
        continue;
      }
      let qClose = oc + dir * tClosest;
      let dMin = (length(qClose) - 1.0) * min(radii.x, radii.y);
      let cov = 1.0 - smoothstep(-rCoCDrop * 1.15, rCoCDrop * 1.15, dMin);
      if (cov <= 0.004) {
        continue;
      }

      let rimChordMin = 0.38 * min(radii.x, radii.y);
      var tHit = tClosest;
      var t1 = tClosest + rimChordMin;
      var normLocal = qClose / max(length(qClose), 1e-5);
      if (disc > 0.0) {
        let sDisc = sqrt(disc);
        let t0 = (-b - sDisc) / a;
        let tExitExact = (-b + sDisc) / a;
        if (tExitExact <= tMinRay || t0 < tMinRay - 0.004) {
          continue;
        }
        tHit = max(t0, tMinRay);
        let rimFade = 1.0 - smoothstep(0.0, 0.16, (tExitExact - tHit) / max(min(radii.x, radii.y), 1e-4));
        t1 = max(tExitExact, tHit + rimChordMin * rimFade);
        normLocal = (ro + rd * tHit - center) * invR;
      }

      if (tHit < best.tEnter) {
        best.hit = true;
        best.slotIdx = d;
        best.tEnter = tHit;
        best.tExit = t1;
        best.normal = normalize(normLocal * invR);
        best.neckBlend = 1.0;
        best.edgeCov = cov * domeAboveFade;
      }
    } else {
      // Bounding sphere ray gate for pendant teardrop or concave capillary wetting-neck SDF
      let boundR = max(rMax + select(0.010 + rCoCDrop * 1.25, kNeck * 1.25 + rCoCDrop, nearMeniscus), pendantNeckPad + rCoCDrop * 1.25);
      let oc = ro - center;
      let b = dot(oc, rd);
      let c = dot(oc, oc) - boundR * boundR;
      let disc = b * b - c;
      if (disc <= 0.0) {
        continue;
      }
      let sDisc = sqrt(disc);
      let tS0 = max(tMinRay, -b - sDisc);
      let tS1 = -b + sDisc;
      if (tS1 <= tS0 || tS0 >= best.tEnter || (-b - sDisc) < tMinRay - 0.012) {
        continue;
      }

      let steps = 36;
      let dt = (tS1 - tS0) / f32(steps);
      var prevT = tS0;
      let pStart = ro + rd * prevT;
      var prevEval = select(
        vec2<f32>(evalDropBodySDF(pStart, center, radii, true), 1.0),
        evalWettingNeckUnion(pStart, center, radii, kNeck),
        nearMeniscus
      );

      var found = false;
      var hitT = tS0;
      var hitH = 1.0;
      var minD = prevEval.x;
      var minT = prevT;

      if (prevEval.x <= 0.0 && prevEval.y > 0.04) {
        found = true;
        hitT = tS0;
        hitH = prevEval.y;
      } else {
        for (var s = 1; s <= steps; s = s + 1) {
          let currT = tS0 + f32(s) * dt;
          let pCurr = ro + rd * currT;
          let currEval = select(
            vec2<f32>(evalDropBodySDF(pCurr, center, radii, true), 1.0),
            evalWettingNeckUnion(pCurr, center, radii, kNeck),
            nearMeniscus
          );
          if (currEval.x < minD) {
            minD = currEval.x;
            minT = currT;
          }

          if (prevEval.x > 0.0 && currEval.x <= 0.0) {
            var aT = prevT;
            var bT = currT;
            var hMid = currEval.y;
            for (var m = 0; m < 5; m = m + 1) {
              let midT = 0.5 * (aT + bT);
              let pMid = ro + rd * midT;
              let mEval = select(
                vec2<f32>(evalDropBodySDF(pMid, center, radii, true), 1.0),
                evalWettingNeckSmooth(pMid, center, radii, kNeck),
                nearMeniscus
              );
              hMid = mEval.y;
              if (mEval.x <= 0.0) {
                bT = midT;
              } else {
                aT = midT;
              }
            }
            let candT = 0.5 * (aT + bT);
            let pCand = ro + rd * candT;
            let surfH = evalSurfaceHeightSmooth(pCand.xz);
            // Accept drop body + concave capillary wetting collar joining smoothly into the crater
            if (hMid > 0.02 && (!nearMeniscus || pCand.y >= surfH - 0.004)) {
              found = true;
              hitT = candT;
              hitH = hMid;
              break;
            }
          }
          prevT = currT;
          prevEval = currEval;
        }
      }

      // Smooth outer optical CoC / Retina pixel-footprint penumbra for pendant and wetting-neck silhouettes
      if (!found && minD < rCoCDrop * 1.15) {
        if (!nearMeniscus) {
          found = true;
          hitT = minT;
          hitH = 1.0;
        } else {
          let pMin = ro + rd * minT;
          let mMin = evalWettingNeckSmooth(pMin, center, radii, kNeck);
          if (mMin.y > 0.06 && pMin.y >= evalSurfaceHeightSmooth(pMin.xz) - 0.003) {
            found = true;
            hitT = minT;
            hitH = mMin.y;
          }
        }
      }

      if (found && hitT < best.tEnter) {
        // Continuous exit chord across both drop dome AND capillary wetting neck column (never collapses at bare sphere base!)
        let invR = 1.0 / radii;
        let ocE = (ro - center) * invR;
        let dirE = rd * invR;
        let aE = dot(dirE, dirE);
        let bE = dot(ocE, dirE);
        let cE = dot(ocE, ocE) - 1.0;
        let discE = bE * bE - aE * cE;
        let rimChordSdf = 0.38 * min(radii.x, radii.y);
        let tExitExactE = (-bE + sqrt(max(0.0, discE))) / max(aE, 1e-5);
        let rimFadeE = 1.0 - smoothstep(0.0, 0.16, max(0.0, tExitExactE - hitT) / max(min(radii.x, radii.y), 1e-4));
        let tExitEllip = select(hitT + rimChordSdf, max(tExitExactE, hitT + rimChordSdf * rimFadeE), discE > 0.0);

        var tExitEst = tExitEllip;
        if (nearMeniscus) {
          let relXZ0 = ro.xz - center.xz;
          let aCyl = dot(rd.xz, rd.xz);
          let bCyl = dot(relXZ0, rd.xz);
          let rNeckCol = max(radii.x * 0.86, radii.x * smoothstep(0.0, 0.72, hitH) + kNeck * (1.0 - hitH) * 0.88);
          let cCyl = dot(relXZ0, relXZ0) - rNeckCol * rNeckCol;
          let discCyl = max(0.0, bCyl * bCyl - aCyl * cCyl);
          let tExitCyl = select(hitT + 2.0 * rNeckCol, max(hitT + 0.003, (-bCyl + sqrt(discCyl)) / aCyl), aCyl > 1e-5);
          let domeMix = smoothstep(0.58, 0.94, hitH);
          tExitEst = max(tExitEllip, mix(tExitCyl, max(tExitEllip, tExitCyl * 0.90), domeMix));
        }

        var cov = 1.0;
        if (!nearMeniscus) {
          let tDeep = clamp(-b, tS0, tS1);
          let tMid = clamp(0.5 * (hitT + tExitEst), tS0, tS1);
          let dDeep = min(
            minD,
            min(
              evalDropBodySDF(ro + rd * tDeep, center, radii, true),
              evalDropBodySDF(ro + rd * tMid, center, radii, true)
            )
          );
          cov = 1.0 - smoothstep(-rCoCDrop * 1.15, rCoCDrop * 1.15, dDeep);
        } else {
          let tProbe = clamp(hitT + 0.0035, tS0, tS1);
          let dDeepNeck = min(minD, evalWettingNeckSmooth(ro + rd * tProbe, center, radii, kNeck).x);
          cov = 1.0 - smoothstep(-rCoCDrop * 1.15, rCoCDrop * 1.15, dDeepNeck);
        }
        if (cov > 0.004) {
          let pHit = ro + rd * hitT;
          let eps = 0.0010;
          var grad = vec3<f32>(0.0, 1.0, 0.0);
          if (nearMeniscus) {
            let dx = evalWettingNeckSmooth(pHit + vec3<f32>(eps, 0.0, 0.0), center, radii, kNeck).x
                   - evalWettingNeckSmooth(pHit - vec3<f32>(eps, 0.0, 0.0), center, radii, kNeck).x;
            let dy = evalWettingNeckSmooth(pHit + vec3<f32>(0.0, eps, 0.0), center, radii, kNeck).x
                   - evalWettingNeckSmooth(pHit - vec3<f32>(0.0, eps, 0.0), center, radii, kNeck).x;
            let dz = evalWettingNeckSmooth(pHit + vec3<f32>(0.0, 0.0, eps), center, radii, kNeck).x
                   - evalWettingNeckSmooth(pHit - vec3<f32>(0.0, 0.0, eps), center, radii, kNeck).x;
            grad = normalize(vec3<f32>(dx, dy, dz));
          } else {
            let dx = evalDropBodySDF(pHit + vec3<f32>(eps, 0.0, 0.0), center, radii, true)
                   - evalDropBodySDF(pHit - vec3<f32>(eps, 0.0, 0.0), center, radii, true);
            let dy = evalDropBodySDF(pHit + vec3<f32>(0.0, eps, 0.0), center, radii, true)
                   - evalDropBodySDF(pHit - vec3<f32>(0.0, eps, 0.0), center, radii, true);
            let dz = evalDropBodySDF(pHit + vec3<f32>(0.0, 0.0, eps), center, radii, true)
                   - evalDropBodySDF(pHit - vec3<f32>(0.0, 0.0, eps), center, radii, true);
            grad = normalize(vec3<f32>(dx, dy, dz));
          }

          var tExitFinal = min(tS1 + kNeck, tExitEst);
          if (isPendant) {
            let rLocalXZ = length(pHit.xz - center.xz);
            let cosHit = clamp(dot(grad, -rd), 0.16, 1.0);
            let tExitNeckCyl = hitT + 2.0 * max(0.0016, rLocalXZ * cosHit);
            let inUpperNeck = smoothstep(0.35 * radii.y, 0.92 * radii.y, pHit.y - center.y);
            tExitFinal = mix(tExitFinal, tExitNeckCyl, inUpperNeck);
          }

          best.hit = true;
          best.slotIdx = d;
          best.tEnter = hitT;
          best.tExit = tExitFinal;
          best.normal = grad;
          best.neckBlend = select(1.0, smoothstep(0.02, 0.64, hitH), nearMeniscus);
          best.edgeCov = cov * domeAboveFade;
        }
      }
    }
  }

  return best;
}

fn intersectAerialDrop(ro: vec3<f32>, rd: vec3<f32>, tFocus: f32, apRadius: f32) -> DropHit {
  return intersectAerialDropFrom(ro, rd, 0.001, tFocus, apRadius);
}

fn evalDropStudioWindowSpec(
  reflSpec: vec3<f32>,
  fresnel: f32,
  cocSoften: f32,
  darkMode: bool
) -> f32 {
  let worldUp = vec3<f32>(0.0, 1.0, 0.0);

  // Primary vertical-oblique studio softbox window positioned near 100 deg from view axis so drop curvature
  // wraps the rectangular diffuser into a slender limb-hugging crescent weighted by physical dielectric Fresnel
  let boxCenter = normalize(vec3<f32>(-0.74, 0.65, 0.17));
  let boxRight = normalize(cross(worldUp, boxCenter));
  let boxUp = cross(boxCenter, boxRight);
  var primaryLum = 0.0;
  let alignMain = dot(reflSpec, boxCenter);
  if (alignMain > 0.0) {
    let uBox = dot(reflSpec, boxRight);
    let vBox = dot(reflSpec, boxUp);
    let rCorner = 0.046;
    let q = abs(vec2<f32>(uBox, vBox)) - vec2<f32>(0.132 - rCorner, 0.40 - rCorner);
    let dBox = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - rCorner;
    let feather = mix(0.046, 0.145, cocSoften);
    let windowMask = 1.0 - smoothstep(-0.016, feather, dBox);
    let lampFalloff = exp(-(uBox * uBox) * 20.0 - (vBox * vBox) * 3.6);
    primaryLum = windowMask * (10.5 + 16.5 * lampFalloff);
  }

  // Secondary opposite-limb studio fill strip card
  let fillCenter = normalize(vec3<f32>(0.80, 0.48, 0.36));
  let fillRight = normalize(cross(worldUp, fillCenter));
  let fillUp = cross(fillCenter, fillRight);
  var fillLum = 0.0;
  if (dot(reflSpec, fillCenter) > 0.0) {
    let uFill = dot(reflSpec, fillRight);
    let vFill = dot(reflSpec, fillUp);
    let rFill = 0.052;
    let qFill = abs(vec2<f32>(uFill, vFill)) - vec2<f32>(0.145 - rFill, 0.33 - rFill);
    let dFill = length(max(qFill, vec2<f32>(0.0))) + min(max(qFill.x, qFill.y), 0.0) - rFill;
    let featherFill = mix(0.062, 0.165, cocSoften);
    let fillMask = 1.0 - smoothstep(-0.018, featherFill, dFill);
    fillLum = fillMask * (4.2 + 4.8 * exp(-(uFill * uFill) * 16.0 - (vFill * vFill) * 4.2));
  }

  let specScale = select(1.0, 0.58, darkMode);
  return (primaryLum + fillLum) * fresnel * (1.0 - 0.52 * cocSoften) * specScale;
}

fn shadeAerialDrop(
  ro: vec3<f32>,
  rd: vec3<f32>,
  hit: DropHit,
  backdropBehind: vec3<f32>
) -> vec3<f32> {
  let slot = u.drops[hit.slotIdx];
  let pHit = ro + rd * hit.tEnter;
  let pExit = ro + rd * hit.tExit;
  let n = hit.normal;
  let v = -rd;
  let cosI = clamp(dot(n, v), 0.0, 1.0);

  let collarFade = smoothstep(0.02, 0.62, hit.neckBlend);
  let domeSpecGate = smoothstep(0.84, 0.994, hit.neckBlend);
  let darkMode = u.opticalParams.w > 0.5;
  let specScale = select(1.0, 0.55, darkMode);

  // Whole human blood (Hct ~ 45%) bulk refractive index n = 1.378 (grading smoothly to water n = 1.333 down wetting collar)
  let iorDrop = mix(1.333, 1.378, collarFade);
  let fresnel = fresnelDielectric(cosI, 1.0, iorDrop);
  let reflDir = reflect(rd, n);
  let envRefl = evalStudioBackdrop(reflDir, false);

  let center = evalDropCenterSlot(slot);
  let radii = evalDropRadiiSlot(slot);
  let invR = 1.0 / radii;
  let localSurfY = evalSurfaceHeightSmooth(pHit.xz);

  // Curved rectangular studio softbox strip evaluated on strictly convex drop dome normal (with concave-fillet rejection)
  // weighted by physical dielectric Fresnel (F0 ~ 0.025) — zero circular Phong sticker dot!
  let nConvex = normalize(((pHit - center) * invR) * invR);
  let isFreeDrop = slot.velOscAndDrain.w > 0.0;
  let nSpec = select(n, nConvex, isFreeDrop);
  let reflSpec = reflect(rd, nSpec);
  let convexMatch = select(1.0, smoothstep(0.955, 0.998, dot(n, nConvex)), isFreeDrop);
  let domeBaseY = select(center.y, max(center.y, localSurfY + 0.002), isFreeDrop);
  let upperDomeVis = smoothstep(-0.28 * radii.y, 0.22 * radii.y, pHit.y - domeBaseY) * convexMatch;
  let specKey = evalDropStudioWindowSpec(reflSpec, fresnel, 0.0, darkMode) * domeSpecGate * upperDomeVis;
  let rimSheen = fresnel * pow(1.0 - cosI, 2.2) * smoothstep(-0.12, 0.55, reflSpec.y) * 0.28 * domeSpecGate * upperDomeVis * specScale;

  let nExit = normalize(((pExit - center) * invR) * invR);
  let rdIn = refract(rd, n, 1.0 / iorDrop);
  var rdOut = refract(rdIn, -nExit, iorDrop);
  if (dot(rdOut, rdOut) < 0.25) {
    rdOut = reflect(rdIn, -nExit);
  }
  let refrBg = evalStudioBackdrop(rdOut, false);

  // Turbid whole-blood (Hct ~ 45%, erythrocyte Mie mu_s' ~ 2.2 mm^-1) exponential chord extinction + Prahl (1999) spectral LUT:
  // Core chord (~3-4 mm, tau' > 8) is optically opaque dark oxblood; ruby background transmission is strictly confined
  // to < 0.5 mm grazing rims, stretched capillary pinch-off necks, and the meniscus wetting collar.
  let rawChord = max(0.0, hit.tExit - hit.tEnter);
  let hCollar = max(0.0, pHit.y - localSurfY);
  let chord = mix(hCollar * 1.65, rawChord + 0.0025, smoothstep(0.04, 0.48, hit.neckBlend)) * smoothstep(0.0, 0.24, hit.neckBlend);
  let dropOxState = mix(0.86, 0.24, smoothstep(0.010, 0.058, chord));
  let turbidGate = smoothstep(0.08, 0.68, hit.neckBlend);
  let ballisticTrans = exp(-chord * 92.0 * u.opticalParams.x * turbidGate);
  let trans = sampleSpectralLut(chord * 3.85 * u.opticalParams.x, dropOxState) * ballisticTrans;

  let distAboveMeniscus = pHit.y - localSurfY;
  let studioWrap = clamp(0.46 + 0.34 * n.y - 0.22 * n.x + 0.24 * n.z, 0.0, 1.0);
  let thinChordGlow = (1.0 - smoothstep(0.006, 0.034, rawChord)) * collarFade;
  let grazingRubyRim = pow(1.0 - cosI, 2.3) * smoothstep(0.004, 0.040, max(0.0, distAboveMeniscus)) * collarFade;
  let bellyCaustic = smoothstep(-0.15, 0.72, -n.y) * smoothstep(0.18, 0.85, cosI) * collarFade;
  let oxbloodCore = mix(vec3<f32>(0.018, 0.00030, 0.00036), vec3<f32>(0.054, 0.00090, 0.00105), studioWrap);
  let rubyScatter = vec3<f32>(0.155, 0.0036, 0.0030) * clamp(0.62 * thinChordGlow + 0.52 * grazingRubyRim + 0.34 * bellyCaustic, 0.0, 1.15);
  let internalScatter = (oxbloodCore + rubyScatter) * select(1.0, 0.42, darkMode);

  // Rays entering the lower hemisphere or capillary wetting collar pass into the submerged stem (backdropBehind);
  // turbid whole-blood core blocks upside-down glass-marble sky lensing (ballisticTrans < 0.001 in core).
  let coreLensHit = smoothstep(0.55, 0.96, hit.edgeCov);
  let airExitGate = smoothstep(0.020, 0.078, distAboveMeniscus) * smoothstep(0.36, 0.88, hit.neckBlend) * coreLensHit;
  let bgThrough = mix(backdropBehind, refrBg, 0.24 * airExitGate * ballisticTrans);
  let bodyColor = bgThrough * trans + internalScatter * (1.0 - trans.g) * mix(0.28, 1.0, collarFade);
  let skyReflGate = smoothstep(-0.18, 0.22, reflDir.y);
  let effEnvRefl = envRefl * mix(0.14, 1.0, skyReflGate);
  let domeEmergence = smoothstep(0.006, 0.052, max(0.0, distAboveMeniscus));
  let edgeFresDamp = hit.edgeCov * hit.edgeCov * smoothstep(0.25, 0.92, hit.edgeCov);
  let fresnelWeight = clamp(fresnel * mix(0.10, mix(0.18, 0.62, domeEmergence), collarFade) * edgeFresDamp, 0.0, 0.52);
  let dropFinal = mix(bodyColor, effEnvRefl, fresnelWeight)
                + vec3<f32>(0.97, 0.985, 1.0) * (specKey + rimSheen) * edgeFresDamp;

  let alpha = smoothstep(0.0, 0.28, hit.neckBlend) * hit.edgeCov;
  return mix(backdropBehind, dropFinal, alpha);
}

fn evalPipetteOuterRadius(hAbove: f32) -> f32 {
  return mix(0.0110, 0.0255, smoothstep(0.0, 0.14, hAbove)) + 0.0025 * smoothstep(0.14, 0.34, hAbove);
}

/**
 * Raycasts & shades a macro borosilicate glass Pasteur pipette tip (n = 1.474, yTip = 0.546)
 * anchored above press-and-hold pendant drops (slot.velOscAndDrain.z < -0.005 -> pipettePresence in (0, 1]).
 * Returns vec4(compositedRgb, tHit) where tHit < 0 when missed.
 */
fn evalCapillaryPipette(
  ro: vec3<f32>,
  rd: vec3<f32>,
  tMaxClip: f32,
  backdropBehind: vec3<f32>
) -> vec4<f32> {
  if (u.surfaceParams.z < 0.5) {
    return vec4<f32>(backdropBehind, -1.0);
  }
  let darkMode = u.opticalParams.w > 0.5;
  let stripKeyXZ = normalize(vec2<f32>(0.96, 0.28));
  let stripRimXZ = normalize(vec2<f32>(-0.94, 0.34));

  var bestT = tMaxClip;
  var bestCol = backdropBehind;
  var hitAny = false;

  for (var d = 0u; d < ${MAX_ACTIVE_DROPS}u; d = d + 1u) {
    let slot = u.drops[d];
    let pipetteVis = clamp(-slot.velOscAndDrain.z, 0.0, 1.0);
    if (pipetteVis <= 0.005 || slot.posAndRadius.w <= 0.0005) {
      continue;
    }

    let isHeld = slot.velOscAndDrain.w < 0.0;
    let yTip = 0.546 + (1.0 - smoothstep(0.0, 1.0, pipetteVis)) * 0.135;
    let yTop = 1.250;
    let axisXZ = slot.posAndRadius.xz;
    let rBound = 0.0295;
    let rel0 = ro.xz - axisXZ;
    let aXZ = dot(rd.xz, rd.xz);
    if (aXZ < 1e-6) {
      continue;
    }
    let bXZ = dot(rel0, rd.xz);
    let cXZ = dot(rel0, rel0) - rBound * rBound;
    let discXZ = bXZ * bXZ - aXZ * cXZ;
    if (discXZ <= 0.0) {
      continue;
    }
    let sXZ = sqrt(discXZ);
    var t0 = max(0.002, (-bXZ - sXZ) / aXZ);
    var t1 = min(bestT, (-bXZ + sXZ) / aXZ);
    if (abs(rd.y) > 1e-5) {
      let ty0 = (yTip - ro.y) / rd.y;
      let ty1 = (yTop - ro.y) / rd.y;
      t0 = max(t0, min(ty0, ty1));
      t1 = min(t1, max(ty0, ty1));
    } else if (ro.y < yTip || ro.y > yTop) {
      continue;
    }
    if (t1 <= t0) {
      continue;
    }

    let steps = 16;
    let dt = (t1 - t0) / f32(steps);
    var prevT = t0;
    let pStart = ro + rd * prevT;
    let rStart = evalPipetteOuterRadius(max(0.0, pStart.y - yTip));
    var prevF = length(pStart.xz - axisXZ) - rStart;
    var foundPip = prevF <= 0.0 && pStart.y >= yTip && pStart.y <= yTop;
    var hitTPip = prevT;

    if (!foundPip) {
      for (var s = 1; s <= steps; s = s + 1) {
        let currT = t0 + f32(s) * dt;
        let pC = ro + rd * currT;
        let rC = evalPipetteOuterRadius(max(0.0, pC.y - yTip));
        let currF = length(pC.xz - axisXZ) - rC;
        if (prevF > 0.0 && currF <= 0.0) {
          var lo = prevT;
          var hi = currT;
          for (var b = 0; b < 4; b = b + 1) {
            let mid = 0.5 * (lo + hi);
            let pM = ro + rd * mid;
            let rM = evalPipetteOuterRadius(max(0.0, pM.y - yTip));
            if (length(pM.xz - axisXZ) - rM <= 0.0) {
              hi = mid;
            } else {
              lo = mid;
            }
          }
          hitTPip = 0.5 * (lo + hi);
          let pCheck = ro + rd * hitTPip;
          if (pCheck.y >= yTip - 0.0004 && pCheck.y <= yTop) {
            foundPip = true;
          }
          break;
        }
        prevT = currT;
        prevF = currF;
      }
    }

    if (foundPip && hitTPip < bestT) {
      let pHit = ro + rd * hitTPip;
      let hAbove = max(0.0, pHit.y - yTip);
      let taperS = smoothstep(0.0, 0.14, hAbove);
      let barrelS = smoothstep(0.14, 0.34, hAbove);
      let rOut = mix(0.0110, 0.0255, taperS) + 0.0025 * barrelS;
      let rBore = mix(0.0066, 0.0130, taperS) + 0.0018 * barrelS;
      let uTap = clamp(hAbove / 0.14, 0.0, 1.0);
      let uBar = clamp((hAbove - 0.14) / 0.20, 0.0, 1.0);
      let drDy = 0.0145 * (6.0 * uTap * (1.0 - uTap)) / 0.14 + 0.0025 * (6.0 * uBar * (1.0 - uBar)) / 0.20;
      let radXZ = pHit.xz - axisXZ;
      let radDir = normalize(radXZ + vec2<f32>(1e-6, 0.0));
      let nGlass = normalize(vec3<f32>(radDir.x, -drDy, radDir.y));

      let cosIGlass = clamp(dot(-rd, nGlass), 0.0, 1.0);
      let fresnelGlass = fresnelDielectric(cosIGlass, 1.0, 1.474);
      let reflGlass = reflect(rd, nGlass);

      // Exact 2D horizontal ray-to-axis distance: invariant under camera elevation tilt & off-center placement!
      let perpDistXZ = abs(rel0.x * rd.z - rel0.y * rd.x) / sqrt(aXZ);
      let lumenRatio = perpDistXZ / max(rBore, 1e-4);
      let sinIGlassXZ = clamp(perpDistXZ / max(rOut, 1e-4), 0.0, 1.0);

      // Cylindrical borosilicate refraction (n = 1.474) across solid wall vs hollow central lumen
      let rdGlassIn = refract(rd, nGlass, 1.0 / 1.474);
      let dGlassXZ = normalize(rdGlassIn.xz + vec2<f32>(1e-6, 0.0));
      let exitRadXZ = radDir - 2.0 * dot(radDir, dGlassXZ) * dGlassXZ;
      let nGlassExit = normalize(vec3<f32>(exitRadXZ.x, -drDy, exitRadXZ.y));
      var rdGlassOut = refract(rdGlassIn, -nGlassExit, 1.474);
      if (dot(rdGlassOut, rdGlassOut) < 0.25) {
        rdGlassOut = reflect(rdGlassIn, -nGlassExit);
      }
      let wallLensMix = smoothstep(0.82, 1.08, lumenRatio);
      let rdTrans = normalize(mix(rd, rdGlassOut, 0.16 + 0.84 * wallLensMix));
      let refrBackdrop = evalStudioBackdrop(rdTrans, false);

      // Short capillary liquid meniscus near tip while held; immediate flush retraction upon release
      let bloodRetract = pipetteVis * pipetteVis;
      let rMeniscNorm = clamp(lumenRatio, 0.0, 1.0);
      let hMenisc = select(0.002 * bloodRetract, 0.020 + 0.0036 * rMeniscNorm * rMeniscNorm, isHeld);
      let meniscusSoft = select(0.0010, 0.0022, isHeld);
      let bloodFill = (1.0 - smoothstep(hMenisc - meniscusSoft, hMenisc + meniscusSoft, hAbove))
                    * select(bloodRetract * 0.22, 1.0, isHeld);

      // Optical total-deviation outer glass edge bands + Bouguer inner-bore TIR contrast line + flame-polished tip lip
      let outerEdgeBand = smoothstep(0.70, 0.96, sinIGlassXZ);
      let innerWallLine = exp(-pow((lumenRatio - 0.98) / 0.125, 2.0)) * (1.0 - 0.52 * bloodFill);
      let meniscusRimLine = exp(-pow((hAbove - hMenisc) / 0.0018, 2.0)) * (1.0 - smoothstep(0.96, 1.06, lumenRatio)) * select(0.0, 1.0, isHeld);
      let tipLip = exp(-pow(hAbove / 0.0026, 2.0));

      var coreTrans: vec3<f32>;
      if (!darkMode) {
        let glassTint = mix(vec3<f32>(0.975, 0.988, 0.996), vec3<f32>(0.885, 0.932, 0.968), wallLensMix);
        let baseGlass = mix(backdropBehind, refrBackdrop, 0.58 * wallLensMix) * glassTint;
        let opticalEdgeDarken = (1.0 - 0.46 * outerEdgeBand)
                              * (1.0 - 0.34 * innerWallLine)
                              * (1.0 - 0.40 * meniscusRimLine)
                              * (1.0 - 0.22 * tipLip);
        coreTrans = baseGlass * opticalEdgeDarken;
      } else {
        let dfWallRim = vec3<f32>(0.24, 0.30, 0.38) * (0.68 * outerEdgeBand + 0.52 * innerWallLine + 0.45 * tipLip + 0.35 * meniscusRimLine);
        coreTrans = backdropBehind + dfWallRim;
      }

      if (lumenRatio < 0.995 && bloodFill > 0.005) {
        let cylDepth = sqrt(max(0.0, 1.0 - rMeniscNorm * rMeniscNorm));
        let chordLumen = 2.0 * rBore * cylDepth;
        let lumenTrans = sampleSpectralLut((chordLumen * 2.35 + 0.008) * u.opticalParams.x, mix(0.68, 0.28, cylDepth));
        let lumenBlood = (
          backdropBehind * lumenTrans +
          vec3<f32>(0.056, 0.0012, 0.0016) * select(1.0, 0.42, darkMode)
        ) * select(1.0 - 0.30 * outerEdgeBand, 1.0, darkMode);
        coreTrans = mix(coreTrans, lumenBlood, bloodFill * 0.97);
      }

      // Vertical studio strip-bank specular reflection along curved cylindrical borosilicate walls
      let reflXZ = normalize(reflGlass.xz + vec2<f32>(1e-6, 0.0));
      let keyAlignXZ = dot(reflXZ, stripKeyXZ);
      let rimAlignXZ = dot(reflXZ, stripRimXZ);
      let specPrimary = (smoothstep(0.935, 0.996, keyAlignXZ) * 0.44 + pow(max(keyAlignXZ, 0.0), 72.0) * 0.34)
                      * select(1.0, 0.72, darkMode);
      let specRim = (smoothstep(0.955, 0.998, rimAlignXZ) * 0.18 + pow(max(rimAlignXZ, 0.0), 48.0) * 0.16)
                  * select(0.75, 1.0, darkMode);
      let lipGlint = (0.14 * tipLip + 0.08 * innerWallLine) * select(1.0, 0.75, darkMode);

      let envGlass = evalStudioBackdrop(reflGlass, false);
      let fresnelBlend = select(fresnelGlass * 0.20 * (1.0 - 0.75 * outerEdgeBand), fresnelGlass * 0.55, darkMode);
      let glassBody = mix(coreTrans, envGlass, clamp(fresnelBlend, 0.0, 0.65))
                    + vec3<f32>(0.96, 0.985, 1.0) * (specPrimary + specRim + lipGlint);

      let clipHit = u.viewProj * vec4<f32>(pHit, 1.0);
      let ndcHitY = clipHit.y / max(clipHit.w, 1e-4);
      let topFade = (1.0 - smoothstep(0.88, 0.995, ndcHitY)) * (1.0 - smoothstep(1.10, 1.24, pHit.y));
      let edgeAA = smoothstep(0.0, 0.0011, rOut - perpDistXZ + 0.0009);
      let visAlpha = smoothstep(0.0, 0.28, pipetteVis) * topFade * edgeAA;
      bestCol = mix(backdropBehind, glassBody, clamp(visAlpha, 0.0, 1.0));
      bestT = hitTPip;
      hitAny = true;
    }
  }

  return vec4<f32>(bestCol, select(-1.0, bestT, hitAny));
}

fn marchSurfaceSegment(
  ro: vec3<f32>,
  rd: vec3<f32>,
  segLo: f32,
  segHi: f32,
  steps: i32,
  fromAir: bool
) -> f32 {
  if (segHi <= segLo + 1e-5) {
    return -1.0;
  }
  let dt = (segHi - segLo) / f32(steps);
  var prevT = segLo;
  let p0 = ro + rd * prevT;
  var prevF = p0.y - evalSurfaceHeightOnly(p0.xz);

  for (var i = 1; i <= steps; i = i + 1) {
    let currT = segLo + f32(i) * dt;
    let pC = ro + rd * currT;
    let currF = pC.y - evalSurfaceHeightOnly(pC.xz);

    let crossed = select(
      prevF <= 0.0 && currF >= 0.0,
      prevF >= 0.0 && currF <= 0.0,
      fromAir
    );
    if (crossed) {
      var a = prevT;
      var b = currT;
      let pa0 = ro + rd * a;
      let pb0 = ro + rd * b;
      let faSmooth = pa0.y - evalSurfaceHeightSmooth(pa0.xz);
      let fbSmooth = pb0.y - evalSurfaceHeightSmooth(pb0.xz);
      let useSmoothBracket = select(
        faSmooth <= 0.0 && fbSmooth >= 0.0,
        faSmooth >= 0.0 && fbSmooth <= 0.0,
        fromAir
      );
      var fa = select(prevF, faSmooth, useSmoothBracket);
      var fb = select(currF, fbSmooth, useSmoothBracket);
      for (var j = 0; j < 4; j = j + 1) {
        let mid = 0.5 * (a + b);
        let pM = ro + rd * mid;
        let hM = select(evalSurfaceHeightOnly(pM.xz), evalSurfaceHeightSmooth(pM.xz), useSmoothBracket);
        let fm = pM.y - hM;
        let midCross = select(fa <= 0.0 && fm >= 0.0, fa >= 0.0 && fm <= 0.0, fromAir);
        if (midCross) {
          b = mid;
          fb = fm;
        } else {
          a = mid;
          fa = fm;
        }
      }
      let denom = fb - fa;
      let frac = select(0.5, clamp(-fa / denom, 0.0, 1.0), abs(denom) > 1e-6);
      return a + (b - a) * frac;
    }

    prevT = currT;
    prevF = currF;
  }
  return -1.0;
}

fn intersectWaterSurface(ro: vec3<f32>, rd: vec3<f32>, tMin: f32, tMax: f32, fromAir: bool) -> f32 {
  let baseY = u.surfaceParams.x;
  let minH = baseY - 0.145;
  let maxH = baseY + 0.145;

  var tLo = tMin;
  var tHi = tMax;
  if (abs(rd.y) < 1e-4) {
    if (ro.y < minH || ro.y > maxH) {
      return -1.0;
    }
    tHi = min(tMax, tMin + 2.8);
  } else {
    let tA = (minH - ro.y) / rd.y;
    let tB = (maxH - ro.y) / rd.y;
    tLo = max(tMin, min(tA, tB));
    tHi = min(tMax, max(tA, tB));
    if (tHi < tLo) {
      return -1.0;
    }
  }

  let rdXZ2 = dot(rd.xz, rd.xz);
  var wLo = tLo;
  var wHi = min(tHi, tLo + 2.8);
  var checkWaves = true;

  if (rdXZ2 > 1e-6) {
    let bXZ = dot(ro.xz, rd.xz) / rdXZ2;
    let cXZ = (dot(ro.xz, ro.xz) - ${WAVE_RAY_RADIUS} * ${WAVE_RAY_RADIUS}) / rdXZ2;
    let discXZ = bXZ * bXZ - cXZ;
    if (discXZ <= 0.0) {
      checkWaves = false;
    } else {
      let sXZ = sqrt(discXZ);
      wLo = max(tLo, -bXZ - sXZ);
      wHi = min(tHi, -bXZ + sXZ);
      if (wHi <= wLo) {
        checkWaves = false;
      }
    }
  }

  if (checkWaves) {
    let horizSpan = sqrt(max(rdXZ2, 0.0)) * (wHi - wLo);
    let adaptiveSteps = clamp(i32(ceil(horizSpan / 0.0052)), 14, 56);
    let hHit = marchSurfaceSegment(ro, rd, wLo, wHi, adaptiveSteps, fromAir);
    if (hHit > 0.0) {
      return hHit;
    }
  }

  let validFlatDir = select(ro.y <= baseY && rd.y > 1e-4, ro.y >= baseY && rd.y < -1e-4, fromAir);
  if (validFlatDir) {
    let tFlat = (baseY - ro.y) / rd.y;
    if (tFlat >= tMin && tFlat <= tMax) {
      return tFlat;
    }
  }

  return -1.0;
}

/**
 * Raycasts all 8 aerial & pendant drops along topside water-reflected ray (pReflOrig, dropReflDir)
 * using exact unfolded virtual-mirror focal path remFocusRefl = tFocus - tSurf, Prahl spectral hemoglobin
 * transmittance, and inverted studio softbox specular highlight. Returns vec4(rgb, coverage).
 */
fn evalTopsideDropReflection(
  pReflOrig: vec3<f32>,
  dropReflDir: vec3<f32>,
  remFocusRefl: f32,
  apRadius: f32,
  waveSlope: f32,
  surfY: f32,
  skyReflCol: vec3<f32>
) -> vec4<f32> {
  if (u.surfaceParams.z < 0.5 || dropReflDir.y <= 0.005) {
    return vec4<f32>(skyReflCol, 0.0);
  }

  let kNeckBase = max(0.008, u.particleParams.w);
  let focalRef = max(u.lensAndRheology.x, 0.25);
  let pixRad = 0.85 / max(u.viewportAndDt.y, 600.0);
  var bestHit = false;
  var bestSlot = 0u;
  var bestT0 = 1e9;
  var bestChord = 0.0;
  var bestN = vec3<f32>(0.0, 1.0, 0.0);
  var bestNeckBlend = 1.0;
  var bestCov = 0.0;
  var bestCoC = 0.0;

  for (var d = 0u; d < ${MAX_ACTIVE_DROPS}u; d = d + 1u) {
    let slot = u.drops[d];
    let rBase = slot.posAndRadius.w;
    if (slot.velOscAndDrain.w == 0.0 || rBase <= 0.0005) {
      continue;
    }

    let isPend = slot.velOscAndDrain.w < 0.0;
    let center = evalDropCenterSlot(slot);
    let radii = evalDropRadiiSlot(slot);
    if (center.y + radii.y <= surfY + 0.002) {
      continue;
    }

    let tCl = dot(center - pReflOrig, dropReflDir);
    if (tCl <= 0.001) {
      continue;
    }

    let kNeckMax = kNeckBase * clamp(rBase / 0.048, 0.35, 1.15);
    let hClear = max(0.0, center.y - radii.y - surfY);
    let heightAtten = exp(-hClear * 26.0);
    let rCoC = apRadius * abs(tCl - remFocusRefl) / focalRef;
    let rPen = max(0.0036 + tCl * (0.012 + pixRad * 2.5), rCoC * 2.5 + 0.0048 * waveSlope);
    let boundExtra = select(0.008, kNeckMax * 1.20, !isPend && hClear < 0.004);
    let rMax = max(radii.x, radii.y) + rPen * 1.35 + boundExtra;
    let ocB = pReflOrig - center;
    let bB = dot(ocB, dropReflDir);
    let cB = dot(ocB, ocB) - rMax * rMax;
    if (bB * bB - cB <= 0.0) {
      continue;
    }

    let surfCenter = evalSurfaceHeightSmooth(center.xz);
    let contactGap = (center.y - radii.y) - surfCenter;
    let contactRamp = 1.0 - smoothstep(-0.009, 0.0012, contactGap);
    let kNeck = kNeckMax * contactRamp;
    let nearMeniscus = !isPend && (contactRamp > 0.002);

    if (!isPend && !nearMeniscus) {
      let invR = 1.0 / radii;
      let oc = (pReflOrig - center) * invR;
      let dir = dropReflDir * invR;
      let a = dot(dir, dir);
      let b = dot(oc, dir);
      let c = dot(oc, oc) - 1.0;
      let disc = b * b - a * c;

      let tClosest = max(0.002, -b / max(a, 1e-6));
      let qClose = oc + dir * tClosest;
      let dMin = (length(qClose) - 1.0) * min(radii.x, radii.y);
      let cov = 1.0 - smoothstep(-rPen * 1.25, rPen * 1.25, dMin);
      if (cov <= 0.004) {
        continue;
      }

      var t0 = tClosest;
      var exactChord = 0.0;
      if (disc > 0.0) {
        let sD = sqrt(disc);
        let tEnter = (-b - sD) / a;
        let tExit = (-b + sD) / a;
        if (tExit <= 0.002) {
          continue;
        }
        t0 = max(0.002, tEnter);
        exactChord = max(0.0, tExit - t0);
      }

      let pHit = pReflOrig + dropReflDir * t0;
      let aboveMirror = smoothstep(-0.002, 0.008, pHit.y - surfY);
      let effCov = cov * aboveMirror * heightAtten;
      if (effCov > 0.004 && t0 < bestT0) {
        let cocChord = 1.15 * sqrt(max(0.0, min(radii.x, radii.y) * max(0.0, rPen * 1.25 - dMin))) * cov;
        let localP = (pHit - center) * invR;
        bestHit = true;
        bestSlot = d;
        bestT0 = t0;
        bestChord = max(exactChord, cocChord);
        bestN = normalize(localP * invR);
        bestNeckBlend = 1.0;
        bestCov = effCov;
        bestCoC = rCoC;
      }
    } else {
      let sB = sqrt(max(0.0, bB * bB - cB));
      let tS0 = max(0.002, -bB - sB);
      let tS1 = -bB + sB;
      if (tS1 <= tS0 || tS0 >= bestT0) {
        continue;
      }
      let steps = 28;
      let dt = (tS1 - tS0) / f32(steps);
      var prevT = tS0;
      let pStart = pReflOrig + dropReflDir * prevT;
      var prevEval = select(
        vec2<f32>(evalDropBodySDF(pStart, center, radii, true), 1.0),
        evalWettingNeckSmooth(pStart, center, radii, kNeck),
        nearMeniscus
      );
      var minD = prevEval.x;
      var minT = prevT;
      var minH = prevEval.y;
      var found = false;
      var crossed = false;
      var hitT = tS0;
      var hitH = prevEval.y;

      if (prevEval.x <= 0.0 && prevEval.y > 0.05 && (!nearMeniscus || pStart.y >= surfY + 0.001)) {
        found = true;
        crossed = true;
        hitT = tS0;
        hitH = prevEval.y;
      } else {
        for (var s = 1; s <= steps; s = s + 1) {
          let currT = tS0 + f32(s) * dt;
          let pCurr = pReflOrig + dropReflDir * currT;
          let currEval = select(
            vec2<f32>(evalDropBodySDF(pCurr, center, radii, true), 1.0),
            evalWettingNeckSmooth(pCurr, center, radii, kNeck),
            nearMeniscus
          );
          if (currEval.x < minD && (!nearMeniscus || currEval.y > 0.05)) {
            minD = currEval.x;
            minT = currT;
            minH = currEval.y;
          }
          if (prevEval.x > 0.0 && currEval.x <= 0.0) {
            var aT = prevT;
            var bT = currT;
            var hMid = currEval.y;
            for (var m = 0; m < 4; m = m + 1) {
              let midT = 0.5 * (aT + bT);
              let pMid = pReflOrig + dropReflDir * midT;
              let mEval = select(
                vec2<f32>(evalDropBodySDF(pMid, center, radii, true), 1.0),
                evalWettingNeckSmooth(pMid, center, radii, kNeck),
                nearMeniscus
              );
              hMid = mEval.y;
              if (mEval.x <= 0.0) {
                bT = midT;
              } else {
                aT = midT;
              }
            }
            let candT = 0.5 * (aT + bT);
            let pCand = pReflOrig + dropReflDir * candT;
            if (hMid > 0.04 && (!nearMeniscus || pCand.y >= surfY + 0.001)) {
              found = true;
              crossed = true;
              hitT = candT;
              hitH = hMid;
              break;
            }
          }
          prevT = currT;
          prevEval = currEval;
        }
      }

      if (!found && minD < rPen * 1.25 && (!nearMeniscus || minH > 0.06)) {
        found = true;
        hitT = minT;
        hitH = minH;
      }

      if (found && hitT < bestT0) {
        // Compute true 3D exit chord through pendant teardrop or wetting-neck column (smoothly -> 0 at 3D limb!)
        var exactChord = 0.0;
        if (crossed) {
          if (!nearMeniscus) {
            let exitSteps = 10;
            let exitSpan = max(0.002, tS1 - hitT);
            let exitDt = exitSpan / f32(exitSteps);
            var ePrevT = hitT;
            var eExitT = hitT;
            for (var es = 1; es <= exitSteps; es = es + 1) {
              let eCurrT = hitT + f32(es) * exitDt;
              let eD = evalDropBodySDF(pReflOrig + dropReflDir * eCurrT, center, radii, true);
              if (eD <= 0.0) {
                eExitT = eCurrT;
              } else {
                var ea = ePrevT;
                var eb = eCurrT;
                for (var em = 0; em < 3; em = em + 1) {
                  let emid = 0.5 * (ea + eb);
                  if (evalDropBodySDF(pReflOrig + dropReflDir * emid, center, radii, true) <= 0.0) {
                    ea = emid;
                  } else {
                    eb = emid;
                  }
                }
                eExitT = 0.5 * (ea + eb);
                break;
              }
              ePrevT = eCurrT;
            }
            exactChord = max(0.0, eExitT - hitT);
          } else {
            let relXZ0 = pReflOrig.xz - center.xz;
            let aCyl = dot(dropReflDir.xz, dropReflDir.xz);
            let bCyl = dot(relXZ0, dropReflDir.xz);
            let rNeckCol = max(radii.x * 0.84, radii.x * smoothstep(0.0, 0.72, hitH) + kNeck * (1.0 - hitH) * 0.82);
            let cCyl = dot(relXZ0, relXZ0) - rNeckCol * rNeckCol;
            let discCyl = max(0.0, bCyl * bCyl - aCyl * cCyl);
            let tExitCyl = select(hitT + 1.8 * rNeckCol, max(hitT, (-bCyl + sqrt(discCyl)) / aCyl), aCyl > 1e-5);
            exactChord = max(0.0, tExitCyl - hitT);
          }
        }

        let tDeep = clamp(select(-bB, hitT + 0.5 * exactChord, crossed), tS0, tS1);
        let dDeep = min(
          minD,
          select(
            evalDropBodySDF(pReflOrig + dropReflDir * tDeep, center, radii, true),
            evalWettingNeckSmooth(pReflOrig + dropReflDir * tDeep, center, radii, kNeck).x,
            nearMeniscus
          )
        );
        let pHit = pReflOrig + dropReflDir * hitT;
        let aboveMirror = select(1.0, smoothstep(-0.002, 0.007, pHit.y - surfY), nearMeniscus);
        let rawCov = 1.0 - smoothstep(-rPen * 1.25, rPen * 1.25, dDeep);
        let cov = rawCov * aboveMirror * heightAtten;
        if (cov > 0.004) {
          let eps = 0.0011;
          var grad = vec3<f32>(0.0, 1.0, 0.0);
          if (nearMeniscus) {
            let dx = evalWettingNeckSmooth(pHit + vec3<f32>(eps, 0.0, 0.0), center, radii, kNeck).x
                   - evalWettingNeckSmooth(pHit - vec3<f32>(eps, 0.0, 0.0), center, radii, kNeck).x;
            let dy = evalWettingNeckSmooth(pHit + vec3<f32>(0.0, eps, 0.0), center, radii, kNeck).x
                   - evalWettingNeckSmooth(pHit - vec3<f32>(0.0, eps, 0.0), center, radii, kNeck).x;
            let dz = evalWettingNeckSmooth(pHit + vec3<f32>(0.0, 0.0, eps), center, radii, kNeck).x
                   - evalWettingNeckSmooth(pHit - vec3<f32>(0.0, 0.0, eps), center, radii, kNeck).x;
            grad = normalize(vec3<f32>(dx, dy, dz));
          } else {
            let dx = evalDropBodySDF(pHit + vec3<f32>(eps, 0.0, 0.0), center, radii, true)
                   - evalDropBodySDF(pHit - vec3<f32>(eps, 0.0, 0.0), center, radii, true);
            let dy = evalDropBodySDF(pHit + vec3<f32>(0.0, eps, 0.0), center, radii, true)
                   - evalDropBodySDF(pHit - vec3<f32>(0.0, eps, 0.0), center, radii, true);
            let dz = evalDropBodySDF(pHit + vec3<f32>(0.0, 0.0, eps), center, radii, true)
                   - evalDropBodySDF(pHit - vec3<f32>(0.0, 0.0, eps), center, radii, true);
            grad = normalize(vec3<f32>(dx, dy, dz));
          }
          let cocChord = 1.15 * sqrt(max(0.0, min(radii.x, radii.y) * max(0.0, rPen * 1.25 - dDeep))) * rawCov;
          bestHit = true;
          bestSlot = d;
          bestT0 = hitT;
          bestChord = max(exactChord, cocChord);
          bestN = grad;
          bestNeckBlend = select(1.0, smoothstep(0.02, 0.64, hitH), nearMeniscus);
          bestCov = cov;
          bestCoC = rCoC;
        }
      }
    }
  }

  if (!bestHit) {
    return vec4<f32>(skyReflCol, 0.0);
  }

  let slot = u.drops[bestSlot];
  let center = evalDropCenterSlot(slot);
  let radii = evalDropRadiiSlot(slot);
  let pHit = pReflOrig + dropReflDir * bestT0;
  let collarFade = smoothstep(0.02, 0.62, bestNeckBlend);
  let domeSpecGate = smoothstep(0.90, 0.996, bestNeckBlend);
  let iorDrop = mix(1.333, 1.378, collarFade);
  let cosDrop = clamp(dot(bestN, -dropReflDir), 0.0, 1.0);
  let fresnelDrop = fresnelDielectric(cosDrop, 1.0, iorDrop);
  let reflOnDrop = reflect(dropReflDir, bestN);
  let skyFace = smoothstep(-0.05, 0.38, reflOnDrop.y);
  let envOnDrop = evalStudioBackdrop(reflOnDrop, false);

  let darkMode = u.opticalParams.w > 0.5;
  let specScale = select(1.0, 0.55, darkMode);
  let invR = 1.0 / radii;
  let nConvexRefl = normalize(((pHit - center) * invR) * invR);
  let isFreeRefl = slot.velOscAndDrain.w > 0.0;
  let nSpecRefl = select(bestN, nConvexRefl, isFreeRefl);
  let reflSpecDrop = reflect(dropReflDir, nSpecRefl);
  let convexMatchRefl = select(1.0, smoothstep(0.94, 0.996, dot(bestN, nConvexRefl)), isFreeRefl);
  let upperVis = smoothstep(-0.42 * radii.y, 0.24 * radii.y, pHit.y - center.y) * domeSpecGate * convexMatchRefl;
  let cocSoften = smoothstep(0.0004, 0.0045, bestCoC + 0.0016 * waveSlope);
  let specKey = evalDropStudioWindowSpec(reflSpecDrop, fresnelDrop, cocSoften, darkMode) * upperVis;
  let rimSheen = fresnelDrop * pow(1.0 - cosDrop, 2.2) * 0.24 * skyFace * domeSpecGate * convexMatchRefl * specScale;

  // Turbid whole-blood reflection body + Prahl hemoglobin thin-limb rim + lower-hemisphere ruby belly glow (-bestN.y > 0)
  let pExit = pHit + dropReflDir * bestChord;
  let nExit = normalize(((pExit - center) * invR) * invR);
  let rdIn = refract(dropReflDir, bestN, 1.0 / iorDrop);
  var rdOut = refract(rdIn, -nExit, iorDrop);
  if (dot(rdOut, rdOut) < 0.25) {
    rdOut = reflect(rdIn, -nExit);
  }
  let refrBg = mix(skyReflCol, evalStudioBackdrop(rdOut, false), 0.22);

  let neckVis = smoothstep(0.0, 0.12, bestNeckBlend);
  let effChord = bestChord * neckVis;
  let hbReflTau = effChord * 1.65 * u.opticalParams.x;
  let ballisticRefl = exp(-effChord * 88.0 * u.opticalParams.x * collarFade);
  let trans = sampleSpectralLut(hbReflTau, 0.08) * ballisticRefl;
  let bellyCaustic = smoothstep(-0.18, 0.70, -bestN.y) * smoothstep(0.16, 0.88, cosDrop) * collarFade;
  let internalScatter = (
    vec3<f32>(0.065, 0.0008, 0.0010) +
    vec3<f32>(0.29, 0.0045, 0.0040) * bellyCaustic * 0.88
  ) * select(1.0, 0.40, darkMode);
  let bodyCol = refrBg * trans + internalScatter * (1.0 - trans.g) * mix(0.32, 1.0, collarFade);
  let dropCol = mix(bodyCol, envOnDrop, clamp(fresnelDrop * mix(0.08, 0.28, collarFade) * skyFace, 0.0, 0.28))
              + vec3<f32>(0.97, 0.985, 1.0) * (specKey + rimSheen);

  let chordNorm = clamp(effChord / max(1.65 * radii.x, 1e-4), 0.0, 1.0);
  let opticalPresence = mix(0.48, 0.94, smoothstep(0.0, 0.62, chordNorm));
  return vec4<f32>(dropCol, bestCov * neckVis * opticalPresence);
}
`;

export const RENDER_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var concTex: texture_3d<f32>;
@group(0) @binding(2) var lightTex: texture_3d<f32>;
@group(0) @binding(3) var waveTex: texture_2d<f32>;
@group(0) @binding(4) var spectralLutTex: texture_2d<f32>;
@group(0) @binding(5) var linSampler: sampler;
@group(0) @binding(6) var brickOccTex: texture_3d<f32>;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) ndc: vec2<f32>,
};

@vertex
fn vsMain(@builtin(vertex_index) vid: u32) -> VertexOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  var out: VertexOut;
  out.position = vec4<f32>(pos[vid], 0.0, 1.0);
  out.ndc = pos[vid];
  return out;
}

${SHARED_OPTICAL_DROP_WGSL}

struct PlumeResult {
  radiance: vec3<f32>,
  transmittance: vec3<f32>,
  bentDir: vec3<f32>,
  tauIntact: f32,
  tauTotal: f32,
};

fn copysignVec3(mag: vec3<f32>, sgn: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    select(-mag.x, mag.x, sgn.x >= 0.0),
    select(-mag.y, mag.y, sgn.y >= 0.0),
    select(-mag.z, mag.z, sgn.z >= 0.0)
  );
}

fn intersectBox(ro: vec3<f32>, rd: vec3<f32>, bMin: vec3<f32>, bMax: vec3<f32>) -> vec2<f32> {
  let safeRd = select(rd, copysignVec3(vec3<f32>(1e-6), rd), abs(rd) < vec3<f32>(1e-6));
  let invRd = 1.0 / safeRd;
  let t0 = (bMin - ro) * invRd;
  let t1 = (bMax - ro) * invRd;
  let tSmaller = min(t0, t1);
  let tBigger = max(t0, t1);
  let tNear = max(max(tSmaller.x, tSmaller.y), tSmaller.z);
  let tFar = min(min(tBigger.x, tBigger.y), tBigger.z);
  return vec2<f32>(tNear, tFar);
}

/**
 * Ruijters & Thevenaz (2012) 8-tap hardware-trilinear C2 cubic B-spline 3D texture reconstruction.
 * Strictly non-negative weights (w0..w3 >= 0, sum == 1) eliminate trilinear voxel Mach bands with zero ringing!
 */
fn sampleConcTricubic(uvwIn: vec3<f32>) -> vec4<f32> {
  let uvw = clamp(uvwIn, vec3<f32>(0.002), vec3<f32>(0.998));
  let dims = vec3<f32>(${CONC_GRID_X}.0, ${CONC_GRID_Y}.0, ${CONC_GRID_Z}.0);
  let invDims = 1.0 / dims;
  let coord = uvw * dims - 0.5;
  let idx = floor(coord);
  let f = coord - idx;
  let f2 = f * f;
  let f3 = f2 * f;
  let omf = vec3<f32>(1.0) - f;

  let w0 = (1.0 / 6.0) * omf * omf * omf;
  let w1 = (1.0 / 6.0) * (3.0 * f3 - 6.0 * f2 + vec3<f32>(4.0));
  let w2 = (1.0 / 6.0) * (-3.0 * f3 + 3.0 * f2 + 3.0 * f + vec3<f32>(1.0));
  let w3 = (1.0 / 6.0) * f3;

  let g0 = w0 + w1;
  let g1 = w2 + w3;
  let h0 = clamp((idx - 1.0 + (w1 / max(g0, vec3<f32>(1e-5))) + 0.5) * invDims, vec3<f32>(0.001), vec3<f32>(0.999));
  let h1 = clamp((idx + 1.0 + (w3 / max(g1, vec3<f32>(1e-5))) + 0.5) * invDims, vec3<f32>(0.001), vec3<f32>(0.999));

  let c000 = textureSampleLevel(concTex, linSampler, vec3<f32>(h0.x, h0.y, h0.z), 0.0);
  let c100 = textureSampleLevel(concTex, linSampler, vec3<f32>(h1.x, h0.y, h0.z), 0.0);
  let c00  = mix(c100, c000, g0.x);

  let c010 = textureSampleLevel(concTex, linSampler, vec3<f32>(h0.x, h1.y, h0.z), 0.0);
  let c110 = textureSampleLevel(concTex, linSampler, vec3<f32>(h1.x, h1.y, h0.z), 0.0);
  let c10  = mix(c110, c010, g0.x);
  let c0   = mix(c10, c00, g0.y);

  let c001 = textureSampleLevel(concTex, linSampler, vec3<f32>(h0.x, h0.y, h1.z), 0.0);
  let c101 = textureSampleLevel(concTex, linSampler, vec3<f32>(h1.x, h0.y, h1.z), 0.0);
  let c01  = mix(c101, c001, g0.x);

  let c011 = textureSampleLevel(concTex, linSampler, vec3<f32>(h0.x, h1.y, h1.z), 0.0);
  let c111 = textureSampleLevel(concTex, linSampler, vec3<f32>(h1.x, h1.y, h1.z), 0.0);
  let c11  = mix(c111, c011, g0.x);
  let c1   = mix(c11, c01, g0.y);

  return mix(c1, c0, g0.z);
}

/**
 * Raymarches the 192x256x192 hybrid RBC/Hemoglobin concentration field with:
 *  - True 380-720 nm CIE D65 Prahl Soret + Q-band dichromatic LUT driven by physical SO2 & shear/rouleaux state (conc.b)
 *  - Intact-RBC Mie forward scattering (g = 0.94) + deep 650-680 nm warm venous oxblood core transport
 *  - Local filament optical density preserved prior to thin-lens CoC bokeh softening (eliminating pink defocus washout)
 *  - Transverse Schlieren refractive-index ray bending grad_perp(n(c)) (n = 1.333 + 0.032 * c_total)
 */
fn marchPlume(
  ro: vec3<f32>,
  rdIn: vec3<f32>,
  tMinClamp: f32,
  tMaxClamp: f32,
  tFocus: f32,
  apRadius: f32,
  jitter: f32,
  fineQuality: bool
) -> PlumeResult {
  var out: PlumeResult;
  out.radiance = vec3<f32>(0.0);
  out.transmittance = vec3<f32>(1.0);
  out.bentDir = rdIn;
  out.tauIntact = 0.0;
  out.tauTotal = 0.0;

  let spongeHalfXZ = min(u.domainBounds.x, 0.435);
  let bMin = vec3<f32>(-spongeHalfXZ, u.domainBounds.y, -spongeHalfXZ);
  let bMax = vec3<f32>( spongeHalfXZ, u.domainBounds.z + 0.002, spongeHalfXZ);
  let boxHit = intersectBox(ro, rdIn, bMin, bMax);
  // Inset box entry by 1.5e-4 so f32 rounding at jitter -> 0 never puts initial uvw at 1.0000001 and aborts iteration 0!
  let tStart = max(boxHit.x + 1.5e-4, tMinClamp);
  let tEnd = min(boxHit.y - 1.5e-4, tMaxClamp);

  if (tEnd <= tStart) {
    return out;
  }

  let fineBase = select(0.0033, 0.0026, fineQuality);
  let coarseSkip = 0.015;
  let maxSteps = select(80, 160, fineQuality);

  let focalRef = max(u.lensAndRheology.x, 0.25);
  let effJitter = 0.5 + 0.08 * (jitter - 0.5);

  var t = tStart + effJitter * fineBase * 0.40;
  var rayPos = ro + rdIn * t;
  var rayDir = rdIn;
  var prevLitA = 0.0;
  var prevCoarseStep = 0.0;
  var prevConeD = 0.0;
  var prevHaloStep = 0.0;

  let darkMode = u.opticalParams.w > 0.5;
  let camPos = u.cameraPosAndTime.xyz;
  let topsideCam = camPos.y > u.surfaceParams.x;
  let toOrigin = vec3<f32>(0.0, -0.05, 0.0) - camPos;
  let camRightXZ = normalize(vec2<f32>(-toOrigin.z, toOrigin.x) + vec2<f32>(1e-6, 0.0));

  let keyDir = select(
    normalize(vec3<f32>(0.22, 0.95, 0.20)),
    normalize(vec3<f32>(0.64, 0.70, 0.32)),
    darkMode
  );
  let studioBackDir = normalize(vec3<f32>(toOrigin.x, 0.22, toOrigin.z));
  let chiaroRimDir  = normalize(vec3<f32>(-0.78 * camRightXZ.x + 0.38 * toOrigin.x, 0.24, -0.78 * camRightXZ.y + 0.38 * toOrigin.z));
  let backDir = select(studioBackDir, chiaroRimDir, darkMode);

  // Intact human erythrocyte Mie scattering: strongly forward-peaked g = 0.94 + weak cellular backscatter
  let cosKey = dot(rdIn, keyDir);
  let cosBack = dot(rdIn, backDir);
  let phaseKey = 0.82 * min(henyeyGreenstein(cosKey, 0.94), 5.8) + 0.18 * henyeyGreenstein(cosKey, -0.22);
  let phaseBack = 0.88 * min(henyeyGreenstein(cosBack, 0.94), 6.4) + 0.12 * henyeyGreenstein(cosBack, -0.20);

  let albedoScale = u.opticalParams.y;
  let tauScale = 1.68 * u.opticalParams.x;
  let schlierenStr = u.particleParams.x;
  let domSize = domainSize(u);
  let hasActiveDrops = u.surfaceParams.z > 0.5;
  let brickDimsF = vec3<f32>(${BRICK_GRID_X}.0, ${BRICK_GRID_Y}.0, ${BRICK_GRID_Z}.0);
  let brickMaxI = vec3<i32>(${BRICK_GRID_X - 1}, ${BRICK_GRID_Y - 1}, ${BRICK_GRID_Z - 1});

  // Orthonormal cross-ray aperture disk basis in UVW space (rotated 26.565 deg -> zero stem-axis aliasing or stipple!)
  let refUp = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(rdIn.y) > 0.92);
  let uPerpWorld = normalize(cross(rdIn, refUp));
  let vPerpWorld = cross(rdIn, uPerpWorld);
  let uLens = uPerpWorld / domSize;
  let vLens = vPerpWorld / domSize;
  let diskAxis1 =  0.894427 * uLens + 0.447214 * vLens;
  let diskAxis2 = -0.447214 * uLens + 0.894427 * vLens;

  let lightDims = vec3<f32>(96.0, 128.0, 96.0);
  let hTet = 1.62 / lightDims;
  let tetK0 = vec3<f32>( 1.0, -1.0, -1.0);
  let tetK1 = vec3<f32>(-1.0, -1.0,  1.0);
  let tetK2 = vec3<f32>(-1.0,  1.0, -1.0);
  let tetK3 = vec3<f32>( 1.0,  1.0,  1.0);
  let invTetDenom = 1.0 / (4.0 * hTet * domSize);
  let transStopThresh = select(0.018, 0.010, fineQuality);

  var tauTotal = 0.0;
  var tauLysed = 0.0;
  var tauIntact = 0.0;
  var tauOx = 0.0;

  for (var i = 0; i < maxSteps; i = i + 1) {
    if (t >= tEnd) {
      break;
    }

    let uvwRaw = worldToUVW(rayPos, u);
    if (
      uvwRaw.x < -0.002 || uvwRaw.x > 1.002 ||
      uvwRaw.z < -0.002 || uvwRaw.z > 1.002 ||
      uvwRaw.y < -0.002 || uvwRaw.y > 1.002
    ) {
      break;
    }
    var uvw = clamp(uvwRaw, vec3<f32>(0.0005), vec3<f32>(0.9995));

    // Hierarchical 24x32x24 3D occupancy brick skip:
    //   occ < 0.5 -> >80mm from any active lightTex voxel (safe 72mm leap)
    //   occ < 1.5 -> 40..80mm from any active lightTex voxel (safe 34mm leap)
    let brickCoord = clamp(vec3<i32>(floor(uvw * brickDimsF)), vec3<i32>(0), brickMaxI);
    let occ = textureLoad(brickOccTex, brickCoord, 0).r;
    if (occ < 1.5) {
      let brickSkip = select(0.034, 0.072, occ < 0.5);
      prevLitA = 0.0;
      prevCoarseStep = 0.0;
      prevConeD = 0.0;
      prevHaloStep = 0.0;
      t = t + brickSkip;
      rayPos = rayPos + rayDir * brickSkip;
      continue;
    }

    // Exact thin-lens Circle-of-Confusion radius r_CoC(t) referenced to camera focal distance
    var rCoC = apRadius * abs(t - tFocus) / focalRef;
    var cocBlur = smoothstep(0.00032, 0.0036, rCoC);

    // lit.a includes center tap + 28mm dilated tetrahedral support; linear-crossing rewind guarantees C0 phase continuity!
    var lit = textureSampleLevel(lightTex, linSampler, uvw, 0.0);
    if (lit.a < 0.000005) {
      prevLitA = lit.a;
      prevCoarseStep = coarseSkip;
      prevConeD = 0.0;
      prevHaloStep = 0.0;
      t = t + coarseSkip;
      rayPos = rayPos + rayDir * coarseSkip;
      continue;
    }
    if (prevCoarseStep > 0.0) {
      let crossFrac = clamp((0.000005 - prevLitA) / max(lit.a - prevLitA, 1e-8), 0.0, 1.0);
      let tCross = max(tStart, (t - prevCoarseStep) + crossFrac * prevCoarseStep + (effJitter - 0.5) * fineBase * 0.30);
      rayPos = rayPos + rayDir * (tCross - t);
      t = tCross;
      prevCoarseStep = 0.0;
      uvw = clamp(worldToUVW(rayPos, u), vec3<f32>(0.0005), vec3<f32>(0.9995));
      rCoC = apRadius * abs(t - tFocus) / focalRef;
      cocBlur = smoothstep(0.00032, 0.0036, rCoC);
      lit = textureSampleLevel(lightTex, linSampler, uvw, 0.0);
    }

    // Continuous 3D rCoC-scaled optical cone filter + secondary grazing TIR top-slice anti-voxel filter
    var tirGrazingSmooth = select(
      (1.0 - smoothstep(0.05, 0.22, abs(rayDir.y))) * smoothstep(u.surfaceParams.x - 0.028, u.surfaceParams.x - 0.003, rayPos.y),
      0.0,
      fineQuality
    );
    var coneDensity = lit.a;
    if ((cocBlur > 0.14 || tirGrazingSmooth > 0.04) && lit.a > 0.000008) {
      let rWorldCone = max(clamp(rCoC * 1.55, 0.0009, 0.022), tirGrazingSmooth * 0.012);
      let rayStepUVW = (rayDir / domSize) * min(rWorldCone * 0.50, 0.0038);
      let l1 = textureSampleLevel(lightTex, linSampler, clamp(uvw + diskAxis1 * rWorldCone + rayStepUVW, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0);
      let l2 = textureSampleLevel(lightTex, linSampler, clamp(uvw + diskAxis2 * rWorldCone - rayStepUVW, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0);
      let l3 = textureSampleLevel(lightTex, linSampler, clamp(uvw - diskAxis1 * rWorldCone - rayStepUVW, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0);
      let l4 = textureSampleLevel(lightTex, linSampler, clamp(uvw - diskAxis2 * rWorldCone + rayStepUVW, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0);
      let litAvg = 0.20 * (lit + l1 + l2 + l3 + l4);
      lit = mix(lit, litAvg, max(smoothstep(0.14, 0.68, cocBlur), tirGrazingSmooth * 0.92));
      coneDensity = lit.a;
    }
    if (coneDensity < 0.000034) {
      let skipHalo = fineBase * 2.15;
      prevConeD = coneDensity;
      prevHaloStep = skipHalo;
      t = t + skipHalo;
      rayPos = rayPos + rayDir * skipHalo;
      continue;
    }
    if (prevHaloStep > 0.0) {
      let haloFrac = clamp((0.000034 - prevConeD) / max(coneDensity - prevConeD, 1e-8), 0.0, 1.0);
      let tHaloCross = max(tStart, (t - prevHaloStep) + haloFrac * prevHaloStep + (effJitter - 0.5) * fineBase * 0.25);
      rayPos = rayPos + rayDir * (tHaloCross - t);
      t = tHaloCross;
      prevHaloStep = 0.0;
      uvw = clamp(worldToUVW(rayPos, u), vec3<f32>(0.0005), vec3<f32>(0.9995));
      rCoC = apRadius * abs(t - tFocus) / focalRef;
      cocBlur = smoothstep(0.00032, 0.0036, rCoC);
      tirGrazingSmooth = select(
        (1.0 - smoothstep(0.05, 0.22, abs(rayDir.y))) * smoothstep(u.surfaceParams.x - 0.028, u.surfaceParams.x - 0.003, rayPos.y),
        0.0,
        fineQuality
      );
      lit = textureSampleLevel(lightTex, linSampler, uvw, 0.0);
      coneDensity = lit.a;
      if ((cocBlur > 0.14 || tirGrazingSmooth > 0.04) && lit.a > 0.000008) {
        let rWorldCone = max(clamp(rCoC * 1.55, 0.0009, 0.022), tirGrazingSmooth * 0.012);
        let rayStepUVW = (rayDir / domSize) * min(rWorldCone * 0.50, 0.0038);
        let l1 = textureSampleLevel(lightTex, linSampler, clamp(uvw + diskAxis1 * rWorldCone + rayStepUVW, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0);
        let l2 = textureSampleLevel(lightTex, linSampler, clamp(uvw + diskAxis2 * rWorldCone - rayStepUVW, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0);
        let l3 = textureSampleLevel(lightTex, linSampler, clamp(uvw - diskAxis1 * rWorldCone - rayStepUVW, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0);
        let l4 = textureSampleLevel(lightTex, linSampler, clamp(uvw - diskAxis2 * rWorldCone + rayStepUVW, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0);
        let litAvg = 0.20 * (lit + l1 + l2 + l3 + l4);
        lit = mix(lit, litAvg, max(smoothstep(0.14, 0.68, cocBlur), tirGrazingSmooth * 0.92));
        coneDensity = lit.a;
      }
    }

    // Focal-plane C2 cubic B-spline reconstruction + chroma-preserving thin-lens CoC softening:
    // Always sample true local concTex so close-range defocused threads never dilute 50x into pale pink milk!
    let tirSmooth = select(0.24 + 0.68 * tirGrazingSmooth, 0.0, fineQuality);
    let wideConeMix = max(smoothstep(0.14, 0.72, cocBlur), tirSmooth);
    var conc: vec4<f32>;
    let nearMeniscusCollar = smoothstep(u.surfaceParams.x - 0.072, u.surfaceParams.x - 0.020, rayPos.y);
    if (fineQuality && (cocBlur < 0.36 || nearMeniscusCollar > 0.02) && tauTotal < 0.22) {
      let cocSplineW = max(1.0 - smoothstep(0.20, 0.36, cocBlur), nearMeniscusCollar);
      let splineW = cocSplineW * (1.0 - smoothstep(0.12, 0.22, tauTotal));
      let cBspline = sampleConcTricubic(uvw);
      if (splineW >= 0.995) {
        conc = cBspline;
      } else {
        let cTrilin = textureSampleLevel(concTex, linSampler, uvw, 0.0);
        conc = mix(cTrilin, cBspline, splineW);
      }
    } else {
      conc = textureSampleLevel(concTex, linSampler, uvw, 0.0);
    }

    let cLocalRaw = max(0.0, conc.r) + max(0.0, conc.g);
    let fLocalLysis = select(0.22, clamp(conc.g / max(cLocalRaw, 1e-5), 0.0, 1.0), cLocalRaw > 0.00008);
    let vOxShear = clamp(conc.b, 0.0, 1.0);
    if (wideConeMix > 0.001) {
      let collarConeGate = mix(1.0, smoothstep(0.00018, 0.0028, cLocalRaw), nearMeniscusCollar);
      let cPreserve = max(cLocalRaw * mix(0.72, 0.32, cocBlur), coneDensity * collarConeGate);
      let cSoft = mix(cLocalRaw, cPreserve, wideConeMix);
      conc.r = cSoft * (1.0 - fLocalLysis);
      conc.g = cSoft * fLocalLysis;
    }

    // Nyquist-exact rCoC-proportional step size: keeps ds <= 1.35*rCoC on moderately defocused stems (rCoC ~ 3.7mm -> 5.2mm)
    // while scaling smoothly up to ~11mm inside wide >10mm bokeh clouds!
    let haloCoarsen = 1.05 * (1.0 - smoothstep(0.00006, 0.00038, max(conc.r, coneDensity * 0.65)));
    let depthCoarsen = 1.35 * smoothstep(0.025, 0.22, tauTotal);
    let cocCoarsen = clamp((rCoC / fineBase) * 0.72, 0.0, 2.35) * cocBlur;
    let ds = min(fineBase * (1.0 + haloCoarsen + depthCoarsen + cocCoarsen), tEnd - t);

    // Meniscus wave height clip with tight collar-interior bridge, circular sub-collar voxel containment,
    // and strict sub-meniscus containment when underwater/TIR so zero voxel pokes above a crater into an inverted spike:
    var meniscusMask = 1.0;
    var subCollarContain = 1.0;
    var collarSmoothMix = 0.0;
    var orificeCoreBoost = 1.0;
    var orificeWeight = 0.0;
    if (rayPos.y > u.surfaceParams.x - 0.145) {
      let useSmoothWaveH = fineQuality && rayPos.y > u.surfaceParams.x - 0.065;
      let hWaveLocal = select(evalSurfaceHeightOnly(rayPos.xz), evalSurfaceHeightSmooth(rayPos.xz), useSmoothWaveH);
      let depthBelowCrater = hWaveLocal - rayPos.y;
      var bridgeWeight = 0.0;
      if (hasActiveDrops) {
        var deepASampled = false;
        var deepACache = 0.0;
        for (var d = 0u; d < ${MAX_ACTIVE_DROPS}u; d = d + 1u) {
          let slot = u.drops[d];
          if (slot.velOscAndDrain.w > 0.0 && (slot.posAndRadius.w > 0.001 || slot.velOscAndDrain.z > 0.001)) {
            let rCol = max(slot.posAndRadius.w * 0.84, 0.018);
            let rD = length(rayPos.xz - slot.posAndRadius.xz);
            bridgeWeight = max(bridgeWeight, 1.0 - smoothstep(rCol * 0.48, rCol, rD));
            if (slot.posAndRadius.w > 0.015 && slot.velOscAndDrain.z > 0.012 && depthBelowCrater > -0.008 && depthBelowCrater < 0.044) {
              if (!deepASampled) {
                deepACache = textureSampleLevel(lightTex, linSampler, vec3<f32>(uvw.x, max(0.002, uvw.y - 0.033), uvw.z), 0.0).a;
                deepASampled = true;
              }
              let emerg = (slot.posAndRadius.y - u.surfaceParams.x) / max(slot.posAndRadius.w, 0.010);
              let domeStillEmerging = smoothstep(-0.85, 0.05, emerg);
              let stemBelowGate = 1.0 - smoothstep(0.00025, 0.0018, deepACache);
              let nascentTouch = smoothstep(-0.28, 0.24, emerg) * stemBelowGate;
              let depthFade = (1.0 - smoothstep(0.015, 0.036, depthBelowCrater)) * stemBelowGate * domeStillEmerging;
              let rSubFoot = slot.posAndRadius.w * 0.78 + max(0.0, depthBelowCrater) * 0.44;
              let rFeather = mix(0.0016, 0.0044, max(nascentTouch, select(1.0, 0.0, topsideCam)));
              let radialClip = 1.0 - smoothstep(rSubFoot * 0.84, rSubFoot + rFeather, rD);
              subCollarContain = min(subCollarContain, mix(1.0, radialClip, depthFade));
              let innerSmooth = smoothstep(rSubFoot * 0.18, rSubFoot * 0.68, rD) * (1.0 - smoothstep(rSubFoot * 0.72, rSubFoot * 0.96, rD));
              collarSmoothMix = max(collarSmoothMix, depthFade * innerSmooth);
              // Compensate free-surface C2 B-spline zero-air/water boundary dilution across the annular draining orifice
              // while preserving natural 3D translucent liquid depth beneath craters and central rebound swells!
              let rOrifice = rSubFoot * 0.98;
              let orificeRadial = 1.0 - smoothstep(rOrifice * 0.78, rOrifice + 0.0015, rD);
              let orificeVertical = 1.0 - smoothstep(0.014, 0.038, max(0.0, depthBelowCrater));
              let drainActive = smoothstep(0.012, 0.085, slot.velOscAndDrain.z);
              let slotOrifice = orificeRadial * orificeVertical * drainActive * domeStillEmerging;
              orificeWeight = max(orificeWeight, slotOrifice);
              orificeCoreBoost = max(orificeCoreBoost, 1.0 + select(0.40, 0.65, topsideCam) * slotOrifice);
            }
          }
        }
      }
      let surfLocal = select(
        hWaveLocal,
        mix(hWaveLocal, max(hWaveLocal, u.surfaceParams.x + 0.003), bridgeWeight),
        topsideCam && fineQuality
      );
      let meniscusBokeh = smoothstep(0.00028, 0.0018, rCoC);
      let grazingTopside = select(0.0, 1.0 - smoothstep(0.06, 0.24, abs(rayDir.y)), topsideCam);
      let cocMeniscusFade = 0.008 + meniscusBokeh * (0.022 + 0.024 * grazingTopside);
      let topsideMask = mix(
        smoothstep(0.0, cocMeniscusFade, surfLocal - rayPos.y),
        smoothstep(-0.003, 0.008, surfLocal - rayPos.y),
        bridgeWeight
      );
      meniscusMask = select(smoothstep(0.0005, 0.0085, hWaveLocal - rayPos.y), topsideMask, topsideCam) * subCollarContain;
    }

    if (collarSmoothMix > 0.001) {
      conc.r = mix(conc.r, max(conc.r * 0.80, coneDensity * (1.0 - fLocalLysis)), collarSmoothMix * 0.35);
      conc.g = mix(conc.g, max(conc.g * 0.80, coneDensity * fLocalLysis), collarSmoothMix * 0.35);
    }

    // Low-threshold intact-RBC hairline mask + diffused lysed-hemoglobin halo mask preserving continuous trailing stems
    let intactMask = smoothstep(0.00006, 0.00085, conc.r) * meniscusMask;
    let lysedMask  = smoothstep(0.00028, 0.0016, conc.g) * meniscusMask;
    let cIntact = max(0.0, conc.r) * intactMask * orificeCoreBoost;
    let cLysed  = max(0.0, conc.g) * lysedMask * (1.0 - 0.55 * orificeWeight);
    let cTotal  = cIntact + cLysed;

    if (cTotal > 0.00004) {
      let fLysis = clamp(cLysed / max(cTotal, 1e-5), 0.0, 1.0);
      let intactFrac = 1.0 - fLysis;

      // Duysens (1956) cellular package/sieve optical path vs dispersed molecular lysed-Hb absorption,
      // coupled to Prahl (1999) 380-720nm dichromatic Soret + Q-band LUT driven by physical SO2 & shear (vOxShear):
      let dTauIntact = cIntact * ds * (1.22 * tauScale);
      let dTauLysed  = cLysed  * ds * (3.45 * tauScale);
      let dTau = dTauIntact + dTauLysed;
      tauTotal = tauTotal + dTau;
      tauLysed = tauLysed + dTauLysed;
      tauIntact = tauIntact + dTauIntact;
      let coherentOxWeight = smoothstep(0.0014, 0.0068, conc.r) * (1.0 - 0.85 * smoothstep(0.28, 0.72, cocBlur));
      tauOx = tauOx + dTau * (vOxShear * coherentOxWeight);
      let meanOxShear = clamp(tauOx / max(tauTotal, 1e-6), 0.0, 1.0);

      // Duysens (1956) Erythrocyte Sieve / Package Effect + optical CoC bokeh:
      // Below a single erythrocyte's optical thickness (tauCell = 0.0042..0.0065 in Prahl LUT units),
      // G and B attenuate linearly together while 650nm red stays open (rAnchor) -> natural rose-ruby
      // with zero yellow-tan Soret tea and zero magenta tint, joining C0-exact Prahl LUT at tau >= tauCell!
      let lutSharp = sampleSpectralLut(tauTotal, meanOxShear);
      let tauCell = mix(0.0042, 0.0065, smoothstep(0.10, 0.55, cocBlur));
      let tauSieveRef = max(tauTotal, tauCell);
      let sieveAlpha = clamp(tauTotal / tauCell, 0.0, 1.0);
      let tCell = sampleSpectralLut(tauSieveRef, meanOxShear * mix(1.0, 0.35, cocBlur));
      let rAnchor = tCell.r + (0.96 - tCell.r) * (1.0 - sieveAlpha);
      let lutSieve = vec3<f32>(
        1.0 - sieveAlpha * (1.0 - rAnchor),
        1.0 - sieveAlpha * (1.0 - tCell.g),
        1.0 - sieveAlpha * (1.0 - tCell.b)
      );
      let sieveWeight = clamp(max(intactFrac, smoothstep(0.08, 0.45, cocBlur)), 0.0, 1.0);
      let lutTrans = min(out.transmittance, mix(lutSharp, lutSieve, sieveWeight));

      let keyIllum = lit.r;
      let backIllum = lit.g;
      let multiScat = lit.b;
      let orificeScatterDamp = 1.0 - select(0.08, 0.16, topsideCam) * orificeWeight;

      // Dichromatic thin-sheet factors strictly confined to genuinely thin, high-shear grazing/transilluminated veils & vortex lips:
      let thinNeighborhood = 1.0 - smoothstep(0.016, 0.042, coneDensity);
      let subPlugZone = 1.0 - 0.96 * smoothstep(u.surfaceParams.x - 0.115, u.surfaceParams.x - 0.042, rayPos.y);
      let dfSubPlugZone = 1.0 - smoothstep(u.surfaceParams.x - 0.135, u.surfaceParams.x - 0.058, rayPos.y);
      let thinSheetFactor = smoothstep(0.16, 0.52, vOxShear)
                          * (1.0 - smoothstep(0.006, 0.030, tauTotal))
                          * (1.0 - smoothstep(0.018, 0.056, cLocalRaw))
                          * thinNeighborhood
                          * subPlugZone;
      let dfDichroFactor = smoothstep(0.09, 0.32, vOxShear)
                         * smoothstep(0.14, 0.52, max(backIllum, keyIllum * 0.78))
                         * (1.0 - smoothstep(0.009, 0.032, tauTotal))
                         * (1.0 - smoothstep(0.022, 0.060, cLocalRaw))
                         * thinNeighborhood
                         * dfSubPlugZone
                         * intactFrac;

      // Strictly energy-conserving cellular Mie scattering coupled to intact RBC cross-section (cIntact -> scatterProb)
      // with deep 650-680 nm diffusive red photon transport warming thick venous cores (R >> G, B):
      var stepRad = vec3<f32>(0.0);
      var dfCoherentRim = vec3<f32>(0.0);
      if (!darkMode) {
        let fwdTint = mix(vec3<f32>(0.052, 0.0020, 0.0016), vec3<f32>(0.078, 0.0124, 0.0014), thinSheetFactor);
        let keyTint = mix(vec3<f32>(0.018, 0.0005, 0.0010), vec3<f32>(0.025, 0.0020, 0.0009), vOxShear);
        let coreTint = mix(vec3<f32>(0.0155, 0.00038, 0.00062), vec3<f32>(0.0120, 0.00046, 0.00072), vOxShear);
        let rubyForward = fwdTint * (backIllum * (0.20 + 0.56 * phaseBack)) * orificeScatterDamp;
        let carmineKey  = keyTint * (keyIllum * (0.28 + 0.48 * phaseKey));
        let venousMulti = coreTint * (0.24 + 1.18 * multiScat);
        stepRad = (rubyForward + carmineKey + venousMulti) * albedoScale;
      } else {
        // Dark-Field Chiaroscuro: 590-610nm scarlet-vermilion (G/R ~ 0.16) on thin sheared vortex lips -> arterial ruby (G/R ~ 0.03) -> venous oxblood (G/R ~ 0.02)!
        let rimTint = mix(vec3<f32>(1.08, 0.032, 0.020), vec3<f32>(2.85, 0.465, 0.018), dfDichroFactor);
        let baseKeyTint = mix(vec3<f32>(0.58, 0.014, 0.013), vec3<f32>(0.88, 0.028, 0.015), smoothstep(0.24, 0.78, vOxShear));
        let keyTint = mix(baseKeyTint, vec3<f32>(2.05, 0.330, 0.018), dfDichroFactor * 0.65);
        let plugSelfShadow = 1.0 - 0.76 * smoothstep(0.028, 0.088, coneDensity);
        let darkRimFwd  = rimTint * (backIllum * (0.22 + 1.08 * phaseBack)) * mix(0.32, 1.0, plugSelfShadow);
        let darkKey     = keyTint * (keyIllum * (0.24 + 0.72 * phaseKey)) * plugSelfShadow;
        let oxbloodCore = vec3<f32>(0.052, 0.0010, 0.0011) * (0.040 + 1.22 * multiScat);
        stepRad = (darkRimFwd + darkKey + oxbloodCore) * albedoScale;
      }

      // 4-tap O(h^2) symmetric tetrahedral density gradient (diagonal corner sampling -> zero Y-plane ridges!):
      //  (A) C1-continuous sheetGate eliminates any cocBlur/tauTotal step crease across reflected TIR ripples!
      //  (B) Schlieren transverse refractive-index ray deflection (n = 1.333 + 0.032 * c_total)
      let sheetGate = smoothstep(0.0032, 0.0110, coneDensity)
                    * (1.0 - smoothstep(0.18, 0.34, tauTotal))
                    * smoothstep(0.032, 0.11, out.transmittance.r)
                    * (1.0 - smoothstep(0.44, 0.88, cocBlur));
      if (sheetGate > 0.002) {
        let d0 = textureSampleLevel(lightTex, linSampler, clamp(uvw + tetK0 * hTet, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0).a;
        let d1 = textureSampleLevel(lightTex, linSampler, clamp(uvw + tetK1 * hTet, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0).a;
        let d2 = textureSampleLevel(lightTex, linSampler, clamp(uvw + tetK2 * hTet, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0).a;
        let d3 = textureSampleLevel(lightTex, linSampler, clamp(uvw + tetK3 * hTet, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0).a;
        let gradWorld = (tetK0 * d0 + tetK1 * d1 + tetK2 * d2 + tetK3 * d3) * invTetDenom;
        let gradLen = length(gradWorld);

        if (gradLen > 0.12) {
          // (A) Grazing & oblique side-lit intact-RBC cellular sheet glow (dichromatic on thin high-shear veils)
          let nSheet = -gradWorld / gradLen;
          let grazing = pow(1.0 - clamp(abs(dot(nSheet, -rayDir)), 0.0, 1.0), 1.90);
          let sheetCrisp = smoothstep(0.15, 2.0, gradLen) * intactFrac * sheetGate;
          if (!darkMode) {
            let sheetTint = mix(vec3<f32>(0.115, 0.0038, 0.0028), vec3<f32>(0.175, 0.028, 0.0026), thinSheetFactor);
            stepRad = stepRad + sheetTint * (grazing * sheetCrisp * backIllum * orificeScatterDamp);
          } else {
            let nChiaro = normalize(vec3<f32>(-gradWorld.x, -gradWorld.y * 0.40, -gradWorld.z) + vec3<f32>(1e-6, 0.0, 0.0));
            let sideKeyFace = smoothstep(-0.14, 0.76, dot(nChiaro, keyDir)) * keyIllum;
            let rimEdgeFace = smoothstep(-0.14, 0.76, dot(nChiaro, backDir)) * backIllum;
            let chiaroSheet = sheetCrisp * (
              1.48 * sideKeyFace +
              1.82 * rimEdgeFace +
              2.35 * grazing * (0.04 + 0.96 * max(keyIllum, backIllum))
            );
            let chiaroTint = mix(vec3<f32>(0.82, 0.025, 0.016), vec3<f32>(3.15, 0.510, 0.018), dfDichroFactor);
            stepRad = stepRad + chiaroTint * chiaroSheet;
            // Coherent annular Dark-Field thin-interface rim scatter strictly on thin rolled vortex lips
            let thinRimDrive = sheetCrisp * dfDichroFactor * (0.28 + 0.90 * grazing + 0.38 * max(sideKeyFace, rimEdgeFace));
            let dfRimProb = 1.0 - exp(-thinRimDrive * ds * 11.5);
            dfCoherentRim = vec3<f32>(1.36, 0.220, 0.008) * dfRimProb * albedoScale;
          }

          // (B) Eikonal Schlieren ray deflection: d(rayDir)/ds = grad_perp(n) / n_water
          let gradN = (schlierenStr * sheetGate) * gradWorld;
          let gradPerp = gradN - dot(gradN, rayDir) * rayDir;
          rayDir = normalize(rayDir + clamp(gradPerp / 1.333, vec3<f32>(-1.2), vec3<f32>(1.2)) * ds);
        }
      }

      // Physical RTE in-scattering step
      let scatterProb = 1.0 - exp(-cIntact * ds * 24.0);
      var inScatter = scatterProb * stepRad + dfCoherentRim;
      if (darkMode && cLysed > 0.00018) {
        let beamHalo = 0.26 * (keyIllum + backIllum) + 1.65 * multiScat;
        let haloTransillum = (1.0 - exp(-cLysed * ds * 18.0)) * clamp(beamHalo, 0.0, 0.55);
        inScatter = inScatter + vec3<f32>(0.32, 0.016, 0.012) * haloTransillum;
      }
      out.radiance = out.radiance + out.transmittance * inScatter;
      out.transmittance = lutTrans;

      let maxTrans = max(out.transmittance.r, max(out.transmittance.g, out.transmittance.b));
      if (maxTrans < transStopThresh) {
        out.transmittance = vec3<f32>(0.0);
        break;
      }
    }

    t = t + ds;
    rayPos = rayPos + rayDir * ds;
  }

  out.bentDir = rayDir;
  out.tauIntact = tauIntact;
  out.tauTotal = tauTotal;
  return out;
}

fn evalCausticShafts(ro: vec3<f32>, rd: vec3<f32>, tStart: f32, tMax: f32, jitter: f32, fineQuality: bool) -> vec3<f32> {
  if (u.opticalParams.z <= 0.01 || u.opticalParams.w > 0.5) {
    return vec3<f32>(0.0);
  }
  let t0 = max(tStart, 0.15);
  let t1 = min(tMax, 2.35);
  if (t1 <= t0) {
    return vec3<f32>(0.0);
  }
  let keyDir = normalize(vec3<f32>(0.22, 0.95, 0.20));
  let span = t1 - t0;
  let steps = select(3, 5, fineQuality);
  let dt = span / f32(steps);
  var accum = 0.0;

  for (var i = 0; i < steps; i = i + 1) {
    let t = t0 + (f32(i) + 0.5) * dt;
    let p = ro + rd * t;
    if (p.y < u.surfaceParams.x && p.y > -1.15) {
      let dUp = (u.surfaceParams.x - p.y) / keyDir.y;
      let hitXZ = p.xz + keyDir.xz * dUp;
      let r = length(hitXZ);
      if (r < 1.35) {
        let rDir = normalize(hitXZ + vec2<f32>(1e-5, 0.0));
        // Second-moment-matched symmetric 2-tap filter (0.5 * 0.0195^2 == 0.33 * 0.024^2) halves waveTex fetches!
        let lapBroad = 0.50 * (
          evalWaveHeightRaw(hitXZ + rDir * 0.0195).b +
          evalWaveHeightRaw(hitXZ - rDir * 0.0195).b
        );
        let depthFade = smoothstep(0.02, 0.28, u.surfaceParams.x - p.y) * (1.0 - smoothstep(0.55, 1.35, r));
        let focus = clamp(-lapBroad * 42.0, 0.0, 1.15) * depthFade;
        accum = accum + focus * dt;
      }
    }
  }

  let shaftTint = vec3<f32>(0.055, 0.072, 0.088);
  return shaftTint * (accum * u.opticalParams.z);
}

fn evalTopsideBranch(
  ro: vec3<f32>,
  rd: vec3<f32>,
  tFocus: f32,
  apRadius: f32,
  jitter: f32,
  camRelY: f32,
  fineQuality: bool
) -> vec4<f32> {
  let opticalDome = smoothstep(0.004, 0.058, camRelY);
  let fresnelCap = 0.88 * opticalDome;
  let airBgMix = opticalDome;
  let darkTop = u.opticalParams.w > 0.5;

  var tSurf = -1.0;
  if (camRelY > -0.004) {
    tSurf = intersectWaterSurface(ro, rd, 0.002, 12.0, true);
  }
  var airBackdrop = mix(evalStudioBackdrop(rd, true), evalStudioBackdrop(rd, false), airBgMix);
  var tSurfOut = 0.0;

  if (tSurf > 0.0) {
    tSurfOut = tSurf;
    let pSurf = ro + rd * tSurf;
    let sSurf = evalSurface(pSurf.xz);
    let nSurf = normalize(mix(vec3<f32>(0.0, 1.0, 0.0), sSurf.normal, 0.35 + 0.65 * opticalDome));
    let snellDown = refract(rd, nSurf, 0.75);
    let hasSnell = dot(snellDown, snellDown) > 0.25;
    let validSnell = select(rd, snellDown, hasSnell);
    let refrDir = normalize(mix(rd, validSnell, 0.48 * opticalDome));

    // Step positive liquid swells (pSurf.y > u.surfaceParams.x) down to concTex top plane y = u.surfaceParams.x
    // via convex-dome converging lensing into the sub-swell blood throat (preventing grazing -Z miss into clear water!)
    var pPlumeOrigin = pSurf;
    var tPlumeOrigin = tSurf;
    var marchDir = refrDir;
    let swellLift = max(0.0, pSurf.y - u.surfaceParams.x);
    if (swellLift > 0.0) {
      let focusInward = -sSurf.normal.xz * (swellLift * 0.52);
      pPlumeOrigin = vec3<f32>(pSurf.x + focusInward.x, u.surfaceParams.x - 0.0015, pSurf.z + focusInward.y);
      tPlumeOrigin = tSurf + swellLift;
      let domeLensMix = smoothstep(0.002, 0.016, swellLift);
      let steepThroatDir = normalize(vec3<f32>(
        refrDir.x * 0.42 - sSurf.normal.x * 0.30,
        -0.78,
        refrDir.z * 0.42 - sSurf.normal.z * 0.30
      ));
      marchDir = normalize(mix(refrDir, steepThroatDir, domeLensMix));
    }
    let crestCont = 1.0 / (1.0 + 6.0 * abs(tSurf - tPlumeOrigin));
    let grazingWaveMix = smoothstep(0.06, 0.22, -rd.y);
    let pWaveXZ = mix(pPlumeOrigin.xz, pSurf.xz, grazingWaveMix);
    let sInfo = evalSurface(pWaveXZ);
    let cosI = clamp(dot(-rd, nSurf), 0.0, 1.0);
    let fresnel = fresnelDielectric(cosI, 1.0, 1.333);

    let slopeLen = length(sInfo.normal.xz);
    let waveSlope = smoothstep(0.00018, 0.010, slopeLen);
    let capillaryRing = smoothstep(0.0010, 0.014, slopeLen);
    let activeCurve = smoothstep(0.015, 0.14, slopeLen);
    let reflDir = reflect(rd, sInfo.normal);
    var reflCol = evalStudioBackdrop(reflDir, false) * select(vec3<f32>(1.03, 1.04, 1.05), vec3<f32>(1.0), darkTop);
    let studioGobo = mix(0.42, 1.24, smoothstep(0.06, 0.56, reflDir.y));
    reflCol = reflCol * select(mix(1.0, studioGobo, waveSlope), 1.0, darkTop);
    if (darkTop) {
      let dfTopRing = exp(-pow((reflDir.y - 0.14) / 0.095, 2.0));
      reflCol = reflCol + vec3<f32>(0.18, 0.22, 0.28) * (0.30 + 0.70 * dfTopRing) * waveSlope;
    }

    // Signed unfolded optical focal path for virtual water-reflected drops (evaluated at 1:1 Retina in COMPOSITE_WGSL)
    let remFocusRefl = tFocus - tSurf;
    let unusedReflGuard = remFocusRefl * 0.0;

    // Signed remaining focal path: negative when camera is focused above/in-front of water surface!
    let remFocus = tFocus - tPlumeOrigin;
    let plume = marchPlume(pPlumeOrigin, marchDir, 0.001, 5.0, remFocus, apRadius, jitter, fineQuality);
    let plumePresence = smoothstep(0.0006, 0.018, plume.tauTotal);
    let bloodInColumn = plumePresence;

    // Optical glass-tank meniscus depth tone clearly distinguishes calm water plane from warm studio air sky
    let zTable = smoothstep(-0.85, 0.65, pPlumeOrigin.z);
    let calmWaterTone = mix(vec3<f32>(0.865, 0.914, 0.956), vec3<f32>(0.918, 0.952, 0.982), zTable);
    let waterPlaneTint = select(mix(vec3<f32>(1.0), calmWaterTone, opticalDome), vec3<f32>(1.0), darkTop);
    var refrCol = evalStudioBackdrop(plume.bentDir, true) * waterPlaneTint + vec3<f32>(unusedReflGuard);

    let shafts = evalCausticShafts(pPlumeOrigin, marchDir, 0.02, 2.5, jitter, fineQuality);
    refrCol = refrCol * plume.transmittance + plume.radiance + shafts * plume.transmittance;

    // 3D translucent liquid subsurface hemoglobin optics across convex rebound swells (s2_06) & concave craters (s2_05):
    // Driven continuously by smooth optical depth plume.tauTotal (zero synthetic cookie-cutter cylinder!)
    let domeKeyDir = normalize(vec3<f32>(0.22, 0.95, 0.20));
    let domeBlend = smoothstep(0.0005, 0.014, swellLift) * smoothstep(0.0012, 0.032, plume.tauTotal);
    if (domeBlend > 0.001) {
      let cosDom = clamp(dot(-rd, sInfo.normal), 0.0, 1.0);
      let domeChord = (swellLift * 1.85) / max(cosDom, 0.42);
      let domeTau = max(0.0065, domeChord * bloodInColumn * 0.46 * u.opticalParams.x);
      let domeEdgeThin = smoothstep(0.18, 0.84, 1.0 - cosDom);
      let domeTrans = sampleSpectralLut(domeTau, mix(0.24, 0.72, domeEdgeThin));
      let domeKeyWrap = smoothstep(0.24, 0.96, clamp(dot(sInfo.normal, domeKeyDir), 0.0, 1.0));
      let domeFrontLit = smoothstep(-0.18, 0.32, sInfo.normal.z + 0.25 * sInfo.normal.x);
      let domeBodyLit = mix(vec3<f32>(0.038, 0.0008, 0.0009), vec3<f32>(0.185, 0.0048, 0.0030), 0.55 * domeKeyWrap + 0.45 * domeFrontLit);
      let domeRubyRim = mix(domeBodyLit, vec3<f32>(0.245, 0.0085, 0.0038), domeEdgeThin * 0.68);
      let domeSubsurf = select(
        domeRubyRim,
        vec3<f32>(0.18, 0.008, 0.005),
        darkTop
      );
      let domeShaded = refrCol * domeTrans + domeSubsurf * (1.0 - domeTrans.g);
      refrCol = mix(refrCol, domeShaded, domeBlend);
    }

    let craterDepth = max(0.0, u.surfaceParams.x - pSurf.y);
    let craterBowl = smoothstep(0.0015, 0.014, craterDepth) * bloodInColumn;
    if (!darkTop && craterBowl > 0.002) {
      let craterSunWall = smoothstep(-0.06, 0.28, sInfo.normal.z + 0.35 * sInfo.normal.x);
      let pitShadow = 1.0 - 0.38 * smoothstep(0.008, 0.020, craterDepth);
      refrCol = refrCol * pitShadow
              + vec3<f32>(0.145, 0.0038, 0.0024) * (0.20 + 0.80 * craterSunWall) * craterBowl * activeCurve;
    }

    // Chromaticity-preserving 3D meniscus curvature shading over coalescing craters & rebound swells
    if (!darkTop && bloodInColumn > 0.02 && waveSlope > 0.01) {
      let curveShade = clamp(-sInfo.lap * 7.5 + sInfo.normal.z * 0.95, -0.36, 0.58) * waveSlope * bloodInColumn;
      refrCol = refrCol * (1.0 + curveShade) + vec3<f32>(0.026, 0.0007, 0.0005) * bloodInColumn * activeCurve;
    }

    // Attenuate horizontal meniscus stain & dimple rings at near-horizon grazing angles (-rd.y < 0.065)
    // so 15cm planar patches never foreshorten 18:1 into a 2px razor line across defocused plumes!
    let obliquePlanarVis = smoothstep(0.065, 0.22, -rd.y);

    // Concave capillary meniscus collar rim tightly localized to dense stem-surface contact (zero drifting parcel boxes!)
    let halfXZ = u.domainBounds.x;
    if (obliquePlanarVis > 0.002 && max(abs(pPlumeOrigin.x), abs(pPlumeOrigin.z)) < halfXZ * 0.94) {
      let uSurf = (pPlumeOrigin.x + halfXZ) / (2.0 * halfXZ);
      let wSurf = (pPlumeOrigin.z + halfXZ) / (2.0 * halfXZ);
      let s0 = textureSampleLevel(lightTex, linSampler, vec3<f32>(uSurf, 0.982, wSurf), 0.0).a;
      let sz1 = textureSampleLevel(lightTex, linSampler, vec3<f32>(uSurf, 0.982, clamp(wSurf + 0.010, 0.0, 1.0)), 0.0).a;
      let sz2 = textureSampleLevel(lightTex, linSampler, vec3<f32>(uSurf, 0.982, clamp(wSurf - 0.010, 0.0, 1.0)), 0.0).a;
      let meniscusDimple = smoothstep(0.0055, 0.026, s0) * (1.0 - smoothstep(0.028, 0.085, s0)) * 0.35 * opticalDome * obliquePlanarVis;
      if (!darkTop) {
        let dimpleGradZ = sz1 - sz2;
        let rimShade = clamp(-0.032 + 5.0 * dimpleGradZ, -0.055, 0.055) * meniscusDimple * (1.0 - 0.92 * bloodInColumn);
        refrCol = refrCol * mix(vec3<f32>(1.0), vec3<f32>(0.955, 0.968, 0.985), meniscusDimple)
                + vec3<f32>(rimShade * 0.85, rimShade * 0.93, rimShade);
      } else {
        refrCol = refrCol + vec3<f32>(0.028, 0.0026, 0.0020) * meniscusDimple * bloodInColumn;
      }
    }

    // Surfactant crimson film optical absorption via Duysens-sieved Prahl spectral LUT:
    // Confined strictly to active meniscus collar curvature so no flat 2D red coin lingers on calm water!
    let stainCore = smoothstep(0.68, 0.96, sInfo.stain) * 0.06 * activeCurve;
    let stainStrength = stainCore * (0.14 + 0.86 * opticalDome) * obliquePlanarVis;
    if (stainStrength > 0.005) {
      let filmTau = stainStrength * 0.0052;
      let filmTrans = mix(vec3<f32>(1.0), sampleSpectralLut(max(filmTau, 0.0065), 0.55), clamp(filmTau / 0.0065, 0.0, 1.0));
      refrCol = refrCol * filmTrans;
    }

    let rimLightDir = normalize(vec3<f32>(-rd.x, 0.24, -rd.z));
    let vDir = -rd;
    let hRimRaw = rimLightDir + vDir;
    let hRim = hRimRaw / max(length(hRimRaw), 1e-5);

    // Preserve deep crimson/ruby saturation on flat submerged plumes (avoiding grey sky washout!) while
    // keeping full dielectric Fresnel reflection & 1:1 Retina softbox speculars across active wave slopes/swells!
    let grazingFresnelRim = pow(1.0 - cosI, 3.0);
    let plumeGlareDamp = select(mix(1.0, mix(0.055, 0.42, activeCurve * (0.35 + 0.65 * grazingFresnelRim)), bloodInColumn), 1.0, darkTop);
    let clearWaterReliefGate = select(mix(1.0, mix(0.18, 0.72, activeCurve), bloodInColumn), 1.0, darkTop);
    let viewSteepness = 0.20 + 0.80 * smoothstep(0.03, 0.28, -rd.y);
    let tangentContinuity = mix(crestCont, 1.0, grazingWaveMix);

    // Pure colorless Fresnel reflection in Bright-Field (zero painted grey rippleRelief bands); subtle rim in Dark-Field
    let waveRim = pow(max(dot(sInfo.normal, hRim), 0.0), 18.0) * select(0.0, 0.48, darkTop)
                * opticalDome * (0.12 + 0.88 * waveSlope) * tangentContinuity;
    let rippleRelief = select(
      0.0,
      clamp(-sInfo.lap * 11.0 + sInfo.normal.z * 0.55, 0.0, 0.14) * waveSlope * opticalDome * tangentContinuity,
      darkTop
    );
    let darkPlumeRim = select(
      vec3<f32>(0.0),
      vec3<f32>(0.06, 0.005, 0.004) * bloodInColumn * waveSlope * opticalDome,
      darkTop
    );

    let maxCalmVeil = select(
      (0.16 * plumeGlareDamp + (0.24 * waveSlope + 0.10 * activeCurve) * clearWaterReliefGate) * opticalDome * viewSteepness,
      (0.20 + 0.34 * waveSlope) * opticalDome,
      darkTop
    );
    let effTopFresnel = fresnel;
    let reflWeight = clamp((effTopFresnel + 0.038 * capillaryRing * clearWaterReliefGate) * fresnelCap, 0.0, maxCalmVeil);
    let horizonFade = mix(1.0 - smoothstep(1.75, 4.2, length(pPlumeOrigin.xz)), 1.0, bloodInColumn);
    let waterTopCol = mix(refrCol, reflCol, reflWeight)
                    + vec3<f32>(waveRim * 0.86, waveRim * 0.93, waveRim)
                    + vec3<f32>(rippleRelief * 0.80, rippleRelief * 0.91, rippleRelief)
                    + darkPlumeRim;
    airBackdrop = mix(airBackdrop, waterTopCol, horizonFade);
  } else if (camRelY <= 0.022 && rd.y < 0.025) {
    let plume = marchPlume(ro, rd, 0.02, 5.5, tFocus, apRadius, jitter, fineQuality);
    let bgBase = mix(evalStudioBackdrop(plume.bentDir, true), airBackdrop, airBgMix);
    let shafts = evalCausticShafts(ro, rd, 0.02, 5.5, jitter, fineQuality) * (1.0 - airBgMix);
    airBackdrop = bgBase * plume.transmittance + plume.radiance + shafts * plume.transmittance;
  }

  return vec4<f32>(airBackdrop, tSurfOut);
}

fn evalSubmergedBranch(
  ro: vec3<f32>,
  rd: vec3<f32>,
  tFocus: f32,
  apRadius: f32,
  jitter: f32,
  camRelY: f32,
  fineQuality: bool
) -> vec4<f32> {
  let tWaterStart = 0.02;
  var tUnderSurf = -1.0;
  if (camRelY < -0.0005 && rd.y > 0.001) {
    let tFlatApprox = (u.surfaceParams.x - ro.y) / rd.y;
    if (rd.y > 0.052 && tFlatApprox < 4.5) {
      tUnderSurf = intersectWaterSurface(ro, rd, 0.004, min(12.0, tFlatApprox + 0.8), false);
    } else {
      tUnderSurf = min(12.0, tFlatApprox);
    }
  }

  let tPlumeEnd = select(5.5, tUnderSurf + 0.004, tUnderSurf > 0.0);

  // 1. Primary volumetric march (with Prahl LUT, intact-RBC Mie, C2 tricubic, CoC DOF & Schlieren bending)
  let fgPlume = marchPlume(ro, rd, tWaterStart, tPlumeEnd, tFocus, apRadius, jitter, fineQuality);

  // Forward Mie turbidity & total hemoglobin extinction modulation: prevents caustic shafts or backdrop bleach through dense/defocused plumes
  let turbidSharpness = exp(-fgPlume.tauIntact * 680.0);
  let plumeClearance = exp(-(fgPlume.tauTotal * 460.0 + fgPlume.tauIntact * 380.0));
  let fgMaxTrans = max(fgPlume.transmittance.r, max(fgPlume.transmittance.g, fgPlume.transmittance.b));
  let shaftWeight = fgMaxTrans * turbidSharpness * plumeClearance;

  var fgShafts = vec3<f32>(0.0);
  if (shaftWeight > 0.015) {
    fgShafts = evalCausticShafts(ro, rd, tWaterStart, select(5.5, tUnderSurf, tUnderSurf > 0.0), jitter, fineQuality);
  }

  let smoothBehind = evalStudioBackdrop(fgPlume.bentDir, true);
  var behindPlume = smoothBehind;

  // 2. Physical Underside Total Internal Reflection + Pupil/Dispersion-Coupled Dielectric Snell's Window:
  if (fgMaxTrans > 0.012 && camRelY < -0.0005 && rd.y > 0.001) {
    let tMirror = (u.surfaceParams.x - ro.y) / rd.y;
    if (tMirror > 0.004 && tMirror < 12.0) {
      let underGrazingMix = smoothstep(0.065, 0.22, rd.y);
      let tRawHit = select(tMirror, tUnderSurf, tUnderSurf > 0.0);
      let foldCont = 1.0 / (1.0 + 10.0 * abs(tRawHit - tMirror));
      let tSurfHit = mix(tMirror, tRawHit, underGrazingMix * foldCont);
      let pMirror = ro + rd * tMirror;
      let pRipple = ro + rd * tSurfHit;
      let sInfo = evalSurface(pRipple.xz);
      let nRaw = -sInfo.normal; // Downward normal into water medium

      let craterDepth = max(0.0, u.surfaceParams.x - sInfo.height);
      let craterDamp = 1.0 - smoothstep(0.008, 0.026, craterDepth);
      let waveEnv = sqrt(dot(sInfo.normal.xz, sInfo.normal.xz) + (0.022 * sInfo.lap) * (0.022 * sInfo.lap));

      // Grazing anti-fold clamp prevents back-facing wave-back inversion while keeping physical wave slopes
      let cosRaw = dot(-rd, nRaw);
      let nUnder = normalize(nRaw + max(0.0, 0.015 - cosRaw) * (-rd));
      let cosI = clamp(dot(-rd, nUnder), 0.0, 1.0);
      let fresnel = fresnelDielectric(cosI, 1.333, 1.0);

      var tirDir = reflect(rd, nUnder);
      tirDir = normalize(vec3<f32>(tirDir.x, -sqrt(tirDir.y * tirDir.y + 0.0009), tirDir.z));

      // Signed unfolded virtual-mirror focal path
      let remFocusTir = tFocus - tSurfHit;

      // Macro-lens pupil filter centered at displaced hit pRipple.xz preserving genuine capillary & gravity wave slopes
      // so the underside meniscus shimmers with liquid distortion rather than acting as a flat 1:1 vanity mirror!
      let rPupil = clamp(0.0032 + 0.0028 / max(rd.y, 0.20), 0.0035, 0.0078);
      let hPx1 = evalSurfaceHeightOnly(pRipple.xz + vec2<f32>(rPupil, 0.0));
      let hPx0 = evalSurfaceHeightOnly(pRipple.xz - vec2<f32>(rPupil, 0.0));
      let hPz1 = evalSurfaceHeightOnly(pRipple.xz + vec2<f32>(0.0, rPupil));
      let hPz0 = evalSurfaceHeightOnly(pRipple.xz - vec2<f32>(0.0, rPupil));
      let gradPupil = vec2<f32>(hPx1 - hPx0, hPz1 - hPz0) / (2.0 * rPupil);
      let gradSoft = (gradPupil * 1.15) / sqrt(1.0 + dot(gradPupil, gradPupil) * 3.5);
      let nPupilRaw = normalize(vec3<f32>(gradSoft.x, -1.0, gradSoft.y));
      let cosPupilRaw = dot(-rd, nPupilRaw);
      let nPupilUnder = normalize(nPupilRaw + max(0.0, 0.035 - cosPupilRaw) * (-rd));
      var tirPlumeDir = reflect(rd, nPupilUnder);
      tirPlumeDir = normalize(mix(tirPlumeDir, tirDir, 0.25));
      tirPlumeDir = normalize(vec3<f32>(tirPlumeDir.x, -sqrt(tirPlumeDir.y * tirPlumeDir.y + 0.0016), tirPlumeDir.z));

      var tirPlume: PlumeResult;
      tirPlume.radiance = vec3<f32>(0.0);
      tirPlume.transmittance = vec3<f32>(1.0);
      tirPlume.bentDir = tirPlumeDir;
      tirPlume.tauIntact = 0.0;
      tirPlume.tauTotal = 0.0;
      let pTirOrigin = vec3<f32>(pRipple.x, min(sInfo.height, u.surfaceParams.x) - 0.0012, pRipple.z);
      let tirSubjBox = intersectBox(pTirOrigin, tirPlumeDir, vec3<f32>(-0.42, -0.96, -0.42), vec3<f32>(0.42, 0.302, 0.42));
      if (fgMaxTrans * max(fresnel, 0.35) > 0.018 && tirSubjBox.y > max(tirSubjBox.x, 0.002)) {
        let apRadiusTir = apRadius + 0.0046 + clamp(waveEnv * 0.085, 0.0, 0.016);
        tirPlume = marchPlume(
          pTirOrigin,
          tirPlumeDir,
          max(0.002, tirSubjBox.x),
          min(3.0, tirSubjBox.y),
          remFocusTir,
          apRadiusTir,
          jitter,
          false
        );
      }

      // Folded water-path attenuation & meniscus micro-wave dispersion soften reflected plume contrast
      // so Sub-Meniscus views read as a watery underside reflection rather than a sharp 1:1 duplicate
      let tirFoldedAtten = exp(-tSurfHit * 0.52) * mix(0.30, 0.46, craterDamp);
      tirPlume.transmittance = mix(vec3<f32>(1.0), tirPlume.transmittance, tirFoldedAtten);
      tirPlume.radiance = tirPlume.radiance * tirFoldedAtten;

      let darkMode = u.opticalParams.w > 0.5;
      let slopeMag = length(sInfo.normal.xz);
      let slopeSignal = smoothstep(0.00012, 0.011, slopeMag);

      let tirBloodPresence = clamp(1.0 - tirPlume.transmittance.g, 0.0, 1.0);
      let clearReflDir = normalize(mix(tirDir, tirPlume.bentDir, tirBloodPresence));
      var tirBg = evalStudioBackdrop(clearReflDir, true);
      if (!darkMode) {
        let horizonBand = smoothstep(-0.36, -0.020, clearReflDir.y);
        let clearWaterMod = mix(0.93, 1.05, horizonBand);
        let clearTirBg = tirBg * clearWaterMod;

        let flatTirDir = normalize(vec3<f32>(rd.x, -max(abs(rd.y), 0.08), rd.z));
        let diffusedTirBg = evalStudioBackdrop(flatTirDir, true) * mix(0.96, 1.03, horizonBand);
        tirBg = mix(clearTirBg, diffusedTirBg, max(tirBloodPresence * 0.85, (1.0 - craterDamp) * 0.65));
      } else {
        let dfRingBand = exp(-pow((tirPlume.bentDir.y + 0.13) / 0.095, 2.0));
        let crestCurv = clamp(0.28 - 14.0 * sInfo.lap, 0.0, 1.35) * slopeSignal * mix(0.35, 1.0, craterDamp);
        tirBg = tirBg
              + vec3<f32>(0.16, 0.20, 0.26) * (0.35 + 0.65 * dfRingBand) * crestCurv;
      }
      tirBg = mix(smoothBehind, tirBg, turbidSharpness);

      let tirCoreGlow = select(
        vec3<f32>(0.024, 0.0005, 0.0010),
        vec3<f32>(0.009, 0.0002, 0.0004),
        darkMode
      ) * smoothstep(0.002, 0.018, tirPlume.tauIntact) * (1.0 - tirPlume.transmittance.r);
      let tirCol = tirBg * tirPlume.transmittance + tirPlume.radiance + tirCoreGlow;

      var underSurfCol = tirCol;
      let steepSnell = smoothstep(0.68, 0.76, rd.y);
      let nSnellEval = normalize(mix(nUnder, vec3<f32>(0.0, -1.0, 0.0), steepSnell * 0.72));
      let cosSnell = clamp(dot(-rd, nSnellEval), 0.0, 1.0);
      let fresnelSnellCenter = fresnelDielectric(cosSnell, 1.333, 1.0);
      let dCosCone = clamp(0.012 + apRadius / max(tSurfHit, 0.25) + waveEnv * 0.28, 0.012, 0.085);
      let fresnelWindow = (
        fresnelDielectric(cosSnell - dCosCone, 1.333, 1.0) +
        fresnelSnellCenter +
        fresnelDielectric(cosSnell + dCosCone, 1.333, 1.0)
      ) * (1.0 / 3.0);

      // Couple Snell's Window opening to BOTH macroscopic view elevation AND pupil-filtered meniscus incidence:
      // At shallow underwater angles (rd.y < 0.615, e.g. elevation -0.21..-0.45), localized concave crater walls
      // NEVER punch a [225, 232, 238] floating eyeball cutout through the TIR mirror, while steep overhead Snell's
      // Window views (rd.y > 0.76) remain 100% open across central impact craters without grey crater shadows!
      let macroSnellGate = smoothstep(0.615, 0.675, rd.y);
      let pupilCosI = clamp(dot(-rd, nPupilUnder), 0.0, 1.0);
      let pupilSnellGate = smoothstep(0.655, 0.735, pupilCosI);
      let snellOpenGate = macroSnellGate * mix(pupilSnellGate * mix(0.28, 1.0, craterDamp), 1.0, steepSnell);

      if (fresnelWindow < 0.999 && snellOpenGate > 0.001) {
        let cosRefr = max(cosSnell, 0.6645);
        let nRefr = normalize(nSnellEval + max(0.0, cosRefr - cosSnell) * (-rd));
        let airDir = refract(rd, nRefr, 1.333);
        if (dot(airDir, airDir) > 0.25) {
          // Grazing cuvette-rim horizon baffle at the outer critical rim without darkening central steep craters
          let snellGrazingBaffle = mix(0.32, 0.95, smoothstep(0.04, 0.44, max(airDir.y, steepSnell * 0.55)));
          var snellAir = evalStudioBackdrop(airDir, false) * snellGrazingBaffle;
          let pSnellOrig = pRipple + sInfo.normal * 0.003;
          let apSnell = min(apRadius + waveEnv * 0.0065, 0.0068);
          let snellDrop = intersectAerialDrop(pSnellOrig, airDir, tFocus - tSurfHit, apSnell);
          if (snellDrop.hit) {
            let shadedSnellDrop = shadeAerialDrop(pSnellOrig, airDir, snellDrop, snellAir);
            // Meniscus wave curvature damps only specular excess above snellAir (zero sky-shadow halo!)
            let curvSpecDamp = 1.0 / (1.0 + 48.0 * waveEnv);
            let specExcess = max(shadedSnellDrop - snellAir, vec3<f32>(0.0));
            snellAir = min(shadedSnellDrop, snellAir) + specExcess * curvSpecDamp;
          }
          let maxSnellPipT = select(8.0, snellDrop.tEnter + 0.004, snellDrop.hit);
          let snellPip = evalCapillaryPipette(pSnellOrig, airDir, maxSnellPipT, snellAir);
          if (snellPip.w > 0.0) {
            snellAir = snellPip.rgb;
          }
          snellAir = mix(smoothBehind, snellAir, turbidSharpness);

          // 3-channel RGB Cauchy dispersion across Snell's Window critical rim (n_R = 1.330, n_G = 1.333, n_B = 1.339)
          let fresnelRgb = vec3<f32>(
            0.5 * (fresnelDielectric(cosSnell - dCosCone * 0.65, 1.330, 1.0) + fresnelDielectric(cosSnell + dCosCone * 0.65, 1.330, 1.0)),
            fresnelWindow,
            0.5 * (fresnelDielectric(cosSnell - dCosCone * 0.65, 1.339, 1.0) + fresnelDielectric(cosSnell + dCosCone * 0.65, 1.339, 1.0))
          );
          let rawWindowRgb = mix(fresnelRgb, vec3<f32>(1.0), tirBloodPresence * (1.0 - max(0.65 * macroSnellGate, steepSnell)));
          let windowEffRgb = mix(vec3<f32>(1.0), rawWindowRgb, snellOpenGate);
          underSurfCol = mix(snellAir, tirCol, windowEffRgb);
        }
      }

      let keyDir = select(
        normalize(vec3<f32>(0.22, 0.95, 0.20)),
        normalize(vec3<f32>(0.64, 0.70, 0.32)),
        darkMode
      );
      let glintScale = select(0.34, 0.42, darkMode) * mix(0.30, 1.0, craterDamp);
      let lapScale = select(8.0, 10.5, darkMode) * craterDamp;
      let rippleGlint = slopeSignal * (
        pow(max(dot(sInfo.normal, keyDir), 0.0), 36.0) * glintScale
        + clamp(-sInfo.lap * lapScale, select(-0.012, -0.005, darkMode), 0.115)
      ) * turbidSharpness * mix(0.03, 1.0, fresnelWindow);
      underSurfCol = underSurfCol + vec3<f32>(rippleGlint * 0.85, rippleGlint * 0.94, rippleGlint) * tirPlume.transmittance;

      let submergeGate = 1.0 - smoothstep(-0.0045, -0.0005, camRelY);
      let underFade = (1.0 - smoothstep(1.32, 2.75, length(pMirror.xz)))
                    * submergeGate
                    * smoothstep(0.001, 0.028, rd.y);
      behindPlume = mix(behindPlume, underSurfCol, underFade);
    }
  }

  let col = behindPlume * fgPlume.transmittance + fgPlume.radiance + fgShafts * fgPlume.transmittance * turbidSharpness * plumeClearance;
  return vec4<f32>(col, 0.0);
}

fn evalLensRay(
  roLens: vec3<f32>,
  rdLens: vec3<f32>,
  tFocus: f32,
  apRadius: f32,
  jitter: f32,
  camRelY: f32,
  airWeight: f32,
  fineQuality: bool
) -> vec4<f32> {
  if (airWeight >= 0.999) {
    return evalTopsideBranch(roLens, rdLens, tFocus, apRadius, jitter, camRelY, fineQuality);
  }
  if (airWeight <= 0.001) {
    return evalSubmergedBranch(roLens, rdLens, tFocus, apRadius, jitter, camRelY, fineQuality);
  }
  let topOut = evalTopsideBranch(roLens, rdLens, tFocus, apRadius, jitter, camRelY, fineQuality);
  let subOut = evalSubmergedBranch(roLens, rdLens, tFocus, apRadius, jitter, camRelY, fineQuality);
  return vec4<f32>(mix(subOut.rgb, topOut.rgb, airWeight), select(0.0, topOut.a, airWeight > 0.001));
}

@fragment
fn fsMain(in: VertexOut) -> @location(0) vec4<f32> {
  // Base pinhole camera center (uniform across viewport -> zero split-screen meniscus seam!)
  let ro = u.cameraPosAndTime.xyz;
  let pFar4 = u.invViewProj * vec4<f32>(in.ndc, 1.0, 1.0);
  let pFar = pFar4.xyz / pFar4.w;
  let rd = normalize(pFar - ro);

  let pC4 = u.invViewProj * vec4<f32>(0.0, 0.0, 0.5, 1.0);
  let pCenter = pC4.xyz / pC4.w;
  let camFwd = normalize(pCenter - ro);

  let fragXY = in.position.xy + vec2<f32>(u.viewportAndDt.w * 5.588238, u.viewportAndDt.w * 3.14159);
  let jitter = fract(52.9829189 * fract(0.06711056 * fragXY.x + 0.00583715 * fragXY.y));

  let camSurfH = evalSurfaceHeightOnly(ro.xz);
  let camRelY = ro.y - camSurfH;
  let airWeight = smoothstep(-0.004, 0.004, camRelY);

  let focalDist = max(0.25, u.lensAndRheology.x);
  let cosAxis = max(dot(rd, camFwd), 0.35);
  let tFocus = focalDist / cosAxis;
  let apRadius = max(0.0, u.lensAndRheology.y);

  // Single-pass thin-lens optical cone evaluation in linear HDR radiance (rgba16float):
  // Stores topside surface hit distance tSurf in .a so COMPOSITE_WGSL refines meniscus micro-glints,
  // topside drop reflections, and foreground drops/collars at 100% native 1:1 Retina resolution!
  let lensOut = evalLensRay(ro, rd, tFocus, apRadius, jitter, camRelY, airWeight, true);
  return vec4<f32>(clamp(lensOut.rgb, vec3<f32>(0.0), vec3<f32>(64.0)), clamp(lensOut.a, 0.0, 32.0));
}
`;

export const COMPOSITE_WGSL = /* wgsl */ `
override HDR_EXTENDED: u32 = 0u;
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> u: SimUniforms;
@group(0) @binding(1) var volTex: texture_2d<f32>;
@group(0) @binding(2) var linSampler: sampler;
@group(0) @binding(3) var waveTex: texture_2d<f32>;
@group(0) @binding(4) var spectralLutTex: texture_2d<f32>;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) ndc: vec2<f32>,
};

@vertex
fn vsMain(@builtin(vertex_index) vid: u32) -> VertexOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  var out: VertexOut;
  out.position = vec4<f32>(pos[vid], 0.0, 1.0);
  out.ndc = pos[vid];
  return out;
}

${SHARED_OPTICAL_DROP_WGSL}

fn pbrNeutralToneMap(colorIn: vec3<f32>) -> vec3<f32> {
  var color = max(colorIn, vec3<f32>(0.0));
  let startCompression = 0.80 - 0.04;
  let desaturation = 0.08;

  let x = min(color.r, min(color.g, color.b));
  let toe = select(0.04, x - 6.25 * x * x, x < 0.08);
  color = max(color - 0.55 * clamp(toe, 0.0, 0.04), vec3<f32>(0.0));

  let peak = max(color.r, max(color.g, color.b));
  if (peak < startCompression) {
    return color;
  }

  let d = 1.0 - startCompression;
  let newPeak = 1.0 - d * d / (peak + d - startCompression);
  color = color * (newPeak / max(peak, 1e-5));

  let g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
  return mix(color, vec3<f32>(newPeak), g);
}

fn linearToSRGB(c: vec3<f32>) -> vec3<f32> {
  let clamped = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
  let lo = clamped * 12.92;
  let hi = 1.055 * pow(clamped, vec3<f32>(1.0 / 2.4)) - 0.055;
  return select(hi, lo, clamped <= vec3<f32>(0.0031308));
}

/**
 * 5-tap Jorge Jimenez / Filmic Catmull-Rom bicubic HDR reconstruction clamped to the exact
 * 2x2 bracketing texel [boxMin, boxMax] interval (zero negative-lobe dark halos on speculars!).
 */
fn sampleVolCatmullRomClamped(uvIn: vec2<f32>) -> vec3<f32> {
  let dimsU = textureDimensions(volTex, 0);
  let dims = vec2<f32>(dimsU);
  let invDims = 1.0 / dims;
  let uv = clamp(uvIn, invDims * 0.5, vec2<f32>(1.0) - invDims * 0.5);

  let samplePos = uv * dims;
  let baseIdx = floor(samplePos - 0.5);
  let texPos1 = baseIdx + 0.5;
  let f = samplePos - texPos1;

  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);

  let w12 = w1 + w2;
  let offset12 = w2 / w12;

  let texPos0 = clamp((texPos1 - 1.0) * invDims, invDims * 0.5, vec2<f32>(1.0) - invDims * 0.5);
  let texPos3 = clamp((texPos1 + 2.0) * invDims, invDims * 0.5, vec2<f32>(1.0) - invDims * 0.5);
  let texPos12 = clamp((texPos1 + offset12) * invDims, invDims * 0.5, vec2<f32>(1.0) - invDims * 0.5);

  let c12_12 = textureSampleLevel(volTex, linSampler, vec2<f32>(texPos12.x, texPos12.y), 0.0).rgb * (w12.x * w12.y);
  let c0_12  = textureSampleLevel(volTex, linSampler, vec2<f32>(texPos0.x,  texPos12.y), 0.0).rgb * (w0.x  * w12.y);
  let c3_12  = textureSampleLevel(volTex, linSampler, vec2<f32>(texPos3.x,  texPos12.y), 0.0).rgb * (w3.x  * w12.y);
  let c12_0  = textureSampleLevel(volTex, linSampler, vec2<f32>(texPos12.x, texPos0.y),  0.0).rgb * (w12.x * w0.y);
  let c12_3  = textureSampleLevel(volTex, linSampler, vec2<f32>(texPos12.x, texPos3.y),  0.0).rgb * (w12.x * w3.y);

  let wSum = w12.x * w12.y + (w0.x + w3.x) * w12.y + w12.x * (w0.y + w3.y);
  let bicubic = (c12_12 + c0_12 + c3_12 + c12_0 + c12_3) / max(wSum, 1e-5);

  let maxCoord = vec2<i32>(dimsU) - vec2<i32>(1);
  let i1 = clamp(vec2<i32>(baseIdx), vec2<i32>(0), maxCoord);
  let i2 = min(i1 + vec2<i32>(1), maxCoord);
  let p00 = textureLoad(volTex, vec2<i32>(i1.x, i1.y), 0).rgb;
  let p10 = textureLoad(volTex, vec2<i32>(i2.x, i1.y), 0).rgb;
  let p01 = textureLoad(volTex, vec2<i32>(i1.x, i2.y), 0).rgb;
  let p11 = textureLoad(volTex, vec2<i32>(i2.x, i2.y), 0).rgb;
  let boxMin = min(min(p00, p10), min(p01, p11));
  let boxMax = max(max(p00, p10), max(p01, p11));

  return clamp(bicubic, boxMin, boxMax);
}

fn sampleVolSurfBounds(uvIn: vec2<f32>) -> vec3<f32> {
  let dimsU = textureDimensions(volTex, 0);
  let dims = vec2<f32>(dimsU);
  let baseIdx = floor(uvIn * dims - 0.5);
  let maxCoord = vec2<i32>(dimsU) - vec2<i32>(1);
  let i1 = clamp(vec2<i32>(baseIdx), vec2<i32>(0), maxCoord);
  let i2 = min(i1 + vec2<i32>(1), maxCoord);
  let a00 = textureLoad(volTex, vec2<i32>(i1.x, i1.y), 0).a;
  let a10 = textureLoad(volTex, vec2<i32>(i2.x, i1.y), 0).a;
  let a01 = textureLoad(volTex, vec2<i32>(i1.x, i2.y), 0).a;
  let a11 = textureLoad(volTex, vec2<i32>(i2.x, i2.y), 0).a;
  let minA = min(min(a00, a10), min(a01, a11));
  let maxA = max(max(a00, a10), max(a01, a11));
  return vec3<f32>(0.25 * (a00 + a10 + a01 + a11), minA, maxA);
}

fn sampleVolExtSurfSpan(uvIn: vec2<f32>, baseBounds: vec3<f32>) -> vec2<f32> {
  let dimsU = textureDimensions(volTex, 0);
  let dims = vec2<f32>(dimsU);
  let baseIdx = floor(uvIn * dims - 0.5);
  let maxCoord = vec2<i32>(dimsU) - vec2<i32>(1);
  let i1 = clamp(vec2<i32>(baseIdx), vec2<i32>(0), maxCoord);
  let i2 = min(i1 + vec2<i32>(1), maxCoord);
  let yUp = max(i1.y - 1, 0);
  let yDn = min(i2.y + 1, maxCoord.y);
  let aU0 = textureLoad(volTex, vec2<i32>(i1.x, yUp), 0).a;
  let aU1 = textureLoad(volTex, vec2<i32>(i2.x, yUp), 0).a;
  let aD0 = textureLoad(volTex, vec2<i32>(i1.x, yDn), 0).a;
  let aD1 = textureLoad(volTex, vec2<i32>(i2.x, yDn), 0).a;
  let extMin = min(baseBounds.y, min(min(aU0, aU1), min(aD0, aD1)));
  let extMax = max(baseBounds.z, max(max(aU0, aU1), max(aD0, aD1)));
  return vec2<f32>(extMin, extMax);
}

fn evalFlatWaterDistAtUV(uvQuery: vec2<f32>, ro: vec3<f32>, baseY: f32) -> f32 {
  let ndcQ = vec2<f32>(uvQuery.x * 2.0 - 1.0, 1.0 - uvQuery.y * 2.0);
  let pF4 = u.invViewProj * vec4<f32>(ndcQ, 1.0, 1.0);
  let dir = pF4.xyz / pF4.w - ro;
  let lenDir = max(length(dir), 1e-6);
  let rdY = dir.y / lenDir;
  if (ro.y > baseY && rdY < -1e-4) {
    let tF = (baseY - ro.y) / rdY;
    if (tF >= 0.002 && tF <= 12.0) {
      return tF;
    }
  }
  return 0.0;
}

/**
 * Measures flat-plane-detrended non-planar wave depth span across the 2x2 and 2x4 half-res
 * volTex neighborhoods so perspective foreshortening on calm water (dtFlat ~ t^2 / camRelY)
 * never triggers expensive silhouette ray-marching on flat water.
 */
fn sampleVolNonPlanarSpan(uvIn: vec2<f32>, ro: vec3<f32>, baseY: f32) -> vec2<f32> {
  let dimsU = textureDimensions(volTex, 0);
  let dims = vec2<f32>(dimsU);
  let invDims = 1.0 / dims;
  let baseIdx = floor(uvIn * dims - 0.5);
  let maxCoord = vec2<i32>(dimsU) - vec2<i32>(1);
  let i1 = clamp(vec2<i32>(baseIdx), vec2<i32>(0), maxCoord);
  let i2 = min(i1 + vec2<i32>(1), maxCoord);
  let yUp = max(i1.y - 1, 0);
  let yDn = min(i2.y + 1, maxCoord.y);

  var min2 = 1e9;
  var max2 = -1e9;
  var minE = 1e9;
  var maxE = -1e9;
  var valid2 = 0;
  var validE = 0;
  var anyHit = false;
  var anyMiss = false;

  let ys = array<i32, 4>(yUp, i1.y, i2.y, yDn);
  let xs = array<i32, 2>(i1.x, i2.x);
  for (var ry = 0; ry < 4; ry = ry + 1) {
    let cy = ys[ry];
    let isCoreY = ry == 1 || ry == 2;
    for (var rx = 0; rx < 2; rx = rx + 1) {
      let cx = xs[rx];
      let a = textureLoad(volTex, vec2<i32>(cx, cy), 0).a;
      let uvTex = (vec2<f32>(f32(cx), f32(cy)) + 0.5) * invDims;
      let tF = evalFlatWaterDistAtUV(uvTex, ro, baseY);
      if (a > 0.0) {
        anyHit = true;
        if (tF > 0.0) {
          let res = a - tF;
          minE = min(minE, res);
          maxE = max(maxE, res);
          validE = validE + 1;
          if (isCoreY) {
            min2 = min(min2, res);
            max2 = max(max2, res);
            valid2 = valid2 + 1;
          }
        } else {
          // Near-zero camRelY grazing ray hitting an elevated wave crest where tFlat == 0
          return vec2<f32>(1.0, 1.0);
        }
      } else {
        anyMiss = true;
      }
    }
  }

  let span2 = select(0.0, max2 - min2, valid2 >= 2);
  var spanE = select(0.0, maxE - minE, validE >= 2);
  if (anyHit && anyMiss && minE < 3.6) {
    spanE = max(spanE, 0.05);
  }
  return vec2<f32>(span2, spanE);
}

/**
 * Exact C2 cubic B-spline topside ray-surface intersection over a tight [tLo, tHi] bracket so
 * grazing swell silhouettes never skip sub-millimeter crest chords or facet on bilinear wave cells.
 */
fn intersectWaterSurfaceFine(ro: vec3<f32>, rd: vec3<f32>, tLoIn: f32, tHiIn: f32) -> f32 {
  let segLo = max(0.002, tLoIn);
  let segHi = max(segLo + 0.02, tHiIn);
  let steps = 36;
  let dt = (segHi - segLo) / f32(steps);
  var prevT = segLo;
  let p0 = ro + rd * prevT;
  var prevF = p0.y - evalSurfaceHeightSmooth(p0.xz);
  for (var i = 1; i <= steps; i = i + 1) {
    let currT = segLo + f32(i) * dt;
    let pC = ro + rd * currT;
    let currF = pC.y - evalSurfaceHeightSmooth(pC.xz);
    if (prevF >= 0.0 && currF <= 0.0) {
      var a = prevT;
      var b = currT;
      var fa = prevF;
      var fb = currF;
      for (var j = 0; j < 5; j = j + 1) {
        let mid = 0.5 * (a + b);
        let pM = ro + rd * mid;
        let fm = pM.y - evalSurfaceHeightSmooth(pM.xz);
        if (fa >= 0.0 && fm <= 0.0) {
          b = mid;
          fb = fm;
        } else {
          a = mid;
          fa = fm;
        }
      }
      let denom = fb - fa;
      let frac = select(0.5, clamp(-fa / denom, 0.0, 1.0), abs(denom) > 1e-6);
      return a + (b - a) * frac;
    }
    prevT = currT;
    prevF = currF;
  }
  return intersectWaterSurface(ro, rd, 0.002, 12.0, true);
}

/**
 * Depth-aware tSurf-bilateral upsample over the 4x4 half-res volTex neighborhood:
 * weights rgba texels by smooth Gaussian spatial kernel * exp(-abs(texel.a - tSurf) / sigmaDepth)
 * (detrended by flat-plane perspective gradient) so grazing swell crests never leak 2x2 half-res blocks.
 */
fn sampleVolDepthBilateral(uvIn: vec2<f32>, tRef: f32, ro: vec3<f32>, baseY: f32) -> vec3<f32> {
  let dimsU = textureDimensions(volTex, 0);
  let dims = vec2<f32>(dimsU);
  let invDims = 1.0 / dims;
  let maxCoord = vec2<i32>(dimsU) - vec2<i32>(1);
  let samplePos = uvIn * dims - 0.5;
  let base = vec2<i32>(floor(samplePos));
  let f = samplePos - vec2<f32>(base);
  let sigmaDepth = max(0.014, 0.011 * tRef);
  let tFlatRef = evalFlatWaterDistAtUV(uvIn, ro, baseY);

  var rgbSum = vec3<f32>(0.0);
  var wSum = 0.0;
  var fallbackRgb = vec3<f32>(0.0);
  var fallbackW = 0.0;

  for (var oy = -1; oy <= 2; oy = oy + 1) {
    for (var ox = -1; ox <= 2; ox = ox + 1) {
      let coord = clamp(base + vec2<i32>(ox, oy), vec2<i32>(0), maxCoord);
      let texel = textureLoad(volTex, coord, 0);
      let dTex = vec2<f32>(f32(ox), f32(oy)) - f;
      let wSpatial = exp(-0.82 * dot(dTex, dTex));
      fallbackRgb = fallbackRgb + texel.rgb * wSpatial;
      fallbackW = fallbackW + wSpatial;
      if (texel.a > 0.0) {
        let uvTex = (vec2<f32>(coord) + 0.5) * invDims;
        let tFlatTex = evalFlatWaterDistAtUV(uvTex, ro, baseY);
        let depthDiff = select(
          abs(texel.a - tRef),
          abs((texel.a - tFlatTex) - (tRef - tFlatRef)),
          tFlatRef > 0.0 && tFlatTex > 0.0
        );
        let wDepth = exp(-min(depthDiff / sigmaDepth, 14.0));
        let w = wSpatial * wDepth;
        rgbSum = rgbSum + texel.rgb * w;
        wSum = wSum + w;
      }
    }
  }

  let spatialAvg = fallbackRgb / max(fallbackW, 1e-5);
  if (wSum > 1e-4) {
    return rgbSum / wSum;
  }
  return spatialAvg;
}

@fragment
fn fsMain(in: VertexOut) -> @location(0) vec4<f32> {
  // WebGPU top-left UV convention: ndc (-1, +1) -> uv (0, 0), ndc (+1, -1) -> uv (1, 1)
  let uv = vec2<f32>(in.ndc.x * 0.5 + 0.5, 0.5 - in.ndc.y * 0.5);
  var finalLinear = sampleVolCatmullRomClamped(uv);

  // Native 1:1 Retina ray reconstruction (2560 x 1600) for foreground drops, collars, topside reflections & meniscus micro-glints!
  let ro = u.cameraPosAndTime.xyz;
  let pFar4 = u.invViewProj * vec4<f32>(in.ndc, 1.0, 1.0);
  let pFar = pFar4.xyz / pFar4.w;
  let rd = normalize(pFar - ro);

  let pC4 = u.invViewProj * vec4<f32>(0.0, 0.0, 0.5, 1.0);
  let pCenter = pC4.xyz / pC4.w;
  let camFwd = normalize(pCenter - ro);

  let camSurfH = evalSurfaceHeightOnly(ro.xz);
  let camRelY = ro.y - camSurfH;
  let airWeight = smoothstep(-0.004, 0.004, camRelY);

  let focalDist = max(0.25, u.lensAndRheology.x);
  let cosAxis = max(dot(rd, camFwd), 0.35);
  let tFocus = focalDist / cosAxis;
  let apRadius = max(0.0, u.lensAndRheology.y);

  var tSurf = -1.0;
  var surfBounds = vec3<f32>(0.0);
  var extBounds = vec2<f32>(0.0);
  var nonPlanarSpan = vec2<f32>(0.0);
  if (airWeight > 0.001 && camRelY > -0.004) {
    let baseY = u.surfaceParams.x;
    let tFlatCenter = select(-1.0, (baseY - ro.y) / rd.y, ro.y > baseY && rd.y < -1e-4);
    let outsideWaveDisk = tFlatCenter > 0.002 && tFlatCenter <= 12.0 && length((ro + rd * tFlatCenter).xz) > 1.66;
    if (outsideWaveDisk) {
      tSurf = tFlatCenter;
    } else {
      surfBounds = sampleVolSurfBounds(uv);
      extBounds = sampleVolExtSurfSpan(uv, surfBounds);
      nonPlanarSpan = sampleVolNonPlanarSpan(uv, ro, baseY);
      let extSpan = extBounds.y - extBounds.x;
      if (surfBounds.z > 0.0) {
        if (
          surfBounds.y > 0.0 &&
          ((surfBounds.z - surfBounds.y) < 0.005 || (nonPlanarSpan.x < 0.005 && nonPlanarSpan.y < 0.012)) &&
          abs(rd.y) > 0.025
        ) {
          tSurf = surfBounds.x;
          let invDy = 1.0 / rd.y;
          let p1 = ro + rd * tSurf;
          tSurf = clamp(tSurf + (evalSurfaceHeightSmooth(p1.xz) - p1.y) * invDy, surfBounds.y - 0.012, surfBounds.z + 0.012);
          let p2 = ro + rd * tSurf;
          tSurf = clamp(tSurf + (evalSurfaceHeightSmooth(p2.xz) - p2.y) * invDy, surfBounds.y - 0.016, surfBounds.z + 0.016);
        } else if (extBounds.x > 0.0 && extSpan < 0.85 && nonPlanarSpan.y >= 0.010) {
          tSurf = intersectWaterSurfaceFine(ro, rd, extBounds.x - 0.05, extBounds.y + 0.05);
        } else {
          tSurf = intersectWaterSurface(ro, rd, 0.002, 12.0, true);
        }
      } else if (rd.y < 0.06 && (ro.y + rd.y * 12.0) <= baseY + 0.145) {
        tSurf = intersectWaterSurface(ro, rd, 0.002, 12.0, true);
      }
    }
  }

  var dropHit: DropHit;
  dropHit.hit = false;
  dropHit.slotIdx = 0u;
  dropHit.tEnter = 1e9;
  dropHit.tExit = 0.0;
  dropHit.normal = vec3<f32>(0.0, 1.0, 0.0);
  dropHit.neckBlend = 1.0;
  dropHit.edgeCov = 0.0;
  if (camRelY > -0.02 && u.surfaceParams.z > 0.5) {
    dropHit = intersectAerialDrop(ro, rd, tFocus, apRadius);
  }

  if (airWeight > 0.001 && tSurf > 0.0) {
    let surfSpan = surfBounds.z - surfBounds.y;
    let extSpan = extBounds.y - extBounds.x;
    let nonPlanarGate = max(
      smoothstep(0.010, 0.024, nonPlanarSpan.x),
      smoothstep(0.016, 0.036, nonPlanarSpan.y)
    );
    let silhouetteBlend = nonPlanarGate * max(
      smoothstep(0.020, 0.044, surfSpan),
      smoothstep(0.032, 0.060, extSpan)
    );
    if (silhouetteBlend > 0.001 && (surfSpan > 0.020 || extSpan > 0.032)) {
      let invRetina = vec2<f32>(1.0) / max(u.viewportAndDt.xy, vec2<f32>(1.0));
      let tLoHint = select(0.002, extBounds.x - 0.05, extBounds.x > 0.0 && extSpan < 0.85);
      let tHiHint = select(12.0, extBounds.y + 0.05, extBounds.x > 0.0 && extSpan < 0.85);
      let useFine = extBounds.x > 0.0 && extSpan < 0.85;
      let subOffsets = array<vec2<f32>, 8>(
        vec2<f32>(-0.375, -0.4375),
        vec2<f32>( 0.125, -0.3125),
        vec2<f32>(-0.125, -0.1875),
        vec2<f32>( 0.375, -0.0625),
        vec2<f32>(-0.4375, 0.0625),
        vec2<f32>( 0.0625, 0.1875),
        vec2<f32>(-0.1875, 0.3125),
        vec2<f32>( 0.3125, 0.4375)
      );
      var aaColor = vec3<f32>(0.0);
      for (var s = 0u; s < 8u; s = s + 1u) {
        let off = subOffsets[s];
        let ndcSub = in.ndc + vec2<f32>(off.x * 2.0 * invRetina.x, -off.y * 2.0 * invRetina.y);
        let pSub4 = u.invViewProj * vec4<f32>(ndcSub, 1.0, 1.0);
        let rdSub = normalize(pSub4.xyz / pSub4.w - ro);
        var tSub = -1.0;
        if (useFine) {
          tSub = intersectWaterSurfaceFine(ro, rdSub, tLoHint, tHiHint);
        } else {
          tSub = intersectWaterSurface(ro, rdSub, 0.002, 12.0, true);
        }
        let tQuery = select(tSurf, tSub, tSub > 0.0);
        let uvSub = uv + off * invRetina;
        aaColor = aaColor + sampleVolDepthBilateral(uvSub, tQuery, ro, u.surfaceParams.x);
      }
      finalLinear = mix(finalLinear, aaColor * 0.125, silhouetteBlend * airWeight);
    }

    let darkTop = u.opticalParams.w > 0.5;
    let opticalDome = smoothstep(0.004, 0.058, camRelY);
    let pSurf = ro + rd * tSurf;
    let pWaveXZ = pSurf.xz;
    let sInfo = evalSurface(pWaveXZ);
    let slopeLen = length(sInfo.normal.xz);
    let waveSlope = smoothstep(0.00018, 0.010, slopeLen);
    let capillaryRing = smoothstep(0.0012, 0.018, slopeLen);
    let activeCurve = smoothstep(0.015, 0.14, slopeLen);
    let horizonFade = 1.0 - smoothstep(1.75, 4.2, length(pWaveXZ));

    // 1. Native 1:1 Retina meniscus wave micro-specular glints + crisp overhead studio softbox gloss on curved swells/craters
    let keyDir = select(
      normalize(vec3<f32>(0.22, 0.95, 0.20)),
      normalize(vec3<f32>(0.64, 0.70, 0.32)),
      darkTop
    );
    let softboxDir = normalize(vec3<f32>(0.14, 0.97, -0.18));
    let hWaveRaw = keyDir - rd;
    let hWave = hWaveRaw / max(length(hWaveRaw), 1e-5);
    let hSoftRaw = softboxDir - rd;
    let hSoft = hSoftRaw / max(length(hSoftRaw), 1e-5);
    let nDotHWave = max(dot(sInfo.normal, hWave), 0.0);
    let nDotHSoft = max(dot(sInfo.normal, hSoft), 0.0);
    let waterClear = select(
      smoothstep(0.016, 0.16, min(finalLinear.g, finalLinear.b)),
      1.0,
      darkTop
    );
    let steepDomeCurve = smoothstep(0.058, 0.18, slopeLen) * smoothstep(0.0022, 0.0085, abs(pSurf.y - u.surfaceParams.x));
    let softboxDome = (
      smoothstep(0.912, 0.974, nDotHSoft) * select(0.36, 0.16, darkTop) +
      pow(nDotHSoft, 115.0) * select(0.52, 0.24, darkTop)
    ) * steepDomeCurve;
    let capillarySheen = smoothstep(0.80, 0.982, max(nDotHWave, nDotHSoft))
                       * select(0.24, 0.14, darkTop) * capillaryRing * mix(0.45, 1.0, waterClear);
    let keySharpSpec = pow(nDotHWave, 155.0) * select(1.12, 0.94, darkTop) * (0.12 + 0.88 * waveSlope);
    let keyBroadSpec = (
      pow(nDotHWave, 76.0) * select(0.38, 0.36, darkTop) +
      pow(nDotHWave, 24.0) * select(0.135, 0.22, darkTop)
    ) * waveSlope * mix(0.35, 1.0, waterClear);
    let waveSpecRetina = (
      keySharpSpec +
      keyBroadSpec +
      softboxDome +
      capillarySheen
    ) * opticalDome * horizonFade;
    var topSurfaceCol = finalLinear + vec3<f32>(waveSpecRetina * 0.88, waveSpecRetina * 0.94, waveSpecRetina);

    // 2. Native 1:1 Retina topside water-surface reflection of aerial & pendant drops (fixed sInfo.height origin!)
    if (u.surfaceParams.z > 0.5) {
      let nSurf = normalize(mix(vec3<f32>(0.0, 1.0, 0.0), sInfo.normal, 0.35 + 0.65 * opticalDome));
      let cosI = clamp(dot(-rd, nSurf), 0.0, 1.0);
      let fresnel = fresnelDielectric(cosI, 1.0, 1.333);
      let reflDir = reflect(rd, nSurf);
      let skyRefl = evalStudioBackdrop(reflDir, false);
      let remFocusRefl = tFocus - tSurf;
      let pReflOrig = vec3<f32>(pWaveXZ.x, sInfo.height, pWaveXZ.y) + nSurf * 0.0020;
      let dropRefl = evalTopsideDropReflection(
        pReflOrig,
        reflDir,
        remFocusRefl,
        apRadius,
        waveSlope,
        sInfo.height,
        skyRefl
      );
      if (dropRefl.a > 0.002) {
        var contactReflRamp = 1.0;
        if (dropHit.hit && dropHit.tEnter > tSurf) {
          contactReflRamp = smoothstep(0.0, 0.018, dropHit.tEnter - tSurf);
        }
        let clearWaterGate = select(
          smoothstep(0.018, 0.28, min(topSurfaceCol.g, topSurfaceCol.b)),
          1.0,
          darkTop
        );
        let reflFresnel = clamp(fresnel * opticalDome * smoothstep(0.030, 0.12, -rd.y), 0.0, 0.28);
        let dropReflWeight = reflFresnel * dropRefl.a * contactReflRamp * horizonFade;
        let skyOccWeight = min(dropReflWeight * 0.06, 0.015) * clearWaterGate;
        let waterReflBase = select(
          max(topSurfaceCol - skyRefl * skyOccWeight, vec3<f32>(0.0)),
          topSurfaceCol,
          darkTop
        );
        let effReflBlend = dropReflWeight * select(mix(0.06, 0.65, clearWaterGate), 1.0, darkTop);
        let darkReflDelta = max(dropRefl.rgb - skyRefl, vec3<f32>(0.0)) * dropReflWeight;
        topSurfaceCol = select(
          mix(waterReflBase, dropRefl.rgb, effReflBlend),
          topSurfaceCol + darkReflDelta,
          darkTop
        );
      }
    }

    finalLinear = mix(finalLinear, topSurfaceCol, airWeight);
  }

  // 3. Native 1:1 Retina foreground aerial & pendant drops, concave capillary wetting collars, and borosilicate capillary pipette!
  if (dropHit.hit) {
    var dropVis = smoothstep(-0.02, 0.004, camRelY);
    if (tSurf > 0.0 && dropHit.tEnter > tSurf) {
      dropVis = dropVis * (1.0 - smoothstep(0.0005, 0.010, dropHit.tEnter - tSurf));
    }
    if (dropVis > 0.001) {
      let fgAlpha = smoothstep(0.0, 0.28, dropHit.neckBlend) * dropHit.edgeCov;
      if (fgAlpha < 0.992 && u.surfaceParams.z > 1.5) {
        let backDrop = intersectAerialDropFrom(ro, rd, dropHit.tExit + 0.0015, tFocus, apRadius);
        if (backDrop.hit) {
          var backVis = smoothstep(-0.02, 0.004, camRelY);
          if (tSurf > 0.0 && backDrop.tEnter > tSurf) {
            backVis = backVis * (1.0 - smoothstep(0.0005, 0.010, backDrop.tEnter - tSurf));
          }
          if (backVis > 0.001) {
            let shadedBack = shadeAerialDrop(ro, rd, backDrop, finalLinear);
            finalLinear = mix(finalLinear, shadedBack, backVis);
          }
        }
      }
      let shadedDrop = shadeAerialDrop(ro, rd, dropHit, finalLinear);
      finalLinear = mix(finalLinear, shadedDrop, dropVis);
    }
  }

  if (camRelY > -0.02 && u.surfaceParams.z > 0.5) {
    let pipAirVis = smoothstep(-0.02, 0.004, camRelY);
    let maxSurfT = select(12.0, tSurf, tSurf > 0.0);
    let maxPipT = select(maxSurfT, min(maxSurfT, dropHit.tEnter + 0.0005), dropHit.hit);
    let pipOut = evalCapillaryPipette(ro, rd, maxPipT, finalLinear);
    if (pipOut.w > 0.0) {
      finalLinear = mix(finalLinear, pipOut.rgb, pipAirVis);
    }
  }

  let rVig = length(in.ndc * vec2<f32>(0.82, 0.92));
  let lensVignette = 1.0 - 0.16 * smoothstep(0.55, 1.45, rVig);
  finalLinear = finalLinear * lensVignette;

  let mapped = pbrNeutralToneMap(finalLinear);
  if (HDR_EXTENDED != 0u) {
    // Preserve exact pbrNeutral IEC linear sRGB across [0, 0.76] while opening linear
    // extended-sRGB XDR headroom (> 1.0..4.0x) on studio strip glints and meniscus speculars
    let excess = max(finalLinear - vec3<f32>(0.76), vec3<f32>(0.0));
    let excessPeak = max(excess.r, max(excess.g, excess.b));
    let maxHeadroom = 3.0;
    let boostPeak = (excessPeak * maxHeadroom) / (excessPeak + maxHeadroom * 1.30 + 1e-4);
    let edrBoost = select(vec3<f32>(0.0), excess * (boostPeak / max(excessPeak, 1e-4)), excessPeak > 1e-4);
    return vec4<f32>(mapped + edrBoost, 1.0);
  }
  let srgb = linearToSRGB(mapped);

  // Sub-LSB (1/512) Interleaved Gradient Noise dither prevents 8-bit Mach banding in deep obsidian gradients
  let fragXY = in.position.xy + vec2<f32>(u.viewportAndDt.w * 3.17, u.viewportAndDt.w * 1.93);
  let ign = fract(52.9829189 * fract(0.06711056 * fragXY.x + 0.00583715 * fragXY.y));
  let dithered = clamp(srgb + vec3<f32>((ign - 0.5) * (1.0 / 512.0)), vec3<f32>(0.0), vec3<f32>(1.0));
  return vec4<f32>(dithered, 1.0);
}
`;


