import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

if (typeof globalThis.WebSocket === 'undefined') {
  const r = spawnSync(process.execPath, ['--experimental-websocket', ...process.argv.slice(1)], {
    stdio: 'inherit',
  });
  process.exit(r.status ?? 1);
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');
const OUT_DIR = path.join(ROOT_DIR, '.verify-out');

const QUICK = process.argv.includes('--quick');
const PRESET_MATRIX = process.argv.includes('--preset-matrix');
const LIVE_AUDIT = process.argv.includes('--live-audit');
const DEV_PORT = 5179;
const CDP_PORT = 9234 + (process.pid % 400);
const APP_URL = `http://localhost:${DEV_PORT}/?zen=1`;
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const USER_DIR = `/tmp/blood_verify_chrome_${process.pid}`;

fs.mkdirSync(OUT_DIR, { recursive: true });

function writeOut(filename, bufOrStr) {
  const outPath = path.join(OUT_DIR, filename);
  fs.writeFileSync(outPath, bufOrStr);
  return outPath;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function isBloodDropServerOpen(port) {
  for (const host of ['localhost', '127.0.0.1']) {
    try {
      const res = await fetch(`http://${host}:${port}/`, { signal: AbortSignal.timeout(600) });
      if (res.ok) {
        const html = await res.text();
        if (html.includes('Drop of Blood') || html.includes('1000 FPS SLOW-MO')) {
          return true;
        }
      }
    } catch {}
  }
  return false;
}

async function isPortInUse(port) {
  for (const host of ['localhost', '127.0.0.1']) {
    try {
      const res = await fetch(`http://${host}:${port}/`, { signal: AbortSignal.timeout(400) });
      if (res.status > 0) return true;
    } catch {}
  }
  return false;
}

let activeDevPort = DEV_PORT;
async function ensureDevServer() {
  for (let p = DEV_PORT; p < DEV_PORT + 15; p++) {
    if (await isBloodDropServerOpen(p)) {
      activeDevPort = p;
      return null;
    }
    if (!(await isPortInUse(p))) {
      activeDevPort = p;
      break;
    }
  }
  const viteBin = path.join(ROOT_DIR, 'node_modules', '.bin', 'vite');
  const devProc = spawn(viteBin, ['--port', String(activeDevPort), '--strictPort'], {
    cwd: ROOT_DIR,
    stdio: 'ignore',
  });
  for (let i = 0; i < 40; i++) {
    if (await isBloodDropServerOpen(activeDevPort)) return devProc;
    await sleep(150);
  }
  devProc.kill('SIGKILL');
  throw new Error(`Timed out waiting for Vite dev server on port ${activeDevPort}`);
}

async function getWsUrl() {
  for (let i = 0; i < 50; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`);
      const pages = await res.json();
      const page = pages.find((p) => p.type === 'page' && p.webSocketDebuggerUrl);
      if (page) return page.webSocketDebuggerUrl;
    } catch {}
    await sleep(150);
  }
  throw new Error('Timed out waiting for Chrome CDP');
}

class CdpClient {
  constructor(ws) {
    this.ws = ws;
    this.nextId = 1;
    this.pending = new Map();
    this.consoleErrors = [];
    this.pageErrors = [];
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(typeof ev.data === 'string' ? ev.data : ev.data.toString());
      if (msg.method === 'Runtime.consoleAPICalled' && msg.params?.type === 'error') {
        this.consoleErrors.push(
          (msg.params.args || []).map((a) => a.value ?? a.description ?? '').join(' ')
        );
      }
      if (msg.method === 'Runtime.exceptionThrown') {
        this.pageErrors.push(JSON.stringify(msg.params.exceptionDetails));
      }
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(JSON.stringify(msg.error)));
        else resolve(msg.result);
      }
    });
  }
  send(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  async eval(expr) {
    const res = await this.send('Runtime.evaluate', {
      expression: expr,
      awaitPromise: true,
      returnByValue: true,
    });
    if (res.exceptionDetails) {
      throw new Error('Runtime exception: ' + JSON.stringify(res.exceptionDetails));
    }
    return res.result?.value;
  }
}

async function buildContactSheet(cdp, outName, cols, rows, tiles) {
  const tileW = Math.floor(1920 / cols);
  const tileH = Math.round(tileW * (1600 / 2560)) + 34;
  const sheetW = tileW * cols;
  const sheetH = tileH * rows;
  const dataUrls = tiles.map((t) => ({
    label: t.label,
    b64: 'data:image/png;base64,' + fs.readFileSync(path.join(OUT_DIR, t.file)).toString('base64'),
  }));
  const jpgB64 = await cdp.eval(`(async () => {
    const items = ${JSON.stringify(dataUrls)};
    const cols = ${cols}, tileW = ${tileW}, tileH = ${tileH};
    const cv = document.createElement('canvas');
    cv.width = ${sheetW};
    cv.height = ${sheetH};
    const ctx = cv.getContext('2d');
    ctx.fillStyle = '#0f1115';
    ctx.fillRect(0, 0, cv.width, cv.height);
    for (let i = 0; i < items.length; i++) {
      const img = new Image();
      img.src = items[i].b64;
      await img.decode();
      const cx = (i % cols) * tileW;
      const cy = Math.floor(i / cols) * tileH;
      ctx.fillStyle = '#181b22';
      ctx.fillRect(cx + 2, cy + 2, tileW - 4, 30);
      ctx.fillStyle = '#e8ecf2';
      ctx.font = '600 14px -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.fillText(items[i].label, cx + 10, cy + 21);
      ctx.drawImage(img, cx + 2, cy + 32, tileW - 4, tileH - 34);
    }
    return cv.toDataURL('image/jpeg', 0.90).split(',')[1];
  })()`);
  return writeOut(outName, Buffer.from(jpgB64, 'base64'));
}

async function runPresetMatrix(cdp, snap) {
  // 1. Default 'hero' (Follow Shot) at t = 0.45s: Aerial drop before entry
  await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    eng.setLightingMode(0);
    eng.setCameraPreset('hero', true);
    eng.resetSimulation(1.0);
    await eng.seekToSimTime(0.45, 1.0);
    eng.camera.syncHeroFollow(0.45, true);
    eng.camera.updateMatrices(eng.canvas.width / eng.canvas.height);
  })()`);
  await snap('pm_01_hero_t0p45_aerial_drop.png');

  // 2. Default 'hero' at t = 1.15s: Crater & ripple propagation before submersion completes
  await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    await eng.seekToSimTime(1.15, 1.0);
    eng.camera.syncHeroFollow(1.15, true);
    eng.camera.updateMatrices(eng.canvas.width / eng.canvas.height);
  })()`);
  await snap('pm_02_hero_t1p15_ripples.png');

  // 3. Default 'hero' at t = 3.60s: Post-entry auto-follow state (exposes Rorschach TIR duplication if present)
  const heroSubState = await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    await eng.seekToSimTime(3.60, 1.0);
    eng.camera.syncHeroFollow(3.60, true);
    eng.camera.updateMatrices(eng.canvas.width / eng.canvas.height);
    return {
      camPosY: +eng.camera.position[1].toFixed(4),
      camElev: +eng.camera.elevation.toFixed(4),
      targetY: +eng.camera.target[1].toFixed(4),
    };
  })()`);
  await snap('pm_03_hero_t3p60_post_entry.png', { isShallowUnderwater: heroSubState.camPosY < 0.30 });

  // 4. Real 'crown' (Surface Crown) preset + real DOM pointerdown hold at canvas center (exposes PIPETTE_TIP_Y frustum clipping)
  const crownHold = await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    eng.resetSimulation(1.0);
    eng.multiDrops.clearSlot(eng.multiDrops.slots[0]);
    eng.setCameraPreset('crown', true);
    eng.camera.updateMatrices(eng.canvas.width / eng.canvas.height);
    const rect = eng.canvas.getBoundingClientRect();
    const cx = rect.left + rect.width * 0.5;
    const cy = rect.top + rect.height * 0.5;
    eng.canvas.dispatchEvent(new PointerEvent('pointerdown', {
      bubbles: true, clientX: cx, clientY: cy, pointerId: 1, pointerType: 'mouse', button: 0
    }));
    eng.paused = false;
    for (let i = 0; i < 40; i++) {
      window.__REAL_RENDER_FRAME__(1 / 60);
      await new Promise((r) => setTimeout(r, 12));
    }
    eng.paused = true;
    window.dispatchEvent(new PointerEvent('pointerup', {
      bubbles: true, clientX: cx, clientY: cy, pointerId: 1, pointerType: 'mouse', button: 0
    }));
    // Compute NDC Y of PIPETTE_TIP_Y = 0.55 at x=0, z=0
    const vp = eng.camera.viewProjMatrix;
    const clipY = vp[1] * 0 + vp[5] * 0.55 + vp[9] * 0 + vp[13];
    const clipW = vp[3] * 0 + vp[7] * 0.55 + vp[11] * 0 + vp[15];
    return { pipetteTipNdcY: +(clipY / clipW).toFixed(4), visibleInFrustum: Math.abs(clipY / clipW) <= 0.98 };
  })()`);
  await snap('pm_04_crown_real_dom_hold.png', { isHeldPipette: true });

  // 5. Real 'crown' off-center impact at [+0.18, -0.10] (exposes world-origin (0,0) locked ripple shading)
  await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    eng.resetSimulation(1.0);
    eng.multiDrops.clearSlot(eng.multiDrops.slots[0]);
    eng.setCameraPreset('crown', true);
    eng.triggerAdditionalDrop([0.18, -0.10], 0.058);
    for (let i = 0; i < 68; i++) {
      eng.paused = false;
      window.__REAL_RENDER_FRAME__(1 / 60);
    }
    eng.paused = true;
    eng.camera.updateMatrices(eng.canvas.width / eng.canvas.height);
  })()`);
  await snap('pm_05_crown_offcenter_ripples.png');

  // 6. Real 'abyss' (Sub-Meniscus) preset at t = 3.40s
  await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    eng.resetSimulation(1.0);
    eng.setCameraPreset('abyss', true);
    await eng.seekToSimTime(3.40, 1.0);
    eng.camera.updateMatrices(eng.canvas.width / eng.canvas.height);
  })()`);
  await snap('pm_06_abyss_t3p40_submeniscus.png', { isShallowUnderwater: true });

  // 7. Unpaused 6-frame live temporal entry strip (t = 0.72s -> 1.47s in 0.15s steps) under 'hero'
  const stripFiles = [];
  await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    eng.setLightingMode(0);
    eng.setCameraPreset('hero', true);
    eng.resetSimulation(1.0);
    await eng.seekToSimTime(0.72, 1.0);
  })()`);
  for (let k = 0; k < 6; k++) {
    if (k > 0) {
      await cdp.eval(`(async () => {
        const eng = window.__BLOOD_ENGINE__;
        eng.paused = false;
        for (let s = 0; s < 9; s++) {
          window.__REAL_RENDER_FRAME__(1 / 60);
        }
        eng.paused = true;
        eng.camera.syncHeroFollow(eng.multiDrops.simTime, true);
        eng.camera.updateMatrices(eng.canvas.width / eng.canvas.height);
      })()`);
    }
    const fname = `pm_strip_${k}_t${(0.72 + k * 0.15).toFixed(2)}s.png`;
    await snap(fname);
    stripFiles.push({ file: fname, label: `Live rAF t=${(0.72 + k * 0.15).toFixed(2)}s` });
  }

  const sheet1 = await buildContactSheet(cdp, 'preset_matrix_sheet.jpg', 3, 2, [
    { file: 'pm_01_hero_t0p45_aerial_drop.png', label: 'Hero Preset t=0.45s (Aerial Drop)' },
    { file: 'pm_02_hero_t1p15_ripples.png', label: 'Hero Preset t=1.15s (Entry Ripples)' },
    { file: 'pm_03_hero_t3p60_post_entry.png', label: `Hero t=3.60s (elev=${heroSubState.camElev})` },
    { file: 'pm_04_crown_real_dom_hold.png', label: `Crown DOM Hold (tipNdcY=${crownHold.pipetteTipNdcY})` },
    { file: 'pm_05_crown_offcenter_ripples.png', label: 'Crown Off-Center [+0.18,-0.10] Ripples' },
    { file: 'pm_06_abyss_t3p40_submeniscus.png', label: 'Abyss Preset t=3.40s (Sub-Meniscus)' },
  ]);
  const sheet2 = await buildContactSheet(cdp, 'temporal_entry_strip.jpg', 3, 2, stripFiles);

  return { heroSubState, crownHold, sheets: [sheet1, sheet2] };
}

async function runLiveExplorationAudit(cdp, snap) {
  // 1. Quick click pipette squeeze + XDR extended-sRGB midtone parity & specular headroom
  const quickClickState = await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    eng.setLightingMode(0);
    eng.setCameraPreset('hero', true);
    eng.resetSimulation(1.0);
    eng.multiDrops.clearSlot(eng.multiDrops.slots[0]);
    const rect = eng.canvas.getBoundingClientRect();
    const cx = rect.left + rect.width * 0.50;
    const cy = rect.top + rect.height * 0.46;
    eng.canvas.dispatchEvent(new PointerEvent('pointerdown', {
      bubbles: true, clientX: cx, clientY: cy, pointerId: 2, pointerType: 'mouse', button: 0
    }));
    window.dispatchEvent(new PointerEvent('pointerup', {
      bubbles: true, clientX: cx, clientY: cy, pointerId: 2, pointerType: 'mouse', button: 0
    }));
    eng.paused = false;
    for (let i = 0; i < 6; i++) {
      window.__REAL_RENDER_FRAME__(1 / 60);
    }
    eng.paused = true;
    eng.camera.updateMatrices(eng.canvas.width / eng.canvas.height);
    const activeSlot = eng.multiDrops.slots.find((s) => s.active);
    return {
      mode: activeSlot?.mode ?? 'none',
      centerY: activeSlot ? +activeSlot.center[1].toFixed(4) : null,
      pipettePresence: activeSlot ? +activeSlot.pipettePresence.toFixed(3) : 0,
    };
  })()`);
  await snap('lv_00_quick_pipette_squeeze.png');
  const xdrCheck = await cdp.eval('window.__QA__.verifyExtendedHdrToggle()');

  // 2. Settling ripples decay monotonically without inward drumhead sonar bounce or red clear-water sweep
  await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    eng.setLightingMode(0);
    eng.setCameraPreset('crown', true);
    eng.resetSimulation(1.0);
    await eng.seekToSimTime(2.00, 1.0);
    eng.camera.updateMatrices(eng.canvas.width / eng.canvas.height);
  })()`);
  const wave2p0 = await cdp.eval('window.__QA__.readWaveStats()');
  await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    await eng.seekToSimTime(3.20, 1.0);
  })()`);
  const wave3p2 = await cdp.eval('window.__QA__.readWaveStats()');
  await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    await eng.seekToSimTime(4.20, 1.0);
    eng.camera.updateMatrices(eng.canvas.width / eng.canvas.height);
  })()`);
  const wave4p2 = await cdp.eval('window.__QA__.readWaveStats()');
  await snap('lv_01_settling_ripples_t4p2s.png', { checkClearOuterRipples: true });
  await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    await eng.seekToSimTime(5.40, 1.0);
  })()`);
  const wave5p4 = await cdp.eval('window.__QA__.readWaveStats()');

  // 3. Multi-drop organic plume unfurling (3 drops across [-0.27, -0.01, +0.26] at t = 8.5s)
  await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    eng.setLightingMode(0);
    eng.setCameraPreset('hero', true);
    eng.resetSimulation(1.3);
    eng.multiDrops.clearSlot(eng.multiDrops.slots[0]);
    eng.triggerQuickPipetteDrop([-0.27, 0.03], 0.056);
    await eng.seekToSimTime(1.45, 1.3);
    eng.triggerQuickPipetteDrop([-0.01, -0.02], 0.060);
    await eng.seekToSimTime(2.90, 1.3);
    eng.triggerQuickPipetteDrop([0.26, 0.04], 0.054);
    await eng.seekToSimTime(8.50, 1.3);
    window.__QA__.setCam(0.04, 0.24, 1.58, [0.0, -0.14, 0.0]);
  })()`);
  await snap('lv_02_multi_drop_plumes_t8p5s.png');

  // 4. Post-11s aging & hemolysis (t = 19.0s -> zero olive/chartreuse yellow-green streaks!)
  await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    await eng.seekToSimTime(19.00, 1.3);
    window.__QA__.setCam(0.04, 0.22, 1.56, [0.0, -0.22, 0.0]);
  })()`);
  await snap('lv_03_aging_hemolysis_t19s.png');

  // 5. Low floor orbit & borosilicate glass cuvette base (clamped above y = -0.88, flat crimson sediment)
  const floorOrbitState = await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    await eng.seekToSimTime(21.50, 1.3);
    window.__QA__.setCam(0.28, -0.22, 1.48, [0.0, -0.52, 0.0]);
    eng.camera.orbit(0.0, -6.0);
    eng.camera.updateMatrices(eng.canvas.width / eng.canvas.height);
    return {
      camPosY: +eng.camera.position[1].toFixed(4),
      camElev: +eng.camera.elevation.toFixed(4),
    };
  })()`);
  await snap('lv_04_cuvette_floor_orbit.png');

  // 6. Side-on eye-level aquarium view at t = 2.30s (full 3D vortex plume through vertical cuvette wall, not 11px sliver!)
  await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    eng.setLightingMode(0);
    eng.resetSimulation(1.0);
    await eng.seekToSimTime(2.30, 1.0);
    window.__QA__.setCam(0.42, 0.03, 1.44, [0.0, -0.04, 0.0]);
  })()`);
  await snap('lv_05_sideon_aquarium_t2p3s.png');

  // 7. Top-down camera view (elev = +1.36) across multiple drops + late settling (verifying zero bioluminescent bloom, zero square checkerboard, and full-canvas ripples)
  await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    eng.setLightingMode(0);
    eng.resetSimulation(1.3);
    eng.multiDrops.clearSlot(eng.multiDrops.slots[0]);
    eng.triggerQuickPipetteDrop([-0.14, 0.05], 0.056);
    await eng.seekToSimTime(1.10, 1.3);
    eng.triggerQuickPipetteDrop([0.11, -0.06], 0.060);
    await eng.seekToSimTime(2.05, 1.3);
    eng.triggerQuickPipetteDrop([-0.03, -0.13], 0.054);
    await eng.seekToSimTime(2.65, 1.3);
    window.__QA__.setCam(0.0, 1.36, 1.32, [0.0, 0.10, 0.0]);
  })()`);
  await snap('lv_06_topdown_multidrop.png');
  await cdp.eval(`(async () => {
    const eng = window.__BLOOD_ENGINE__;
    eng.resetSimulation(1.0);
    await eng.seekToSimTime(3.35, 1.0);
    window.__QA__.setCam(0.0, 1.36, 1.32, [0.0, 0.10, 0.0]);
  })()`);
  await snap('lv_07_topdown_settling_t3p35s.png');

  const sheet = await buildContactSheet(cdp, 'live_fixes_sheet.jpg', 3, 2, [
    { file: 'lv_00_quick_pipette_squeeze.png', label: '1. Quick Pipette Squeeze + XDR sRGB' },
    { file: 'lv_01_settling_ripples_t4p2s.png', label: '5. Settling Ripples (0 Red Sonar Echo)' },
    { file: 'lv_06_topdown_multidrop.png', label: 'Top-Down Multi-Drop (Full-Canvas Ripples)' },
    { file: 'lv_07_topdown_settling_t3p35s.png', label: 'Top-Down t=3.35s (0 Square Checkerboard)' },
    { file: 'lv_02_multi_drop_plumes_t8p5s.png', label: '2. 3-Drop Unfurling Curtains (t=8.5s)' },
    { file: 'lv_05_sideon_aquarium_t2p3s.png', label: '4C. Side-On Aquarium View (t=2.3s)' },
  ]);

  return {
    quickClickState,
    xdrCheck,
    waveDecay: { wave2p0, wave3p2, wave4p2, wave5p4 },
    floorOrbitState,
    sheet,
  };
}

async function main() {
  const devServer = await ensureDevServer();
  const chrome = spawn(
    CHROME,
    [
      '--headless=new',
      `--remote-debugging-port=${CDP_PORT}`,
      '--remote-allow-origins=*',
      `--user-data-dir=${USER_DIR}`,
      '--no-first-run',
      '--no-default-browser-check',
      '--use-angle=metal',
      '--enable-unsafe-webgpu',
      '--ignore-gpu-blocklist',
      '--enable-gpu-rasterization',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
      '--window-size=1280,800',
      '--force-device-scale-factor=2',
      `http://localhost:${activeDevPort}/?zen=1`,
    ],
    { stdio: 'ignore' }
  );

  try {
    const wsUrl = await getWsUrl();
    const ws = new WebSocket(wsUrl);
    await new Promise((res, rej) => {
      ws.addEventListener('open', res, { once: true });
      ws.addEventListener('error', rej, { once: true });
    });
    const cdp = new CdpClient(ws);
    await cdp.send('Page.enable');
    await cdp.send('Runtime.enable');
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width: 1280,
      height: 800,
      deviceScaleFactor: 2,
      mobile: false,
    });

    for (let i = 0; i < 60; i++) {
      const ready = await cdp.eval('Boolean(window.__BLOOD_ENGINE__)');
      if (ready) break;
      await sleep(150);
    }
    const engineExists = await cdp.eval('Boolean(window.__BLOOD_ENGINE__)');
    if (!engineExists) {
      const domText = await cdp.eval('document.body ? document.body.innerText : ""');
      throw new Error(
        `window.__BLOOD_ENGINE__ failed to initialize!\nDOM: ${domText}\nConsoleErrors: ${JSON.stringify(cdp.consoleErrors)}\nPageErrors: ${JSON.stringify(cdp.pageErrors)}`
      );
    }

    await cdp.eval(`(async () => {
      const eng = window.__BLOOD_ENGINE__;
      const realRenderFrame = eng.renderFrame.bind(eng);
      window.__REAL_RENDER_FRAME__ = realRenderFrame;
      eng.renderFrame = () => ({ simTime: eng.multiDrops.simTime, phaseLabel: 'QA' });
      eng.paused = true;
      await eng.device.queue.onSubmittedWorkDone();

      const adapter = await navigator.gpu.requestAdapter();
      const info = adapter?.info || {};
      window.__INIT_INFO__ = {
        canvasWidth: eng.canvas.width,
        canvasHeight: eng.canvas.height,
        format: eng.format,
        isExtendedHdr: eng.isExtendedHdr,
        supportsTimestamps: eng.supportsTimestamps,
        autoLoop: eng.autoLoop,
        validationErrors: [...eng.validationErrors],
        adapterInfo: {
          vendor: info.vendor || '',
          architecture: info.architecture || '',
          device: info.device || '',
          description: info.description || '',
        },
      };

      const w = eng.canvas.width, h = eng.canvas.height;
      const pixFormat = eng.format;
      const isFloat16Swap = pixFormat === 'rgba16float';
      const bpp = isFloat16Swap ? 8 : 4;
      const bytesPerRow = Math.ceil((w * bpp) / 256) * 256;
      const pixTex = eng.device.createTexture({
        size: [w, h, 1],
        format: pixFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      const pixBuf = eng.device.createBuffer({
        size: bytesPerRow * h,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      const waveProbeMod = eng.device.createShaderModule({
        code: \`
          @group(0) @binding(0) var waveTex: texture_2d<f32>;
          @group(0) @binding(1) var<storage, read_write> outBuf: array<vec4<f32>>;
          @compute @workgroup_size(64)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            if (gid.x < 64u) {
              let dims = vec2<i32>(textureDimensions(waveTex));
              let halfDim = dims.x / 2;
              let stride = max(1, dims.x / 256);
              outBuf[gid.x] = textureLoad(waveTex, vec2<i32>(halfDim + i32(gid.x) * stride, halfDim), 0);
            }
          }
        \`
      });
      const waveProbePipe = eng.device.createComputePipeline({
        layout: 'auto',
        compute: { module: waveProbeMod, entryPoint: 'main' }
      });
      const waveOutBuf = eng.device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const waveReadBuf = eng.device.createBuffer({ size: 1024, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

      const concProbeMod = eng.device.createShaderModule({
        code: \`
          @group(0) @binding(0) var concTex: texture_3d<f32>;
          @group(0) @binding(1) var<storage, read_write> outSlice: array<vec4<f32>>;
          @compute @workgroup_size(64)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let iy = i32(gid.x);
            if (iy >= 256) { return; }
            var sum = vec4<f32>(0.0);
            for (var iz = 0; iz < 192; iz = iz + 2) {
              for (var ix = 0; ix < 192; ix = ix + 2) {
                let c = textureLoad(concTex, vec3<i32>(ix, iy, iz), 0);
                sum = sum + vec4<f32>(c.r, c.g, c.a, select(0.0, 1.0, c.r + c.g > 0.00025));
              }
            }
            outSlice[gid.x] = sum * 4.0;
          }
        \`
      });
      const concProbePipe = eng.device.createComputePipeline({
        layout: 'auto',
        compute: { module: concProbeMod, entryPoint: 'main' }
      });
      const concOutBuf = eng.device.createBuffer({ size: 4096, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const concReadBuf = eng.device.createBuffer({ size: 4096, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

      window.__QA__ = {
        setCam(az, el, dist, target) {
          eng.camera.followHeroEnabled = false;
          eng.camera.azimuth = az;
          eng.camera.elevation = el;
          eng.camera.distance = dist;
          eng.camera.target = [...target];
          eng.camera.targetAzimuth = az;
          eng.camera.targetElevation = el;
          eng.camera.targetDistance = dist;
          eng.camera.targetCenter = [...target];
          eng.camera.updateMatrices(eng.canvas.width / eng.canvas.height);
        },

        async readWaveStats() {
          const bg = eng.device.createBindGroup({
            layout: waveProbePipe.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: eng.texWave[eng.waveState].createView() },
              { binding: 1, resource: { buffer: waveOutBuf } }
            ]
          });
          const enc = eng.device.createCommandEncoder();
          const pass = enc.beginComputePass();
          pass.setPipeline(waveProbePipe);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(1);
          pass.end();
          enc.copyBufferToBuffer(waveOutBuf, 0, waveReadBuf, 0, 1024);
          eng.device.queue.submit([enc.finish()]);
          await waveReadBuf.mapAsync(GPUMapMode.READ);
          const f = new Float32Array(waveReadBuf.getMappedRange().slice(0));
          waveReadBuf.unmap();
          let minEta = Infinity, maxEta = -Infinity, maxSlope = 0, sumE2 = 0;
          for (let i = 0; i < 64; i++) {
            const e = f[i * 4];
            sumE2 += e * e;
            if (e < minEta) minEta = e;
            if (e > maxEta) maxEta = e;
            if (i > 0) {
              const s = Math.abs(e - f[(i - 1) * 4]) / (3.2 / 256.0);
              if (s > maxSlope) maxSlope = s;
            }
          }
          return {
            centerEta: +f[0].toFixed(5),
            minEta: +minEta.toFixed(5),
            maxEta: +maxEta.toFixed(5),
            rmsEta: +Math.sqrt(sumE2 / 64).toFixed(6),
            maxSlope: +maxSlope.toFixed(4),
          };
        },

        async readConcSlabs() {
          const bg = eng.device.createBindGroup({
            layout: concProbePipe.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: eng.texConc[eng.concState].createView() },
              { binding: 1, resource: { buffer: concOutBuf } }
            ]
          });
          const enc = eng.device.createCommandEncoder();
          const pass = enc.beginComputePass();
          pass.setPipeline(concProbePipe);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(4);
          pass.end();
          enc.copyBufferToBuffer(concOutBuf, 0, concReadBuf, 0, 4096);
          eng.device.queue.submit([enc.finish()]);
          await concReadBuf.mapAsync(GPUMapMode.READ);
          const f = new Float32Array(concReadBuf.getMappedRange().slice(0));
          concReadBuf.unmap();
          const sumRange = (y0, y1) => {
            let r = 0, g = 0, bed = 0, vox = 0;
            for (let y = y0; y <= y1; y++) {
              r += f[y * 4 + 0];
              g += f[y * 4 + 1];
              bed += f[y * 4 + 2];
              vox += f[y * 4 + 3];
            }
            return { intact: +r.toFixed(1), lysed: +g.toFixed(1), bed: +bed.toFixed(1), occupiedVox: Math.round(vox) };
          };
          let iyLow = -1, iyHigh = -1;
          for (let y = 20; y <= 253; y++) {
            if (f[y * 4 + 3] >= 4) {
              if (iyLow < 0) iyLow = y;
              iyHigh = y;
            }
          }
          let emptySlicesInSpan = 0;
          if (iyLow >= 0 && iyHigh > iyLow) {
            for (let y = iyLow; y <= iyHigh; y++) {
              if (f[y * 4 + 3] < 4) emptySlicesInSpan++;
            }
          }
          return {
            floor_0_18: sumRange(0, 18),
            lower_19_85: sumRange(19, 85),
            mid_86_170: sumRange(86, 170),
            upper_171_255: sumRange(171, 255),
            voxVerticalContinuity: { iyLow, iyHigh, emptySlicesInSpan },
          };
        },

        async renderAndMeasure(opts = {}) {
          eng.paused = true;
          eng.camera.updateMatrices(w / h);
          await new Promise((resolve) => {
            requestAnimationFrame(() => {
              eng.uploadSnapshotUniforms(0.0, eng.spawnCursor, 0);
              const enc = eng.device.createCommandEncoder();
              const lPass = enc.beginComputePass();
              eng.encodeLightingPasses(lPass);
              lPass.end();
              const swapView = eng.context.getCurrentTexture().createView();
              eng.encodeOpticalRenderPasses(enc, swapView);
              eng.encodeOpticalRenderPasses(enc, pixTex.createView());
              enc.copyTextureToBuffer({ texture: pixTex }, { buffer: pixBuf, bytesPerRow, rowsPerImage: h }, [w, h, 1]);
              eng.device.queue.submit([enc.finish()]);
              resolve();
            });
          });
          await pixBuf.mapAsync(GPUMapMode.READ);
          const mappedRange = pixBuf.getMappedRange();
          const raw = isFloat16Swap ? null : new Uint8Array(mappedRange);
          const rawF16 = isFloat16Swap ? new Float16Array(mappedRange) : null;
          const f16RowStride = bytesPerRow >> 1;
          const isBGRA = pixFormat.startsWith('bgra');
          const linToSrgbByte = (v) => {
            const c = v <= 0 ? 0 : v >= 1 ? 1 : v;
            const s = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
            return Math.round(s * 255);
          };

          let maxHdrLinearPeak = 0;
          let whiteEyeballCutoutPx = 0;
          let muddyYellowTanSmokePx = 0;
          let magentaWaterRipplePx = 0;
          let yellowGreenStreakPx = 0;
          let clearWaterRedRipplePx = 0;
          let upperHalfBloodPx = 0;
          let lowerHalfBloodPx = 0;
          let minBloodY = h;
          let maxBloodY = -1;

          const isShallowUnderwater = Boolean(opts.isShallowUnderwater);
          const isDarkField = Boolean(opts.isDarkField);
          const checkStem = Boolean(opts.checkStemContinuity);
          const checkOblate = Boolean(opts.checkBottomOblateness);
          const checkClearOuterRipples = Boolean(opts.checkClearOuterRipples);
          const rowBloodCounts = checkStem ? new Int32Array(h) : null;
          const xStemMin = Math.round(w * 0.25);
          const xStemMax = Math.round(w * 0.75);
          const oblateY0 = checkOblate ? Math.round(h * (opts.oblateMinYFrac ?? 0.38)) : 0;
          const oblateY1 = checkOblate ? Math.round(h * (opts.oblateMaxYFrac ?? 0.88)) : 0;
          const colBottomY = checkOblate ? new Int32Array(w).fill(-1) : null;
          let oblateSumX = 0;
          let oblateCnt = 0;

          for (let y = 0; y < h; y++) {
            const row = y * bytesPerRow;
            const rowF16 = y * f16RowStride;
            let rowBlood = 0;
            for (let x = 0; x < w; x++) {
              let r, g, b;
              if (rawF16) {
                const j = rowF16 + x * 4;
                const lr = rawF16[j];
                const lg = rawF16[j + 1];
                const lb = rawF16[j + 2];
                const pk = Math.max(lr, lg, lb);
                if (pk > maxHdrLinearPeak) maxHdrLinearPeak = pk;
                // rgba16float swapchain uses extended-sRGB transfer directly on [0, 1]
                r = Math.round(Math.min(1, Math.max(0, lr)) * 255);
                g = Math.round(Math.min(1, Math.max(0, lg)) * 255);
                b = Math.round(Math.min(1, Math.max(0, lb)) * 255);
              } else {
                const i = row + x * 4;
                r = isBGRA ? raw[i + 2] : raw[i + 0];
                g = raw[i + 1];
                b = isBGRA ? raw[i + 0] : raw[i + 2];
              }

              if (isShallowUnderwater && y > h * 0.04 && y < h * 0.42 && x > w * 0.20 && x < w * 0.80) {
                if (r > 218 && g > 224 && b > 230) whiteEyeballCutoutPx++;
              }
              if (r > 150 && g > 105 && b > 55 && b < 175 && (r - b) > 35 && (r - g) < 90 && (g - b) > 20) {
                muddyYellowTanSmokePx++;
              }
              if (isDarkField && y < h * 0.34 && r > 70 && b > 65 && b > g + 10 && r > g + 12 && r < b * 1.55) {
                magentaWaterRipplePx++;
              }
              if (!isDarkField && g >= 48 && g > r + 6 && g > b + 6) {
                yellowGreenStreakPx++;
              }
              if (
                checkClearOuterRipples &&
                y > h * 0.14 &&
                y < h * 0.36 &&
                Math.abs(x - 0.5 * w) > 0.18 * w &&
                Math.abs(x - 0.5 * w) < 0.40 * w
              ) {
                if (r > b + 8 || r > g + 14) {
                  clearWaterRedRipplePx++;
                }
              }

              const isBlood = isDarkField
                ? (r >= 16 && r > g * 1.28 && r > b * 1.55)
                : ((r > g + 12 && r > b + 12) || (r >= 16 && g < 145 && b < 150 && r > g * 1.38));

              if (isBlood) {
                if (y < h * 0.46) upperHalfBloodPx++;
                else lowerHalfBloodPx++;
                if (x > w * 0.15 && x < w * 0.85) {
                  if (y < minBloodY) minBloodY = y;
                  if (y > maxBloodY) maxBloodY = y;
                }
              }

              if (isBlood && (checkStem || checkOblate) && x >= xStemMin && x <= xStemMax) {
                if (checkStem) rowBlood++;
                if (checkOblate && y >= oblateY0 && y <= oblateY1) {
                  oblateSumX += x;
                  oblateCnt++;
                  if (y > colBottomY[x]) colBottomY[x] = y;
                }
              }
            }
            if (checkStem) rowBloodCounts[y] = rowBlood;
          }
          pixBuf.unmap();

          let stemContinuity = null;
          if (checkStem && rowBloodCounts) {
            const minScanY = Math.round(h * (opts.stemMinYFrac ?? 0.22));
            const maxScanY = Math.round(h * (opts.stemMaxYFrac ?? 0.88));
            let topPlumeRow = -1;
            let bottomPlumeRow = -1;
            for (let y = minScanY; y <= maxScanY; y++) {
              if (rowBloodCounts[y] >= 12) { topPlumeRow = y; break; }
            }
            for (let y = maxScanY; y >= minScanY; y--) {
              if (rowBloodCounts[y] >= 24) { bottomPlumeRow = y; break; }
            }
            let emptyGapScanlines = 0;
            if (topPlumeRow >= 0 && bottomPlumeRow > topPlumeRow) {
              for (let y = topPlumeRow; y <= bottomPlumeRow; y++) {
                if (rowBloodCounts[y] === 0) emptyGapScanlines++;
              }
            }
            stemContinuity = {
              topPlumeRow,
              bottomPlumeRow,
              emptyGapScanlines,
              continuous: topPlumeRow >= 0 && bottomPlumeRow > topPlumeRow + 150 && emptyGapScanlines === 0,
            };
          }

          let bottomOblateness = null;
          if (checkOblate && colBottomY && oblateCnt > 100) {
            const plumeCenterX = Math.round(oblateSumX / oblateCnt);
            let centerMaxY = -1, leftLobeMaxY = -1, rightLobeMaxY = -1;
            for (let dx = -18; dx <= 18; dx++) {
              const cx = plumeCenterX + dx;
              if (cx >= 0 && cx < w && colBottomY[cx] > centerMaxY) centerMaxY = colBottomY[cx];
            }
            for (let dx = -130; dx <= -45; dx++) {
              const cx = plumeCenterX + dx;
              if (cx >= 0 && cx < w && colBottomY[cx] > leftLobeMaxY) leftLobeMaxY = colBottomY[cx];
            }
            for (let dx = 45; dx <= 130; dx++) {
              const cx = plumeCenterX + dx;
              if (cx >= 0 && cx < w && colBottomY[cx] > rightLobeMaxY) rightLobeMaxY = colBottomY[cx];
            }
            const lobeMeanMaxY = +(0.5 * (leftLobeMaxY + rightLobeMaxY)).toFixed(1);
            const centerSpearExcessPx = +(centerMaxY - lobeMeanMaxY).toFixed(1);
            bottomOblateness = {
              centerMaxY,
              lobeMeanMaxY,
              centerSpearExcessPx,
              oblatePass: centerSpearExcessPx <= 14,
            };
          }

          return {
            width: w,
            height: h,
            maxHdrLinearPeak: isFloat16Swap ? +maxHdrLinearPeak.toFixed(4) : null,
            whiteEyeballCutoutPx,
            muddyYellowTanSmokePx,
            magentaWaterRipplePx,
            yellowGreenStreakPx,
            clearWaterRedRipplePx,
            plumeVerticalSpanPx: maxBloodY >= minBloodY ? maxBloodY - minBloodY : 0,
            upperToLowerBloodRatio: lowerHalfBloodPx > 0 ? +(upperHalfBloodPx / lowerHalfBloodPx).toFixed(3) : 0,
            stemContinuity,
            bottomOblateness,
          };
        },

        async verifyExtendedHdrToggle() {
          const wasHdr = eng.isExtendedHdr;
          // 1. Capture SDR reference on identical frame (with an aerial drop softbox glint in view so >0.78 speculars exist)
          eng.setExtendedHdr(false);
          const sdrFormat = eng.format;
          const isSdrBGRA = sdrFormat.startsWith('bgra');
          const sdrBytesPerRow = Math.ceil((w * 4) / 256) * 256;
          const sdrTex = eng.device.createTexture({
            size: [w, h, 1],
            format: sdrFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
          });
          const sdrBuf = eng.device.createBuffer({
            size: sdrBytesPerRow * h,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          });
          await new Promise((resolve) => {
            requestAnimationFrame(() => {
              eng.uploadSnapshotUniforms(0.0, eng.spawnCursor, 0);
              const enc = eng.device.createCommandEncoder();
              eng.encodeOpticalRenderPasses(enc, sdrTex.createView());
              enc.copyTextureToBuffer(
                { texture: sdrTex },
                { buffer: sdrBuf, bytesPerRow: sdrBytesPerRow, rowsPerImage: h },
                [w, h, 1]
              );
              eng.device.queue.submit([enc.finish()]);
              resolve();
            });
          });
          await sdrBuf.mapAsync(GPUMapMode.READ);
          const sdrU8 = new Uint8Array(sdrBuf.getMappedRange()).slice();
          sdrBuf.unmap();
          sdrTex.destroy();
          sdrBuf.destroy();

          // 2. Capture HDR extended-sRGB on the exact same scene
          eng.setExtendedHdr(true);
          if (!eng.isExtendedHdr || eng.format !== 'rgba16float') {
            eng.setExtendedHdr(wasHdr);
            return { supported: false, maxHdrLinearPeak: 0, hdrSuperWhitePx: 0, maxMidtoneSdrDiff: 999 };
          }
          const hdrBytesPerRow = Math.ceil((w * 8) / 256) * 256;
          const hdrTex = eng.device.createTexture({
            size: [w, h, 1],
            format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
          });
          const hdrBuf = eng.device.createBuffer({
            size: hdrBytesPerRow * h,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          });
          await new Promise((resolve) => {
            requestAnimationFrame(() => {
              eng.uploadSnapshotUniforms(0.0, eng.spawnCursor, 0);
              const enc = eng.device.createCommandEncoder();
              const swapView = eng.context.getCurrentTexture().createView();
              eng.encodeOpticalRenderPasses(enc, swapView);
              eng.encodeOpticalRenderPasses(enc, hdrTex.createView());
              enc.copyTextureToBuffer(
                { texture: hdrTex },
                { buffer: hdrBuf, bytesPerRow: hdrBytesPerRow, rowsPerImage: h },
                [w, h, 1]
              );
              eng.device.queue.submit([enc.finish()]);
              resolve();
            });
          });
          await hdrBuf.mapAsync(GPUMapMode.READ);
          const f16 = new Float16Array(hdrBuf.getMappedRange());
          const stride = hdrBytesPerRow >> 1;
          let maxPeak = 0;
          let hdrSuperWhitePx = 0;
          let maxMidtoneSdrDiff = 0;
          for (let y = 0; y < h; y += 2) {
            const row = y * stride;
            const sRow = y * sdrBytesPerRow;
            for (let x = 0; x < w; x += 2) {
              const j = row + x * 4;
              const pk = Math.max(f16[j], f16[j + 1], f16[j + 2]);
              if (pk > maxPeak) maxPeak = pk;
              if (pk > 1.001) hdrSuperWhitePx++;
              const si = sRow + x * 4;
              const sr = (isSdrBGRA ? sdrU8[si + 2] : sdrU8[si + 0]) / 255;
              const sg = sdrU8[si + 1] / 255;
              const sb = (isSdrBGRA ? sdrU8[si + 0] : sdrU8[si + 2]) / 255;
              const sMax = Math.max(sr, sg, sb);
              if (sMax >= 0.05 && sMax <= 0.78) {
                const d = Math.max(Math.abs(f16[j] - sr), Math.abs(f16[j + 1] - sg), Math.abs(f16[j + 2] - sb));
                if (d > maxMidtoneSdrDiff) maxMidtoneSdrDiff = d;
              }
            }
          }
          hdrBuf.unmap();
          hdrTex.destroy();
          hdrBuf.destroy();
          eng.setExtendedHdr(wasHdr);
          await eng.device.queue.onSubmittedWorkDone();
          return {
            supported: true,
            maxHdrLinearPeak: +maxPeak.toFixed(4),
            hdrSuperWhitePx,
            maxMidtoneSdrDiff: +maxMidtoneSdrDiff.toFixed(4),
            restoredFormat: eng.format,
            validationErrors: eng.validationErrors.length,
          };
        },

        async profile60Frames(label) {
          const q = eng.device.queue;
          const gpuTimes = [];
          eng.paused = false;
          for (let i = 0; i < 4; i++) {
            realRenderFrame(1 / 60);
            await q.onSubmittedWorkDone();
          }
          for (let i = 0; i < 60; i++) {
            const t0 = performance.now();
            realRenderFrame(1 / 60);
            await q.onSubmittedWorkDone();
            gpuTimes.push(performance.now() - t0);
          }
          eng.paused = true;
          await new Promise((r) => setTimeout(r, 20));
          const sorted = [...gpuTimes].sort((a, b) => a - b);
          const avg = gpuTimes.reduce((a, b) => a + b, 0) / gpuTimes.length;
          return {
            label,
            avgGpuMs: +avg.toFixed(2),
            p95GpuMs: +sorted[Math.floor(sorted.length * 0.95)].toFixed(2),
            maxGpuMs: +sorted[sorted.length - 1].toFixed(2),
            gpuPassTimingsMs: { ...eng.gpuPassTimingsMs },
          };
        }
      };
    })()`);

    const initInfo = await cdp.eval('window.__INIT_INFO__');
    if (initInfo.validationErrors.length > 0) {
      throw new Error('WebGPU startup validation errors: ' + JSON.stringify(initInfo.validationErrors));
    }

    const captures = [];
    async function snap(filename, opts = {}) {
      const metrics = await cdp.eval(`window.__QA__.renderAndMeasure(${JSON.stringify(opts)})`);
      const shot = await cdp.send('Page.captureScreenshot', { format: 'png' });
      const outPath = writeOut(filename, Buffer.from(shot.data, 'base64'));
      captures.push({ filename, metrics });
      console.log(
        `Captured ${filename}: upper/lowerBlood=${metrics.upperToLowerBloodRatio}, eyeball=${metrics.whiteEyeballCutoutPx}, smoke=${metrics.muddyYellowTanSmokePx}`
      );
      return outPath;
    }

    if (LIVE_AUDIT) {
      const lvResult = await runLiveExplorationAudit(cdp, snap);
      const byLv = Object.fromEntries(captures.map((c) => [c.filename, c.metrics]));
      const lvFailures = [];
      if (!lvResult.xdrCheck?.supported || lvResult.xdrCheck.maxMidtoneSdrDiff > 0.015 || lvResult.xdrCheck.maxHdrLinearPeak <= 1.0) {
        lvFailures.push(`XDR extended-sRGB parity/headroom failed: ${JSON.stringify(lvResult.xdrCheck)}`);
      }
      if (lvResult.quickClickState.pipettePresence < 0.85) {
        lvFailures.push(`Quick click did not squeeze from pipette tip: ${JSON.stringify(lvResult.quickClickState)}`);
      }
      if ((byLv['lv_01_settling_ripples_t4p2s.png']?.clearWaterRedRipplePx ?? 999) > 20) {
        lvFailures.push(`Settling outer ripples showed red sonar tint: ${JSON.stringify(byLv['lv_01_settling_ripples_t4p2s.png'])}`);
      }
      if (lvResult.waveDecay.wave5p4.rmsEta >= lvResult.waveDecay.wave2p0.rmsEta * 0.45) {
        lvFailures.push(`Wave radial decay failed (sonar echo): ${JSON.stringify(lvResult.waveDecay)}`);
      }
      if ((byLv['lv_03_aging_hemolysis_t19s.png']?.yellowGreenStreakPx ?? 999) > 0) {
        lvFailures.push(`Post-11s yellow-green streaks detected: ${JSON.stringify(byLv['lv_03_aging_hemolysis_t19s.png'])}`);
      }
      if (lvResult.floorOrbitState.camPosY < -0.885) {
        lvFailures.push(`Camera orbited below cuvette floor: ${JSON.stringify(lvResult.floorOrbitState)}`);
      }
      if ((byLv['lv_05_sideon_aquarium_t2p3s.png']?.plumeVerticalSpanPx ?? 0) < 140) {
        lvFailures.push(`Side-on aquarium view flattened plume into sliver: ${JSON.stringify(byLv['lv_05_sideon_aquarium_t2p3s.png'])}`);
      }
      writeOut('live_audit_metrics.json', JSON.stringify({ initInfo, ...lvResult, lvFailures, captures }, null, 2));
      if (lvFailures.length > 0) {
        throw new Error('Live Exploration Audit Gate Failures:\n' + lvFailures.join('\n'));
      }
      console.log('LIVE EXPLORATION AUDIT PASSED:', JSON.stringify(lvResult, null, 2));
      ws.close();
      return;
    }

    if (PRESET_MATRIX) {
      const pmResult = await runPresetMatrix(cdp, snap);
      const pmFailures = [];
      if (pmResult.heroSubState.camPosY >= 0.20 || pmResult.heroSubState.camElev >= -0.14) {
        pmFailures.push(`Hero Follow camera did not submerge cleanly below meniscus: ${JSON.stringify(pmResult.heroSubState)}`);
      }
      if (!pmResult.crownHold.visibleInFrustum || Math.abs(pmResult.crownHold.pipetteTipNdcY) > 0.96) {
        pmFailures.push(`Crown preset clipped PIPETTE_TIP_Y = 0.55: ${JSON.stringify(pmResult.crownHold)}`);
      }
      writeOut('preset_matrix_metrics.json', JSON.stringify({ initInfo, ...pmResult, pmFailures, captures }, null, 2));
      if (pmFailures.length > 0) {
        throw new Error('Preset Matrix Gate Failures:\n' + pmFailures.join('\n'));
      }
      console.log('PRESET MATRIX COMPLETE:', JSON.stringify(pmResult, null, 2));
      ws.close();
      return;
    }

    // Standard Regression Pass (--quick or full)
    await cdp.eval(`(async () => {
      const eng = window.__BLOOD_ENGINE__;
      eng.setLightingMode(0);
      eng.resetSimulation(1.0);
      await eng.seekToSimTime(0.9203 + 0.33, 1.0);
      window.__QA__.setCam(0.0, 0.34, 1.04, [0.0, 0.24, 0.0]);
    })()`);
    await snap('s2_06_central_rebound_swell.png');
    const hdrExtendedCheck = await cdp.eval('window.__QA__.verifyExtendedHdrToggle()');
    console.log('XDR rgba16float extended swapchain check:', JSON.stringify(hdrExtendedCheck));

    await cdp.eval(`(async () => {
      const eng = window.__BLOOD_ENGINE__;
      eng.resetSimulation(1.0);
      await eng.seekToSimTime(3.25, 1.0);
      eng.triggerAdditionalDrop([0.0, 0.0], 0.062);
      for (let i = 0; i < 90; i++) {
        eng.paused = false;
        window.__REAL_RENDER_FRAME__(1 / 60);
        const c = eng.multiDrops.slots.find(s => s.active && s.mode === 'coalescing' && s.ageInPhase >= 0.10);
        if (c) break;
      }
      eng.paused = true;
      eng.multiDrops.spawnAerialDrop([0.0, 0.0], 0.056, 3.1, 0.44, -0.04);
      window.__QA__.setCam(0.0, -0.22, 1.24, [0.0, 0.04, 0.0]);
    })()`);
    await snap('s1_01_underwater_crater_elev22.png', {
      isShallowUnderwater: true,
      checkStemContinuity: true,
      stemMinYFrac: 0.24,
      stemMaxYFrac: 0.90,
      checkBottomOblateness: true,
      oblateMinYFrac: 0.44,
      oblateMaxYFrac: 0.88,
    });

    await cdp.eval(`(async () => {
      const eng = window.__BLOOD_ENGINE__;
      eng.setLightingMode(0);
      eng.resetSimulation(1.0);
      await eng.seekToSimTime(2.9, 1.0);
      eng.triggerAdditionalDrop([0.06, -0.03], 0.058);
      await eng.seekToSimTime(4.55, 1.0);
      window.__QA__.setCam(0.12, 0.06, 1.38, [0.02, -0.22, 0.0]);
    })()`);
    await snap('s3_01_dichromatic_brightfield.png', {
      checkStemContinuity: true,
      stemMinYFrac: 0.02,
      stemMaxYFrac: 0.88,
      checkBottomOblateness: true,
      oblateMinYFrac: 0.18,
      oblateMaxYFrac: 0.62,
    });

    const profiles = [];
    let pmFullResult = null;
    let lvFullResult = null;
    if (!QUICK) {
      await cdp.send('Page.bringToFront');
      profiles.push(await cdp.eval(`window.__QA__.profile60Frames('dichromatic_vortex_plume')`));
      pmFullResult = await runPresetMatrix(cdp, snap);
      lvFullResult = await runLiveExplorationAudit(cdp, snap);
    }

    await cdp.eval(`(async () => {
      const eng = window.__BLOOD_ENGINE__;
      eng.setLightingMode(1);
      eng.resetSimulation(1.0);
      await eng.seekToSimTime(2.9, 1.0);
      eng.triggerAdditionalDrop([0.06, -0.03], 0.058);
      await eng.seekToSimTime(4.55, 1.0);
      window.__QA__.setCam(0.12, 0.06, 1.38, [0.02, -0.22, 0.0]);
    })()`);
    await snap('s3_02_dichromatic_darkfield.png', {
      isDarkField: true,
      checkStemContinuity: true,
      stemMinYFrac: 0.02,
      stemMaxYFrac: 0.88,
      checkBottomOblateness: true,
      oblateMinYFrac: 0.18,
      oblateMaxYFrac: 0.62,
    });

    await buildContactSheet(cdp, 'regression_4up_sheet.jpg', 2, 2, [
      { file: 's1_01_underwater_crater_elev22.png', label: 's1_01 Underwater Crater' },
      { file: 's2_06_central_rebound_swell.png', label: 's2_06 Rebound Swell' },
      { file: 's3_01_dichromatic_brightfield.png', label: 's3_01 Bright-Field Dichromatism' },
      { file: 's3_02_dichromatic_darkfield.png', label: 's3_02 Dark-Field Dichromatism' },
    ]);

    const finalValidationErrors = await cdp.eval('window.__BLOOD_ENGINE__.validationErrors');
    const byFile = Object.fromEntries(captures.map((c) => [c.filename, c.metrics]));
    const gateFailures = [];
    if (finalValidationErrors.length > 0) {
      gateFailures.push(`WebGPU validationErrors: ${JSON.stringify(finalValidationErrors)}`);
    }
    if (cdp.consoleErrors.length > 0 || cdp.pageErrors.length > 0) {
      gateFailures.push(`Console/page errors: ${JSON.stringify({ console: cdp.consoleErrors, page: cdp.pageErrors })}`);
    }
    for (const f of ['s1_01_underwater_crater_elev22.png', 's3_01_dichromatic_brightfield.png', 's3_02_dichromatic_darkfield.png']) {
      if (!byFile[f]?.stemContinuity?.continuous) {
        gateFailures.push(`${f} stemContinuity failed: ${JSON.stringify(byFile[f]?.stemContinuity)}`);
      }
      if (!byFile[f]?.bottomOblateness?.oblatePass) {
        gateFailures.push(`${f} bottomOblateness failed: ${JSON.stringify(byFile[f]?.bottomOblateness)}`);
      }
    }
    if (!hdrExtendedCheck?.supported || hdrExtendedCheck.maxHdrLinearPeak <= 1.0 || hdrExtendedCheck.maxMidtoneSdrDiff > 0.015) {
      gateFailures.push(`XDR rgba16float extended check failed: ${JSON.stringify(hdrExtendedCheck)}`);
    }
    if (lvFullResult) {
      if ((byFile['lv_01_settling_ripples_t4p2s.png']?.clearWaterRedRipplePx ?? 999) > 20) {
        gateFailures.push(`Settling outer ripples showed red sonar tint: ${JSON.stringify(byFile['lv_01_settling_ripples_t4p2s.png'])}`);
      }
      if (lvFullResult.waveDecay.wave5p4.rmsEta >= lvFullResult.waveDecay.wave2p0.rmsEta * 0.45) {
        gateFailures.push(`Wave radial decay failed (sonar echo): ${JSON.stringify(lvFullResult.waveDecay)}`);
      }
      if ((byFile['lv_03_aging_hemolysis_t19s.png']?.yellowGreenStreakPx ?? 999) > 0) {
        gateFailures.push(`Post-11s yellow-green streaks detected: ${JSON.stringify(byFile['lv_03_aging_hemolysis_t19s.png'])}`);
      }
      if ((byFile['lv_05_sideon_aquarium_t2p3s.png']?.plumeVerticalSpanPx ?? 0) < 140) {
        gateFailures.push(`Side-on aquarium view flattened plume into sliver: ${JSON.stringify(byFile['lv_05_sideon_aquarium_t2p3s.png'])}`);
      }
    }
    if (pmFullResult) {
      if (pmFullResult.heroSubState.camPosY >= 0.20 || pmFullResult.heroSubState.camElev >= -0.14) {
        gateFailures.push(`Hero Follow camera did not submerge cleanly below meniscus: ${JSON.stringify(pmFullResult.heroSubState)}`);
      }
      if (!pmFullResult.crownHold.visibleInFrustum || Math.abs(pmFullResult.crownHold.pipetteTipNdcY) > 0.96) {
        gateFailures.push(`Crown preset clipped PIPETTE_TIP_Y = 0.55: ${JSON.stringify(pmFullResult.crownHold)}`);
      }
    }
    for (const prof of profiles) {
      if (prof.avgGpuMs > 12.5 || prof.p95GpuMs > 15.5) {
        gateFailures.push(`GPU 60Hz budget exceeded on ${prof.label}: ${JSON.stringify(prof)}`);
      }
    }

    writeOut('verification_metrics.json', JSON.stringify({ initInfo, hdrExtendedCheck, profiles, gateFailures, captures }, null, 2));
    if (gateFailures.length > 0) {
      throw new Error('Automated Verification Gate Failures:\n' + gateFailures.join('\n'));
    }
    console.log('VERIFICATION PASSED', JSON.stringify({ quick: QUICK, hdrExtendedCheck, profiles }, null, 2));
    ws.close();
  } finally {
    if (chrome.exitCode === null) {
      const exited = new Promise((r) => chrome.once('exit', r));
      chrome.kill('SIGKILL');
      await Promise.race([exited, sleep(800)]);
    }
    if (devServer && devServer.exitCode === null) {
      devServer.kill('SIGTERM');
    }
    await sleep(120);
    fs.rmSync(USER_DIR, { recursive: true, force: true, maxRetries: 8, retryDelay: 120 });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
