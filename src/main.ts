import { BloodDropEngine } from './engine/BloodDropEngine';
import type { CameraPreset } from './math/camera';

declare global {
  interface Window {
    __BLOOD_ENGINE__?: BloodDropEngine;
  }
}

async function bootstrap(): Promise<void> {
  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement | null;
  const fallbackBanner = document.getElementById('fallback-banner');
  const fallbackMessage = document.getElementById('fallback-message');

  if (!canvas) return;

  const params = new URLSearchParams(window.location.search);
  if (params.get('zen') === '1') {
    document.body.classList.add('zen-mode');
  }

  const syncCanvasResolution = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  };
  syncCanvasResolution();

  let engine: BloodDropEngine;
  try {
    engine = await BloodDropEngine.create(canvas);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('[BloodDropEngine Init Failed]', msg);
    if (fallbackMessage) fallbackMessage.textContent = msg;
    if (fallbackBanner) fallbackBanner.hidden = false;
    return;
  }

  window.__BLOOD_ENGINE__ = engine;

  // Apply optional URL query parameters for deterministic presets/inspection
  const presetParam = params.get('preset');
  if (presetParam === 'hero' || presetParam === 'crown' || presetParam === 'abyss') {
    engine.setCameraPreset(presetParam, true);
  }
  if (params.get('lighting') === 'dark') {
    engine.setLightingMode(true);
  }
  if (params.get('paused') === '1') {
    engine.paused = true;
  }

  const resizeObserver = new ResizeObserver(() => {
    syncCanvasResolution();
  });
  resizeObserver.observe(canvas);

  // Pointer 3D orbit, pinch-zoom, stationary click-to-drop & press-and-hold pendant swelling
  const DRAG_ORBIT_THRESHOLD_PX = 5.0;
  const PENDANT_HOLD_DELAY_MS = 160.0;

  const activePointers = new Map<number, { x: number; y: number }>();
  let prevPinchDist = 0;

  const gesture = {
    pointerId: null as number | null,
    downX: 0,
    downY: 0,
    downTimeMs: 0,
    ndcX: 0,
    ndcY: 0,
    isOrbiting: false,
    pendantStarted: false,
  };

  const computeCanvasNdc = (clientX: number, clientY: number): [number, number] => {
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, rect.width);
    const h = Math.max(1, rect.height);
    const ndcX = ((clientX - rect.left) / w) * 2.0 - 1.0;
    const ndcY = 1.0 - ((clientY - rect.top) / h) * 2.0;
    return [ndcX, ndcY];
  };

  const getPinchDistance = (): number => {
    const pts = Array.from(activePointers.values());
    if (pts.length < 2) return 0;
    return Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
  };

  canvas.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {
      // Ignore synthetic events without active pointerId registration
    }

    if (activePointers.size === 1) {
      const [ndcX, ndcY] = computeCanvasNdc(e.clientX, e.clientY);
      gesture.pointerId = e.pointerId;
      gesture.downX = e.clientX;
      gesture.downY = e.clientY;
      gesture.downTimeMs = performance.now();
      gesture.ndcX = ndcX;
      gesture.ndcY = ndcY;
      gesture.isOrbiting = false;
      gesture.pendantStarted = false;
    } else if (activePointers.size >= 2) {
      if (gesture.pendantStarted) {
        engine.cancelPendantDrop();
        gesture.pendantStarted = false;
      }
      gesture.isOrbiting = true;
      prevPinchDist = getPinchDistance();
    }
  });

  window.addEventListener('pointermove', (e) => {
    const prev = activePointers.get(e.pointerId);
    if (!prev) return;
    const dx = e.clientX - prev.x;
    const dy = e.clientY - prev.y;
    prev.x = e.clientX;
    prev.y = e.clientY;

    if (activePointers.size === 1 && e.pointerId === gesture.pointerId) {
      if (!gesture.isOrbiting) {
        const dragDist = Math.hypot(e.clientX - gesture.downX, e.clientY - gesture.downY);
        if (dragDist < DRAG_ORBIT_THRESHOLD_PX) {
          return;
        }
        if (gesture.pendantStarted) {
          engine.cancelPendantDrop();
          gesture.pendantStarted = false;
        }
        gesture.isOrbiting = true;
      }
      engine.camera.orbit(-dx * 0.0048, dy * 0.0042);
      for (const b of camButtons) b.classList.remove('is-active');
    } else if (activePointers.size === 2) {
      const currDist = getPinchDistance();
      if (prevPinchDist > 8 && currDist > 8) {
        engine.camera.dolly(prevPinchDist / currDist);
        for (const b of camButtons) b.classList.remove('is-active');
      }
      prevPinchDist = currDist;
    }
  });

  const endPointer = (e: PointerEvent) => {
    const wasTracked = activePointers.delete(e.pointerId);
    if (activePointers.size < 2) {
      prevPinchDist = 0;
    }
    if (!wasTracked || e.pointerId !== gesture.pointerId) {
      return;
    }

    if (e.type === 'pointercancel') {
      if (gesture.pendantStarted) {
        engine.cancelPendantDrop();
      }
    } else if (!gesture.isOrbiting) {
      if (gesture.pendantStarted) {
        if (engine.hasActivePendantDrop()) {
          engine.releasePendantDrop();
          syncPlayButton();
        }
      } else {
        engine.spawnDropAtNdc(gesture.ndcX, gesture.ndcY);
        syncPlayButton();
      }
    }

    gesture.pointerId = null;
    gesture.isOrbiting = false;
    gesture.pendantStarted = false;
  };
  window.addEventListener('pointerup', endPointer);
  window.addEventListener('pointercancel', endPointer);

  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      const zoomFactor = Math.exp(e.deltaY * 0.0011);
      engine.camera.dolly(zoomFactor);
      for (const b of camButtons) b.classList.remove('is-active');
    },
    { passive: false }
  );

  // HUD bindings
  const hudTime = document.getElementById('hud-time');
  const hudPhase = document.getElementById('hud-phase');
  const hudFpsMode = document.getElementById('hud-fps-mode');
  const btnPlay = document.getElementById('btn-play');
  const btnReplay = document.getElementById('btn-replay');
  const btnDrop = document.getElementById('btn-drop');
  const btnLighting = document.getElementById('btn-lighting');
  const btnZen = document.getElementById('btn-zen');
  const speedButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('.speed-btn'));
  const camButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('.cam-btn'));
  if (presetParam === 'hero' || presetParam === 'crown' || presetParam === 'abyss') {
    for (const b of camButtons) b.classList.toggle('is-active', b.dataset.cam === presetParam);
  }

  const syncPlayButton = () => {
    if (btnPlay) {
      btnPlay.textContent = engine.paused ? 'Play' : 'Pause';
      btnPlay.classList.toggle('is-active', engine.paused);
    }
  };
  syncPlayButton();

  const syncLightingButton = () => {
    const isDark = engine.config.lightingMode > 0.5;
    if (btnLighting) {
      btnLighting.textContent = isDark ? 'Dark Field' : 'Backlit';
      btnLighting.classList.toggle('is-active', isDark);
    }
  };
  syncLightingButton();

  btnPlay?.addEventListener('click', () => {
    engine.paused = !engine.paused;
    syncPlayButton();
  });

  btnReplay?.addEventListener('click', () => {
    engine.paused = false;
    engine.resetSimulation();
    syncPlayButton();
  });

  btnDrop?.addEventListener('click', () => {
    const jitterX = (Math.random() - 0.5) * 0.14;
    const jitterZ = (Math.random() - 0.5) * 0.14;
    engine.triggerAdditionalDrop([jitterX, jitterZ]);
    syncPlayButton();
  });

  btnLighting?.addEventListener('click', () => {
    engine.setLightingMode(engine.config.lightingMode < 0.5);
    syncLightingButton();
  });

  btnZen?.addEventListener('click', () => {
    document.body.classList.toggle('zen-mode');
  });

  for (const btn of speedButtons) {
    btn.addEventListener('click', () => {
      const spd = Number(btn.dataset.speed || '1');
      engine.timeScale = spd;
      engine.paused = false;
      syncPlayButton();
      for (const b of speedButtons) b.classList.toggle('is-active', b === btn);
      if (hudFpsMode && btn.dataset.label) {
        hudFpsMode.textContent = btn.dataset.label;
      }
    });
  }

  for (const btn of camButtons) {
    btn.addEventListener('click', () => {
      const id = btn.dataset.cam as CameraPreset['id'] | undefined;
      if (!id) return;
      engine.setCameraPreset(id, false);
      for (const b of camButtons) b.classList.toggle('is-active', b === btn);
    });
  }

  window.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
    if (e.code === 'Space') {
      e.preventDefault();
      engine.paused = !engine.paused;
      syncPlayButton();
    } else if (e.key === 'r' || e.key === 'R') {
      engine.paused = false;
      engine.resetSimulation();
      syncPlayButton();
    } else if (e.key === 'd' || e.key === 'D') {
      const jitterX = (Math.random() - 0.5) * 0.14;
      const jitterZ = (Math.random() - 0.5) * 0.14;
      engine.triggerAdditionalDrop([jitterX, jitterZ]);
      syncPlayButton();
    } else if (e.key === 'l' || e.key === 'L') {
      engine.setLightingMode(engine.config.lightingMode < 0.5);
      syncLightingButton();
    } else if (e.key === 'h' || e.key === 'H' || e.key === 'Escape') {
      document.body.classList.toggle('zen-mode');
    }
  });

  const fallbackTitle = document.getElementById('fallback-title');
  const btnReload = document.getElementById('btn-reload') as HTMLButtonElement | null;
  btnReload?.addEventListener('click', () => {
    window.location.reload();
  });

  let rafId = 0;
  engine.onDeviceLost = (info) => {
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = 0;
    }
    const reason = info.message || info.reason || 'driver reset';
    if (fallbackTitle) fallbackTitle.textContent = 'GPU Context Lost';
    if (fallbackMessage) {
      fallbackMessage.textContent = `The WebGPU graphics device was lost (${reason}). Reload to re-initialize the fluid & optical simulation.`;
    }
    if (btnReload) btnReload.hidden = false;
    if (fallbackBanner) fallbackBanner.hidden = false;
  };

  window.addEventListener('pagehide', (ev) => {
    if (!ev.persisted) {
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
      engine.destroy();
    }
  });

  // Main high-refresh animation loop
  let lastNow = performance.now();
  const frame = (now: number) => {
    if (engine.isDeviceLost) return;
    const dtSec = (now - lastNow) * 0.001;
    lastNow = now;
    syncCanvasResolution();

    if (
      gesture.pointerId !== null &&
      activePointers.size === 1 &&
      !gesture.isOrbiting
    ) {
      const heldMs = now - gesture.downTimeMs;
      if (!gesture.pendantStarted && heldMs >= PENDANT_HOLD_DELAY_MS) {
        engine.beginPendantDropAtNdc(gesture.ndcX, gesture.ndcY);
        gesture.pendantStarted = true;
      } else if (gesture.pendantStarted && engine.hasActivePendantDrop()) {
        const pinched = engine.updatePendantDrop(dtSec);
        if (pinched) {
          syncPlayButton();
        }
      }
    }

    const kin = engine.renderFrame(dtSec);
    if (hudTime) {
      hudTime.textContent = `T+${kin.simTime.toFixed(2).padStart(5, '0')}s`;
    }
    if (hudPhase && hudPhase.textContent !== kin.phaseLabel) {
      hudPhase.textContent = kin.phaseLabel;
    }

    if (!engine.isDeviceLost) {
      rafId = requestAnimationFrame(frame);
    }
  };
  rafId = requestAnimationFrame(frame);
}

void bootstrap();
