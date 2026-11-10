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

  // Pointer 3D orbit, 1:1 3D pan, macro zoom, subject focus, click-to-drop & press-and-hold pendant swelling
  const DRAG_ORBIT_THRESHOLD_PX = 5.0;
  const PENDANT_HOLD_DELAY_MS = 160.0;

  const activePointers = new Map<number, { x: number; y: number }>();
  let prevPinchDist = 0;
  let prevPinchMidX = 0;
  let prevPinchMidY = 0;

  const gesture = {
    pointerId: null as number | null,
    button: 0,
    shiftDown: false,
    altDown: false,
    downX: 0,
    downY: 0,
    downTimeMs: 0,
    ndcX: 0,
    ndcY: 0,
    isDragging: false,
    dragMode: 'orbit' as 'orbit' | 'pan',
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

  const getPinchMetrics = (): { dist: number; midX: number; midY: number } => {
    const pts = Array.from(activePointers.values());
    if (pts.length < 2) return { dist: 0, midX: 0, midY: 0 };
    return {
      dist: Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y),
      midX: 0.5 * (pts[0].x + pts[1].x),
      midY: 0.5 * (pts[0].y + pts[1].y),
    };
  };

  canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
  });

  canvas.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse' && e.button !== 0 && e.button !== 1 && e.button !== 2) return;
    if (e.button === 1) e.preventDefault();
    activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {
      // Ignore synthetic events without active pointerId registration
    }

    if (activePointers.size === 1) {
      const [ndcX, ndcY] = computeCanvasNdc(e.clientX, e.clientY);
      const wantPan = e.button === 1 || e.button === 2 || e.shiftKey;
      gesture.pointerId = e.pointerId;
      gesture.button = e.button;
      gesture.shiftDown = e.shiftKey;
      gesture.altDown = e.altKey;
      gesture.downX = e.clientX;
      gesture.downY = e.clientY;
      gesture.downTimeMs = performance.now();
      gesture.ndcX = ndcX;
      gesture.ndcY = ndcY;
      gesture.isDragging = false;
      gesture.dragMode = wantPan ? 'pan' : 'orbit';
      gesture.pendantStarted = false;
    } else if (activePointers.size >= 2) {
      if (gesture.pendantStarted) {
        engine.cancelPendantDrop();
        gesture.pendantStarted = false;
      }
      gesture.isDragging = true;
      const pm = getPinchMetrics();
      prevPinchDist = pm.dist;
      prevPinchMidX = pm.midX;
      prevPinchMidY = pm.midY;
    }
  });

  window.addEventListener('pointermove', (e) => {
    const prev = activePointers.get(e.pointerId);
    if (!prev) return;
    const dx = e.clientX - prev.x;
    const dy = e.clientY - prev.y;
    prev.x = e.clientX;
    prev.y = e.clientY;

    const viewH = Math.max(1, canvas.clientHeight || 800);

    if (activePointers.size === 1 && e.pointerId === gesture.pointerId) {
      if (!gesture.isDragging) {
        const dragDist = Math.hypot(e.clientX - gesture.downX, e.clientY - gesture.downY);
        const thresh = gesture.dragMode === 'pan' ? 3.5 : DRAG_ORBIT_THRESHOLD_PX;
        if (dragDist < thresh) {
          return;
        }
        if (gesture.pendantStarted) {
          engine.cancelPendantDrop();
          gesture.pendantStarted = false;
        }
        gesture.isDragging = true;
      }
      if (gesture.dragMode === 'pan' || e.shiftKey) {
        engine.clearSubjectLock();
        engine.camera.panScreen(dx / viewH, dy / viewH);
        syncFocusButton();
      } else {
        engine.camera.orbit(-dx * 0.0048, dy * 0.0042);
      }
      for (const b of camButtons) b.classList.remove('is-active');
    } else if (activePointers.size === 2) {
      const pm = getPinchMetrics();
      if (prevPinchDist > 8 && pm.dist > 8) {
        engine.camera.dolly(prevPinchDist / pm.dist);
        const panDx = pm.midX - prevPinchMidX;
        const panDy = pm.midY - prevPinchMidY;
        if (Math.hypot(panDx, panDy) > 0.5) {
          engine.clearSubjectLock();
          engine.camera.panScreen(panDx / viewH, panDy / viewH);
          syncFocusButton();
        }
        for (const b of camButtons) b.classList.remove('is-active');
      }
      prevPinchDist = pm.dist;
      prevPinchMidX = pm.midX;
      prevPinchMidY = pm.midY;
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
    } else if (!gesture.isDragging) {
      if (gesture.button === 2 || (gesture.button === 0 && gesture.altDown)) {
        engine.focusNearestSubjectAtNdc(gesture.ndcX, gesture.ndcY);
        for (const b of camButtons) b.classList.remove('is-active');
        syncFocusButton();
      } else if (gesture.button === 0 && !gesture.shiftDown) {
        if (gesture.pendantStarted) {
          if (engine.hasActivePendantDrop()) {
            engine.releasePendantDrop();
            syncPlayButton();
          }
        } else {
          engine.triggerQuickPipetteDropAtNdc(gesture.ndcX, gesture.ndcY);
          syncPlayButton();
        }
      }
    }

    gesture.pointerId = null;
    gesture.isDragging = false;
    gesture.pendantStarted = false;
  };
  window.addEventListener('pointerup', endPointer);
  window.addEventListener('pointercancel', endPointer);

  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      if (e.shiftKey) {
        const viewH = Math.max(1, canvas.clientHeight || 800);
        engine.clearSubjectLock();
        engine.camera.panScreen(-e.deltaX / viewH, -e.deltaY / viewH);
        syncFocusButton();
      } else {
        const zoomFactor = Math.exp(e.deltaY * 0.0011);
        engine.camera.dolly(zoomFactor);
      }
      for (const b of camButtons) b.classList.remove('is-active');
    },
    { passive: false }
  );

  // HUD bindings
  const hud = document.getElementById('hud');
  const hudTime = document.getElementById('hud-time');
  const hudPhase = document.getElementById('hud-phase');
  const hudFpsMode = document.getElementById('hud-fps-mode');
  const btnPlay = document.getElementById('btn-play');
  const btnStep = document.getElementById('btn-step');
  const btnReplay = document.getElementById('btn-replay');
  const btnDrop = document.getElementById('btn-drop');
  const btnFocus = document.getElementById('btn-focus');
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
    hud?.classList.toggle('is-paused', engine.paused);
  };
  syncPlayButton();

  const syncFocusButton = () => {
    btnFocus?.classList.toggle('is-active', engine.hasSubjectLock());
  };
  syncFocusButton();

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

  btnStep?.addEventListener('click', (e) => {
    engine.stepPausedSubsteps(e.shiftKey ? 5 : 1);
    syncPlayButton();
  });

  btnReplay?.addEventListener('click', () => {
    engine.paused = false;
    engine.resetSimulation();
    syncPlayButton();
    syncFocusButton();
  });

  btnDrop?.addEventListener('click', () => {
    const jitterX = (Math.random() - 0.5) * 0.14;
    const jitterZ = (Math.random() - 0.5) * 0.14;
    engine.triggerQuickPipetteDrop([jitterX, jitterZ]);
    syncPlayButton();
  });

  btnFocus?.addEventListener('click', () => {
    engine.cycleFocusSubject();
    for (const b of camButtons) b.classList.remove('is-active');
    syncFocusButton();
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
      syncFocusButton();
    });
  }

  window.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
    if (e.code === 'Space' || e.key === ' ') {
      e.preventDefault();
      engine.paused = !engine.paused;
      syncPlayButton();
    } else if (
      e.code === 'Period' ||
      e.code === 'BracketRight' ||
      e.key === '.' ||
      e.key === '>' ||
      e.key === ']'
    ) {
      e.preventDefault();
      engine.stepPausedSubsteps(e.shiftKey ? 5 : 1);
      syncPlayButton();
    } else if (e.key === 'f' || e.key === 'F') {
      e.preventDefault();
      engine.cycleFocusSubject();
      for (const b of camButtons) b.classList.remove('is-active');
      syncFocusButton();
    } else if (e.code === 'ArrowUp' || e.code === 'ArrowDown' || e.code === 'ArrowLeft' || e.code === 'ArrowRight') {
      e.preventDefault();
      const step = e.shiftKey ? 0.11 : 0.045;
      const dx = e.code === 'ArrowLeft' ? step : e.code === 'ArrowRight' ? -step : 0.0;
      const dy = e.code === 'ArrowUp' ? -step : e.code === 'ArrowDown' ? step : 0.0;
      engine.clearSubjectLock();
      engine.camera.panScreen(dx, dy);
      for (const b of camButtons) b.classList.remove('is-active');
      syncFocusButton();
    } else if (e.key === '=' || e.key === '+') {
      e.preventDefault();
      engine.camera.dolly(0.84);
      for (const b of camButtons) b.classList.remove('is-active');
    } else if (e.key === '-' || e.key === '_') {
      e.preventDefault();
      engine.camera.dolly(1.19);
      for (const b of camButtons) b.classList.remove('is-active');
    } else if (e.key === 'r' || e.key === 'R') {
      engine.paused = false;
      engine.resetSimulation();
      syncPlayButton();
      syncFocusButton();
    } else if (e.key === 'd' || e.key === 'D') {
      const jitterX = (Math.random() - 0.5) * 0.14;
      const jitterZ = (Math.random() - 0.5) * 0.14;
      engine.triggerQuickPipetteDrop([jitterX, jitterZ]);
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
      !gesture.isDragging &&
      gesture.button === 0 &&
      !gesture.shiftDown &&
      !gesture.altDown
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
    } else if (engine.hasAutoSqueezePendantDrop()) {
      const pinched = engine.updatePendantDrop(dtSec);
      if (pinched) {
        syncPlayButton();
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
