# `drop-of-blood` — Workspace Agent Runbook & Invariants

## 0. Fast Triage & Anti-Relearning Runbook

1. **Mandatory Turn-0 Skill Read**: Read `/Users/andrewvoirol/.gemini/config/skills/webgpu-interactive-simulation/SKILL.md` before designing, editing, or delegating any shader, fluid, camera, or verification work.
2. **Core Commands**:
   - **Unit Tests (`44/44`)**: `npx vitest run`
   - **Typecheck + Production Bundle**: `npm run build`
   - **Real User Preset + DOM Hold + Unpaused Temporal Strip (`~12 s`)**: `node scripts/verify.mjs --preset-matrix` (outputs stitched `.verify-out/preset_matrix_sheet.jpg` and `.verify-out/temporal_entry_strip.jpg`)
   - **Fast 4-Shot Regression + `1:1` Crops (`~10 s`)**: `node scripts/verify.mjs --quick`
   - **Full Regression + `60`-Frame Metal GPU Latency Gate**: `node scripts/verify.mjs`
3. **Iteration & Git Discipline**:
   - Enforce `<= 2` edit-and-capture passes before surfacing side-by-side results to the user. Never chain `> 2` `/boost` `DeepCoder` rounds.
   - Keep `screenshots/*.png` and `screenshots/demo.gif` untouched (`git diff --stat screenshots/` must stay empty) until explicit user sign-off for README/portfolio media regeneration.

---

## 1. Coordinate, Grid, Camera-Preset Frustum & Clock Contracts

### A. World Coordinates & Simulation Grids
- **Quiescent Meniscus (`WATER_SURFACE_Y`)**: `y = +0.30` ([`src/sim/dropKinematics.ts`](src/sim/dropKinematics.ts#L5), [`src/sim/uniformLayout.ts`](src/sim/uniformLayout.ts#L11)).
- **Pasteur Pipette Orifice (`PIPETTE_TIP_Y`)**: `y = +0.55` ([`src/sim/dropKinematics.ts`](src/sim/dropKinematics.ts#L306-L364)).
- **Tank Floor**: `y = -0.98` (3D MAC fluid box spans `x, z in [-0.48, +0.48]`, `y in [-0.98, +0.30]`, resolution `192 × 256 × 192`, voxel size `~5.0 mm`).
- **2D Free-Surface Wave Grid**: `WAVE_GRID_SIZE = 256` ([`src/shaders/common.ts`](src/shaders/common.ts#L1-L2)). Currently `WAVE_HALF_EXTENT = 1.60` (`3.20 m` span, `dx = 12.5 mm`, Nyquist `= 25.0 mm` — too coarse for `2–8 mm` capillary wavelets; see Backlog #2).
- **Dual Clocks**:
  - Physics substep: `FIXED_SIM_DT = 1 / 60` ([`src/engine/BloodDropEngine.ts`](src/engine/BloodDropEngine.ts#L82)) representing `1000 fps` slow-motion simulation seconds.
  - User pointer hold (`PENDANT_HOLD_DELAY_MS = 160 ms` in [`src/main.ts`](src/main.ts#L65)): measured in real wall-clock seconds (`0..2.2 s` -> $V \propto t$).
  - Continuous vessel: `autoLoop = false` by default (`393,216`-particle ring buffer wraps cleanly across `> 8` drops; floor *rouleaux* sedimentation accumulates in `concTex.a`).

### B. User HUD `CAMERA_PRESETS` (`src/math/camera.ts` lines `14–42` & `102–120`)
Every visual change MUST be verified through these actual user presets, never solely through custom `window.__QA__.setCam(...)` overrides:
1. **`hero` (`Follow Shot`, default on boot)**:
   - `t in [0, 1.18 s]`: `elev = +0.29`, `dist = 1.28`, `target = [0, 0.22, 0]`, `fovY = 40°` (above meniscus).
   - `t in [1.18, 2.85 s]`: `syncHeroFollow` currently dives underwater to `elev = -0.125 -> -0.155`, `target.y = -0.015 -> -0.26` — **causes Backlog #1b (hides `y = 0.55` pipette on subsequent holds) and Backlog #3 (50/50 planar TIR Rorschach plume clone)**.
2. **`crown` (`Surface Crown`)**:
   - `az = 0.32`, `elev = +0.46`, `dist = 1.44`, `target = [0, 0.06, 0]`, `fovY = 39°` — top of frame at `(0, 0)` cuts off at **`y ≈ +0.544`**, currently clipping `PIPETTE_TIP_Y = +0.55` out of frame!
3. **`abyss` (`Sub-Meniscus`)**:
   - `az = -0.16`, `elev = -0.21`, `dist = 1.32`, `target = [0, 0.06, 0]`, `fovY = 43°` (intentional submerged upward look).

---

## 2. Coupled Shader & Texture Contracts

- **`concTex` (`192 × 256 × 192` `rgba16float`)**:
  - `.r`: Intact erythrocyte concentration (turbid Mie scatter + Duysens discrete-cell sieve).
  - `.g`: Lysed free-hemoglobin plasma concentration.
  - `.b`: Oxygenation / shear tracer (`vOxShear` driving `380–720 nm` $\text{HbO}_2$/$\text{dHb}$ dichromatic LUT).
  - `.a`: Settled floor *rouleaux* sediment bed (`floorRouleauxBed`, symmetric mass transfer $\Delta m = \min(\rho, J\,\Delta t)$).
- **Two-Stage Optical Pipeline ([`src/shaders/render.ts`](src/shaders/render.ts))**:
  - Half-res volumetric pass (`1088 × 680` `volTex`: `.rgb` linear radiance, `.a` ray hit distance `tSurf`).
  - `1:1` Retina `COMPOSITE_WGSL` (`2560 × 1600`): perspective-detrended `volTex.a - tFlat` bilateral upsampling + native `1:1` foreground aerial/pendant drop intersection + meniscus speculars.
- **Hard Regression Invariants Across All 4 Optical Regimes** (`Bright-Field Topside`, `Dark-Field Topside`, `Sub-Meniscus TIR`, `Steep Snell's Window`):
  - Unbroken vertical stem continuity (`emptyGapScanlines === 0`).
  - Oblate toroidal vortex ring bottom (`centerSpearExcessPx <= 14`, zero pointed axial V-tip spear).
  - Zero `[225, 232, 238]` underwater white eyeball cutouts, zero yellow-tan Soret smoke, zero Dark-Field magenta water ripples.
  - `avgGpuMs <= 12.5 ms` and `p95GpuMs <= 15.5 ms` at `2560 × 1600` (`2x` Retina).
  - Staggered MAC radial forcing evaluated at true face coordinates (`uvwX`, `uvwY`, `uvwZ` in [`src/shaders/fluid.ts`](src/shaders/fluid.ts)).

---

## 3. Priority Next-Session Live-Realism Backlog (Verified Root Causes)

The user live-tested Round 8 and identified **three primary realism breaks** to resolve in the next session:

### Issue 1 — "The drop is very fake looking before it enters"
- **Exact Code Locations**:
  - [`src/shaders/render.ts`](src/shaders/render.ts#L637-L721) (`shadeAerialDrop`)
  - [`src/sim/dropKinematics.ts`](src/sim/dropKinematics.ts#L300-L565) (pendant growth, pinch-off, free-fall kinematics & shape oscillation)
  - [`src/math/camera.ts`](src/math/camera.ts#L15-L42) (`crown` frustum top `y ≈ 0.544 < PIPETTE_TIP_Y = 0.55` and submerged `hero` hiding aerial drops)
- **Root Causes**:
  1. **Clear Red Glass-Marble Refraction (`render.ts:682-710`)**: `bgThrough = mix(backdropBehind, refrBg, 0.88 * airExitGate)` refracts `88%` of the bright studio backdrop upside-down through the drop core. Real `3–4 mm` whole blood (`Hct ~ 45%`, erythrocyte Mie reduced scattering $\mu_s' \approx 2.2\text{ mm}^{-1}$, $\tau' > 8$) is optically turbid/opaque dark oxblood in its core, with ruby translucency confined to `< 0.5 mm` grazing rims and stretched capillary necks.
  2. **Circular Phong Sticker Highlight (`render.ts:676-679`)**: `smoothstep(0.925, 0.988, keyAlign) * 1.42 + pow(..., 150.0) * 0.92` stamps a hard round white cartoon dot onto a rigid ellipsoid instead of a curved studio strip/window area-light reflection weighted by physical dielectric Fresnel ($F_0 \approx 0.025$).
  3. **Rigid Spheroid Float, Gumdrop Slump & Preset Framing**: During the slow-motion fall (`0..0.92 s`), the drop reads as a stiff analytical sphere with a sticker dot, melts into a smooth gumdrop dome at `t = 1.02 s`, and leaves a flat dark-red coin disc on the surface at `t = 1.15..1.47 s` (captured in `temporal_entry_strip.jpg`); meanwhile `Crown` clips `PIPETTE_TIP_Y = 0.55` above the top bezel (`tipNdcY = +1.0142` in `preset_matrix_sheet.jpg`) and post-`1.18 s` `Follow` hides aerial drops completely.

### Issue 2 — "The ripples are very fake looking"
- **Exact Code Locations**:
  - [`src/shaders/common.ts`](src/shaders/common.ts#L1-L2) (`WAVE_GRID_SIZE = 256`, `WAVE_HALF_EXTENT = 1.60`)
  - [`src/shaders/wave.ts`](src/shaders/wave.ts#L108-L127) (`WAVE_SPATIAL_FORCE_WGSL` dual Ricker-wavelet pistons)
  - [`src/shaders/render.ts`](src/shaders/render.ts#L2364-L2392) (`radialOut = normalize(pRipple.xz)` + `mercuryBand` + `slopeContrast`) and [`src/shaders/render.ts`](src/shaders/render.ts#L3003-L3021) (`capillarySheen` / `softboxDome`)
- **Root Causes**:
  1. **Coarse `12.5 mm` Grid Nyquist Limit**: Spanning `3.20 m` with `256` cells (`dx = 12.5 mm`, $\lambda_{\min} = 25\text{ mm}$) makes it mathematically impossible to resolve `2–8 mm` capillary precursor wavelets ($\omega^2 = gk + (\sigma/\rho)k^3$), turning every impact into wide, rubbery `10–18 cm` bullseye swells.
  2. **Synthetic Double-Piston Forcing (`wave.ts:120-126`)**: Injects a second hardcoded `reboundImpulse = 4.10 * swellShape` Ricker wavelet after impact on top of the FFT PDE instead of letting cavity collapse emerge from a single momentum-coupled crater impulse on a properly scaled grid.
  3. **Painted Origin-Locked Airbrush Rings (`render.ts:2367-2391`)**: Computes `radialOut = normalize(pRipple.xz)` locked to `(0, 0)` (breaking off-center drops!) and adds synthetic grey/metallic `slopeContrast` + `mercuryBand` paint instead of pure Fresnel environment reflection and Snell refraction.

### Issue 3 — "The reflection or mirror of the drop after it enters that duplicates the above and below"
- **Exact Code Locations**:
  - [`src/math/camera.ts`](src/math/camera.ts#L102-L120) (`syncHeroFollow` diving to `elev = -0.125`, `target.y = -0.015`)
  - [`src/shaders/render.ts`](src/shaders/render.ts#L2325-L2401) (`evalSubmergedBranch` `gradPupil` / `nPupilUnder` / `tirPlume`)
- **Root Causes**:
  1. **Default `Follow` Camera Parks in a 50/50 Grazing TIR Split**: After `t = 1.18 s`, `syncHeroFollow` automatically submerges the default camera just below `y = +0.30` at shallow upward grazing angle (`elev = -0.125 rad`, $\theta_i > 48.6^\circ \implies F = 1.0$ Total Internal Reflection), bisecting the screen with the water ceiling.
  2. **Over-Smoothed Planar Ceiling Mirror (`render.ts:2327-2339`)**: `gradPupil` box-filters wave height over `4.4–9.6 cm`, damps slopes by `0.56`, and blends `72%` back toward flat `tirDir`, turning the disturbed meniscus into a flat vanity mirror that raymarches `tirPlume` a second time and renders a giant **1:1 upside-down Rorschach clone** of the blood plume above the real plume (visible in `phase2_s1_01_underwater_elev22.png`).
  3. **Resolution Path**: Keep `Follow` viewing cleanly through the lateral cuvette without parking in a shallow 50/50 ceiling mirror split (while keeping aerial/pendant drops visible on click-and-hold!), and in `Sub-Meniscus` (`abyss`) preserve genuine multi-scale wave slope distortion + physical path attenuation on `tirPlume` so underside reflections ripple naturally instead of cloning 1:1.
