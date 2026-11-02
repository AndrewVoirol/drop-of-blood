# `drop-of-blood` — Workspace Agent Runbook & Invariants

## 0. Fast Triage & Anti-Relearning Runbook

1. **Mandatory Turn-0 Skill Read**: Read `/Users/andrewvoirol/.gemini/config/skills/webgpu-interactive-simulation/SKILL.md` before designing, editing, or delegating any shader, fluid, camera, or verification work.
2. **Core Commands**:
   - **Unit Tests (`46/46`)**: `npx vitest run` (or `npm run test:blast -- <file-or-symbol>`)
   - **Static WGSL AST, Struct Alignment & Control-Flow Lint**: `npm run lint:wgsl`
   - **Symbol Line-Number Indexer (Before Reading Monoliths)**: `npm run map:symbols -- <keyword>`
   - **Typecheck + Production Bundle (`TS 7` + `Vite 8` Rolldown)**: `npm run build`
   - **Real User Preset + DOM Hold + Unpaused Temporal Strip (`~12 s`)**: `node scripts/verify.mjs --preset-matrix` (outputs stitched `.verify-out/preset_matrix_sheet.jpg` and `.verify-out/temporal_entry_strip.jpg`)
   - **Live Exploration Audit (`XDR` + Pipette Squeeze + Wave Decay + Hemolysis + Cuvette + Side View)**: `node scripts/verify.mjs --live-audit` (outputs stitched `.verify-out/live_fixes_sheet.jpg`)
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
- **2D Free-Surface Capillary–Gravity Wave Grid**: `WAVE_GRID_SIZE = 256`, `WAVE_HALF_EXTENT = 1.65` ([`src/shaders/common.ts`](src/shaders/common.ts#L1-L2)), spanning `[-1.65, +1.65]` (`3.30 m` full-canvas span, radial absorber `WAVE_SPONGE_INNER = 1.188`, `WAVE_SPONGE_OUTER = 1.617`), so ripples propagate seamlessly across the entire open water surface without a visible container wall.
- **Dual Clocks**:
  - Physics substep: `FIXED_SIM_DT = 1 / 60` ([`src/engine/BloodDropEngine.ts`](src/engine/BloodDropEngine.ts#L82)) representing `1000 fps` slow-motion simulation seconds.
  - User pointer hold (`PENDANT_HOLD_DELAY_MS = 160 ms` in [`src/main.ts`](src/main.ts#L65)): measured in real wall-clock seconds (`0..2.2 s` -> $V \propto t$).
  - Continuous vessel: `autoLoop = false` by default (`393,216`-particle ring buffer wraps cleanly across `> 8` drops; floor *rouleaux* sedimentation accumulates in `concTex.a`).

### B. User HUD `CAMERA_PRESETS` (`src/math/camera.ts` lines `14–42` & `102–125`)
Every visual change MUST be verified through these actual user presets, never solely through custom `window.__QA__.setCam(...)` overrides:
1. **`hero` (`Follow Shot`, default on boot)**:
   - Stays strictly topside (`camPos.y > +0.54 > WATER_SURFACE_Y = +0.30`) across all timestamps (`elev = +0.29 -> +0.365`, `dist = 1.28 -> 1.49`, `target.y = +0.22 -> +0.075`, `fovY = 40° -> 41°`), tilting down to follow the descending vortex ring through the cuvette without diving into a 50/50 TIR ceiling mirror split and keeping `PIPETTE_TIP_Y = +0.55` inside the frustum (`|ndcY| <= 0.95`) on subsequent holds.
2. **`crown` (`Surface Crown`)**:
   - `az = 0.32`, `elev = +0.44`, `dist = 1.48`, `target = [0, 0.11, 0]`, `fovY = 40°` — frames both the entire capillary wave field and `PIPETTE_TIP_Y = +0.55` (`tipNdcY ≈ +0.846`).
3. **`abyss` (`Sub-Meniscus`)**:
   - `az = -0.16`, `elev = -0.21`, `dist = 1.32`, `target = [0, 0.06, 0]`, `fovY = 43°` (dedicated submerged upward view into the wave-warped TIR ceiling & Snell's Window).

---

## 2. Coupled Shader & Texture Contracts

- **`concTex` (`192 × 256 × 192` `rgba16float`)**:
  - `.r`: Intact erythrocyte concentration (turbid Mie scatter + Duysens discrete-cell sieve).
  - `.g`: Lysed free-hemoglobin plasma concentration.
  - `.b`: Oxygenation / shear tracer (`vOxShear` driving `380–720 nm` $\text{HbO}_2$/$\text{dHb}$ dichromatic LUT).
  - `.a`: Settled floor *rouleaux* sediment bed (`floorRouleauxBed`, symmetric mass transfer $\Delta m = \min(\rho, J\,\Delta t)$).
- **Two-Stage Optical Pipeline ([`src/shaders/render.ts`](src/shaders/render.ts))**:
  - Half-res volumetric pass (`1088 × 680` `volTex`: `.rgb` linear radiance, `.a` ray hit distance `tSurf`).
  - `1:1` Retina `COMPOSITE_WGSL` (`2560 × 1600`): perspective-detrended `volTex.a - tFlat` bilateral upsampling + native `1:1` foreground turbid whole-blood aerial/pendant drop intersection + meniscus speculars.
  - **Apple Silicon XDR / EDR Swapchain (`override HDR_EXTENDED`)**: Pre-compiles `pCompositeSdr` (`bgra8unorm`, IEC `linearToSRGB`) and `pCompositeHdr` (`rgba16float`, `toneMapping: { mode: 'extended' }`, `srgb + edrBoost` matching SDR midtones 1:1 with `maxMidtoneSdrDiff <= 0.015` across `[0.05, 0.78]` while opening `> 1.0` specular headroom above linear `0.78`), switched dynamically via `window.matchMedia('(dynamic-range: high)')` or `engine.setExtendedHdr(boolean)`. Never return raw linear `mapped + edrBoost` to an extended-sRGB swapchain (macOS expects piecewise sRGB transfer across `[0, 1]` and otherwise crushes midtones ~`3x` dark).
  - **Hardware `timestamp-query` 5-Pass Telemetry**: When `adapter.features.has('timestamp-query')`, `BloodDropEngine` resolves per-pass GPU durations into `engine.gpuPassTimingsMs` (`macAndWaveMs`, `parcelAndConcMs`, `lightingAndBricksMs`, `halfResVolumetricMs`, `retinaCompositeMs`, `totalGpuMs`) via double-buffered 80-byte readback (`engine.getWorkbenchAdapter()`).
- **Hard Regression Invariants Across All 4 Optical Regimes** (`Bright-Field Topside`, `Dark-Field Topside`, `Sub-Meniscus TIR`, `Steep Snell's Window`):
  - Unbroken vertical stem continuity (`emptyGapScanlines === 0`).
  - Oblate toroidal vortex ring bottom (`centerSpearExcessPx <= 14`, zero pointed axial V-tip spear).
  - Zero `[225, 232, 238]` underwater white eyeball cutouts, zero yellow-tan Soret smoke, zero Dark-Field magenta water ripples.
  - `avgGpuMs <= 12.5 ms` and `p95GpuMs <= 15.5 ms` at `2560 × 1600` (`2x` Retina).
  - Staggered MAC radial forcing evaluated at true face coordinates (`uvwX`, `uvwY`, `uvwZ` in [`src/shaders/fluid.ts`](src/shaders/fluid.ts)).

---

## 3. Physical-Realism Guardrails (Round 9 & Round 10 Contracts)

1. **Turbid Whole-Blood Aerial & Pendant Drops + Quick Pipette Squeeze ([`src/shaders/render.ts`](src/shaders/render.ts), [`src/sim/dropKinematics.ts`](src/sim/dropKinematics.ts))**:
   - Whole human blood (`Hct ~ 45%`, $n \approx 1.378$, erythrocyte Mie $\mu_s' \approx 2.2\text{ mm}^{-1}$) extinguishes ballistic background sky transmission exponentially (`ballisticTrans = exp(-chord * 92.0 * ...)` `< 0.05%` in core) so `3–4 mm` drops read as warm dark oxblood in the core with ruby translucency restricted to `< 0.5 mm` rims and capillary pinch-off necks—never refracting `88%` of the backdrop upside-down like a glass marble.
   - Both quick clicks (`< 160 ms`) and `+ Drop` (`beginQuickPipetteDrop`) squeeze briskly (`~0.15 s`) from the borosilicate Pasteur pipette tip (`PIPETTE_TIP_Y = +0.55`) rather than spawning detached spheres in empty mid-air.
   - Specular reflections evaluate extended rectangular studio softbox strip windows (`evalDropStudioWindowSpec`) weighted by physical dielectric Fresnel ($F_0 \approx 0.025$), never circular Phong point dots (`pow(..., 150)`), paired with Rayleigh–Lamb $l = 2, 3$ capillary shape oscillations and smooth (`plungeFade`) crater coalescence handoff.
2. **Full-Canvas Wave Absorber, Zero Top-Down Bioluminescence & Zero Red "Sonar" Sweep ([`src/shaders/wave.ts`](src/shaders/wave.ts), [`src/shaders/render.ts`](src/shaders/render.ts))**:
   - Drive the FFT PDE with a single momentum-coupled cavity + wetting-line capillary impulse at each drop's true `dropXZ`, damped at the outer horizon by a circular radial Rayleigh velocity absorber (`rayleighDamp`) + physical viscous decay (`viscRate = 0.56 + 0.00018 * k2`) + drainage-gated `forceSubsurface` (`subsurfGate`)—never a pinned circular `eta * sponge` ring that bounces drumhead echoes inward.
   - Gate `evalCausticShafts` strictly to low-angle side/underwater views (`sideViewGate = 1.0 - smoothstep(0.18, 0.42, abs(rd.y))`), damp top-down meniscus speculars (`topDownSpecDamp`), and use min-channel XDR headroom (`excessMin = max(min(r, g, b) - 0.78, 0.0)`) so top-down views never exhibit cyan/bioluminescent bloom or 2×2 half-res `volTex` checkerboard patterns.
   - Gate meniscus blood tinting & swell-throat lensing strictly by `surfaceBloodContact` (`surfBlood`) and keep `plumeGlareDamp` / `clearWaterReliefGate` independent of `activeCurve` (`clearWaterRedRipplePx === 0`) so clear-water ripples over submerged plumes never pulse red or wash oxblood cores into grey discs.
3. **Size- & Dilution-Coupled Dispersion, Pure Absorptive Bright-Field Hemolysis & Circular Floor Pool ([`src/shaders/particles.ts`](src/shaders/particles.ts), [`src/shaders/advect.ts`](src/shaders/advect.ts), [`src/shaders/fluid.ts`](src/shaders/fluid.ts), [`src/shaders/render.ts`](src/shaders/render.ts))**:
   - Keep Stokes settling continuous across age (`stackSettle <= 0.062`, zero `0.165` O-ring cliff), scale initial osmotic pre-swelling (`initLysis`) by inverse drop radius (`1.0 - smoothstep(0.022, 0.056, rDrop)`) and radial exposure, lower `minIntactMass` (`0.46..0.07`) with neighborhood dilution (`diluteExposure`), and start divergence-free curl unfurling at `depthBelowSurf > 0.035`.
   - In Bright-Field (`!darkMode`), dissolved hemoglobin (`cLysed`) acts strictly as a pure translucent ruby-rose absorber (zero additive `haloTransillum` milky-white smoke) while enforcing `gRoseGuard` (`yellowGreenStreakPx === 0`).
   - Use circular radial `length(p.xz)` boundary sponges across `particles.ts`, `advect.ts`, and `fluid.ts` (`smoothstep(halfXZ * 0.68, halfXZ * 0.94, rHoriz)`) plus radial outward floor-cushion glide and slow `bedDecay` on `cRouleauxBed` in an infinite open water canvas (`evalCuvetteFloorBackdrop` = seamless studio backdrop) so settled blood forms a soft circular, gradually clearing pool rather than a square/diamond floor tile.
4. **Topside `Follow` Trajectory & Wave-Warped `Sub-Meniscus` TIR ([`src/math/camera.ts`](src/math/camera.ts), [`src/shaders/render.ts`](src/shaders/render.ts))**:
   - Never auto-dive `syncHeroFollow` barely below `y = +0.30` into a grazing 50/50 TIR ceiling clone.
   - In `Sub-Meniscus` (`abyss`), keep `rPupil` at macro lens scale (`3.5–7.8 mm`) with un-crushed wave slopes and folded water-path attenuation (`tirFoldedAtten`) so underside reflections ripple and soften naturally rather than cloning the plume 1:1.
