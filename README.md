# Drop of Blood

![Slow-motion WebGPU simulation of a blood-red drop entering clear water](screenshots/demo.gif)

Real-time WebGPU slow-motion fluid and optical simulation of a crimson drop falling through air, striking calm water, driving concentric capillary-gravity ripples, and rolling into a 3D toroidal vortex ring with Rayleigh-Taylor secondary tendrils.

## Quick Start

Requires Node.js 20+ and a WebGPU-enabled browser (Chrome 113+, Edge 113+, or Safari 18+).

```bash
git clone https://github.com/andrewvoirol/drop-of-blood.git
cd drop-of-blood
npm install
npm run dev
```

## States

| Surface Impact & Capillary Ripples | Submerged Vortex Plume & HUD |
| :---: | :---: |
| ![Drop impacting calm water surface](screenshots/surface-impact.png) | ![Toroidal vortex plume with underside ripple reflection](screenshots/vortex-plume.png) |

| Dark-Field Rim Illumination | Late Rayleigh-Taylor Tendrils |
| :---: | :---: |
| ![Dark-field studio illumination](screenshots/dark-field.png) | ![Bifurcating secondary tendrils](screenshots/tendrils-bloom.png) |

## How It Works

Drag anywhere on the canvas to orbit the macro camera around the tank, or scroll to zoom in on the meniscus and vortex core. Click **+ Drop** (`D`) to release additional drops into an existing plume, switch slow-motion playback rates (**2400 fps**, **1000 fps**, **500 fps**), jump between **Follow**, **Crown**, and **Sub-Meniscus** camera angles, toggle **Backlit** vs. **Dark Field** studio lighting (`L`), or press `H` for full-bleed **Zen** mode (`Space` pauses, `R` resets the tank).

Under the hood, each frame executes a zero-allocation WebGPU compute and optical pipeline in a single command submission: a `256 × 256` volume-conserving capillary-gravity wave solver with a dimensionless 9-point isotropic Laplacian and $-\nabla^4 \eta$ surface-tension dispersion updates the water meniscus; a `96 × 128 × 96` 3D incompressible Navier-Stokes solver (`r32float` 26-iteration warm-started Jacobi pressure projection, central-difference vorticity confinement, and Boussinesq negative buoyancy) drives toroidal roll-up; a super-resolved `192 × 256 × 192` monotonic MacCormack (BFECC) pass advects erythrocyte and plasma concentration fields; and a full-screen pinhole volumetric raytracer evaluates bisection heightfield refraction, underside Total Internal Reflection, RGB Beer-Lambert hemoglobin extinction (`σ = [136, 820, 1100]`), dual-lobe Henyey-Greenstein scattering, and C1-continuous Khronos PBR Neutral tone mapping.

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| Build & Dev Server | Vite 6 |
| Language | TypeScript 5.8 (strict) |
| GPU Compute & Rendering | WebGPU + WGSL (zero runtime dependencies) |
| Unit Testing | Vitest 3 |
