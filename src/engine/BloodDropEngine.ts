import { CAMERA_PRESETS, OrbitCamera } from '../math/camera';
import type { CameraPreset } from '../math/camera';
import {
  CONC_CLEAR_WGSL,
  CONC_PREDICT_WGSL,
  CONC_RESOLVE_WGSL,
} from '../shaders/advect';
import {
  BRICK_GRID_X,
  BRICK_GRID_Y,
  BRICK_GRID_Z,
  CONC_GRID_X,
  CONC_GRID_Y,
  CONC_GRID_Z,
  CONC_TOTAL_VOXELS,
  PARTICLE_COUNT,
  PARTICLE_WORKGROUP_SIZE,
  SPECTRAL_LUT_HEIGHT,
  SPECTRAL_LUT_WIDTH,
  VEL_GRID_X,
  VEL_GRID_Y,
  VEL_GRID_Z,
  WAVE_GRID_SIZE,
} from '../shaders/common';
import {
  FLUID_ADVECT_MAC_WGSL,
  FLUID_CASSON_FORCES_WGSL,
  FLUID_CLEAR_R32_WGSL,
  FLUID_CURL_WGSL,
  FLUID_INIT_VEL_WGSL,
  FLUID_MAC_PROJECT_WGSL,
  FLUID_MAC_RHS_WGSL,
  FLUID_MAC_TO_CENTER_WGSL,
  MG_COARSE_SOLVE_L3_WGSL,
  MG_PROLONG_ADD_WGSL,
  MG_RBGS_BLACK_WGSL,
  MG_RBGS_RED_WGSL,
  MG_RBGS_RED_ZERO_WGSL,
  MG_RESIDUAL_WGSL,
  MG_RESTRICT_WGSL,
} from '../shaders/fluid';
import {
  LIGHTING_VOLUME_WGSL,
  PLUME_OCCUPANCY_DILATE_WGSL,
  PLUME_OCCUPANCY_RAW_WGSL,
} from '../shaders/lighting';
import {
  PARCELS_PER_REF_DROP,
  PARTICLE_ADVECT_SPLAT_WGSL,
  PARTICLE_EMIT_WGSL,
  PARTICLE_RESET_WGSL,
} from '../shaders/particles';
import { COMPOSITE_WGSL, RENDER_WGSL } from '../shaders/render';
import {
  WAVE_CLEAR_SPEC_WGSL,
  WAVE_CLEAR_WGSL,
  WAVE_FFT_COL_WGSL,
  WAVE_FFT_ROW_WGSL,
  WAVE_FINALIZE_WGSL,
  WAVE_IFFT_COL_WGSL,
  WAVE_IFFT_ROW_WGSL,
  WAVE_SPATIAL_FORCE_WGSL,
  WAVE_SPECTRAL_PROPAGATE_WGSL,
} from '../shaders/wave';
import {
  INITIAL_DROP_RADIUS,
  MultiDropSystem,
  REFERENCE_DROP_VOLUME,
  SHOT_CYCLE_DURATION,
  raycastMeniscusXZ,
} from '../sim/dropKinematics';
import type { DropSlotState, InspectionSubject, MultiDropSnapshot } from '../sim/dropKinematics';
import { generatePrahlSpectralLut } from '../sim/spectralLut';
import {
  DEFAULT_FLUID_OPTICAL_CONFIG,
  SIM_UNIFORM_BYTES,
  SIM_UNIFORM_FLOATS,
  SimUniformPacker,
} from '../sim/uniformLayout';
import type { FluidOpticalConfig } from '../sim/uniformLayout';

export const FIXED_SIM_DT = 1.0 / 60.0;
const MAX_SUBSTEPS = 3;
const PAUSED_ACCUM_TARGET = 16;
const PARTICLE_STRIDE_BYTES = 32; // 2 * vec4<f32> per RbcParcel
const VOL_RENDER_MAX_PIXELS = 739_840; // Exact 1088x680 (16:10 isotropic, 8px aligned) + native 1:1 2560x1600 Retina drop/collar/meniscus/reflection composite

const MG_L1_X = VEL_GRID_X / 2; // 48
const MG_L1_Y = VEL_GRID_Y / 2; // 64
const MG_L1_Z = VEL_GRID_Z / 2; // 48

const MG_L2_X = VEL_GRID_X / 4; // 24
const MG_L2_Y = VEL_GRID_Y / 4; // 32
const MG_L2_Z = VEL_GRID_Z / 4; // 24

const MG_L3_X = VEL_GRID_X / 8; // 12
const MG_L3_Y = VEL_GRID_Y / 8; // 16
const MG_L3_Z = VEL_GRID_Z / 8; // 12

export interface GpuPassTimingsMs {
  macAndWaveMs: number;
  parcelAndConcMs: number;
  lightingAndBricksMs: number;
  halfResVolumetricMs: number;
  retinaCompositeMs: number;
  totalGpuMs: number;
  sampleCount: number;
}

export class BloodDropEngine {
  public readonly canvas: HTMLCanvasElement;
  public readonly device: GPUDevice;
  public readonly context: GPUCanvasContext;
  public format: GPUTextureFormat;
  public readonly preferredSdrFormat: GPUTextureFormat;
  public readonly supportsTimestamps: boolean;
  public isExtendedHdr = false;
  public readonly gpuPassTimingsMs: GpuPassTimingsMs = {
    macAndWaveMs: 0,
    parcelAndConcMs: 0,
    lightingAndBricksMs: 0,
    halfResVolumetricMs: 0,
    retinaCompositeMs: 0,
    totalGpuMs: 0,
    sampleCount: 0,
  };
  public readonly camera = new OrbitCamera(CAMERA_PRESETS.hero);
  public readonly config: FluidOpticalConfig = { ...DEFAULT_FLUID_OPTICAL_CONFIG };
  public readonly multiDrops = new MultiDropSystem();

  public simTime = 0.0;
  public timeScale = 1.0; // 1.0 = ~1000 fps Hero Slow Motion
  public paused = false;
  public autoLoop = false;
  public dropSeed = 1.0;
  public dropOffsetXZ: [number, number] = [0.0, 0.0];
  public frameIndex = 0;
  public isDeviceLost = false;
  public onDeviceLost?: (info: GPUDeviceLostInfo) => void;
  public readonly validationErrors: string[] = [];

  private simAccumulator = 0.0;
  private waveState: 0 | 1 = 0;
  private velState: 0 | 1 = 0;
  private pressState: 0 | 1 = 0;
  private concState: 0 | 1 = 0;
  private spawnCursor = 0;
  private highWaterParticles = 0;
  private parcelFractionCarry = 0.0;
  private destroyed = false;

  private readonly uniformPacker = new SimUniformPacker();
  private readonly multiStepScratch = new Float32Array(SIM_UNIFORM_FLOATS * MAX_SUBSTEPS);
  private readonly substepSpawnCounts = new Int32Array(MAX_SUBSTEPS);
  private heroTopsideCue = 0.0;
  private pendingManualSubsteps = 0;
  private lockedSubjectId: number | null = null;
  private pausedAccumCount = 0;
  private pausedBaseFrame = 0;
  private readonly lastPausedViewProj = new Float32Array(16);
  private lastPausedLightingMode = -1.0;
  private lastPausedSimTime = -1.0;
  private lastPausedDropSig = -1.0;

  private readonly uniformBuffer: GPUBuffer;
  private readonly uniformStagingBuffer: GPUBuffer;
  private readonly particlesBuffer: GPUBuffer;
  private readonly splatIntactBuffer: GPUBuffer;
  private readonly splatLysedBuffer: GPUBuffer;
  private readonly linearSampler: GPUSampler;

  // Textures
  private readonly texWave: [GPUTexture, GPUTexture];
  private readonly texWaveSpecAux: GPUTexture;
  private readonly texWaveSpecA: GPUTexture;
  private readonly texWaveSpecB: GPUTexture;

  private readonly texVel: [GPUTexture, GPUTexture];
  private readonly texVelCenter: GPUTexture;
  private readonly texCurl: GPUTexture;

  // 3-Level Geometric Multigrid r32float textures
  private readonly texPress: [GPUTexture, GPUTexture];
  private readonly texRhs0: GPUTexture;
  private readonly texRes0: GPUTexture;

  private readonly texMgErr1: [GPUTexture, GPUTexture];
  private readonly texMgRhs1: GPUTexture;
  private readonly texMgRes1: GPUTexture;

  private readonly texMgErr2: [GPUTexture, GPUTexture];
  private readonly texMgRhs2: GPUTexture;
  private readonly texMgRes2: GPUTexture;

  private readonly texMgErr3: [GPUTexture, GPUTexture];
  private readonly texMgRhs3: GPUTexture;

  private readonly texConc: [GPUTexture, GPUTexture];
  private readonly texConcPred: GPUTexture;
  private readonly texLight: GPUTexture;
  private readonly texBrickRaw: GPUTexture;
  private readonly texBrickDilated: GPUTexture;
  private readonly texSpectralLut: GPUTexture;
  private texVolRender: GPUTexture;
  private viewVolRender: GPUTextureView;
  private volSwapW = 0;
  private volSwapH = 0;

  // Compute & Render Pipelines
  private readonly pWaveClear: GPUComputePipeline;
  private readonly pWaveClearSpec: GPUComputePipeline;
  private readonly pWaveSpatialForce: GPUComputePipeline;
  private readonly pWaveFftRow: GPUComputePipeline;
  private readonly pWaveFftCol: GPUComputePipeline;
  private readonly pWaveSpectralProp: GPUComputePipeline;
  private readonly pWaveIfftCol: GPUComputePipeline;
  private readonly pWaveIfftRow: GPUComputePipeline;
  private readonly pWaveFinalize: GPUComputePipeline;

  private readonly pFluidInitVel: GPUComputePipeline;
  private readonly pFluidClearR32: GPUComputePipeline;
  private readonly pFluidAdvectMac: GPUComputePipeline;
  private readonly pFluidCurl: GPUComputePipeline;
  private readonly pFluidCassonForces: GPUComputePipeline;
  private readonly pFluidMacRhs: GPUComputePipeline;

  private readonly pMgRedZero: GPUComputePipeline;
  private readonly pMgRed: GPUComputePipeline;
  private readonly pMgBlack: GPUComputePipeline;
  private readonly pMgCoarseL3: GPUComputePipeline;
  private readonly pMgResidual: GPUComputePipeline;
  private readonly pMgRestrict: GPUComputePipeline;
  private readonly pMgProlong: GPUComputePipeline;

  private readonly pFluidMacProject: GPUComputePipeline;
  private readonly pFluidMacToCenter: GPUComputePipeline;

  private readonly pParticleReset: GPUComputePipeline;
  private readonly pParticleEmit: GPUComputePipeline;
  private readonly pParticleAdvectSplat: GPUComputePipeline;

  private readonly pConcClear: GPUComputePipeline;
  private readonly pConcPredict: GPUComputePipeline;
  private readonly pConcResolve: GPUComputePipeline;
  private readonly pLighting: GPUComputePipeline;
  private readonly pBrickRaw: GPUComputePipeline;
  private readonly pBrickDilate: GPUComputePipeline;
  private readonly bglComposite: GPUBindGroupLayout;
  private readonly pRender: GPURenderPipeline;
  private readonly pRenderMacro: GPURenderPipeline;
  private readonly pRenderAccum: GPURenderPipeline;
  private readonly pCompositeSdr: GPURenderPipeline;
  private readonly pCompositeHdr: GPURenderPipeline;
  private pComposite: GPURenderPipeline;

  private readonly tsQuerySet: GPUQuerySet | null = null;
  private readonly tsResolveBuffer: GPUBuffer | null = null;
  private readonly tsReadBuffers: [GPUBuffer, GPUBuffer] | null = null;
  private readonly tsReadInFlight: [boolean, boolean] = [false, false];
  private tsRingCursor: 0 | 1 = 0;
  private readonly tsPass0: GPUComputePassTimestampWrites | undefined;
  private readonly tsPass1: GPUComputePassTimestampWrites | undefined;
  private readonly tsPass2: GPUComputePassTimestampWrites | undefined;
  private readonly tsPass3: GPURenderPassTimestampWrites | undefined;
  private readonly tsPass4: GPURenderPassTimestampWrites | undefined;
  private hdrMediaQuery: MediaQueryList | null = null;
  private onHdrMediaChange: ((e: MediaQueryListEvent) => void) | null = null;

  // Pre-allocated BindGroup permutations (Zero-allocation render loop invariant)
  private readonly bgWaveClear: [GPUBindGroup, GPUBindGroup];
  private readonly bgWaveClearSpec: [GPUBindGroup, GPUBindGroup, GPUBindGroup];
  private readonly bgWaveSpatialForce: [[GPUBindGroup, GPUBindGroup], [GPUBindGroup, GPUBindGroup]];
  private readonly bgWaveFftRow: GPUBindGroup;
  private readonly bgWaveFftCol: GPUBindGroup;
  private readonly bgWaveSpectralProp: GPUBindGroup;
  private readonly bgWaveIfftCol: GPUBindGroup;
  private readonly bgWaveIfftRow: GPUBindGroup;
  private readonly bgWaveFinalize: [GPUBindGroup, GPUBindGroup];

  private readonly bgFluidInitVel: [GPUBindGroup, GPUBindGroup, GPUBindGroup, GPUBindGroup];
  private readonly bgFluidClearR32: [GPUBindGroup, GPUBindGroup, GPUBindGroup, GPUBindGroup];
  private readonly bgFluidAdvectMac: [[GPUBindGroup, GPUBindGroup], [GPUBindGroup, GPUBindGroup]];
  private readonly bgFluidCurl: GPUBindGroup;
  private readonly bgFluidCassonForces: [[GPUBindGroup, GPUBindGroup], [GPUBindGroup, GPUBindGroup]];
  private readonly bgFluidMacRhs: [[GPUBindGroup, GPUBindGroup], [GPUBindGroup, GPUBindGroup]];

  // Multigrid V-cycle BindGroups
  private readonly bgMgL0Red: [GPUBindGroup, GPUBindGroup];
  private readonly bgMgL0Black: [GPUBindGroup, GPUBindGroup];
  private readonly bgMgL0Res: [GPUBindGroup, GPUBindGroup];
  private readonly bgMgRestrict0to1: GPUBindGroup;

  private readonly bgMgL1RedZero: GPUBindGroup;
  private readonly bgMgL1Red0to1: GPUBindGroup;
  private readonly bgMgL1Red1to0: GPUBindGroup;
  private readonly bgMgL1Black1to0: GPUBindGroup;
  private readonly bgMgL1Black0to1: GPUBindGroup;
  private readonly bgMgL1Res: GPUBindGroup;
  private readonly bgMgRestrict1to2: GPUBindGroup;

  private readonly bgMgL2RedZero: GPUBindGroup;
  private readonly bgMgL2Red0to1: GPUBindGroup;
  private readonly bgMgL2Red1to0: GPUBindGroup;
  private readonly bgMgL2Black1to0: GPUBindGroup;
  private readonly bgMgL2Black0to1: GPUBindGroup;
  private readonly bgMgL2Res: GPUBindGroup;
  private readonly bgMgRestrict2to3: GPUBindGroup;

  private readonly bgMgL3Coarse: GPUBindGroup;

  private readonly bgMgProlong3to2: GPUBindGroup;
  private readonly bgMgProlong2to1: GPUBindGroup;
  private readonly bgMgProlong1to0: [GPUBindGroup, GPUBindGroup];

  private readonly bgFluidMacProject: [
    [[GPUBindGroup, GPUBindGroup], [GPUBindGroup, GPUBindGroup]],
    [[GPUBindGroup, GPUBindGroup], [GPUBindGroup, GPUBindGroup]],
  ];
  private readonly bgFluidMacToCenter: [GPUBindGroup, GPUBindGroup];

  private readonly bgParticleReset: GPUBindGroup;
  private readonly bgParticleEmit: [GPUBindGroup, GPUBindGroup];
  private readonly bgParticleAdvectSplat: [GPUBindGroup, GPUBindGroup];

  private readonly bgConcClear: [GPUBindGroup, GPUBindGroup, GPUBindGroup];
  private readonly bgConcPredict: [GPUBindGroup, GPUBindGroup];
  private readonly bgConcResolve: [GPUBindGroup, GPUBindGroup];
  private readonly bgLighting: [[GPUBindGroup, GPUBindGroup], [GPUBindGroup, GPUBindGroup]];
  private readonly bgBrickRaw: GPUBindGroup;
  private readonly bgBrickDilate: GPUBindGroup;
  private readonly viewWave: [GPUTextureView, GPUTextureView];
  private readonly viewSpectralLut: GPUTextureView;
  private readonly bgRender: [[GPUBindGroup, GPUBindGroup], [GPUBindGroup, GPUBindGroup]];
  private bgComposite: [GPUBindGroup, GPUBindGroup];

  public static async create(canvas: HTMLCanvasElement): Promise<BloodDropEngine> {
    if (!navigator.gpu) {
      throw new Error('WebGPU is not supported in this browser.');
    }
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) {
      throw new Error('Unable to acquire a high-performance WebGPU adapter.');
    }
    const supportsTimestamps = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice(
      supportsTimestamps ? { requiredFeatures: ['timestamp-query'] } : undefined
    );
    const context = canvas.getContext('webgpu');
    if (!context) {
      throw new Error('Unable to acquire a WebGPU canvas context.');
    }
    const preferredSdrFormat = navigator.gpu.getPreferredCanvasFormat();
    const hdrQuery: MediaQueryList | null = null;
    const format: GPUTextureFormat = preferredSdrFormat;
    const isExtendedHdr = false;
    try {
      context.configure({
        device,
        format: preferredSdrFormat,
        alphaMode: 'opaque',
        toneMapping: { mode: 'standard' },
      });
    } catch {
      context.configure({
        device,
        format: preferredSdrFormat,
        alphaMode: 'opaque',
      });
    }

    device.pushErrorScope('validation');
    device.pushErrorScope('internal');

    const engine = new BloodDropEngine(
      canvas,
      device,
      context,
      format,
      preferredSdrFormat,
      isExtendedHdr,
      supportsTimestamps,
      hdrQuery
    );
    await engine.verifyCleanInit();
    engine.resetSimulation(1.0);
    engine.dispatchLightingOnly();

    return engine;
  }

  private constructor(
    canvas: HTMLCanvasElement,
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    preferredSdrFormat: GPUTextureFormat,
    isExtendedHdr: boolean,
    supportsTimestamps: boolean,
    hdrQuery: MediaQueryList | null
  ) {
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.format = format;
    this.preferredSdrFormat = preferredSdrFormat;
    this.isExtendedHdr = isExtendedHdr;
    this.supportsTimestamps = supportsTimestamps;

    if (supportsTimestamps) {
      this.tsQuerySet = device.createQuerySet({
        label: 'BloodDropPassTimestamps',
        type: 'timestamp',
        count: 10,
      });
      this.tsResolveBuffer = device.createBuffer({
        label: 'BloodDropTimestampResolve',
        size: 80,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.tsReadBuffers = [
        device.createBuffer({
          label: 'BloodDropTimestampRead0',
          size: 80,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        device.createBuffer({
          label: 'BloodDropTimestampRead1',
          size: 80,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
      ];
      this.tsPass0 = { querySet: this.tsQuerySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
      this.tsPass1 = { querySet: this.tsQuerySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 };
      this.tsPass2 = { querySet: this.tsQuerySet, beginningOfPassWriteIndex: 4, endOfPassWriteIndex: 5 };
      this.tsPass3 = { querySet: this.tsQuerySet, beginningOfPassWriteIndex: 6, endOfPassWriteIndex: 7 };
      this.tsPass4 = { querySet: this.tsQuerySet, beginningOfPassWriteIndex: 8, endOfPassWriteIndex: 9 };
    }

    if (hdrQuery) {
      this.hdrMediaQuery = hdrQuery;
      this.onHdrMediaChange = (e: MediaQueryListEvent) => {
        if (!this.destroyed) {
          this.setExtendedHdr(e.matches);
        }
      };
      hdrQuery.addEventListener('change', this.onHdrMediaChange);
    }

    this.device.addEventListener('uncapturederror', (ev) => {
      const msg = ev.error?.message ?? String(ev.error);
      this.validationErrors.push(msg);
      console.error('[WebGPU Uncaptured Error]', msg);
    });

    void this.device.lost.then((info) => {
      if (this.destroyed) return;
      this.isDeviceLost = true;
      this.destroyed = true;
      const msg = info.message || info.reason || 'WebGPU device lost';
      this.validationErrors.push(`[DeviceLost] ${msg}`);
      console.error('[WebGPU Device Lost]', msg);
      this.onDeviceLost?.(info);
    });

    this.uniformBuffer = device.createBuffer({
      label: 'SimUniforms',
      size: SIM_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.uniformStagingBuffer = device.createBuffer({
      label: 'SimUniformsStaging',
      size: SIM_UNIFORM_BYTES * MAX_SUBSTEPS,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    this.particlesBuffer = device.createBuffer({
      label: 'LagrangianRbcParcels',
      size: PARTICLE_COUNT * PARTICLE_STRIDE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.splatIntactBuffer = device.createBuffer({
      label: 'AtomicSplatIntact',
      size: CONC_TOTAL_VOXELS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.splatLysedBuffer = device.createBuffer({
      label: 'AtomicSplatLysed',
      size: CONC_TOTAL_VOXELS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.linearSampler = device.createSampler({
      label: 'LinearClampSampler',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });

    const texUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;

    const createWave2DF16 = (label: string) =>
      device.createTexture({
        label,
        size: [WAVE_GRID_SIZE, WAVE_GRID_SIZE, 1],
        dimension: '2d',
        format: 'rgba16float',
        usage: texUsage,
      });

    const createWave2DF32 = (label: string) =>
      device.createTexture({
        label,
        size: [WAVE_GRID_SIZE, WAVE_GRID_SIZE, 1],
        dimension: '2d',
        format: 'rgba32float',
        usage: texUsage,
      });

    const createGrid3DF16 = (label: string, w: number, h: number, d: number) =>
      device.createTexture({
        label,
        size: [w, h, d],
        dimension: '3d',
        format: 'rgba16float',
        usage: texUsage,
      });

    const createGrid3DR32 = (label: string, w: number, h: number, d: number) =>
      device.createTexture({
        label,
        size: [w, h, d],
        dimension: '3d',
        format: 'r32float',
        usage: texUsage,
      });

    this.texWave = [createWave2DF16('WaveA'), createWave2DF16('WaveB')];
    this.texWaveSpecAux = createWave2DF32('WaveSpecAux');
    this.texWaveSpecA = createWave2DF32('WaveSpecA');
    this.texWaveSpecB = createWave2DF32('WaveSpecB');

    this.texVel = [
      createGrid3DF16('VelMacA', VEL_GRID_X, VEL_GRID_Y, VEL_GRID_Z),
      createGrid3DF16('VelMacB', VEL_GRID_X, VEL_GRID_Y, VEL_GRID_Z),
    ];
    this.texVelCenter = createGrid3DF16('VelCenter', VEL_GRID_X, VEL_GRID_Y, VEL_GRID_Z);
    this.texCurl = createGrid3DF16('Curl', VEL_GRID_X, VEL_GRID_Y, VEL_GRID_Z);

    // 3-Level Multigrid Hierarchy
    this.texPress = [
      createGrid3DR32('PressL0A', VEL_GRID_X, VEL_GRID_Y, VEL_GRID_Z),
      createGrid3DR32('PressL0B', VEL_GRID_X, VEL_GRID_Y, VEL_GRID_Z),
    ];
    this.texRhs0 = createGrid3DR32('MgRhs0', VEL_GRID_X, VEL_GRID_Y, VEL_GRID_Z);
    this.texRes0 = createGrid3DR32('MgRes0', VEL_GRID_X, VEL_GRID_Y, VEL_GRID_Z);

    this.texMgErr1 = [
      createGrid3DR32('MgErr1A', MG_L1_X, MG_L1_Y, MG_L1_Z),
      createGrid3DR32('MgErr1B', MG_L1_X, MG_L1_Y, MG_L1_Z),
    ];
    this.texMgRhs1 = createGrid3DR32('MgRhs1', MG_L1_X, MG_L1_Y, MG_L1_Z);
    this.texMgRes1 = createGrid3DR32('MgRes1', MG_L1_X, MG_L1_Y, MG_L1_Z);

    this.texMgErr2 = [
      createGrid3DR32('MgErr2A', MG_L2_X, MG_L2_Y, MG_L2_Z),
      createGrid3DR32('MgErr2B', MG_L2_X, MG_L2_Y, MG_L2_Z),
    ];
    this.texMgRhs2 = createGrid3DR32('MgRhs2', MG_L2_X, MG_L2_Y, MG_L2_Z);
    this.texMgRes2 = createGrid3DR32('MgRes2', MG_L2_X, MG_L2_Y, MG_L2_Z);

    this.texMgErr3 = [
      createGrid3DR32('MgErr3A', MG_L3_X, MG_L3_Y, MG_L3_Z),
      createGrid3DR32('MgErr3B', MG_L3_X, MG_L3_Y, MG_L3_Z),
    ];
    this.texMgRhs3 = createGrid3DR32('MgRhs3', MG_L3_X, MG_L3_Y, MG_L3_Z);

    this.texConc = [
      createGrid3DF16('ConcA', CONC_GRID_X, CONC_GRID_Y, CONC_GRID_Z),
      createGrid3DF16('ConcB', CONC_GRID_X, CONC_GRID_Y, CONC_GRID_Z),
    ];
    this.texConcPred = createGrid3DF16('ConcPred', CONC_GRID_X, CONC_GRID_Y, CONC_GRID_Z);
    this.texLight = createGrid3DF16('LightVol', VEL_GRID_X, VEL_GRID_Y, VEL_GRID_Z);
    this.texBrickRaw = createGrid3DR32('BrickOccRaw', BRICK_GRID_X, BRICK_GRID_Y, BRICK_GRID_Z);
    this.texBrickDilated = createGrid3DR32('BrickOccDilated', BRICK_GRID_X, BRICK_GRID_Y, BRICK_GRID_Z);

    // Precompute Prahl (1999) CIE 1931 spectral hemoglobin transmittance LUT once at startup
    this.texSpectralLut = device.createTexture({
      label: 'PrahlSpectralHemoglobinLut',
      size: [SPECTRAL_LUT_WIDTH, SPECTRAL_LUT_HEIGHT, 1],
      dimension: '2d',
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const lut = generatePrahlSpectralLut();
    device.queue.writeTexture(
      { texture: this.texSpectralLut },
      lut.rgba16.buffer,
      {
        offset: lut.rgba16.byteOffset,
        bytesPerRow: SPECTRAL_LUT_WIDTH * 8,
        rowsPerImage: SPECTRAL_LUT_HEIGHT,
      },
      [SPECTRAL_LUT_WIDTH, SPECTRAL_LUT_HEIGHT, 1]
    );

    // Explicit BindGroupLayouts eliminate dead-binding reflection mismatches
    const C = GPUShaderStage.COMPUTE;
    const F = GPUShaderStage.FRAGMENT;
    const uEntry: GPUBindGroupLayoutEntry = {
      binding: 0,
      visibility: C,
      buffer: { type: 'uniform' },
    };

    const bglStore2DF16 = device.createBindGroupLayout({
      label: 'BGL_Store2DF16',
      entries: [
        uEntry,
        { binding: 1, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' } },
      ],
    });

    const bglStore2DF32 = device.createBindGroupLayout({
      label: 'BGL_Store2DF32',
      entries: [
        uEntry,
        { binding: 1, visibility: C, storageTexture: { access: 'write-only', format: 'rgba32float', viewDimension: '2d' } },
      ],
    });

    const bglWaveSpatialForce = device.createBindGroupLayout({
      label: 'BGL_WaveSpatialForce',
      entries: [
        uEntry,
        { binding: 1, visibility: C, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 2, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 3, visibility: C, sampler: { type: 'filtering' } },
        { binding: 4, visibility: C, storageTexture: { access: 'write-only', format: 'rgba32float', viewDimension: '2d' } },
      ],
    });

    const bglWaveSpecPass = device.createBindGroupLayout({
      label: 'BGL_WaveSpecPass',
      entries: [
        uEntry,
        { binding: 1, visibility: C, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 2, visibility: C, storageTexture: { access: 'write-only', format: 'rgba32float', viewDimension: '2d' } },
      ],
    });

    const bglWaveFinalize = device.createBindGroupLayout({
      label: 'BGL_WaveFinalize',
      entries: [
        uEntry,
        { binding: 1, visibility: C, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 2, visibility: C, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 3, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' } },
      ],
    });

    const bglStore3DF16 = device.createBindGroupLayout({
      label: 'BGL_Store3DF16',
      entries: [
        uEntry,
        { binding: 1, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
      ],
    });

    const bglStore3DR32 = device.createBindGroupLayout({
      label: 'BGL_Store3DR32',
      entries: [
        uEntry,
        { binding: 1, visibility: C, storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' } },
      ],
    });

    const bglFluidAdvectMac = device.createBindGroupLayout({
      label: 'BGL_FluidAdvectMac',
      entries: [
        uEntry,
        { binding: 1, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 3, visibility: C, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 4, visibility: C, sampler: { type: 'filtering' } },
        { binding: 5, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
      ],
    });

    const bglTex3DToStore3DF16 = device.createBindGroupLayout({
      label: 'BGL_Tex3DToStore3DF16',
      entries: [
        uEntry,
        { binding: 1, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
      ],
    });

    const bglFluidCassonForces = device.createBindGroupLayout({
      label: 'BGL_FluidCassonForces',
      entries: [
        uEntry,
        { binding: 1, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 3, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 4, visibility: C, sampler: { type: 'filtering' } },
        { binding: 5, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
      ],
    });

    const bglFluidMacRhs = device.createBindGroupLayout({
      label: 'BGL_FluidMacRhs',
      entries: [
        uEntry,
        { binding: 1, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: C, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 3, visibility: C, sampler: { type: 'filtering' } },
        { binding: 4, visibility: C, storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' } },
      ],
    });

    const bglMg1In = device.createBindGroupLayout({
      label: 'BGL_Mg1In',
      entries: [
        uEntry,
        { binding: 1, visibility: C, texture: { sampleType: 'unfilterable-float', viewDimension: '3d' } },
        { binding: 2, visibility: C, storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' } },
      ],
    });

    const bglMg2In = device.createBindGroupLayout({
      label: 'BGL_Mg2In',
      entries: [
        uEntry,
        { binding: 1, visibility: C, texture: { sampleType: 'unfilterable-float', viewDimension: '3d' } },
        { binding: 2, visibility: C, texture: { sampleType: 'unfilterable-float', viewDimension: '3d' } },
        { binding: 3, visibility: C, storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' } },
      ],
    });

    const bglFluidMacProject = device.createBindGroupLayout({
      label: 'BGL_FluidMacProject',
      entries: [
        uEntry,
        { binding: 1, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: C, texture: { sampleType: 'unfilterable-float', viewDimension: '3d' } },
        { binding: 3, visibility: C, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 4, visibility: C, sampler: { type: 'filtering' } },
        { binding: 5, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
      ],
    });

    const bglParticleReset = device.createBindGroupLayout({
      label: 'BGL_ParticleReset',
      entries: [
        uEntry,
        { binding: 1, visibility: C, buffer: { type: 'storage' } },
      ],
    });

    const bglParticleEmit = device.createBindGroupLayout({
      label: 'BGL_ParticleEmit',
      entries: [
        uEntry,
        { binding: 1, visibility: C, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 2, visibility: C, sampler: { type: 'filtering' } },
        { binding: 3, visibility: C, buffer: { type: 'storage' } },
        { binding: 4, visibility: C, buffer: { type: 'storage' } },
      ],
    });

    const bglParticleAdvectSplat = device.createBindGroupLayout({
      label: 'BGL_ParticleAdvectSplat',
      entries: [
        uEntry,
        { binding: 1, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 3, visibility: C, sampler: { type: 'filtering' } },
        { binding: 4, visibility: C, buffer: { type: 'storage' } },
        { binding: 5, visibility: C, buffer: { type: 'storage' } },
        { binding: 6, visibility: C, buffer: { type: 'storage' } },
      ],
    });

    const bglConcPredict = device.createBindGroupLayout({
      label: 'BGL_ConcPredict',
      entries: [
        uEntry,
        { binding: 1, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 3, visibility: C, sampler: { type: 'filtering' } },
        { binding: 4, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
      ],
    });

    const bglConcResolve = device.createBindGroupLayout({
      label: 'BGL_ConcResolve',
      entries: [
        uEntry,
        { binding: 1, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 3, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 4, visibility: C, sampler: { type: 'filtering' } },
        { binding: 5, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 7, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
      ],
    });

    const bglLighting = device.createBindGroupLayout({
      label: 'BGL_Lighting',
      entries: [
        uEntry,
        { binding: 1, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: C, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 3, visibility: C, sampler: { type: 'filtering' } },
        { binding: 4, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
      ],
    });

    const bglBrickRaw = device.createBindGroupLayout({
      label: 'BGL_BrickRaw',
      entries: [
        uEntry,
        { binding: 1, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: C, storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' } },
      ],
    });

    const bglRender = device.createBindGroupLayout({
      label: 'BGL_Render',
      entries: [
        { binding: 0, visibility: F, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: F, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 3, visibility: F, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 4, visibility: F, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 5, visibility: F, sampler: { type: 'filtering' } },
        { binding: 6, visibility: F, texture: { sampleType: 'unfilterable-float', viewDimension: '3d' } },
      ],
    });

    this.bglComposite = device.createBindGroupLayout({
      label: 'BGL_Composite',
      entries: [
        { binding: 0, visibility: F, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 2, visibility: F, sampler: { type: 'filtering' } },
        { binding: 3, visibility: F, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 4, visibility: F, texture: { sampleType: 'float', viewDimension: '2d' } },
      ],
    });

    const makeCompute = (label: string, code: string, bgl: GPUBindGroupLayout) => {
      const mod = device.createShaderModule({ label, code });
      this.inspectCompilation(label, mod);
      return device.createComputePipeline({
        label,
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: { module: mod, entryPoint: 'main' },
      });
    };

    this.pWaveClear = makeCompute('WaveClear', WAVE_CLEAR_WGSL, bglStore2DF16);
    this.pWaveClearSpec = makeCompute('WaveClearSpec', WAVE_CLEAR_SPEC_WGSL, bglStore2DF32);
    this.pWaveSpatialForce = makeCompute('WaveSpatialForce', WAVE_SPATIAL_FORCE_WGSL, bglWaveSpatialForce);
    this.pWaveFftRow = makeCompute('WaveFftRow', WAVE_FFT_ROW_WGSL, bglWaveSpecPass);
    this.pWaveFftCol = makeCompute('WaveFftCol', WAVE_FFT_COL_WGSL, bglWaveSpecPass);
    this.pWaveSpectralProp = makeCompute('WaveSpectralProp', WAVE_SPECTRAL_PROPAGATE_WGSL, bglWaveSpecPass);
    this.pWaveIfftCol = makeCompute('WaveIfftCol', WAVE_IFFT_COL_WGSL, bglWaveSpecPass);
    this.pWaveIfftRow = makeCompute('WaveIfftRow', WAVE_IFFT_ROW_WGSL, bglWaveSpecPass);
    this.pWaveFinalize = makeCompute('WaveFinalize', WAVE_FINALIZE_WGSL, bglWaveFinalize);

    this.pFluidInitVel = makeCompute('FluidInitVel', FLUID_INIT_VEL_WGSL, bglStore3DF16);
    this.pFluidClearR32 = makeCompute('FluidClearR32', FLUID_CLEAR_R32_WGSL, bglStore3DR32);
    this.pFluidAdvectMac = makeCompute('FluidAdvectMac', FLUID_ADVECT_MAC_WGSL, bglFluidAdvectMac);
    this.pFluidCurl = makeCompute('FluidCurl', FLUID_CURL_WGSL, bglTex3DToStore3DF16);
    this.pFluidCassonForces = makeCompute('FluidCassonForces', FLUID_CASSON_FORCES_WGSL, bglFluidCassonForces);
    this.pFluidMacRhs = makeCompute('FluidMacRhs', FLUID_MAC_RHS_WGSL, bglFluidMacRhs);

    this.pMgRedZero = makeCompute('MgRedZero', MG_RBGS_RED_ZERO_WGSL, bglMg1In);
    this.pMgRed = makeCompute('MgRed', MG_RBGS_RED_WGSL, bglMg2In);
    this.pMgBlack = makeCompute('MgBlack', MG_RBGS_BLACK_WGSL, bglMg2In);
    this.pMgCoarseL3 = makeCompute('MgCoarseL3', MG_COARSE_SOLVE_L3_WGSL, bglMg1In);
    this.pMgResidual = makeCompute('MgResidual', MG_RESIDUAL_WGSL, bglMg2In);
    this.pMgRestrict = makeCompute('MgRestrict', MG_RESTRICT_WGSL, bglMg1In);
    this.pMgProlong = makeCompute('MgProlong', MG_PROLONG_ADD_WGSL, bglMg2In);

    this.pFluidMacProject = makeCompute('FluidMacProject', FLUID_MAC_PROJECT_WGSL, bglFluidMacProject);
    this.pFluidMacToCenter = makeCompute('FluidMacToCenter', FLUID_MAC_TO_CENTER_WGSL, bglTex3DToStore3DF16);

    this.pParticleReset = makeCompute('ParticleReset', PARTICLE_RESET_WGSL, bglParticleReset);
    this.pParticleEmit = makeCompute('ParticleEmit', PARTICLE_EMIT_WGSL, bglParticleEmit);
    this.pParticleAdvectSplat = makeCompute('ParticleAdvectSplat', PARTICLE_ADVECT_SPLAT_WGSL, bglParticleAdvectSplat);

    this.pConcClear = makeCompute('ConcClear', CONC_CLEAR_WGSL, bglStore3DF16);
    this.pConcPredict = makeCompute('ConcPredict', CONC_PREDICT_WGSL, bglConcPredict);
    this.pConcResolve = makeCompute('ConcResolve', CONC_RESOLVE_WGSL, bglConcResolve);
    this.pLighting = makeCompute('LightingVol', LIGHTING_VOLUME_WGSL, bglLighting);
    this.pBrickRaw = makeCompute('PlumeBrickRaw', PLUME_OCCUPANCY_RAW_WGSL, bglBrickRaw);
    this.pBrickDilate = makeCompute('PlumeBrickDilate', PLUME_OCCUPANCY_DILATE_WGSL, bglMg1In);

    const renderMod = device.createShaderModule({ label: 'RenderShader', code: RENDER_WGSL });
    this.inspectCompilation('RenderShader', renderMod);
    const renderLayout = device.createPipelineLayout({ bindGroupLayouts: [bglRender] });
    this.pRender = device.createRenderPipeline({
      label: 'RenderPipeline',
      layout: renderLayout,
      vertex: { module: renderMod, entryPoint: 'vsMain' },
      fragment: {
        module: renderMod,
        entryPoint: 'fsMain',
        constants: { MACRO_INSPECT_MODE: 0 },
        targets: [{ format: 'rgba16float' }],
      },
      primitive: { topology: 'triangle-list' },
    });
    this.pRenderMacro = device.createRenderPipeline({
      label: 'RenderMacroPipeline',
      layout: renderLayout,
      vertex: { module: renderMod, entryPoint: 'vsMain' },
      fragment: {
        module: renderMod,
        entryPoint: 'fsMain',
        constants: { MACRO_INSPECT_MODE: 1 },
        targets: [{ format: 'rgba16float' }],
      },
      primitive: { topology: 'triangle-list' },
    });
    this.pRenderAccum = device.createRenderPipeline({
      label: 'RenderAccumPipeline',
      layout: renderLayout,
      vertex: { module: renderMod, entryPoint: 'vsMain' },
      fragment: {
        module: renderMod,
        entryPoint: 'fsMain',
        constants: { MACRO_INSPECT_MODE: 1 },
        targets: [
          {
            format: 'rgba16float',
            blend: {
              color: {
                srcFactor: 'constant',
                dstFactor: 'one-minus-constant',
                operation: 'add',
              },
              alpha: {
                srcFactor: 'one',
                dstFactor: 'zero',
                operation: 'add',
              },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    });

    const compMod = device.createShaderModule({ label: 'CompositeShader', code: COMPOSITE_WGSL });
    this.inspectCompilation('CompositeShader', compMod);
    const compLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bglComposite] });
    this.pCompositeSdr = device.createRenderPipeline({
      label: 'CompositePipelineSDR',
      layout: compLayout,
      vertex: { module: compMod, entryPoint: 'vsMain' },
      fragment: {
        module: compMod,
        entryPoint: 'fsMain',
        constants: { HDR_EXTENDED: 0 },
        targets: [{ format: preferredSdrFormat }],
      },
      primitive: { topology: 'triangle-list' },
    });
    this.pCompositeHdr = device.createRenderPipeline({
      label: 'CompositePipelineHDR',
      layout: compLayout,
      vertex: { module: compMod, entryPoint: 'vsMain' },
      fragment: {
        module: compMod,
        entryPoint: 'fsMain',
        constants: { HDR_EXTENDED: 1 },
        targets: [{ format: 'rgba16float' }],
      },
      primitive: { topology: 'triangle-list' },
    });
    this.pComposite = isExtendedHdr ? this.pCompositeHdr : this.pCompositeSdr;

    // Views & Pre-Allocated BindGroup Matrices
    const uBuf = { buffer: this.uniformBuffer };
    const pBuf = { buffer: this.particlesBuffer };
    const sIntactBuf = { buffer: this.splatIntactBuffer };
    const sLysedBuf = { buffer: this.splatLysedBuffer };

    const vWave = [this.texWave[0].createView(), this.texWave[1].createView()] as const;
    const vWaveSpecAux = this.texWaveSpecAux.createView();
    const vWaveSpecA = this.texWaveSpecA.createView();
    const vWaveSpecB = this.texWaveSpecB.createView();

    const vVel = [this.texVel[0].createView(), this.texVel[1].createView()] as const;
    const vVelCenter = this.texVelCenter.createView();
    const vCurl = this.texCurl.createView();

    const vPress = [this.texPress[0].createView(), this.texPress[1].createView()] as const;
    const vRhs0 = this.texRhs0.createView();
    const vRes0 = this.texRes0.createView();

    const vMgErr1 = [this.texMgErr1[0].createView(), this.texMgErr1[1].createView()] as const;
    const vMgRhs1 = this.texMgRhs1.createView();
    const vMgRes1 = this.texMgRes1.createView();

    const vMgErr2 = [this.texMgErr2[0].createView(), this.texMgErr2[1].createView()] as const;
    const vMgRhs2 = this.texMgRhs2.createView();
    const vMgRes2 = this.texMgRes2.createView();

    const vMgErr3 = [this.texMgErr3[0].createView(), this.texMgErr3[1].createView()] as const;
    const vMgRhs3 = this.texMgRhs3.createView();

    const vConc = [this.texConc[0].createView(), this.texConc[1].createView()] as const;
    const vConcPred = this.texConcPred.createView();
    const vLight = this.texLight.createView();
    const vBrickRaw = this.texBrickRaw.createView();
    const vBrickDilated = this.texBrickDilated.createView();
    const vSpectralLut = this.texSpectralLut.createView();
    this.viewWave = [vWave[0], vWave[1]];
    this.viewSpectralLut = vSpectralLut;

    // Wave BindGroups
    this.bgWaveClear = [
      device.createBindGroup({ layout: bglStore2DF16, entries: [{ binding: 0, resource: uBuf }, { binding: 1, resource: vWave[0] }] }),
      device.createBindGroup({ layout: bglStore2DF16, entries: [{ binding: 0, resource: uBuf }, { binding: 1, resource: vWave[1] }] }),
    ];

    this.bgWaveClearSpec = [
      device.createBindGroup({ layout: bglStore2DF32, entries: [{ binding: 0, resource: uBuf }, { binding: 1, resource: vWaveSpecAux }] }),
      device.createBindGroup({ layout: bglStore2DF32, entries: [{ binding: 0, resource: uBuf }, { binding: 1, resource: vWaveSpecA }] }),
      device.createBindGroup({ layout: bglStore2DF32, entries: [{ binding: 0, resource: uBuf }, { binding: 1, resource: vWaveSpecB }] }),
    ];

    const makeWaveForceBG = (ws: 0 | 1, vs: 0 | 1) =>
      device.createBindGroup({
        layout: bglWaveSpatialForce,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: vWave[ws] },
          { binding: 2, resource: vVel[vs] },
          { binding: 3, resource: this.linearSampler },
          { binding: 4, resource: vWaveSpecAux },
        ],
      });
    this.bgWaveSpatialForce = [
      [makeWaveForceBG(0, 0), makeWaveForceBG(0, 1)],
      [makeWaveForceBG(1, 0), makeWaveForceBG(1, 1)],
    ];

    const makeSpecPassBG = (inView: GPUTextureView, outView: GPUTextureView) =>
      device.createBindGroup({
        layout: bglWaveSpecPass,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: inView },
          { binding: 2, resource: outView },
        ],
      });

    this.bgWaveFftRow = makeSpecPassBG(vWaveSpecAux, vWaveSpecA);
    this.bgWaveFftCol = makeSpecPassBG(vWaveSpecA, vWaveSpecB);
    this.bgWaveSpectralProp = makeSpecPassBG(vWaveSpecB, vWaveSpecA);
    this.bgWaveIfftCol = makeSpecPassBG(vWaveSpecA, vWaveSpecB);
    this.bgWaveIfftRow = makeSpecPassBG(vWaveSpecB, vWaveSpecA);

    this.bgWaveFinalize = [
      device.createBindGroup({
        layout: bglWaveFinalize,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: vWaveSpecA },
          { binding: 2, resource: vWaveSpecAux },
          { binding: 3, resource: vWave[1] },
        ],
      }),
      device.createBindGroup({
        layout: bglWaveFinalize,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: vWaveSpecA },
          { binding: 2, resource: vWaveSpecAux },
          { binding: 3, resource: vWave[0] },
        ],
      }),
    ];

    // Fluid MAC & Casson BindGroups
    this.bgFluidInitVel = [
      device.createBindGroup({ layout: bglStore3DF16, entries: [{ binding: 0, resource: uBuf }, { binding: 1, resource: vVel[0] }] }),
      device.createBindGroup({ layout: bglStore3DF16, entries: [{ binding: 0, resource: uBuf }, { binding: 1, resource: vVel[1] }] }),
      device.createBindGroup({ layout: bglStore3DF16, entries: [{ binding: 0, resource: uBuf }, { binding: 1, resource: vVelCenter }] }),
      device.createBindGroup({ layout: bglStore3DF16, entries: [{ binding: 0, resource: uBuf }, { binding: 1, resource: vCurl }] }),
    ];

    this.bgFluidClearR32 = [
      device.createBindGroup({ layout: bglStore3DR32, entries: [{ binding: 0, resource: uBuf }, { binding: 1, resource: vPress[0] }] }),
      device.createBindGroup({ layout: bglStore3DR32, entries: [{ binding: 0, resource: uBuf }, { binding: 1, resource: vPress[1] }] }),
      device.createBindGroup({ layout: bglStore3DR32, entries: [{ binding: 0, resource: uBuf }, { binding: 1, resource: vRhs0 }] }),
      device.createBindGroup({ layout: bglStore3DR32, entries: [{ binding: 0, resource: uBuf }, { binding: 1, resource: vRes0 }] }),
    ];

    const makeAdvectMacBG = (vs: 0 | 1, ws: 0 | 1) =>
      device.createBindGroup({
        layout: bglFluidAdvectMac,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: vVel[vs] },
          { binding: 2, resource: vVelCenter },
          { binding: 3, resource: vWave[ws] },
          { binding: 4, resource: this.linearSampler },
          { binding: 5, resource: vVel[1 - vs] },
        ],
      });
    this.bgFluidAdvectMac = [
      [makeAdvectMacBG(0, 0), makeAdvectMacBG(0, 1)],
      [makeAdvectMacBG(1, 0), makeAdvectMacBG(1, 1)],
    ];

    this.bgFluidCurl = device.createBindGroup({
      layout: bglTex3DToStore3DF16,
      entries: [
        { binding: 0, resource: uBuf },
        { binding: 1, resource: vVelCenter },
        { binding: 2, resource: vCurl },
      ],
    });

    const makeCassonForcesBG = (vs: 0 | 1, cs: 0 | 1) =>
      device.createBindGroup({
        layout: bglFluidCassonForces,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: vVel[vs] },
          { binding: 2, resource: vCurl },
          { binding: 3, resource: vConc[cs] },
          { binding: 4, resource: this.linearSampler },
          { binding: 5, resource: vVel[1 - vs] },
        ],
      });
    this.bgFluidCassonForces = [
      [makeCassonForcesBG(0, 0), makeCassonForcesBG(0, 1)],
      [makeCassonForcesBG(1, 0), makeCassonForcesBG(1, 1)],
    ];

    const makeMacRhsBG = (vs: 0 | 1, ws: 0 | 1) =>
      device.createBindGroup({
        layout: bglFluidMacRhs,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: vVel[vs] },
          { binding: 2, resource: vWave[ws] },
          { binding: 3, resource: this.linearSampler },
          { binding: 4, resource: vRhs0 },
        ],
      });
    this.bgFluidMacRhs = [
      [makeMacRhsBG(0, 0), makeMacRhsBG(0, 1)],
      [makeMacRhsBG(1, 0), makeMacRhsBG(1, 1)],
    ];

    // Multigrid BindGroup Helpers
    const makeMg1 = (in1: GPUTextureView, out: GPUTextureView) =>
      device.createBindGroup({
        layout: bglMg1In,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: in1 },
          { binding: 2, resource: out },
        ],
      });

    const makeMg2 = (in1: GPUTextureView, in2: GPUTextureView, out: GPUTextureView) =>
      device.createBindGroup({
        layout: bglMg2In,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: in1 },
          { binding: 2, resource: in2 },
          { binding: 3, resource: out },
        ],
      });

    // Level 0
    this.bgMgL0Red = [makeMg2(vPress[0], vRhs0, vPress[1]), makeMg2(vPress[1], vRhs0, vPress[0])];
    this.bgMgL0Black = [makeMg2(vPress[0], vRhs0, vPress[1]), makeMg2(vPress[1], vRhs0, vPress[0])];
    this.bgMgL0Res = [makeMg2(vPress[0], vRhs0, vRes0), makeMg2(vPress[1], vRhs0, vRes0)];
    this.bgMgRestrict0to1 = makeMg1(vRes0, vMgRhs1);

    // Level 1
    this.bgMgL1RedZero = makeMg1(vMgRhs1, vMgErr1[1]);
    this.bgMgL1Red0to1 = makeMg2(vMgErr1[0], vMgRhs1, vMgErr1[1]);
    this.bgMgL1Red1to0 = makeMg2(vMgErr1[1], vMgRhs1, vMgErr1[0]);
    this.bgMgL1Black1to0 = makeMg2(vMgErr1[1], vMgRhs1, vMgErr1[0]);
    this.bgMgL1Black0to1 = makeMg2(vMgErr1[0], vMgRhs1, vMgErr1[1]);
    this.bgMgL1Res = makeMg2(vMgErr1[0], vMgRhs1, vMgRes1);
    this.bgMgRestrict1to2 = makeMg1(vMgRes1, vMgRhs2);

    // Level 2
    this.bgMgL2RedZero = makeMg1(vMgRhs2, vMgErr2[1]);
    this.bgMgL2Red0to1 = makeMg2(vMgErr2[0], vMgRhs2, vMgErr2[1]);
    this.bgMgL2Red1to0 = makeMg2(vMgErr2[1], vMgRhs2, vMgErr2[0]);
    this.bgMgL2Black1to0 = makeMg2(vMgErr2[1], vMgRhs2, vMgErr2[0]);
    this.bgMgL2Black0to1 = makeMg2(vMgErr2[0], vMgRhs2, vMgErr2[1]);
    this.bgMgL2Res = makeMg2(vMgErr2[0], vMgRhs2, vMgRes2);
    this.bgMgRestrict2to3 = makeMg1(vMgRes2, vMgRhs3);

    // Level 3 (single shared-memory workgroup coarse solve -> MgErr3[0])
    this.bgMgL3Coarse = makeMg1(vMgRhs3, vMgErr3[0]);

    // Prolongation
    this.bgMgProlong3to2 = makeMg2(vMgErr2[0], vMgErr3[0], vMgErr2[1]);
    this.bgMgProlong2to1 = makeMg2(vMgErr1[0], vMgErr2[1], vMgErr1[1]);
    this.bgMgProlong1to0 = [
      makeMg2(vPress[0], vMgErr1[1], vPress[1]),
      makeMg2(vPress[1], vMgErr1[1], vPress[0]),
    ];

    const makeMacProjectBG = (vs: 0 | 1, ps: 0 | 1, ws: 0 | 1) =>
      device.createBindGroup({
        layout: bglFluidMacProject,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: vVel[vs] },
          { binding: 2, resource: vPress[ps] },
          { binding: 3, resource: vWave[ws] },
          { binding: 4, resource: this.linearSampler },
          { binding: 5, resource: vVel[1 - vs] },
        ],
      });
    this.bgFluidMacProject = [
      [
        [makeMacProjectBG(0, 0, 0), makeMacProjectBG(0, 0, 1)],
        [makeMacProjectBG(0, 1, 0), makeMacProjectBG(0, 1, 1)],
      ],
      [
        [makeMacProjectBG(1, 0, 0), makeMacProjectBG(1, 0, 1)],
        [makeMacProjectBG(1, 1, 0), makeMacProjectBG(1, 1, 1)],
      ],
    ];

    this.bgFluidMacToCenter = [
      device.createBindGroup({
        layout: bglTex3DToStore3DF16,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: vVel[0] },
          { binding: 2, resource: vVelCenter },
        ],
      }),
      device.createBindGroup({
        layout: bglTex3DToStore3DF16,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: vVel[1] },
          { binding: 2, resource: vVelCenter },
        ],
      }),
    ];

    // Lagrangian RBC Parcel BindGroups
    this.bgParticleReset = device.createBindGroup({
      layout: bglParticleReset,
      entries: [
        { binding: 0, resource: uBuf },
        { binding: 1, resource: pBuf },
      ],
    });

    this.bgParticleEmit = [
      device.createBindGroup({
        layout: bglParticleEmit,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: vWave[0] },
          { binding: 2, resource: this.linearSampler },
          { binding: 3, resource: pBuf },
          { binding: 4, resource: sLysedBuf },
        ],
      }),
      device.createBindGroup({
        layout: bglParticleEmit,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: vWave[1] },
          { binding: 2, resource: this.linearSampler },
          { binding: 3, resource: pBuf },
          { binding: 4, resource: sLysedBuf },
        ],
      }),
    ];

    const makeParticleAdvectBG = (cs: 0 | 1) =>
      device.createBindGroup({
        layout: bglParticleAdvectSplat,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: vVelCenter },
          { binding: 2, resource: vConc[cs] },
          { binding: 3, resource: this.linearSampler },
          { binding: 4, resource: pBuf },
          { binding: 5, resource: sIntactBuf },
          { binding: 6, resource: sLysedBuf },
        ],
      });
    this.bgParticleAdvectSplat = [makeParticleAdvectBG(0), makeParticleAdvectBG(1)];

    // Eulerian Lysed-Hb & Composite Concentration BindGroups
    this.bgConcClear = [
      device.createBindGroup({ layout: bglStore3DF16, entries: [{ binding: 0, resource: uBuf }, { binding: 1, resource: vConc[0] }] }),
      device.createBindGroup({ layout: bglStore3DF16, entries: [{ binding: 0, resource: uBuf }, { binding: 1, resource: vConc[1] }] }),
      device.createBindGroup({ layout: bglStore3DF16, entries: [{ binding: 0, resource: uBuf }, { binding: 1, resource: vConcPred }] }),
    ];

    const makeConcPredictBG = (cs: 0 | 1) =>
      device.createBindGroup({
        layout: bglConcPredict,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: vConc[cs] },
          { binding: 2, resource: vVelCenter },
          { binding: 3, resource: this.linearSampler },
          { binding: 4, resource: vConcPred },
        ],
      });
    this.bgConcPredict = [makeConcPredictBG(0), makeConcPredictBG(1)];

    const makeConcResolveBG = (cs: 0 | 1) =>
      device.createBindGroup({
        layout: bglConcResolve,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: vConc[cs] },
          { binding: 2, resource: vConcPred },
          { binding: 3, resource: vVelCenter },
          { binding: 4, resource: this.linearSampler },
          { binding: 5, resource: sIntactBuf },
          { binding: 6, resource: sLysedBuf },
          { binding: 7, resource: vConc[1 - cs] },
        ],
      });
    this.bgConcResolve = [makeConcResolveBG(0), makeConcResolveBG(1)];

    const makeLightBG = (cs: 0 | 1, ws: 0 | 1) =>
      device.createBindGroup({
        layout: bglLighting,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: vConc[cs] },
          { binding: 2, resource: vWave[ws] },
          { binding: 3, resource: this.linearSampler },
          { binding: 4, resource: vLight },
        ],
      });
    this.bgLighting = [
      [makeLightBG(0, 0), makeLightBG(0, 1)],
      [makeLightBG(1, 0), makeLightBG(1, 1)],
    ];

    this.bgBrickRaw = device.createBindGroup({
      layout: bglBrickRaw,
      entries: [
        { binding: 0, resource: uBuf },
        { binding: 1, resource: vLight },
        { binding: 2, resource: vBrickRaw },
      ],
    });
    this.bgBrickDilate = makeMg1(vBrickRaw, vBrickDilated);

    const makeRenderBG = (cs: 0 | 1, ws: 0 | 1) =>
      device.createBindGroup({
        layout: bglRender,
        entries: [
          { binding: 0, resource: uBuf },
          { binding: 1, resource: vConc[cs] },
          { binding: 2, resource: vLight },
          { binding: 3, resource: vWave[ws] },
          { binding: 4, resource: vSpectralLut },
          { binding: 5, resource: this.linearSampler },
          { binding: 6, resource: vBrickDilated },
        ],
      });
    this.bgRender = [
      [makeRenderBG(0, 0), makeRenderBG(0, 1)],
      [makeRenderBG(1, 0), makeRenderBG(1, 1)],
    ];

    const initSwapW = Math.max(2, canvas.width || 1280);
    const initSwapH = Math.max(2, canvas.height || 800);
    const initVol = this.createVolRenderResources(initSwapW, initSwapH);
    this.volSwapW = initSwapW;
    this.volSwapH = initSwapH;
    this.texVolRender = initVol.tex;
    this.viewVolRender = initVol.view;
    this.bgComposite = initVol.bg;
  }

  private createVolRenderResources(swapW: number, swapH: number): {
    tex: GPUTexture;
    view: GPUTextureView;
    bg: [GPUBindGroup, GPUBindGroup];
  } {
    const scale = Math.min(1.0, Math.sqrt(VOL_RENDER_MAX_PIXELS / Math.max(1, swapW * swapH)));
    const volW = scale >= 0.999 ? swapW : Math.max(2, Math.round(swapW * scale) & ~1);
    const volH = scale >= 0.999 ? swapH : Math.max(2, Math.round(swapH * scale) & ~1);
    const tex = this.device.createTexture({
      label: 'VolRenderHDR_F16',
      size: [volW, volH, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const view = tex.createView();
    const makeCompBG = (ws: 0 | 1) =>
      this.device.createBindGroup({
        layout: this.bglComposite,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: view },
          { binding: 2, resource: this.linearSampler },
          { binding: 3, resource: this.viewWave[ws] },
          { binding: 4, resource: this.viewSpectralLut },
        ],
      });
    return { tex, view, bg: [makeCompBG(0), makeCompBG(1)] };
  }

  private ensureVolRenderTarget(): void {
    const cw = Math.max(2, this.canvas.width || 1280);
    const ch = Math.max(2, this.canvas.height || 800);
    if (cw === this.volSwapW && ch === this.volSwapH) {
      return;
    }
    this.texVolRender.destroy();
    const next = this.createVolRenderResources(cw, ch);
    this.volSwapW = cw;
    this.volSwapH = ch;
    this.texVolRender = next.tex;
    this.viewVolRender = next.view;
    this.bgComposite = next.bg;
  }

  private inspectCompilation(label: string, module: GPUShaderModule): void {
    module.getCompilationInfo().then((info) => {
      for (const msg of info.messages) {
        if (msg.type === 'error') {
          const formatted = `[${label}] L${msg.lineNum}:${msg.linePos} ${msg.message}`;
          this.validationErrors.push(formatted);
          console.error(formatted);
        }
      }
    });
  }

  private async verifyCleanInit(): Promise<void> {
    const internalErr = await this.device.popErrorScope();
    const valErr = await this.device.popErrorScope();
    if (internalErr) {
      this.validationErrors.push(internalErr.message);
      throw new Error(`WebGPU internal pipeline error: ${internalErr.message}`);
    }
    if (valErr) {
      this.validationErrors.push(valErr.message);
      throw new Error(`WebGPU validation error: ${valErr.message}`);
    }
  }

  public getCurrentKinematics(): MultiDropSnapshot {
    return this.multiDrops.getSnapshot();
  }

  public setCameraPreset(id: CameraPreset['id'], instant = false): void {
    this.lockedSubjectId = null;
    this.pausedAccumCount = 0;
    this.camera.applyPreset(CAMERA_PRESETS[id], instant);
    if (id === 'hero') {
      this.camera.syncHeroFollow(this.simTime, instant);
    }
  }

  public setLightingMode(darkChiaroscuro: boolean): void {
    this.config.lightingMode = darkChiaroscuro ? 1.0 : 0.0;
    this.pausedAccumCount = 0;
  }

  /**
   * Switches the canvas swapchain between Apple Silicon XDR linear extended-sRGB (`rgba16float` +
   * `toneMapping: { mode: 'extended' }`) and standard SDR (`bgra8unorm`/`rgba8unorm`) with zero
   * shader recompilation via pre-specialized `HDR_EXTENDED` composite pipelines.
   */
  public setExtendedHdr(enabled: boolean): void {
    if (this.destroyed || this.isDeviceLost) return;
    if (enabled === this.isExtendedHdr) return;
    this.pausedAccumCount = 0;
    if (enabled) {
      try {
        this.context.configure({
          device: this.device,
          format: 'rgba16float',
          alphaMode: 'opaque',
          toneMapping: { mode: 'extended' },
        });
        this.format = 'rgba16float';
        this.isExtendedHdr = true;
        this.pComposite = this.pCompositeHdr;
        return;
      } catch {
        // Fallback to standard SDR swapchain if extended HDR is rejected
      }
    }
    try {
      this.context.configure({
        device: this.device,
        format: this.preferredSdrFormat,
        alphaMode: 'opaque',
        toneMapping: { mode: 'standard' },
      });
    } catch {
      this.context.configure({
        device: this.device,
        format: this.preferredSdrFormat,
        alphaMode: 'opaque',
      });
    }
    this.format = this.preferredSdrFormat;
    this.isExtendedHdr = false;
    this.pComposite = this.pCompositeSdr;
  }

  /**
   * Compact telemetry and deterministic control surface for QA / Workbench tooling.
   */
  public getWorkbenchAdapter() {
    return {
      getStatus: () => ({
        simTime: this.simTime,
        frameIndex: this.frameIndex,
        paused: this.paused,
        pausedAccumCount: this.pausedAccumCount,
        lockedSubjectId: this.lockedSubjectId,
        timeScale: this.timeScale,
        format: this.format,
        isExtendedHdr: this.isExtendedHdr,
        supportsTimestamps: this.supportsTimestamps,
        canvasSize: [this.canvas.width, this.canvas.height] as [number, number],
        gpuPassTimingsMs: { ...this.gpuPassTimingsMs },
        validationErrorCount: this.validationErrors.length,
      }),
      getPresets: () => ['hero', 'crown', 'abyss'] as const,
      applyPreset: (id: CameraPreset['id'], instant = true) => this.setCameraPreset(id, instant),
      setPaused: (paused: boolean) => {
        this.paused = paused;
      },
      stepFrames: (count = 1) => this.stepPausedSubsteps(count),
      focusSubject: () => this.cycleFocusSubject(),
      seekSimTime: (t: number, seed = 1.0) => this.seekToSimTime(t, seed),
      setExtendedHdr: (enabled: boolean) => this.setExtendedHdr(enabled),
      getGpuPassTimings: () => ({ ...this.gpuPassTimingsMs }),
    };
  }

  private nextSeed(): number {
    this.dropSeed = ((this.dropSeed * 2.7182818 + 1.37) % 100.0) + 0.5;
    return this.dropSeed;
  }

  /**
   * Raycasts normalized device coordinates ([-1, 1] x [-1, 1]) onto y = WATER_SURFACE_Y.
   */
  public raycastCanvasNdcToMeniscus(ndcX: number, ndcY: number): [number, number] {
    const aspect = (this.canvas.width || 1) / (this.canvas.height || 1);
    this.camera.updateMatrices(aspect);
    return raycastMeniscusXZ(
      this.camera.invViewProjMatrix,
      this.camera.position,
      this.camera.target,
      ndcX,
      ndcY
    );
  }

  /**
   * Releases an aerial blood drop above the raycast meniscus coordinate (ndcX, ndcY).
   */
  public spawnDropAtNdc(ndcX: number, ndcY: number, radius = INITIAL_DROP_RADIUS): DropSlotState {
    const xz = this.raycastCanvasNdcToMeniscus(ndcX, ndcY);
    this.dropOffsetXZ[0] = xz[0];
    this.dropOffsetXZ[1] = xz[1];
    const slot = this.multiDrops.spawnAerialDrop(xz, radius, this.nextSeed());
    this.paused = false;
    return slot;
  }

  /**
   * Suspends a swelling capillary pendant drop above the raycast meniscus coordinate (ndcX, ndcY).
   */
  public beginPendantDropAtNdc(ndcX: number, ndcY: number): DropSlotState {
    const xz = this.raycastCanvasNdcToMeniscus(ndcX, ndcY);
    this.dropOffsetXZ[0] = xz[0];
    this.dropOffsetXZ[1] = xz[1];
    return this.multiDrops.beginPendantDrop(xz, this.nextSeed());
  }

  /**
   * Squeezes and detaches a quick-clicked drop briskly (~0.15s) from the glass Pasteur pipette tip
   * at PIPETTE_TIP_Y = +0.55 above the raycast meniscus coordinate (ndcX, ndcY).
   */
  public triggerQuickPipetteDropAtNdc(
    ndcX: number,
    ndcY: number,
    radius?: number
  ): DropSlotState | null {
    if (this.destroyed) return null;
    const xz = this.raycastCanvasNdcToMeniscus(ndcX, ndcY);
    return this.triggerQuickPipetteDrop(xz, radius);
  }

  /**
   * Squeezes and detaches an additional drop briskly (~0.15s) from the glass Pasteur pipette tip
   * at PIPETTE_TIP_Y = +0.55 above offsetXZ without popping a bare sphere into mid-air.
   */
  public triggerQuickPipetteDrop(
    offsetXZ: [number, number] = [0.0, 0.0],
    radius?: number
  ): DropSlotState | null {
    if (this.destroyed) return null;
    const seed = this.nextSeed();
    const frac = (Math.sin(seed * 12.9898) * 43758.5453) % 1.0;
    const unitFrac = frac < 0.0 ? frac + 1.0 : frac;
    const pipetteRadius = radius ?? 0.053 + unitFrac * 0.010;
    const slot = this.multiDrops.beginQuickPipetteDrop(offsetXZ, pipetteRadius, seed);
    this.dropOffsetXZ[0] = slot.center[0];
    this.dropOffsetXZ[1] = slot.center[2];
    this.paused = false;
    return slot;
  }

  public hasAutoSqueezePendantDrop(): boolean {
    return this.multiDrops.hasAutoSqueezePendantDrop();
  }

  public updatePendantDrop(dtRealSec: number): boolean {
    if (this.destroyed) return false;
    const pinchedOff = this.multiDrops.updatePendantDrop(dtRealSec);
    if (pinchedOff) {
      this.paused = false;
    }
    return pinchedOff;
  }

  public releasePendantDrop(): DropSlotState | null {
    if (this.destroyed) return null;
    const slot = this.multiDrops.releasePendantDrop();
    if (slot) {
      this.paused = false;
    }
    return slot;
  }

  public cancelPendantDrop(): void {
    if (this.destroyed) return;
    this.multiDrops.cancelPendantDrop();
  }

  public hasActivePendantDrop(): boolean {
    return this.multiDrops.hasActivePendantDrop();
  }

  /**
   * Resets the water tank to calm water, clears all Lagrangian RBC parcels, and releases a new hero drop.
   */
  public resetSimulation(seed?: number, offsetXZ: [number, number] = [0.0, 0.0]): void {
    if (this.destroyed) return;
    this.simTime = 0.0;
    this.simAccumulator = 0.0;
    this.waveState = 0;
    this.velState = 0;
    this.pressState = 0;
    this.concState = 0;
    this.spawnCursor = 0;
    this.highWaterParticles = 0;
    this.parcelFractionCarry = 0.0;

    if (seed !== undefined) {
      this.dropSeed = Math.max(0.5, seed);
    } else {
      this.dropSeed = ((this.dropSeed * 1.6180339 + 1.0) % 100.0) + 0.5;
    }

    this.multiDrops.reset(this.dropSeed, offsetXZ);
    this.dropOffsetXZ[0] = offsetXZ[0];
    this.dropOffsetXZ[1] = offsetXZ[1];
    this.heroTopsideCue = 0.0;
    this.pendingManualSubsteps = 0;
    this.lockedSubjectId = null;
    this.pausedAccumCount = 0;
    this.lastPausedSimTime = -1.0;
    this.camera.syncHeroFollow(0.0, true, 0.0);

    this.uploadSnapshotUniforms(FIXED_SIM_DT, 0, 0);

    const enc = this.device.createCommandEncoder({ label: 'ResetSimEncoder' });
    enc.clearBuffer(this.splatIntactBuffer);
    enc.clearBuffer(this.splatLysedBuffer);

    const pass = enc.beginComputePass({ label: 'ResetSimCompute' });

    const wgWave = WAVE_GRID_SIZE / 8;
    for (let i = 0; i < 2; i++) {
      pass.setPipeline(this.pWaveClear);
      pass.setBindGroup(0, this.bgWaveClear[i]);
      pass.dispatchWorkgroups(wgWave, wgWave, 1);
    }
    for (let i = 0; i < 3; i++) {
      pass.setPipeline(this.pWaveClearSpec);
      pass.setBindGroup(0, this.bgWaveClearSpec[i]);
      pass.dispatchWorkgroups(wgWave, wgWave, 1);
    }

    const wgVelX = VEL_GRID_X / 8;
    const wgVelY = VEL_GRID_Y / 4;
    const wgVelZ = VEL_GRID_Z / 4;

    for (let i = 0; i < 4; i++) {
      pass.setPipeline(this.pFluidInitVel);
      pass.setBindGroup(0, this.bgFluidInitVel[i]);
      pass.dispatchWorkgroups(wgVelX, wgVelY, wgVelZ);
    }

    for (let i = 0; i < 4; i++) {
      pass.setPipeline(this.pFluidClearR32);
      pass.setBindGroup(0, this.bgFluidClearR32[i]);
      pass.dispatchWorkgroups(wgVelX, wgVelY, wgVelZ);
    }

    pass.setPipeline(this.pParticleReset);
    pass.setBindGroup(0, this.bgParticleReset);
    pass.dispatchWorkgroups(PARTICLE_COUNT / PARTICLE_WORKGROUP_SIZE, 1, 1);

    const wgConcX = CONC_GRID_X / 8;
    const wgConcY = CONC_GRID_Y / 4;
    const wgConcZ = CONC_GRID_Z / 4;

    for (let i = 0; i < 3; i++) {
      pass.setPipeline(this.pConcClear);
      pass.setBindGroup(0, this.bgConcClear[i]);
      pass.dispatchWorkgroups(wgConcX, wgConcY, wgConcZ);
    }

    this.encodeLightingPasses(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Releases an additional concurrent blood drop into the active tank without resetting shot time,
   * yanking the Follow camera, or cancelling any in-flight or coalescing drops.
   */
  public triggerAdditionalDrop(
    offsetXZ: [number, number] = [0.0, 0.0],
    radius?: number
  ): DropSlotState | null {
    if (this.destroyed) return null;
    const seed = this.nextSeed();
    const frac = (Math.sin(seed * 12.9898) * 43758.5453) % 1.0;
    const unitFrac = frac < 0.0 ? frac + 1.0 : frac;
    const pipetteRadius = radius ?? 0.053 + unitFrac * 0.010;
    const slot = this.multiDrops.spawnAerialDrop(offsetXZ, pipetteRadius, seed);
    this.dropOffsetXZ[0] = slot.center[0];
    this.dropOffsetXZ[1] = slot.center[2];
    this.paused = false;
    return slot;
  }

  private packSnapshotUniforms(
    drops: MultiDropSnapshot,
    dt: number,
    spawnCursor: number,
    spawnCount: number
  ): Float32Array {
    const w = this.canvas.width || 1;
    const h = this.canvas.height || 1;
    return this.uniformPacker.pack({
      invViewProj: this.camera.invViewProjMatrix,
      viewProj: this.camera.viewProjMatrix,
      cameraPos: this.camera.position,
      cameraTarget: this.camera.target,
      viewportWidth: w,
      viewportHeight: h,
      dt,
      frameIndex: this.frameIndex,
      globalSeed: this.dropSeed,
      drops,
      spawnCursor,
      spawnCount,
      config: this.config,
    });
  }

  private uploadSnapshotUniforms(
    dt: number,
    spawnCursor: number,
    spawnCount: number
  ): MultiDropSnapshot {
    const snap = this.multiDrops.getSnapshot();
    const packed = this.packSnapshotUniforms(snap, dt, spawnCursor, spawnCount);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, packed.buffer, 0, SIM_UNIFORM_BYTES);
    return snap;
  }

  private advanceSubstepUniforms(dt: number): {
    packed: Float32Array;
    drops: MultiDropSnapshot;
    spawnCount: number;
  } {
    const snap = this.multiDrops.step(dt);
    this.simTime = snap.simTime;
    if (this.paused && this.lockedSubjectId !== null) {
      this.syncLockedSubjectTarget(true);
    }

    const exactParcels =
      (snap.totalStepDrainedVolume / REFERENCE_DROP_VOLUME) * PARCELS_PER_REF_DROP +
      this.parcelFractionCarry;
    const spawnCount = Math.min(8192, Math.max(0, Math.floor(exactParcels)));
    this.parcelFractionCarry = Math.max(0.0, exactParcels - spawnCount);

    const cursorBefore = this.spawnCursor;
    this.spawnCursor = (this.spawnCursor + spawnCount) % PARTICLE_COUNT;
    this.highWaterParticles = Math.min(PARTICLE_COUNT, this.highWaterParticles + spawnCount);

    const packed = this.packSnapshotUniforms(snap, dt, cursorBefore, spawnCount);
    return { packed, drops: snap, spawnCount };
  }

  private acquireSwapchainView(): GPUTextureView | null {
    if (this.destroyed || this.isDeviceLost) return null;
    try {
      return this.context.getCurrentTexture().createView();
    } catch {
      return null;
    }
  }

  private encodePhysicsStep(
    enc: GPUCommandEncoder,
    substepSpawnCount: number,
    recordTimestamps = false
  ): void {
    const passA = enc.beginComputePass({
      label: 'PhysicsSubstepPreSplat',
      timestampWrites: recordTimestamps ? this.tsPass0 : undefined,
    });

    const wgWave = WAVE_GRID_SIZE / 8; // 32
    const wgVelX = VEL_GRID_X / 8;     // 12
    const wgVelY = VEL_GRID_Y / 4;     // 32
    const wgVelZ = VEL_GRID_Z / 4;     // 12

    const wgMg0X = VEL_GRID_X / 8;     // 12 (@workgroup_size(8, 4, 4))
    const wgMg0Y = VEL_GRID_Y / 4;     // 32
    const wgMg0Z = VEL_GRID_Z / 4;     // 24

    const wgMg1X = MG_L1_X / 8;        // 6
    const wgMg1Y = MG_L1_Y / 4;        // 16
    const wgMg1Z = MG_L1_Z / 4;        // 12

    const wgMg2X = MG_L2_X / 8;        // 3
    const wgMg2Y = MG_L2_Y / 4;        // 8
    const wgMg2Z = MG_L2_Z / 4;        // 6

    const wgMg3X = Math.ceil(MG_L3_X / 8); // 2
    const wgMg3Y = MG_L3_Y / 4;            // 4
    const wgMg3Z = MG_L3_Z / 4;            // 3

    // 1. Exact Spectral Gravity-Capillary Propagator + Marangoni Surface Stress
    passA.setPipeline(this.pWaveSpatialForce);
    passA.setBindGroup(0, this.bgWaveSpatialForce[this.waveState][this.velState]);
    passA.dispatchWorkgroups(wgWave, wgWave, 1);

    passA.setPipeline(this.pWaveFftRow);
    passA.setBindGroup(0, this.bgWaveFftRow);
    passA.dispatchWorkgroups(WAVE_GRID_SIZE, 1, 1);

    passA.setPipeline(this.pWaveFftCol);
    passA.setBindGroup(0, this.bgWaveFftCol);
    passA.dispatchWorkgroups(WAVE_GRID_SIZE, 1, 1);

    passA.setPipeline(this.pWaveSpectralProp);
    passA.setBindGroup(0, this.bgWaveSpectralProp);
    passA.dispatchWorkgroups(wgWave, wgWave, 1);

    passA.setPipeline(this.pWaveIfftCol);
    passA.setBindGroup(0, this.bgWaveIfftCol);
    passA.dispatchWorkgroups(WAVE_GRID_SIZE, 1, 1);

    passA.setPipeline(this.pWaveIfftRow);
    passA.setBindGroup(0, this.bgWaveIfftRow);
    passA.dispatchWorkgroups(WAVE_GRID_SIZE, 1, 1);

    passA.setPipeline(this.pWaveFinalize);
    passA.setBindGroup(0, this.bgWaveFinalize[this.waveState]);
    passA.dispatchWorkgroups(wgWave, wgWave, 1);
    this.waveState = (1 - this.waveState) as 0 | 1;

    // 2. Face-Staggered MAC Velocity Advection + Wetting-Neck Momentum Coupling
    passA.setPipeline(this.pFluidAdvectMac);
    passA.setBindGroup(0, this.bgFluidAdvectMac[this.velState][this.waveState]);
    passA.dispatchWorkgroups(wgVelX, wgVelY, wgVelZ);
    this.velState = (1 - this.velState) as 0 | 1;

    // Reconstruct cell-centered velocity for curl evaluation
    passA.setPipeline(this.pFluidMacToCenter);
    passA.setBindGroup(0, this.bgFluidMacToCenter[this.velState]);
    passA.dispatchWorkgroups(wgVelX, wgVelY, wgVelZ);

    // 3. Curl + Casson Viscoplastic Shear-Thinning Stress + Linear Boussinesq Gravity
    passA.setPipeline(this.pFluidCurl);
    passA.setBindGroup(0, this.bgFluidCurl);
    passA.dispatchWorkgroups(wgVelX, wgVelY, wgVelZ);

    passA.setPipeline(this.pFluidCassonForces);
    passA.setBindGroup(0, this.bgFluidCassonForces[this.velState][this.concState]);
    passA.dispatchWorkgroups(wgVelX, wgVelY, wgVelZ);
    this.velState = (1 - this.velState) as 0 | 1;

    // 4. Two-Way Free-Surface Dirichlet Poisson Solve via 3-Level Geometric Multigrid V-Cycle
    passA.setPipeline(this.pFluidMacRhs);
    passA.setBindGroup(0, this.bgFluidMacRhs[this.velState][this.waveState]);
    passA.dispatchWorkgroups(wgVelX, wgVelY, wgVelZ);

    // Level 0 pre-smooth (1 RBGS sweep on temporally warm-started L0 pressure)
    passA.setPipeline(this.pMgRed);
    passA.setBindGroup(0, this.bgMgL0Red[this.pressState]);
    passA.dispatchWorkgroups(wgMg0X, wgMg0Y, wgMg0Z);

    passA.setPipeline(this.pMgBlack);
    passA.setBindGroup(0, this.bgMgL0Black[(1 - this.pressState) as 0 | 1]);
    passA.dispatchWorkgroups(wgMg0X, wgMg0Y, wgMg0Z);

    // Restrict L0 -> L1
    passA.setPipeline(this.pMgResidual);
    passA.setBindGroup(0, this.bgMgL0Res[this.pressState]);
    passA.dispatchWorkgroups(wgMg0X, wgMg0Y, wgMg0Z);

    passA.setPipeline(this.pMgRestrict);
    passA.setBindGroup(0, this.bgMgRestrict0to1);
    passA.dispatchWorkgroups(wgMg1X, wgMg1Y, wgMg1Z);

    // Level 1 pre-smooth (2 RBGS sweeps from zero initial error -> MgErr1[0])
    passA.setPipeline(this.pMgRedZero);
    passA.setBindGroup(0, this.bgMgL1RedZero);
    passA.dispatchWorkgroups(wgMg1X, wgMg1Y, wgMg1Z);

    passA.setPipeline(this.pMgBlack);
    passA.setBindGroup(0, this.bgMgL1Black1to0);
    passA.dispatchWorkgroups(wgMg1X, wgMg1Y, wgMg1Z);

    passA.setPipeline(this.pMgRed);
    passA.setBindGroup(0, this.bgMgL1Red0to1);
    passA.dispatchWorkgroups(wgMg1X, wgMg1Y, wgMg1Z);

    passA.setPipeline(this.pMgBlack);
    passA.setBindGroup(0, this.bgMgL1Black1to0);
    passA.dispatchWorkgroups(wgMg1X, wgMg1Y, wgMg1Z);

    // Restrict L1 -> L2
    passA.setPipeline(this.pMgResidual);
    passA.setBindGroup(0, this.bgMgL1Res);
    passA.dispatchWorkgroups(wgMg1X, wgMg1Y, wgMg1Z);

    passA.setPipeline(this.pMgRestrict);
    passA.setBindGroup(0, this.bgMgRestrict1to2);
    passA.dispatchWorkgroups(wgMg2X, wgMg2Y, wgMg2Z);

    // Level 2 pre-smooth (2 RBGS sweeps from zero initial error -> MgErr2[0])
    passA.setPipeline(this.pMgRedZero);
    passA.setBindGroup(0, this.bgMgL2RedZero);
    passA.dispatchWorkgroups(wgMg2X, wgMg2Y, wgMg2Z);

    passA.setPipeline(this.pMgBlack);
    passA.setBindGroup(0, this.bgMgL2Black1to0);
    passA.dispatchWorkgroups(wgMg2X, wgMg2Y, wgMg2Z);

    passA.setPipeline(this.pMgRed);
    passA.setBindGroup(0, this.bgMgL2Red0to1);
    passA.dispatchWorkgroups(wgMg2X, wgMg2Y, wgMg2Z);

    passA.setPipeline(this.pMgBlack);
    passA.setBindGroup(0, this.bgMgL2Black1to0);
    passA.dispatchWorkgroups(wgMg2X, wgMg2Y, wgMg2Z);

    // Restrict L2 -> L3
    passA.setPipeline(this.pMgResidual);
    passA.setBindGroup(0, this.bgMgL2Res);
    passA.dispatchWorkgroups(wgMg2X, wgMg2Y, wgMg2Z);

    passA.setPipeline(this.pMgRestrict);
    passA.setBindGroup(0, this.bgMgRestrict2to3);
    passA.dispatchWorkgroups(wgMg3X, wgMg3Y, wgMg3Z);

    // Level 3 shared-memory coarse solve (6 full RBGS sweeps in 1 workgroup -> MgErr3[0])
    passA.setPipeline(this.pMgCoarseL3);
    passA.setBindGroup(0, this.bgMgL3Coarse);
    passA.dispatchWorkgroups(1, 1, 1);

    // Prolong L3 -> L2 & Post-Smooth (2 RBGS sweeps -> MgErr2[1])
    passA.setPipeline(this.pMgProlong);
    passA.setBindGroup(0, this.bgMgProlong3to2);
    passA.dispatchWorkgroups(wgMg2X, wgMg2Y, wgMg2Z);

    for (let s = 0; s < 2; s++) {
      passA.setPipeline(this.pMgRed);
      passA.setBindGroup(0, this.bgMgL2Red1to0);
      passA.dispatchWorkgroups(wgMg2X, wgMg2Y, wgMg2Z);

      passA.setPipeline(this.pMgBlack);
      passA.setBindGroup(0, this.bgMgL2Black0to1);
      passA.dispatchWorkgroups(wgMg2X, wgMg2Y, wgMg2Z);
    }

    // Prolong L2 -> L1 & Post-Smooth (2 RBGS sweeps -> MgErr1[1])
    passA.setPipeline(this.pMgProlong);
    passA.setBindGroup(0, this.bgMgProlong2to1);
    passA.dispatchWorkgroups(wgMg1X, wgMg1Y, wgMg1Z);

    for (let s = 0; s < 2; s++) {
      passA.setPipeline(this.pMgRed);
      passA.setBindGroup(0, this.bgMgL1Red1to0);
      passA.dispatchWorkgroups(wgMg1X, wgMg1Y, wgMg1Z);

      passA.setPipeline(this.pMgBlack);
      passA.setBindGroup(0, this.bgMgL1Black0to1);
      passA.dispatchWorkgroups(wgMg1X, wgMg1Y, wgMg1Z);
    }

    // Prolong L1 -> L0 & Post-Smooth (1 RBGS sweep -> Press[pressState])
    passA.setPipeline(this.pMgProlong);
    passA.setBindGroup(0, this.bgMgProlong1to0[this.pressState]);
    passA.dispatchWorkgroups(wgMg0X, wgMg0Y, wgMg0Z);
    this.pressState = (1 - this.pressState) as 0 | 1;

    passA.setPipeline(this.pMgRed);
    passA.setBindGroup(0, this.bgMgL0Red[this.pressState]);
    passA.dispatchWorkgroups(wgMg0X, wgMg0Y, wgMg0Z);

    passA.setPipeline(this.pMgBlack);
    passA.setBindGroup(0, this.bgMgL0Black[(1 - this.pressState) as 0 | 1]);
    passA.dispatchWorkgroups(wgMg0X, wgMg0Y, wgMg0Z);

    // Project MAC faces & reconstruct divergence-free cell-centered velocity
    passA.setPipeline(this.pFluidMacProject);
    passA.setBindGroup(0, this.bgFluidMacProject[this.velState][this.pressState][this.waveState]);
    passA.dispatchWorkgroups(wgVelX, wgVelY, wgVelZ);
    this.velState = (1 - this.velState) as 0 | 1;

    passA.setPipeline(this.pFluidMacToCenter);
    passA.setBindGroup(0, this.bgFluidMacToCenter[this.velState]);
    passA.dispatchWorkgroups(wgVelX, wgVelY, wgVelZ);

    passA.end();

    // Hardware DMA clear of atomic fixed-point splat buffers before parcel emission & advection/splat
    enc.clearBuffer(this.splatIntactBuffer);
    enc.clearBuffer(this.splatLysedBuffer);

    const passB = enc.beginComputePass({
      label: 'PhysicsSubstepTransport',
      timestampWrites: recordTimestamps ? this.tsPass1 : undefined,
    });

    // Emit newly drained Lagrangian RBC parcels across all active wetting necks (transferring any wrapped parcel mass into splatLysed)
    if (substepSpawnCount > 0) {
      passB.setPipeline(this.pParticleEmit);
      passB.setBindGroup(0, this.bgParticleEmit[this.waveState]);
      passB.dispatchWorkgroups(Math.ceil(substepSpawnCount / PARTICLE_WORKGROUP_SIZE), 1, 1);
    }

    // 5. Lagrangian RBC Parcel Advection + Stretch Anisotropy + Osmotic Hemolysis + Splatting
    if (this.highWaterParticles > 0) {
      passB.setPipeline(this.pParticleAdvectSplat);
      passB.setBindGroup(0, this.bgParticleAdvectSplat[this.concState]);
      passB.dispatchWorkgroups(Math.ceil(this.highWaterParticles / PARTICLE_WORKGROUP_SIZE), 1, 1);
    }

    // 6. Super-Resolved 192x256x192 Eulerian Lysed-Hb Advection/Diffusion + Composite Resolve (@workgroup_size(8, 8, 4))
    const wgConcX = CONC_GRID_X / 8;
    const wgConcY = CONC_GRID_Y / 8;
    const wgConcZ = CONC_GRID_Z / 4;

    passB.setPipeline(this.pConcPredict);
    passB.setBindGroup(0, this.bgConcPredict[this.concState]);
    passB.dispatchWorkgroups(wgConcX, wgConcY, wgConcZ);

    passB.setPipeline(this.pConcResolve);
    passB.setBindGroup(0, this.bgConcResolve[this.concState]);
    passB.dispatchWorkgroups(wgConcX, wgConcY, wgConcZ);
    this.concState = (1 - this.concState) as 0 | 1;

    passB.end();
  }

  private encodeLightingPasses(pass: GPUComputePassEncoder): void {
    pass.setPipeline(this.pLighting);
    pass.setBindGroup(0, this.bgLighting[this.concState][this.waveState]);
    pass.dispatchWorkgroups(VEL_GRID_X / 8, VEL_GRID_Y / 4, VEL_GRID_Z / 4);

    const wgBrickX = BRICK_GRID_X / 4;
    const wgBrickY = BRICK_GRID_Y / 4;
    const wgBrickZ = BRICK_GRID_Z / 4;

    pass.setPipeline(this.pBrickRaw);
    pass.setBindGroup(0, this.bgBrickRaw);
    pass.dispatchWorkgroups(wgBrickX, wgBrickY, wgBrickZ);

    pass.setPipeline(this.pBrickDilate);
    pass.setBindGroup(0, this.bgBrickDilate);
    pass.dispatchWorkgroups(wgBrickX, wgBrickY, wgBrickZ);
  }

  private dispatchLightingOnly(): void {
    const enc = this.device.createCommandEncoder({ label: 'InitLighting' });
    const pass = enc.beginComputePass();
    this.encodeLightingPasses(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Deterministically steps the simulation forward to `targetSimTime` using exact FIXED_SIM_DT increments.
   */
  public async seekToSimTime(targetSimTime: number, seed = 1.0): Promise<void> {
    if (this.destroyed) return;
    if (targetSimTime < this.simTime - 1e-5) {
      this.resetSimulation(seed);
    }
    const aspect = (this.canvas.width || 1280) / (this.canvas.height || 720);
    this.camera.updateMatrices(aspect);

    const stepsNeeded = Math.max(0, Math.round((targetSimTime - this.simTime) / FIXED_SIM_DT));
    const batchSize = 6;
    for (let i = 0; i < stepsNeeded; i++) {
      this.frameIndex++;
      if (this.hasAutoSqueezePendantDrop()) {
        this.multiDrops.updatePendantDrop(FIXED_SIM_DT);
      }
      const { packed, spawnCount } = this.advanceSubstepUniforms(FIXED_SIM_DT);
      this.device.queue.writeBuffer(this.uniformBuffer, 0, packed.buffer, 0, SIM_UNIFORM_BYTES);

      const enc = this.device.createCommandEncoder();
      this.encodePhysicsStep(enc, spawnCount);
      if (i === stepsNeeded - 1) {
        const lPass = enc.beginComputePass();
        this.encodeLightingPasses(lPass);
        lPass.end();
      }
      this.device.queue.submit([enc.finish()]);
      if ((i + 1) % batchSize === 0) {
        await this.device.queue.onSubmittedWorkDone();
      }
    }
    if (stepsNeeded === 0) {
      this.uploadSnapshotUniforms(FIXED_SIM_DT, this.spawnCursor, 0);
      this.dispatchLightingOnly();
    }
    this.camera.syncHeroFollow(this.simTime, true);
    this.renderCurrentFrameOnly();
    await this.device.queue.onSubmittedWorkDone();
  }

  /**
   * Pauses playback (if running) and queues `count` exact FIXED_SIM_DT slow-motion substeps
   * to execute cleanly on the next renderFrame without breaking camera zoom/orbit state.
   */
  public stepPausedSubsteps(count = 1): void {
    if (this.destroyed) return;
    const clamped = Math.max(1, Math.min(12, Math.round(count)));
    this.paused = true;
    this.pendingManualSubsteps = Math.min(12, this.pendingManualSubsteps + clamped);
    this.pausedAccumCount = 0;
  }

  public clearSubjectLock(): void {
    this.lockedSubjectId = null;
  }

  public hasSubjectLock(): boolean {
    return this.lockedSubjectId !== null;
  }

  /**
   * Centers the orbit & thin-lens focal pivot on the newest active drop or descending vortex ring,
   * cycling through concurrent drops/plumes on repeated presses.
   */
  public cycleFocusSubject(maxInspectDistance = 0.68): InspectionSubject | null {
    if (this.destroyed) return null;
    const subjects = this.multiDrops.getInspectionSubjects();
    if (subjects.length === 0) {
      this.lockedSubjectId = null;
      this.camera.focusOn([0.0, 0.02, 0.0], maxInspectDistance, false);
      return null;
    }
    let chosen = subjects[0];
    if (this.lockedSubjectId !== null) {
      const curIdx = subjects.findIndex((s) => s.id === this.lockedSubjectId);
      if (curIdx >= 0) {
        chosen = subjects[(curIdx + 1) % subjects.length];
      }
    }
    this.lockedSubjectId = chosen.id;
    this.camera.focusOn(chosen.center, maxInspectDistance, false);
    this.pausedAccumCount = 0;
    return chosen;
  }

  /**
   * Projects all active drops and descending vortex rings into screen NDC and locks camera focus
   * onto the subject closest to `(ndcX, ndcY)`.
   */
  public focusNearestSubjectAtNdc(
    ndcX: number,
    ndcY: number,
    maxInspectDistance = 0.68
  ): InspectionSubject | null {
    if (this.destroyed) return null;
    const subjects = this.multiDrops.getInspectionSubjects();
    if (subjects.length === 0) {
      return this.cycleFocusSubject(maxInspectDistance);
    }
    const aspect = (this.canvas.width || 1) / (this.canvas.height || 1);
    this.camera.updateMatrices(aspect);
    const vp = this.camera.viewProjMatrix;

    let best = subjects[0];
    let bestDist2 = Infinity;
    for (const s of subjects) {
      const [x, y, z] = s.center;
      const clipX = vp[0] * x + vp[4] * y + vp[8] * z + vp[12];
      const clipY = vp[1] * x + vp[5] * y + vp[9] * z + vp[13];
      const clipW = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
      if (clipW <= 1e-4) continue;
      const sx = clipX / clipW;
      const sy = clipY / clipW;
      const d2 = (sx - ndcX) * (sx - ndcX) + (sy - ndcY) * (sy - ndcY);
      if (d2 < bestDist2) {
        bestDist2 = d2;
        best = s;
      }
    }

    this.lockedSubjectId = best.id;
    this.camera.focusOn(best.center, maxInspectDistance, false);
    this.pausedAccumCount = 0;
    return best;
  }

  private syncLockedSubjectTarget(instant = false): void {
    if (this.lockedSubjectId === null || this.camera.followHeroEnabled) return;
    const subjects = this.multiDrops.getInspectionSubjects();
    if (subjects.length === 0) return;

    let match = subjects.find((s) => s.id === this.lockedSubjectId);
    if (!match) {
      // Seamless handoff when a locked aerial drop finishes draining into its nascent submerged vortex ring
      let closestPlume: InspectionSubject | null = null;
      let bestD2 = 0.04; // within 20 cm of current camera target
      for (const s of subjects) {
        const dx = s.center[0] - this.camera.target[0];
        const dy = s.center[1] - this.camera.target[1];
        const dz = s.center[2] - this.camera.target[2];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < bestD2) {
          bestD2 = d2;
          closestPlume = s;
        }
      }
      if (closestPlume) {
        this.lockedSubjectId = closestPlume.id;
        match = closestPlume;
      } else {
        this.lockedSubjectId = null;
        return;
      }
    }

    this.camera.focusOn(match.center, undefined, instant);
    if (instant) {
      this.camera.updateMatrices((this.canvas.width || 1) / (this.canvas.height || 1));
    }
  }

  private computePausedDropSignature(snap: MultiDropSnapshot): number {
    let sig = snap.activeCount * 101.0;
    const slots = snap.slots;
    for (let i = 0; i < slots.length; i++) {
      const s = slots[i];
      if (!s.active) continue;
      sig +=
        (i + 1) *
        (s.center[0] * 13.0 +
          s.center[1] * 31.0 +
          s.center[2] * 47.0 +
          s.radius * 97.0 +
          s.pipettePresence * 7.0);
    }
    return sig;
  }

  private isPausedSceneStationary(dropSig: number): boolean {
    if (Math.abs(this.simTime - this.lastPausedSimTime) > 1e-7) return false;
    if (Math.abs(this.config.lightingMode - this.lastPausedLightingMode) > 1e-5) return false;
    if (Math.abs(dropSig - this.lastPausedDropSig) > 1e-6) return false;
    const cw = Math.max(2, this.canvas.width || 1280);
    const ch = Math.max(2, this.canvas.height || 800);
    if (cw !== this.volSwapW || ch !== this.volSwapH) return false;
    const vp = this.camera.viewProjMatrix;
    for (let i = 0; i < 16; i++) {
      if (Math.abs(vp[i] - this.lastPausedViewProj[i]) > 1e-5) {
        return false;
      }
    }
    return true;
  }

  private recordPausedSceneSignature(dropSig: number): void {
    this.lastPausedSimTime = this.simTime;
    this.lastPausedLightingMode = this.config.lightingMode;
    this.lastPausedDropSig = dropSig;
    this.lastPausedViewProj.set(this.camera.viewProjMatrix);
  }

  private encodeOpticalRenderPasses(
    enc: GPUCommandEncoder,
    swapView: GPUTextureView,
    recordTimestamps = false,
    accumBlendWeight = 1.0,
    skipVolumetricPass = false
  ): void {
    this.ensureVolRenderTarget();

    if (!skipVolumetricPass) {
      const useAccumBlend = accumBlendWeight < 0.999;
      const volPass = enc.beginRenderPass({
        label: 'VolumetricHDRPass',
        timestampWrites: recordTimestamps ? this.tsPass3 : undefined,
        colorAttachments: [
          {
            view: this.viewVolRender,
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            loadOp: useAccumBlend ? 'load' : 'clear',
            storeOp: 'store',
          },
        ],
      });
      if (useAccumBlend) {
        volPass.setPipeline(this.pRenderAccum);
        volPass.setBlendConstant({
          r: accumBlendWeight,
          g: accumBlendWeight,
          b: accumBlendWeight,
          a: 1.0,
        });
      } else {
        const useMacroInspectPso = this.paused || this.camera.distance < 1.08;
        volPass.setPipeline(useMacroInspectPso ? this.pRenderMacro : this.pRender);
      }
      volPass.setBindGroup(0, this.bgRender[this.concState][this.waveState]);
      volPass.draw(3, 1, 0, 0);
      volPass.end();
    }

    const compPass = enc.beginRenderPass({
      label: 'CompositeToneMapPass',
      timestampWrites: recordTimestamps ? this.tsPass4 : undefined,
      colorAttachments: [
        {
          view: swapView,
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    compPass.setPipeline(this.pComposite);
    compPass.setBindGroup(0, this.bgComposite[this.waveState]);
    compPass.draw(3, 1, 0, 0);
    compPass.end();
  }

  private resolvePassTimestamps(enc: GPUCommandEncoder, hasPhysicsPasses: boolean): number {
    if (!this.supportsTimestamps || !this.tsQuerySet || !this.tsResolveBuffer || !this.tsReadBuffers) {
      return -1;
    }
    const slot = this.tsRingCursor;
    if (this.tsReadInFlight[slot]) {
      return -1;
    }
    if (hasPhysicsPasses) {
      enc.resolveQuerySet(this.tsQuerySet, 0, 10, this.tsResolveBuffer, 0);
    } else {
      // W3C WebGPU requires resolveQuerySet destinationOffset to be 256-byte aligned (0 here)
      enc.resolveQuerySet(this.tsQuerySet, 4, 6, this.tsResolveBuffer, 0);
    }
    enc.copyBufferToBuffer(this.tsResolveBuffer, 0, this.tsReadBuffers[slot], 0, 80);
    this.tsRingCursor = (1 - slot) as 0 | 1;
    return slot;
  }

  private collectPassTimestamps(slot: 0 | 1, hasPhysicsPasses: boolean): void {
    if (!this.tsReadBuffers || this.destroyed || this.isDeviceLost) return;
    const readBuf = this.tsReadBuffers[slot];
    this.tsReadInFlight[slot] = true;
    void readBuf
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        if (this.destroyed || this.isDeviceLost) return;
        const ns = new BigInt64Array(readBuf.getMappedRange());
        const deltaMs = (i0: number, i1: number): number => {
          const dt = Number(ns[i1] - ns[i0]) * 1e-6;
          return dt >= 0 && Number.isFinite(dt) ? dt : 0;
        };
        const base = hasPhysicsPasses ? 0 : -4;
        const macMs = hasPhysicsPasses ? deltaMs(0, 1) : 0;
        const parcelMs = hasPhysicsPasses ? deltaMs(2, 3) : 0;
        const lightMs = deltaMs(4 + base, 5 + base);
        const volMs = deltaMs(6 + base, 7 + base);
        const compMs = deltaMs(8 + base, 9 + base);
        // Measure true end-to-end GPU span (ns[last] - ns[0]) rather than summing overlapped Apple Silicon TBDR stages
        const totalSpanMs = deltaMs(0, 9 + base);
        readBuf.unmap();

        this.gpuPassTimingsMs.macAndWaveMs = +macMs.toFixed(3);
        this.gpuPassTimingsMs.parcelAndConcMs = +parcelMs.toFixed(3);
        this.gpuPassTimingsMs.lightingAndBricksMs = +lightMs.toFixed(3);
        this.gpuPassTimingsMs.halfResVolumetricMs = +volMs.toFixed(3);
        this.gpuPassTimingsMs.retinaCompositeMs = +compMs.toFixed(3);
        this.gpuPassTimingsMs.totalGpuMs = +totalSpanMs.toFixed(3);
        this.gpuPassTimingsMs.sampleCount++;
      })
      .catch(() => {
        // Ignore aborted mapAsync on engine teardown
      })
      .finally(() => {
        this.tsReadInFlight[slot] = false;
      });
  }

  public renderCurrentFrameOnly(): void {
    if (this.destroyed) return;
    const swapView = this.acquireSwapchainView();
    if (!swapView) return;

    this.pausedAccumCount = 0;
    this.lastPausedSimTime = -1.0;

    const aspect = (this.canvas.width || 1) / (this.canvas.height || 1);
    this.camera.updateMatrices(aspect);
    this.uploadSnapshotUniforms(0.0, this.spawnCursor, 0);

    const enc = this.device.createCommandEncoder({ label: 'RenderOnlyEncoder' });
    const lPass = enc.beginComputePass({ label: 'LightingRefresh' });
    this.encodeLightingPasses(lPass);
    lPass.end();

    this.encodeOpticalRenderPasses(enc, swapView);

    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Main per-frame update and consolidated single-encoder GPU submission across ALL playback speeds
   */
  public renderFrame(dtRealSec: number): MultiDropSnapshot {
    if (this.destroyed) {
      return this.getCurrentKinematics();
    }
    const swapView = this.acquireSwapchainView();
    if (!swapView) {
      return this.getCurrentKinematics();
    }

    const clampedRealDt = Math.min(Math.max(dtRealSec, 0.0), 0.05);
    const aspect = (this.canvas.width || 1) / (this.canvas.height || 1);
    let hasSubsequentTopsideDrop = false;
    if (this.simTime > 1.35) {
      const slots = this.multiDrops.slots;
      for (let i = 0; i < slots.length; i++) {
        const sl = slots[i];
        if (!sl.active) continue;
        if (sl.mode === 'pendant' || sl.mode === 'aerial' || (sl.mode === 'coalescing' && sl.ageInPhase < 0.28)) {
          hasSubsequentTopsideDrop = true;
          break;
        }
      }
    }
    const cueTarget = hasSubsequentTopsideDrop ? 1.0 : 0.0;
    const cueRate = cueTarget > this.heroTopsideCue ? 7.5 : 2.2;
    this.heroTopsideCue += (cueTarget - this.heroTopsideCue) * (1.0 - Math.exp(-clampedRealDt * cueRate));
    this.camera.syncHeroFollow(this.simTime, false, this.heroTopsideCue);
    this.syncLockedSubjectTarget();
    this.camera.tick(clampedRealDt, aspect);

    let substeps = 0;
    if (this.pendingManualSubsteps > 0) {
      substeps = Math.min(MAX_SUBSTEPS, this.pendingManualSubsteps);
      this.pendingManualSubsteps -= substeps;
      this.pausedAccumCount = 0;
      this.lastPausedSimTime = -1.0;
    } else if (!this.paused && this.timeScale > 0.0) {
      const stepBudget = this.timeScale > 1.05 ? MAX_SUBSTEPS : 1;
      this.simAccumulator += clampedRealDt * this.timeScale;
      while (this.simAccumulator >= FIXED_SIM_DT && substeps < stepBudget) {
        this.simAccumulator -= FIXED_SIM_DT;
        substeps++;
      }
      if (this.simAccumulator > FIXED_SIM_DT) {
        this.simAccumulator = Math.min(this.simAccumulator, FIXED_SIM_DT * 0.5);
      }
    }

    const cycleLimit = this.multiDrops.getEffectiveCycleDuration(SHOT_CYCLE_DURATION);
    if (
      this.autoLoop &&
      !this.paused &&
      !this.multiDrops.hasActivePendantDrop() &&
      this.simTime >= cycleLimit
    ) {
      this.resetSimulation();
      substeps = 1;
    }

    if (substeps === 0) {
      const snap = this.multiDrops.getSnapshot();
      const dropSig = this.computePausedDropSignature(snap);
      const stationary = this.paused && this.pausedAccumCount > 0 && this.isPausedSceneStationary(dropSig);

      if (!stationary) {
        this.frameIndex++;
        this.pausedBaseFrame = this.frameIndex;
        this.pausedAccumCount = 1;
        this.recordPausedSceneSignature(dropSig);

        const kin = this.uploadSnapshotUniforms(0.0, this.spawnCursor, 0);
        const enc = this.device.createCommandEncoder({ label: 'PausedFrameEncoder' });
        const lPass = enc.beginComputePass({
          label: 'PausedLightingPass',
          timestampWrites: this.tsPass2,
        });
        this.encodeLightingPasses(lPass);
        lPass.end();

        this.encodeOpticalRenderPasses(enc, swapView, true, 1.0, false);
        const tsSlot = this.resolvePassTimestamps(enc, false);

        this.device.queue.submit([enc.finish()]);
        if (tsSlot >= 0) {
          this.collectPassTimestamps(tsSlot as 0 | 1, false);
        }
        return kin;
      }

      if (this.pausedAccumCount < PAUSED_ACCUM_TARGET) {
        // Stationary paused progressive super-sampling (frames 2..16): skip static 3D lighting compute,
        // stratify golden-ratio raymarch phase (dt = -1.0 sentinel), and hardware-blend into VolRenderHDR_F16.
        const sampleIdx = this.pausedAccumCount;
        this.pausedAccumCount++;
        this.frameIndex = this.pausedBaseFrame + sampleIdx;
        const blendW = 1.0 / this.pausedAccumCount;
        const kin = this.uploadSnapshotUniforms(-1.0, this.spawnCursor, 0);
        const enc = this.device.createCommandEncoder({ label: 'PausedAccumEncoder' });
        this.encodeOpticalRenderPasses(enc, swapView, false, blendW, false);
        this.device.queue.submit([enc.finish()]);
        return kin;
      }

      // Fully converged stationary pause (pausedAccumCount >= 16): skip lighting & volumetric passes completely,
      // presenting the converged 16-sample VolRenderHDR_F16 buffer at native Retina with zero shimmer (< 0.5 ms GPU).
      this.frameIndex = this.pausedBaseFrame + PAUSED_ACCUM_TARGET - 1;
      const kin = this.uploadSnapshotUniforms(0.0, this.spawnCursor, 0);
      const enc = this.device.createCommandEncoder({ label: 'PausedConvergedEncoder' });
      this.encodeOpticalRenderPasses(enc, swapView, false, 1.0, true);
      this.device.queue.submit([enc.finish()]);
      return kin;
    }

    this.pausedAccumCount = 0;
    this.lastPausedSimTime = -1.0;

    if (substeps === 1) {
      this.frameIndex++;
      const { packed, drops, spawnCount } = this.advanceSubstepUniforms(FIXED_SIM_DT);
      this.device.queue.writeBuffer(this.uniformBuffer, 0, packed.buffer, 0, SIM_UNIFORM_BYTES);

      const enc = this.device.createCommandEncoder({ label: 'SimFrameEncoder' });
      this.encodePhysicsStep(enc, spawnCount, true);

      const lPass = enc.beginComputePass({
        label: 'LightingPass',
        timestampWrites: this.tsPass2,
      });
      this.encodeLightingPasses(lPass);
      lPass.end();

      this.encodeOpticalRenderPasses(enc, swapView, true);
      const tsSlot = this.resolvePassTimestamps(enc, true);

      this.device.queue.submit([enc.finish()]);
      if (tsSlot >= 0) {
        this.collectPassTimestamps(tsSlot as 0 | 1, true);
      }
      return drops;
    }

    // Multi-substep (e.g. 500 fps / 2.0x speed): pack all substep uniform blocks into staging buffer
    // in a SINGLE writeBuffer and execute copyBufferToBuffer + compute + render in a SINGLE submit
    let latestKin = this.getCurrentKinematics();
    for (let s = 0; s < substeps; s++) {
      this.frameIndex++;
      const { packed, drops, spawnCount } = this.advanceSubstepUniforms(FIXED_SIM_DT);
      this.multiStepScratch.set(packed, s * SIM_UNIFORM_FLOATS);
      this.substepSpawnCounts[s] = spawnCount;
      latestKin = drops;
    }
    this.device.queue.writeBuffer(
      this.uniformStagingBuffer,
      0,
      this.multiStepScratch.buffer,
      0,
      substeps * SIM_UNIFORM_BYTES
    );

    const enc = this.device.createCommandEncoder({ label: 'MultiSubstepFrameEncoder' });
    for (let s = 0; s < substeps; s++) {
      enc.copyBufferToBuffer(
        this.uniformStagingBuffer,
        s * SIM_UNIFORM_BYTES,
        this.uniformBuffer,
        0,
        SIM_UNIFORM_BYTES
      );
      this.encodePhysicsStep(enc, this.substepSpawnCounts[s], s === substeps - 1);
    }

    const lPass = enc.beginComputePass({
      label: 'LightingPass',
      timestampWrites: this.tsPass2,
    });
    this.encodeLightingPasses(lPass);
    lPass.end();

    this.encodeOpticalRenderPasses(enc, swapView, true);
    const tsSlot = this.resolvePassTimestamps(enc, true);

    this.device.queue.submit([enc.finish()]);
    if (tsSlot >= 0) {
      this.collectPassTimestamps(tsSlot as 0 | 1, true);
    }
    return latestKin;
  }

  public destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    if (this.hdrMediaQuery && this.onHdrMediaChange) {
      this.hdrMediaQuery.removeEventListener('change', this.onHdrMediaChange);
      this.hdrMediaQuery = null;
      this.onHdrMediaChange = null;
    }
    try {
      this.context.unconfigure();
    } catch {
      // Ignore if context already detached
    }
    this.tsQuerySet?.destroy();
    this.tsResolveBuffer?.destroy();
    if (this.tsReadBuffers) {
      this.tsReadBuffers[0].destroy();
      this.tsReadBuffers[1].destroy();
    }
    this.uniformBuffer.destroy();
    this.uniformStagingBuffer.destroy();
    this.particlesBuffer.destroy();
    this.splatIntactBuffer.destroy();
    this.splatLysedBuffer.destroy();

    this.texWave[0].destroy();
    this.texWave[1].destroy();
    this.texWaveSpecAux.destroy();
    this.texWaveSpecA.destroy();
    this.texWaveSpecB.destroy();

    this.texVel[0].destroy();
    this.texVel[1].destroy();
    this.texVelCenter.destroy();
    this.texCurl.destroy();

    this.texPress[0].destroy();
    this.texPress[1].destroy();
    this.texRhs0.destroy();
    this.texRes0.destroy();

    this.texMgErr1[0].destroy();
    this.texMgErr1[1].destroy();
    this.texMgRhs1.destroy();
    this.texMgRes1.destroy();

    this.texMgErr2[0].destroy();
    this.texMgErr2[1].destroy();
    this.texMgRhs2.destroy();
    this.texMgRes2.destroy();

    this.texMgErr3[0].destroy();
    this.texMgErr3[1].destroy();
    this.texMgRhs3.destroy();

    this.texConc[0].destroy();
    this.texConc[1].destroy();
    this.texConcPred.destroy();
    this.texLight.destroy();
    this.texBrickRaw.destroy();
    this.texBrickDilated.destroy();
    this.texSpectralLut.destroy();
    this.texVolRender.destroy();

    this.device.destroy();
  }
}
