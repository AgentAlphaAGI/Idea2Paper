
export type Language = 'en' | 'zh';

export enum PipelineStatus {
  IDLE = 'IDLE',
  RUNNING = 'RUNNING',
  COMPLETE = 'COMPLETE',
  ERROR = 'ERROR'
}

export enum PipelineStep {
  INIT = 'INIT',
  RETRIEVAL = 'RETRIEVAL', // Combines: Recall + Pattern Selection
  GENERATION = 'GENERATION',
  REVIEW = 'REVIEW', // Combines: Critic Review + Refinement
  VALIDATION = 'VALIDATION', // Combines: Novelty Check + Verification + Bundling
  DONE = 'DONE',
  ERROR = 'ERROR'
}

export interface PipelineLog {
  timestamp: string;
  step: PipelineStep;
  message: string;
  details?: string;
}

export interface GeneratedStory {
  title: string;
  abstract: string;
  introduction: string;
  methodology: string;
  experiments: string;
  contributions: string[];
}

export interface ReviewScore {
  reviewer: string;
  role: string;
  score: number; // 1-10
  feedback: string;
  anchors?: Array<{
    paper_id: string;
    title: string;
    score10: number;
    review_count: number;
  }>;
}

export interface PipelineResult {
  story: GeneratedStory;
  reviews: ReviewScore[];
  logs: PipelineLog[];
}

export interface ApiEndpoints {
  runPipeline: string;
  checkStatus: string;
  cancelPipeline: string;
  queryGraph: string;
  exportResult: string;
}

export interface AppConfig {
  // System Config
  apiKey: string; // Backend Access Key
  baseUrl: string;
  endpoints: ApiEndpoints;
  useMock: boolean;
  theme: 'light' | 'dark';

  // SiliconFlow / LLM Keys
  siliconFlowApiKey: string;

  // LLM Config
  llmUrl: string;
  llmModel: string;
  llmTemperatures: {
    default: number;
    storyGenerator: number;
    storyGeneratorRewrite: number;
    storyReflector: number;
    patternSelector: number;
    ideaFusion: number;
    ideaFusionStage2: number;
    ideaFusionStage3: number;
    criticMain: number;
    criticRepair: number;
    criticAnchored: number;
    ideaPackagingParse: number;
    ideaPackagingPatternGuided: number;
    ideaPackagingJudge: number;
  };

  // Embedding Config
  embeddingUrl: string;
  embeddingModel: string;
  embeddingApiKey: string;
  indexDirMode: string;

  // Feature: Idea Packaging
  ideaPackaging: {
    enable: boolean;
    topNPatterns: number;
    maxExemplarPapers: number;
    candidateK: number;
    selectMode: string;
    forceEnQuery: boolean;
  };

  // Logging & Results
  logging: {
    enable: boolean;
    logDir: string;
    maxTextChars: number;
  };
  results: {
    enable: boolean;
    resultsDir: string;
  };

  // Critic Strictness
  critic: {
    strictJson: boolean;
    jsonRetries: number;
  };

  // Pass Rule
  passRule: {
    mode: string;
    minPatternPapers: number;
    fallback: string;
    score: number;
  };

  // Anchors
  anchors: {
    densifyEnable: boolean;
    quantiles: string;
  };

  // Novelty Check
  novelty: {
    enable: boolean;
    autoBuildIndex: boolean;
    buildBatchSize: number;
    action: 'report_only' | 'pivot' | 'fail';
    maxPivots: number;
  };

  // Verification
  verification: {
    enable: boolean;
    collisionThreshold: number;
  };

  // Indexing
  index: {
    autoPrepare: boolean;
    allowBuild: boolean;
  };

  // Recall Audit
  recall: {
    auditEnable: boolean;
    auditTopN: number;
  };
}
