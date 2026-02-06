import { PipelineLog, PipelineStep, GeneratedStory, ReviewScore, Language, AppConfig, PipelineResult, PipelineStatus } from '../types';

// ==========================================
// 1. Interfaces & Types
// ==========================================

export interface PipelineCallbacks {
  onLog: (log: PipelineLog) => void;
  onStatusChange?: (status: PipelineStatus) => void;
  onRunIdReceived?: (uiRunId: string) => void;
}

// ==========================================
// 2. Mock Data Definition
// ==========================================

const MOCK_DATA = {
  stories: {
    en: (idea: string) => ({
      title: `Automated Discovery of ${idea.substring(0, 20)}... via Anchored Graphs`,
      abstract: `In this work, we propose a novel framework for ${idea}. Unlike traditional approaches that rely on runtime reasoning, we leverage a pre-computed knowledge graph derived from ICLR papers. Our method demonstrates a 15% improvement in coherence and novelty scores compared to baselines.`,
      introduction: `Scientific discovery is often hindered by the challenge of transforming raw ideas into structured narratives. \n\nWe introduce Idea2Story, which decomposes this process. The core problem we address is ${idea}, which has traditionally been solved by manual heuristics.`,
      methodology: `Our approach consists of three phases:\n1. Knowledge Graph Construction: Utilizing SiliconFlow embeddings.\n2. Three-Path Retrieval: Combining Jaccard similarity with dense retrieval.\n3. Anchored Refinement: Using a multi-agent critic system.`,
      experiments: `We evaluated our model on the Idea2Story benchmark. Results show state-of-the-art performance in logical consistency (Score: 8.5/10).`,
      contributions: [
          "A comprehensive KG built from ICLR data.",
          "A deterministic anchored review mechanism.",
          "First end-to-end open-source research agent."
      ]
    }),
    zh: (idea: string) => ({
      title: `基于锚点图谱的${idea.substring(0, 10)}...自动发现研究`,
      abstract: `在这项工作中，我们提出了一种针对${idea}的新颖框架。与依赖运行时推理的传统方法不同，我们利用了从 ICLR 论文中提取的预计算知识图谱。我们的方法在连贯性和新颖性评分上比基线提高了 15%。`,
      introduction: `科学发现往往受到将原始想法转化为结构化叙事的挑战的阻碍。\n\n我们介绍了 Idea2Story, 它分解了这一过程。我们解决的核心问题是${idea}，这在传统上是通过人工启发式方法解决的。`,
      methodology: `我们的方法包含三个阶段：\n1. 知识图谱构建：利用 SiliconFlow 嵌入。\n2. 三路检索：结合 Jaccard 相似度和密集检索。\n3. 锚点润色：使用多智能体评论系统。`,
      experiments: `我们在 Idea2Story 基准测试上评估了我们的模型。结果显示在逻辑一致性方面达到了最先进的性能（评分：8.5/10）。`,
      contributions: [
          "一个基于 ICLR 数据构建的综合知识图谱。",
          "一种确定性的锚点评审机制。",
          "首个端到端的开源研究智能体。"
      ]
    })
  },
  reviews: {
    en: [
      { criterion: "Novelty", score: 8, reasoning: "The approach to using anchored graphs for this specific problem is unique compared to baselines." },
      { criterion: "Technical Soundness", score: 7, reasoning: "Methodology is solid, though the embedding model choice could be more flexible." },
      { criterion: "Clarity", score: 9, reasoning: "The narrative structure flows logically from problem definition to solution." }
    ],
    zh: [
      { criterion: "新颖性", score: 8, reasoning: "与基线相比，使用锚点图谱解决此特定问题的方法是独特的。" },
      { criterion: "技术稳健性", score: 7, reasoning: "方法论很扎实，尽管嵌入模型的选择可以更加灵活。" },
      { criterion: "清晰度", score: 9, reasoning: "叙事结构从问题定义到解决方案的逻辑流畅。" }
    ]
  },
  logs: {
    en: {
        init: "Initializing Idea2Paper environment...",
        loading: "Loading Configuration",
        kg_query: "Querying ICLR Knowledge Graph...",
        kg_found: "Found 124 related nodes",
        retrieval: "Executing Three-Path Retrieval",
        ranking: "Ranking candidates...",
        gen_skeleton: "Synthesizing narrative skeleton...",
        gen_draft: "Drafting abstract and contributions...",
        review_init: "Initiating Anchored Multi-Agent Review",
        review_compare: "Comparing against anchor papers...",
        review_score: "Scores calculated",
        refine: "Applying reviewer feedback...",
        done: "Story generation complete."
    },
    zh: {
        init: "初始化 Idea2Paper 环境...",
        loading: "加载配置",
        kg_query: "查询 ICLR 知识图谱...",
        kg_found: "发现 124 个相关节点",
        retrieval: "执行三路检索",
        ranking: "候选排序中...",
        gen_skeleton: "合成叙事骨架...",
        gen_draft: "起草摘要和贡献...",
        review_init: "启动锚点多智能体评审",
        review_compare: "与锚点论文对比中...",
        review_score: "分数已计算",
        refine: "应用评审反馈...",
        done: "故事生成完成。"
    }
  }
};

// ==========================================
// 3. Helper Functions
// ==========================================

const wait = (ms: number, signal?: AbortSignal) => {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort);
  });
};

// ==========================================
// 4. Mock Strategy Implementation
// ==========================================

const MockStrategy = {
  runPipeline: async (
    idea: string, 
    lang: Language,
    callbacks: PipelineCallbacks,
    signal?: AbortSignal
  ): Promise<PipelineResult> => {
    const { onLog } = callbacks;
    const msgs = MOCK_DATA.logs[lang];
    const logs: PipelineLog[] = [];

    const addLog = (step: PipelineStep, message: string, details?: string) => {
      const log = { timestamp: new Date().toLocaleTimeString(), step, message, details };
      logs.push(log);
      onLog(log);
    };

    // Simulate Network & Processing Delays
    addLog(PipelineStep.INIT, msgs.init);
    await wait(800, signal);
    addLog(PipelineStep.INIT, msgs.loading, "Mode: Mock Simulation");

    await wait(1200, signal);
    addLog(PipelineStep.RECALL, msgs.kg_query);
    await wait(1000, signal);
    addLog(PipelineStep.RECALL, msgs.kg_found, "Domains: [AI, Graph Theory]");

    await wait(1500, signal);
    addLog(PipelineStep.PATTERN_SELECTION, msgs.retrieval);
    await wait(800, signal);
    addLog(PipelineStep.PATTERN_SELECTION, msgs.ranking);

    await wait(2000, signal);
    addLog(PipelineStep.GENERATION, msgs.gen_skeleton);
    addLog(PipelineStep.GENERATION, msgs.gen_draft);
    await wait(1500, signal);

    addLog(PipelineStep.REVIEW, msgs.review_init);
    addLog(PipelineStep.REVIEW, msgs.review_compare);
    await wait(2000, signal);
    addLog(PipelineStep.REVIEW, msgs.review_score);

    addLog(PipelineStep.REFINEMENT, msgs.refine);
    await wait(1000, signal);
    addLog(PipelineStep.DONE, msgs.done);

    return {
      story: MOCK_DATA.stories[lang](idea),
      reviews: MOCK_DATA.reviews[lang],
      logs
    };
  }
};

// ==========================================
// 5. Real Strategy Implementation
// ==========================================

const RealStrategy = {
  runPipeline: async (
    idea: string,
    lang: Language,
    config: AppConfig,
    callbacks: PipelineCallbacks,
    signal?: AbortSignal
  ): Promise<PipelineResult> => {
    const { onLog, onStatusChange, onRunIdReceived } = callbacks;

    // Construct base URL from config
    const baseUrl = config.baseUrl.replace(/\/$/, '');
    const runEndpoint = `${baseUrl}/api/runs`;

    onLog({
        timestamp: new Date().toLocaleTimeString(),
        step: PipelineStep.INIT,
        message: lang === 'zh' ? "连接后端服务..." : "Connecting to backend...",
        details: `POST ${runEndpoint}`
    });

    try {
        // Start the pipeline
        const response = await fetch(runEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                idea,
                config: {
                   config_overrides: {
                       SILICONFLOW_API_KEY: config.siliconFlowApiKey,
                       LLM_API_URL: config.llmUrl,
                       LLM_MODEL: config.llmModel,
                       EMBEDDING_API_URL: config.embeddingUrl,
                       EMBEDDING_MODEL: config.embeddingModel,

                       // Idea Packaging
                       I2P_IDEA_PACKAGING_ENABLE: config.ideaPackaging.enable ? '1' : '0',
                       I2P_IDEA_PACKAGING_TOPN_PATTERNS: config.ideaPackaging.topNPatterns,
                       I2P_IDEA_PACKAGING_MAX_EXEMPLAR_PAPERS: config.ideaPackaging.maxExemplarPapers,
                       I2P_IDEA_PACKAGING_CANDIDATE_K: config.ideaPackaging.candidateK,
                       I2P_IDEA_PACKAGING_SELECT_MODE: config.ideaPackaging.selectMode,

                       // Novelty
                       I2P_NOVELTY_ENABLE: config.novelty.enable ? '1' : '0',
                       I2P_NOVELTY_ACTION: config.novelty.action,
                       I2P_NOVELTY_MAX_PIVOTS: config.novelty.maxPivots,

                       // Verification
                       I2P_VERIFICATION_ENABLE: config.verification.enable ? '1' : '0',
                       I2P_COLLISION_THRESHOLD: config.verification.collisionThreshold,

                       // Temperatures
                       I2P_LLM_TEMPERATURE_DEFAULT: config.llmTemperatures.default,
                       I2P_LLM_TEMPERATURE_STORY_GENERATOR: config.llmTemperatures.storyGenerator,
                       I2P_LLM_TEMPERATURE_CRITIC_MAIN: config.llmTemperatures.criticMain,

                       // Critic
                       I2P_CRITIC_STRICT_JSON: config.critic.strictJson ? '1' : '0',
                       I2P_CRITIC_JSON_RETRIES: config.critic.jsonRetries,

                       // Logging
                       I2P_ENABLE_LOGGING: config.logging.enable ? '1' : '0',
                       I2P_LOG_MAX_TEXT_CHARS: config.logging.maxTextChars,

                       // Results
                       I2P_RESULTS_ENABLE: config.results.enable ? '1' : '0',
                   }
                }
            }),
            signal
        });

        if (!response.ok) {
            let errorDetails = '';

            if (response.status === 404) {
               throw new Error(lang === 'zh'
                 ? `后端接口未找到 (404)。请检查路径: ${runEndpoint}`
                 : `Backend endpoint not found (404). Checked path: ${runEndpoint}`);
            }

            try {
                const errData = await response.json();
                errorDetails = errData.error || errData.message || JSON.stringify(errData);
            } catch (e) {
                errorDetails = response.statusText;
            }
            throw new Error(`API Request Failed (${response.status}): ${errorDetails}`);
        }

        const startData = await response.json();
        if (!startData.ok) {
            throw new Error(startData.error || 'Failed to start pipeline');
        }

        const uiRunId = startData.ui_run_id;

        // Notify caller of the run ID
        if (onRunIdReceived) {
            onRunIdReceived(uiRunId);
        }

        onLog({
            timestamp: new Date().toLocaleTimeString(),
            step: PipelineStep.INIT,
            message: lang === 'zh' ? "Pipeline 已启动" : "Pipeline started",
            details: `Run ID: ${uiRunId}`
        });

        // Poll for status and events
        return await pollPipelineStatus(baseUrl, uiRunId, lang, onLog, onStatusChange, signal);

    } catch (error: any) {
        if (error.name === 'AbortError') throw error;

        const errorMsg = error instanceof Error ? error.message : "Unknown network error";
        onLog({
            timestamp: new Date().toLocaleTimeString(),
            step: PipelineStep.ERROR,
            message: lang === 'zh' ? "API请求失败" : "API Request Failed",
            details: errorMsg
        });
        throw error;
    }
  }
};

// Helper function to poll pipeline status
async function pollPipelineStatus(
  baseUrl: string,
  uiRunId: string,
  lang: Language,
  onLog: (log: PipelineLog) => void,
  onStatusChange?: (status: PipelineStatus) => void,
  signal?: AbortSignal
): Promise<PipelineResult> {
  const statusEndpoint = `${baseUrl}/api/runs/${uiRunId}`;
  const eventsEndpoint = `${baseUrl}/api/runs/${uiRunId}/events`;
  const resultEndpoint = `${baseUrl}/api/runs/${uiRunId}/result`;

  let lastEventCount = 0;
  let isDone = false;
  let lastStageDetail = '';

  const stageToStep: Record<string, PipelineStep> = {
    'Initializing': PipelineStep.INIT,
    'Recall': PipelineStep.RETRIEVAL,
    'Pattern Selection': PipelineStep.RETRIEVAL,
    'Story Generation': PipelineStep.GENERATION,
    'Critic Review': PipelineStep.REVIEW,
    'Refinement': PipelineStep.REVIEW,
    'Novelty Check': PipelineStep.VALIDATION,
    'Verification': PipelineStep.VALIDATION,
    'Bundling': PipelineStep.VALIDATION,
    'Done': PipelineStep.DONE,
    'Failed': PipelineStep.ERROR,
  };

  while (!isDone && !signal?.aborted) {
    try {
      // Check status
      const statusResp = await fetch(statusEndpoint, { signal });
      if (statusResp.ok) {
        const statusData = await statusResp.json();

        if (statusData.ok) {
          const stageInfo = statusData.stage || {};
          const stage = stageInfo.name || 'Running';
          const stageDetail = stageInfo.detail || '';
          const status = statusData.status;

          // Update status
          if (status === 'done') {
            isDone = true;
            if (onStatusChange) onStatusChange(PipelineStatus.COMPLETE);
          } else if (status === 'failed') {
            isDone = true;
            if (onStatusChange) onStatusChange(PipelineStatus.ERROR);
          } else if (onStatusChange) {
            onStatusChange(PipelineStatus.RUNNING);
          }

          // Fetch events
          const eventsResp = await fetch(eventsEndpoint, { signal });
          if (eventsResp.ok) {
            const eventsData = await eventsResp.json();
            if (eventsData.ok && eventsData.events) {
              const newEvents = eventsData.events.slice(lastEventCount);
              lastEventCount = eventsData.events.length;

              // Convert backend events to frontend logs
              for (const event of newEvents) {
                const eventType = event.data?.event_type || event.event_type || '';
                const step = stageToStep[stage] || PipelineStep.INIT;

                onLog({
                  timestamp: new Date(event.ts || Date.now()).toLocaleTimeString(),
                  step,
                  message: `${stage}: ${eventType}`,
                  details: JSON.stringify(event.data?.payload || {}, null, 2).substring(0, 200)
                });
              }

              // If stage detail changed (e.g., building index progress), send a status update
              if (stageDetail && stageDetail !== lastStageDetail) {
                lastStageDetail = stageDetail;
                onLog({
                  timestamp: new Date().toLocaleTimeString(),
                  step: stageToStep[stage] || PipelineStep.INIT,
                  message: stage,
                  details: stageDetail
                });
              }
            }
          }
        }
      }

      if (!isDone) {
        await wait(2000, signal); // Poll every 2 seconds
      }
    } catch (error: any) {
      if (error.name === 'AbortError') throw error;
      // Continue polling on errors
      await wait(2000, signal);
    }
  }

  // Fetch final result
  const resultResp = await fetch(resultEndpoint, { signal });
  if (!resultResp.ok) {
    throw new Error('Failed to fetch result');
  }

  const resultData = await resultResp.json();
  if (!resultData.ok) {
    throw new Error(resultData.error || 'Failed to get result');
  }

  const finalStory = resultData.final_story || {};
  const pipelineResult = resultData.pipeline_result || {};

  // Convert to frontend format
  const story: GeneratedStory = {
    title: finalStory.title || '',
    abstract: finalStory.abstract || '',
    // Map backend fields to frontend fields
    introduction: finalStory.introduction || finalStory.problem_framing || '',
    methodology: finalStory.methodology || finalStory.method_skeleton || finalStory.solution || '',
    experiments: finalStory.experiments || finalStory.experiments_plan || '',
    contributions: finalStory.contributions || finalStory.innovation_claims || []
  };

  const reviews: ReviewScore[] = (pipelineResult.review_history || []).flatMap((reviewRound: any) => {
    const audit = reviewRound.audit || {};
    const anchors = (audit.anchors || []).slice(0, 5).map((anchor: any) => ({
      paper_id: anchor.paper_id || '',
      title: anchor.title || '',
      score10: anchor.score10 || 0,
      review_count: anchor.review_count || 0
    }));

    return (reviewRound.reviews || []).map((r: any) => ({
      reviewer: r.reviewer || '',
      role: r.role || '',
      score: r.score || 0,
      feedback: r.feedback || '',
      anchors: anchors
    }));
  });

  onLog({
    timestamp: new Date().toLocaleTimeString(),
    step: PipelineStep.DONE,
    message: lang === 'zh' ? "生成完成" : "Generation Complete"
  });

  return {
    story,
    reviews,
    logs: []
  };
}

// ==========================================
// 6. Main API Export
// ==========================================

export const api = {
    /**
     * Main entry point for running the Idea2Paper pipeline.
     */
    runPipeline: async (
        idea: string,
        lang: Language,
        config: AppConfig,
        callbacks: PipelineCallbacks,
        signal?: AbortSignal
    ): Promise<PipelineResult> => {
        if (config.useMock) {
            return MockStrategy.runPipeline(idea, lang, callbacks, signal);
        } else {
            return RealStrategy.runPipeline(idea, lang, config, callbacks, signal);
        }
    },

    /**
     * Terminate a running pipeline
     */
    terminatePipeline: async (
        uiRunId: string,
        config: AppConfig
    ): Promise<boolean> => {
        try {
            const baseUrl = config.baseUrl || 'http://localhost:8080';
            const response = await fetch(`${baseUrl}/api/runs/${uiRunId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                console.error('Failed to terminate pipeline:', response.statusText);
                return false;
            }

            const data = await response.json();
            return data.ok && data.terminated;
        } catch (error) {
            console.error('Error terminating pipeline:', error);
            return false;
        }
    },

    /**
     * List all available results
     */
    listResults: async (config: AppConfig): Promise<any[]> => {
        try {
            const baseUrl = config.baseUrl || 'http://localhost:8080';
            const response = await fetch(`${baseUrl}/api/results`);

            if (!response.ok) {
                console.error('Failed to fetch results:', response.statusText);
                return [];
            }

            const data = await response.json();
            return data.ok ? data.results : [];
        } catch (error) {
            console.error('Error fetching results:', error);
            return [];
        }
    },

    /**
     * Get a specific result by run_id
     */
    getResultById: async (runId: string, config: AppConfig): Promise<PipelineResult | null> => {
        try {
            const baseUrl = config.baseUrl || 'http://localhost:8080';
            const response = await fetch(`${baseUrl}/api/results/${runId}`);

            if (!response.ok) {
                console.error('Failed to fetch result:', response.statusText);
                return null;
            }

            const data = await response.json();
            if (!data.ok) {
                return null;
            }

            const finalStory = data.final_story || {};
            const pipelineResult = data.pipeline_result || {};

            // Convert to frontend format
            const story: GeneratedStory = {
                title: finalStory.title || '',
                abstract: finalStory.abstract || '',
                introduction: finalStory.introduction || finalStory.problem_framing || '',
                methodology: finalStory.methodology || finalStory.method_skeleton || finalStory.solution || '',
                experiments: finalStory.experiments || finalStory.experiments_plan || '',
                contributions: finalStory.contributions || finalStory.innovation_claims || []
            };

            const reviews: ReviewScore[] = (pipelineResult.review_history || []).flatMap((reviewRound: any) => {
                const audit = reviewRound.audit || {};
                const anchors = (audit.anchors || []).slice(0, 5).map((anchor: any) => ({
                    paper_id: anchor.paper_id || '',
                    title: anchor.title || '',
                    score10: anchor.score10 || 0,
                    review_count: anchor.review_count || 0
                }));

                return (reviewRound.reviews || []).map((r: any) => ({
                    reviewer: r.reviewer || '',
                    role: r.role || '',
                    score: r.score || 0,
                    feedback: r.feedback || '',
                    anchors: anchors
                }));
            });

            return {
                story,
                reviews,
                logs: []
            };
        } catch (error) {
            console.error('Error fetching result:', error);
            return null;
        }
    }
};
