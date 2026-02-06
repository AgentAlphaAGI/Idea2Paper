import React, { useState } from 'react';
import { AppConfig } from '../types';
import { Copy, Check, Globe, ChevronDown, ChevronRight } from 'lucide-react';

interface ApiDocsProps {
  config: AppConfig;
  t: any;
}

const CodeBlock = ({ code }: { code: string }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group mt-2">
      <div className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <button 
          onClick={handleCopy}
          className="p-1.5 bg-slate-700/50 hover:bg-slate-700 text-slate-300 rounded-lg backdrop-blur-sm transition-colors"
        >
          {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
        </button>
      </div>
      <pre className="bg-slate-900 text-slate-50 p-4 rounded-xl text-xs font-mono overflow-x-auto border border-slate-800 shadow-inner">
        <code>{code}</code>
      </pre>
    </div>
  );
};

const EndpointCard = ({ method, path, description, request, response }: any) => {
  const [isOpen, setIsOpen] = useState(false);

  const methodColors: any = {
    POST: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 border-blue-200 dark:border-blue-800',
    GET: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300 border-green-200 dark:border-green-800',
    DELETE: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300 border-red-200 dark:border-red-800',
  };

  return (
    <div className="border border-slate-200 dark:border-slate-700 rounded-2xl bg-white dark:bg-slate-800 overflow-hidden transition-colors duration-300">
      <div 
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-4 p-4 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-700/30 transition-colors"
      >
        <div className={`px-2.5 py-1 rounded-md text-[10px] font-bold border ${methodColors[method]}`}>
          {method}
        </div>
        <div className="font-mono text-sm text-slate-700 dark:text-slate-300 font-medium truncate flex-1">
          {path}
        </div>
        <div className="text-slate-400">
          {isOpen ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
        </div>
      </div>
      
      {isOpen && (
        <div className="px-6 pb-6 pt-2 border-t border-slate-100 dark:border-slate-700">
          <p className="text-slate-600 dark:text-slate-400 text-sm mb-6 mt-2">{description}</p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">Request Body</h4>
              {request ? (
                 <CodeBlock code={JSON.stringify(request, null, 2)} />
              ) : (
                 <div className="text-xs text-slate-400 italic py-4">No request body required</div>
              )}
            </div>
            <div>
              <h4 className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">Response Example</h4>
              <CodeBlock code={JSON.stringify(response, null, 2)} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export const ApiDocs: React.FC<ApiDocsProps> = ({ config, t }) => {
  const endpoints = [
    {
      method: 'GET',
      path: '/config',
      description: 'Retrieves the current system configuration and available parameters.',
      request: null,
      response: {
        version: "0.1.0",
        default_model: "Qwen/Qwen3-Embedding-8B",
        supported_languages: ["en", "zh"],
        features: {
          novelty_check: true,
          verification: true
        }
      }
    },
    {
      method: 'GET',
      path: '/models',
      description: 'Lists all available models supported by the backend LLM service.',
      request: null,
      response: {
        data: [
          { id: "Qwen/Qwen3-Embedding-8B", object: "model", created: 1699999999, owned_by: "system" },
          { id: "deepseek-v3", object: "model", created: 1699999999, owned_by: "system" }
        ],
        object: "list"
      }
    },
    {
      method: 'POST',
      path: config.endpoints.runPipeline,
      description: 'Initiates the research story generation pipeline. This is a long-running process that returns the final result synchronously or a job ID for polling (depending on implementation).',
      request: {
        idea: "Exploring multi-agent reinforcement learning...",
        model: "Qwen/Qwen3-Embedding-8B",
        language: "en",
        options: {
          check_novelty: true,
          verify_logic: false
        }
      },
      response: {
        story: {
          title: "Automated Discovery via Anchored Graphs",
          abstract: "...",
          introduction: "...",
          methodology: "...",
          experiments: "...",
          contributions: ["..."]
        },
        reviews: [
          { criterion: "Novelty", score: 8, reasoning: "..." }
        ],
        logs: []
      }
    },
    {
      method: 'GET',
      path: `${config.endpoints.checkStatus}/{job_id}`,
      description: 'Checks the status of an ongoing pipeline execution. Useful for polling if the run endpoint handles requests asynchronously.',
      request: null,
      response: {
        status: "RUNNING",
        progress: 45,
        current_step: "RETRIEVAL",
        logs: [
           { timestamp: "12:00:01", step: "INIT", message: "Started" }
        ]
      }
    },
    {
      method: 'POST',
      path: config.endpoints.cancelPipeline,
      description: 'Terminates a running pipeline job immediately.',
      request: {
        job_id: "12345-abcde"
      },
      response: {
        success: true,
        message: "Job terminated"
      }
    },
    {
      method: 'POST',
      path: config.endpoints.queryGraph,
      description: 'Queries the local Knowledge Graph (ICLR dataset) for nodes related to specific keywords or concepts.',
      request: {
        query: "Large Language Models",
        limit: 10
      },
      response: {
        nodes: [
            { id: "n1", label: "LLM", type: "concept" }
        ],
        edges: [
            { source: "n1", target: "n2", relation: "related_to" }
        ]
      }
    },
    {
      method: 'POST',
      path: config.endpoints.exportResult,
      description: 'Generates a PDF or formatted JSON export of the generated story.',
      request: {
        format: "pdf",
        story_data: { "...": "..." }
      },
      response: {
        download_url: "https://api.example.com/downloads/paper_123.pdf"
      }
    }
  ];

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
       <div>
          <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight flex items-center gap-3">
             <Globe className="text-violet-500" />
             API Reference
          </h1>
          <p className="text-slate-500 dark:text-slate-400 mt-2 font-medium max-w-3xl">
            Complete documentation for the Idea2Paper backend services. Use these endpoints to integrate the research agent into your own workflows.
          </p>
       </div>

       <div className="space-y-4">
          {endpoints.map((ep, idx) => (
            <EndpointCard key={idx} {...ep} />
          ))}
       </div>
    </div>
  );
};