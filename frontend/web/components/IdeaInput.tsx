import React, { useState } from 'react';
import { Sparkles, ArrowRight, Lightbulb } from 'lucide-react';
import { Language } from '../types';

interface IdeaInputProps {
  onSubmit: (idea: string) => void;
  isLoading: boolean;
  t: any;
  lang: Language;
}

export const IdeaInput: React.FC<IdeaInputProps> = ({ onSubmit, isLoading, t, lang }) => {
  const [idea, setIdea] = useState('');

  const examples = lang === 'en' ? [
    "A multi-agent system for automated code refactoring using AST parsing",
    "Improving diffusion models for temporal consistency in video generation",
    "Reinforcement learning for optimizing supply chain logistics"
  ] : [
    "基于AST解析的自动化代码重构多智能体系统",
    "改进扩散模型以提高视频生成的时间一致性",
    "用于优化供应链物流的强化学习方法"
  ];

  return (
    <div className="bg-white dark:bg-slate-800 rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border border-slate-100 dark:border-slate-700 overflow-hidden group hover:border-violet-200 dark:hover:border-violet-800 transition-colors duration-300">
      <div className="p-8 md:p-10 bg-gradient-to-br from-violet-600 via-indigo-600 to-purple-700 text-white relative overflow-hidden">
        {/* Decorative background circles */}
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-64 h-64 bg-white/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-0 -ml-10 -mb-10 w-40 h-40 bg-indigo-500/30 rounded-full blur-2xl"></div>
        
        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-3 text-indigo-100">
             <Lightbulb size={24} className="text-yellow-300 fill-yellow-300/20" />
             <span className="text-sm font-bold tracking-wider uppercase opacity-80">Idea Phase</span>
          </div>
          <h2 className="text-3xl md:text-4xl font-extrabold mb-4 tracking-tight leading-tight">{t.input.title}</h2>
          <p className="text-indigo-100/90 text-lg max-w-2xl leading-relaxed font-light">{t.input.subtitle}</p>
        </div>
      </div>
      
      <div className="p-8 md:p-10">
        <label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-3 uppercase tracking-wide">{t.input.label}</label>
        <div className="relative">
          <textarea
            value={idea}
            onChange={(e) => setIdea(e.target.value)}
            disabled={isLoading}
            className="w-full h-44 p-6 border border-slate-200 dark:border-slate-600 rounded-2xl focus:ring-4 focus:ring-violet-100 dark:focus:ring-violet-900/30 focus:border-violet-400 outline-none resize-none text-slate-700 dark:text-slate-200 placeholder-slate-400 dark:placeholder-slate-500 text-lg transition-all shadow-sm bg-slate-50/30 dark:bg-slate-900/50"
            placeholder={t.input.placeholder}
          />
        </div>

        <div className="mt-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="flex-1 w-full md:w-auto">
            <span className="text-xs text-slate-500 dark:text-slate-400 font-bold uppercase tracking-wider block mb-3 flex items-center gap-2">
              <Sparkles size={12} className="text-violet-400" />
              {t.input.try_example}
            </span>
            <div className="flex flex-wrap gap-2">
              {examples.map((ex, i) => (
                <button 
                  key={i}
                  onClick={() => setIdea(ex)}
                  disabled={isLoading}
                  className="text-xs px-4 py-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 font-medium rounded-full transition-colors truncate max-w-[240px] border border-slate-200 dark:border-slate-600"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={() => {
              if (idea.trim()) onSubmit(idea);
            }}
            disabled={!idea.trim() || isLoading}
            className={`
              w-full md:w-auto flex items-center justify-center gap-3 px-8 py-4 rounded-xl font-bold text-white shadow-lg shadow-violet-200 dark:shadow-none transition-all transform
              ${!idea.trim() || isLoading 
                ? 'bg-slate-300 dark:bg-slate-700 cursor-not-allowed transform-none shadow-none' 
                : 'bg-violet-600 hover:bg-violet-700 hover:shadow-violet-300 hover:-translate-y-1 active:scale-95'
              }
            `}
          >
            {isLoading ? (
              <span className="flex items-center gap-2">
                 <span className="w-2 h-2 bg-white rounded-full animate-bounce"></span>
                 <span className="w-2 h-2 bg-white rounded-full animate-bounce delay-100"></span>
                 <span className="w-2 h-2 bg-white rounded-full animate-bounce delay-200"></span>
                 <span className="ml-2">{t.input.btn_running}</span>
              </span>
            ) : (
              <>
                <Sparkles size={20} className="animate-pulse" />
                {t.input.btn_generate}
                <ArrowRight size={20} />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};