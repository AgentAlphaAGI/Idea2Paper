import React, { useState } from 'react';
import { AppConfig } from '../types';
import {
  Server,
  Key,
  Database,
  Cpu,
  Settings2,
  ChevronDown,
  ChevronUp,
  Sliders,
  Thermometer,
  ShieldCheck,
  BrainCircuit,
  FileSearch,
  Zap,
  Eye,
  EyeOff
} from 'lucide-react';

interface ConfigPanelProps {
  config: AppConfig;
  setConfig: React.Dispatch<React.SetStateAction<AppConfig>>;
  t: any;
}

const CollapsibleSection = ({ title, icon: Icon, children, defaultOpen = false }: any) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="bg-white dark:bg-slate-800 rounded-3xl shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden transition-colors duration-300">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-6 hover:bg-slate-50 dark:hover:bg-slate-700/30 transition-colors text-left"
      >
        <div className="flex items-center gap-3">
           <div className={`p-2 rounded-xl ${isOpen ? 'bg-violet-100 text-violet-600 dark:bg-violet-900/30 dark:text-violet-400' : 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400'}`}>
              <Icon size={20} />
           </div>
           <span className="font-bold text-slate-800 dark:text-white">{title}</span>
        </div>
        <div className="text-slate-400">
          {isOpen ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </div>
      </button>
      
      {isOpen && (
        <div className="p-6 pt-0 border-t border-slate-100 dark:border-slate-700 animate-in slide-in-from-top-2 duration-200">
          <div className="mt-6 space-y-6">
            {children}
          </div>
        </div>
      )}
    </div>
  );
};

const InputGroup = ({ label, value, onChange, type = "text", placeholder = "", desc = "", required = false }: any) => {
  const [showPassword, setShowPassword] = useState(false);
  const isPassword = type === "password";
  const inputType = isPassword && showPassword ? "text" : type;

  return (
    <div className="space-y-2">
      <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
        {label}
        {required && <span className="text-red-500 ml-1">*</span>}
      </label>
      <div className="relative">
        <input
          type={inputType}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          required={required}
          className={`w-full px-4 py-2.5 ${isPassword ? 'pr-12' : ''} bg-slate-50 dark:bg-slate-900 border ${required && !value ? 'border-red-300 dark:border-red-700' : 'border-slate-200 dark:border-slate-700'} rounded-xl focus:ring-2 focus:ring-violet-500 focus:border-violet-500 outline-none transition-all font-mono text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400`}
        />
        {isPassword && (
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
            title={showPassword ? "Hide password" : "Show password"}
          >
            {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
          </button>
        )}
      </div>
      {required && !value && <p className="text-xs text-red-500 dark:text-red-400">This field is required</p>}
      {desc && <p className="text-xs text-slate-400 dark:text-slate-500">{desc}</p>}
    </div>
  );
};

const ToggleGroup = ({ label, checked, onChange, desc = "" }: any) => (
  <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-100 dark:border-slate-800">
    <div className="pr-4">
      <div className="font-bold text-slate-700 dark:text-slate-300 text-sm">{label}</div>
      {desc && <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{desc}</div>}
    </div>
    <button 
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-violet-500 focus:ring-offset-2 dark:focus:ring-offset-slate-800 ${checked ? 'bg-violet-600' : 'bg-slate-300 dark:bg-slate-600'}`}
    >
      <span className={`inline-block h-4 w-4 transform rounded-full bg-white shadow-md transition-transform ${checked ? 'translate-x-6' : 'translate-x-1'}`} />
    </button>
  </div>
);

export const ConfigPanel: React.FC<ConfigPanelProps> = ({ config, setConfig, t }) => {
  const [saveStatus, setSaveStatus] = React.useState<'idle' | 'saving' | 'saved'>('idle');

  // Show save indicator when config changes
  React.useEffect(() => {
    setSaveStatus('saving');
    const timer = setTimeout(() => {
      setSaveStatus('saved');
      setTimeout(() => setSaveStatus('idle'), 2000);
    }, 500);
    return () => clearTimeout(timer);
  }, [config]);

  const handleResetConfig = () => {
    if (confirm('Are you sure you want to reset all configuration to defaults? This will clear your API keys and all settings.')) {
      localStorage.removeItem('idea2paper_config');
      window.location.reload();
    }
  };

  return (
    <div className="space-y-6 max-w-3xl pb-20">

      {/* Reset Button and Save Status */}
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          {saveStatus === 'saved' && (
            <span className="text-sm text-green-600 dark:text-green-400 flex items-center gap-1">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Configuration saved
            </span>
          )}
        </div>
        <button
          onClick={handleResetConfig}
          className="px-4 py-2 text-sm font-medium text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 border border-red-300 dark:border-red-700 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
        >
          Reset to Defaults
        </button>
      </div>

      {/* 1. LLM Configuration (Primary - High Priority) */}
      <CollapsibleSection title={t.config.headers.llm} icon={Cpu} defaultOpen={true}>
         <InputGroup
            label={t.config.labels.silicon_key}
            value={config.siliconFlowApiKey}
            type="password"
            onChange={(e: any) => setConfig({...config, siliconFlowApiKey: e.target.value})}
            desc={t.config.descriptions.silicon_key}
            required={true}
         />
         <InputGroup
            label={t.config.labels.llm_url}
            value={config.llmUrl}
            onChange={(e: any) => setConfig({...config, llmUrl: e.target.value})}
            required={true}
         />
         <InputGroup
            label={t.config.labels.llm_model}
            value={config.llmModel}
            onChange={(e: any) => setConfig({...config, llmModel: e.target.value})}
            required={true}
         />

         <div className="bg-slate-50 dark:bg-slate-900 rounded-xl p-4 border border-slate-100 dark:border-slate-800">
             <div className="flex items-center gap-2 mb-4 text-slate-700 dark:text-slate-300 font-bold text-xs uppercase tracking-wider">
                <Thermometer size={14} />
                {t.config.headers.temperatures}
             </div>
             <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                 {Object.entries(config.llmTemperatures).map(([key, val]) => (
                    <div key={key}>
                        <label className="block text-[10px] text-slate-500 dark:text-slate-400 mb-1 truncate" title={key}>{key}</label>
                        <input
                            type="number"
                            step="0.1"
                            min="0"
                            max="2"
                            value={val}
                            onChange={(e) => setConfig({
                                ...config,
                                llmTemperatures: {
                                    ...config.llmTemperatures,
                                    [key]: parseFloat(e.target.value)
                                }
                            })}
                            className="w-full px-2 py-1 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded text-xs text-center font-mono focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none"
                        />
                    </div>
                 ))}
             </div>
         </div>
      </CollapsibleSection>

      {/* 2. Embedding Configuration (Primary - High Priority) */}
      <CollapsibleSection title={t.config.headers.embedding} icon={Database} defaultOpen={true}>
         <InputGroup
            label={t.config.labels.embed_url}
            value={config.embeddingUrl}
            onChange={(e: any) => setConfig({...config, embeddingUrl: e.target.value})}
            required={true}
         />
         <InputGroup
            label={t.config.labels.embed_model}
            value={config.embeddingModel}
            onChange={(e: any) => setConfig({...config, embeddingModel: e.target.value})}
            required={true}
         />
         <InputGroup
            label="Embedding API Key"
            value={config.embeddingApiKey}
            type="password"
            onChange={(e: any) => setConfig({...config, embeddingApiKey: e.target.value})}
            placeholder="Enter your Embedding API Key"
            required={true}
            desc="API Key for the embedding service"
         />
      </CollapsibleSection>

      {/* 3. Pipeline Control */}
      <CollapsibleSection title={t.config.headers.pipeline} icon={Sliders} defaultOpen={false}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
             <ToggleGroup
               label={t.config.labels.enable_logging}
               checked={config.logging.enable}
               onChange={(val: boolean) => setConfig({...config, logging: {...config.logging, enable: val}})}
             />
             <ToggleGroup
               label={t.config.labels.enable_results}
               checked={config.results.enable}
               onChange={(val: boolean) => setConfig({...config, results: {...config.results, enable: val}})}
             />
          </div>
          <ToggleGroup
            label={t.config.labels.novelty_check}
            checked={config.novelty.enable}
            onChange={(val: boolean) => setConfig({...config, novelty: {...config.novelty, enable: val}})}
            desc={t.config.descriptions.novelty}
          />
           <ToggleGroup
            label={t.config.labels.verification}
            checked={config.verification.enable}
            onChange={(val: boolean) => setConfig({...config, verification: {...config.verification, enable: val}})}
            desc={t.config.descriptions.verification}
          />
      </CollapsibleSection>

      {/* 4. Advanced Logic (Collapsed by default) */}
      <CollapsibleSection title={t.config.headers.advanced} icon={Settings2}>
          {/* Idea Packaging */}
          <div className="space-y-4 border-b border-slate-100 dark:border-slate-700 pb-6">
              <div className="flex items-center gap-2 text-violet-600 dark:text-violet-400 font-bold text-sm">
                  <BrainCircuit size={16} />
                  Idea Packaging
              </div>
              <ToggleGroup 
                label={t.config.labels.idea_packaging} 
                checked={config.ideaPackaging.enable} 
                onChange={(val: boolean) => setConfig({...config, ideaPackaging: {...config.ideaPackaging, enable: val}})}
                desc={t.config.descriptions.idea_packaging}
              />
              {config.ideaPackaging.enable && (
                  <div className="grid grid-cols-2 gap-4 pl-4 border-l-2 border-violet-100 dark:border-violet-900">
                      <InputGroup 
                        label="Top N Patterns" 
                        value={config.ideaPackaging.topNPatterns} 
                        type="number"
                        onChange={(e: any) => setConfig({...config, ideaPackaging: {...config.ideaPackaging, topNPatterns: parseInt(e.target.value)}})} 
                      />
                      <InputGroup 
                        label="Candidate K" 
                        value={config.ideaPackaging.candidateK} 
                        type="number"
                        onChange={(e: any) => setConfig({...config, ideaPackaging: {...config.ideaPackaging, candidateK: parseInt(e.target.value)}})} 
                      />
                  </div>
              )}
          </div>

          {/* Critic */}
          <div className="space-y-4 border-b border-slate-100 dark:border-slate-700 pb-6 mt-6">
              <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400 font-bold text-sm">
                  <ShieldCheck size={16} />
                  Critic Settings
              </div>
               <ToggleGroup 
                label={t.config.labels.critic_strict} 
                checked={config.critic.strictJson} 
                onChange={(val: boolean) => setConfig({...config, critic: {...config.critic, strictJson: val}})}
                desc={t.config.descriptions.critic}
              />
              <InputGroup 
                label="JSON Retries" 
                value={config.critic.jsonRetries} 
                type="number"
                onChange={(e: any) => setConfig({...config, critic: {...config.critic, jsonRetries: parseInt(e.target.value)}})} 
              />
          </div>

          {/* Novelty Extended */}
           <div className="space-y-4 border-b border-slate-100 dark:border-slate-700 pb-6 mt-6">
              <div className="flex items-center gap-2 text-blue-600 dark:text-blue-400 font-bold text-sm">
                  <FileSearch size={16} />
                  Novelty Extended
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <ToggleGroup 
                    label={t.config.labels.auto_index} 
                    checked={config.novelty.autoBuildIndex} 
                    onChange={(val: boolean) => setConfig({...config, novelty: {...config.novelty, autoBuildIndex: val}})}
                  />
                  <div className="space-y-2">
                     <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">{t.config.labels.novelty_action}</label>
                     <select 
                        value={config.novelty.action}
                        onChange={(e: any) => setConfig({...config, novelty: {...config.novelty, action: e.target.value}})}
                        className="w-full px-4 py-2.5 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl focus:ring-2 focus:ring-violet-500 outline-none text-sm text-slate-900 dark:text-slate-100"
                     >
                        <option value="report_only">Report Only</option>
                        <option value="pivot">Pivot (Retry)</option>
                        <option value="fail">Fail</option>
                     </select>
                  </div>
              </div>
           </div>

           {/* Misc */}
           <div className="mt-6 pt-2">
               <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400 font-bold text-sm mb-4">
                  <Zap size={16} />
                  Misc Thresholds
              </div>
               <div className="grid grid-cols-2 gap-4">
                  <InputGroup 
                    label={t.config.labels.collision_th} 
                    value={config.verification.collisionThreshold} 
                    type="number"
                    step="0.01"
                    onChange={(e: any) => setConfig({...config, verification: {...config.verification, collisionThreshold: parseFloat(e.target.value)}})} 
                  />
                  <InputGroup 
                    label="Recall Audit Top N" 
                    value={config.recall.auditTopN} 
                    type="number"
                    onChange={(e: any) => setConfig({...config, recall: {...config.recall, auditTopN: parseInt(e.target.value)}})} 
                  />
               </div>
           </div>

      </CollapsibleSection>

    </div>
  );
};