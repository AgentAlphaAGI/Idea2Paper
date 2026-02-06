import React, { useState } from 'react';
import { FileText, Settings, Menu, X, Database, Activity, Sun, Moon } from 'lucide-react';
import { Language } from '../types';

interface LayoutProps {
  children: React.ReactNode;
  activeTab: string;
  onTabChange: (tab: string) => void;
  lang: Language;
  setLang: (lang: Language) => void;
  t: any;
  theme: 'light' | 'dark';
  onThemeToggle: () => void;
}

// Custom Icons for Socials
const WeChatIcon = ({ size = 20 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 1024 1024" fill="currentColor" className="transform transition-transform hover:scale-110">
    <path d="M664.250054 368.541681c10.015098 0 19.892049 0.732687 29.67281 1.795902-26.647917-122.810047-159.358451-214.077703-310.826188-214.077703-169.353083 0-308.085774 114.232694-308.085774 259.274068 0 83.708494 46.165436 152.460344 123.281791 205.78483l-30.80868 91.730191 107.688651-53.455469c38.558178 7.53665 69.459978 15.308661 107.924012 15.308661 9.66308 0 19.230993-0.470721 28.752858-1.225921-6.025227-20.36584-9.521864-41.723264-9.521864-63.862493C402.328693 476.632491 517.908058 368.541681 664.250054 368.541681zM498.62897 285.87389c23.200398 0 38.557154 15.120372 38.557154 38.061874 0 22.846334-15.356756 38.156018-38.557154 38.156018-23.107277 0-46.260603-15.309684-46.260603-38.156018C452.368366 300.994262 475.522716 285.87389 498.62897 285.87389zM283.016307 362.090758c-23.107277 0-46.402843-15.309684-46.402843-38.156018 0-22.941502 23.295566-38.061874 46.402843-38.061874 23.081695 0 38.46301 15.120372 38.46301 38.061874C321.479317 346.782098 306.098002 362.090758 283.016307 362.090758zM945.448458 606.151333c0-121.888048-123.258255-221.236753-261.683954-221.236753-146.57838 0-262.015505 99.348706-262.015505 221.236753 0 122.06508 115.437126 221.200938 262.015505 221.200938 30.66644 0 61.617359-7.609305 92.423993-15.262612l84.513836 45.786813-23.178909-76.17082C899.379213 735.776599 945.448458 674.90216 945.448458 606.151333zM598.803483 567.994292c-15.332197 0-30.807656-15.096836-30.807656-30.501688 0-15.190981 15.47546-30.477129 30.807656-30.477129 23.295566 0 38.558178 15.286148 38.558178 30.477129C637.361661 552.897456 622.099049 567.994292 598.803483 567.994292zM768.25071 567.994292c-15.213493 0-30.594809-15.096836-30.594809-30.501688 0-15.190981 15.381315-30.477129 30.594809-30.477129 23.107277 0 38.558178 15.286148 38.558178 30.477129C806.808888 552.897456 791.357987 567.994292 768.25071 567.994292z"/>
  </svg>
);

const DiscordIcon = ({ size = 20 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 1280 1024" fill="currentColor" className="transform transition-transform hover:scale-110">
    <path d="M1049.062 139.672a3 3 0 0 0-1.528-1.4A970.13 970.13 0 0 0 808.162 64.06a3.632 3.632 0 0 0-3.846 1.82 674.922 674.922 0 0 0-29.8 61.2 895.696 895.696 0 0 0-268.852 0 619.082 619.082 0 0 0-30.27-61.2 3.78 3.78 0 0 0-3.848-1.82 967.378 967.378 0 0 0-239.376 74.214 3.424 3.424 0 0 0-1.576 1.352C78.136 367.302 36.372 589.38 56.86 808.708a4.032 4.032 0 0 0 1.53 2.75 975.332 975.332 0 0 0 293.65 148.378 3.8 3.8 0 0 0 4.126-1.352A696.4 696.4 0 0 0 416.24 860.8a3.72 3.72 0 0 0-2.038-5.176 642.346 642.346 0 0 1-91.736-43.706 3.77 3.77 0 0 1-0.37-6.252 502.094 502.094 0 0 0 18.218-14.274 3.638 3.638 0 0 1 3.8-0.512c192.458 87.834 400.82 87.834 591 0a3.624 3.624 0 0 1 3.848 0.466 469.066 469.066 0 0 0 18.264 14.32 3.768 3.768 0 0 1-0.324 6.252 602.814 602.814 0 0 1-91.78 43.66 3.75 3.75 0 0 0-2 5.222 782.11 782.11 0 0 0 60.028 97.63 3.728 3.728 0 0 0 4.126 1.4A972.096 972.096 0 0 0 1221.4 811.458a3.764 3.764 0 0 0 1.53-2.704c24.528-253.566-41.064-473.824-173.868-669.082zM444.982 675.16c-57.944 0-105.688-53.174-105.688-118.478s46.818-118.482 105.688-118.482c59.33 0 106.612 53.64 105.686 118.478 0 65.308-46.82 118.482-105.686 118.482z m390.76 0c-57.942 0-105.686-53.174-105.686-118.478s46.818-118.482 105.686-118.482c59.334 0 106.614 53.64 105.688 118.478 0 65.308-46.354 118.482-105.688 118.482z"/>
  </svg>
);

const GithubIcon = ({ size = 20 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 1024 1024" fill="currentColor" className="transform transition-transform hover:scale-110">
    <path d="M511.6 76.3C264.3 76.2 64 276.4 64 523.5 64 718.9 189.3 885 363.8 946c23.5 5.9 19.9-10.8 19.9-22.2v-77.5c-135.7 15.9-141.2-73.9-150.3-88.9C215 726 171.5 718 184.5 703c30.9-15.9 62.4 4 98.9 57.9 26.4 39.1 77.9 32.5 104 26 5.7-23.5 17.9-44.5 34.7-60.8-140.6-25.2-199.2-111-199.2-213 0-49.5 16.3-95 48.3-131.7-20.4-60.5 1.9-112.3 4.9-120 58.1-5.2 118.5 41.6 123.2 45.3 33-8.9 70.7-13.6 112.9-13.6 42.4 0 80.2 4.9 113.5 13.9 11.3-8.6 67.3-48.8 121.3-43.9 2.9 7.7 24.7 58.3 5.5 118 32.4 36.8 48.9 82.7 48.9 132.3 0 102.2-59 188.1-200 212.9 23.5 23.2 38.1 55.4 38.1 91v112.5c0.8 9 0 17.9 15 17.9 177.1-59.7 304.6-227 304.6-424.1 0-247.2-200.4-447.3-447.5-447.3z"/>
  </svg>
);

const HuggingFaceIcon = ({ size = 20 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 95 88" fill="none" className="transform transition-transform hover:scale-110">
    <path fill="#FFD21E" d="M47.21 76.5a34.75 34.75 0 1 0 0-69.5 34.75 34.75 0 0 0 0 69.5Z" />
    <path fill="#FF9D0B" d="M81.96 41.75a34.75 34.75 0 1 0-69.5 0 34.75 34.75 0 0 0 69.5 0Zm-73.5 0a38.75 38.75 0 1 1 77.5 0 38.75 38.75 0 0 1-77.5 0Z" />
    <path fill="#3A3B45" d="M58.5 32.3c1.28.44 1.78 3.06 3.07 2.38a5 5 0 1 0-6.76-2.07c.61 1.15 2.55-.72 3.7-.32ZM34.95 32.3c-1.28.44-1.79 3.06-3.07 2.38a5 5 0 1 1 6.76-2.07c-.61 1.15-2.56-.72-3.7-.32Z" />
    <path fill="#FF323D" d="M46.96 56.29c9.83 0 13-8.76 13-13.26 0-2.34-1.57-1.6-4.09-.36-2.33 1.15-5.46 2.74-8.9 2.74-7.19 0-13-6.88-13-2.38s3.16 13.26 13 13.26Z" />
    <path fill="#3A3B45" fillRule="evenodd" d="M39.43 54a8.7 8.7 0 0 1 5.3-4.49c.4-.12.81.57 1.24 1.28.4.68.82 1.37 1.24 1.37.45 0 .9-.68 1.33-1.35.45-.7.89-1.38 1.32-1.25a8.61 8.61 0 0 1 5 4.17c3.73-2.94 5.1-7.74 5.1-10.7 0-2.34-1.57-1.6-4.09-.36l-.14.07c-2.31 1.15-5.39 2.67-8.77 2.67s-6.45-1.52-8.77-2.67c-2.6-1.29-4.23-2.1-4.23.29 0 3.05 1.46 8.06 5.47 10.97Z" clipRule="evenodd" />
    <path fill="#FF9D0B" d="M70.71 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM24.21 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM17.52 48c-1.62 0-3.06.66-4.07 1.87a5.97 5.97 0 0 0-1.33 3.76 7.1 7.1 0 0 0-1.94-.3c-1.55 0-2.95.59-3.94 1.66a5.8 5.8 0 0 0-.8 7 5.3 5.3 0 0 0-1.79 2.82c-.24.9-.48 2.8.8 4.74a5.22 5.22 0 0 0-.37 5.02c1.02 2.32 3.57 4.14 8.52 6.1 3.07 1.22 5.89 2 5.91 2.01a44.33 44.33 0 0 0 10.93 1.6c5.86 0 10.05-1.8 12.46-5.34 3.88-5.69 3.33-10.9-1.7-15.92-2.77-2.78-4.62-6.87-5-7.77-.78-2.66-2.84-5.62-6.25-5.62a5.7 5.7 0 0 0-4.6 2.46c-1-1.26-1.98-2.25-2.86-2.82A7.4 7.4 0 0 0 17.52 48Zm0 4c.51 0 1.14.22 1.82.65 2.14 1.36 6.25 8.43 7.76 11.18.5.92 1.37 1.31 2.14 1.31 1.55 0 2.75-1.53.15-3.48-3.92-2.93-2.55-7.72-.68-8.01.08-.02.17-.02.24-.02 1.7 0 2.45 2.93 2.45 2.93s2.2 5.52 5.98 9.3c3.77 3.77 3.97 6.8 1.22 10.83-1.88 2.75-5.47 3.58-9.16 3.58-3.81 0-7.73-.9-9.92-1.46-.11-.03-13.45-3.8-11.76-7 .28-.54.75-.76 1.34-.76 2.38 0 6.7 3.54 8.57 3.54.41 0 .7-.17.83-.6.79-2.85-12.06-4.05-10.98-8.17.2-.73.71-1.02 1.44-1.02 3.14 0 10.2 5.53 11.68 5.53.11 0 .2-.03.24-.1.74-1.2.33-2.04-4.9-5.2-5.21-3.16-8.88-5.06-6.8-7.33.24-.26.58-.38 1-.38 3.17 0 10.66 6.82 10.66 6.82s2.02 2.1 3.25 2.1c.28 0 .52-.1.68-.38.86-1.46-8.06-8.22-8.56-11.01-.34-1.9.24-2.85 1.31-2.85Z" />
    <path fill="#FFD21E" d="M38.6 76.69c2.75-4.04 2.55-7.07-1.22-10.84-3.78-3.77-5.98-9.3-5.98-9.3s-.82-3.2-2.69-2.9c-1.87.3-3.24 5.08.68 8.01 3.91 2.93-.78 4.92-2.29 2.17-1.5-2.75-5.62-9.82-7.76-11.18-2.13-1.35-3.63-.6-3.13 2.2.5 2.79 9.43 9.55 8.56 11-.87 1.47-3.93-1.71-3.93-1.71s-9.57-8.71-11.66-6.44c-2.08 2.27 1.59 4.17 6.8 7.33 5.23 3.16 5.64 4 4.9 5.2-.75 1.2-12.28-8.53-13.36-4.4-1.08 4.11 11.77 5.3 10.98 8.15-.8 2.85-9.06-5.38-10.74-2.18-1.7 3.21 11.65 6.98 11.76 7.01 4.3 1.12 15.25 3.49 19.08-2.12Z" />
    <path fill="#FF9D0B" d="M77.4 48c1.62 0 3.07.66 4.07 1.87a5.97 5.97 0 0 1 1.33 3.76 7.1 7.1 0 0 1 1.95-.3c1.55 0 2.95.59 3.94 1.66a5.8 5.8 0 0 1 .8 7 5.3 5.3 0 0 1 1.78 2.82c.24.9.48 2.8-.8 4.74a5.22 5.22 0 0 1 .37 5.02c-1.02 2.32-3.57 4.14-8.51 6.1-3.08 1.22-5.9 2-5.92 2.01a44.33 44.33 0 0 1-10.93 1.6c-5.86 0-10.05-1.8-12.46-5.34-3.88-5.69-3.33-10.9 1.7-15.92 2.78-2.78 4.63-6.87 5.01-7.77.78-2.66 2.83-5.62 6.24-5.62a5.7 5.7 0 0 1 4.6 2.46c1-1.26 1.98-2.25 2.87-2.82A7.4 7.4 0 0 1 77.4 48Zm0 4c-.51 0-1.13.22-1.82.65-2.13 1.36-6.25 8.43-7.76 11.18a2.43 2.43 0 0 1-2.14 1.31c-1.54 0-2.75-1.53-.14-3.48 3.91-2.93 2.54-7.72.67-8.01a1.54 1.54 0 0 0-.24-.02c-1.7 0-2.45 2.93-2.45 2.93s-2.2 5.52-5.97 9.3c-3.78 3.77-3.98 6.8-1.22 10.83 1.87 2.75 5.47 3.58 9.15 3.58 3.82 0 7.73-.9 9.93-1.46.1-.03 13.45-3.8 11.76-7-.29-.54-.75-.76-1.34-.76-2.38 0-6.71 3.54-8.57 3.54-.42 0-.71-.17-.83-.6-.8-2.85 12.05-4.05 10.97-8.17-.19-.73-.7-1.02-1.44-1.02-3.14 0-10.2 5.53-11.68 5.53-.1 0-.19-.03-.23-.1-.74-1.2-.34-2.04 4.88-5.2 5.23-3.16 8.9-5.06 6.8-7.33-.23-.26-.57-.38-.98-.38-3.18 0-10.67 6.82-10.67 6.82s-2.02 2.1-3.24 2.1a.74.74 0 0 1-.68-.38c-.87-1.46 8.05-8.22 8.55-11.01.34-1.9-.24-2.85-1.31-2.85Z" />
    <path fill="#FFD21E" d="M56.33 76.69c-2.75-4.04-2.56-7.07 1.22-10.84 3.77-3.77 5.97-9.3 5.97-9.3s.82-3.2 2.7-2.9c1.86.3 3.23 5.08-.68 8.01-3.92 2.93.78 4.92 2.28 2.17 1.51-2.75 5.63-9.82 7.76-11.18 2.13-1.35 3.64-.6 3.13 2.2-.5 2.79-9.42 9.55-8.55 11 .86 1.47 3.92-1.71 3.92-1.71s9.58-8.71 11.66-6.44c2.08 2.27-1.58 4.17-6.8 7.33-5.23 3.16-5.63 4-4.9 5.2.75 1.2 12.28-8.53 13.36-4.4 1.08 4.11-11.76 5.3-10.97 8.15.8 2.85 9.05-5.38 10.74-2.18 1.69 3.21-11.65 6.98-11.76 7.01-4.31 1.12-15.26 3.49-19.08-2.12Z" />
  </svg>
);


export const Layout: React.FC<LayoutProps> = ({ children, activeTab, onTabChange, lang, setLang, t, theme, onThemeToggle }) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [showWechatQR, setShowWechatQR] = useState(false);

  const navItems = [
    { id: 'dashboard', label: t.nav.dashboard, icon: Activity },
    { id: 'results', label: t.nav.results, icon: FileText },
    // { id: 'kg', label: t.nav.kg, icon: Database },  // Hidden for now
    { id: 'settings', label: t.nav.settings, icon: Settings },
  ];

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50 dark:bg-slate-950 font-sans selection:bg-violet-100 selection:text-violet-900 transition-colors duration-300">
      {/* Mobile Sidebar Overlay */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 z-20 bg-black/20 backdrop-blur-sm lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* WeChat QR Modal */}
      {showWechatQR && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm" onClick={() => setShowWechatQR(false)}>
           <div className="bg-white dark:bg-slate-800 p-6 rounded-3xl shadow-2xl max-w-sm w-full text-center animate-in zoom-in-95 duration-200" onClick={e => e.stopPropagation()}>
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-2">{t.social.wechat_group}</h3>
              <p className="text-slate-500 dark:text-slate-400 mb-4 text-sm">{t.social.scan_qr}</p>
           <div className="bg-white dark:bg-slate-700 w-48 h-48 mx-auto rounded-xl flex items-center justify-center mb-4 overflow-hidden">
                 <img
                   src="https://github.com/AgentAlphaAGI/Idea2Paper/raw/main/assets/images/idea2paper_code.png"
                   alt="WeChat QR Code"
                   className="w-full h-full object-contain"
                 />
              </div>
              <button onClick={() => setShowWechatQR(false)} className="px-6 py-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 rounded-full text-slate-600 dark:text-slate-300 font-medium transition-colors">
                {t.common.close}
              </button>
           </div>
        </div>
      )}

      {/* Modern Sidebar */}
      <aside className={`
        fixed inset-y-0 left-0 z-30 w-72 bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl border-r border-slate-200/60 dark:border-slate-800/60 transform transition-transform duration-300 ease-[cubic-bezier(0.2,0.8,0.2,1)] lg:translate-x-0 lg:static lg:inset-auto
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} shadow-[4px_0_24px_-12px_rgba(0,0,0,0.1)]
      `}>
        <div className="flex flex-col h-full p-6">
          {/* Logo Area */}
          <div className="mb-10 relative">
             {/* Mobile Close Button - Absolute to avoid layout shifting */}
             <button 
               onClick={() => setIsSidebarOpen(false)} 
               className="lg:hidden absolute top-0 right-0 p-2 text-slate-400 hover:text-slate-600 transition-colors rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800"
             >
               <X size={20} />
             </button>

            <div className="flex flex-col gap-0.5">
               {/* Organization Logo - Width 100% */}
               <div className="w-full">
                 <img 
                   src="https://agentalpha.top/logo-light.png" 
                   alt="Agent Alpha" 
                   className="w-full h-auto object-contain dark:hidden"
                 />
                 {/* Dark mode logo - reusing same image but inverted for better visibility on dark backgrounds if needed */}
                  <img 
                   src="https://agentalpha.top/logo-light.png" 
                   alt="Agent Alpha" 
                   className="w-full h-auto object-contain hidden dark:block brightness-0 invert"
                 />
               </div>
               
               {/* Product Title Group */}
               <div className="space-y-1 text-center">
                   <h1 className="font-extrabold text-2xl text-slate-900 dark:text-white tracking-tight leading-none">Idea2Paper</h1>
                   <div className="flex items-center justify-center gap-2">
                      <div className="h-1.5 w-1.5 rounded-full bg-violet-500"></div>
                      <span className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                        Research Agent
                      </span>
                   </div>
               </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-2">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => {
                  onTabChange(item.id);
                  setIsSidebarOpen(false);
                }}
                className={`group flex items-center w-full gap-4 px-4 py-3.5 rounded-2xl transition-all duration-200 font-medium text-sm
                  ${activeTab === item.id 
                    ? 'bg-violet-50 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 shadow-sm ring-1 ring-violet-100 dark:ring-violet-900' 
                    : 'text-slate-500 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-200'
                  }`}
              >
                <item.icon size={20} className={`transition-colors ${activeTab === item.id ? 'text-violet-600 dark:text-violet-400' : 'text-slate-400 dark:text-slate-500 group-hover:text-slate-600 dark:group-hover:text-slate-300'}`} />
                <span>{item.label}</span>
                {activeTab === item.id && <div className="ml-auto w-1.5 h-1.5 rounded-full bg-violet-600 dark:bg-violet-400"></div>}
              </button>
            ))}
          </nav>

          {/* Footer Area */}
          <div className="mt-auto space-y-6">
            
            {/* Language Switcher */}
            <div className="bg-slate-50 dark:bg-slate-800 p-1.5 rounded-xl flex items-center relative border border-slate-100 dark:border-slate-700">
               <div className={`absolute inset-y-1.5 w-1/2 bg-white dark:bg-slate-600 rounded-lg shadow-sm border border-slate-200 dark:border-slate-500 transition-transform duration-300 ease-out ${lang === 'zh' ? 'translate-x-full' : 'translate-x-0'}`}></div>
               <button onClick={() => setLang('en')} className={`relative z-10 w-1/2 py-1.5 text-xs font-bold text-center transition-colors ${lang === 'en' ? 'text-slate-800 dark:text-white' : 'text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300'}`}>
                 English
               </button>
               <button onClick={() => setLang('zh')} className={`relative z-10 w-1/2 py-1.5 text-xs font-bold text-center transition-colors ${lang === 'zh' ? 'text-slate-800 dark:text-white' : 'text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300'}`}>
                 中文
               </button>
            </div>

            {/* Social Links */}
            <div className="flex items-center justify-between px-2 pt-4 border-t border-slate-100 dark:border-slate-800">
              <div className="flex gap-3">
                 {/* WeChat */}
                 <button
                  onClick={() => setShowWechatQR(true)}
                  className="text-green-600 dark:text-green-400 hover:text-green-700 dark:hover:text-green-300 transition-colors"
                  title="WeChat Group"
                 >
                    <WeChatIcon size={20} />
                 </button>

                 {/* Discord */}
                 <a
                  href="https://discord.gg/FfXtbREb"
                  target="_blank"
                  rel="noreferrer"
                  className="text-indigo-500 dark:text-indigo-400 hover:text-indigo-600 dark:hover:text-indigo-300 transition-colors"
                  title="Join Discord"
                 >
                    <DiscordIcon size={20} />
                 </a>

                 {/* GitHub */}
                 <a
                  href="https://github.com/AgentAlphaAGI/Idea2Paper"
                  target="_blank"
                  rel="noreferrer"
                  className="text-slate-700 dark:text-slate-300 hover:text-slate-900 dark:hover:text-slate-100 transition-colors"
                  title="GitHub Repository"
                 >
                    <GithubIcon size={20} />
                 </a>

                 {/* Hugging Face */}
                 <a
                  href="https://huggingface.co/datasets/AgentAlphaAGI/Paper-Review-Dataset"
                  target="_blank"
                  rel="noreferrer"
                  className="text-yellow-500 dark:text-yellow-400 hover:text-yellow-600 dark:hover:text-yellow-300 transition-colors"
                  title="Hugging Face Dataset"
                 >
                    <HuggingFaceIcon size={20} />
                 </a>
              </div>
            </div>
            
            <p className="text-[10px] text-slate-400 dark:text-slate-600 text-center font-medium">
              v0.1.0-alpha • Built with ❤️
            </p>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden bg-slate-50/50 dark:bg-slate-950/50 transition-colors duration-300 relative">
        {/* Mobile Header */}
        <header className="lg:hidden flex items-center justify-between p-4 bg-white/80 dark:bg-slate-900/80 backdrop-blur border-b border-slate-200 dark:border-slate-800 sticky top-0 z-10 transition-colors duration-300">
          <button onClick={() => setIsSidebarOpen(true)} className="text-slate-600 dark:text-slate-300">
            <Menu size={24} />
          </button>
          <span className="font-bold text-slate-800 dark:text-slate-100">Idea2Paper</span>
          <button onClick={onThemeToggle} className="text-slate-600 dark:text-slate-300 p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            {theme === 'light' ? <Sun size={24} /> : <Moon size={24} />}
          </button>
        </header>

        {/* Desktop Theme Toggle - Absolute Top Right */}
        <div className="hidden lg:block absolute top-6 right-8 z-20">
             <button 
                onClick={onThemeToggle} 
                className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm p-3 rounded-full shadow-sm border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:text-violet-600 dark:hover:text-violet-400 transition-all hover:scale-110"
                title="Toggle Theme"
             >
                {theme === 'light' ? <Sun size={20} /> : <Moon size={20} />}
             </button>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 lg:p-10 scroll-smooth">
          <div className="w-full mx-auto">
            {children}
          </div>
        </div>
      </main>
    </div>
  );
};