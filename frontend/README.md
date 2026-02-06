# Idea2Paper Frontend

基于 **React + TypeScript + Vite** 构建的现代化 Web 界面，用于运行 Idea2Story 流水线并可视化结果。

## 🚀 快速开始

### 用户使用

直接启动后端服务器：

```bash
cd server
python app.py --host 127.0.0.1 --port 8080
```

在浏览器中打开 `http://127.0.0.1:8080`

前端已预先构建在 `web/dist/` 目录中，后端会自动提供静态文件。

### 开发模式

如果需要修改前端代码：

**1. 安装依赖**

```bash
cd web
npm install
```

**2. 启动后端服务器**

```bash
cd server
python app.py --host 127.0.0.1 --port 8080
```

**3. 启动前端开发服务器**

```bash
cd web
npm run dev
```

前端开发服务器运行在 `http://localhost:3000`

**4. 构建生产版本**

修改完成后，构建新的生产版本：

```bash
cd web
npm run build
```

构建产物输出到 `web/dist/` 目录，提交到版本控制中。

## ✨ 功能特性

### 📊 Dashboard 页面
- **一键运行**：输入研究想法并启动流水线
- **实时进度**：6 阶段进度可视化（Init → KG Search → Retrieval → Generation → Review → Refinement）
- **进程管理**：支持终止运行和孤立进程恢复

### 📄 Results 页面
- **历史记录**：左侧边栏显示所有历史运行结果
- **三列布局**：历史记录（20%）+ 论文内容（50%）+ 评审信息（30%）
- **评审详情**：显示评审者、反馈、锚定论文和分数
- **导出功能**：支持 JSON 和 Markdown 格式导出

### ⚙️ Settings 页面
- **配置持久化**：自动保存到 localStorage
- **LLM 配置**：API Key、模型选择、温度参数
- **流水线配置**：Embedding、检索、评审和验证选项

### 🎨 界面特性
- 深色/亮色主题切换
- 响应式设计
- 中英文切换
- 现代化 UI（Tailwind CSS）

## 🔒 安全说明

- API Key 保存在浏览器 localStorage，不会上传到服务器
- 运行时注入到子进程环境变量，运行结束后销毁
- 所有数据本地处理

## 🛠️ 技术栈

- **前端**：React 19 + TypeScript + Vite
- **UI**：Tailwind CSS + Lucide React
- **图谱**：ReactFlow
- **后端**：Python 3.10+ (Flask)

## 📁 目录结构

```
frontend/
├── server/
│   ├── app.py              # 后端服务器
│   ├── stage_mapper.py     # 阶段映射
│   ├── log_zipper.py       # 日志打包
│   └── run_registry.py     # 运行状态管理
├── web/
│   ├── components/         # React 组件
│   │   ├── Layout.tsx
│   │   ├── ResultViewer.tsx
│   │   └── KnowledgeGraph.tsx
│   ├── services/
│   │   └── api.ts          # API 封装
│   ├── types.ts            # 类型定义
│   ├── App.tsx
│   ├── main.tsx
│   ├── package.json
│   ├── vite.config.ts
│   └── dist/               # 构建产物（已提交到版本控制）
└── README.md
```

## 🛠️ 常见问题

**Q: 用户需要安装 Node.js 吗？**
A: 不需要。前端已预先构建，用户只需要 Python 3.10+ 即可运行。

**Q: 如何查看详细日志？**
A: 点击"下载日志"按钮获取完整日志文件，或直接在终端运行流水线。

**Q: 配置会丢失吗？**
A: 不会，配置保存在浏览器 localStorage 中。

**Q: 如何修改前端代码？**
A: 按照"开发模式"章节的步骤操作。修改后记得运行 `npm run build` 并提交 `dist/` 目录。

## 📝 开发指南

### 添加新功能

1. 在 `components/` 下创建新组件
2. 在 `App.tsx` 中集成
3. 如需后端支持，在 `server/app.py` 中添加 API
4. 更新 `types.ts` 添加类型定义
5. 运行 `npm run build` 构建生产版本
6. 提交 `dist/` 目录到版本控制

## 📄 许可证

与主项目保持一致（MIT License）
