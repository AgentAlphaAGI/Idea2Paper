// 中文注释：本文件是 Web UI 的纯原生 JS（不依赖框架/打包器），负责：提交 Job、订阅 SSE、渲染对话与步骤卡片。

(() => {
  // ===== DOM 引用 =====
  const chatEl = document.getElementById("chat");
  const stepsEl = document.getElementById("steps");
  const inputEl = document.getElementById("inputText");
  const runBtn = document.getElementById("runBtn");
  const useLlmEl = document.getElementById("useLlm");
  const snapshotPathEl = document.getElementById("snapshotPath");
  const configPathEl = document.getElementById("configPath");
  const seedEl = document.getElementById("seed");
  const statusBadge = document.getElementById("statusBadge");
  const progressEl = document.getElementById("progress");
  const exportBtn = document.getElementById("exportBtn");
  const themeBtn = document.getElementById("themeBtn");

  let es = null; // 当前 EventSource
  let lastReport = null; // 最终 report（用于导出）
  let running = false;

  // ===== 工具函数 =====
  function nowLocalTime() {
    return new Date().toLocaleTimeString();
  }

  function setStatus(kind, text) {
    statusBadge.classList.remove("idle", "running", "done", "error");
    statusBadge.classList.add(kind);
    statusBadge.textContent = text;
  }

  function setRunning(flag) {
    running = flag;
    runBtn.disabled = flag;
    useLlmEl.disabled = flag;
    snapshotPathEl.disabled = flag;
    configPathEl.disabled = flag;
    seedEl.disabled = flag;
    exportBtn.disabled = flag || !lastReport;
    progressEl.classList.toggle("hidden", !flag);
    setStatus(flag ? "running" : "idle", flag ? "Running" : "Idle");
  }

  function escapeHtml(str) {
    return String(str)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function formatJson(obj) {
    try {
      return JSON.stringify(obj, null, 2);
    } catch (e) {
      return String(obj);
    }
  }

  async function copyText(text) {
    const value = String(text ?? "");
    try {
      await navigator.clipboard.writeText(value);
      return true;
    } catch (e) {
      // 中文注释：兼容部分浏览器的降级复制方式。
      const ta = document.createElement("textarea");
      ta.value = value;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      try {
        document.execCommand("copy");
        return true;
      } finally {
        document.body.removeChild(ta);
      }
    }
  }

  function scrollToBottom(el) {
    el.scrollTop = el.scrollHeight;
  }

  function el(tag, className, html) {
    const n = document.createElement(tag);
    if (className) n.className = className;
    if (html !== undefined) n.innerHTML = html;
    return n;
  }

  function addChatMessage(role, title, text, jsonObj) {
    const msg = el("div", `message ${role}`);

    const meta = el(
      "div",
      "meta",
      `<span>${escapeHtml(role === "user" ? "User" : "System")}</span><span>${escapeHtml(nowLocalTime())}</span>`
    );

    const bubble = el("div", "bubble");
    if (title) bubble.appendChild(el("div", "title", escapeHtml(title)));
    if (text) bubble.appendChild(el("div", "text", escapeHtml(text)));

    if (jsonObj !== undefined && jsonObj !== null) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "查看 JSON";
      details.appendChild(summary);

      const pre = el("pre", "code");
      pre.textContent = formatJson(jsonObj);
      details.appendChild(pre);

      const actions = el("div", "step-actions");
      const copyBtn = el("button", "btn small ghost", "复制 JSON");
      copyBtn.addEventListener("click", async () => {
        await copyText(pre.textContent);
        copyBtn.textContent = "已复制";
        setTimeout(() => (copyBtn.textContent = "复制 JSON"), 900);
      });
      actions.appendChild(copyBtn);
      details.appendChild(actions);

      bubble.appendChild(details);
    }

    msg.appendChild(meta);
    msg.appendChild(bubble);
    chatEl.appendChild(msg);
    scrollToBottom(chatEl);
  }

  function addStepCard({ node, ts, seq, summary, data }) {
    const card = el("div", "step-card");
    const head = el("div", "step-head");
    head.appendChild(el("div", "step-node", escapeHtml(`${seq}. ${node}`)));
    head.appendChild(el("div", "step-ts", escapeHtml(new Date(ts).toLocaleTimeString())));
    card.appendChild(head);

    card.appendChild(el("div", "step-summary", escapeHtml(summary || "")));

    // 中文注释：review 节点做特殊展示：拆分每个 reviewer 的输出。
    if (node === "review" && data && Array.isArray(data.reviews)) {
      const grid = el("div", "review-grid");
      for (const r of data.reviews) {
        const rc = el("div", "review-card");
        const title = el("div", "review-title");
        const left = `${r.reviewer_id || ""} ${r.persona ? "· " + r.persona : ""}`.trim();
        const right = `overall=${Number(r.overall_score ?? 0).toFixed(2)}`;
        title.appendChild(el("div", "", escapeHtml(left || "reviewer")));
        title.appendChild(el("div", "", escapeHtml(right)));
        rc.appendChild(title);

        const meta = el("div", "review-meta");
        meta.appendChild(el("span", "", escapeHtml(`issue=${r.issue_type || ""}`)));
        meta.appendChild(el("span", "", escapeHtml(`decision=${r.decision || ""}`)));
        rc.appendChild(meta);

        const d = document.createElement("details");
        const s = document.createElement("summary");
        s.textContent = "查看该 Reviewer JSON";
        d.appendChild(s);
        const pre = el("pre", "code");
        pre.textContent = formatJson(r);
        d.appendChild(pre);
        rc.appendChild(d);

        grid.appendChild(rc);
      }
      card.appendChild(grid);
    }

    const details = document.createElement("details");
    const sum = document.createElement("summary");
    sum.textContent = "查看节点 JSON";
    details.appendChild(sum);
    const pre = el("pre", "code");
    pre.textContent = formatJson(data);
    details.appendChild(pre);

    const actions = el("div", "step-actions");
    const copyBtn = el("button", "btn small ghost", "复制 JSON");
    copyBtn.addEventListener("click", async () => {
      await copyText(pre.textContent);
      copyBtn.textContent = "已复制";
      setTimeout(() => (copyBtn.textContent = "复制 JSON"), 900);
    });
    actions.appendChild(copyBtn);
    details.appendChild(actions);

    card.appendChild(details);
    stepsEl.appendChild(card);
    scrollToBottom(stepsEl);
  }

  function clearUI() {
    chatEl.innerHTML = "";
    stepsEl.innerHTML = "";
    lastReport = null;
    exportBtn.disabled = true;
  }

  function downloadJson(obj, filename) {
    const data = new Blob([formatJson(obj)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(data);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // ===== 主题切换 =====
  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
    themeBtn.textContent = theme === "dark" ? "浅色模式" : "暗色模式";
  }

  const savedTheme = localStorage.getItem("theme");
  applyTheme(savedTheme === "dark" ? "dark" : "light");
  themeBtn.addEventListener("click", () => {
    const current = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
    applyTheme(current === "dark" ? "light" : "dark");
  });

  // ===== 导出按钮 =====
  exportBtn.addEventListener("click", () => {
    if (!lastReport) return;
    downloadJson(lastReport, "paper_workflow_report.json");
  });

  // ===== 核心：创建 Job + 订阅 SSE =====
  async function startWorkflow() {
    const text = String(inputEl.value || "").trim();
    if (!text) {
      addChatMessage("system", "提示", "请输入一句自然语言的论文想法。");
      return;
    }

    // 中文注释：如果上一次还在运行，先关闭连接并清理。
    if (es) {
      try {
        es.close();
      } catch (_) {}
      es = null;
    }

    clearUI();
    addChatMessage("user", "用户输入", text);

    setRunning(true);

    const snapshotPath = String(snapshotPathEl.value || "./paper_kg_snapshot.json").trim();
    const configPath = String(configPathEl.value || "").trim();
    const seedRaw = String(seedEl.value || "").trim();
    const seed = seedRaw ? Number(seedRaw) : null;
    const useLlm = !!useLlmEl.checked;

    try {
      const resp = await fetch("/api/paper/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          snapshot_path: snapshotPath,
          config_path: configPath || null,
          use_llm: useLlm,
          seed: Number.isFinite(seed) ? seed : null,
        }),
      });

      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(`创建 job 失败：${resp.status} ${errText}`);
      }

      const body = await resp.json();
      const jobId = body.job_id;
      if (!jobId) throw new Error("服务端未返回 job_id");

      es = new EventSource(`/api/paper/jobs/${jobId}/events`);

      es.onmessage = (ev) => {
        let msg = null;
        try {
          msg = JSON.parse(ev.data);
        } catch (e) {
          return;
        }

        if (msg.type === "step") {
          // 对话区：每步追加一条系统消息
          addChatMessage("system", `${msg.seq}. ${msg.node}`, msg.summary || "", msg.data || null);
          // 右侧：步骤卡片
          addStepCard(msg);
          return;
        }

        if (msg.type === "final") {
          lastReport = msg.report || null;
          exportBtn.disabled = !lastReport;
          setStatus("done", "Done");
          progressEl.classList.add("hidden");
          runBtn.disabled = false;
          addChatMessage("system", "最终稿", msg.final_reply || "", msg.report || null);
          if (es) es.close();
          es = null;
          running = false;
          return;
        }

        if (msg.type === "error") {
          setStatus("error", "Error");
          progressEl.classList.add("hidden");
          runBtn.disabled = false;
          addChatMessage("system", "运行失败", msg.message || "未知错误", msg);
          if (es) es.close();
          es = null;
          running = false;
        }
      };

      es.onerror = () => {
        if (!running) return;
        setStatus("error", "Error");
        progressEl.classList.add("hidden");
        runBtn.disabled = false;
        addChatMessage("system", "连接错误", "SSE 连接异常中断，请重试。");
        try {
          es.close();
        } catch (_) {}
        es = null;
        running = false;
      };
    } catch (err) {
      setStatus("error", "Error");
      progressEl.classList.add("hidden");
      runBtn.disabled = false;
      addChatMessage("system", "启动失败", String(err?.message || err));
      running = false;
    }
  }

  runBtn.addEventListener("click", startWorkflow);

  // 中文注释：回车快捷键（Ctrl/Cmd+Enter）提交。
  inputEl.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      startWorkflow();
    }
  });
})();

