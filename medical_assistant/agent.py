"""Local, auditable clinical-support workflow.

LangGraph is used when installed by downstream deployments, while this module
ships a small deterministic state machine so the application remains usable in
an offline virtual environment without an additional runtime service.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import AppConfig
from .models import AgentEvent, AgentResult, Source
from .rag import DocumentStore
from .storage import SQLiteStore
from .tools import MedicalToolset


def find_ollama_executable() -> str | None:
    """Locate the ollama binary via PATH or the standard Windows install folders."""
    exe = shutil.which("ollama") or shutil.which("ollama.exe")
    if exe:
        return exe
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Ollama" / "ollama.exe",
        Path("D:/app/ollama/ollama.exe"),
    ]
    return next((str(path) for path in candidates if path.is_file()), None)


def is_ollama_running(ollama_url: str, timeout: float = 1.0) -> bool:
    parsed = urlparse(ollama_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 11434), timeout=timeout):
            return True
    except OSError:
        return False


def launch_ollama_detached(logger: Callable[[str], None] | None = None) -> bool:
    """Start the Ollama desktop app (which hosts the local API) in the background."""
    exe = find_ollama_executable()
    if not exe:
        if logger:
            logger("未找到 ollama 可执行文件，无法自动启动")
        return False
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        subprocess.Popen(
            [exe],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=creationflags,
        )
    except OSError as exc:
        if logger:
            logger(f"自动启动 Ollama 失败：{exc}")
        return False
    if logger:
        logger(f"Ollama 服务未运行，已通过 {exe} 发起自动启动")
    return True


def ensure_ollama_running(ollama_url: str, wait_seconds: float = 30.0, logger: Callable[[str], None] | None = None) -> bool:
    """Return True when the local API answers, launching Ollama first if needed."""
    if is_ollama_running(ollama_url):
        return True
    if not launch_ollama_detached(logger):
        return False
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if is_ollama_running(ollama_url):
            if logger:
                logger("Ollama 服务已就绪")
            return True
        time.sleep(0.5)
    if logger:
        logger("等待 Ollama 服务启动超时")
    return False


class LocalAgent:
    MAX_REACT_STEPS = 5

    TOOL_CATALOG = [
        {"name": "calculate_dosage", "desc": "按体重计算药品单次/每日剂量（参数：drug, weight_kg, dose_mg_per_kg?, frequency_per_day?）"},
        {"name": "assess_emergency", "desc": "急症风险分级，识别胸痛/呼吸困难/昏迷/偏瘫等红旗信号（参数：symptoms）"},
        {"name": "check_drug_interactions", "desc": "校验药品相互作用与配伍禁忌（参数：drugs）"},
        {"name": "recommend_department", "desc": "根据症状推荐首诊科室（参数：symptoms）"},
        {"name": "query_inventory", "desc": "查询院内药品库存余量与缺货/紧缺状态（参数：drug_name?, department?, low_only?）"},
        {"name": "search_knowledge", "desc": "检索本地医疗知识库（药典/指南/规范）（参数：query）"},
        {"name": "finish", "desc": "给出最终回答（参数：answer）"},
    ]

    def __init__(self, config: AppConfig, store: SQLiteStore, rag: DocumentStore, event_callback: Callable[[AgentEvent], None] | None = None):
        self.config = config
        self.store = store
        self.rag = rag
        self.event_callback = event_callback

    def _emit(self, events: list[AgentEvent], node: str, event_type: str, detail: str, session_id: str | None = None) -> None:
        event = AgentEvent(node, event_type, detail)
        events.append(event)
        self.store.add_audit(event_type, f"[{node}] {detail}", session_id)
        if self.event_callback:
            self.event_callback(event)

    def _tool_callback(self, events: list[AgentEvent], session_id: str | None) -> Callable[[AgentEvent], None]:
        def receive(event: AgentEvent) -> None:
            events.append(event)
            self.store.add_audit(event.event_type, f"[{event.node}] {event.detail}", session_id)
            if self.event_callback:
                self.event_callback(event)
        return receive

    def run(self, query: str, session_id: str | None = None, history: list[dict[str, Any]] | None = None, patient: dict[str, Any] | None = None, stop_check: Callable[[], bool] | None = None) -> AgentResult:
        query = query.strip()
        events: list[AgentEvent] = []
        tool_results: list[dict[str, Any]] = []
        sources: list[Source] = []
        if not query:
            return AgentResult("请输入需要咨询的内容。", events=events)
        self._emit(events, "router", "node_start", "接收请求并识别临床任务", session_id)
        toolset = MedicalToolset(self.store, self.rag, self._tool_callback(events, session_id))
        # Use the model for every answer. The deterministic workflow remains an
        # auditable safety net and is clearly labelled if generation is unavailable.
        if self.config.use_ollama:
            try:
                react = self._run_react(query, toolset, events, session_id, history, patient, stop_check)
                if react is not None:
                    answer, sources, tool_results = react
                    answer = self._ensure_inventory_answer(query, answer, toolset)
                    if not answer.startswith("免责声明"):
                        answer = "免责声明：本工具仅提供院内临床辅助信息，不替代医生诊断、处方或现场急救。\n\n" + answer
                    self._emit(events, "workflow", "node_end", "ReAct 工作流完成", session_id)
                    return AgentResult(answer, sources, tool_results, events)
            except Exception as exc:
                self._emit(events, "react", "error", f"ReAct 异常，降级到规则工作流：{exc}", session_id)
        # ----- deterministic fallback -----
        lower = query.lower()
        try:
            basic_answer = self._basic_answer(query)
            if basic_answer is not None:
                self._emit(events, "router", "decision", "识别为基础问候/系统说明，无需检索", session_id)
                answer = basic_answer
            elif any(word in query for word in ("剂量", "用量", "mg/kg", "多少毫克", "服用几次")):
                self._emit(events, "router", "decision", "路由到用药剂量计算工具", session_id)
                result = self._run_dosage(toolset, query)
                tool_results.append({"tool": "calculate_dosage", "result": result})
                answer = self._format_dosage(result)
            elif any(word in query for word in ("急症", "危险", "风险", "胸痛", "呼吸困难", "昏迷", "偏瘫", "大出血", "抽搐")):
                self._emit(events, "router", "decision", "路由到急症风险分级工具", session_id)
                result = toolset.assess_emergency(query)
                tool_results.append({"tool": "assess_emergency", "result": result})
                answer = self._format_emergency(result)
            elif any(word in query for word in ("库存", "余量", "缺货", "紧缺", "存量")):
                self._emit(events, "router", "decision", "路由到院内库存查询工具", session_id)
                drug = self._find_drug(query)
                department = self._find_department(query)
                result = toolset.query_inventory(drug, department, "缺货" in query or "紧缺" in query)
                tool_results.append({"tool": "query_inventory", "result": result})
                answer = self._format_inventory(result)
            elif any(word in query for word in ("配伍", "相互作用", "禁忌", "能不能一起", "一起吃")):
                self._emit(events, "router", "decision", "路由到配伍禁忌校验工具", session_id)
                names = self._find_drugs(query)
                result = toolset.check_drug_interactions(names)
                tool_results.append({"tool": "check_drug_interactions", "result": result})
                answer = self._format_interactions(result)
            elif any(word in query for word in ("挂号", "分诊", "看什么科", "哪个科", "科室")):
                self._emit(events, "router", "decision", "路由到科室分诊推荐工具", session_id)
                result = toolset.recommend_department(query)
                tool_results.append({"tool": "recommend_department", "result": result})
                answer = self._format_departments(result)
            else:
                self._emit(events, "router", "decision", "路由到医疗知识库检索工具", session_id)
                result = toolset.search_knowledge(query, self.config.top_k, self.config.rrf_weight)
                tool_results.append({"tool": "search_knowledge", "result": result})
                sources = [Source(item["title"], item["text"], item["score"], item.get("path", ""), item.get("chunk_id", "")) for item in result["results"]]
                answer = self._answer_from_context(sources)
        except Exception as exc:  # keep a clinical UI responsive even if optional integrations fail
            self._emit(events, "workflow", "error", f"工具执行异常：{exc}", session_id)
            answer = "本次辅助流程未能完成，请改用人工核对或联系系统管理员。"
        if self.config.use_ollama:
            generated = self._synthesize_fallback_answer(query, answer, history, patient, events, session_id, stop_check)
            if generated:
                answer = generated
            else:
                answer = "【模型未参与本次回复】本机模型当前不可用，以下内容由规则工作流生成。\n\n" + answer
        else:
            answer = "【规则模式】本机模型未启用，以下内容由规则工作流生成。\n\n" + answer
        answer = self._ensure_inventory_answer(query, answer, toolset)
        if not answer.startswith("免责声明"):
            answer = "免责声明：本工具仅提供院内临床辅助信息，不替代医生诊断、处方或现场急救。\n\n" + answer
        self._emit(events, "workflow", "node_end", "工作流完成，生成可追溯结果", session_id)
        return AgentResult(answer, sources, tool_results, events)

    # ---------------------------------------------------------------------- ReAct

    def _build_guardrails(self, patient: dict[str, Any] | None) -> str:
        """Inject patient safety context (weight, allergy, pregnancy, liver/kidney) into the prompt."""
        if not patient:
            return "【患者信息】当前未绑定病例。若问题涉及用药剂量，请先要求用户提供体重；若涉及特殊人群用药，请提醒核对过敏史、孕哺状态和肝肾功能。"
        name = patient.get("patient_name", "")
        age = patient.get("age", "")
        weight = patient.get("weight", "")
        preg = patient.get("pregnancy_status", "非孕哺")
        allergy = patient.get("allergy_history", "无")
        lkf = patient.get("liver_kidney_function", "正常")
        diag = patient.get("primary_diagnosis", "")
        return (
            f"【患者信息】姓名：{name}，年龄：{age}岁，体重：{weight}kg，孕哺状态：{preg}，"
            f"过敏史：{allergy}，肝肾功能：{lkf}，初步诊断：{diag}。\n"
            "安全规则：\n"
            "- 剂量必须按体重核对；\n"
            "- 过敏药物绝对禁用并明确警告；\n"
            "- 孕哺期、肝肾功能异常须提示用药风险与调整建议。"
        )

    def _drug_safety_warnings(self, drug: str, patient: dict[str, Any] | None) -> list[str]:
        if not patient:
            return []
        warnings: list[str] = []
        allergy = patient.get("allergy_history", "") or ""
        preg = patient.get("pregnancy_status", "") or ""
        lkf = patient.get("liver_kidney_function", "") or ""
        if drug and drug in allergy:
            warnings.append(f"【过敏拦截】患者对 {drug} 过敏，禁用！")
        if "妊娠" in preg or "哺乳" in preg:
            if drug in ("华法林", "布洛芬", "阿司匹林"):
                warnings.append(f"【孕哺提醒】{drug} 在孕哺期禁用/慎用，建议改用对乙酰氨基酚并咨询产科医生。")
        if "肝功能" in lkf or "ALT" in lkf:
            if drug == "对乙酰氨基酚":
                warnings.append("【肝功能提醒】肝功能不全患者使用对乙酰氨基酚须减量（每日≤2g），严重肝功能损害禁用。")
        if "肾功能" in lkf or "eGFR" in lkf:
            if drug == "布洛芬":
                warnings.append("【肾功能提醒】肾功能不全（eGFR<30）患者禁用布洛芬。")
        return warnings

    def _allergy_hits(self, drugs: list[str], patient: dict[str, Any] | None) -> list[str]:
        if not patient:
            return []
        allergy = patient.get("allergy_history", "") or ""
        return [f"【过敏拦截】患者对 {d} 过敏，禁用！" for d in drugs if d and d in allergy]

    @staticmethod
    def _parse_decision(text: str) -> dict[str, Any] | None:
        """Strip <think> blocks and extract the first JSON object (tool decision)."""
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        start = clean.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(clean)):
                if clean[i] == "{":
                    depth += 1
                elif clean[i] == "}":
                    depth -= 1
                    if depth == 0:
                        snippet = clean[start : i + 1]
                        try:
                            return json.loads(snippet)
                        except json.JSONDecodeError:
                            break
            start = clean.find("{", start + 1)
        return None

    def _execute_react_tool(self, toolset: MedicalToolset, decision: dict[str, Any], patient: dict[str, Any] | None, events: list[AgentEvent], session_id: str | None) -> tuple[str, list[dict[str, Any]], list[Source]]:
        name = decision.get("tool_name", decision.get("tool", "search_knowledge"))
        args = decision.get("arguments", decision.get("args", decision)) or {}
        self._emit(events, "react", "tool_call", f"{name}({json.dumps(args, ensure_ascii=False)[:120]})", session_id)
        safety_alerts: list[str] = []
        sources: list[Source] = []
        try:
            if name == "calculate_dosage":
                drug = str(args.get("drug", "未指定药品"))
                weight = float(args.get("weight_kg", 0) or 0)
                result = toolset.calculate_dosage(
                    drug, weight,
                    float(args["dose_mg_per_kg"]) if args.get("dose_mg_per_kg") else None,
                    int(args["frequency_per_day"]) if args.get("frequency_per_day") else None,
                )
                safety_alerts = self._drug_safety_warnings(drug, patient)
                obs = self._format_dosage(result)
            elif name == "assess_emergency":
                result = toolset.assess_emergency(str(args.get("symptoms", args.get("query", ""))))
                obs = self._format_emergency(result)
            elif name == "check_drug_interactions":
                drugs = args.get("drugs", [])
                if isinstance(drugs, str):
                    drugs = [d for d in re.split(r"[,，、\s]+", drugs) if d]
                result = toolset.check_drug_interactions(drugs)
                safety_alerts = self._allergy_hits(drugs, patient)
                obs = self._format_interactions(result)
                if drugs:
                    knowledge = toolset.search_knowledge(" ".join(drugs), self.config.top_k, self.config.rrf_weight)
                    sources = [
                        Source(item["title"], item["text"], item["score"], item.get("path", ""), item.get("chunk_id", ""))
                        for item in knowledge["results"]
                    ]
                    if sources:
                        evidence = "\n".join(
                            f"[{index + 1}] {source.title}：{source.text[:900]}"
                            for index, source in enumerate(sources[:2])
                        )
                        obs += f"\n\n院内知识库依据：\n{evidence}"
            elif name == "recommend_department":
                result = toolset.recommend_department(str(args.get("symptoms", args.get("query", ""))))
                obs = self._format_departments(result)
            elif name == "query_inventory":
                result = toolset.query_inventory(
                    str(args.get("drug_name", "")),
                    str(args.get("department", "")),
                    bool(args.get("low_only", False)),
                )
                obs = self._format_inventory(result)
            elif name == "search_knowledge":
                q = str(args.get("query", args.get("symptoms", "")))
                result = toolset.search_knowledge(q, self.config.top_k, self.config.rrf_weight)
                sources = [Source(item["title"], item["text"], item["score"], item.get("path", ""), item.get("chunk_id", "")) for item in result["results"]]
                obs = "\n".join(f"[{i+1}] {s.title}：{s.text[:200]}" for i, s in enumerate(sources[:3])) or "未检索到相关内容。"
            elif name == "finish":
                obs = str(args.get("answer", ""))
            else:
                obs = f"未知工具：{name}"
        except Exception as exc:
            obs = f"工具执行异常：{exc}"
        for alert in safety_alerts:
            self._emit(events, "safety", "alert", alert, session_id)
        self._emit(events, "react", "tool_observe", obs[:300], session_id)
        return obs, safety_alerts, sources

    def _render_scratch(self, query: str, history: list[dict[str, Any]] | None, steps: list[dict[str, str]]) -> str:
        recent = ""
        if history:
            for msg in history[-4:]:
                role = "用户" if msg.get("role") == "user" else "助手"
                recent += f"{role}：{msg.get('content','')[:200]}\n"
        step_text = ""
        for s in steps:
            step_text += f"思考：{s.get('think','')}\n动作：{s.get('action','')}\n观察：{s.get('obs','')[:300]}\n"
        return f"{recent}\n当前问题：{query}\n{step_text}"

    def _ollama_json(self, prompt: str) -> str | None:
        payload = json.dumps({
            "model": self.config.model_name, "prompt": prompt, "stream": False,
            "options": {"temperature": min(self.config.temperature, 0.3)}, "format": "json",
        }).encode()
        request = urllib.request.Request(f"{self.config.ollama_url}/api/generate", data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
            return str(body.get("response", "")).strip()
        except (OSError, ValueError, urllib.error.URLError):
            return None

    def _stream_final_answer(self, prompt: str, events: list[AgentEvent], session_id: str | None, stop_check: Callable[[], bool] | None) -> str:
        payload = json.dumps({
            "model": self.config.model_name, "prompt": prompt, "stream": True,
            "options": {"temperature": self.config.temperature},
        }).encode()
        request = urllib.request.Request(f"{self.config.ollama_url}/api/generate", data=payload, headers={"Content-Type": "application/json"})
        chunks: list[str] = []
        self._emit(events, "ollama", "request", f"调用模型 {self.config.model_name} 生成最终回答", session_id)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                for line in response:
                    if stop_check and stop_check():
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = str(data.get("response", ""))
                    if token:
                        chunks.append(token)
                    if data.get("done"):
                        break
        except (OSError, ValueError, urllib.error.URLError) as exc:
            self._emit(events, "ollama", "error", f"最终回答生成失败：{exc}", session_id)
            return ""
        raw_answer = "".join(chunks)
        answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL)
        answer = re.sub(r"<think>.*$", "", answer, flags=re.DOTALL)
        answer = answer.replace("</think>", "").replace("**", "").strip()
        if answer:
            self._emit(events, "ollama", "response", "模型最终回答生成成功", session_id)
            if self.event_callback:
                self.event_callback(AgentEvent("react", "answer_chunk", answer))
        else:
            self._emit(events, "ollama", "error", "模型最终回答为空", session_id)
        return answer

    @staticmethod
    def _history_text(history: list[dict[str, Any]] | None, query: str) -> str:
        messages: list[str] = []
        recent = (history or [])[-6:]
        for index, message in enumerate(recent):
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if index == len(recent) - 1 and message.get("role") == "user" and content == query:
                continue
            role = "用户" if message.get("role") == "user" else "助手"
            messages.append(f"{role}：{content[:500]}")
        return "\n".join(messages) or "无"

    def _final_answer_prompt(
        self,
        query: str,
        history: list[dict[str, Any]] | None,
        patient: dict[str, Any] | None,
        evidence: str,
        proposed_answer: str = "",
        safety_alerts: list[str] | None = None,
    ) -> str:
        alerts = "\n".join(safety_alerts or []) or "无"
        proposal = proposed_answer.strip() or "无"
        return (
            "你是院内临床智能辅助助手。请先在内部完成分析，再直接给出自然、针对性强的中文答复；"
            "不要输出 JSON，不要展示隐藏思维过程，也不要套用固定开场白。使用适合桌面聊天框的纯文本，"
            "不要使用 Markdown 加粗标记。\n"
            "事实边界：医疗事实、剂量、库存和配伍结论只能来自下方已验证结果；资料不足时明确说明并追问必要信息，禁止编造。"
            "必须保留已验证结果中的数值、风险警告和人工复核要求。不得建议患者自行停药、换药或调整处方。"
            "不要从常识自行添加替代药物、监测指标、适应证、禁忌证或具体数值；这些内容未出现在已验证结果中时，只能说明资料不足。"
            "能力仅限剂量核对、急症风险初筛、配伍校验、科室推荐、院内库存查询和本地知识库检索，"
            "不要声称能执行其他功能。对于普通寒暄，可以自然回应。\n"
            "库存查询若用户未指定药品而工具返回多条记录，必须逐条列出全部库存记录，不得要求用户先补充药品名称；"
            "库存结果为空时才说明没有符合条件的记录。\n"
            f"{self._build_guardrails(patient)}\n"
            f"【最近对话】\n{self._history_text(history, query)}\n"
            f"【用户当前问题】\n{query}\n"
            f"【已验证的工具/知识库结果】\n{evidence or '无'}\n"
            f"【上一阶段拟答（仅供参考）】\n{proposal}\n"
            f"【必须强调的安全提醒】\n{alerts}\n"
            "请生成最终答复。"
        )

    def _synthesize_fallback_answer(
        self,
        query: str,
        grounded_result: str,
        history: list[dict[str, Any]] | None,
        patient: dict[str, Any] | None,
        events: list[AgentEvent],
        session_id: str | None,
        stop_check: Callable[[], bool] | None,
    ) -> str:
        """Turn deterministic results into a contextual model answer after a ReAct failure."""
        prompt = self._final_answer_prompt(query, history, patient, grounded_result)
        return self._stream_final_answer(prompt, events, session_id, stop_check)

    def _run_react(self, query: str, toolset: MedicalToolset, events: list[AgentEvent], session_id: str | None, history: list[dict[str, Any]] | None, patient: dict[str, Any] | None, stop_check: Callable[[], bool] | None) -> tuple[str, list[Source], list[dict[str, Any]]] | None:
        """ReAct loop: think -> tool -> observe -> finish. Returns (answer, sources, tool_results) or None on failure."""
        if not ensure_ollama_running(self.config.ollama_url, logger=lambda d: self._emit(events, "ollama", "status", d, session_id)):
            return None
        if not self._server_has_model(events, session_id):
            return None
        guardrails = self._build_guardrails(patient)
        tools_desc = "\n".join(f"- {t['name']}：{t['desc']}" for t in self.TOOL_CATALOG)
        system = (
            "你是院内临床辅助 ReAct 智能体。请在内部分析，但每次只输出一个 JSON 对象，"
            "不要输出思维过程。JSON 必须包含 tool_name 和 arguments 两个字段。"
            "涉及医疗事实、剂量、配伍、分诊或库存时，先调用合适的工具收集依据；"
            "普通寒暄、致谢以及无需外部事实的追问可以直接调用 finish。已有足够依据后调用 finish。"
            "不要重复调用参数相同的工具。\n"
            f"{guardrails}\n【可用工具】\n{tools_desc}\n"
            "输出格式示例：{\"tool_name\": \"calculate_dosage\", \"arguments\": {\"drug\": \"对乙酰氨基酚\", \"weight_kg\": 20}}"
        )
        steps: list[dict[str, str]] = []
        all_sources: list[Source] = []
        all_safety: list[str] = []
        used_tools: set[str] = set()
        fail_count = 0
        for step in range(self.MAX_REACT_STEPS):
            if stop_check and stop_check():
                return ("已停止生成。", all_sources, [])
            scratch = self._render_scratch(query, history, steps)
            prompt = f"{system}\n{scratch}\n请输出下一步的 JSON 决策："
            self._emit(events, "react", "think", f"第 {step+1} 步：分析问题并选择工具", session_id)
            raw = self._ollama_json(prompt)
            if raw is None:
                fail_count += 1
                self._emit(events, "react", "error", f"第 {step+1} 步未获得有效模型输出", session_id)
                if fail_count >= 2:
                    return None
                continue
            decision = self._parse_decision(raw)
            if not decision:
                fail_count += 1
                self._emit(events, "react", "error", f"第 {step+1} 步模型输出不是有效工具 JSON", session_id)
                if fail_count >= 2:
                    return None
                continue
            name = decision.get("tool_name", "")
            if name == "finish":
                answer = str(decision.get("arguments", {}).get("answer", ""))
                if not answer:
                    answer = str(decision.get("answer", ""))
                obs_context = "\n".join(s.get("obs", "")[:1200] for s in steps)
                final_prompt = self._final_answer_prompt(
                    query, history, patient, obs_context, answer, all_safety
                )
                answer = self._stream_final_answer(final_prompt, events, session_id, stop_check) or answer
                return (answer, all_sources, [{"tool": s.get("action", ""), "result": s.get("obs", "")} for s in steps])
            if name in used_tools:
                self._emit(events, "react", "decision", f"跳过重复工具 {name}，使用现有结果生成回答", session_id)
                break
            used_tools.add(name)
            obs, safety, srcs = self._execute_react_tool(toolset, decision, patient, events, session_id)
            all_sources.extend(srcs)
            all_safety.extend(safety)
            steps.append({"think": f"调用 {name}", "action": name, "obs": obs})
        # Step limit reached: summarize observations
        obs_context = "\n".join(s.get("obs", "")[:1200] for s in steps)
        final_prompt = self._final_answer_prompt(query, history, patient, obs_context, safety_alerts=all_safety)
        answer = self._stream_final_answer(final_prompt, events, session_id, stop_check)
        if not answer:
            return None
        return (answer, all_sources, [{"tool": s.get("action", ""), "result": s.get("obs", "")} for s in steps])

    @staticmethod
    def _basic_answer(query: str) -> str | None:
        """Answer common conversational questions locally instead of treating them as clinical searches."""
        normalized = re.sub(r"[，。！？,.!?\s]+", "", query.lower())
        greetings = {"你好", "您好", "嗨", "hi", "hello", "hey", "在吗", "早上好", "下午好", "晚上好"}
        if normalized in greetings or any(normalized.startswith(item) for item in ("你好呀", "您好呀", "嗨呀")):
            return "您好，我是院内临床智能辅助助手。您可以咨询剂量核对、急症风险、配伍禁忌、科室分诊和药品库存。"
        if any(item in normalized for item in ("你是谁", "你叫什么", "你能做什么", "有什么功能", "怎么使用", "帮助", "功能")):
            return "我可以在本地帮助核对常用药品剂量、初步识别急症风险、推荐就诊科室、检查内置配伍风险规则、查询科室库存，并检索已导入的院内指南。结果仅供专业人员复核。"
        if any(item in normalized for item in ("谢谢", "感谢", "多谢")):
            return "不客气。涉及诊疗、用药或急症时，请以现场医生、药师和院内规范为准。"
        return None

    def _run_dosage(self, toolset: MedicalToolset, query: str) -> dict[str, Any]:
        weight_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|公斤|千克)", query, re.I)
        dose_match = re.search(r"(\d+(?:\.\d+)?)\s*mg\s*/\s*kg", query, re.I)
        freq_match = re.search(r"(?:每日|一天|每天)\s*(\d+)\s*次", query)
        drug = self._find_drug(query) or "未指定药品"
        if not weight_match:
            return {"ok": False, "error": "请提供体重（例如：儿童 20 kg）后再计算。", "requires_review": True}
        return toolset.calculate_dosage(drug, float(weight_match.group(1)), float(dose_match.group(1)) if dose_match else None, int(freq_match.group(1)) if freq_match else None)

    @staticmethod
    def _find_drug(query: str) -> str:
        for drug in ("对乙酰氨基酚", "扑热息痛", "阿司匹林", "华法林", "布洛芬", "阿莫西林", "二甲双胍", "头孢曲松", "布洛芬混悬液"):
            if drug in query:
                return drug
        return ""

    def _find_drugs(self, query: str) -> list[str]:
        drugs = [drug for drug in ("对乙酰氨基酚", "扑热息痛", "阿司匹林", "华法林", "布洛芬", "阿莫西林", "二甲双胍", "头孢曲松", "酒精") if drug in query]
        return drugs or [part for part in re.split(r"[,，、\s]+", query) if part]

    @staticmethod
    def _find_department(query: str) -> str:
        for department in ("急诊科", "内科", "儿科", "外科", "心内科", "呼吸内科", "妇产科"):
            if department in query:
                return department
        return ""

    @staticmethod
    def _format_dosage(result: dict[str, Any]) -> str:
        if not result.get("ok"):
            return f"剂量计算无法完成：{result.get('error', '参数不足')}"
        warnings = "\n".join(f"- {warning}" for warning in result.get("warnings", []))
        return f"剂量核对结果（{result['drug']}）：\n- 单次参考剂量：{result['single_dose_mg']} mg\n- 频次：每日 {result['frequency_per_day']} 次\n- 日累计参考剂量：{result['daily_dose_mg']} mg\n\n注意事项：\n{warnings}"

    @staticmethod
    def _format_emergency(result: dict[str, Any]) -> str:
        flags = "、".join(result["matched_red_flags"]) or "无"
        return f"急症风险分级：{result['level']}\n识别到的高危信号：{flags}\n建议：{result['recommendation']}\n\n{result['disclaimer']}"

    @staticmethod
    def _format_inventory(result: dict[str, Any]) -> str:
        if not result["items"]:
            return "未找到符合条件的库存记录，请确认药品名称或科室。"
        lines = [f"院内库存查询结果（共 {result['count']} 条，紧缺/缺货 {result['low_stock_count']} 条）："]
        for item in result["items"]:
            lines.append(f"- {item['department']}｜{item['drug_name']} {item['specification']}：{item['quantity']} {item['unit']}（{item['status']}）")
        return "\n".join(lines)

    def _ensure_inventory_answer(self, query: str, answer: str, toolset: MedicalToolset) -> str:
        """Ensure an all-inventory request includes every validated inventory row."""
        if not any(word in query for word in ("库存", "余量", "缺货", "紧缺", "存量")):
            return answer
        # A named-drug request stays focused on that drug. Without one, the
        # user is asking for the complete department or hospital inventory.
        if self._find_drug(query):
            return answer
        department = self._find_department(query)
        result = toolset.query_inventory(
            "", department, "缺货" in query or "紧缺" in query
        )
        items = result.get("items", [])
        if not items or all(str(item.get("drug_name", "")) in answer for item in items):
            return answer
        listing = self._format_inventory(result)
        return f"{answer.rstrip()}\n\n系统已验证的完整库存明细：\n{listing}"

    @staticmethod
    def _format_interactions(result: dict[str, Any]) -> str:
        if result["safe"]:
            return f"未命中内置配伍风险规则（已校验：{'、'.join(result['drugs'])}）。\n{result['warning']}"
        lines = ["发现需要复核的配伍风险："]
        lines.extend(f"- {' + '.join(item['drugs'])}：{item['risk']}。{item['advice']}" for item in result["interactions"])
        return "\n".join(lines) + f"\n\n{result['warning']}"

    @staticmethod
    def _format_departments(result: dict[str, Any]) -> str:
        return "推荐就诊科室：\n" + "\n".join(f"- {item['department']}：{item['reason']}" for item in result["recommendations"])

    @staticmethod
    def _answer_from_context(sources: list[Source]) -> str:
        if not sources:
            return "本地知识库中没有检索到足够的依据。请导入经审核的院内药典/指南文档，或联系药师、临床医生进行人工确认。"
        snippets = "\n".join(f"- {source.text[:240]}" for source in sources[:3])
        return f"根据本地知识库检索到的相关内容：\n{snippets}\n\n以上为检索片段，具体适用范围、禁忌和处置仍需结合患者完整病史及院内规范由专业人员确认。"

    def _server_has_model(self, events: list[AgentEvent], session_id: str | None) -> bool:
        """Confirm the configured model is actually installed on the running server."""
        request = urllib.request.Request(f"{self.config.ollama_url}/api/tags")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError, urllib.error.URLError) as exc:
            self._emit(events, "ollama", "error", f"获取服务模型列表失败：{exc}", session_id)
            return False
        names = [str(item.get("name", "")) for item in body.get("models", [])]
        if self.config.model_name in names:
            return True
        self._emit(events, "ollama", "error",
                   f"服务中未找到模型 {self.config.model_name}，当前可用：{', '.join(names) or '无'}", session_id)
        return False

    def _ollama_answer(
        self,
        query: str,
        sources: list[Source],
        history: list[dict[str, Any]] | None,
        events: list[AgentEvent],
        session_id: str | None,
    ) -> str:
        context = "\n".join(source.text for source in sources[: self.config.top_k])
        prompt = f"你是院内离线临床辅助助手。只根据给定资料回答，不得编造诊疗建议。问题：{query}\n资料：{context}\n请用简洁中文回答，并明确提示需专业人员复核。"
        payload = json.dumps({"model": self.config.model_name, "prompt": prompt, "stream": False, "options": {"temperature": self.config.temperature}}).encode()
        request = urllib.request.Request(f"{self.config.ollama_url}/api/generate", data=payload, headers={"Content-Type": "application/json"})
        self._emit(events, "ollama", "request", f"调用模型 {self.config.model_name} 生成回答", session_id)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
            answer = str(body.get("response", "")).strip()
            if answer:
                self._emit(events, "ollama", "response", "模型回答生成成功", session_id)
            else:
                self._emit(events, "ollama", "error", "模型返回了空回答", session_id)
            return answer
        except (OSError, ValueError, urllib.error.URLError) as exc:
            self._emit(events, "ollama", "error", f"模型生成失败：{exc}", session_id)
            return ""
