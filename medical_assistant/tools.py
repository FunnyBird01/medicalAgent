"""Deterministic clinical-support tools used by the agent.

These tools are intentionally conservative: they expose assumptions and warn
when required clinical context is missing. They do not diagnose or prescribe.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .models import AgentEvent, InventoryItem, SearchResult
from .rag import DocumentStore
from .storage import SQLiteStore


EventCallback = Callable[[AgentEvent], None]


class MedicalToolset:
    DOSING_STANDARDS = {
        "对乙酰氨基酚": {"dose_mg_per_kg": 10.0, "frequency_per_day": 4, "max_daily_mg_per_kg": 60.0, "unit": "mg"},
        "扑热息痛": {"dose_mg_per_kg": 10.0, "frequency_per_day": 4, "max_daily_mg_per_kg": 60.0, "unit": "mg"},
        "阿莫西林": {"dose_mg_per_kg": 25.0, "frequency_per_day": 3, "max_daily_mg_per_kg": 90.0, "unit": "mg"},
        "布洛芬": {"dose_mg_per_kg": 10.0, "frequency_per_day": 3, "max_daily_mg_per_kg": 40.0, "unit": "mg"},
    }
    INTERACTIONS = [
        ({"华法林", "阿司匹林"}, "出血风险增加", "联合使用通常需要专科医生评估并监测凝血指标。"),
        ({"华法林", "布洛芬"}, "出血风险增加", "NSAIDs可能增加消化道出血风险，避免自行合用。"),
        ({"阿司匹林", "布洛芬"}, "胃肠道及出血风险增加", "非必要不建议同时使用两种NSAID/抗血小板药。"),
        ({"对乙酰氨基酚", "酒精"}, "肝损伤风险增加", "有饮酒或肝病史时应由医生确认剂量。"),
    ]
    DEPARTMENTS = {
        "胸痛": ("急诊科/心内科", "胸痛、胸闷、出汗或呼吸困难需优先排除急性冠脉综合征。"),
        "呼吸困难": ("急诊科/呼吸内科", "突发或进行性呼吸困难应先进行急诊分级。"),
        "咳嗽": ("呼吸内科", "持续咳嗽、咯血或伴高热时建议尽快就医。"),
        "发热": ("感染科/呼吸内科", "高热不退、意识改变或免疫抑制患者应急诊评估。"),
        "腹痛": ("急诊科/消化内科", "剧烈腹痛、腹膜刺激征或伴休克表现时直接急诊。"),
        "皮疹": ("皮肤科", "伴呼吸困难、口唇肿胀时按过敏急症处理。"),
        "骨折": ("骨科", "开放性损伤、肢体缺血或明显畸形需急诊处理。"),
        "视物模糊": ("眼科/神经内科", "突发单眼视力下降或伴神经功能缺损需急诊。"),
    }

    def __init__(self, store: SQLiteStore, rag: DocumentStore, callback: EventCallback | None = None):
        self.store = store
        self.rag = rag
        self.callback = callback

    def _event(self, node: str, event_type: str, detail: str) -> AgentEvent:
        event = AgentEvent(node, event_type, detail)
        if self.callback:
            self.callback(event)
        return event

    def search_knowledge(self, query: str, top_k: int = 5, rrf_weight: float = 0.65) -> dict[str, Any]:
        self._event("knowledge_retrieval", "tool_start", f"混合检索：{query}")
        results = self.rag.search(query, top_k, rrf_weight)
        self._event("knowledge_retrieval", "tool_end", f"检索到 {len(results)} 个文档片段")
        return {"query": query, "results": [result.__dict__ if hasattr(result, "__dict__") else {"title": result.title, "text": result.text, "score": result.score, "path": result.path, "chunk_id": result.chunk_id} for result in results]}

    def calculate_dosage(self, drug: str, weight_kg: float, dose_mg_per_kg: float | None = None, frequency_per_day: int | None = None, max_daily_mg_per_kg: float | None = None) -> dict[str, Any]:
        self._event("dosage_calculator", "tool_start", f"计算 {drug}，体重 {weight_kg:g} kg")
        if weight_kg <= 0 or weight_kg > 500:
            return {"ok": False, "error": "体重必须在 0～500 kg 范围内。"}
        standard = next((value for name, value in self.DOSING_STANDARDS.items() if name in drug or drug in name), None)
        dose = dose_mg_per_kg if dose_mg_per_kg is not None else (standard or {}).get("dose_mg_per_kg")
        frequency = frequency_per_day if frequency_per_day is not None else (standard or {}).get("frequency_per_day")
        max_daily = max_daily_mg_per_kg if max_daily_mg_per_kg is not None else (standard or {}).get("max_daily_mg_per_kg")
        if dose is None or frequency is None:
            return {"ok": False, "error": "缺少该药的可靠剂量标准，请提供医嘱剂量或先检索院内药典。", "requires_review": True}
        if dose <= 0 or frequency <= 0 or frequency > 24:
            return {"ok": False, "error": "剂量或频次参数无效。"}
        single = weight_kg * dose
        daily = single * frequency
        warnings: list[str] = ["结果仅用于核对计算，必须以当前药品说明书、患者年龄/肝肾功能和医嘱为准。"]
        if max_daily and daily > weight_kg * max_daily:
            warnings.append(f"计算的日剂量超过参考上限 {weight_kg * max_daily:.1f} mg，请立即复核。")
        result = {"ok": True, "drug": drug, "weight_kg": weight_kg, "single_dose_mg": round(single, 2), "frequency_per_day": frequency, "daily_dose_mg": round(daily, 2), "warnings": warnings}
        self._event("dosage_calculator", "tool_end", f"单次 {single:.1f} mg，每日 {daily:.1f} mg")
        return result

    def assess_emergency(self, symptoms: str | list[str]) -> dict[str, Any]:
        text = " ".join(symptoms) if isinstance(symptoms, list) else symptoms
        self._event("emergency_triage", "tool_start", f"症状分级：{text[:100]}")
        red_flags = {"意识丧失": "意识丧失", "昏迷": "意识障碍", "呼吸停止": "呼吸停止", "呼吸困难": "呼吸困难", "胸痛": "胸痛", "偏瘫": "偏瘫", "言语不清": "言语不清", "大出血": "大出血", "抽搐": "抽搐", "休克": "休克"}
        hits = [label for key, label in red_flags.items() if key in text]
        if any(item in hits for item in {"意识丧失", "意识障碍", "呼吸停止", "大出血", "休克"}):
            level, action = "一级（紧急）", "立即拨打急救电话并启动院内急救流程，不要等待线上咨询。"
        elif hits:
            level, action = "二级（高风险）", "建议立即到急诊科评估，必要时进行心电图、生命体征和相关检查。"
        else:
            level, action = "三级（一般）", "未识别到规则中的高危信号；如症状持续、加重或出现新症状，应及时就医。"
        result = {"level": level, "matched_red_flags": hits, "recommendation": action, "disclaimer": "该分级不能排除急症，不能替代现场诊疗。"}
        self._event("emergency_triage", "tool_end", f"分级结果：{level}")
        return result

    def recommend_department(self, symptoms: str) -> dict[str, Any]:
        self._event("department_router", "tool_start", f"分诊输入：{symptoms[:100]}")
        matches = [(department, reason) for keyword, (department, reason) in self.DEPARTMENTS.items() if keyword in symptoms]
        if not matches:
            matches = [("全科医学科", "症状信息不足，建议先由全科/接诊医生进行初步评估。")]
        result = {"recommendations": [{"department": d, "reason": r} for d, r in matches[:3]], "requires_triage": True}
        self._event("department_router", "tool_end", f"推荐 {len(result['recommendations'])} 个科室")
        return result

    def check_drug_interactions(self, drugs: list[str] | str) -> dict[str, Any]:
        names = [item.strip() for item in (drugs.split(",") if isinstance(drugs, str) else drugs) if item.strip()]
        self._event("interaction_checker", "tool_start", f"校验药品：{'、'.join(names)}")
        normalized = set(names)
        found = [{"drugs": sorted(pair), "risk": risk, "advice": advice} for pair, risk, advice in self.INTERACTIONS if pair.issubset(normalized)]
        result = {"drugs": names, "interactions": found, "safe": not found, "warning": "未发现内置规则命中不代表绝对安全，请以药典/药师审核为准。"}
        self._event("interaction_checker", "tool_end", f"发现 {len(found)} 条配伍风险")
        return result

    def query_inventory(self, drug_name: str = "", department: str = "", low_only: bool = False) -> dict[str, Any]:
        self._event("inventory_query", "tool_start", f"库存查询：药品={drug_name or '全部'}，科室={department or '全部'}")
        items = self.store.query_inventory(drug_name, department, low_only)
        result = {"items": [item.as_dict() for item in items], "count": len(items), "low_stock_count": sum(item.status != "正常" for item in items)}
        self._event("inventory_query", "tool_end", f"返回 {len(items)} 条库存记录")
        return result

