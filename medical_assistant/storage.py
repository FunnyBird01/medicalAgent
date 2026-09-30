"""SQLite persistence: conversations, inventory and an auditable event trail."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from .models import InventoryItem, now_iso


class SQLiteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.initialize()

    def initialize(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant','system')), content TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS inventory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, department TEXT NOT NULL, drug_name TEXT NOT NULL,
                    specification TEXT NOT NULL DEFAULT '', quantity INTEGER NOT NULL DEFAULT 0,
                    reorder_level INTEGER NOT NULL DEFAULT 10, unit TEXT NOT NULL DEFAULT '盒', updated_at TEXT NOT NULL,
                    UNIQUE(department, drug_name, specification)
                );
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, event_type TEXT NOT NULL,
                    detail TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS app_meta (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY, case_no TEXT NOT NULL UNIQUE, patient_name TEXT NOT NULL,
                    gender TEXT NOT NULL DEFAULT '', age TEXT NOT NULL, weight TEXT NOT NULL,
                    pregnancy_status TEXT NOT NULL, allergy_history TEXT NOT NULL, liver_kidney_function TEXT NOT NULL,
                    department TEXT NOT NULL DEFAULT '', chief_complaint TEXT NOT NULL DEFAULT '',
                    present_illness TEXT NOT NULL DEFAULT '', past_history TEXT NOT NULL DEFAULT '',
                    primary_diagnosis TEXT NOT NULL DEFAULT '', attending_doctor TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '', admission_date TEXT NOT NULL DEFAULT '',
                    patient_status TEXT NOT NULL DEFAULT '入院中',
                    follow_up_date TEXT NOT NULL DEFAULT '', doctor_name TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS prescriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
                    drug_name TEXT NOT NULL, specification TEXT NOT NULL DEFAULT '', dosage TEXT NOT NULL DEFAULT '',
                    frequency TEXT NOT NULL DEFAULT '', duration TEXT NOT NULL DEFAULT '', quantity TEXT NOT NULL DEFAULT '',
                    route TEXT NOT NULL DEFAULT '口服', prescribing_doctor TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '', prescribed_at TEXT NOT NULL, created_at TEXT NOT NULL
                );
                """
            )
            self._migrate_session_case_binding()
            self._migrate_case_extra_fields()
            self._repair_orphaned_case_owners()
        self.seed_inventory()

    def _migrate_session_case_binding(self) -> None:
        """Add the case_id link column to existing session databases (SQLite has no ADD COLUMN IF NOT EXISTS)."""
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(sessions)")}
        if "case_id" not in columns:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN case_id TEXT")

    def _migrate_case_extra_fields(self) -> None:
        """Add admission_date and patient_status to cases tables created before this feature."""
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(cases)")}
        if "admission_date" not in columns:
            self._conn.execute("ALTER TABLE cases ADD COLUMN admission_date TEXT NOT NULL DEFAULT ''")
        if "patient_status" not in columns:
            self._conn.execute("ALTER TABLE cases ADD COLUMN patient_status TEXT NOT NULL DEFAULT '入院中'")
        if "follow_up_date" not in columns:
            self._conn.execute("ALTER TABLE cases ADD COLUMN follow_up_date TEXT NOT NULL DEFAULT ''")
        if "doctor_name" not in columns:
            self._conn.execute("ALTER TABLE cases ADD COLUMN doctor_name TEXT NOT NULL DEFAULT ''")

    def _repair_orphaned_case_owners(self) -> None:
        """Restore ownership erased by older edit forms that omitted doctor_name."""
        self._conn.execute(
            """
            UPDATE cases
            SET doctor_name=attending_doctor
            WHERE trim(doctor_name)='' AND trim(attending_doctor)!=''
            """
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -------------------------------------------------------------------- meta

    def get_meta(self, key: str, default: str = "") -> str:
        with self._lock:
            row = self._conn.execute("SELECT value FROM app_meta WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO app_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def create_session(self, title: str = "新建会话") -> str:
        session_id = uuid.uuid4().hex
        stamp = now_iso()
        with self._lock, self._conn:
            self._conn.execute("INSERT INTO sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)", (session_id, title, stamp, stamp))
        return session_id

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT s.id, s.title, s.created_at, s.updated_at,
                       c.case_no AS bound_case_no, c.patient_name AS bound_patient_name
                FROM sessions s
                LEFT JOIN cases c ON s.case_id = c.id
                ORDER BY s.updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_session_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT role,content,sources_json,created_at FROM messages WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
        return [{**dict(row), "sources": json.loads(row["sources_json"] or "[]")} for row in rows]

    def delete_session(self, session_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    # ------------------------------------------------------------------ cases

    CASE_FIELDS = [
        "case_no", "patient_name", "gender", "age", "weight", "pregnancy_status",
        "allergy_history", "liver_kidney_function", "department", "chief_complaint",
        "present_illness", "past_history", "primary_diagnosis", "attending_doctor", "notes",
        "admission_date", "patient_status", "follow_up_date", "doctor_name",
    ]

    def next_case_no(self) -> str:
        """Generate a human-friendly case number: BL<yyyymmdd>-<seq>."""
        day = datetime.now().strftime("%Y%m%d")
        prefix = f"BL{day}-"
        with self._lock:
            rows = self._conn.execute(
                "SELECT case_no FROM cases WHERE case_no LIKE ? ORDER BY case_no", (prefix + "%",)
            ).fetchall()
        seq = len(rows) + 1
        number = f"{prefix}{seq:03d}"
        existing = {row[0] for row in rows}
        while number in existing:  # tolerate manual deletions/collisions
            seq += 1
            number = f"{prefix}{seq:03d}"
        return number

    def create_case(self, data: dict[str, Any]) -> dict[str, Any]:
        case_id = uuid.uuid4().hex
        stamp = now_iso()
        values = {field: str(data.get(field, "")).strip() for field in self.CASE_FIELDS}
        if not values["case_no"]:
            values["case_no"] = self.next_case_no()
        if not values["patient_status"]:
            values["patient_status"] = "入院中"
        columns = ["id", *self.CASE_FIELDS, "created_at", "updated_at"]
        placeholders = ",".join("?" for _ in columns)
        params = [case_id, *(values[field] for field in self.CASE_FIELDS), stamp, stamp]
        with self._lock, self._conn:
            self._conn.execute(f"INSERT INTO cases({','.join(columns)}) VALUES({placeholders})", params)
        return self.get_case(case_id)

    def list_cases(self, keyword: str = "", doctor_name: str = "") -> list[dict[str, Any]]:
        sql = "SELECT * FROM cases WHERE 1=1"
        params: list[Any] = []
        if doctor_name:
            sql += " AND doctor_name=?"
            params.append(doctor_name)
        keyword = keyword.strip()
        if keyword:
            sql += " AND (case_no LIKE ? OR patient_name LIKE ?)"
            params += [f"%{keyword}%", f"%{keyword}%"]
        sql += " ORDER BY updated_at DESC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        return dict(row) if row else None

    def update_case(self, case_id: str, data: dict[str, Any]) -> dict[str, Any] | None:
        existing = self.get_case(case_id)
        if not existing:
            return None
        # Edit forms intentionally expose only user-editable fields. Preserve
        # ownership and any future hidden fields instead of blanking them.
        values = {
            field: str(data[field] if field in data else existing.get(field, "")).strip()
            for field in self.CASE_FIELDS
        }
        assignments = ",".join(f"{field}=?" for field in self.CASE_FIELDS)
        params = [*(values[field] for field in self.CASE_FIELDS), now_iso(), case_id]
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE cases SET {assignments},updated_at=? WHERE id=?", params
            )
        return self.get_case(case_id)

    def seed_demo_cases(self, doctor_name: str) -> list[dict[str, Any]]:
        """Create one private, idempotent demonstration set for a doctor."""
        doctor_name = doctor_name.strip()
        if not doctor_name:
            return []
        marker_key = f"demo_cases_v1:{doctor_name}"
        if self.get_meta(marker_key):
            return []

        today = datetime.now().date()
        date_text = lambda offset: (today + timedelta(days=offset)).isoformat()
        prefix = f"DEMO-{doctor_name}-"
        demo_specs: list[dict[str, Any]] = [
            {
                "case_no": prefix + "01",
                "patient_name": "李小童",
                "gender": "男",
                "age": "5",
                "weight": "20",
                "pregnancy_status": "不适用",
                "allergy_history": "无",
                "liver_kidney_function": "正常",
                "department": "儿科",
                "chief_complaint": "发热伴鼻塞2天",
                "present_illness": "最高体温38.6℃，精神尚可，无抽搐、呼吸困难或持续呕吐。",
                "past_history": "既往体健，疫苗接种完整。",
                "primary_diagnosis": "上呼吸道感染待查",
                "admission_date": date_text(0),
                "patient_status": "观察中",
                "follow_up_date": date_text(0),
                "notes": "【系统演示病例】体重剂量核对。推荐提问：对乙酰氨基酚每次10mg/kg，20kg儿童每日4次，单次和每日剂量是多少？",
                "prescriptions": [
                    {"drug_name": "对乙酰氨基酚", "specification": "0.5g*20片", "dosage": "200mg", "frequency": "每日4次", "duration": "2天", "quantity": "1盒", "route": "口服", "notes": "系统演示处方：用于展示处方历史，不作为真实医嘱。"},
                ],
            },
            {
                "case_no": prefix + "02",
                "patient_name": "张小安",
                "gender": "男",
                "age": "8",
                "weight": "25",
                "pregnancy_status": "不适用",
                "allergy_history": "青霉素（既往皮试阳性并出现皮疹）",
                "liver_kidney_function": "正常",
                "department": "儿科",
                "chief_complaint": "发热、咽痛伴扁桃体脓点",
                "present_illness": "发热2天，最高38.3℃，咽痛明显，尚未使用抗菌药。",
                "past_history": "青霉素过敏史。",
                "primary_diagnosis": "急性化脓性扁桃体炎待查",
                "admission_date": date_text(-1),
                "patient_status": "治疗中",
                "follow_up_date": date_text(2),
                "notes": "【系统演示病例】过敏安全拦截。推荐提问：这个病例能否使用阿莫西林？请结合过敏史核对。",
            },
            {
                "case_no": prefix + "03",
                "patient_name": "王静怡",
                "gender": "女",
                "age": "30",
                "weight": "58",
                "pregnancy_status": "孕期（孕24周）",
                "allergy_history": "无",
                "liver_kidney_function": "正常",
                "department": "产科",
                "chief_complaint": "低热伴头痛、肌肉酸痛",
                "present_illness": "受凉后低热1天，无阴道流血、腹痛或胎动异常。",
                "past_history": "孕2产0，无慢性病史。",
                "primary_diagnosis": "妊娠期上呼吸道感染待查",
                "admission_date": date_text(0),
                "patient_status": "观察中",
                "follow_up_date": date_text(1),
                "notes": "【系统演示病例】孕期用药护栏。推荐提问：孕24周头痛，可以自行服用布洛芬吗？",
            },
            {
                "case_no": prefix + "04",
                "patient_name": "陈国强",
                "gender": "男",
                "age": "72",
                "weight": "65",
                "pregnancy_status": "不适用",
                "allergy_history": "无",
                "liver_kidney_function": "肾功能减退（eGFR 35 mL/min/1.73m²）",
                "department": "心内科",
                "chief_complaint": "房颤抗凝治疗期间出现膝关节疼痛",
                "present_illness": "长期服用华法林，近期膝关节疼痛，拟自行购买布洛芬。",
                "past_history": "房颤、高血压、慢性肾病3期。",
                "primary_diagnosis": "房颤抗凝中；骨关节炎；慢性肾病3期",
                "admission_date": date_text(-2),
                "patient_status": "治疗中",
                "follow_up_date": date_text(3),
                "notes": "【系统演示病例】配伍与肾功能双重核对。推荐提问：服用华法林期间能否同时吃布洛芬止痛？",
                "prescriptions": [
                    {"drug_name": "华法林", "specification": "2.5mg*30片", "dosage": "遵医嘱", "frequency": "每日1次", "duration": "长期", "quantity": "1盒", "route": "口服", "notes": "系统演示处方：用于配伍检查展示，不作为真实医嘱。"},
                ],
            },
            {
                "case_no": prefix + "05",
                "patient_name": "赵建国",
                "gender": "男",
                "age": "78",
                "weight": "52",
                "pregnancy_status": "不适用",
                "allergy_history": "无",
                "liver_kidney_function": "正常",
                "department": "呼吸内科",
                "chief_complaint": "突发喘息、胸闷和呼吸困难",
                "present_illness": "哮喘病史20年，今晨受凉后喘息加重，出现端坐呼吸。",
                "past_history": "支气管哮喘、高血压、2型糖尿病。",
                "primary_diagnosis": "支气管哮喘急性发作待查",
                "admission_date": date_text(0),
                "patient_status": "入院中",
                "follow_up_date": date_text(0),
                "notes": "【系统演示病例】急症风险与科室推荐。推荐提问：患者胸闷、喘息并端坐呼吸，急症风险如何，应去哪个科室？",
            },
            {
                "case_no": prefix + "06",
                "patient_name": "孙丽华",
                "gender": "女",
                "age": "62",
                "weight": "58",
                "pregnancy_status": "不适用",
                "allergy_history": "无",
                "liver_kidney_function": "正常",
                "department": "内分泌科",
                "chief_complaint": "血糖控制欠佳，需续配二甲双胍",
                "present_illness": "2型糖尿病8年，近期空腹血糖波动，现有药物即将用完。",
                "past_history": "2型糖尿病、高血压。",
                "primary_diagnosis": "2型糖尿病",
                "admission_date": date_text(-4),
                "patient_status": "已出院",
                "follow_up_date": date_text(7),
                "notes": "【系统演示病例】库存查询、处方历史与复查管理。推荐提问：查询内科二甲双胍库存余量；再打开处方管理查看历史。",
                "prescriptions": [
                    {"drug_name": "二甲双胍", "specification": "0.5g*20片", "dosage": "0.5g", "frequency": "每日2次", "duration": "14天", "quantity": "2盒", "route": "口服", "notes": "系统演示处方：仅用于界面功能展示。"},
                ],
            },
            {
                "case_no": prefix + "07",
                "patient_name": "刘秀英",
                "gender": "女",
                "age": "55",
                "weight": "60",
                "pregnancy_status": "非孕哺",
                "allergy_history": "无",
                "liver_kidney_function": "肝功能异常（ALT 120 U/L）",
                "department": "消化内科",
                "chief_complaint": "慢性乙肝患者发热伴肌肉酸痛",
                "present_illness": "慢性乙肝、肝硬化代偿期，近2天出现低热和肌肉酸痛。",
                "past_history": "慢性乙肝20年，肝硬化5年。",
                "primary_diagnosis": "慢性乙型肝炎；上呼吸道感染待查",
                "admission_date": date_text(-1),
                "patient_status": "治疗中",
                "follow_up_date": date_text(6),
                "notes": "【系统演示病例】肝功能与本地知识库检索。推荐提问：肝功能异常时使用对乙酰氨基酚需要注意什么？请引用院内资料。",
            },
        ]

        created: list[dict[str, Any]] = []
        try:
            for spec in demo_specs:
                prescriptions = spec.pop("prescriptions", [])
                spec["attending_doctor"] = doctor_name
                spec["doctor_name"] = doctor_name
                with self._lock:
                    row = self._conn.execute(
                        "SELECT * FROM cases WHERE case_no=?", (spec["case_no"],)
                    ).fetchone()
                case = dict(row) if row else self.create_case(spec)
                if not row:
                    created.append(case)
                existing_prescriptions = self.list_prescriptions(case["id"])
                existing_demo_drugs = {
                    item["drug_name"] for item in existing_prescriptions
                    if str(item.get("notes", "")).startswith("系统演示处方")
                }
                for prescription in prescriptions:
                    if prescription["drug_name"] in existing_demo_drugs:
                        continue
                    prescription["prescribing_doctor"] = doctor_name
                    self.create_prescription(case["id"], prescription)
            self.set_meta(marker_key, now_iso())
        except Exception:
            # Keep the marker unset so a later startup can complete the set.
            raise
        return created

    def delete_case(self, case_id: str) -> None:
        """Delete a case and detach it from every linked session."""
        with self._lock, self._conn:
            self._conn.execute("UPDATE sessions SET case_id=NULL WHERE case_id=?", (case_id,))
            self._conn.execute("DELETE FROM cases WHERE id=?", (case_id,))

    def bind_session_case(self, session_id: str, case_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE sessions SET case_id=? WHERE id=?", (case_id, session_id))

    def clear_session_case(self, session_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE sessions SET case_id=NULL WHERE id=?", (session_id,))

    def get_session_case(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT c.* FROM sessions s JOIN cases c ON c.id = s.case_id WHERE s.id=?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    # ----------------------------------------------------------- prescriptions

    def create_prescription(self, case_id: str, data: dict[str, Any]) -> dict[str, Any]:
        stamp = now_iso()
        prescribed_at = str(data.get("prescribed_at") or stamp)
        fields = ["drug_name", "specification", "dosage", "frequency", "duration",
                  "quantity", "route", "prescribing_doctor", "notes"]
        values = {field: str(data.get(field, "")).strip() for field in fields}
        columns = ["case_id", *fields, "prescribed_at", "created_at"]
        placeholders = ",".join("?" for _ in columns)
        params = [case_id, *(values[field] for field in fields), prescribed_at, stamp]
        with self._lock, self._conn:
            cur = self._conn.execute(
                f"INSERT INTO prescriptions({','.join(columns)}) VALUES({placeholders})", params
            )
            rx_id = cur.lastrowid
        return self.get_prescription(rx_id)

    def get_prescription(self, rx_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM prescriptions WHERE id=?", (rx_id,)).fetchone()
        return dict(row) if row else None

    def list_prescriptions(self, case_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM prescriptions WHERE case_id=? ORDER BY prescribed_at DESC, id DESC",
                (case_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_prescription(self, rx_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM prescriptions WHERE id=?", (rx_id,))

    def update_case_status(self, case_id: str, patient_status: str = "", admission_date: str = "",
                           follow_up_date: str = "", session_id: str | None = None) -> dict[str, Any] | None:
        """Update patient status and/or key dates without rewriting the whole case."""
        case = self.get_case(case_id)
        if not case:
            return None
        data = dict(case)
        changes = []
        if patient_status:
            data["patient_status"] = patient_status
            changes.append(f"状况={patient_status}")
        if admission_date:
            data["admission_date"] = admission_date
            changes.append(f"入院时间={admission_date}")
        if follow_up_date:
            data["follow_up_date"] = follow_up_date
            changes.append(f"复查时间={follow_up_date}")
        result = self.update_case(case_id, data)
        if changes:
            self.add_audit("case_status", f"更新病例 {case['case_no']}：" + "，".join(changes), session_id)
        return result

    def get_case_stats(self, doctor_name: str = "") -> dict[str, int]:
        """Startup dashboard: total cases, in-hospital patients, pending and today's follow-ups."""
        today = datetime.now().strftime("%Y-%m-%d")
        doctor_filter = " AND doctor_name=?" if doctor_name else ""
        params = [doctor_name] if doctor_name else []
        with self._lock:
            total = self._conn.execute(f"SELECT COUNT(*) FROM cases WHERE 1=1{doctor_filter}", params).fetchone()[0]
            in_hospital = self._conn.execute(
                f"SELECT COUNT(*) FROM cases WHERE patient_status NOT IN ('已出院','死亡'){doctor_filter}", params
            ).fetchone()[0]
            pending_followup = self._conn.execute(
                f"SELECT COUNT(*) FROM cases WHERE follow_up_date != '' AND substr(follow_up_date,1,10) >= ?{doctor_filter}",
                [today, *params],
            ).fetchone()[0]
            today_followup = self._conn.execute(
                f"SELECT COUNT(*) FROM cases WHERE substr(follow_up_date,1,10) = ?{doctor_filter}",
                [today, *params],
            ).fetchone()[0]
        return {"total": total, "in_hospital": in_hospital,
                "pending_followup": pending_followup, "today_followup": today_followup}

    def add_message(self, session_id: str, role: str, content: str, sources: list[dict[str, Any]] | None = None) -> None:
        stamp = now_iso()
        with self._lock, self._conn:
            self._conn.execute("INSERT OR IGNORE INTO sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)", (session_id, content[:30] if role == "user" else "新建会话", stamp, stamp))
            self._conn.execute("INSERT INTO messages(session_id,role,content,sources_json,created_at) VALUES(?,?,?,?,?)", (session_id, role, content, json.dumps(sources or [], ensure_ascii=False), stamp))
            self._conn.execute("UPDATE sessions SET updated_at=?, title=CASE WHEN title='新建会话' AND ?='user' THEN ? ELSE title END WHERE id=?", (stamp, role, content[:30], session_id))

    def add_audit(self, event_type: str, detail: str, session_id: str | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT INTO audit_logs(session_id,event_type,detail,created_at) VALUES(?,?,?,?)", (session_id, event_type, detail, now_iso()))

    def list_audit(self, session_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            if session_id:
                rows = self._conn.execute("SELECT event_type,detail,created_at FROM audit_logs WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit)).fetchall()
            else:
                rows = self._conn.execute("SELECT event_type,detail,created_at FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def seed_inventory(self) -> None:
        with self._lock, self._conn:
            # Seed only missing demo rows. Existing hospital/imported quantities
            # are never overwritten during application startup.
            items = [
                ("急诊科", "对乙酰氨基酚", "0.5g*20片", 86, 20, "盒"),
                ("急诊科", "阿司匹林", "100mg*30片", 12, 20, "盒"),
                ("内科", "华法林", "2.5mg*30片", 8, 10, "盒"),
                ("内科", "二甲双胍", "0.5g*20片", 64, 15, "盒"),
                ("儿科", "布洛芬混悬液", "100ml:2g", 4, 8, "瓶"),
                ("外科", "头孢曲松", "1g/瓶", 0, 10, "瓶"),
                ("急诊科", "硝酸甘油", "0.5mg*100片", 32, 10, "盒"),
                ("急诊科", "肾上腺素注射液", "1mg/ml*10支", 18, 10, "盒"),
                ("急诊科", "沙丁胺醇吸入气雾剂", "100μg*200揿", 7, 10, "瓶"),
                ("急诊科", "地塞米松磷酸钠注射液", "5mg/ml*10支", 24, 8, "盒"),
                ("急诊科", "氯化钠注射液", "0.9% 250ml*40袋", 120, 30, "箱"),
                ("内科", "阿莫西林", "0.5g*24粒", 45, 15, "盒"),
                ("内科", "奥美拉唑", "20mg*28粒", 9, 15, "盒"),
                ("内科", "阿托伐他汀", "20mg*14片", 36, 12, "盒"),
                ("内科", "恩格列净", "10mg*10片", 22, 8, "盒"),
                ("儿科", "阿莫西林颗粒", "0.125g*12袋", 6, 10, "盒"),
                ("儿科", "对乙酰氨基酚混悬滴剂", "15ml", 18, 8, "瓶"),
                ("儿科", "口服补液盐III", "5.125g*10袋", 40, 12, "盒"),
                ("呼吸内科", "布地奈德混悬液", "1mg/2ml*5支", 5, 8, "盒"),
                ("呼吸内科", "沙丁胺醇雾化吸入溶液", "2.5mg/2.5ml*10支", 14, 8, "盒"),
                ("呼吸内科", "氨溴索", "30mg*20片", 28, 10, "盒"),
                ("心内科", "氯吡格雷", "75mg*28片", 16, 10, "盒"),
                ("心内科", "呋塞米", "20mg*100片", 11, 15, "瓶"),
                ("心内科", "硝酸异山梨酯", "5mg*100片", 20, 8, "瓶"),
                ("妇产科", "叶酸", "0.4mg*31片", 52, 15, "盒"),
                ("妇产科", "硫酸镁注射液", "10ml*5支", 12, 8, "盒"),
                ("妇产科", "对乙酰氨基酚", "0.5g*20片", 18, 10, "盒"),
                ("外科", "头孢呋辛", "0.75g/瓶", 9, 12, "瓶"),
                ("外科", "破伤风抗毒素", "1500IU*10支", 6, 8, "盒"),
                ("外科", "利多卡因注射液", "5ml:0.1g*10支", 23, 8, "盒"),
            ]
            self._conn.executemany(
                """
                INSERT OR IGNORE INTO inventory(
                    department,drug_name,specification,quantity,reorder_level,unit,updated_at
                ) VALUES(?,?,?,?,?,?,?)
                """,
                [(*item, now_iso()) for item in items],
            )

    def upsert_inventory(self, item: InventoryItem) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO inventory(department,drug_name,specification,quantity,reorder_level,unit,updated_at)
                VALUES(?,?,?,?,?,?,?) ON CONFLICT(department,drug_name,specification) DO UPDATE SET quantity=excluded.quantity,reorder_level=excluded.reorder_level,unit=excluded.unit,updated_at=excluded.updated_at""",
                (item.department, item.drug_name, item.specification, item.quantity, item.reorder_level, item.unit, now_iso()),
            )

    def query_inventory(self, drug_name: str = "", department: str = "", low_only: bool = False) -> list[InventoryItem]:
        clauses: list[str] = []
        params: list[Any] = []
        if drug_name:
            clauses.append("drug_name LIKE ?")
            params.append(f"%{drug_name}%")
        if department:
            clauses.append("department LIKE ?")
            params.append(f"%{department}%")
        if low_only:
            clauses.append("quantity <= reorder_level")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM inventory" + where + " ORDER BY quantity ASC, department", params).fetchall()
        return [InventoryItem(row["id"], row["department"], row["drug_name"], row["specification"], row["quantity"], row["reorder_level"], row["unit"], row["updated_at"]) for row in rows]

    def import_inventory(self, rows: Iterable[dict[str, Any]]) -> int:
        count = 0
        for row in rows:
            try:
                self.upsert_inventory(InventoryItem(None, str(row["department"]), str(row["drug_name"]), str(row.get("specification", "")), int(row["quantity"]), int(row.get("reorder_level", 10)), str(row.get("unit", "盒"))))
                count += 1
            except (KeyError, TypeError, ValueError):
                continue
        return count

