"""PyQt desktop interface for the local medical assistant."""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

try:  # PyQt6 is the supported runtime; PyQt5 keeps the demo runnable on older hospital PCs.
    from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
    from PyQt6.QtGui import QAction, QKeySequence
    from PyQt6.QtWidgets import (
        QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDockWidget, QFileDialog,
        QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
        QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSlider,
        QSpinBox, QSplitter, QStatusBar, QTableWidget, QTableWidgetItem, QTextBrowser, QTextEdit,
        QVBoxLayout, QWidget,
    )
    QT6 = True
except ImportError:  # pragma: no cover - depends on machine environment
    from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
    from PyQt5.QtGui import QKeySequence
    from PyQt5.QtWidgets import (
        QAbstractItemView, QAction, QApplication, QCheckBox, QComboBox, QDialog, QDockWidget,
        QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
        QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
        QSlider, QSpinBox, QSplitter, QStatusBar, QTableWidget, QTableWidgetItem, QTextBrowser,
        QTextEdit, QVBoxLayout, QWidget,
    )
    QT6 = False

from .agent import LocalAgent, is_ollama_running, launch_ollama_detached
from .auth import account_names, verify_login
from .config import AppConfig
from .models import AgentEvent, AgentResult
from .rag import DocumentStore
from .storage import SQLiteStore


if QT6:
    _RETURN_KEYS = {Qt.Key.Key_Return, Qt.Key.Key_Enter}
    _SHIFT_MODIFIER = Qt.KeyboardModifier.ShiftModifier
    _SELECT_ROWS = QAbstractItemView.SelectionBehavior.SelectRows
    _SINGLE_SELECTION = QAbstractItemView.SelectionMode.SingleSelection
    _HEADER_STRETCH = QHeaderView.ResizeMode.Stretch
else:
    _RETURN_KEYS = {Qt.Key_Return, Qt.Key_Enter}
    _SHIFT_MODIFIER = Qt.ShiftModifier
    _SELECT_ROWS = QAbstractItemView.SelectRows
    _SINGLE_SELECTION = QAbstractItemView.SingleSelection
    _HEADER_STRETCH = QHeaderView.Stretch


class ChatInput(QTextEdit):
    """Multiline input where Enter sends and Shift+Enter inserts a newline."""

    send_requested = pyqtSignal()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if event.key() in _RETURN_KEYS and not event.modifiers() & _SHIFT_MODIFIER:
            self.send_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class AgentWorker(QThread):
    event = pyqtSignal(object)
    result = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config: AppConfig, store: SQLiteStore, rag: DocumentStore, query: str, session_id: str, history: list[dict], patient: dict | None = None):
        super().__init__()
        self.config, self.store, self.rag = config, store, rag
        self.query, self.session_id, self.history, self.patient = query, session_id, history, patient

    def run(self) -> None:
        try:
            agent = LocalAgent(self.config, self.store, self.rag, self.event.emit)
            self.result.emit(agent.run(self.query, self.session_id, self.history, patient=self.patient))
        except Exception as exc:  # surface worker errors without crashing the GUI thread
            self.failed.emit(str(exc))


class CaseEditDialog(QDialog):
    """Modal form for creating / editing a patient case."""

    REQUIRED_FIELDS = {"age", "weight", "pregnancy_status", "allergy_history", "liver_kidney_function"}

    FIELD_LABELS: list[tuple[str, str]] = [
        ("case_no", "病例号"),
        ("patient_name", "姓名"),
        ("gender", "性别"),
        ("age", "年龄"),
        ("weight", "体重（kg）"),
        ("pregnancy_status", "孕哺状态"),
        ("allergy_history", "过敏史"),
        ("liver_kidney_function", "肝肾功能"),
        ("department", "就诊科室"),
        ("chief_complaint", "主诉"),
        ("present_illness", "现病史"),
        ("past_history", "既往史"),
        ("primary_diagnosis", "初步诊断"),
        ("attending_doctor", "收治医生"),
        ("admission_date", "入院时间"),
        ("patient_status", "病人状况"),
        ("follow_up_date", "复查时间"),
        ("notes", "备注"),
    ]

    MULTILINE_FIELDS = {"allergy_history", "chief_complaint", "present_illness", "past_history", "notes"}

    def __init__(
        self,
        store: SQLiteStore,
        prefill: dict | None = None,
        edit_id: str = "",
        log_callback=None,
        doctor_name: str = "",
    ):
        super().__init__()
        self.store = store
        self.edit_id = edit_id
        self.log_callback = log_callback
        self.doctor_name = doctor_name
        self.saved_case: dict | None = None
        self.widgets: dict[str, QWidget] = {}
        self.setWindowTitle("编辑病例" if edit_id else "新建病例")
        self.resize(460, 640)
        self._build_form(prefill or {})
        self._build_buttons()

    def _build_form(self, prefill: dict) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight if QT6 else Qt.AlignRight)
        for field, label_text in self.FIELD_LABELS:
            widget = self._create_widget(field)
            self.widgets[field] = widget
            value = prefill.get(field, "")
            self._set_widget_value(field, widget, value)
            caption = label_text
            if field in self.REQUIRED_FIELDS:
                caption = f'<span style="color:#d23b3b">*</span> {label_text}'
            label = QLabel(caption)
            form.addRow(label, widget)
        root.addLayout(form)

    def _create_widget(self, field: str) -> QWidget:
        if field == "gender":
            widget = QComboBox()
            widget.addItems(["男", "女"])
        elif field == "pregnancy_status":
            widget = QComboBox()
            widget.addItems(["非孕哺", "孕期", "哺乳期", "不适用"])
        elif field == "liver_kidney_function":
            widget = QComboBox()
            widget.setEditable(True)
            widget.addItems(["正常", "肝功能异常", "肾功能异常", "肝肾均异常"])
        elif field == "patient_status":
            widget = QComboBox()
            widget.setEditable(True)
            widget.addItems(["入院中", "观察中", "治疗中", "已出院", "死亡"])
        elif field in self.MULTILINE_FIELDS:
            widget = QTextEdit()
            widget.setFixedHeight(72 if field == "present_illness" else 54)
        else:
            widget = QLineEdit()
            if field == "age":
                widget.setPlaceholderText("如 45")
            elif field == "weight":
                widget.setPlaceholderText("如 70")
            elif field == "admission_date":
                widget.setPlaceholderText("如 2026-09-27")
            elif field == "follow_up_date":
                widget.setPlaceholderText("如 2026-10-04")
        if field == "case_no":
            widget.setPlaceholderText("保存时自动生成")
        return widget

    @staticmethod
    def _set_widget_value(field: str, widget: QWidget, value: Any) -> None:
        if isinstance(widget, QLineEdit):
            widget.setText(str(value))
        elif isinstance(widget, QComboBox):
            text = str(value)
            if text:
                index = widget.findText(text)
                if index >= 0:
                    widget.setCurrentIndex(index)
                elif widget.isEditable():
                    widget.setEditText(text)
        elif isinstance(widget, QTextEdit):
            widget.setPlainText(str(value))

    def _build_buttons(self) -> None:
        row = QHBoxLayout()
        row.addStretch()
        save_button = QPushButton("保存")
        save_button.setObjectName("primaryButton")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        row.addWidget(save_button)
        row.addWidget(cancel_button)
        self.layout().addLayout(row)

    def _read_widget(self, field: str) -> str:
        widget = self.widgets[field]
        if isinstance(widget, QLineEdit):
            return widget.text().strip()
        if isinstance(widget, QComboBox):
            return widget.currentText().strip()
        if isinstance(widget, QTextEdit):
            return widget.toPlainText().strip()
        return ""

    def collect_data(self) -> dict[str, str]:
        return {field: self._read_widget(field) for field, _ in self.FIELD_LABELS}

    def accept(self) -> None:  # noqa: N802 - Qt API name
        data = self.collect_data()
        missing = [label for field, label in self.FIELD_LABELS if field in self.REQUIRED_FIELDS and not data[field]]
        if missing:
            QMessageBox.warning(self, "必填项不完整", "请填写标星必填项：" + "、".join(missing))
            for field, _ in self.FIELD_LABELS:
                if field in self.REQUIRED_FIELDS and not data[field]:
                    self.widgets[field].setFocus()
                    break
            return
        try:
            if self.edit_id:
                self.saved_case = self.store.update_case(self.edit_id, data)
                event_type, action = "case_update", "编辑"
            else:
                data["doctor_name"] = self.doctor_name
                self.saved_case = self.store.create_case(data)
                event_type, action = "case_create", "新建"
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "保存失败", f"病例号 {data['case_no']} 已存在，请使用其他病例号。")
            return
        case_no = self.saved_case["case_no"]
        self.store.add_audit(event_type, f"{action}病例：{case_no}（{data['patient_name']}）")
        if self.log_callback:
            self.log_callback(f"{action}病例成功：{case_no}")
        super().accept()


class CaseManagerDialog(QDialog):
    """Search / create / edit / duplicate / delete cases and bind one to the active chat session."""

    COLUMNS: list[tuple[str, str]] = [
        ("case_no", "病例号"), ("patient_name", "姓名"), ("gender", "性别"),
        ("age", "年龄"), ("department", "就诊科室"), ("primary_diagnosis", "初步诊断"),
    ]

    def __init__(self, main_window: "ChatWindow", keyword: str = ""):
        super().__init__(main_window)
        self.main_window = main_window
        self.store = main_window.store
        self.setWindowTitle("病例管理")
        self.resize(780, 540)
        self._build_ui(keyword)
        self._reload()

    def _build_ui(self, keyword: str) -> None:
        root = QVBoxLayout(self)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("搜索病例"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("输入病例号或姓名快速过滤")
        self.search_edit.setText(keyword)
        self.search_edit.textChanged.connect(lambda _: self._reload())
        self.search_edit.returnPressed.connect(lambda: self._reload())
        search_row.addWidget(self.search_edit, 1)
        search_button = QPushButton("搜索")
        search_button.clicked.connect(lambda: self._reload())
        search_row.addWidget(search_button)
        root.addLayout(search_row)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _, label in self.COLUMNS])
        self.table.setSelectionMode(_SINGLE_SELECTION)
        self.table.setSelectionBehavior(_SELECT_ROWS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers if QT6 else QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(_HEADER_STRETCH)
        self.table.verticalHeader().setVisible(False)
        self.table.doubleClicked.connect(lambda _: self._show_case_detail())
        root.addWidget(self.table, 1)

        self.hint = QLabel("")
        self.hint.setObjectName("muted")
        root.addWidget(self.hint)

        buttons = QHBoxLayout()
        new_button = QPushButton("新建病例")
        new_button.clicked.connect(self._new_case)
        edit_button = QPushButton("编辑病例")
        edit_button.clicked.connect(self._edit_case)
        copy_button = QPushButton("复制病例")
        copy_button.clicked.connect(self._copy_case)
        delete_button = QPushButton("删除病例")
        delete_button.clicked.connect(self._delete_case)
        buttons.addWidget(new_button)
        buttons.addWidget(edit_button)
        buttons.addWidget(copy_button)
        buttons.addWidget(delete_button)
        buttons.addStretch()
        bind_button = QPushButton("绑定病例")
        bind_button.setObjectName("primaryButton")
        bind_button.clicked.connect(self._bind_case)
        buttons.addWidget(bind_button)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        buttons.addWidget(close_button)
        root.addLayout(buttons)

    def _reload(self, select_case_id: str = "") -> None:
        self.cases = self.store.list_cases(self.search_edit.text(), doctor_name=self.main_window.doctor_name)
        self.table.setRowCount(len(self.cases))
        target_row = 0
        for row, case in enumerate(self.cases):
            for column, (key, _) in enumerate(self.COLUMNS):
                item = QTableWidgetItem(str(case.get(key, "")))
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole if QT6 else Qt.UserRole, case["id"])
                self.table.setItem(row, column, item)
            if select_case_id and case["id"] == select_case_id:
                target_row = row
        if self.cases:
            self.table.selectRow(target_row)
        self.table.setEnabled(bool(self.cases))

    def _selected_case(self) -> dict | None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.cases):
            return None
        return self.cases[row]

    def _open_editor(self, case: dict | None, copy_mode: bool = False) -> None:
        if copy_mode and case:
            prefill = {key: case.get(key, "") for key in self.store.CASE_FIELDS}
            prefill["case_no"] = ""
            prefill["patient_name"] = ""
            dialog = CaseEditDialog(self.store, prefill=prefill, log_callback=self._gui_log,
                                    doctor_name=self.main_window.doctor_name)
        elif case:
            dialog = CaseEditDialog(self.store, prefill=case, edit_id=case["id"], log_callback=self._gui_log,
                                    doctor_name=self.main_window.doctor_name)
        else:
            dialog = CaseEditDialog(self.store, log_callback=self._gui_log,
                                    doctor_name=self.main_window.doctor_name)
        if dialog.exec() and dialog.saved_case:
            self.search_edit.clear()
            self._reload(dialog.saved_case["id"])
            self.main_window._load_sessions()
            self.main_window._update_current_case_label()

    def _new_case(self) -> None:
        self._open_editor(None)

    def _edit_case(self) -> None:
        case = self._selected_case()
        if case:
            self._open_editor(case)

    def _show_case_detail(self) -> None:
        case = self._selected_case()
        if not case:
            return
        dialog = CaseDetailDialog(self.main_window, case=case)
        dialog.exec()
        self._reload(case["id"])
        self.main_window._load_sessions()
        self.main_window._update_current_case_label()

    def _copy_case(self) -> None:
        case = self._selected_case()
        if case:
            self._open_editor(case, copy_mode=True)

    def _delete_case(self) -> None:
        case = self._selected_case()
        if not case:
            return
        confirmed = QMessageBox.question(
            self, "删除病例",
            f"确定删除病例 {case['case_no']}（{case['patient_name']}）吗？\n删除后将同时解除该病例与会话的绑定关系。",
        )
        yes_value = QMessageBox.StandardButton.Yes if QT6 else QMessageBox.Yes
        if confirmed != yes_value:
            return
        self.store.delete_case(case["id"])
        self.store.add_audit("case_delete", f"删除病例：{case['case_no']}（{case['patient_name']}）")
        self._gui_log(f"删除病例：{case['case_no']}")
        self._reload()
        self.main_window._load_sessions()
        self.main_window._update_current_case_label()

    def _bind_case(self) -> None:
        case = self._selected_case()
        if not case:
            QMessageBox.information(self, "无法绑定", "请先选择一个病例。")
            return
        self.main_window.set_current_case(case)
        self.accept()

    def _gui_log(self, detail: str) -> None:
        self.main_window._append_log(AgentEvent("case", "status", detail))


class PrescriptionManagerDialog(QDialog):
    """Prescribe drugs for a bound patient and review the prescription history."""

    COLUMNS: list[tuple[str, str]] = [
        ("drug_name", "药品名称"), ("specification", "规格"), ("dosage", "单次剂量"),
        ("frequency", "频次"), ("duration", "疗程"), ("route", "途径"),
        ("quantity", "数量"), ("prescribing_doctor", "开方医生"),
        ("prescribed_at", "开方时间"), ("notes", "备注"),
    ]

    STATUS_OPTIONS = ["入院中", "观察中", "治疗中", "已出院", "死亡"]

    def __init__(self, main_window: "ChatWindow"):
        super().__init__(main_window)
        self.main_window = main_window
        self.store = main_window.store
        self.case: dict | None = main_window.store.get_session_case(main_window.session_id) if main_window.session_id else None
        if not self.case:
            QMessageBox.information(self, "无法开药", "请先绑定病例后再进行处方管理。")
            return
        self.setWindowTitle(f"处方管理 - {self.case['patient_name']}（{self.case['case_no']}）")
        self.resize(920, 720)
        self.setMinimumSize(760, 520)
        self._build_ui()
        self._reload()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame if QT6 else QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if QT6 else Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if QT6 else Qt.ScrollBarAsNeeded)
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(4, 4, 4, 8)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # Patient summary + status editor
        info = QFrame()
        info.setObjectName("patientInfo")
        info_layout = QFormLayout(info)
        info_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight if QT6 else Qt.AlignRight)
        info_layout.addRow("姓名", QLabel(self.case["patient_name"]))
        info_layout.addRow("病例号", QLabel(self.case["case_no"]))
        info_layout.addRow("性别/年龄", QLabel(f"{self.case['gender']} / {self.case['age']} 岁"))
        info_layout.addRow("体重", QLabel(f"{self.case['weight']} kg"))
        info_layout.addRow("过敏史", QLabel(self.case["allergy_history"] or "无"))
        info_layout.addRow("孕哺/肝肾", QLabel(f"{self.case['pregnancy_status']} / {self.case['liver_kidney_function']}"))

        self.admission_edit = QLineEdit(self.case.get("admission_date", ""))
        self.admission_edit.setPlaceholderText("如 2026-09-27")
        info_layout.addRow("入院时间", self.admission_edit)

        self.follow_up_edit = QLineEdit(self.case.get("follow_up_date", ""))
        self.follow_up_edit.setPlaceholderText("如 2026-10-04")
        info_layout.addRow("复查时间", self.follow_up_edit)

        self.status_combo = QComboBox()
        self.status_combo.setEditable(True)
        self.status_combo.addItems(self.STATUS_OPTIONS)
        status_text = self.case.get("patient_status", "入院中")
        idx = self.status_combo.findText(status_text)
        if idx >= 0:
            self.status_combo.setCurrentIndex(idx)
        else:
            self.status_combo.setEditText(status_text)
        info_layout.addRow("病人状况", self.status_combo)

        save_status_btn = QPushButton("保存状况与时间节点")
        save_status_btn.clicked.connect(self._save_status)
        info_layout.addRow("", save_status_btn)
        root.addWidget(info)

        # Prescription form
        form_box = QGroupBox("开处方")
        form = QFormLayout(form_box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight if QT6 else Qt.AlignRight)
        self.drug_edit = QLineEdit()
        self.drug_edit.setPlaceholderText("如 对乙酰氨基酚")
        self.spec_edit = QLineEdit()
        self.spec_edit.setPlaceholderText("如 0.5g*20片")
        self.dosage_edit = QLineEdit()
        self.dosage_edit.setPlaceholderText("如 200 mg")
        self.freq_edit = QLineEdit()
        self.freq_edit.setPlaceholderText("如 每日 4 次")
        self.duration_edit = QLineEdit()
        self.duration_edit.setPlaceholderText("如 3 天")
        self.route_combo = QComboBox()
        self.route_combo.setEditable(True)
        self.route_combo.addItems(["口服", "静脉滴注", "静脉注射", "肌肉注射", "皮下注射", "外用", "吸入"])
        self.quantity_edit = QLineEdit()
        self.quantity_edit.setPlaceholderText("如 2 盒")
        self.doctor_edit = QLineEdit(self.main_window.doctor_name)
        self.doctor_edit.setPlaceholderText("开方医生姓名")
        self.rx_notes_edit = QLineEdit()
        self.rx_notes_edit.setPlaceholderText("用药说明/注意事项")
        self.rx_time_edit = QLineEdit()
        self.rx_time_edit.setPlaceholderText("留空则使用当前时间")
        form.addRow("药品名称 *", self.drug_edit)
        form.addRow("规格", self.spec_edit)
        form.addRow("单次剂量", self.dosage_edit)
        form.addRow("频次", self.freq_edit)
        form.addRow("疗程", self.duration_edit)
        form.addRow("给药途径", self.route_combo)
        form.addRow("数量", self.quantity_edit)
        form.addRow("开方医生", self.doctor_edit)
        form.addRow("开方时间", self.rx_time_edit)
        form.addRow("备注", self.rx_notes_edit)
        rx_btn = QPushButton("开具处方")
        rx_btn.setObjectName("primaryButton")
        rx_btn.clicked.connect(self._add_prescription)
        form.addRow("", rx_btn)
        root.addWidget(form_box)

        # History table
        root.addWidget(QLabel("处方记录"))
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _, label in self.COLUMNS])
        self.table.setSelectionBehavior(_SELECT_ROWS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers if QT6 else QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(_HEADER_STRETCH)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(180)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if QT6 else Qt.ScrollBarAsNeeded)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if QT6 else Qt.ScrollBarAsNeeded)
        root.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        del_btn = QPushButton("删除选中处方")
        del_btn.clicked.connect(self._delete_prescription)
        buttons.addWidget(del_btn)
        buttons.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        outer.addLayout(buttons)

    def _save_status(self) -> None:
        admission = self.admission_edit.text().strip()
        follow_up = self.follow_up_edit.text().strip()
        status = self.status_combo.currentText().strip()
        self.store.update_case_status(self.case["id"], patient_status=status, admission_date=admission,
                                       follow_up_date=follow_up, session_id=self.main_window.session_id)
        self.case = self.store.get_case(self.case["id"])
        self.main_window._update_current_case_label()
        QMessageBox.information(self, "已保存", "病人状况与时间节点已更新。")

    def _add_prescription(self) -> None:
        drug = self.drug_edit.text().strip()
        if not drug:
            QMessageBox.warning(self, "缺少药品名称", "请填写药品名称后再开具处方。")
            return
        data = {
            "drug_name": drug,
            "specification": self.spec_edit.text().strip(),
            "dosage": self.dosage_edit.text().strip(),
            "frequency": self.freq_edit.text().strip(),
            "duration": self.duration_edit.text().strip(),
            "route": self.route_combo.currentText().strip(),
            "quantity": self.quantity_edit.text().strip(),
            "prescribing_doctor": self.doctor_edit.text().strip(),
            "notes": self.rx_notes_edit.text().strip(),
            "prescribed_at": self.rx_time_edit.text().strip(),
        }
        rx = self.store.create_prescription(self.case["id"], data)
        self.store.add_audit(
            "prescription",
            f"为 {self.case['patient_name']} 开具处方：{drug} {data['dosage']} {data['frequency']}",
            self.main_window.session_id,
        )
        # Clear form for next entry, keep doctor/route
        self.drug_edit.clear()
        self.spec_edit.clear()
        self.dosage_edit.clear()
        self.freq_edit.clear()
        self.duration_edit.clear()
        self.quantity_edit.clear()
        self.rx_notes_edit.clear()
        self._reload()

    def _delete_prescription(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.prescriptions):
            return
        rx = self.prescriptions[row]
        confirmed = QMessageBox.question(
            self, "删除处方",
            f"确定删除处方「{rx['drug_name']}」（开方时间 {rx['prescribed_at'][:16]}）吗？",
        )
        yes_value = QMessageBox.StandardButton.Yes if QT6 else QMessageBox.Yes
        if confirmed != yes_value:
            return
        self.store.delete_prescription(rx["id"])
        self.store.add_audit("prescription_delete", f"删除处方：{rx['drug_name']}", self.main_window.session_id)
        self._reload()

    def _reload(self) -> None:
        self.prescriptions = self.store.list_prescriptions(self.case["id"])
        self.table.setRowCount(len(self.prescriptions))
        for row, rx in enumerate(self.prescriptions):
            for column, (key, _) in enumerate(self.COLUMNS):
                value = rx.get(key, "")
                if key == "prescribed_at":
                    value = str(value)[:16].replace("T", " ")
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole if QT6 else Qt.UserRole, rx["id"])
                self.table.setItem(row, column, item)
        self.table.setEnabled(bool(self.prescriptions))


class CenteredDockWidget(QDockWidget):
    """Floating dock that centers itself over its parent window whenever shown."""

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().showEvent(event)
        if not self.isFloating():
            return
        parent = self.parentWidget()
        if parent is not None:
            target_center = parent.frameGeometry().center()
        else:
            target_center = QApplication.primaryScreen().availableGeometry().center()
        frame = self.frameGeometry()
        frame.moveCenter(target_center)
        self.move(frame.topLeft())


class LoginDialog(QDialog):
    """Doctor login with developer-managed built-in accounts (no registration)."""

    def __init__(self):
        super().__init__()
        self.doctor_name = ""
        self.setWindowTitle("医生登录")
        self.setFixedSize(380, 280)

        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 20)

        title = QLabel("院内临床智能辅助系统")
        title.setStyleSheet("font-size:17px; font-weight:700; color:#17324d;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)
        subtitle = QLabel("请使用医生账号登录")
        subtitle.setStyleSheet("color:#667785; font-size:12px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(subtitle)
        root.addSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight if QT6 else Qt.AlignRight)
        self.name_combo = QComboBox()
        self.name_combo.setEditable(True)
        self.name_combo.addItems(account_names())
        form.addRow("医生姓名", self.name_combo)
        self.password_edit = QLineEdit()
        echo_password = QLineEdit.EchoMode.Password if QT6 else QLineEdit.Password
        self.password_edit.setEchoMode(echo_password)
        self.password_edit.returnPressed.connect(self._try_login)
        form.addRow("登录密码", self.password_edit)
        root.addLayout(form)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color:#c62828; font-size:11px;")
        root.addWidget(self.error_label)
        root.addStretch()

        login_btn = QPushButton("登录")
        login_btn.setObjectName("primaryButton")
        login_btn.setMinimumHeight(34)
        login_btn.clicked.connect(self._try_login)
        root.addWidget(login_btn)

    def _try_login(self) -> None:
        name = self.name_combo.currentText().strip()
        password = self.password_edit.text()
        if verify_login(name, password):
            self.doctor_name = name
            self.accept()
        else:
            self.error_label.setText("医生姓名或密码错误，账号仅由开发者预置。")
            self.password_edit.selectAll()
            self.password_edit.setFocus()


class CaseDetailDialog(QDialog):
    """Read-only details for either a specified case or the current binding."""

    DETAIL_ROWS: list[tuple[str, str]] = [
        ("patient_name", "姓名"), ("case_no", "病例号"), ("gender", "性别"),
        ("age", "年龄"), ("weight", "体重（kg）"),
        ("pregnancy_status", "孕哺状态"), ("allergy_history", "过敏史"),
        ("liver_kidney_function", "肝肾功能"), ("department", "就诊科室"),
        ("chief_complaint", "主诉"), ("present_illness", "现病史"),
        ("past_history", "既往史"), ("primary_diagnosis", "初步诊断"),
        ("attending_doctor", "收治医生"), ("admission_date", "入院时间"),
        ("patient_status", "病人状况"), ("follow_up_date", "复查时间"),
        ("notes", "备注"), ("created_at", "创建时间"), ("updated_at", "更新时间"),
    ]

    def __init__(self, main_window: "ChatWindow", case: dict | None = None):
        super().__init__(main_window)
        self.main_window = main_window
        self.case_id = str(case.get("id", "")) if case else ""
        self.setWindowTitle("病例详情")
        self.resize(520, 660)
        self._build_ui()

    def _current_case(self) -> dict | None:
        if self.case_id:
            return self.main_window.store.get_case(self.case_id)
        if self.main_window.session_id:
            return self.main_window.store.get_session_case(self.main_window.session_id)
        return None

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)

        title = QLabel("病例详情")
        title.setStyleSheet("font-size:17px; font-weight:700; color:#17324d;")
        root.addWidget(title)
        root.addSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight if QT6 else Qt.AlignRight)
        case = self._current_case()
        for key, label in self.DETAIL_ROWS:
            value = str(case.get(key, "")) if case else ""
            field = QLabel(value or "—")
            field.setWordWrap(True)
            field.setStyleSheet("color:#17212b; background:transparent; border:none;")
            form.addRow(label, field)
        root.addLayout(form)
        root.addStretch()

        buttons = QHBoxLayout()
        pick_btn = QPushButton("选择病例")
        pick_btn.clicked.connect(self._pick_case)
        edit_btn = QPushButton("编辑病例")
        edit_btn.clicked.connect(self._edit_case)
        buttons.addWidget(pick_btn)
        buttons.addWidget(edit_btn)
        buttons.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        root.addLayout(buttons)

    def _pick_case(self) -> None:
        manager = CaseManagerDialog(self.main_window)
        manager.exec()
        case = self.main_window.store.get_session_case(self.main_window.session_id) if self.main_window.session_id else None
        self.case_id = str(case.get("id", "")) if case else ""
        self._refresh()

    def _edit_case(self) -> None:
        case = self._current_case()
        if not case:
            QMessageBox.information(self, "无当前病例", "请先选择一个病例。")
            return
        dialog = CaseEditDialog(self.main_window.store, prefill=case, edit_id=case["id"],
                                log_callback=self.main_window._case_log,
                                doctor_name=self.main_window.doctor_name)
        if dialog.exec():
            self.main_window._update_current_case_label()
            self._refresh()

    def _refresh(self) -> None:
        # Rebuild in place so freshly chosen case details are shown.
        while self.layout().count():
            item = self.layout().takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._build_ui()


class ChatWindow(QMainWindow):
    def __init__(self, config: AppConfig | None = None, doctor_name: str = ""):
        super().__init__()
        self.config = config or AppConfig.from_env()
        self.doctor_name = doctor_name
        self.config.ensure_directories()
        self.store = SQLiteStore(self.config.db_path)
        self.store.seed_demo_cases(self.doctor_name)
        self.rag = DocumentStore(self.config.index_path, self.config.chunk_size, self.config.chunk_overlap)
        self.rag.ingest_directory(self.config.data_dir / "knowledge")
        self.session_id = ""
        self.worker: AgentWorker | None = None
        # Case bound to the current session (one session ↔ one case).
        self.current_case: dict | None = None
        self.setWindowTitle("院内临床智能辅助系统")
        self.resize(1380, 860)
        self._build_ui()
        self._load_sessions()
        self._open_most_recent_or_new()
        self._update_current_case_label()
        self._show_case_stats()
        QTimer.singleShot(250, self._show_case_stats_dialog)
        QTimer.singleShot(700, self._auto_start_ollama)

    def _build_ui(self) -> None:
        self.setStatusBar(QStatusBar(self))
        # Permanent doctor badge on the right side of the status bar, never overwritten.
        self.doctor_badge = QLabel(f"👤 {self.doctor_name}")
        self.doctor_badge.setStyleSheet("color:#1f6a3c; font-weight:600; padding:2px 10px;")
        self.statusBar().addPermanentWidget(self.doctor_badge)
        self.statusBar().showMessage("本地离线模式 · 数据仅存储于当前电脑")
        self._build_menu()

        central = QWidget(self)
        root = QHBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 10)
        splitter = QSplitter(Qt.Orientation.Horizontal if QT6 else Qt.Horizontal)
        root.addWidget(splitter)

        history_panel = QWidget()
        history_layout = QVBoxLayout(history_panel)
        history_layout.setContentsMargins(0, 0, 8, 0)
        title_row = QHBoxLayout()
        title = QLabel("会话历史")
        title.setObjectName("sectionTitle")
        title_row.addWidget(title)
        title_row.addStretch()
        new_button = QPushButton("＋ 新建")
        new_button.clicked.connect(self._new_session)
        title_row.addWidget(new_button)
        history_layout.addLayout(title_row)
        self.sessions = QListWidget()
        self.sessions.currentItemChanged.connect(self._session_changed)
        self.sessions.itemDoubleClicked.connect(self._open_session_case_detail)
        history_layout.addWidget(self.sessions)
        delete_button = QPushButton("删除当前会话")
        delete_button.clicked.connect(self._delete_session)
        history_layout.addWidget(delete_button)
        splitter.addWidget(history_panel)

        chat_panel = QWidget()
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(8, 0, 8, 0)
        chat_heading = QHBoxLayout()
        heading = QLabel("临床咨询")
        heading.setObjectName("pageTitle")
        chat_heading.addWidget(heading)
        chat_heading.addStretch()
        self.clear_button = QPushButton("清空显示")
        self.clear_button.clicked.connect(self._clear_chat_view)
        chat_heading.addWidget(self.clear_button)
        chat_layout.addLayout(chat_heading)
        self.binding_label = QLabel("【未关联病例】")
        self.binding_label.setObjectName("bindingLabel")
        binding_row = QHBoxLayout()
        binding_row.addWidget(self.binding_label)
        binding_row.addStretch()
        self.rx_button = QPushButton("处方管理")
        self.rx_button.clicked.connect(self._open_prescription_manager)
        self.rx_button.setEnabled(False)
        binding_row.addWidget(self.rx_button)
        chat_layout.addLayout(binding_row)
        # Fixed stats panel above the chat area
        self.stats_label = QLabel()
        self.stats_label.setObjectName("statsPanel")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter if QT6 else Qt.AlignCenter)
        chat_layout.addWidget(self.stats_label)
        self.chat = QTextBrowser()
        self.chat.document().setDefaultStyleSheet(self._chat_document_style())
        self.chat.setOpenExternalLinks(False)
        chat_layout.addWidget(self.chat, 1)
        # Quick command bar: one-click common clinical queries for fast demo.
        quick_row = QHBoxLayout()
        quick_row.setSpacing(6)
        quick_row.addWidget(QLabel("快捷："))
        quick_commands = [
            "20kg儿童对乙酰氨基酚剂量",
            "华法林和阿司匹林能同服吗",
            "胸痛伴大汗风险分级",
            "腹痛挂什么科",
            "查询急诊科全部库存",
        ]
        for text in quick_commands:
            btn = QPushButton(text)
            btn.setObjectName("chipButton")
            btn.clicked.connect(lambda checked=False, t=text: self._quick_send(t))
            quick_row.addWidget(btn)
        quick_row.addStretch()
        chat_layout.addLayout(quick_row)
        input_row = QHBoxLayout()
        self.input = ChatInput()
        self.input.setPlaceholderText("输入临床问题，例如：对乙酰氨基酚 20 kg 每日剂量，或查询急诊科全部库存")
        self.input.setFixedHeight(76)
        self.input.setAcceptRichText(False)
        self.input.send_requested.connect(self._send)
        input_row.addWidget(self.input, 1)
        send_button = QPushButton("发送")
        send_button.setObjectName("primaryButton")
        send_button.setMinimumWidth(90)
        send_button.clicked.connect(self._send)
        input_row.addWidget(send_button)
        chat_layout.addLayout(input_row)
        splitter.addWidget(chat_panel)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([245, 900])
        self.setCentralWidget(central)

        self._build_config_dock()
        self._build_log_dock()
        self.setStyleSheet(self._style_sheet())

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("文件")
        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self.close)
        menu.addAction(quit_action)

        settings_menu = self.menuBar().addMenu("设置")
        self.settings_action = QAction("系统配置", self)
        self.settings_action.setCheckable(True)
        self.settings_action.setChecked(False)
        self.settings_action.triggered.connect(self._toggle_settings)
        settings_menu.addAction(self.settings_action)
        settings_menu.addSeparator()
        import_action = QAction("导入知识库文档", self)
        import_action.triggered.connect(self._import_documents)
        settings_menu.addAction(import_action)

        case_menu = self.menuBar().addMenu("病例")
        detail_action = QAction("病例详情", self)
        detail_action.triggered.connect(self._open_case_detail)
        case_menu.addAction(detail_action)
        new_case_action = QAction("新建病例", self)
        new_case_action.triggered.connect(self._new_case_direct)
        case_menu.addAction(new_case_action)
        find_case_action = QAction("查找病例", self)
        find_case_action.triggered.connect(self._open_case_manager)
        case_menu.addAction(find_case_action)
        export_case_action = QAction("导出病例记录", self)
        export_case_action.triggered.connect(self._export_cases)
        case_menu.addAction(export_case_action)
        case_menu.addSeparator()
        rx_action = QAction("处方管理", self)
        rx_action.triggered.connect(self._open_prescription_manager)
        case_menu.addAction(rx_action)
        clear_binding_action = QAction("清除病例绑定", self)
        clear_binding_action.triggered.connect(self._clear_case_binding)
        case_menu.addAction(clear_binding_action)

        help_menu = self.menuBar().addMenu("帮助")
        disclaimer = QAction("查看免责声明", self)
        disclaimer.triggered.connect(self._show_disclaimer)
        help_menu.addAction(disclaimer)

    def _build_config_dock(self) -> None:
        dock = CenteredDockWidget("系统配置", self)
        dock.setObjectName("configDock")
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea if QT6 else Qt.RightDockWidgetArea)
        dock.visibilityChanged.connect(self._sync_settings_action)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        form = QFormLayout()
        self.model_box = QComboBox()
        self.model_box.setEditable(True)
        self.model_box.addItem(self.config.model_name)
        self.model_box.addItem("deepseek-r1:8b")
        form.addRow("Ollama 模型", self.model_box)
        self.ollama_check = QCheckBox("启用真实模型回复（本机 Ollama）")
        self.ollama_check.setChecked(self.config.use_ollama)
        form.addRow("本地模型", self.ollama_check)
        self.temperature = QSlider(Qt.Orientation.Horizontal if QT6 else Qt.Horizontal)
        self.temperature.setRange(0, 100)
        self.temperature.setValue(int(self.config.temperature * 100))
        form.addRow("温度", self.temperature)
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 20)
        self.top_k.setValue(self.config.top_k)
        form.addRow("检索 Top-K", self.top_k)
        self.rrf = QSlider(Qt.Orientation.Horizontal if QT6 else Qt.Horizontal)
        self.rrf.setRange(0, 100)
        self.rrf.setValue(int(self.config.rrf_weight * 100))
        form.addRow("RRF 权重", self.rrf)
        layout.addLayout(form)
        apply_button = QPushButton("应用配置")
        apply_button.clicked.connect(self._apply_config)
        layout.addWidget(apply_button)
        import_button = QPushButton("导入 PDF / TXT / MD")
        import_button.clicked.connect(self._import_documents)
        layout.addWidget(import_button)
        hint = QLabel("默认由本机模型结合会话和工具结果生成回复；关闭后将明确标注为规则模式。所有数据均保存在本地。")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        layout.addWidget(hint)
        layout.addStretch()
        dock.setWidget(panel)
        self.config_dock = dock
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea if QT6 else Qt.RightDockWidgetArea, dock)
        dock.setFloating(True)
        dock.resize(360, 480)
        dock.hide()

    def _toggle_settings(self, visible: bool) -> None:
        self.config_dock.setVisible(visible)

    def _sync_settings_action(self, visible: bool) -> None:
        if self.settings_action.isChecked() != visible:
            self.settings_action.blockSignals(True)
            self.settings_action.setChecked(visible)
            self.settings_action.blockSignals(False)

    def _build_log_dock(self) -> None:
        dock = QDockWidget("模型思考与工具过程（固定）", self)
        dock.setObjectName("logDock")
        bottom_area = Qt.DockWidgetArea.BottomDockWidgetArea if QT6 else Qt.BottomDockWidgetArea
        no_features = (
            QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
            if QT6 else QDockWidget.NoDockWidgetFeatures
        )
        dock.setAllowedAreas(bottom_area)
        dock.setFeatures(no_features)
        dock.setFloating(False)
        dock.setMinimumHeight(170)
        dock.setMaximumHeight(230)
        dock.toggleViewAction().setEnabled(False)
        dock.toggleViewAction().setVisible(False)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setPlaceholderText("模型分析、工具调用、知识检索和安全检查将在这里持续显示。")
        dock.setWidget(self.log)
        self.log_dock = dock
        self.addDockWidget(bottom_area, dock)
        dock.show()
        self.resizeDocks([dock], [190], Qt.Orientation.Vertical if QT6 else Qt.Vertical)

    def _load_sessions(self) -> None:
        self.sessions.blockSignals(True)
        self.sessions.clear()
        role_key = Qt.ItemDataRole.UserRole if QT6 else Qt.UserRole
        for session in self.store.list_sessions():
            if session.get("bound_case_no"):
                label = f"{session.get('bound_patient_name', '')}（{session['bound_case_no']}）"
            else:
                label = session["title"]
            item = QListWidgetItem(label)
            item.setData(role_key, session["id"])
            self.sessions.addItem(item)
            if session["id"] == self.session_id:
                self.sessions.setCurrentItem(item)
        self.sessions.blockSignals(False)

    def _new_session(self) -> None:
        self.session_id = self.store.create_session()
        self._load_sessions()
        for index in range(self.sessions.count()):
            item = self.sessions.item(index)
            if item.data(Qt.ItemDataRole.UserRole if QT6 else Qt.UserRole) == self.session_id:
                self.sessions.setCurrentItem(item)
                break
        self.chat.clear()
        self._show_case_stats()
        self._append_system("我是院内临床智能辅助系统，我能为您做点什么？")
        self.current_case = None
        self._update_current_case_label()
        # Prompt to select a case for the new session immediately.
        dialog = CaseManagerDialog(self)
        dialog.exec()
        self._update_current_case_label()

    def _open_most_recent_or_new(self) -> None:
        """Reuse the latest existing session on startup; only create one when none exist."""
        if self.sessions.count():
            self.sessions.setCurrentRow(0)
        else:
            self._new_session()

    def _delete_session(self) -> None:
        if not self.session_id:
            return
        current_row = self.sessions.currentRow()
        self.store.delete_session(self.session_id)
        self._load_sessions()
        if self.sessions.count():
            next_row = current_row if current_row < self.sessions.count() else self.sessions.count() - 1
            self.sessions.setCurrentRow(max(0, next_row))
        else:
            self._new_session()

    def _session_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        if not current:
            return
        role_key = Qt.ItemDataRole.UserRole if QT6 else Qt.UserRole
        self.session_id = current.data(role_key)
        self.chat.clear()
        self._show_case_stats()
        for message in self.store.get_session_messages(self.session_id):
            if message["role"] == "user":
                self._append_user(message["content"])
            elif message["role"] == "assistant":
                self._append_assistant(message["content"], message.get("sources", []))
        self.current_case = self.store.get_session_case(self.session_id)
        self._update_current_case_label()

    def _auto_start_ollama(self) -> None:
        """Launch the local Ollama service in the background unless it is already running."""
        if not self.config.use_ollama:
            return
        if is_ollama_running(self.config.ollama_url):
            self._append_log(AgentEvent("ollama", "status", "Ollama 服务已在运行"))
            return
        launch_ollama_detached(lambda detail: self._append_log(AgentEvent("ollama", "status", detail)))

    def _new_case_direct(self) -> None:
        dialog = CaseEditDialog(self.store, log_callback=self._case_log, doctor_name=self.doctor_name)
        if dialog.exec() and dialog.saved_case:
            self.set_current_case(dialog.saved_case)

    def _open_case_manager(self) -> None:
        dialog = CaseManagerDialog(self)
        dialog.exec()
        self._update_current_case_label()

    def _open_prescription_manager(self) -> None:
        case = self.store.get_session_case(self.session_id) if self.session_id else None
        if not case:
            QMessageBox.information(self, "无法开药", "请先绑定病例后再进行处方管理。")
            return
        dialog = PrescriptionManagerDialog(self)
        dialog.exec()
        self._update_current_case_label()

    def _open_case_detail(self) -> None:
        dialog = CaseDetailDialog(self)
        dialog.exec()
        self._update_current_case_label()

    def _open_session_case_detail(self, item: QListWidgetItem) -> None:
        """Open the case attached to the exact history row that was double-clicked."""
        role_key = Qt.ItemDataRole.UserRole if QT6 else Qt.UserRole
        session_id = str(item.data(role_key) or "")
        case = self.store.get_session_case(session_id) if session_id else None
        if not case:
            QMessageBox.information(self, "未绑定病例", "该会话尚未绑定病例，请先选择病例。")
            return
        if session_id != self.session_id:
            self.sessions.setCurrentItem(item)
        dialog = CaseDetailDialog(self, case=case)
        dialog.exec()
        self._load_sessions()
        self._update_current_case_label()

    def set_current_case(self, case: dict) -> None:
        """Bind a case, keeping each case's conversation in its own session."""
        if not self.session_id:
            return

        current_case = self.store.get_session_case(self.session_id)
        messages = self.store.get_session_messages(self.session_id)
        same_case = bool(current_case and current_case.get("id") == case.get("id"))

        # A session that already has a case or chat history belongs to that
        # case.  Switch to a fresh session before binding a different case so
        # the previous conversation remains intact.
        if not same_case and (current_case or messages):
            self.session_id = self.store.create_session(
                f"{case['patient_name']}（{case['case_no']}）"
            )
            self.chat.clear()
            self._show_case_stats()
            self._append_system(f"已为病例 {case['patient_name']} 创建新会话。")

        self.store.bind_session_case(self.session_id, case["id"])
        self.store.add_audit(
            "case_bind",
            f"绑定病例 {case['case_no']}（{case['patient_name']}）到会话",
            self.session_id,
        )
        self._case_log(f"绑定病例：{case['case_no']}")
        self._load_sessions()
        self._update_current_case_label()

    def _clear_case_binding(self) -> None:
        case = self.current_case
        if not case:
            self.statusBar().showMessage("当前会话未关联病例", 3000)
            return
        self.store.clear_session_case(self.session_id)
        self.store.add_audit(
            "case_unbind",
            f"解除病例 {case['case_no']}（{case['patient_name']}）与会话的绑定",
            self.session_id,
        )
        self._case_log(f"清除病例绑定：{case['case_no']}")
        self.current_case = None
        self._load_sessions()
        self._update_current_case_label()
        self.statusBar().showMessage("已清除病例绑定", 3000)

    def _export_cases(self) -> None:
        cases = self.store.list_cases(doctor_name=self.doctor_name)
        if not cases:
            QMessageBox.information(self, "导出病例记录", "当前没有可导出的病例。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出病例记录", "病例记录.csv", "CSV 文件 (*.csv)")
        if not path:
            return
        headers = SQLiteStore.CASE_FIELDS + ["created_at", "updated_at"]
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for case in cases:
                writer.writerow([case.get(key, "") for key in headers])
        self.store.add_audit("case_export", f"导出 {len(cases)} 条病例记录到 {Path(path).name}")
        self._case_log(f"导出病例记录 {len(cases)} 条：{Path(path).name}")
        QMessageBox.information(self, "导出病例记录", f"已导出 {len(cases)} 条病例记录。")

    def _show_case_stats(self) -> None:
        """Update the fixed stats panel above the chat area."""
        stats = self.store.get_case_stats(doctor_name=self.doctor_name)
        today = datetime.now().strftime("%Y-%m-%d")
        self.stats_label.setText(
            f"今日病例概览（{today}）  ·  "
            f"累计病例 {stats['total']} 人  ·  "
            f"在院 {stats['in_hospital']} 人  ·  "
            f"待复查 {stats['pending_followup']} 人  ·  "
            f"今日复查 {stats['today_followup']} 人"
        )
        self.statusBar().showMessage(
            f"在院 {stats['in_hospital']} 人 · 累计病例 {stats['total']} · 待复查 {stats['pending_followup']} · 今日复查 {stats['today_followup']}"
        )

    def _update_current_case_label(self) -> None:
        case = self.store.get_session_case(self.session_id) if self.session_id else None
        self.current_case = case
        if case:
            status = case.get("patient_status", "")
            status_text = f" · {status}" if status else ""
            self.binding_label.setText(f"当前病例：{case['patient_name']}（{case['case_no']}）{status_text}")
            self.binding_label.setProperty("bound", True)
        else:
            self.binding_label.setText("【未选择病例】")
            self.binding_label.setProperty("bound", False)
        self.binding_label.style().unpolish(self.binding_label)
        self.binding_label.style().polish(self.binding_label)
        self.rx_button.setEnabled(case is not None)

    def _case_log(self, detail: str) -> None:
        self._append_log(AgentEvent("case", "status", detail))

    def _quick_send(self, text: str) -> None:
        """Fill the input box with a preset query and submit immediately."""
        if self.worker and self.worker.isRunning():
            return
        self.input.setPlainText(text)
        self._send()

    def _send(self) -> None:
        query = self.input.toPlainText().strip()
        if not query or self.worker and self.worker.isRunning():
            return
        self.input.clear()
        self._append_user(query)
        self.store.add_message(self.session_id, "user", query)
        self._load_sessions()
        self._append_log(AgentEvent("ui", "request", f"用户请求：{query}"))
        self.input.setEnabled(False)
        history = self.store.get_session_messages(self.session_id)
        patient = dict(self.current_case) if self.current_case else None
        self.worker = AgentWorker(self.config, self.store, self.rag, query, self.session_id, history, patient)
        self.worker.event.connect(self._append_log)
        self.worker.result.connect(self._handle_result)
        self.worker.failed.connect(self._handle_error)
        self.worker.finished.connect(lambda: self.input.setEnabled(True))
        self.worker.start()

    def _handle_result(self, result: AgentResult) -> None:
        self._append_assistant(result.answer, [source.as_dict() for source in result.sources])
        self.store.add_message(self.session_id, "assistant", result.answer, [source.as_dict() for source in result.sources])
        self._load_sessions()

    def _handle_error(self, message: str) -> None:
        self._append_assistant(f"工作流执行失败：{message}", [])

    def _append_user(self, text: str) -> None:
        self.chat.append(
            f'<table align="right" class="chatTable"><tr><td class="userBubble">{self._escape(text)}</td></tr></table>'
        )

    def _append_assistant(self, text: str, sources: list[dict]) -> None:
        rendered = self._escape(text).replace("\n", "<br>")
        source_html = ""
        if sources:
            source_html = "<div class='sourceTitle'>引用来源</div>" + "".join(f"<div class='source'>[{index}] {self._escape(item.get('title', '本地文档'))} · 相关度 {float(item.get('score', 0)):.3f}<br>{self._escape(item.get('text', '')[:180])}</div>" for index, item in enumerate(sources, 1))
        self.chat.append(
            f'<table align="left" class="chatTable"><tr><td class="assistantBubble">{rendered}{source_html}</td></tr></table>'
        )

    def _append_system(self, text: str) -> None:
        self.chat.append(f'<div class="systemMessage">{self._escape(text)}</div>')

    def _append_log(self, event: AgentEvent) -> None:
        self.log.appendPlainText(f"[{event.timestamp}] [{event.node}] {event.event_type} · {event.detail}")
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _clear_chat_view(self) -> None:
        self.chat.clear()

    def _apply_config(self) -> None:
        self.config.model_name = self.model_box.currentText().strip() or self.config.model_name
        self.config.use_ollama = self.ollama_check.isChecked()
        self.config.temperature = self.temperature.value() / 100
        self.config.top_k = self.top_k.value()
        self.config.rrf_weight = self.rrf.value() / 100
        if self.config.use_ollama:
            if is_ollama_running(self.config.ollama_url):
                message = f"配置已应用 · Ollama 已连接（{self.config.model_name}）"
            else:
                launched = launch_ollama_detached(lambda detail: self._append_log(AgentEvent("ollama", "status", detail)))
                message = "配置已应用 · 正在后台启动 Ollama，稍后即可对话" if launched else "配置已应用 · 未找到 Ollama，可执行文件，请检查安装"
        else:
            message = "配置已应用 · 当前为规则模式，回复将不经过模型生成"
        self.statusBar().showMessage(message, 5000)

    def _import_documents(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "导入医疗知识文档", str(self.config.data_dir), "医疗文档 (*.pdf *.txt *.md)")
        if not paths:
            return
        total = 0
        for path in paths:
            try:
                total += self.rag.ingest_file(path)
            except OSError as exc:
                self._append_log(AgentEvent("knowledge_import", "error", f"{Path(path).name}: {exc}"))
        self._append_log(AgentEvent("knowledge_import", "completed", f"完成导入 {len(paths)} 个文件，生成 {total} 个片段"))
        QMessageBox.information(self, "知识库导入", f"已导入 {len(paths)} 个文件，生成 {total} 个可检索片段。")

    def _show_case_stats_dialog(self) -> None:
        """Startup popup: case overview dashboard (replaces the disclaimer popup)."""
        stats = self.store.get_case_stats(doctor_name=self.doctor_name)
        today = datetime.now().strftime("%Y-%m-%d")
        dialog = QDialog(self)
        dialog.setWindowTitle("今日病例概况")
        dialog.setFixedWidth(460)
        root = QVBoxLayout(dialog)
        root.setContentsMargins(20, 18, 20, 16)

        title = QLabel(f"今日病例概况（{today}）")
        title.setStyleSheet("font-size:17px; font-weight:700; color:#17324d;")
        root.addWidget(title)
        root.addSpacing(12)

        cards = [
            ("累计病例", stats["total"], "#1769aa"),
            ("在院病人", stats["in_hospital"], "#1f6a3c"),
            ("剩余待复查", stats["pending_followup"], "#b06a00"),
            ("今天预计复查", stats["today_followup"], "#7a3fb0"),
        ]
        card_row = QHBoxLayout()
        card_row.setSpacing(8)
        for label, value, color in cards:
            card = QFrame()
            card.setStyleSheet(
                f"QFrame{{background:#ffffff; border:1px solid #dce3ea; border-radius:8px;}}"
                f"QLabel{{border:none; background:transparent;}}"
            )
            vbox = QVBoxLayout(card)
            vbox.setContentsMargins(6, 10, 6, 10)
            num = QLabel(str(value))
            num.setAlignment(Qt.AlignmentFlag.AlignCenter)
            num.setStyleSheet(f"font-size:26px; font-weight:700; color:{color};")
            name = QLabel(label)
            name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name.setStyleSheet("font-size:12px; color:#5a6b7a;")
            vbox.addWidget(num)
            vbox.addWidget(name)
            card_row.addWidget(card)
        root.addLayout(card_row)
        root.addSpacing(12)

        note = QLabel("本工具仅提供临床辅助，不替代医生诊断、处方或现场急救。")
        note.setStyleSheet("color:#8a96a2; font-size:11px;")
        root.addWidget(note)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        enter_btn = QPushButton("进入系统")
        enter_btn.setObjectName("primaryButton")
        enter_btn.setMinimumWidth(100)
        enter_btn.clicked.connect(dialog.accept)
        btn_row.addWidget(enter_btn)
        root.addLayout(btn_row)
        dialog.exec()

    def _show_disclaimer(self) -> None:
        QMessageBox.warning(self, "医疗免责声明", "本工具仅提供院内临床辅助与科普信息，不替代医生诊断、处方或现场急救。\n\n遇到意识丧失、呼吸停止、大出血等情况，请立即启动急救流程。")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(1500)
        self.store.close()
        event.accept()

    @staticmethod
    def _escape(text: str) -> str:
        return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))

    @staticmethod
    def _style_sheet() -> str:
        return """
        QMainWindow, QWidget { background: #f5f7fa; color: #17212b; font-family: 'Microsoft YaHei', 'Segoe UI'; font-size: 13px; }
        QListWidget, QTextBrowser, QTextEdit, QPlainTextEdit { background: #ffffff; border: 1px solid #dce3ea; border-radius: 6px; }
        QListWidget { padding: 4px; }
        QListWidget::item { padding: 11px 8px; border-radius: 4px; }
        QListWidget::item:selected { background: #dceeff; color: #0c4778; }
        QPushButton { background: #ffffff; border: 1px solid #c7d2dc; border-radius: 5px; padding: 7px 12px; }
        QPushButton:hover { background: #edf5fb; border-color: #5d9bc6; }
        QPushButton#primaryButton { background: #1769aa; color: white; border-color: #1769aa; font-weight: 600; }
        QPushButton#primaryButton:hover { background: #0f568f; }
        QPushButton#chipButton { background: #eef3f8; color: #17324d; border: 1px solid #d7e0e8; border-radius: 12px; padding: 3px 12px; font-size: 12px; }
        QPushButton#chipButton:hover { background: #dce8f3; border-color: #9fb9d2; }
        QLabel#statsPanel { background: #f0f5fa; border: 1px solid #d7e0e8; border-radius: 6px; padding: 6px; color: #17324d; font-size: 12px; }
        QLabel#pageTitle { font-size: 20px; font-weight: 700; color: #17324d; padding-bottom: 4px; }
        QLabel#bindingLabel { background: #eef3f8; border: 1px solid #d7e0e8; border-radius: 6px; padding: 7px 10px; color: #5a6b7a; }
        QLabel#bindingLabel[bound="true"] { background: #e8f4ec; border-color: #bfe0c9; color: #1f6a3c; font-weight: 600; }
        QTableWidget { background: #ffffff; border: 1px solid #dce3ea; border-radius: 6px; gridline-color: #eef1f4; selection-background-color: #dceeff; selection-color: #0c4778; }
        QHeaderView::section { background: #eaf0f5; padding: 6px; border: none; border-right: 1px solid #dce3ea; font-weight: 600; }
        QComboBox { background: #ffffff; border: 1px solid #dce3ea; border-radius: 5px; padding: 5px 8px; }
        QLabel#sectionTitle { font-size: 15px; font-weight: 700; color: #17324d; }
        QLabel#muted { color: #667785; line-height: 1.4; }
        QDockWidget { font-weight: 700; }
        QDockWidget::title { background: #eaf0f5; padding: 7px; }
        QSlider::groove:horizontal { height: 4px; background: #d5dde5; }
        QSlider::handle:horizontal { width: 14px; margin: -5px 0; border-radius: 7px; background: #1769aa; }
        """

    @staticmethod
    def _chat_document_style() -> str:
        """CSS consumed by the QTextBrowser document (QSS does not style rich-text elements)."""
        return """
        table.chatTable { margin: 7px 0; }
        td { font-size: 13px; font-family: 'Microsoft YaHei', 'Segoe UI'; padding: 10px 13px; border-radius: 8px; }
        td.userBubble { background: #1769aa; color: #ffffff; }
        td.assistantBubble { background: #f0f3f6; color: #17212b; }
        .sourceTitle { color: #526875; font-weight: 700; margin-top: 10px; }
        .source { color: #5b6f7b; background: #ffffff; padding: 7px; margin-top: 5px; font-size: 11px; border-radius: 5px; }
        .systemMessage { color: #71808a; padding: 8px; text-align: center; font-size: 12px; }
        """


def run(config: AppConfig | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)

    login = LoginDialog()
    accepted_code = QDialog.DialogCode.Accepted if QT6 else QDialog.Accepted
    if login.exec() != accepted_code:
        return 0

    window = ChatWindow(config, doctor_name=login.doctor_name)
    window.show()
    window.statusBar().showMessage(f"登录医生：{login.doctor_name} · 本地离线模式 · 数据仅存储于当前电脑")
    return app.exec() if QT6 else app.exec_()
