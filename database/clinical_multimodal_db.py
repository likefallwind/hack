import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class SearchResult:
    patient_id: str
    visit_id: str
    note_text: str
    score: float


class ClinicalMultimodalDB:
    """Notebook-oriented multimodal clinical database on top of SQLite.

    Features included in this lightweight implementation:
    - Standardized schema for patient/visit/medication/lesion/lab/assets/audit.
    - Intelligent auto-fill extraction from free text (regex-based baseline).
    - Natural language semantic retrieval through SQLite FTS5.
    - Simple RAG-style answer synthesis with evidence references.
    - Longitudinal change detection for lesion and laboratory measurements.
    """

    def __init__(self, db_path: str = "database/clinical_multimodal.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        schema_sql = """
        CREATE TABLE IF NOT EXISTS patient_master (
            patient_id TEXT PRIMARY KEY,
            anonymized_id TEXT NOT NULL,
            sex TEXT,
            birth_year INTEGER,
            center TEXT,
            enrollment_time TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS visit_event (
            visit_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            visit_time TEXT NOT NULL,
            diagnosis TEXT,
            stage TEXT,
            regimen_version TEXT,
            note_text TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patient_master(patient_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS medication_timeline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            visit_id TEXT NOT NULL,
            drug_name TEXT NOT NULL,
            dose_value REAL,
            dose_unit TEXT,
            frequency TEXT,
            change_type TEXT,
            change_reason TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (visit_id) REFERENCES visit_event(visit_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS lesion_measurement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            visit_id TEXT NOT NULL,
            lesion_site TEXT,
            area_cm2 REAL,
            perimeter_cm REAL,
            pigmentation_score REAL,
            model_version TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (visit_id) REFERENCES visit_event(visit_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS lab_result (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            visit_id TEXT NOT NULL,
            item_name TEXT NOT NULL,
            item_value REAL,
            unit TEXT,
            ref_range TEXT,
            abnormal_flag TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (visit_id) REFERENCES visit_event(visit_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS multimodal_asset (
            asset_id TEXT PRIMARY KEY,
            visit_id TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            object_uri TEXT NOT NULL,
            device TEXT,
            iqa_score REAL,
            frame_summary TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (visit_id) REFERENCES visit_event(visit_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT NOT NULL,
            row_ref TEXT NOT NULL,
            field_name TEXT NOT NULL,
            old_value TEXT,
            new_value TEXT,
            source TEXT NOT NULL,
            operator TEXT,
            reason TEXT,
            created_at TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS visit_note_fts USING fts5(
            visit_id UNINDEXED,
            patient_id UNINDEXED,
            note_text,
            tokenize = 'unicode61'
        );
        """
        self.conn.executescript(schema_sql)
        self.conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def close(self) -> None:
        self.conn.close()

    def record_audit(
        self,
        table_name: str,
        row_ref: str,
        field_name: str,
        old_value: Optional[str],
        new_value: Optional[str],
        source: str,
        operator: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO audit_log(table_name, row_ref, field_name, old_value, new_value, source, operator, reason, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (table_name, row_ref, field_name, old_value, new_value, source, operator, reason, self._now()),
        )
        self.conn.commit()

    def upsert_patient(self, patient: Dict[str, Any]) -> None:
        payload = {
            "patient_id": patient["patient_id"],
            "anonymized_id": patient.get("anonymized_id", patient["patient_id"]),
            "sex": patient.get("sex"),
            "birth_year": patient.get("birth_year"),
            "center": patient.get("center"),
            "enrollment_time": patient.get("enrollment_time"),
            "created_at": self._now(),
        }
        self.conn.execute(
            """
            INSERT INTO patient_master(patient_id, anonymized_id, sex, birth_year, center, enrollment_time, created_at)
            VALUES(:patient_id, :anonymized_id, :sex, :birth_year, :center, :enrollment_time, :created_at)
            ON CONFLICT(patient_id) DO UPDATE SET
                anonymized_id=excluded.anonymized_id,
                sex=excluded.sex,
                birth_year=excluded.birth_year,
                center=excluded.center,
                enrollment_time=excluded.enrollment_time
            """,
            payload,
        )
        self.conn.commit()

    def add_visit(self, visit: Dict[str, Any]) -> None:
        payload = {
            "visit_id": visit["visit_id"],
            "patient_id": visit["patient_id"],
            "visit_time": visit["visit_time"],
            "diagnosis": visit.get("diagnosis"),
            "stage": visit.get("stage"),
            "regimen_version": visit.get("regimen_version"),
            "note_text": visit.get("note_text", ""),
            "created_at": self._now(),
        }
        self.conn.execute(
            """
            INSERT INTO visit_event(visit_id, patient_id, visit_time, diagnosis, stage, regimen_version, note_text, created_at)
            VALUES(:visit_id, :patient_id, :visit_time, :diagnosis, :stage, :regimen_version, :note_text, :created_at)
            ON CONFLICT(visit_id) DO UPDATE SET
                patient_id=excluded.patient_id,
                visit_time=excluded.visit_time,
                diagnosis=excluded.diagnosis,
                stage=excluded.stage,
                regimen_version=excluded.regimen_version,
                note_text=excluded.note_text
            """,
            payload,
        )
        self.conn.execute("DELETE FROM visit_note_fts WHERE visit_id = ?", (payload["visit_id"],))
        self.conn.execute(
            "INSERT INTO visit_note_fts(visit_id, patient_id, note_text) VALUES (?, ?, ?)",
            (payload["visit_id"], payload["patient_id"], payload["note_text"]),
        )
        self.conn.commit()

    def add_medication(self, visit_id: str, medication: Dict[str, Any]) -> None:
        payload = {
            "visit_id": visit_id,
            "drug_name": medication.get("drug_name"),
            "dose_value": medication.get("dose_value"),
            "dose_unit": medication.get("dose_unit"),
            "frequency": medication.get("frequency"),
            "change_type": medication.get("change_type"),
            "change_reason": medication.get("change_reason"),
            "created_at": self._now(),
        }
        self.conn.execute(
            """
            INSERT INTO medication_timeline(visit_id, drug_name, dose_value, dose_unit, frequency, change_type, change_reason, created_at)
            VALUES(:visit_id, :drug_name, :dose_value, :dose_unit, :frequency, :change_type, :change_reason, :created_at)
            """,
            payload,
        )
        self.conn.commit()
        self.record_audit(
            table_name="medication_timeline",
            row_ref=visit_id,
            field_name="drug_name",
            old_value=None,
            new_value=str(payload.get("drug_name")),
            source="manual_or_api",
        )

    def add_lesion_measurement(self, visit_id: str, lesion: Dict[str, Any]) -> None:
        payload = {
            "visit_id": visit_id,
            "lesion_site": lesion.get("lesion_site"),
            "area_cm2": lesion.get("area_cm2"),
            "perimeter_cm": lesion.get("perimeter_cm"),
            "pigmentation_score": lesion.get("pigmentation_score"),
            "model_version": lesion.get("model_version"),
            "created_at": self._now(),
        }
        self.conn.execute(
            """
            INSERT INTO lesion_measurement(visit_id, lesion_site, area_cm2, perimeter_cm, pigmentation_score, model_version, created_at)
            VALUES(:visit_id, :lesion_site, :area_cm2, :perimeter_cm, :pigmentation_score, :model_version, :created_at)
            """,
            payload,
        )
        self.conn.commit()
        self.record_audit(
            table_name="lesion_measurement",
            row_ref=visit_id,
            field_name="area_cm2",
            old_value=None,
            new_value=None if payload.get("area_cm2") is None else str(payload.get("area_cm2")),
            source="manual_or_api",
        )

    def add_lab_result(self, visit_id: str, lab: Dict[str, Any]) -> None:
        payload = {
            "visit_id": visit_id,
            "item_name": lab.get("item_name"),
            "item_value": lab.get("item_value"),
            "unit": lab.get("unit"),
            "ref_range": lab.get("ref_range"),
            "abnormal_flag": lab.get("abnormal_flag"),
            "created_at": self._now(),
        }
        self.conn.execute(
            """
            INSERT INTO lab_result(visit_id, item_name, item_value, unit, ref_range, abnormal_flag, created_at)
            VALUES(:visit_id, :item_name, :item_value, :unit, :ref_range, :abnormal_flag, :created_at)
            """,
            payload,
        )
        self.conn.commit()
        self.record_audit(
            table_name="lab_result",
            row_ref=visit_id,
            field_name=str(payload.get("item_name")),
            old_value=None,
            new_value=None if payload.get("item_value") is None else str(payload.get("item_value")),
            source="manual_or_api",
        )

    def add_asset(self, asset: Dict[str, Any]) -> None:
        payload = {
            "asset_id": asset["asset_id"],
            "visit_id": asset["visit_id"],
            "asset_type": asset["asset_type"],
            "object_uri": asset["object_uri"],
            "device": asset.get("device"),
            "iqa_score": asset.get("iqa_score"),
            "frame_summary": asset.get("frame_summary"),
            "created_at": self._now(),
        }
        self.conn.execute(
            """
            INSERT INTO multimodal_asset(asset_id, visit_id, asset_type, object_uri, device, iqa_score, frame_summary, created_at)
            VALUES(:asset_id, :visit_id, :asset_type, :object_uri, :device, :iqa_score, :frame_summary, :created_at)
            """,
            payload,
        )
        self.conn.commit()

    def auto_fill_from_text(self, text: str) -> Dict[str, Any]:
        """Baseline intelligent auto-fill from raw medical note text."""
        meds = []
        for match in re.finditer(r"([A-Za-z\u4e00-\u9fa5]+)\s*(\d+(?:\.\d+)?)\s*(mg|g|ml)?\s*(qd|bid|tid|qod|每[日天周]\\d*次)?", text):
            drug, dose, unit, frequency = match.groups()
            if len(drug) < 2:
                continue
            if unit is None and frequency is None:
                continue
            if drug.lower() in {"cm", "mm"}:
                continue
            meds.append(
                {
                    "drug_name": drug,
                    "dose_value": float(dose),
                    "dose_unit": unit or "mg",
                    "frequency": frequency,
                    "change_type": "unknown",
                }
            )

        area_match = re.search(r"病灶面积\s*[:：]?\s*(\d+(?:\.\d+)?)\s*cm2", text)
        pigmentation_match = re.search(r"色素(?:评分|指数)?\s*[:：]?\s*(\d+(?:\.\d+)?)", text)

        labs = []
        for item, value, unit in re.findall(r"([A-Za-z\u4e00-\u9fa5]+)\s*[:：]\s*(\d+(?:\.\d+)?)\s*([A-Za-z/%μ]+)", text):
            if item in {"病灶面积", "色素评分", "色素指数"}:
                continue
            labs.append({"item_name": item, "item_value": float(value), "unit": unit})

        return {
            "medications": meds,
            "lesion": {
                "area_cm2": float(area_match.group(1)) if area_match else None,
                "pigmentation_score": float(pigmentation_match.group(1)) if pigmentation_match else None,
            },
            "labs": labs,
            "raw_text": text,
        }

    def batch_auto_fill_from_texts(self, texts: Sequence[str]) -> List[Dict[str, Any]]:
        return [self.auto_fill_from_text(text) for text in texts]

    def commit_auto_fill(
        self,
        visit_id: str,
        extracted: Dict[str, Any],
        operator: str = "system",
        manual_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Persist extracted payload with optional manual corrections."""
        manual_overrides = manual_overrides or {}
        medications = manual_overrides.get("medications", extracted.get("medications", []))
        lesion = manual_overrides.get("lesion", extracted.get("lesion", {}))
        labs = manual_overrides.get("labs", extracted.get("labs", []))

        med_count = 0
        for medication in medications:
            self.add_medication(visit_id, medication)
            med_count += 1
        lesion_count = 0
        if lesion and any(v is not None for v in lesion.values()):
            self.add_lesion_measurement(visit_id, lesion)
            lesion_count = 1
        lab_count = 0
        for lab in labs:
            self.add_lab_result(visit_id, lab)
            lab_count += 1

        self.record_audit(
            table_name="visit_event",
            row_ref=visit_id,
            field_name="auto_fill_commit",
            old_value=None,
            new_value=f"medications={med_count},lesion={lesion_count},labs={lab_count}",
            source="auto_fill",
            operator=operator,
            reason="auto_fill_commit_with_optional_manual_overrides",
        )
        return {"medications": med_count, "lesions": lesion_count, "labs": lab_count}

    @staticmethod
    def score_iqa(blur_score: float, exposure_score: float, lesion_ratio: float) -> Dict[str, Any]:
        """Simple IQA scorer for image/video frames.

        All inputs should be normalized to [0, 1], where larger is better.
        """
        blur = max(0.0, min(1.0, blur_score))
        exposure = max(0.0, min(1.0, exposure_score))
        ratio = max(0.0, min(1.0, lesion_ratio))
        total = round(0.45 * blur + 0.35 * exposure + 0.20 * ratio, 4)
        qualified = total >= 0.65 and ratio >= 0.1
        reasons = []
        if blur < 0.5:
            reasons.append("图像清晰度不足")
        if exposure < 0.5:
            reasons.append("曝光质量不足")
        if ratio < 0.1:
            reasons.append("病灶占比过低")
        return {
            "iqa_score": total,
            "qualified": qualified,
            "reasons": reasons,
        }

    def semantic_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        # FTS5 query syntax; quote raw query for better compatibility with Chinese mixed text.
        rows = self.conn.execute(
            """
            SELECT visit_id, patient_id, note_text, bm25(visit_note_fts) AS score
            FROM visit_note_fts
            WHERE visit_note_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query, top_k),
        ).fetchall()
        return [
            SearchResult(
                patient_id=row["patient_id"],
                visit_id=row["visit_id"],
                note_text=row["note_text"],
                score=float(row["score"]),
            )
            for row in rows
        ]

    def rag_answer(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        hits = self.semantic_search(query, top_k=top_k)
        if not hits:
            return {"answer": "未检索到匹配病例，请放宽筛选条件。", "evidence": []}

        evidence = [
            {
                "patient_id": hit.patient_id,
                "visit_id": hit.visit_id,
                "score": hit.score,
                "snippet": hit.note_text[:180],
            }
            for hit in hits
        ]
        answer = f"共检索到 {len(hits)} 条高相关记录，已按相关性排序返回证据。"
        return {"answer": answer, "evidence": evidence}

    def detect_changes(self, patient_id: str, lab_items: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        visits = self.conn.execute(
            """
            SELECT visit_id, visit_time
            FROM visit_event
            WHERE patient_id = ?
            ORDER BY visit_time DESC
            LIMIT 2
            """,
            (patient_id,),
        ).fetchall()
        if len(visits) < 2:
            return {"patient_id": patient_id, "message": "对比至少需要两次就诊记录。"}

        latest_visit, previous_visit = visits[0]["visit_id"], visits[1]["visit_id"]
        lesion_delta = self._lesion_delta(latest_visit, previous_visit)
        lab_delta = self._lab_delta(latest_visit, previous_visit, lab_items)
        return {
            "patient_id": patient_id,
            "latest_visit_id": latest_visit,
            "previous_visit_id": previous_visit,
            "lesion_delta": lesion_delta,
            "lab_delta": lab_delta,
        }

    def _lesion_delta(self, latest_visit_id: str, previous_visit_id: str) -> Dict[str, Any]:
        get_area = "SELECT area_cm2 FROM lesion_measurement WHERE visit_id = ? ORDER BY id DESC LIMIT 1"
        latest = self.conn.execute(get_area, (latest_visit_id,)).fetchone()
        previous = self.conn.execute(get_area, (previous_visit_id,)).fetchone()
        if not latest or not previous or latest["area_cm2"] is None or previous["area_cm2"] in (None, 0):
            return {"message": "病灶面积数据不足，无法计算变化率。"}

        delta = latest["area_cm2"] - previous["area_cm2"]
        ratio = delta / previous["area_cm2"] * 100
        return {
            "latest_area_cm2": latest["area_cm2"],
            "previous_area_cm2": previous["area_cm2"],
            "delta_area_cm2": round(delta, 4),
            "delta_ratio_pct": round(ratio, 2),
        }

    def _lab_delta(self, latest_visit_id: str, previous_visit_id: str, lab_items: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
        clause = ""
        params: List[Any] = [latest_visit_id]
        if lab_items:
            placeholders = ",".join(["?" for _ in lab_items])
            clause = f"AND item_name IN ({placeholders})"
            params.extend(lab_items)

        latest_rows = self.conn.execute(
            f"SELECT item_name, item_value, unit FROM lab_result WHERE visit_id = ? {clause}",
            params,
        ).fetchall()

        previous_map = {
            row["item_name"]: row
            for row in self.conn.execute(
                "SELECT item_name, item_value, unit FROM lab_result WHERE visit_id = ?",
                (previous_visit_id,),
            ).fetchall()
        }

        deltas = []
        for row in latest_rows:
            old = previous_map.get(row["item_name"])
            if not old or old["item_value"] is None or row["item_value"] is None:
                continue
            old_value = float(old["item_value"])
            new_value = float(row["item_value"])
            delta = new_value - old_value
            ratio = None if old_value == 0 else (delta / old_value * 100)
            deltas.append(
                {
                    "item_name": row["item_name"],
                    "old_value": old_value,
                    "new_value": new_value,
                    "unit": row["unit"] or old["unit"],
                    "delta": round(delta, 4),
                    "delta_ratio_pct": None if ratio is None else round(ratio, 2),
                }
            )
        return deltas

    def export_patient_snapshot(self, patient_id: str) -> Dict[str, Any]:
        patient = self.conn.execute(
            "SELECT * FROM patient_master WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()
        if not patient:
            return {}

        visits = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM visit_event WHERE patient_id = ? ORDER BY visit_time",
                (patient_id,),
            ).fetchall()
        ]
        return {"patient": dict(patient), "visits": visits}

    def extract_knowledge_graph(self, text: str) -> Dict[str, Any]:
        """Extract a lightweight diagnosis logic graph from plain text.

        Pattern focus: symptom -> drug -> indicator.
        """
        symptom_candidates = re.findall(r"(瘙痒|红斑|脱屑|疼痛|灼热|丘疹|水疱)", text)
        drug_candidates = [item["drug_name"] for item in self.auto_fill_from_text(text)["medications"]]
        indicator_candidates = [item["item_name"] for item in self.auto_fill_from_text(text)["labs"]]

        nodes = []
        edges = []
        seen = set()

        def _add_node(node_id: str, label: str, node_type: str) -> None:
            key = (node_id, node_type)
            if key in seen:
                return
            seen.add(key)
            nodes.append({"id": node_id, "label": label, "type": node_type})

        for symptom in symptom_candidates:
            _add_node(f"symptom:{symptom}", symptom, "symptom")
        for drug in drug_candidates:
            _add_node(f"drug:{drug}", drug, "drug")
        for indicator in indicator_candidates:
            _add_node(f"indicator:{indicator}", indicator, "indicator")

        for symptom in symptom_candidates:
            for drug in drug_candidates:
                edges.append(
                    {
                        "source": f"symptom:{symptom}",
                        "target": f"drug:{drug}",
                        "relation": "treated_by",
                    }
                )
        for drug in drug_candidates:
            for indicator in indicator_candidates:
                edges.append(
                    {
                        "source": f"drug:{drug}",
                        "target": f"indicator:{indicator}",
                        "relation": "affects",
                    }
                )
        return {"nodes": nodes, "edges": edges}

    def build_patient_knowledge_graph(self, patient_id: str) -> Dict[str, Any]:
        rows = self.conn.execute(
            "SELECT note_text FROM visit_event WHERE patient_id = ? ORDER BY visit_time",
            (patient_id,),
        ).fetchall()
        merged_nodes: Dict[str, Dict[str, str]] = {}
        merged_edges: Dict[str, Dict[str, str]] = {}
        for row in rows:
            graph = self.extract_knowledge_graph(row["note_text"] or "")
            for node in graph["nodes"]:
                merged_nodes[node["id"]] = node
            for edge in graph["edges"]:
                edge_id = f'{edge["source"]}->{edge["relation"]}->{edge["target"]}'
                merged_edges[edge_id] = edge
        return {
            "patient_id": patient_id,
            "nodes": list(merged_nodes.values()),
            "edges": list(merged_edges.values()),
        }

    def lesion_evolution(self, patient_id: str) -> Dict[str, Any]:
        rows = self.conn.execute(
            """
            SELECT v.visit_time, l.area_cm2, l.pigmentation_score
            FROM visit_event v
            LEFT JOIN lesion_measurement l ON v.visit_id = l.visit_id
            WHERE v.patient_id = ?
            ORDER BY v.visit_time
            """,
            (patient_id,),
        ).fetchall()
        timeline = []
        prev_area = None
        for row in rows:
            area = row["area_cm2"]
            delta_ratio = None
            if prev_area not in (None, 0) and area is not None:
                delta_ratio = round((area - prev_area) / prev_area * 100, 2)
            timeline.append(
                {
                    "visit_time": row["visit_time"],
                    "area_cm2": area,
                    "pigmentation_score": row["pigmentation_score"],
                    "delta_ratio_pct": delta_ratio,
                }
            )
            if area is not None:
                prev_area = area
        return {"patient_id": patient_id, "timeline": timeline}

    def get_audit_logs(self, row_ref: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        if row_ref:
            rows = self.conn.execute(
                "SELECT * FROM audit_log WHERE row_ref = ? ORDER BY id DESC LIMIT ?",
                (row_ref, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def research_query_dose_up_lesion_down(
        self,
        year: int,
        lesion_reduction_pct: float = 20.0,
    ) -> List[Dict[str, Any]]:
        """Find patients whose dose increases while lesion area decreases over a year."""
        patient_rows = self.conn.execute(
            """
            SELECT DISTINCT patient_id
            FROM visit_event
            WHERE substr(visit_time, 1, 4) = ?
            """,
            (str(year),),
        ).fetchall()
        results: List[Dict[str, Any]] = []
        for row in patient_rows:
            patient_id = row["patient_id"]
            visits = self.conn.execute(
                """
                SELECT visit_id, visit_time
                FROM visit_event
                WHERE patient_id = ? AND substr(visit_time, 1, 4) = ?
                ORDER BY visit_time
                """,
                (patient_id, str(year)),
            ).fetchall()
            if len(visits) < 2:
                continue
            first_visit, last_visit = visits[0]["visit_id"], visits[-1]["visit_id"]

            first_area_row = self.conn.execute(
                "SELECT area_cm2 FROM lesion_measurement WHERE visit_id = ? ORDER BY id DESC LIMIT 1",
                (first_visit,),
            ).fetchone()
            last_area_row = self.conn.execute(
                "SELECT area_cm2 FROM lesion_measurement WHERE visit_id = ? ORDER BY id DESC LIMIT 1",
                (last_visit,),
            ).fetchone()
            if not first_area_row or not last_area_row:
                continue
            first_area = first_area_row["area_cm2"]
            last_area = last_area_row["area_cm2"]
            if first_area in (None, 0) or last_area is None:
                continue
            lesion_change_pct = (last_area - first_area) / first_area * 100.0

            first_dose = self.conn.execute(
                "SELECT COALESCE(SUM(dose_value), 0) AS total_dose FROM medication_timeline WHERE visit_id = ?",
                (first_visit,),
            ).fetchone()["total_dose"]
            last_dose = self.conn.execute(
                "SELECT COALESCE(SUM(dose_value), 0) AS total_dose FROM medication_timeline WHERE visit_id = ?",
                (last_visit,),
            ).fetchone()["total_dose"]
            dose_change_pct = 0.0 if first_dose in (None, 0) else (last_dose - first_dose) / first_dose * 100.0

            if dose_change_pct > 0 and lesion_change_pct <= -abs(lesion_reduction_pct):
                results.append(
                    {
                        "patient_id": patient_id,
                        "first_visit_id": first_visit,
                        "last_visit_id": last_visit,
                        "first_total_dose": round(float(first_dose), 4),
                        "last_total_dose": round(float(last_dose), 4),
                        "dose_change_pct": round(float(dose_change_pct), 2),
                        "first_area_cm2": round(float(first_area), 4),
                        "last_area_cm2": round(float(last_area), 4),
                        "lesion_change_pct": round(float(lesion_change_pct), 2),
                    }
                )
        return results

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
