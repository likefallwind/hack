from pathlib import Path

from flask import Flask, jsonify, request, send_file

from database import ClinicalMultimodalDB


def create_app(db_path: str = "database/clinical_multimodal.db") -> Flask:
    app = Flask(__name__)
    db = ClinicalMultimodalDB(db_path=db_path)

    def _require(payload: dict, keys: list) -> tuple:
        missing = [key for key in keys if key not in payload]
        return (len(missing) == 0, missing)

    @app.get("/health")
    def health() -> tuple:
        return jsonify({"status": "ok"}), 200

    @app.post("/patients/upsert")
    def upsert_patient() -> tuple:
        payload = request.get_json(force=True)
        ok, missing = _require(payload, ["patient_id"])
        if not ok:
            return jsonify({"ok": False, "missing": missing}), 400
        db.upsert_patient(payload)
        return jsonify({"ok": True, "patient_id": payload.get("patient_id")}), 200

    @app.post("/visits")
    def add_visit() -> tuple:
        payload = request.get_json(force=True)
        ok, missing = _require(payload, ["visit_id", "patient_id", "visit_time"])
        if not ok:
            return jsonify({"ok": False, "missing": missing}), 400
        db.add_visit(payload)
        return jsonify({"ok": True, "visit_id": payload.get("visit_id")}), 201

    @app.post("/medications")
    def add_medication() -> tuple:
        payload = request.get_json(force=True)
        ok, missing = _require(payload, ["visit_id", "medication"])
        if not ok:
            return jsonify({"ok": False, "missing": missing}), 400
        db.add_medication(payload["visit_id"], payload["medication"])
        return jsonify({"ok": True}), 201

    @app.post("/lesions")
    def add_lesion() -> tuple:
        payload = request.get_json(force=True)
        ok, missing = _require(payload, ["visit_id", "lesion"])
        if not ok:
            return jsonify({"ok": False, "missing": missing}), 400
        db.add_lesion_measurement(payload["visit_id"], payload["lesion"])
        return jsonify({"ok": True}), 201

    @app.post("/labs")
    def add_lab() -> tuple:
        payload = request.get_json(force=True)
        ok, missing = _require(payload, ["visit_id", "lab"])
        if not ok:
            return jsonify({"ok": False, "missing": missing}), 400
        db.add_lab_result(payload["visit_id"], payload["lab"])
        return jsonify({"ok": True}), 201

    @app.post("/assets")
    def add_asset() -> tuple:
        payload = request.get_json(force=True)
        ok, missing = _require(payload, ["asset_id", "visit_id", "asset_type", "object_uri"])
        if not ok:
            return jsonify({"ok": False, "missing": missing}), 400
        db.add_asset(payload)
        return jsonify({"ok": True, "asset_id": payload.get("asset_id")}), 201

    @app.get("/assets/<asset_id>/download")
    def download_asset(asset_id: str):
        asset = db.get_asset(asset_id)
        if not asset:
            return jsonify({"ok": False, "message": "asset not found"}), 404
        uri = asset.get("object_uri", "")
        if uri.startswith(("http://", "https://", "oss://", "s3://")):
            return jsonify({"ok": True, "download_url": uri, "mode": "redirect_url"}), 200
        file_path = Path(uri)
        if not file_path.exists() or not file_path.is_file():
            return jsonify({"ok": False, "message": "local file not found", "object_uri": uri}), 404
        return send_file(file_path, as_attachment=True)

    @app.post("/extract/auto-fill")
    def auto_fill() -> tuple:
        payload = request.get_json(force=True)
        ok, missing = _require(payload, ["text"])
        if not ok:
            return jsonify({"ok": False, "missing": missing}), 400
        result = db.auto_fill_from_text(payload["text"])
        return jsonify(result), 200

    @app.post("/extract/auto-fill/batch")
    def auto_fill_batch() -> tuple:
        payload = request.get_json(force=True)
        ok, missing = _require(payload, ["texts"])
        if not ok:
            return jsonify({"ok": False, "missing": missing}), 400
        result = db.batch_auto_fill_from_texts(payload["texts"])
        return jsonify({"count": len(result), "items": result}), 200

    @app.post("/quality/iqa")
    def score_iqa() -> tuple:
        payload = request.get_json(force=True)
        ok, missing = _require(payload, ["blur_score", "exposure_score", "lesion_ratio"])
        if not ok:
            return jsonify({"ok": False, "missing": missing}), 400
        result = db.score_iqa(
            blur_score=float(payload["blur_score"]),
            exposure_score=float(payload["exposure_score"]),
            lesion_ratio=float(payload["lesion_ratio"]),
        )
        return jsonify(result), 200

    @app.post("/search/semantic")
    def semantic_search() -> tuple:
        payload = request.get_json(force=True)
        ok, missing = _require(payload, ["query"])
        if not ok:
            return jsonify({"ok": False, "missing": missing}), 400
        query = payload["query"]
        top_k = int(payload.get("top_k", 5))
        hits = db.semantic_search(query, top_k=top_k)
        return (
            jsonify(
                {
                    "query": query,
                    "hits": [
                        {
                            "patient_id": h.patient_id,
                            "visit_id": h.visit_id,
                            "score": h.score,
                            "note_text": h.note_text,
                        }
                        for h in hits
                    ],
                }
            ),
            200,
        )

    @app.post("/search/rag")
    def rag() -> tuple:
        payload = request.get_json(force=True)
        ok, missing = _require(payload, ["query"])
        if not ok:
            return jsonify({"ok": False, "missing": missing}), 400
        query = payload["query"]
        top_k = int(payload.get("top_k", 3))
        return jsonify(db.rag_answer(query, top_k=top_k)), 200

    @app.post("/ingest/multimodal")
    def ingest_multimodal() -> tuple:
        payload = request.get_json(force=True)
        ok, missing = _require(payload, ["patient", "visit", "text"])
        if not ok:
            return jsonify({"ok": False, "missing": missing}), 400
        db.upsert_patient(payload["patient"])
        db.add_visit(payload["visit"])
        extracted = db.auto_fill_from_text(payload["text"])
        persist = bool(payload.get("persist_auto_fill", True))
        committed = None
        if persist:
            committed = db.commit_auto_fill(
                visit_id=payload["visit"]["visit_id"],
                extracted=extracted,
                operator=payload.get("operator", "system"),
                manual_overrides=payload.get("manual_overrides"),
            )
        return jsonify({"ok": True, "extracted": extracted, "committed": committed}), 201

    @app.post("/review/commit")
    def review_commit() -> tuple:
        payload = request.get_json(force=True)
        ok, missing = _require(payload, ["visit_id", "extracted"])
        if not ok:
            return jsonify({"ok": False, "missing": missing}), 400
        result = db.commit_auto_fill(
            visit_id=payload["visit_id"],
            extracted=payload["extracted"],
            operator=payload.get("operator", "reviewer"),
            manual_overrides=payload.get("manual_overrides"),
        )
        return jsonify({"ok": True, "committed": result}), 200

    @app.get("/patients/<patient_id>/changes")
    def detect_changes(patient_id: str) -> tuple:
        item_names = request.args.get("items")
        lab_items = item_names.split(",") if item_names else None
        return jsonify(db.detect_changes(patient_id, lab_items=lab_items)), 200

    @app.get("/patients/<patient_id>/snapshot")
    def snapshot(patient_id: str) -> tuple:
        data = db.export_patient_snapshot(patient_id)
        if not data:
            return jsonify({"message": "patient not found"}), 404
        return jsonify(data), 200

    @app.post("/graph/extract")
    def extract_graph() -> tuple:
        payload = request.get_json(force=True)
        ok, missing = _require(payload, ["text"])
        if not ok:
            return jsonify({"ok": False, "missing": missing}), 400
        return jsonify(db.extract_knowledge_graph(payload["text"])), 200

    @app.get("/patients/<patient_id>/graph")
    def patient_graph(patient_id: str) -> tuple:
        return jsonify(db.build_patient_knowledge_graph(patient_id)), 200

    @app.get("/patients/<patient_id>/evolution")
    def patient_evolution(patient_id: str) -> tuple:
        return jsonify(db.lesion_evolution(patient_id)), 200

    @app.get("/audits")
    def audits() -> tuple:
        row_ref = request.args.get("row_ref")
        limit = int(request.args.get("limit", 100))
        return jsonify({"items": db.get_audit_logs(row_ref=row_ref, limit=limit)}), 200

    @app.get("/research/query")
    def research_query() -> tuple:
        year = request.args.get("year")
        if not year:
            return jsonify({"ok": False, "missing": ["year"]}), 400
        reduction = float(request.args.get("lesion_reduction_pct", 20))
        items = db.research_query_dose_up_lesion_down(
            year=int(year),
            lesion_reduction_pct=reduction,
        )
        return jsonify({"ok": True, "count": len(items), "items": items}), 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=True)
