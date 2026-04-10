import tempfile
import unittest
from pathlib import Path

from server.clinical_api import create_app


class ClinicalApiTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "clinical_api_test.db")
        self.app = create_app(db_path=self.db_path)
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_end_to_end_flow(self):
        patient = {
            "patient_id": "P100",
            "anonymized_id": "AX100",
            "sex": "F",
            "birth_year": 1992,
            "center": "center-a",
        }
        resp = self.client.post("/patients/upsert", json=patient)
        self.assertEqual(resp.status_code, 200)

        visit_1 = {
            "visit_id": "V100-1",
            "patient_id": "P100",
            "visit_time": "2026-03-01T10:00:00Z",
            "note_text": "他克莫司 2 mg bid，病灶面积: 12.5 cm2，CRP: 8.2 mg/L",
        }
        visit_2 = {
            "visit_id": "V100-2",
            "patient_id": "P100",
            "visit_time": "2026-04-01T10:00:00Z",
            "note_text": "剂量增加，病灶面积: 9.0 cm2，CRP: 4.1 mg/L",
        }
        self.assertEqual(self.client.post("/visits", json=visit_1).status_code, 201)
        self.assertEqual(self.client.post("/visits", json=visit_2).status_code, 201)
        ingest_payload = {
            "patient": patient,
            "visit": {
                "visit_id": "V100-0",
                "patient_id": "P100",
                "visit_time": "2026-02-01T10:00:00Z",
                "note_text": "红斑，他克莫司 1 mg qd，CRP: 10.0 mg/L，病灶面积: 15.0 cm2",
            },
            "text": "红斑，他克莫司 1 mg qd，CRP: 10.0 mg/L，病灶面积: 15.0 cm2",
            "persist_auto_fill": True,
            "operator": "bot",
        }
        ingest_resp = self.client.post("/ingest/multimodal", json=ingest_payload)
        self.assertEqual(ingest_resp.status_code, 201)
        self.assertTrue(ingest_resp.get_json()["ok"])

        auto_fill_resp = self.client.post("/extract/auto-fill", json={"text": visit_1["note_text"]})
        self.assertEqual(auto_fill_resp.status_code, 200)
        auto_fill_data = auto_fill_resp.get_json()
        self.assertTrue(len(auto_fill_data["medications"]) >= 1)
        batch_fill_resp = self.client.post("/extract/auto-fill/batch", json={"texts": [visit_1["note_text"], visit_2["note_text"]]})
        self.assertEqual(batch_fill_resp.status_code, 200)
        self.assertEqual(batch_fill_resp.get_json()["count"], 2)

        self.assertEqual(
            self.client.post("/lesions", json={"visit_id": "V100-1", "lesion": {"area_cm2": 12.5}}).status_code,
            201,
        )
        self.assertEqual(
            self.client.post("/lesions", json={"visit_id": "V100-2", "lesion": {"area_cm2": 9.0}}).status_code,
            201,
        )
        self.assertEqual(
            self.client.post("/labs", json={"visit_id": "V100-1", "lab": {"item_name": "CRP", "item_value": 8.2, "unit": "mg/L"}}).status_code,
            201,
        )
        self.assertEqual(
            self.client.post("/labs", json={"visit_id": "V100-2", "lab": {"item_name": "CRP", "item_value": 4.1, "unit": "mg/L"}}).status_code,
            201,
        )
        self.assertEqual(
            self.client.post("/medications", json={"visit_id": "V100-0", "medication": {"drug_name": "他克莫司", "dose_value": 1.0, "dose_unit": "mg", "frequency": "qd"}}).status_code,
            201,
        )
        self.assertEqual(
            self.client.post("/medications", json={"visit_id": "V100-2", "medication": {"drug_name": "他克莫司", "dose_value": 2.0, "dose_unit": "mg", "frequency": "bid"}}).status_code,
            201,
        )

        semantic = self.client.post("/search/semantic", json={"query": "病灶面积 CRP", "top_k": 3})
        self.assertEqual(semantic.status_code, 200)
        self.assertTrue(len(semantic.get_json()["hits"]) >= 1)

        rag = self.client.post("/search/rag", json={"query": "病灶面积变化"})
        self.assertEqual(rag.status_code, 200)
        self.assertIn("answer", rag.get_json())
        iqa = self.client.post("/quality/iqa", json={"blur_score": 0.9, "exposure_score": 0.8, "lesion_ratio": 0.2})
        self.assertEqual(iqa.status_code, 200)
        self.assertTrue(iqa.get_json()["qualified"])

        changes = self.client.get("/patients/P100/changes?items=CRP")
        self.assertEqual(changes.status_code, 200)
        change_data = changes.get_json()
        self.assertEqual(change_data["patient_id"], "P100")
        self.assertEqual(change_data["lesion_delta"]["delta_ratio_pct"], -28.0)

        snapshot = self.client.get("/patients/P100/snapshot")
        self.assertEqual(snapshot.status_code, 200)
        self.assertEqual(snapshot.get_json()["patient"]["patient_id"], "P100")

        review_payload = {
            "visit_id": "V100-2",
            "extracted": {
                "medications": [{"drug_name": "他克莫司", "dose_value": 2.0, "dose_unit": "mg", "frequency": "bid"}],
                "lesion": {"area_cm2": 9.0},
                "labs": [{"item_name": "CRP", "item_value": 4.1, "unit": "mg/L"}],
            },
            "operator": "doctor_a",
            "manual_overrides": {"lesion": {"area_cm2": 8.8, "pigmentation_score": 0.4}},
        }
        review = self.client.post("/review/commit", json=review_payload)
        self.assertEqual(review.status_code, 200)
        self.assertEqual(review.get_json()["committed"]["lesions"], 1)

        graph = self.client.post("/graph/extract", json={"text": "患者红斑，使用他克莫司 2 mg bid，CRP: 8.2 mg/L"})
        self.assertEqual(graph.status_code, 200)
        self.assertTrue(len(graph.get_json()["nodes"]) >= 1)

        patient_graph = self.client.get("/patients/P100/graph")
        self.assertEqual(patient_graph.status_code, 200)
        self.assertEqual(patient_graph.get_json()["patient_id"], "P100")

        evolution = self.client.get("/patients/P100/evolution")
        self.assertEqual(evolution.status_code, 200)
        self.assertTrue(len(evolution.get_json()["timeline"]) >= 2)
        audits = self.client.get("/audits?row_ref=V100-2&limit=20")
        self.assertEqual(audits.status_code, 200)
        self.assertTrue(len(audits.get_json()["items"]) >= 1)
        research = self.client.get("/research/query?year=2026&lesion_reduction_pct=20")
        self.assertEqual(research.status_code, 200)
        self.assertTrue(research.get_json()["count"] >= 1)

    def test_validation_guardrails(self):
        self.assertEqual(self.client.post("/patients/upsert", json={}).status_code, 400)
        self.assertEqual(self.client.post("/extract/auto-fill", json={}).status_code, 400)
        self.assertEqual(self.client.post("/search/semantic", json={}).status_code, 400)
        self.assertEqual(self.client.post("/graph/extract", json={}).status_code, 400)
        self.assertEqual(self.client.post("/ingest/multimodal", json={}).status_code, 400)
        self.assertEqual(self.client.post("/review/commit", json={}).status_code, 400)
        self.assertEqual(self.client.get("/research/query").status_code, 400)


if __name__ == "__main__":
    unittest.main()
