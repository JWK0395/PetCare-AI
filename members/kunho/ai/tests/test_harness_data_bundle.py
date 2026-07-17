from __future__ import annotations

from pathlib import Path

from petcare_agent.harness.data_bundle import DataBundle
from petcare_agent.harness.fake_backend import DataBundleBackendProvider


def test_data_bundle_backend_builds_handoff_context_from_db_tables() -> None:
    bundle = DataBundle.load(_example_bundle_path())
    context = DataBundleBackendProvider(bundle).load_context(1, days=3)

    assert context.pet["name"] == "Mochi"
    assert [entry["record_date"] for entry in context.recent_daily_entries] == [
        "2026-07-16",
        "2026-07-15",
        "2026-07-14",
    ]
    assert "raw_text" not in context.recent_daily_entries[0]
    assert context.data_from == "2026-07-14"
    assert context.data_to == "2026-07-16"
    assert context.diagnoses[0]["diagnosis"] == "mild seasonal allergy"
    assert context.medical_background == {
        "conditions": ["mild seasonal allergy"],
        "medications_or_supplements": ["omega-3 supplement (once daily)"],
        "allergies": [],
    }


def _example_bundle_path() -> Path:
    return Path(__file__).resolve().parents[4] / "examples" / "data_bundles" / "cat_cough_minimal"
