from pathlib import Path
from unittest.mock import MagicMock, patch

from agents import orchestrator


def test_phase_one_runs_profiler_then_sttm_without_changing_checkpoint(tmp_path):
    calls = []
    scratchpad = {}

    class FakeTool:
        def __init__(self, name, output_key, output_path):
            self.name = name
            self.output_key = output_key
            self.output_path = output_path

        def invoke(self, args):
            calls.append((self.name, args["goal"]))
            scratchpad[self.output_key] = str(self.output_path)
            return "completed"

    profile_path = tmp_path / "profile.json"
    sttm_path = tmp_path / "bronze.csv"
    profiler = FakeTool("profiler", "profile_path", profile_path)
    sttm = FakeTool("sttm", "sttm_bronze_path", sttm_path)

    with (
        patch.object(orchestrator, "_make_phase1_tools", return_value=(profiler, sttm, scratchpad)),
        patch.object(orchestrator, "store_document"),
        patch.object(orchestrator, "AuditLogger"),
    ):
        state = orchestrator.run_until_bronze_sttm(["input.csv"], "Analyze sales")

    assert [name for name, _ in calls] == ["profiler", "sttm"]
    assert state["status"] == "awaiting_bronze_sttm_approval"
    assert state["profile_path"] == str(profile_path)
    assert state["sttm_bronze_path"] == str(sttm_path)


def test_phase_four_runs_missing_gold_and_reporter_outputs(tmp_path):
    gold_path = tmp_path / "gold.parquet"
    gold_path.touch()
    report_path = tmp_path / "report.html"
    scratchpad = {}

    gold_tool = MagicMock()
    gold_tool.invoke.side_effect = lambda _: (
        scratchpad.update(gold_output_paths=[str(gold_path)]) or "Gold created"
    )
    reporter_tool = MagicMock()

    def create_report(_):
        report_path.write_text("<html>Report</html>", encoding="utf-8")
        scratchpad["report_path"] = str(report_path)
        return "Report created"

    reporter_tool.invoke.side_effect = create_report

    state = {
        "run_id": "test-run",
        "silver_output_paths": [],
        "sttm_gold_path": "gold-sttm.csv",
        "business_intent": "Test question",
    }

    with (
        patch.object(orchestrator, "_make_phase4_tools", return_value=(gold_tool, reporter_tool, scratchpad)),
        patch.object(orchestrator, "AuditLogger"),
    ):
        result = orchestrator.run_gold_and_report(state)

    gold_tool.invoke.assert_called_once()
    reporter_tool.invoke.assert_called_once()
    assert result["status"] == "completed"
    assert result["report_path"] == str(report_path)
    assert Path(result["report_path"]).is_file()
