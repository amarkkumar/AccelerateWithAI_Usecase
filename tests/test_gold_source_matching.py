import pandas as pd

from agents.gold_agent import _apply_gold_rules


def test_gold_rules_match_sttm_names_with_underscores_to_spaced_files(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.gold_agent.GOLD_DIR", tmp_path)
    source_path = tmp_path / "stores 1 4_silver.parquet"
    pd.DataFrame({"store_id": [1, 2], "region": ["North", "South"]}).to_parquet(
        source_path, index=False
    )

    sttm_path = tmp_path / "sttm_gold.csv"
    pd.DataFrame(
        [
            {
                "source_table": "stores_1_4_silver",
                "source_column": "region",
                "target_table": "region_summary",
                "target_column": "region",
                "transformation_type": "Direct",
                "transformation_logic": "Passthrough",
            }
        ]
    ).to_csv(sttm_path, index=False)

    output_paths = _apply_gold_rules(
        [str(source_path)], str(sttm_path), "test-run"
    )

    assert len(output_paths) == 1
    output = pd.read_parquet(output_paths[0])
    assert output["region"].tolist() == ["North", "South"]
