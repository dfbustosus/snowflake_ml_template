"""Unit tests for the credit card pipeline orchestration."""

from pathlib import Path


def test_creditcard_pipeline_runs_and_calls_session(monkeypatch, tmp_path):
    """Pipeline should call session.sql for DDL and pipeline SQL and call file.put."""

    calls = {"sql": [], "put": []}

    class FakeFileClient:
        def put(self, local, stage, overwrite=False):
            calls["put"].append((local, stage, overwrite))

    class FakeResult:
        def collect(self):
            return []

    class FakeSession:
        def __init__(self):
            self.file = FakeFileClient()

        def sql(self, stmt):
            calls["sql"].append(stmt)
            return FakeResult()

    # Monkeypatch the get_snowflake_session to return our fake session
    import importlib

    mod = importlib.import_module("snowflake_ml_template.pipelines.creditcard_pipeline")

    def fake_get_session():
        return FakeSession()

    monkeypatch.setattr(
        "snowflake_ml_template.pipelines.creditcard_pipeline.get_snowflake_session",
        lambda: fake_get_session(),
    )

    # Create minimal SQL files so pipeline will read them
    root = Path(mod.__file__).parents[3]
    ddl_dir = root / "scripts" / "snowflake" / "ddl"
    pipeline_dir = root / "scripts" / "snowflake" / "pipeline"
    ddl_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    # Add a simple DDL and pipeline SQL file
    (ddl_dir / "01_test_ddl.sql").write_text(
        "CREATE OR REPLACE TABLE test_dummy (id INT);"
    )
    (pipeline_dir / "01_copy_into_raw.sql").write_text("-- copy sql")
    (pipeline_dir / "02_process_stream.sql").write_text("-- merge sql")

    # Create a fake local csv file
    csv = tmp_path / "creditcard.csv"
    csv.write_text("col1,col2\n1,2\n")

    # Run pipeline
    Pipeline = mod.CreditCardPipeline
    p = Pipeline()
    p.run_full_pipeline(str(csv))

    # Assertions: file.put called with our csv and stage
    assert calls["put"], "file.put was not called"
    put_call = calls["put"][0]
    assert put_call[0] == str(csv)
    assert put_call[1] == "@ml_raw_stage"

    # Assertions: SQL was executed (at least the DDL and the embedded pipeline statements)
    executed = " ".join(calls["sql"]) if calls["sql"] else ""
    assert "CREATE OR REPLACE TABLE test_dummy" in executed
    assert "COPY INTO raw_creditcard" in executed
    assert "MERGE INTO creditcard" in executed
