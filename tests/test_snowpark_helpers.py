"""Unit tests for Snowpark helpers (pandas<->snowpark conversions)."""


def test_pandas_to_snowpark_roundtrip():
    """Roundtrip conversion between pandas and a fake Snowpark DataFrame."""

    # Create a fake session with create_dataframe
    class FakeSDF:
        def __init__(self, pdf):
            self._pdf = pdf

        def to_pandas(self):
            return self._pdf

    class FakeSession:
        def create_dataframe(self, pdf):
            return FakeSDF(pdf)

    import pandas as pd

    pdf = pd.DataFrame({"a": [1, 2, 3]})
    from snowflake_ml_template.snowpark import helpers

    session = FakeSession()
    sdf = helpers.pandas_to_snowpark(session, pdf)
    assert hasattr(sdf, "to_pandas")
    restored = helpers.snowpark_to_pandas(sdf)
    pd.testing.assert_frame_equal(restored, pdf)


def test_pandas_to_snowpark_bad_session():
    """Ensure pandas_to_snowpark raises when session does not provide create_dataframe."""
    import pandas as pd

    from snowflake_ml_template.snowpark import helpers

    pdf = pd.DataFrame({"a": [1]})
    with __import__("pytest").raises(TypeError):
        helpers.pandas_to_snowpark(object(), pdf)
