"""Feature Store helper utilities.

These helpers only import Snowflake-specific modules when methods are invoked
so that the package can be imported in environments without Snowflake libs.
"""

from typing import Any, Dict, Optional


class FeatureStoreHelper:
    """Helper to perform common Feature Store operations.

    Usage:
        helper = FeatureStoreHelper()
        helper.register_entity(session, name, join_keys)
    """

    def __init__(self) -> None:
        """Initialize configuration placeholders for the helper."""
        # placeholder for future configuration
        self._cfg: Dict[str, Any] = {}

    def register_entity(self, session: Any, name: str, join_keys: list) -> None:
        """Register an Entity object in the feature store.

        This method expects a live Snowpark session with the Snowflake ML
        Feature Store available. It will import the necessary classes at call time.
        """
        # lazy import to avoid hard dependency at module import time
        from snowflake.ml.feature_store import Entity

        entity = Entity(name=name, join_keys=join_keys)
        # assume `fs` object exists or can be created from session
        from snowflake.ml.feature_store import FeatureStore

        fs = FeatureStore(session=session)
        fs.register_entity(entity)

    def register_feature_view(
        self,
        session: Any,
        name: str,
        feature_df: Any,
        entities: list,
        refresh_freq: Optional[str] = None,
        version: str = "v1",
    ) -> None:
        """Register a FeatureView. Expects a Snowpark DataFrame for feature_df."""
        from snowflake.ml.feature_store import FeatureStore, FeatureView

        fs = FeatureStore(session=session)
        fv = FeatureView(
            name=name,
            entities=entities,
            feature_df=feature_df,
            refresh_freq=refresh_freq,
        )
        fs.register_feature_view(feature_view=fv, version=version)
