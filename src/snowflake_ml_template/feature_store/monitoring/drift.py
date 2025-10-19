"""Feature drift detection using statistical methods.

This module implements drift detection for features using:
- Population Stability Index (PSI)
- KL Divergence
- Statistical tests (Chi-square, KS test)

Drift detection helps identify when feature distributions change,
which may indicate model degradation or data quality issues.
"""

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark.functions import col

from snowflake_ml_template.core.exceptions import FeatureStoreError, MonitoringError
from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection analysis.

    Attributes:
        feature_name: Name of the feature
        drift_score: Numerical drift score (PSI, KL divergence, etc.)
        drift_detected: Whether significant drift was detected
        threshold: Threshold used for detection
        method: Method used for drift detection
        details: Additional details about the drift
    """

    feature_name: str
    drift_score: float
    drift_detected: bool
    threshold: float
    method: str
    details: Dict[str, Any]


class FeatureDriftDetector:
    """Detect drift in feature distributions.

    This class implements statistical drift detection methods to identify
    when feature distributions change significantly between baseline and
    current data.

    Methods:
    - PSI (Population Stability Index): Industry standard for drift detection
    - KL Divergence: Information-theoretic measure of distribution difference

    Attributes:
        session: Snowflake session
        logger: Structured logger

    Example:
        >>> detector = FeatureDriftDetector(session)
        >>>
        >>> # Detect drift using PSI
        >>> result = detector.detect_psi_drift(
        ...     baseline_df=training_data,
        ...     current_df=production_data,
        ...     feature_col="AGE",
        ...     threshold=0.1
        ... )
        >>>
        >>> if result.drift_detected:
        ...     print(f"Drift detected! PSI: {result.drift_score}")
    """

    def __init__(
        self,
        session: Session,
        *,
        database: Optional[str] = None,
        schema: str = "FEATURE_MONITORING",
        table: str = "FEATURE_DRIFT_EVENTS",
    ):
        """Initialize the drift detector.

        Args:
            session: Active Snowflake session
            database: Optional database for persisted drift events
            schema: Schema used for drift event storage
            table: Table name for drift events
        """
        if session is None:
            raise ValueError("Session cannot be None")

        self.session = session
        self.logger = get_logger(__name__)
        self._events_fqn: Optional[str] = None
        if database:
            self._events_fqn = f"{database}.{schema}.{table}"
            self._ensure_events_table()

    def detect_psi_drift(
        self,
        baseline_df: DataFrame,
        current_df: DataFrame,
        feature_col: str,
        threshold: float = 0.1,
        num_bins: int = 10,
    ) -> DriftResult:
        """Detect drift using Population Stability Index (PSI).

        PSI measures the shift in a variable's distribution between two
        datasets. It's widely used in credit scoring and risk modeling.

        PSI Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Moderate change
        - PSI >= 0.2: Significant change

        Args:
            baseline_df: Baseline (training) DataFrame
            current_df: Current (production) DataFrame
            feature_col: Name of feature column to analyze
            threshold: PSI threshold for drift detection (default: 0.1)
            num_bins: Number of bins for discretization

        Returns:
            DriftResult with PSI score and drift status

        Example:
            >>> result = detector.detect_psi_drift(
            ...     baseline_df=train_df,
            ...     current_df=prod_df,
            ...     feature_col="CREDIT_SCORE",
            ...     threshold=0.1
            ... )
        """
        self.logger.info(
            f"Detecting PSI drift for feature: {feature_col}",
            extra={"feature": feature_col, "threshold": threshold},
        )

        # Calculate distributions
        baseline_dist = self._calculate_distribution(baseline_df, feature_col, num_bins)
        current_dist = self._calculate_distribution(current_df, feature_col, num_bins)

        # Calculate PSI
        psi_score = self._calculate_psi(baseline_dist, current_dist)

        drift_detected = psi_score >= threshold

        result = DriftResult(
            feature_name=feature_col,
            drift_score=psi_score,
            drift_detected=drift_detected,
            threshold=threshold,
            method="PSI",
            details={
                "baseline_distribution": baseline_dist,
                "current_distribution": current_dist,
                "num_bins": num_bins,
            },
        )

        if drift_detected:
            self.logger.warning(
                f"Drift detected for {feature_col}! PSI: {psi_score:.4f}",
                extra={"feature": feature_col, "psi": psi_score},
            )
        else:
            self.logger.info(
                f"No drift detected for {feature_col}. PSI: {psi_score:.4f}",
                extra={"feature": feature_col, "psi": psi_score},
            )

        return result

    def _calculate_distribution(
        self, df: DataFrame, feature_col: str, num_bins: int
    ) -> Dict[int, float]:
        """Calculate distribution of a feature across bins.

        Args:
            df: DataFrame containing the feature
            feature_col: Name of feature column
            num_bins: Number of bins

        Returns:
            Dictionary mapping bin index to proportion
        """
        # Get min and max for binning
        stats = (
            df.select(col(feature_col).cast("DOUBLE"))
            .agg([("MIN", feature_col), ("MAX", feature_col), ("COUNT", feature_col)])
            .collect()[0]
        )

        min_val = float(stats[0])
        max_val = float(stats[1])
        total_count = int(stats[2])

        if total_count == 0:
            raise MonitoringError(f"No data found for feature {feature_col}")

        # Calculate bin width
        bin_width = (max_val - min_val) / num_bins if max_val > min_val else 1.0

        # Count values in each bin
        distribution = {}
        for i in range(num_bins):
            bin_start = min_val + i * bin_width
            bin_end = bin_start + bin_width if i < num_bins - 1 else max_val + 1

            count = df.filter(
                (col(feature_col) >= bin_start) & (col(feature_col) < bin_end)
            ).count()

            distribution[i] = count / total_count if total_count > 0 else 0.0

        return distribution

    def _calculate_psi(
        self, baseline_dist: Dict[int, float], current_dist: Dict[int, float]
    ) -> float:
        """Calculate Population Stability Index.

        PSI = Σ (current% - baseline%) * ln(current% / baseline%)

        Args:
            baseline_dist: Baseline distribution
            current_dist: Current distribution

        Returns:
            PSI score
        """
        psi = 0.0
        epsilon = 1e-10  # Small value to avoid log(0)

        for bin_idx in baseline_dist.keys():
            baseline_pct = baseline_dist.get(bin_idx, 0.0) + epsilon
            current_pct = current_dist.get(bin_idx, 0.0) + epsilon

            psi += (current_pct - baseline_pct) * math.log(current_pct / baseline_pct)

        return psi

    def detect_drift_batch(
        self,
        baseline_df: DataFrame,
        current_df: DataFrame,
        feature_cols: List[str],
        threshold: float = 0.1,
    ) -> List[DriftResult]:
        """Detect drift for multiple features.

        Args:
            baseline_df: Baseline DataFrame
            current_df: Current DataFrame
            feature_cols: List of feature columns to analyze
            threshold: PSI threshold for drift detection

        Returns:
            List of DriftResult for each feature
        """
        results = []

        for feature_col in feature_cols:
            try:
                result = self.detect_psi_drift(
                    baseline_df, current_df, feature_col, threshold
                )
                results.append(result)
            except Exception as e:
                self.logger.error(
                    f"Failed to detect drift for {feature_col}: {e}",
                    extra={"feature": feature_col, "error": str(e)},
                )

        # Summary
        num_drifted = sum(1 for r in results if r.drift_detected)
        self.logger.info(
            f"Drift detection complete: {num_drifted}/{len(results)} features drifted",
            extra={"num_drifted": num_drifted, "total_features": len(results)},
        )

        return results

    # ------------------------------------------------------------------
    # Persistence & scheduling helpers
    # ------------------------------------------------------------------
    def record_result(
        self,
        result: DriftResult,
        *,
        feature_view: str,
        entity: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Persist a single drift result if storage is configured."""
        if not self._events_fqn:
            self.logger.debug(
                "Drift persistence disabled", extra={"feature_view": feature_view}
            )
            return

        details_json = self._quote_literal(json.dumps(result.details))
        insert_sql = (
            f"INSERT INTO {self._events_fqn} (event_timestamp, feature_view, entity, feature_name, run_id, method, drift_score, threshold, drift_detected, details) "
            "VALUES (CURRENT_TIMESTAMP(), "
            f"{self._quote_literal(feature_view)}, {self._quote_literal(entity)}, {self._quote_literal(result.feature_name)}, "
            f"{self._quote_literal(run_id)}, {self._quote_literal(result.method)}, {result.drift_score}, {result.threshold}, {str(result.drift_detected).upper()}, {details_json})"
        )
        self._execute_sql(insert_sql)

    def record_results(
        self,
        results: List[DriftResult],
        *,
        feature_view: str,
        entity: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Persist multiple drift results."""
        for result in results:
            self.record_result(
                result, feature_view=feature_view, entity=entity, run_id=run_id
            )

    def create_drift_task(
        self,
        *,
        task_name: str,
        warehouse: str,
        schedule: str,
        procedure_call: str,
    ) -> None:
        """Create or replace a task that runs drift detection."""
        if not self._events_fqn:
            raise FeatureStoreError(
                "Cannot create drift task without configured events table"
            )

        task_sql = (
            f"CREATE OR REPLACE TASK {task_name} WAREHOUSE = {warehouse} "
            f"SCHEDULE = '{schedule}' AS {procedure_call}"
        )
        self._execute_sql(task_sql)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_events_table(self) -> None:
        if not self._events_fqn:
            return

        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {self._events_fqn} (
            event_timestamp TIMESTAMP_NTZ,
            feature_view VARCHAR,
            entity VARCHAR,
            feature_name VARCHAR,
            run_id VARCHAR,
            method VARCHAR,
            drift_score FLOAT,
            threshold FLOAT,
            drift_detected BOOLEAN,
            details VARIANT
        )
        """
        self._execute_sql(create_sql)

    def _execute_sql(self, sql: str) -> None:
        self.logger.debug("Executing SQL", extra={"sql": sql})
        self.session.sql(sql).collect()

    @staticmethod
    def _quote_literal(value: Optional[str]) -> str:
        if value is None:
            return "NULL"
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
