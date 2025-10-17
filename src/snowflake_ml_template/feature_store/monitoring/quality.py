"""Feature quality monitoring.

This module provides data quality monitoring for features, including:
- Completeness (null rate)
- Uniqueness
- Value range validation
- Statistical properties
"""

from dataclasses import dataclass
from typing import List, Optional

from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark.functions import avg, col
from snowflake.snowpark.functions import max as max_
from snowflake.snowpark.functions import min as min_
from snowflake.snowpark.functions import stddev

from snowflake_ml_template.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for a feature.

    Attributes:
        feature_name: Name of the feature
        total_rows: Total number of rows
        null_count: Number of null values
        null_rate: Proportion of null values
        unique_count: Number of unique values
        mean: Mean value (numeric features)
        std: Standard deviation (numeric features)
        min: Minimum value
        max: Maximum value
        quality_score: Overall quality score (0-1)
    """

    feature_name: str
    total_rows: int
    null_count: int
    null_rate: float
    unique_count: Optional[int] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    quality_score: float = 1.0


class FeatureQualityMonitor:
    """Monitor feature data quality.

    This class provides comprehensive quality monitoring for features,
    helping identify data quality issues that may impact model performance.

    Attributes:
        session: Snowflake session
        logger: Structured logger

    Example:
        >>> monitor = FeatureQualityMonitor(session)
        >>>
        >>> # Calculate quality metrics
        >>> metrics = monitor.calculate_quality_metrics(
        ...     df=features_df,
        ...     feature_col="AGE"
        ... )
        >>>
        >>> if metrics.null_rate > 0.1:
        ...     print(f"High null rate: {metrics.null_rate}")
    """

    def __init__(self, session: Session):
        """Initialize the quality monitor.

        Args:
            session: Active Snowflake session
        """
        if session is None:
            raise ValueError("Session cannot be None")

        self.session = session
        self.logger = get_logger(__name__)

    def calculate_quality_metrics(
        self, df: DataFrame, feature_col: str
    ) -> QualityMetrics:
        """Calculate quality metrics for a feature.

        Args:
            df: DataFrame containing the feature
            feature_col: Name of feature column

        Returns:
            QualityMetrics with comprehensive quality information
        """
        self.logger.info(
            f"Calculating quality metrics for: {feature_col}",
            extra={"feature": feature_col},
        )

        # Basic metrics
        total_rows = df.count()
        null_count = df.filter(col(feature_col).isNull()).count()
        null_rate = null_count / total_rows if total_rows > 0 else 0.0

        # Unique count
        unique_count = df.select(feature_col).distinct().count()

        # Statistical metrics (for numeric features)
        try:
            stats = df.select(
                avg(col(feature_col)).alias("mean"),
                stddev(col(feature_col)).alias("std"),
                min_(col(feature_col)).alias("min"),
                max_(col(feature_col)).alias("max"),
            ).collect()[0]

            mean = float(stats["MEAN"]) if stats["MEAN"] is not None else None
            std = float(stats["STD"]) if stats["STD"] is not None else None
            min_val = float(stats["MIN"]) if stats["MIN"] is not None else None
            max_val = float(stats["MAX"]) if stats["MAX"] is not None else None
        except Exception as e:
            self.logger.error(
                f"Failed to calculate quality metrics for {feature_col}: {e}",
                extra={"feature": feature_col, "error": str(e)},
            )
            mean = std = min_val = max_val = None

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            null_rate, unique_count, total_rows
        )

        metrics = QualityMetrics(
            feature_name=feature_col,
            total_rows=total_rows,
            null_count=null_count,
            null_rate=null_rate,
            unique_count=unique_count,
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            quality_score=quality_score,
        )

        self.logger.info(
            f"Quality metrics calculated for {feature_col}",
            extra={
                "feature": feature_col,
                "null_rate": null_rate,
                "quality_score": quality_score,
            },
        )

        return metrics

    def _calculate_quality_score(
        self, null_rate: float, unique_count: int, total_rows: int
    ) -> float:
        """Calculate overall quality score (0-1).

        Args:
            null_rate: Proportion of null values
            unique_count: Number of unique values
            total_rows: Total number of rows

        Returns:
            Quality score between 0 and 1
        """
        # Completeness score (1 - null_rate)
        completeness = 1.0 - null_rate

        # Uniqueness score (penalize if all values are the same)
        uniqueness = min(unique_count / max(total_rows * 0.1, 1), 1.0)

        # Weighted average
        quality_score = 0.7 * completeness + 0.3 * uniqueness

        return quality_score

    def monitor_quality_batch(
        self, df: DataFrame, feature_cols: List[str]
    ) -> List[QualityMetrics]:
        """Monitor quality for multiple features.

        Args:
            df: DataFrame containing features
            feature_cols: List of feature columns to monitor

        Returns:
            List of QualityMetrics for each feature
        """
        results = []

        for feature_col in feature_cols:
            try:
                metrics = self.calculate_quality_metrics(df, feature_col)
                results.append(metrics)
            except Exception as e:
                self.logger.error(
                    f"Failed to calculate quality for {feature_col}: {e}",
                    extra={"feature": feature_col, "error": str(e)},
                )

        # Summary
        avg_quality = (
            sum(m.quality_score for m in results) / len(results) if results else 0.0
        )
        self.logger.info(
            f"Quality monitoring complete. Average quality: {avg_quality:.2f}",
            extra={"num_features": len(results), "avg_quality": avg_quality},
        )

        return results
