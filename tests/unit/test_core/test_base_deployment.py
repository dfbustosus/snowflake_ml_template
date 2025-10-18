"""Unit tests for BaseDeploymentStrategy governance hooks."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytest

from snowflake_ml_template.core.base.deployment import (
    BaseDeploymentStrategy,
    DeploymentConfig,
    DeploymentResult,
    DeploymentStatus,
    DeploymentStrategy,
    DeploymentTarget,
)
from snowflake_ml_template.core.base.tracking import ExecutionEventTracker


class RecordingTracker(ExecutionEventTracker):
    """Tracker implementation that records emitted events."""

    def __init__(self) -> None:
        """Initialize the tracker."""
        self.events: List[Tuple[str, str, Dict[str, object]]] = []

    def record_event(
        self, component: str, event: str, payload: Dict[str, object]
    ) -> None:
        """Record an event."""
        self.events.append((component, event, payload))


class DummyDeploymentStrategy(BaseDeploymentStrategy):
    """Concrete strategy exercising BaseDeploymentStrategy hooks."""

    def __init__(self, config: DeploymentConfig, tracker=None) -> None:
        """Initialize the strategy."""
        super().__init__(config, tracker=tracker)
        self.pre_compliance_actions: List[Tuple[str, Dict[str, object]]] = []
        self.post_compliance_reports: List[
            Tuple[str, Dict[str, object], Dict[str, object]]
        ] = []
        self.compliance_failures: List[Tuple[str, Exception]] = []
        self.raise_compliance_error = False
        self.raise_post_compliance_error = False
        self.undeploy_success = True
        self.health_status: Dict[str, object] = {"healthy": True}

    def pre_compliance_check(self, action: str, context: Dict[str, object]) -> None:
        """Pre-compliance check hook."""
        self.pre_compliance_actions.append((action, context))

    def validate_compliance(
        self, action: str, context: Dict[str, object]
    ) -> Dict[str, object]:
        """Validate compliance hook."""
        if self.raise_compliance_error and action == "deploy":
            raise RuntimeError("compliance failed")
        return {"action": action, "validated": True}

    def post_compliance_check(
        self,
        action: str,
        outcome: Dict[str, object],
        report: Dict[str, object],
    ) -> None:
        """Post-compliance check hook."""
        if self.raise_post_compliance_error and action == "undeploy":
            raise RuntimeError("post compliance error")
        self.post_compliance_reports.append((action, outcome, report))

    def on_compliance_failure(self, action: str, error: Exception) -> None:
        """Error hook."""
        self.compliance_failures.append((action, error))

    def deploy(self, **kwargs) -> DeploymentResult:
        """Deploy hook."""
        return DeploymentResult(
            status=DeploymentStatus.SUCCESS,
            strategy=self.config.strategy,
            target=self.config.target,
            deployment_name=self.config.deployment_name,
        )

    def undeploy(self) -> bool:
        """Undeploy hook."""
        return self.undeploy_success

    def validate(self) -> bool:
        """Validate hook."""
        return True

    def health_check(self) -> Dict[str, object]:
        """Health check hook."""
        return self.health_status


@pytest.fixture()
def deployment_config() -> DeploymentConfig:
    """Build a minimal deployment configuration for tests."""

    return DeploymentConfig(
        strategy=DeploymentStrategy.WAREHOUSE_UDF,
        target=DeploymentTarget.BATCH,
        model_name="model",
        model_version="1.0.0",
        model_artifact_path="@stage/model.joblib",
        deployment_database="DB",
        deployment_schema="SC",
        deployment_name="DEPLOYMENT",
        warehouse="WH",
    )


def test_deployment_compliance_hooks_success(
    deployment_config: DeploymentConfig,
) -> None:
    """Ensure compliance hooks execute on successful deployment lifecycle."""

    tracker = RecordingTracker()
    strategy = DummyDeploymentStrategy(deployment_config, tracker=tracker)

    result = strategy.execute_deploy()

    assert result.deployment_status == DeploymentStatus.SUCCESS
    assert strategy.pre_compliance_actions[0][0] == "deploy"
    assert strategy.post_compliance_reports[0][0] == "deploy"
    assert not strategy.compliance_failures
    assert tracker.events[-1][1] == "deployment_end"

    success = strategy.execute_undeploy()

    assert success is True
    assert strategy.pre_compliance_actions[-1][0] == "undeploy"
    assert strategy.post_compliance_reports[-1][0] == "undeploy"

    status = strategy.execute_health_check()

    assert status == {"healthy": True}
    assert strategy.pre_compliance_actions[-1][0] == "health_check"
    assert strategy.post_compliance_reports[-1][0] == "health_check"


def test_deployment_compliance_failure(deployment_config: DeploymentConfig) -> None:
    """Validate that compliance failures invoke hooks and bubble up."""

    strategy = DummyDeploymentStrategy(deployment_config)
    strategy.raise_compliance_error = True

    with pytest.raises(RuntimeError, match="compliance failed"):
        strategy.execute_deploy()

    assert strategy.compliance_failures
    action, error = strategy.compliance_failures[-1]
    assert action == "deploy"
    assert str(error) == "compliance failed"


def test_post_compliance_failure_on_undeploy(
    deployment_config: DeploymentConfig,
) -> None:
    """Verify post-compliance errors during undeploy are surfaced."""

    strategy = DummyDeploymentStrategy(deployment_config)
    strategy.raise_post_compliance_error = True

    with pytest.raises(RuntimeError, match="post compliance error"):
        strategy.execute_undeploy()

    assert strategy.compliance_failures
    action, error = strategy.compliance_failures[-1]
    assert action == "undeploy"
    assert str(error) == "post compliance error"
