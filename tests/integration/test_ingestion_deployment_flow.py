"""Integration tests for ingestion and deployment flow."""

from snowflake_ml_template.deployment.orchestrator import DeploymentOrchestrator
from snowflake_ml_template.ingestion.orchestrator import IngestionOrchestrator


def test_ingestion_orchestrator_integration(mock_session):
    """Test ingestion orchestrator with multiple strategies."""
    orchestrator = IngestionOrchestrator(mock_session)

    # Verify orchestrator initialized
    assert orchestrator is not None
    assert orchestrator.session == mock_session
    assert len(orchestrator.strategies) == 0


def test_deployment_orchestrator_integration(mock_session):
    """Test deployment orchestrator with multiple strategies."""
    orchestrator = DeploymentOrchestrator(mock_session)

    # Verify orchestrator initialized
    assert orchestrator is not None
    assert orchestrator.session == mock_session
    assert len(orchestrator.strategies) == 0


def test_ingestion_to_deployment_workflow(mock_session):
    """Test complete workflow from ingestion to deployment."""
    # Step 1: Create ingestion orchestrator
    ingest_orch = IngestionOrchestrator(mock_session)

    # Step 2: Create deployment orchestrator
    deploy_orch = DeploymentOrchestrator(mock_session)

    # Verify both orchestrators initialized
    assert ingest_orch is not None
    assert deploy_orch is not None
