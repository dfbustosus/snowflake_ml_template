"""SQL templates for Task orchestration (root check task, feature refresh, train)."""


def render_check_new_data_task(
    task_name: str = "TASK_CHECK_NEW_DATA", stream_name: str = "SRC_STREAM"
) -> str:
    """Render SQL for a root task that triggers when a stream has data.

    The returned SQL creates a task that calls a stored procedure when the
    provided stream contains new rows.
    """
    sql = f"""
CREATE OR REPLACE TASK {task_name}
  WAREHOUSE = TRANSFORM_WH
  WHEN SYSTEM$STREAM_HAS_DATA('{stream_name}')
AS
  CALL TASK_CHECK_NEW_DATA_PROC();
"""
    return sql.strip()


def render_train_task(
    task_name: str = "TASK_TRAIN_MODEL", parent_task: str = "TASK_CHECK_NEW_DATA"
) -> str:
    """Render a training task that runs after a parent task.

    This is a convenience renderer for creating simple orchestration tasks
    used in examples and testing.
    """
    sql = f"""
CREATE OR REPLACE TASK {task_name}
  WAREHOUSE = ML_TRAINING_WH
  AFTER {parent_task}
AS
  CALL TRAIN_MODEL_PROC();
"""
    return sql.strip()
