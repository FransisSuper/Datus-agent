# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
API models for the Datus Agent FastAPI service.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Mode(str, Enum):
    """Execution mode enum."""

    SYNC = "sync"
    ASYNC = "async"


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="Service version")
    database_status: Dict[str, str] = Field(..., description="Database connection status")
    llm_status: str = Field(..., description="LLM service status")


class RunWorkflowRequest(BaseModel):
    """Request model for workflow execution."""

    workflow: str = Field(..., description="Workflow name, e.g., nl2sql")
    namespace: str = Field(..., description="Database namespace")
    task: str = Field(..., description="Natural language task description")
    mode: Mode = Field(Mode.SYNC, description="Execution mode: sync or async")
    task_id: Optional[str] = Field(None, description="Custom task ID for idempotency")
    catalog_name: Optional[str] = Field(None, description="Catalog name")
    database_name: Optional[str] = Field(None, description="Database name")
    schema_name: Optional[str] = Field(None, description="Schema name")
    current_date: Optional[str] = Field(None, description="Current date reference for relative time expressions")
    subject_path: Optional[List[str]] = Field(None, description="Subject path for the task")
    ext_knowledge: Optional[str] = Field(None, description="External knowledge for the task")


class RunWorkflowResponse(BaseModel):
    """Response model for workflow execution."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Workflow execution status")
    workflow: str = Field(..., description="Workflow name")
    sql: Optional[str] = Field(None, description="Generated SQL query")
    result: Optional[List[Dict[str, Any]]] = Field(None, description="Workflow execution results")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if any")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")


class RunChatRequest(BaseModel):
    """Request model for conversational chat execution."""

    namespace: str = Field(..., description="Target namespace")
    message: str = Field(..., description="User message")
    subagent: Optional[str] = Field(
        default=None,
        description="Optional subagent name, e.g. data_governance",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Conversation session identifier for multi-turn chat",
    )
    catalog_name: Optional[str] = Field(None, description="Catalog name")
    database_name: Optional[str] = Field(None, description="Database name")
    schema_name: Optional[str] = Field(None, description="Schema name")
    include_actions: bool = Field(
        default=False,
        description="Whether to include full action history in the response",
    )


class RunChatResponse(BaseModel):
    """Response model for conversational chat execution."""

    status: str = Field(..., description="Execution status")
    namespace: str = Field(..., description="Target namespace")
    subagent: Optional[str] = Field(default=None, description="Resolved subagent name")
    session_id: Optional[str] = Field(default=None, description="Conversation session identifier")
    response: str = Field(default="", description="Assistant response")
    execution_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Execution statistics returned by the node",
    )
    actions: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional action history for debugging",
    )
    error: Optional[str] = Field(default=None, description="Error message if any")


class TokenResponse(BaseModel):
    """Token response model."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(..., description="Token type, always 'Bearer'")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class FeedbackStatus(str, Enum):
    """Feedback status enum."""

    SUCCESS = "success"
    FAILED = "failed"


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""

    task_id: str = Field(..., description="Target task ID")
    status: FeedbackStatus = Field(..., description="Task execution status feedback")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""

    task_id: str = Field(..., description="Task ID that feedback was recorded for")
    acknowledged: bool = Field(..., description="Whether feedback was successfully recorded")
    recorded_at: str = Field(..., description="ISO timestamp when feedback was recorded")
