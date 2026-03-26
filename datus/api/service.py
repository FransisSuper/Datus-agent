# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import csv
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from io import StringIO
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from datus.agent.agent import Agent
from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode
from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode
from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode
from datus.cli.autocomplete import AtReferenceCompleter
from datus.cli.execution_state import auto_submit_interaction
from datus.configuration.agent_config_loader import load_agent_config
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput
from datus.schemas.ext_knowledge_agentic_node_models import ExtKnowledgeNodeInput
from datus.schemas.gen_report_agentic_node_models import GenReportNodeInput
from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput
from datus.schemas.node_models import SqlTask
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput
from datus.storage.task import TaskStore
from datus.utils.loggings import get_logger

from ..utils.json_utils import to_str
from .auth import auth_service, get_current_client
from .models import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    Mode,
    RunChatRequest,
    RunChatResponse,
    RunWorkflowRequest,
    RunWorkflowResponse,
    TokenResponse,
)

logger = get_logger(__name__)

_form_required = Form(...)
_form_client_id = Form(...)
_form_client_secret = Form(...)
_form_grant_type = Form(...)
_depends_get_current_client = Depends(get_current_client)


class DatusAPIService:
    """Main service class for Datus Agent API."""

    def __init__(self, args: argparse.Namespace):
        self.agents: Dict[str, Agent] = {}
        self.chat_nodes: Dict[str, Any] = {}
        self.at_completers: Dict[str, AtReferenceCompleter] = {}
        self.agent_config = None
        self.args = args
        self.task_store = None

    async def initialize(self):
        """Initialize the service with default configurations."""
        # Load default agent configuration
        self.agent_config = load_agent_config(**vars(self.args))
        logger.info("Agent configuration loaded successfully")

        # Initialize task store
        self.task_store = TaskStore()
        logger.info("Task store initialized successfully")

        # Clean up old tasks on startup
        cleaned_count = self.task_store.cleanup_old_tasks(hours=24)
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old task records on startup")

    def _parse_csv_to_list(self, csv_string: str) -> List[Dict[str, Any]]:
        """Parse CSV string to list of dictionaries."""
        try:
            if not csv_string or not csv_string.strip():
                return []

            reader = csv.DictReader(StringIO(csv_string.strip()))
            return [dict(row) for row in reader]
        except Exception as e:
            logger.warning(f"Failed to parse CSV data: {e}")
            return []

    def _generate_task_id(self, client_id: str) -> str:
        """Generate task ID using client_id and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{client_id}_{timestamp}"

    def get_agent(self, namespace: str) -> Agent:
        """Get or create an agent for the specified namespace."""
        if namespace not in self.agents:
            if not self.agent_config:
                raise HTTPException(status_code=500, detail="Agent configuration not available")

            self.agent_config.current_namespace = namespace
            # Create agent instance
            self.agents[namespace] = Agent(self.args, self.agent_config)
            logger.info(f"Created new agent for namespace: {namespace}")

        return self.agents[namespace]

    def _get_at_completer(self, namespace: str) -> Optional[AtReferenceCompleter]:
        """Get or create an @-context completer for chat requests."""
        if not self.agent_config:
            return None

        if namespace not in self.at_completers:
            self.agent_config.current_namespace = namespace
            try:
                self.at_completers[namespace] = AtReferenceCompleter(self.agent_config)
            except Exception as e:
                logger.warning(f"Failed to initialize AtReferenceCompleter for namespace {namespace}: {e}")
                return None

        return self.at_completers.get(namespace)

    def _parse_chat_context(self, namespace: str, message: str) -> Tuple[List[Any], List[Any], List[Any]]:
        """Parse @table/@metric/@sql context from a chat message."""
        completer = self._get_at_completer(namespace)
        if not completer:
            return [], [], []

        try:
            return completer.parse_at_context(message)
        except Exception as e:
            logger.warning(f"Failed to parse @-context for namespace {namespace}: {e}")
            return [], [], []

    def _create_chat_node(self, namespace: str, subagent_name: Optional[str] = None):
        """Create a chat-capable node using the same routing rules as the CLI."""
        if not self.agent_config:
            raise HTTPException(status_code=500, detail="Agent configuration not available")

        self.agent_config.current_namespace = namespace

        if not subagent_name:
            return ChatAgenticNode(
                node_id="chat_api",
                description="Chat node for API interactions",
                node_type="chat",
                input_data=None,
                agent_config=self.agent_config,
                tools=None,
            )

        node_config = {}
        if hasattr(self.agent_config, "agentic_nodes") and self.agent_config.agentic_nodes:
            node_config = self.agent_config.agentic_nodes.get(subagent_name, {})
            if hasattr(node_config, "model_dump"):
                node_config = node_config.model_dump()

        node_class_type = node_config.get("node_class") if isinstance(node_config, dict) else None

        if subagent_name == "gen_semantic_model":
            return GenSemanticModelAgenticNode(
                agent_config=self.agent_config,
                execution_mode="interactive",
            )
        if subagent_name == "gen_metrics":
            return GenMetricsAgenticNode(
                agent_config=self.agent_config,
                execution_mode="interactive",
            )
        if subagent_name == "gen_sql_summary":
            return SqlSummaryAgenticNode(
                node_name=subagent_name,
                agent_config=self.agent_config,
                execution_mode="interactive",
            )
        if subagent_name == "gen_report" or node_class_type == "gen_report":
            return GenReportAgenticNode(
                node_id=f"{subagent_name}_api",
                description=f"Report generation node for {subagent_name}",
                node_type="gen_report",
                input_data=None,
                agent_config=self.agent_config,
                tools=None,
                node_name=subagent_name,
            )
        if subagent_name == "gen_ext_knowledge":
            return GenExtKnowledgeAgenticNode(
                node_name=subagent_name,
                agent_config=self.agent_config,
                execution_mode="interactive",
            )

        return GenSQLAgenticNode(
            node_id=f"{subagent_name}_api",
            description=f"SQL generation node for {subagent_name}",
            node_type="gensql",
            input_data=None,
            agent_config=self.agent_config,
            tools=None,
            node_name=subagent_name,
        )

    def _create_chat_input(self, request: RunChatRequest, node: Any):
        """Create the correct node input payload for a chat request."""
        at_tables, at_metrics, at_sqls = self._parse_chat_context(request.namespace, request.message)

        if isinstance(node, (GenSemanticModelAgenticNode, GenMetricsAgenticNode)):
            return SemanticNodeInput(
                user_message=request.message,
                catalog=request.catalog_name,
                database=request.database_name,
                db_schema=request.schema_name,
                prompt_version=None,
                prompt_language="en",
            )

        if isinstance(node, SqlSummaryAgenticNode):
            return SqlSummaryNodeInput(
                user_message=request.message,
                catalog=request.catalog_name,
                database=request.database_name,
                db_schema=request.schema_name,
                prompt_version=None,
                prompt_language="en",
            )

        if isinstance(node, GenExtKnowledgeAgenticNode):
            return ExtKnowledgeNodeInput(
                user_message=request.message,
                prompt_version=None,
                prompt_language="en",
            )

        if isinstance(node, GenSQLAgenticNode):
            return GenSQLNodeInput(
                user_message=request.message,
                catalog=request.catalog_name,
                database=request.database_name,
                db_schema=request.schema_name,
                schemas=at_tables,
                metrics=at_metrics,
                reference_sql=at_sqls,
                prompt_version=None,
                prompt_language="en",
                plan_mode=False,
            )

        if isinstance(node, GenReportAgenticNode):
            return GenReportNodeInput(
                user_message=request.message,
                catalog=request.catalog_name,
                database=request.database_name,
                db_schema=request.schema_name,
                prompt_version=None,
            )

        return ChatNodeInput(
            user_message=request.message,
            catalog=request.catalog_name,
            database=request.database_name,
            db_schema=request.schema_name,
            schemas=at_tables,
            metrics=at_metrics,
            reference_sql=at_sqls,
            plan_mode=False,
        )

    def _get_or_create_chat_runtime(self, request: RunChatRequest):
        """Resolve the runtime chat node for a request."""
        if request.session_id and request.session_id in self.chat_nodes:
            node = self.chat_nodes[request.session_id]
            current_name = getattr(node, "get_node_name", lambda: "chat")()
            expected_name = request.subagent or "chat"
            if current_name != expected_name:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Session '{request.session_id}' belongs to node '{current_name}', "
                        f"but request asked for '{expected_name}'"
                    ),
                )
            return node

        node = self._create_chat_node(request.namespace, request.subagent)
        if request.session_id:
            node.session_id = request.session_id
        return node

    async def _execute_chat_actions(
        self, request: RunChatRequest
    ) -> Tuple[List[ActionHistory], Optional[str], Optional[Dict[str, Any]], Any]:
        """Execute a chat request and collect its resulting actions."""
        node = self._get_or_create_chat_runtime(request)
        node.input = self._create_chat_input(request, node)

        action_history_manager = ActionHistoryManager()
        actions: List[ActionHistory] = []

        async for action in node.execute_stream_with_interactions(action_history_manager):
            if action.role == ActionRole.TOOL and action.status == ActionStatus.PROCESSING:
                continue

            if action.role == ActionRole.INTERACTION and action.status == ActionStatus.PROCESSING:
                broker = node.interaction_broker
                if broker:
                    await auto_submit_interaction(broker, action)
                continue

            actions.append(action)

        session_info = await node.get_session_info()
        session_id = session_info.get("session_id")
        if session_id:
            self.chat_nodes[session_id] = node

        return actions, session_id, session_info, node

    def _build_chat_response(
        self,
        request: RunChatRequest,
        actions: List[ActionHistory],
        session_id: Optional[str],
        session_info: Optional[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> RunChatResponse:
        """Convert collected chat actions into the public API response."""
        response_text = ""
        execution_stats = session_info or {}
        resolved_error = error

        if not resolved_error:
            for action in reversed(actions):
                if action.status != ActionStatus.FAILED:
                    continue
                if isinstance(action.output, dict) and action.output.get("error"):
                    resolved_error = action.output.get("error")
                else:
                    resolved_error = action.messages or "Chat execution failed"
                break

        status_text = "completed" if not resolved_error else "error"

        for action in reversed(actions):
            if action.status != ActionStatus.SUCCESS or not isinstance(action.output, dict):
                continue

            if action.output.get("response"):
                response_text = action.output.get("response", "")
                execution_stats = action.output.get("execution_stats") or execution_stats
                break

            if action.output.get("content"):
                response_text = action.output.get("content", "")
                execution_stats = action.output.get("execution_stats") or execution_stats
                break

        serialized_actions = [action.model_dump(mode="json") for action in actions] if request.include_actions else None

        return RunChatResponse(
            status=status_text,
            namespace=request.namespace,
            subagent=request.subagent,
            session_id=session_id,
            response=response_text,
            execution_stats=execution_stats,
            actions=serialized_actions,
            error=resolved_error,
        )

    async def run_chat(self, request: RunChatRequest, client_id: str = None) -> RunChatResponse:
        """Execute a chat request synchronously and return the final answer."""
        del client_id  # Auth is enforced at route level; not otherwise used here.
        try:
            actions, session_id, session_info, _ = await self._execute_chat_actions(request)
            return self._build_chat_response(request, actions, session_id, session_info)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error executing chat request: {e}")
            return self._build_chat_response(
                request=request,
                actions=[],
                session_id=request.session_id,
                session_info=None,
                error=str(e),
            )

    async def run_chat_stream(
        self, request: RunChatRequest, client_id: str = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute a chat request with streaming support and yield actions."""
        del client_id
        try:
            node = self._get_or_create_chat_runtime(request)
            node.input = self._create_chat_input(request, node)

            action_history_manager = ActionHistoryManager()

            async for action in node.execute_stream_with_interactions(action_history_manager):
                if action.role == ActionRole.TOOL and action.status == ActionStatus.PROCESSING:
                    continue

                if action.role == ActionRole.INTERACTION and action.status == ActionStatus.PROCESSING:
                    broker = node.interaction_broker
                    if broker:
                        await auto_submit_interaction(broker, action)
                    continue

                yield action

            session_info = await node.get_session_info()
            session_id = session_info.get("session_id")
            if session_id:
                self.chat_nodes[session_id] = node
                request.session_id = session_id

        except Exception as e:
            logger.error(f"Error executing streaming chat request: {e}")
            yield ActionHistory(
                action_id="chat_error",
                role=ActionRole.ASSISTANT,
                messages=f"Chat execution failed: {str(e)}",
                action_type="error",
                input={"namespace": request.namespace, "subagent": request.subagent},
                status=ActionStatus.FAILED,
                output={"error": str(e)},
            )

    def _create_sql_task(self, request: RunWorkflowRequest, task_id: str, agent: Agent) -> SqlTask:
        """Create SQL task from request parameters."""
        subject_path = request.subject_path
        external_knowledge = request.ext_knowledge or ""

        return SqlTask(
            id=task_id,
            task=request.task,
            catalog_name=request.catalog_name or "",
            database_name=request.database_name or "default",
            schema_name=request.schema_name or "",
            subject_path=subject_path or [],
            external_knowledge=external_knowledge,
            output_dir=agent.global_config.output_dir,
            current_date=request.current_date,
        )

    def _create_response(
        self,
        task_id: str,
        request: RunWorkflowRequest,
        status: str,
        sql_query: str = None,
        query_results: list = None,
        metadata: dict = None,
        error: str = None,
        execution_time: float = None,
    ) -> RunWorkflowResponse:
        """Create standardized workflow response."""
        return RunWorkflowResponse(
            task_id=task_id,
            status=status,
            workflow=request.workflow,
            sql=sql_query,
            result=query_results,
            metadata=metadata,
            error=error,
            execution_time=execution_time,
        )

    async def run_workflow(self, request: RunWorkflowRequest, client_id: str = None) -> RunWorkflowResponse:
        """Execute a workflow synchronously and return results."""
        task_id = request.task_id or self._generate_task_id(client_id or "unknown")
        start_time = time.time()

        try:
            # Initialize task tracking in database
            if self.task_store:
                self.task_store.create_task(task_id, request.task)

            # Get agent for the namespace
            agent = self.get_agent(request.namespace)

            # Create SQL task
            sql_task = self._create_sql_task(request, task_id, agent)

            # Execute workflow synchronously using isolated runner
            runner = agent.create_workflow_runner()
            result = runner.run(sql_task)
            workflow = runner.workflow
            execution_time = time.time() - start_time

            if result and result.get("status") == "completed":
                # Extract SQL and results from the workflow
                sql_query = None
                query_results = None

                # Get the last SQL context from workflow using the correct method
                try:
                    if workflow:
                        last_sql_context = workflow.get_last_sqlcontext()
                        sql_query = last_sql_context.sql_query
                        query_results_raw = last_sql_context.sql_return

                        # Convert CSV string to list of dictionaries for API response
                        if query_results_raw and isinstance(query_results_raw, str):
                            query_results = self._parse_csv_to_list(query_results_raw)
                        else:
                            query_results = None

                        # Update task in database (store as string)
                        if self.task_store:
                            self.task_store.update_task(
                                task_id,
                                sql_query=sql_query,
                                sql_result=str(query_results_raw) if query_results_raw else "",
                            )
                except Exception as e:
                    logger.warning(f"Could not extract SQL context for task {task_id}: {e}")
                    # Continue without SQL data

                # Update task status to completed
                if self.task_store:
                    self.task_store.update_task(task_id, status="completed")

                return self._create_response(
                    task_id=task_id,
                    request=request,
                    status="completed",
                    sql_query=sql_query,
                    query_results=query_results,
                    metadata=result,
                    execution_time=execution_time,
                )
            else:
                # Update task status to failed
                if self.task_store:
                    self.task_store.update_task(task_id, status="failed")

                return self._create_response(
                    task_id=task_id,
                    request=request,
                    status="failed",
                    metadata=result,
                    error="Workflow execution failed",
                    execution_time=execution_time,
                )

        except Exception as e:
            logger.error(f"Error executing workflow {task_id}: {e}")
            # Update task status to failed
            if self.task_store:
                self.task_store.update_task(task_id, status="failed")
            return self._create_response(
                task_id=task_id, request=request, status="error", error=str(e), execution_time=time.time() - start_time
            )

    async def run_workflow_stream(
        self, request: RunWorkflowRequest, client_id: str = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute a workflow with streaming support and yield progress updates."""
        task_id = request.task_id or self._generate_task_id(client_id or "unknown")

        try:
            # Get agent for the namespace
            agent = self.get_agent(request.namespace)

            # Create SQL task
            sql_task = self._create_sql_task(request, task_id, agent)

            # Create action history manager for tracking
            action_history_manager = ActionHistoryManager()

            # Execute workflow with streaming
            async for action in agent.run_stream(sql_task, action_history_manager=action_history_manager):
                yield action

        except Exception as e:
            logger.error(f"Error executing streaming workflow {task_id}: {e}")
            # Yield error action
            error_action = ActionHistory(
                action_id="workflow_error",
                role=ActionRole.WORKFLOW,
                messages=f"Workflow execution failed: {str(e)}",
                action_type="error",
                input={"task_id": task_id},
                status=ActionStatus.FAILED,
                output={"error": str(e)},
            )
            yield error_action

    async def record_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """Record user feedback for a task."""
        try:
            if not self.task_store:
                raise HTTPException(status_code=500, detail="Task store not initialized")

            # Record the feedback by updating the user_feedback field
            recorded_data = self.task_store.record_feedback(task_id=request.task_id, status=request.status.value)

            return FeedbackResponse(
                task_id=recorded_data["task_id"], acknowledged=True, recorded_at=recorded_data["recorded_at"]
            )

        except Exception as e:
            logger.error(f"Error recording feedback for task {request.task_id}: {e}")
            return FeedbackResponse(
                task_id=request.task_id,
                acknowledged=False,
                recorded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

    async def health_check(self) -> HealthResponse:
        """Perform health check on the service."""
        try:
            # Check default agent if available
            database_status = {}
            llm_status = "unknown"

            if self.agent_config:
                # Create a temporary agent for health check using service configuration
                temp_agent = Agent(self.args, self.agent_config)

                # Check database connectivity
                db_check = temp_agent.check_db()
                database_status[self.agent_config.current_namespace] = db_check.get("status", "unknown")

                # Check LLM connectivity
                llm_check = temp_agent.probe_llm()
                llm_status = llm_check.get("status", "unknown")

            return HealthResponse(
                status="healthy", version="1.0.0", database_status=database_status, llm_status=llm_status
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy", version="1.0.0", database_status={"error": str(e)}, llm_status="error"
            )


# Global service instance - will be initialized with command line args
service = None


async def generate_sse_stream(req: RunWorkflowRequest, current_client: str):
    """Generate Server-Sent Events stream for workflow execution."""
    task_id = req.task_id or service._generate_task_id(current_client)
    start_time = time.time()

    try:
        # Initialize task tracking in database
        if service.task_store:
            service.task_store.create_task(task_id, req.task)

        # Send started event
        yield f"event: started\ndata: {to_str({'task_id': task_id, 'client': current_client})}\n\n"

        # Execute workflow with streaming
        sql_query = None

        async for action in service.run_workflow_stream(req, current_client):
            # Map different action types to SSE events
            if action.action_type == "sql_generation" and action.status == "success":
                if action.output and "sql_query" in action.output:
                    sql_query = action.output["sql_query"]
                    # Update task in database
                    if service.task_store:
                        service.task_store.update_task(task_id, sql_query=sql_query)
                    yield f"event: sql_generated\ndata: {to_str({'sql': sql_query})}\n\n"

            elif action.action_type == "sql_execution" and action.status == "success":
                output = action.output or {}
                if output.get("has_results"):
                    sql_result = output.get("sql_result", "")
                    # Update task in database
                    if service.task_store:
                        service.task_store.update_task(task_id, sql_result=str(sql_result))
                    result_data = {"row_count": output.get("row_count", 0), "sql_result": sql_result}
                    yield f"event: execution_complete\ndata: {to_str(result_data)}\n\n"

            elif action.action_type == "output_generation" and action.status == "success":
                output = action.output or {}
                output_data = {
                    "output_generated": output.get("output_generated", True),
                    "sql_query": output.get("sql_query", ""),
                    "sql_result": output.get("sql_result", ""),
                }
                yield f"event: output_ready\ndata: {to_str(output_data)}\n\n"

            elif action.action_type == "workflow_completion":
                logger.info(
                    f"Workflow completion action: {action}, action_type: {action.action_type}, status: {action.status}"
                )
                if action.status == "success":
                    # Update task status to completed
                    if service.task_store:
                        service.task_store.update_task(task_id, status="completed")
                    execution_time_ms = int((time.time() - start_time) * 1000)
                    yield f"event: done\ndata: {json.dumps({'exec_time_ms': execution_time_ms})}\n\n"
                elif action.status == "failed":
                    # Update task status to failed
                    if service.task_store:
                        service.task_store.update_task(task_id, status="failed")
                    error_msg = (action.output or {}).get("error", "Unknown error")
                    yield f"event: error\ndata: {to_str({'error': error_msg})}\n\n"
                # For status="processing", do nothing and wait for final status

            elif action.status == "failed":
                error_msg = (action.output or {}).get("error", "Action failed")
                yield f"event: error\ndata: {to_str({'error': error_msg, 'action_id': action.action_id})}\n\n"

            # Send progress updates for workflow steps and node execution
            elif action.action_id == "workflow_initialization":
                progress_data = {"action": "initialization", "status": action.status, "message": action.messages}
                yield f"event: progress\ndata: {to_str(progress_data)}\n\n"

            elif action.action_id.startswith("node_execution_"):
                node_info = action.input or {}
                node_data = {
                    "action": "node_execution",
                    "status": action.status,
                    "node_type": node_info.get("node_type", ""),
                    "description": node_info.get("description", ""),
                    "message": action.messages,
                }
                yield f"event: node_progress\ndata: {to_str(node_data)}\n\n"

            # Send node-specific progress for streaming operations
            elif action.action_type in [
                "schema_linking",
                "sql_preparation",
                "sql_generation",
                "sql_execution",
                "output_preparation",
                "output_generation",
            ]:
                detail_data = {"action_type": action.action_type, "status": action.status, "message": action.messages}
                yield f"event: node_detail\ndata: {to_str(detail_data)}\n\n"

    except Exception as e:
        logger.error(f"SSE stream error for task {task_id}: {e}")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


async def generate_chat_sse_stream(req: RunChatRequest, current_client: str):
    """Generate Server-Sent Events stream for chat execution."""
    action_buffer: List[ActionHistory] = []

    try:
        yield (
            "event: started\n"
            f"data: {to_str({'namespace': req.namespace, 'subagent': req.subagent, 'session_id': req.session_id})}\n\n"
        )

        async for action in service.run_chat_stream(req, current_client):
            action_buffer.append(action)

            action_payload = {
                "action_id": action.action_id,
                "role": action.role,
                "action_type": action.action_type,
                "status": action.status,
                "messages": action.messages,
                "input": action.input,
                "output": action.output,
            }
            yield f"event: action\ndata: {to_str(action_payload)}\n\n"

        final_response = service._build_chat_response(
            request=req,
            actions=action_buffer,
            session_id=req.session_id,
            session_info=None,
        )
        yield f"event: done\ndata: {to_str(final_response.model_dump(mode='json'))}\n\n"
    except Exception as e:
        logger.error(f"Chat SSE stream error: {e}")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifecycle."""
    global service
    args = getattr(app.state, "agent_args", None)
    service = DatusAPIService(args)

    # Startup
    await service.initialize()
    logger.info("Datus API Service started")
    yield
    # Shutdown
    logger.info("Datus API Service shutting down")


def create_app(agent_args: argparse.Namespace) -> FastAPI:
    """Create FastAPI app with agent args."""
    app = FastAPI(
        title="Datus Agent API",
        description="FastAPI service for Datus Agent workflow execution",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.agent_args = agent_args

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Route handlers with decorators
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with API information."""
        return {"message": "Datus Agent API", "version": "1.0.0", "docs": "/docs", "health": "/health"}

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check() -> HealthResponse:
        """Health check endpoint (no authentication required)."""
        return await service.health_check()

    @app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
    async def authenticate(
        client_id: str = _form_client_id, client_secret: str = _form_client_secret, grant_type: str = _form_grant_type
    ) -> TokenResponse:
        """OAuth2 client credentials token endpoint."""
        if grant_type != "client_credentials":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid grant_type. Must be 'client_credentials'"
            )

        if not auth_service.validate_client_credentials(client_id, client_secret):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid client credentials")

        token_data = auth_service.generate_access_token(client_id)
        return TokenResponse(**token_data)

    @app.post("/workflows/run", tags=["workflows"])
    async def run_workflow(
        req: RunWorkflowRequest, request: Request, current_client: str = _depends_get_current_client
    ):
        """Execute a workflow based on the request parameters."""
        try:
            logger.info(f"Workflow request from client: {current_client}, mode: {req.mode}")

            # Check if client accepts server-sent events for async mode
            if req.mode == Mode.ASYNC:
                accept_header = request.headers.get("accept", "")
                if "text/event-stream" not in accept_header:
                    raise HTTPException(
                        status_code=400, detail="For async mode, Accept header must include 'text/event-stream'"
                    )

                # Return streaming response
                return StreamingResponse(
                    generate_sse_stream(req, current_client),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",  # Disable nginx buffering
                    },
                )
            else:
                # Synchronous mode - original behavior
                return await service.run_workflow(req, current_client)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/workflows/feedback", response_model=FeedbackResponse, tags=["workflows"])
    async def record_feedback(req: FeedbackRequest, current_client: str = _depends_get_current_client):
        """Record user feedback for a task."""
        try:
            logger.info(f"Feedback request from client: {current_client} for task: {req.task_id}")
            return await service.record_feedback(req)
        except Exception as e:
            logger.error(f"Feedback recording error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/chat/run", response_model=RunChatResponse, tags=["chat"])
    async def run_chat(req: RunChatRequest, current_client: str = _depends_get_current_client):
        """Execute a chat request for the default chat node or a configured subagent."""
        try:
            logger.info(
                "Chat request from client=%s namespace=%s subagent=%s session_id=%s",
                current_client,
                req.namespace,
                req.subagent,
                req.session_id,
            )
            return await service.run_chat(req, current_client)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Chat execution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/chat/run/stream", tags=["chat"])
    async def run_chat_stream(req: RunChatRequest, current_client: str = _depends_get_current_client):
        """Execute a chat request and return a Server-Sent Events stream."""
        try:
            logger.info(
                "Streaming chat request from client=%s namespace=%s subagent=%s session_id=%s",
                current_client,
                req.namespace,
                req.subagent,
                req.session_id,
            )
            return StreamingResponse(
                generate_chat_sse_stream(req, current_client),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Streaming chat execution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app
