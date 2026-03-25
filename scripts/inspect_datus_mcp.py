#!/usr/bin/env python3
"""
Inspect a registered MCP server from Datus runtime config.

Examples:
    python scripts/inspect_datus_mcp.py --config conf/agent.symphony_sl.yml --namespace local_duckdb --server symphony-sl
    NO_PROXY=127.0.0.1,localhost python scripts/inspect_datus_mcp.py --config conf/agent.mcp_http_demo.yml --namespace local_duckdb --server demo_lineage_metadata
"""

import argparse
import json

from datus.configuration.agent_config_loader import load_agent_config
from datus.tools.mcp_tools.mcp_tool import MCPTool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MCP tools registered in Datus")
    parser.add_argument("--config", required=True, help="Path to Datus agent config")
    parser.add_argument("--namespace", required=True, help="Namespace to load")
    parser.add_argument("--server", required=True, help="Registered MCP server name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_agent_config(config=args.config, namespace=args.namespace)

    tool = MCPTool()
    check = tool.check_connectivity(args.server)
    print("CONNECTIVITY")
    print(json.dumps(check.result or {}, ensure_ascii=False, indent=2, default=str))

    print("\nTOOLS")
    result = tool.list_tools(args.server, apply_filter=False)
    print(json.dumps(result.result or {}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
