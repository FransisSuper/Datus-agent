#!/usr/bin/env python3
"""
Minimal MCP server for local lineage and metadata integration demos.

Recommended:
    python scripts/mock_lineage_metadata_mcp.py

Optional stdio mode:
    python scripts/mock_lineage_metadata_mcp.py --transport stdio

HTTP endpoint:
    http://127.0.0.1:18082/mcp
"""

import argparse
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP


METADATA: Dict[str, Dict[str, Any]] = {
    "sales.orders": {
        "object_name": "sales.orders",
        "object_type": "table",
        "database": "analytics",
        "schema": "sales",
        "description": "Order fact table used by sales, finance, and operations dashboards.",
        "owner": "sales-data@demo.local",
        "tags": ["fact", "orders", "gold"],
        "columns": [
            {"name": "order_id", "type": "bigint", "description": "Primary key."},
            {"name": "customer_id", "type": "bigint", "description": "Customer identifier."},
            {"name": "product_id", "type": "bigint", "description": "Product identifier."},
            {"name": "order_date", "type": "date", "description": "Business order date."},
            {"name": "amount", "type": "decimal(18,2)", "description": "Order amount."},
        ],
    },
    "sales.order_items": {
        "object_name": "sales.order_items",
        "object_type": "table",
        "database": "analytics",
        "schema": "sales",
        "description": "Item-level detail rows for each order.",
        "owner": "sales-data@demo.local",
        "tags": ["detail", "orders", "silver"],
        "columns": [
            {"name": "order_id", "type": "bigint", "description": "Order foreign key."},
            {"name": "sku_id", "type": "bigint", "description": "SKU identifier."},
            {"name": "quantity", "type": "int", "description": "Purchased quantity."},
            {"name": "item_amount", "type": "decimal(18,2)", "description": "Item amount."},
        ],
    },
    "sales.customers": {
        "object_name": "sales.customers",
        "object_type": "table",
        "database": "analytics",
        "schema": "sales",
        "description": "Customer dimension table.",
        "owner": "customer-platform@demo.local",
        "tags": ["dimension", "customer", "gold"],
        "columns": [
            {"name": "customer_id", "type": "bigint", "description": "Customer primary key."},
            {"name": "customer_name", "type": "varchar", "description": "Display name."},
            {"name": "customer_tier", "type": "varchar", "description": "Tier segment."},
        ],
    },
    "bi.sales_order_dashboard": {
        "object_name": "bi.sales_order_dashboard",
        "object_type": "dashboard",
        "database": "analytics",
        "schema": "bi",
        "description": "Executive dashboard for order volume, GMV, and customer mix.",
        "owner": "bi-team@demo.local",
        "tags": ["dashboard", "orders"],
        "upstream_objects": ["sales.orders", "sales.customers"],
    },
}


LINEAGE: Dict[str, Dict[str, List[Dict[str, str]]]] = {
    "sales.orders": {
        "upstream": [
            {"object_name": "ods.raw_orders", "relation": "transform"},
            {"object_name": "sales.order_items", "relation": "aggregate"},
            {"object_name": "sales.customers", "relation": "lookup"},
        ],
        "downstream": [
            {"object_name": "bi.sales_order_dashboard", "relation": "consumed_by"},
            {"object_name": "mart.revenue_daily", "relation": "feeds"},
        ],
    },
    "sales.order_items": {
        "upstream": [{"object_name": "ods.raw_order_items", "relation": "transform"}],
        "downstream": [{"object_name": "sales.orders", "relation": "aggregate_into"}],
    },
    "sales.customers": {
        "upstream": [{"object_name": "crm.customer_master", "relation": "sync"}],
        "downstream": [
            {"object_name": "sales.orders", "relation": "lookup"},
            {"object_name": "bi.sales_order_dashboard", "relation": "consumed_by"},
        ],
    },
}


mcp = FastMCP("demo-lineage-metadata")


def _normalize(text: str) -> str:
    return text.strip().lower()


@mcp.tool()
def search_metadata(query: str, object_type: str = "all", limit: int = 10) -> List[Dict[str, Any]]:
    """Search metadata objects by name, description, owner, or tags."""
    needle = _normalize(query)
    object_type = _normalize(object_type)
    results: List[Dict[str, Any]] = []

    for item in METADATA.values():
        if object_type != "all" and item.get("object_type", "").lower() != object_type:
            continue

        haystack = " ".join(
            [
                item.get("object_name", ""),
                item.get("description", ""),
                item.get("owner", ""),
                " ".join(item.get("tags", [])),
            ]
        ).lower()
        if needle in haystack:
            results.append(
                {
                    "object_name": item["object_name"],
                    "object_type": item["object_type"],
                    "description": item["description"],
                    "owner": item["owner"],
                    "tags": item.get("tags", []),
                }
            )
        if len(results) >= limit:
            break

    return results


@mcp.tool()
def get_metadata(object_name: str) -> Dict[str, Any]:
    """Get detailed metadata for a table, view, or dashboard."""
    key = _normalize(object_name)
    for name, payload in METADATA.items():
        if _normalize(name) == key:
            return payload
    return {"error": f"Object '{object_name}' not found"}


@mcp.tool()
def get_lineage(object_name: str, direction: str = "both", depth: int = 1) -> Dict[str, Any]:
    """Get simplified lineage for an object."""
    direction = _normalize(direction)
    if direction not in {"upstream", "downstream", "both"}:
        return {"error": "direction must be one of: upstream, downstream, both"}

    data = LINEAGE.get(object_name, {"upstream": [], "downstream": []})
    payload: Dict[str, Any] = {"object_name": object_name, "depth": depth}
    if direction in {"upstream", "both"}:
        payload["upstream"] = data.get("upstream", [])
    if direction in {"downstream", "both"}:
        payload["downstream"] = data.get("downstream", [])
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock lineage/metadata MCP server")
    parser.add_argument(
        "--transport",
        default="http",
        choices=["stdio", "http", "streamable-http"],
        help="Transport to use. Default is HTTP for local Datus demos.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=18082, help="Port to bind")
    parser.add_argument("--mount-path", default="/mcp", help="Mount path for streamable HTTP transport")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transport = "streamable-http" if args.transport in {"http", "streamable-http"} else "stdio"

    if transport == "streamable-http":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport=transport, mount_path=args.mount_path)
        return

    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
