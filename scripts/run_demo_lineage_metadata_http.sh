#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-127.0.0.1}"
PORT="${2:-18082}"

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port ${PORT} is already in use. Choose another port, for example:"
  echo "  bash scripts/run_demo_lineage_metadata_http.sh ${HOST} 18082"
  echo "  bash scripts/run_demo_lineage_metadata_http.sh ${HOST} 18083"
  exit 1
fi

echo "Starting demo lineage metadata MCP over HTTP on ${HOST}:${PORT}"
echo "MCP endpoint: http://${HOST}:${PORT}/mcp"
echo
echo "In another terminal:"
echo "  export DEMO_LINEAGE_MCP_URL=http://${HOST}:${PORT}/mcp"
echo "  datus --config conf/agent.mcp_http_demo.yml --namespace local_duckdb"
echo
echo "Inside Datus CLI:"
echo "  .mcp check demo_lineage_metadata"
echo "  .mcp call demo_lineage_metadata.search_metadata {\"query\":\"orders\"}"
echo

exec python scripts/mock_lineage_metadata_mcp.py --transport http --host "${HOST}" --port "${PORT}"
