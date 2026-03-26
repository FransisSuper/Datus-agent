# Datus Chat API 前端接入

这个方案的边界建议是：

- 前端只调用 Datus 暴露出来的 HTTP API。
- `symphony-sl` 的 URL、API Key、MCP 过滤规则全部留在 Datus 服务端。
- 前端不要直连内部 MCP 服务，也不要持有 `symphony-mcp-sl-apikey`。

这样前端只关心 `chat` 接口，后端继续负责：

- 与 LLM 对话
- 调用 `symphony-sl` MCP 工具
- 维护多轮会话 `session_id`
- 控制可调用的治理工具白名单

## 接口

新增了两个接口：

- `POST /chat/run`
- `POST /chat/run/stream`

请求体：

```json
{
  "namespace": "local_duckdb",
  "message": "请搜索订单相关表，并分析其中一张表的上下游血缘和影响范围",
  "subagent": "data_governance",
  "session_id": null,
  "catalog_name": null,
  "database_name": null,
  "schema_name": null,
  "include_actions": false
}
```

返回体：

```json
{
  "status": "completed",
  "namespace": "local_duckdb",
  "subagent": "data_governance",
  "session_id": "data_governance_session_xxxxxxxx",
  "response": "......",
  "execution_stats": {
    "tool_calls_count": 2
  },
  "actions": null,
  "error": null
}
```

`session_id` 是前端多轮对话的关键字段。第一轮为空，后续把服务端返回的 `session_id` 原样带回即可。

## 前端调用方式

### 1. 先取 token

```bash
curl -X POST "http://127.0.0.1:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=datus_client&client_secret=datus_secret_key"
```

### 2. 调同步 chat 接口

```bash
curl -X POST "http://127.0.0.1:8000/chat/run" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "local_duckdb",
    "subagent": "data_governance",
    "message": "请搜索订单相关表，并分析其中一张表的元数据、上下游血缘和影响范围"
  }'
```

### 3. 第二轮继续对话

把第一轮返回的 `session_id` 带回来：

```bash
curl -X POST "http://127.0.0.1:8000/chat/run" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "local_duckdb",
    "subagent": "data_governance",
    "session_id": "data_governance_session_xxxxxxxx",
    "message": "继续展开说明它的下游影响对象"
  }'
```

## 浏览器 Demo

本地 demo 页面：

- [datus_chat_api_demo.html](/Users/fangchao/work/IdeaProjects/github/my-github/Datus-agent/docs/integration/datus_chat_api_demo.html)

这个页面默认走：

- `POST /auth/token`
- `POST /chat/run`

也就是最容易在本机跑通的前后端链路。

## 本机运行步骤

### 1. 启动 Datus API

```bash
cd /Users/fangchao/work/IdeaProjects/github/my-github/Datus-agent
pip install -e .
python datus/api/server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --namespace local_duckdb \
  --config conf/agent.symphony_sl.yml
```

### 2. 注册真实 `symphony-sl`

API 服务启动后，需要先在 Datus 的工作目录里准备 `.mcp.json`。如果之前已经通过 CLI 注册过，可以直接复用。

推荐命令：

```bash
cd /Users/fangchao/work/IdeaProjects/github/my-github/Datus-agent
datus --config conf/agent.symphony_sl.yml --namespace local_duckdb
```

进入 CLI 后执行：

```text
.mcp add --transport http symphony-sl https://kong-symphony-dev.bmwbrill.cn/symphony-mcp-sl/mcp --header "symphony-mcp-sl-apikey: d819472fa68747f99d6f49764e7f901d" --verify-ssl false --use-env-proxy false
.mcp filter set symphony-sl --allowed sl.table_impact_report,sl.table_lineage_report,sl.column_lineage_report,sl.column_impact_report,sl.search_lineage_tables,sl.search_lineage_columns,sl.search_metadata_tables,sl.search_metadata_columns,sl.list_table_columns
.mcp check symphony-sl
```

如果 `.mcp check symphony-sl` 成功，API 侧会读取同一份 MCP 配置。

### 3. 启动前端 demo

```bash
cd /Users/fangchao/work/IdeaProjects/github/my-github/Datus-agent/docs/integration
python -m http.server 8088
```

浏览器打开：

```text
http://127.0.0.1:8088/datus_chat_api_demo.html
```

### 4. 在页面里填写

- API Base URL: `http://127.0.0.1:8000`
- Client ID: `datus_client`
- Client Secret: `datus_secret_key`
- Namespace: `local_duckdb`
- Subagent: `data_governance`

然后点：

- `Connect`
- `Send`

## 验证效果

你应该能看到：

- 第一轮调用后，页面拿到一个新的 `session_id`
- 返回内容里会包含 `data_governance` 的治理分析文本
- 第二轮继续追问时，页面会复用同一个 `session_id`
- 如果 MCP 连通，回答会调用 `symphony-sl` 的真实血缘 / 元数据工具

## 一个最小前端 fetch 示例

```javascript
const tokenResp = await fetch("http://127.0.0.1:8000/auth/token", {
  method: "POST",
  headers: { "Content-Type": "application/x-www-form-urlencoded" },
  body: new URLSearchParams({
    grant_type: "client_credentials",
    client_id: "datus_client",
    client_secret: "datus_secret_key"
  })
});
const { access_token } = await tokenResp.json();

const chatResp = await fetch("http://127.0.0.1:8000/chat/run", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${access_token}`,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    namespace: "local_duckdb",
    subagent: "data_governance",
    session_id: window.sessionId || null,
    message: "请分析订单相关表的元数据和上下游血缘"
  })
});

const data = await chatResp.json();
window.sessionId = data.session_id;
console.log(data.response);
```
