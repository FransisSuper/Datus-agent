
# 在 Datus 中接入外部 MCP 服务

本文说明两件事：

1. 数据血缘、元数据服务接入 Datus 的推荐方案
2. 如何把一个现成的 HTTP MCP 服务接到当前 Datus，并在本机跑通

---

## 1. 最佳接入方案

如果公司内部已经有**权威的元数据/血缘服务**，最佳方案不是把这套能力重新实现到 Datus 里，而是按下面的分层接入：

### 方案建议

- **外部 MCP 服务作为权威实时能力层**
  - 负责对象检索、对象详情、上下游血缘、影响面分析、owner/tag 查询
  - Datus 只负责自然语言理解、工具编排、多轮对话和结果组织

- **Datus 本地知识库作为高频缓存/补充上下文层**
  - 适合沉淀稳定、低频变化的信息，例如语义模型、参考 SQL、业务定义、常见指标说明
  - 不建议把全量血缘图一次性灌入知识库替代在线查询

- **对外暴露为专门的“数据治理问答入口”**
  - 最直接的是把外部 MCP 接到 `chat` 节点
  - 更推荐给治理场景单独做一个 subagent，并限制可用 MCP 工具

### 为什么这是更好的接法

- **职责清晰**：元数据/血缘仍由内部治理平台负责，Datus 只做 AI 交互层
- **实时性更好**：血缘和 owner 这类信息通常变动比语义知识更快，走在线 MCP 更合适
- **风险更低**：不需要改 Datus 的核心 DB tool 或知识库存储结构
- **扩展性更强**：后续新增 `impact_analysis`、`find_owner`、`find_dashboard_dependencies` 等工具时，Datus 基本不用改

### 推荐的 MCP 工具最小集合

第一版建议只暴露只读工具：

- `search_metadata`
- `get_metadata`
- `get_lineage`
- `find_owner` 或 `get_owner`
- `impact_analysis`

不建议第一版就开放写操作，如“改标签”“改 owner”“补充描述”等。

### Datus 侧推荐接法

- 把外部服务配置到 `conf/.mcp.json`
- 在 `agentic_nodes.chat.mcp` 或指定 subagent 的 `mcp` 字段里引用该服务名
- 用 `.mcp filter set` 限制只允许治理相关工具
- 在节点 `rules` 里明确：
  - 元数据问题先查 `get_metadata`
  - 血缘问题先查 `get_lineage`
  - 不确定对象名时先查 `search_metadata`

---

## 2. 你当前内部 Claude Code 命令，如何映射到 Datus

你给的内部接入方式是：

```bash
npx --offline @anthropic-ai/claude-code mcp add --scope user --transport http symphony-sl https://kong-symphony-dev.bmwbrill.cn/symphony-mcp-sl/mcp \
  --header "symphony-mcp-sl-apikey: d819472fa68747f99d6f49764e7f901d"
```

在 Datus 里，等价思路是把这个远端 HTTP MCP 注册到 Datus 的 `.mcp.json`。

### Datus CLI 等价命令

先启动 Datus CLI：

```bash
datus --config conf/agent.mcp_http_demo.yml --namespace local_duckdb
```

然后在 CLI 里执行：

```text
.mcp add --transport http symphony-sl https://kong-symphony-dev.bmwbrill.cn/symphony-mcp-sl/mcp --header "symphony-mcp-sl-apikey: ${SYMPHONY_MCP_SL_APIKEY}"
```

建议把 key 改成环境变量，不要明文写死在配置或命令历史里：

```bash
export SYMPHONY_MCP_SL_APIKEY='your-real-key'
```

添加后，Datus 会把配置保存到：

```text
{agent.home}/conf/.mcp.json
```

本示例里即：

```text
./.datus-demo/conf/.mcp.json
```

### 对应的 `.mcp.json` 结构

大致会是这样：

```json
{
  "mcpServers": {
    "symphony-sl": {
      "type": "http",
      "url": "https://kong-symphony-dev.bmwbrill.cn/symphony-mcp-sl/mcp",
      "headers": {
        "symphony-mcp-sl-apikey": "${SYMPHONY_MCP_SL_APIKEY}"
      },
      "timeout": 10.0
    }
  }
}
```

### 在 Datus 的 `agent.yml` 中启用

如果你希望默认 `chat` 就能调用这个 MCP：

```yaml
agentic_nodes:
  chat:
    model: demo-model
    tools: db_tools.*, context_search_tools.*, date_parsing_tools.*
    mcp: symphony-sl
    rules:
      - For metadata and lineage questions, use MCP tools before answering.
```

如果你不想让所有 chat 都能调它，更推荐单独挂到一个治理 subagent 上。

---

## 3. 本机可运行示例

仓库里已经附带了两个示例文件：

- `scripts/mock_lineage_metadata_mcp.py`
- `conf/agent.mcp_http_demo.yml`

这个 mock 服务模拟了一个内部治理平台，暴露 3 个 MCP 工具：

- `search_metadata`
- `get_metadata`
- `get_lineage`

### 3.1 启动 mock HTTP MCP 服务

在仓库根目录执行：

```bash
python scripts/mock_lineage_metadata_mcp.py
```

默认会启动在：

```text
http://127.0.0.1:18082/mcp
```

如果 `18082` 已被占用，也可以显式指定端口：

```bash
python scripts/mock_lineage_metadata_mcp.py --host 127.0.0.1 --port 18083
```

### 3.2 启动 Datus

新开一个终端，在仓库根目录执行：

```bash
datus --config conf/agent.mcp_http_demo.yml --namespace local_duckdb
```

### 3.3 在 Datus 中注册这个 MCP

进入 Datus CLI 后执行：

```text
.mcp add --transport http demo_lineage_metadata http://127.0.0.1:18082/mcp
```

连通性检查：

```text
.mcp check demo_lineage_metadata
```

建议再加一层工具白名单：

```text
.mcp filter set demo_lineage_metadata --allowed search_metadata,get_metadata,get_lineage
```

### 3.4 手动调用 MCP 工具验证

```text
.mcp call demo_lineage_metadata.search_metadata {"query":"orders"}
```

```text
.mcp call demo_lineage_metadata.get_metadata {"object_name":"sales.orders"}
```

```text
.mcp call demo_lineage_metadata.get_lineage {"object_name":"sales.orders","direction":"both"}
```

### 3.5 通过自然语言调用

如果你的 `OPENAI_API_KEY` 可用，可以直接在 Datus 里问：

```text
/chat 请先查 sales.orders 的元数据，再说明它的上下游血缘和 owner
```

或者：

```text
/chat 帮我找和 orders 相关的数据对象，并告诉我哪个 dashboard 在消费它
```

---

## 4. 推荐的本机操作步骤

下面这套步骤可以直接照着执行。

### 步骤 1：准备环境

```bash
pip install -e .
export OPENAI_API_KEY='your-openai-key'
export NO_PROXY='127.0.0.1,localhost'
export no_proxy='127.0.0.1,localhost'
```

### 步骤 2：启动本地 mock MCP

```bash
python scripts/mock_lineage_metadata_mcp.py
```

### 步骤 3：启动 Datus

```bash
datus --config conf/agent.mcp_http_demo.yml --namespace local_duckdb
```

### 步骤 4：注册 MCP

```text
.mcp add --transport http demo_lineage_metadata http://127.0.0.1:18082/mcp
.mcp check demo_lineage_metadata
.mcp filter set demo_lineage_metadata --allowed search_metadata,get_metadata,get_lineage
```

### 步骤 5：验证工具

```text
.mcp call demo_lineage_metadata.search_metadata {"query":"customer"}
.mcp call demo_lineage_metadata.get_lineage {"object_name":"sales.orders"}
```

### 可选：如果你想换一个本机端口

```bash
python scripts/mock_lineage_metadata_mcp.py --transport http --host 127.0.0.1 --port 18083
```

如果你改成了别的端口，比如 `18083`，启动 Datus 前先设置：

```bash
export DEMO_LINEAGE_MCP_URL=http://127.0.0.1:18083/mcp
```

这样 [`.datus-demo/conf/.mcp.json`](/Users/fangchao/work/IdeaProjects/github/my-github/Datus-agent/.datus-demo/conf/.mcp.json) 会自动读取新的 URL，不需要再手改配置文件。

也可以直接使用启动脚本：

```bash
bash scripts/run_demo_lineage_metadata_http.sh 127.0.0.1 18082
```

### 步骤 6：验证自然语言链路

```text
/chat 查 sales.orders 的字段、owner、上下游血缘
```

---

## 5. 生产接入建议

把本地 demo 换成你们内部服务时，建议这样做：

### Datus 注册命令

```text
.mcp add --transport http symphony-sl https://kong-symphony-dev.bmwbrill.cn/symphony-mcp-sl/mcp --header "symphony-mcp-sl-apikey: ${SYMPHONY_MCP_SL_APIKEY}"
```

### 推荐再加工具过滤

```text
.mcp filter set symphony-sl --allowed search_metadata,get_metadata,get_lineage,impact_analysis
```

### `agent.yml` 示例

```yaml
agentic_nodes:
  chat:
    model: your-model
    tools: db_tools.*, context_search_tools.*, date_parsing_tools.*
    mcp: symphony-sl
    rules:
      - For metadata and lineage questions, call MCP tools first.
      - If object name is ambiguous, call search_metadata before get_metadata.
      - For impact questions, use lineage or impact analysis tools before answering.
```

### 生产注意事项

- 把 API Key 放环境变量，不要写死在仓库里
- 只开放只读工具
- 对工具做 allowlist
- 给 MCP 结果定义稳定字段，如 `object_name`、`owner`、`description`、`upstream`、`downstream`
- 如果返回图结构过大，优先支持 `direction`、`depth`、`limit` 参数
- 高频对象可异步同步一份摘要到 Datus 知识库，降低 token 和网络开销

---

## 6. 什么时候需要更深一层的定制开发

如果你们内部 MCP 的返回值非常原始，例如：

- 只返回图数据库节点 ID
- 字段名不稳定
- 不区分 table/view/dashboard
- 一次返回整个血缘大图

那就建议在内部服务前面再加一层“治理 MCP 适配器”，把能力整理成面向问答的 3 到 5 个工具，再给 Datus 接入。

这样会比直接把底层复杂接口暴露给 LLM 更稳。
