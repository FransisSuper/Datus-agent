# Symphony-SL 接入 Datus

本文给出一套面向真实 `symphony-sl` 服务的正式接入方式，包括：

- Datus 侧正式 `agent.yml`
- 最小治理 subagent 配置
- 将 Claude Code 的接入命令改写为 Datus 的等价方式
- 本机操作步骤与验证方法

---

## 1. Claude Code 命令与 Datus 命令的映射

你当前的 Claude Code 接入命令是：

```bash
npx --offline @anthropic-ai/claude-code mcp add --scope user --transport http symphony-sl https://kong-symphony-dev.bmwbrill.cn/symphony-mcp-sl/mcp \
  --header "symphony-mcp-sl-apikey: ${SYMPHONY_MCP_SL_APIKEY}"
```

它在 Datus 里的等价方式是：

```text
.mcp add --transport http symphony-sl https://kong-symphony-dev.bmwbrill.cn/symphony-mcp-sl/mcp --header "symphony-mcp-sl-apikey: ${SYMPHONY_MCP_SL_APIKEY}" --verify-ssl false --use-env-proxy false
```

建议不要把 key 明文写进仓库或命令历史，先导出环境变量：

```bash
export SYMPHONY_MCP_SL_APIKEY='your-real-key'
```

---

## 2. 正式配置文件

已经补好的正式配置模板在：

- `conf/agent.symphony_sl.yml`
- `conf/mcp.symphony_sl.example.json`

### `agent.symphony_sl.yml`

这个文件里已经包含两类入口：

- `chat`
  - 适合直接 `/chat` 做治理问答
- `data_governance`
  - 一个最小治理 subagent
  - 使用 `node_class: gen_report`
  - 更适合输出结构化解释，而不是 SQL

### 为什么 subagent 用 `gen_report`

治理问题的目标通常是：

- 解释对象定义
- 汇总 owner / tags / 血缘
- 做影响面说明

这类输出更接近分析报告，不适合默认 `gen_sql` 的行为模式。

---

## 3. 本机跑起来的步骤

下面给你一套可直接执行的步骤。

### 步骤 1：准备环境

```bash
cd /Users/fangchao/work/IdeaProjects/github/my-github/Datus-agent
pip install -e .
export OPENAI_API_KEY='your-openai-key'
export SYMPHONY_MCP_SL_APIKEY='your-real-key'
```

### 步骤 2：启动 Datus

```bash
cd /Users/fangchao/work/IdeaProjects/github/my-github/Datus-agent
datus --config conf/agent.symphony_sl.yml --namespace local_duckdb
```

### 步骤 3：在 Datus CLI 中注册真实 MCP

进入 Datus CLI 后执行：

```text
.mcp add --transport http symphony-sl https://kong-symphony-dev.bmwbrill.cn/symphony-mcp-sl/mcp --header "symphony-mcp-sl-apikey: ${SYMPHONY_MCP_SL_APIKEY}" --verify-ssl false --use-env-proxy false
```

连通性检查：

```text
.mcp check symphony-sl
```

如果通过，说明 Datus 已经能连上真实服务。

这里额外带上了：

- `--verify-ssl false`
  - 因为当前服务证书链存在自签名问题
- `--use-env-proxy false`
  - 避免 Python/httpx 走环境代理后再次在代理层握手失败

### 步骤 4：建议增加工具白名单

因为真实服务的工具名你现在没有完整列出，建议先跑一次检查，再把允许的工具收窄到治理相关工具。

你这边当前真实工具名已经验证到是下面这 9 个：

- `sl.table_impact_report`
- `sl.table_lineage_report`
- `sl.column_lineage_report`
- `sl.column_impact_report`
- `sl.search_lineage_tables`
- `sl.search_lineage_columns`
- `sl.search_metadata_tables`
- `sl.search_metadata_columns`
- `sl.list_table_columns`

因此建议直接执行：

```text
.mcp filter set symphony-sl --allowed sl.table_impact_report,sl.table_lineage_report,sl.column_lineage_report,sl.column_impact_report,sl.search_lineage_tables,sl.search_lineage_columns,sl.search_metadata_tables,sl.search_metadata_columns,sl.list_table_columns
```

---

## 4. 如何验证接入效果

### 验证 1：连接是否成功

```text
.mcp check symphony-sl
```

期望结果：

- 显示 `reachable`
- 能看到可用工具数量

如果你还想把真实工具名打印出来，执行：

```bash
python scripts/inspect_datus_mcp.py --config conf/agent.symphony_sl.yml --namespace local_duckdb --server symphony-sl
```

这样可以拿到 `symphony-sl` 的真实工具列表，再决定 `filter allowlist` 要怎么配。

### 推荐的稳定调用路径

- 模糊找表：`sl.search_metadata_tables`
- 模糊找列：`sl.search_metadata_columns`
- 已知表看字段：`sl.list_table_columns`
- 表血缘：`sl.table_lineage_report`
- 表影响面：`sl.table_impact_report`
- 列血缘：`sl.column_lineage_report`
- 列影响面：`sl.column_impact_report`
- 血缘域内模糊找表：`sl.search_lineage_tables`
- 血缘域内模糊找列：`sl.search_lineage_columns`

注意：当前这 9 个真实工具名里没有明显的直接 `owner` 查询工具，所以如果结果里没有明确 owner 字段，建议在 prompt/rules 里要求模型明确写“未返回 owner 信息”，不要猜。

### 验证 2：通过默认 chat 进行问答

```text
/chat 请查询某个数据对象的元数据、owner 和上下游血缘，并按条目总结
```

建议把对象名替换成你们内部真实存在的表、视图或数据集名称。

期望效果：

- Action trace 中出现 `symphony-sl` 的 MCP 工具调用
- 最终回答中明确区分元数据、owner、upstream、downstream

### 验证 3：通过治理 subagent 问答

```text
/data_governance 请分析某个核心表的元数据、owner、上下游血缘，以及它可能影响到的下游对象
```

期望效果：

- 输出更偏报告式、结构化
- 对缺失字段会明确说明“未返回”或“无法确认”

---

## 5. 如果你想预置 `.mcp.json`

除了在 CLI 中执行 `.mcp add`，你也可以直接预置配置文件。

运行时 Datus 会把 MCP 配置读写到：

```text
./.datus-symphony-sl/conf/.mcp.json
```

你可以参考：

- `conf/mcp.symphony_sl.example.json`

把它复制成：

```text
./.datus-symphony-sl/conf/.mcp.json
```

然后再启动 Datus。

不过更推荐先用 `.mcp add`，这样最符合 Datus 当前的使用方式。

---

## 6. 最小治理 subagent 配置说明

`conf/agent.symphony_sl.yml` 中的最小治理 subagent 是：

```yaml
agentic_nodes:
  data_governance:
    node_class: gen_report
    model: governance-model
    system_prompt: gen_report
    tools: ""
    mcp: symphony-sl
```

它的设计原则是：

- 本地工具最小化
- 主要依赖外部治理 MCP
- 输出以解释和总结为主

如果后面你希望它还能结合本地知识库一起回答，可以再加：

```yaml
tools: context_search_tools.*
```

---

## 7. 你在本机应看到的最终效果

接入成功后，通常会看到三层效果：

1. Datus 能成功保存 `symphony-sl` 到 `./.datus-symphony-sl/conf/.mcp.json`
2. `.mcp check symphony-sl` 通过，并识别到真实工具数量
3. `/chat` 或 `/data_governance` 提问时，模型会实际调用 `symphony-sl` 工具，而不是只靠幻觉回答

如果你愿意下一步继续，我可以再给你补一份：

- 针对你们真实 tool 名称的 `.mcp filter set` 白名单
- 一版更严格的治理专用 prompt 规则
- 一份适合提交到仓库的 `README` / 接入说明
