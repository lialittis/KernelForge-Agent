# SketchSkill-AKG 技术设计文档

本文档是初赛 Step 3 提交包中的技术设计说明，面向评审解释 SketchSkill-AKG 的系统边界、模块设计、数据结构、Agent/provider 工作流、验证流程和当前实现状态。

## 1. 设计目标

SketchSkill-AKG 的目标是在昇腾 910 NPU Benchmark 上构建一个可复用的 AI/Agent 算子生成系统。系统输入是官方 Benchmark 任务和参考实现，输出是可提交到 AKG Bench Lite 的 `ModelNew` 实现、正确性与性能结果、Pass@N 报告和可复用技能。

系统不把 LLM 作为不可替代的中心，而把 LLM 作为 provider。确定性模块负责 Benchmark 扫描、AST 解析、OpSpec schema、Sketch 模板、提交布局、结果导入、Pass@N 统计和实验记录；provider 负责候选代码生成、修复建议和技能总结等需要生成能力的环节。

## 2. 总体架构

```text
AKG Bench Lite task
  -> Benchmark Parser
  -> OpSpec
  -> NPU-aware Operator Sketch
  -> Skill Retriever
  -> Prompt Renderer
  -> Provider / Code Agent
  -> Candidate Source
  -> Submission Layout
  -> Backend Probe
  -> Official Benchmark Runner
  -> Result Importer
  -> Pass@N Reporter
  -> Repair / Search / Skill Writer
```

当前实现位于：

| 模块 | 路径 |
| --- | --- |
| Benchmark parser / OpSpec / Sketch | `kernel_forge/benchmark/` |
| Provider / Prompt / Skill retrieval workflow | `kernel_forge/agents/` |
| Submission layout helper | `kernel_forge/submission.py` |
| Result import and Pass@N | `kernel_forge/experiments/` |
| Candidate examples | `kernel_forge/candidates/` |
| CLI scripts | `scripts/` |
| Prompt templates | `prompts/` |
| Skill Library | `skills/` |
| Experiment records | `experiments/` |

## 3. Benchmark 与 OpSpec 设计

官方 Benchmark 固定为：

```text
repo: https://atomgit.com/mindspore/akg.git
branch: br_agents
commit: 47aa428fcdc8c68f78d331dc578bc6c74fb9d91d
path: akg_agents/benchmark/akg_kernels_bench_lite
```

项目通过 submodule 和 sparse checkout 管理 Benchmark，避免直接 vendoring 第三方代码。

OpSpec 是 Benchmark task 到 Agent pipeline 的稳定接口。字段包括：

| 字段 | 作用 |
| --- | --- |
| `id`, `name`, `tier`, `category` | 标识任务和算子类别 |
| `source_path` | 官方 reference 源文件 |
| `inputs`, `outputs` | tensor 名称、shape、dtype、layout |
| `semantics` | 公式、broadcast、reduction axes、layout transform |
| `validation` | rtol、atol、shape/dtype cases |
| `performance` | warmup、repeats、baseline source |
| `sketch` | NPU-aware Operator Sketch |
| `submission` | 需要提交的文件和 entrypoint |

当前支持的 OpSpec 子集：

- `t1/gelu`
- `t1/fused_silu_and_mul`
- `t1/sigmoid_scale_sum`
- `t1/softmax`

T2/T3 任务中存在符号 shape、本地变量构造和复杂输入构造，当前被 registry 明确标记为 deferred，而不是静默失败。

## 4. NPU-aware Operator Sketch

Sketch 是本系统的核心中间层。它不是完整 DSL，而是结构化计划，用于降低 LLM 直接生成底层 NPU kernel 的难度。

统一字段：

```yaml
operator_category: reduction
compute_pattern: sigmoid_scale_sum_row_reduction
parallel_axes:
  - outer_rows
tile_plan:
  strategy: rowwise_last_dim_tiling
  shape: [1000, 8192]
  tunable: true
memory_plan:
  input: row_contiguous_global_read
  output: contiguous_global_write
  intermediate: local_vector_reduction
pipeline_plan:
  stages:
    - load_x_and_bias
    - compute_sigmoid
    - row_sum
    - store
boundary_mask:
  required: true
accumulation_dtype: float32
backend_target: triton_ascend
performance_knobs:
  row_tile: null
  reduction_tile: null
known_risks:
  - broadcast_bias_correctness
  - reduction_accuracy
```

Sketch 的作用：

- 将算子语义转为后端可实现的并行和访存计划。
- 为 Prompt 提供稳定上下文。
- 为 Repair Agent 提供可定位的修复对象。
- 为 Search Agent 提供可解释的调优旋钮。
- 为 Skill Library 写回提供结构化挂载点。

## 5. Skill Library 与检索策略

Skill Library 是项目的可复用知识资产。每个 `SKILL.md` 包含适用边界、shape/dtype 约束、Sketch 建议、后端生成注意事项、常见错误、profiling 解释和 bad-to-good 案例。

检索策略当前为规则化选择：

- reduction 算子检索 `skills/reduction/SKILL.md`。
- elementwise / fused elementwise 检索 `skills/elementwise/SKILL.md`。
- 存在 broadcast 时检索 `skills/broadcast/SKILL.md`。
- 存在 layout transform 时检索 `skills/transpose_layout/SKILL.md`。
- 所有生成任务检索 `skills/ascend_debug/SKILL.md`、`skills/ascend_performance/SKILL.md` 和 `skills/benchmark_evaluation/SKILL.md`。

该策略简单但可测试，后续可替换为 embedding/RAG 检索。

## 6. Provider 和 Prompt 设计

Provider 接口定义在 `kernel_forge/agents/provider.py`。

### 6.1 ProviderRequest

`ProviderRequest` 包含：

- case id
- candidate index
- Pass@N
- backend target
- prompt version
- rendered prompt
- OpSpec
- Sketch
- retrieved skill paths
- run metadata

### 6.2 ProviderResponse

`ProviderResponse` 包含：

- generated source text
- provider name
- model name
- response metadata

### 6.3 已实现 provider

| Provider | 作用 |
| --- | --- |
| `replay` | 确定性回放已验证候选，证明 pipeline、metadata、submission 和 experiment flow |
| `openai` | OpenAI Responses API live provider，凭证和模型通过环境变量配置 |

`openai` provider 的测试不访问网络，而是注入 fake HTTP poster，验证请求构造、响应解析、code fence 清理和缺失配置错误。

## 7. Candidate Generation Workflow

`scripts/generate_candidate.py` 调用 `generate_passn_candidates`，流程如下：

1. 读取 OpSpec。
2. 根据 OpSpec 检索技能。
3. 渲染 `prompts/code_agent.v1.md`。
4. 构造 `ProviderRequest`。
5. 调用 provider。
6. 写入候选源码。
7. 写入 prompt 文件和 metadata JSON。
8. 生成 official submission layout。
9. 写入 experiment YAML。

生成 artifacts 默认位于 `outputs/generated/`，不提交到 Git；提交只保留 provider 代码、Prompt、Skill、测试和简洁实验记录。

## 8. 正确性验证设计

官方 runner 对每个 candidate 执行：

- reference `Model` 与 candidate `ModelNew` 输出对比；
- output 类型、数量、shape 检查；
- `max_abs_diff` 和 `max_rel_diff`；
- 默认 `rtol = 1e-2`，`atol = 1e-2`；
- 正确性通过后才测量性能。

项目额外要求：

- backend probe 必须记录 `_last_backend`，区分真实 Triton path 和 fallback path；
- experiment record 必须保存 correctness、performance、pass_n 和 artifact path；
- Pass@N 汇总必须保留所有候选，而不是只保留最佳候选。

## 9. 性能验证与搜索设计

性能指标：

- baseline median latency
- solution median latency
- speedup
- weighted score
- warmup / iterations / num_trials
- backend path

搜索只在正确性通过后进行。初始搜索空间包括：

- tile size
- row/reduction tile
- sequential chunks
- parallel axis
- vector width
- unroll factor
- double buffering
- boundary strategy

当前项目已有手动搜索案例：

- GELU block size 从 `1024` 到 `16384` 持续改善，`32768` 触发 UB overflow，`16384 x 2` chunks 为当前最佳。
- `sigmoid_scale_sum` 一行一个 program、完整 `8192` reduction tile 最优，拆成多个 chunk 反而变慢。
- `fused_silu_and_mul` 的 flattened-output Triton 变体正确但极慢，应优先考虑框架 intrinsic 或不同后端策略。

## 10. Repair Agent 设计

Repair Agent 的目标是根据错误类别路由修复：

| 错误类别 | 路由 |
| --- | --- |
| syntax/API/type | Code Agent |
| shape/broadcast/layout | Sketch Agent + Code Agent |
| tile/memory/boundary | Sketch Agent |
| numerical | Sketch Agent 检查 dtype、mask、reduction、数学形式 |
| environment/backend | 环境 owner，不计为模型失败 |
| performance regression | Profiler/Search Agent |

当前已有 repair prompt 和技能规则，完整自动修复循环仍是后续工作。

## 11. Skill Writer 设计

Skill Writer 从实验记录中抽取可复用经验：

- 失败现象。
- root cause。
- 修复策略。
- 适用 shape/dtype/backend。
- before/after correctness 和 latency。
- candidate path、experiment path 和 report path。

当前写回以人工整理为主，已覆盖 GELU、`sigmoid_scale_sum` 和 `fused_silu_and_mul` 的关键经验。后续将让 Skill Writer 自动生成候选写回内容，再由人审阅。

## 12. 当前实验证据

| 任务 | Pass@1 | Pass@4 | 最佳候选 | Speedup | 说明 |
| --- | --- | --- | --- | ---: | --- |
| `t1/gelu` | 是 | 未按正式 Pass@4 统计 | `gelu_triton_v13` | `0.6059x` | 正确性正例，性能慢于 baseline |
| `t1/sigmoid_scale_sum` manual | 是 | 是，4/4 | `sigmoid_scale_sum_v2` | `2.0279x` | 第一个正向非 GELU 加速案例 |
| `t1/sigmoid_scale_sum` replay | 是 | 是，4/4 | `sigmoid_scale_sum_replay_v2` | `1.9980x` | provider pipeline 复现 |
| `t1/fused_silu_and_mul` | 是 | 是，4/4 | `fused_silu_and_mul_v1` | `1.0027x` | Triton 变体正确但性能负例 |

## 13. 测试与质量控制

当前测试覆盖：

- Benchmark registry 和 OpSpec 提取。
- GELU OpSpec 和 YAML round-trip。
- generic submission layout。
- experiment result import。
- Pass@N summary。
- candidate Python 编译和安全 import 检查。
- provider workflow、OpenAI request/response parsing、replay generation。
- fused SiLU Pass@4 submission/report handling。

本地建议运行：

```bash
python -m pytest -q tests/test_agent_generation_workflow.py tests/test_fused_silu_and_mul_pass4.py
```

初始化 submodule 后运行完整测试：

```bash
bash scripts/setup_benchmark_submodule.sh
python -m pytest -q
```

## 14. 已知限制

- T2/T3 符号 shape parser 尚未完成。
- live `provider=openai` 生成候选尚未形成正式 Benchmark 对比结果。
- Repair Agent 和 Profiler/Search Agent 仍以规则、prompt 和人工流程为主，自动闭环尚未完全实现。
- GELU Triton-Ascend 候选正确但慢于框架 baseline，需要新策略或后端。
- `fused_silu_and_mul` 说明并非所有 elementwise fusion 都适合简单 Triton flattened-output 替代。

## 15. 下一步

初赛 Step 3 优先级：

1. 完成项目书完整版和本技术设计文档。
2. 准备 GitLink PR package README 和文件清单。
3. 导出项目书提交版本。
4. 打开 GitLink PR。
5. 邮件发送更新版项目书。
6. 在 `docs/status.md` 记录 PR link、邮件状态和后续工作。

技术后续：

1. 运行 live `openai` provider Pass@4。
2. 扩展 T2/T3 parser。
3. 自动化 Repair Agent。
4. 增加 profile import 和小规模搜索。
5. 选择代表性算子尝试 TileLang-Ascend 或 Ascend C。
