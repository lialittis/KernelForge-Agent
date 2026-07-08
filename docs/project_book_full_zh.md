# SketchSkill-AKG 项目书完整版

面向昇腾 NPU 的技能驱动算子自动生成与硬件反馈优化系统

| 项目 | 内容 |
| --- | --- |
| 赛题方向 | 基于 AI/Agent 的 NPU 算子自动生成 |
| 团队名称 | 算子炼金术师 |
| 负责人 | 于天池（帕多瓦大学博士生） & 郑遣俊（慕尼黑工业大学博士生） |
| 联系方式 | yu-tianchi@outlook.com |
| 版本 | 完整版 v1.0 |
| 日期 | 2026 年 7 月 |
| 当前代码仓 | KernelForge-Agent / SketchSkill-AKG |
| GitLink PR | 待提交后填写 |
| 提交包入口 | `projects/operator-alchemists/SketchSkill-AKG/README.md` |

## 摘要

SketchSkill-AKG 面向 CCF x MindSpore 昇腾 AKG 赛题，构建一个用于昇腾 910 NPU Benchmark 的技能驱动算子自动生成与硬件反馈优化系统。项目不把目标定义为一次性手写若干优化 kernel，而是构建可复用的自动化流程：从 AKG Bench Lite 任务中抽取 OpSpec，生成 NPU-aware Operator Sketch，检索算子模式 Skill Library，生成 Triton-Ascend 候选 `ModelNew`，在真实昇腾 NPU 上完成编译、运行、正确性验证、Pass@N 统计、性能测量、失败归因、修复和经验写回。

当前原型已经完成官方 Benchmark 固定版本管理、T1 非矩阵乘子集解析、OpSpec/Sketch 生成、候选提交布局、结果导入、Pass@N 汇总、Prompt 模板、Skill Library、确定性 `replay` 生成 provider、OpenAI Responses live provider 适配器，以及多个真实昇腾硬件实验。已有证据包括：`t1/sigmoid_scale_sum` 手动 Pass@4 全部正确，最佳候选达到约 `2.0279x` 加速；同一任务通过 `replay` provider 复现生成链路，最佳候选约 `1.998x`；`t1/fused_silu_and_mul` Pass@4 全部正确，但 Triton 变体性能显著落后，形成了负向性能经验并写入技能库；GELU 调优形成了数值稳定性、UB 压力、Triton-Ascend backend 诊断等可复用经验。

本项目的核心贡献是一个模型无关、后端可扩展、实验可追踪的算子生成系统，而不是单个算子的临时优化。项目书完整版在基础版基础上补充了当前实现进度、技术设计、复现实验、提交包装方案和下一步计划。

## 一、团队介绍

本团队围绕 AI/Agent 驱动的高性能算子自动生成开展工作，目标后端为昇腾 910 NPU。团队成员关注编译系统、AI 芯片架构、IR/Pass 工程、算子调优、自动化验证、大模型辅助软件工程和工程化实验复现，具备开发本项目所需的系统实现能力。

团队前期能力与本项目匹配点包括：

- 理解昇腾 NPU 算子运行链路，包括 CANN、ACL Runtime、算子编译、运行、profiling 和日志分析。
- 关注 Ascend C、Triton、TileLang、MLIR 等算子语言与编译基础设施，理解 tile、访存、并行、同步、流水线和硬件资源约束。
- 具备代码生成 Agent、RAG、Prompt/Skill 设计、自动化测试、错误归因和修复流程的开发经验。
- 希望通过比赛沉淀可复用的 NPU 算子生成技能库和硬件反馈闭环，而不是只完成若干单点算子。

## 二、项目背景与研究现状

大模型结构、AI 加速硬件和编译器栈持续迭代，使高质量算子的需求不断增长。传统算子开发依赖专家手工设计 kernel、tile 策略、数据搬运、边界处理、编译调试和性能分析，开发成本高，并且难以快速覆盖新模型、新 shape 和新硬件后端。

近年来，LLM 代码生成和 Agent 工作流为“自动生成高性能算子”提供了新的路径。KernelBench 将 LLM 写 CUDA/DSL kernel 作为系统评测任务，强调生成代码必须同时正确和高性能。AKG Kernel Agent 则进一步探索多 Agent、跨 DSL、跨硬件后端的 kernel 生成、迁移和调优。

在昇腾 NPU 场景中，直接让 LLM 生成底层 Ascend C 或复杂 kernel 更困难，原因包括：

- Ascend C API 与 host-side tiling 约束强。
- 公开高质量样例比 CUDA 生态少。
- NPU 的 GM/UB 数据搬运、流水线和 backend API 语义与 GPU 不完全一致。
- 性能依赖真实硬件 profiling，CPU 模拟无法替代。
- Benchmark 结果需要正确性与性能共同验证，不能只依赖静态代码质量。

因此，本项目采用结构化中间层和硬件反馈闭环：先将 Benchmark 任务转成 OpSpec 和 NPU-aware Sketch，再由 provider 和 Code Agent 生成候选代码，通过官方 runner 在真实 NPU 上验证和记录，最后把成功和失败经验写回 Skill Library。

## 三、项目定位与核心目标

项目名称为 **SketchSkill-AKG：面向昇腾 NPU 的技能驱动算子自动生成与硬件反馈优化系统**。

最终产品不是单个优化 kernel，也不是一组手写提交文件，而是一个模型无关的 Agent 系统原型，以及可复现的 Benchmark 证据。系统应能在同一套 OpSpec、Sketch、Skill、Prompt、候选元数据、Benchmark runner 和结果导入流程下比较不同 provider，例如确定性 `replay`、OpenAI/Codex 类模型，以及后续本地/open code 模型。

核心目标如下：

- **算子生成目标**：从 Benchmark 描述、参考实现、shape/dtype 约束生成可执行 NPU `ModelNew` 候选，当前主路径为 Triton-Ascend。
- **正确性目标**：在昇腾 NPU 上编译运行候选，与标准实现对比，记录 Pass@1、Pass@4、误差、失败 shape 和错误类型。
- **性能目标**：只对正确性通过的候选测量 latency、speedup 和 score，引入 tile、并行映射、访存和流水线搜索。
- **AI 技术目标**：实现 Spec、Sketch、Skill Retriever、Code、Verify、Repair、Profiler/Search 和 Skill Writer 的可组合工作流，其中 LLM 是可替换 provider。
- **资产沉淀目标**：形成 `SKILL.md`、Prompt 模板、错误修复规则、调优规则、bad-to-good 轨迹和实验复现脚本。
- **提交目标**：形成 GitLink PR 包、项目书完整版、技术设计文档、Benchmark 证据和复现说明。

## 四、总体技术路线

项目路线为：

```text
AKG Bench Lite task / reference Model
-> Benchmark Parser
-> OpSpec
-> NPU-aware Operator Sketch
-> Skill Retriever
-> Provider / Code Agent
-> ModelNew candidate
-> submission materialization
-> backend probe
-> official correctness/performance benchmark
-> Pass@N report
-> failure classification / repair / tuning
-> Skill Library write-back
```

系统坚持两条原则：

1. LLM 负责需要推理和生成的环节，例如 Sketch 规划、代码生成、修复建议和技能总结。
2. Benchmark 扫描、AST 解析、OpSpec schema、正确性比较、score 导入、Pass@N 汇总和最佳候选选择尽量确定化、可测试化，避免把评测结果建立在模型自由输出上。

当前主路径选择 **AKG Bench Lite + Triton-Ascend + 官方 `run_bench.py` runner**。增强路径保留 TileLang-Ascend、Ascend C、CCE/TBE 知识、MLIR/AscendNPU IR 对齐和 CUDA/Triton 经验迁移，但这些路径不阻塞初赛 Step 3 提交。

## 五、当前实现进展

截至 2026 年 7 月 8 日，项目已从基础方案推进到可运行原型，关键实现如下。

### 5.1 Benchmark 与环境复现

- 官方 AKG 仓库以 Git submodule 固定在 `third_party/akg`。
- Benchmark 源为 `akg_agents/benchmark/akg_kernels_bench_lite`。
- 固定分支与提交：`br_agents` / `47aa428fcdc8c68f78d331dc578bc6c74fb9d91d`。
- 提供 `scripts/setup_benchmark_submodule.sh` 初始化 Benchmark。
- 提供 `scripts/bootstrap_ascend_env.sh` 在昇腾 worker 上重建 venv、CANN 环境、Triton-Ascend backend 和诊断流程。
- 已在真实昇腾设备上运行官方 Benchmark，记录硬件和软件版本。

当前已记录的主要硬件环境：

| 项目 | 值 |
| --- | --- |
| NPU | Ascend910B2C |
| CANN | 8.5.1 |
| Python | 3.11.14 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0rc1 |
| Triton-Ascend | 3.2.0 |
| venv | `/data/venvs/kf-triton-ascend` |

### 5.2 Benchmark Registry、OpSpec 与 Sketch

项目已扫描官方 13 个 `akg_kernels_bench_lite` case，并形成 registry：

| Tier | 数量 |
| --- | ---: |
| T1 | 6 |
| T2 | 4 |
| T3 | 3 |

当前自动 OpSpec 支持的 T1 非矩阵乘子集：

- `t1/gelu`
- `t1/fused_silu_and_mul`
- `t1/sigmoid_scale_sum`
- `t1/softmax`

T2/T3 中存在局部变量和符号 shape 构造，当前 registry 能发现但会标记为 `parse_failed` 或 `unsupported`，后续需要扩展 parser。

OpSpec 记录内容包括：

- task id、tier、算子类别、源码路径；
- 输入输出 tensor name、shape、dtype、layout；
- 算子语义、broadcast、reduction axes、normalization axes、layout transform；
- rtol/atol、shape/dtype cases；
- 官方 Benchmark 性能参数；
- 对应 NPU-aware Sketch；
- 提交文件布局与 `ModelNew` entrypoint。

Sketch 字段包括：

- `operator_category`
- `compute_pattern`
- `parallel_axes`
- `tile_plan`
- `memory_plan`
- `pipeline_plan`
- `boundary_mask`
- `accumulation_dtype`
- `backend_target`
- `performance_knobs`
- `known_risks`

### 5.3 Candidate、Submission 与 Result Pipeline

项目已实现：

- 通用提交布局生成：`scripts/create_submission.py` 和 `kernel_forge/submission.py`。
- Benchmark JSON 导入：`scripts/import_benchmark_result.py`。
- Pass@N 汇总：`scripts/summarize_passn.py`。
- backend probe：GELU、`sigmoid_scale_sum`、`fused_silu_and_mul` 均有独立 probe 脚本。
- 实验记录 schema：`experiments/runs/*.yaml`。
- 汇总报告：`experiments/reports/*.yaml` 和 `experiments/reports/gelu_tuning_summary.md`。

### 5.4 Provider 与 Prompt Workflow

项目已实现第一个 provider 边界：

- `ProviderRequest`：包含 case id、candidate index、Pass@N、backend、prompt version、OpSpec、Sketch、retrieved skills 和元数据。
- `ProviderResponse`：包含候选源码文本、provider、model 和 provider metadata。
- `replay` provider：确定性回放已验证候选，用于 CI/regression 和 pipeline 验证。
- `openai` provider：通过 OpenAI Responses API 接入 live LLM，凭证和模型名来自环境变量，测试通过 fake HTTP poster，不依赖真实网络和密钥。

Prompt 模板已存在：

- `prompts/code_agent.v1.md`
- `prompts/repair_agent.v1.md`
- `prompts/skill_writer.v1.md`

生成命令示例：

```bash
python scripts/generate_candidate.py \
  --opspec benchmarks/parsed/t1_sigmoid_scale_sum.yaml \
  --provider replay \
  --backend triton_ascend \
  --pass-n 4 \
  --run-id replay-sigmoid-scale-sum-pass4 \
  --output-root outputs/generated
```

### 5.5 Skill Library

技能库已按算子模式和工程任务组织：

```text
skills/
  elementwise/SKILL.md
  broadcast/SKILL.md
  reduction/SKILL.md
  transpose_layout/SKILL.md
  normalization/SKILL.md
  matmul_like/SKILL.md
  ascend_debug/SKILL.md
  ascend_performance/SKILL.md
  cuda_to_ascend_migration/SKILL.md
  benchmark_evaluation/SKILL.md
```

已沉淀的实验经验包括：

- GELU 数值稳定形式：避免 `0.5 * x * (1 + tanh(u))` 在负尾部取消误差，使用稳定 sigmoid 等价形式。
- Triton-Ascend backend 诊断：区分真实 Triton path 与 PyTorch fallback path。
- UB 压力边界：GELU block size `32768` 出现 UB overflow，`24576` 可编译。
- Rowwise reduction 正向案例：`sigmoid_scale_sum` 在整行 `8192` 归约轴可放入一个 tile 时，一行一个 program、完整归约 tile 速度最好。
- Fused SwiGLU 负向案例：`fused_silu_and_mul` 的 Triton flattened-output 变体可正确但显著慢于框架 intrinsic，应记录为性能负例。

## 六、关键技术设计

完整技术设计见 `docs/technical_design.md`。本节概述项目书中的关键设计。

### 6.1 OpSpec：稳定的算子语义接口

OpSpec 是 Benchmark task 到生成系统的确定性接口。它把原始 Python reference、输入构造和 runner 约束转换为结构化 YAML。这样做的目标是让后续 Agent 不直接解析复杂源文件，而是面对稳定、可测试、可 diff 的语义记录。

示例字段：

```yaml
id: t1/sigmoid_scale_sum
category: reduction
inputs:
  - name: x
    shape: [1000, 8192]
    dtype: float32
  - name: bias
    shape: [8192]
    dtype: float32
outputs:
  - name: output
    shape: [1000, 1]
semantics:
  formula: sum(sigmoid(x * 2.0 + bias), dim=-1, keepdim=True)
validation:
  rtol: 0.01
  atol: 0.01
```

### 6.2 NPU-aware Operator Sketch

Sketch 是项目核心中间表示，用于连接算子语义和后端代码。它不是完整 DSL，而是 JSON/YAML 风格的结构化计划，便于 LLM 生成、检查、修复和检索。

不同算子类别关注不同字段：

| 类别 | 主要关注 |
| --- | --- |
| elementwise | 连续访存、flattened output elements、尾部 mask、数学表达式 backend lowering |
| fused elementwise | 多输入索引、split/layout、输出连续写、重复索引开销 |
| reduction | reduction axis、accumulation dtype、row mapping、partial reduction、keepdim |
| softmax/normalization | max-subtraction、sum reduction、数值稳定性、归一化轴 |
| transpose/layout | 读写连续性、tile 重排、bank conflict 风险 |
| matmul-like | M/N/K tile、数据复用、double buffer、copy/compute overlap |

### 6.3 Skill Retrieval 与 Prompt 约束

Skill Retriever 根据 OpSpec category、Sketch category、broadcast 和 layout transform 选择技能文件。例如 `sigmoid_scale_sum` 会检索：

- `skills/reduction/SKILL.md`
- `skills/broadcast/SKILL.md`
- `skills/ascend_debug/SKILL.md`
- `skills/ascend_performance/SKILL.md`
- `skills/benchmark_evaluation/SKILL.md`

Prompt 会包含 OpSpec YAML、Sketch YAML、候选序号、Pass@N、backend 和技能摘要，要求 provider 返回单个 Python source file，并定义 `ModelNew`。

### 6.4 Provider 边界

Provider 边界把模型调用从工程流程中隔离出来：

- replay provider 用于确定性验证 pipeline。
- openai provider 用于 live LLM 生成，未来可加入本地/open code model provider。
- provider 输出必须写入候选文件、prompt 文件、metadata JSON 和 experiment YAML。
- experiment 记录必须包含 provider、model、prompt version、retrieved skills、candidate index 和 candidate path。

### 6.5 Correctness、Pass@N 与 Performance

正确性是硬门槛。Benchmark 默认：

- `rtol = 1e-2`
- `atol = 1e-2`
- 同时记录 `max_abs_diff` 和 `max_rel_diff`

Pass@N 定义：

- Pass@1：第一个候选通过正确性。
- Pass@4：四个候选中至少一个通过正确性。

性能只在正确性通过后解释，记录：

- baseline latency
- solution latency
- speedup
- weighted score
- backend probe 结果
- 是否为真实 Triton-Ascend path

## 七、当前 Benchmark 证据

### 7.1 GELU 调优案例

GELU 是第一个端到端调试案例。初始候选曾出现 PyTorch fallback、Triton backend 缺失、相对误差失败、UB overflow、不同数学 lowering 性能差异等问题。

最佳已记录真实 Triton-Ascend 候选为 `gelu_triton_v13`：

| Candidate | 正确性 | Speedup | Score | 说明 |
| --- | --- | ---: | ---: | --- |
| `gelu_triton_v7` | pass | `0.0728x` | `4.37` | 第一个纯 Triton 正确性通过 |
| `gelu_triton_v10` | pass | `0.5635x` | `33.81` | block size `16384` |
| `gelu_triton_v11` | compile fail | - | - | UB overflow |
| `gelu_triton_v13` | pass | `0.6059x` | `36.35` | 当前最佳 GELU Triton 候选 |
| `gelu_triton_v18` | pass | `0.5764x` | `34.58` | `exp2` lowering 未超过 v13 |

结论：GELU 已形成有价值的数值和 backend 调试案例，但当前 Triton-Ascend 版本仍慢于 PyTorch baseline，不适合作为下一阶段单算子调优主目标。

### 7.2 `t1/sigmoid_scale_sum` 手动 Pass@4

任务语义：

```text
sum(sigmoid(x * 2.0 + bias), dim=-1, keepdim=True)
shape: x [1000, 8192], bias [8192], output [1000, 1]
```

官方 Benchmark 结果：

| Candidate | Backend 策略 | 正确性 | Speedup | Score |
| --- | --- | --- | ---: | ---: |
| `sigmoid_scale_sum_v1` | torch reference | pass | `1.0006x` | `60.01` |
| `sigmoid_scale_sum_v2` | Triton one row/program, 8192 tile | pass | `2.0279x` | `70.28` |
| `sigmoid_scale_sum_v3` | Triton 2 x 4096 chunks | pass | `1.9367x` | `69.37` |
| `sigmoid_scale_sum_v4` | Triton 4 x 2048 chunks | pass | `1.5785x` | `65.79` |

结论：这是项目第一个非 GELU 正向性能案例，说明 rowwise reduction 在 reduction axis 可完整放入一个 tile 时，Triton-Ascend 自定义 kernel 可以超过框架 baseline。

### 7.3 `t1/sigmoid_scale_sum` Replay Provider Pass@4

Replay provider 使用同一 provider request、prompt、skill retrieval、candidate metadata 和 submission materialization 流程，返回已知候选模板。其作用不是证明模型质量，而是证明生成 pipeline 和实验追踪链路可复现。

| Candidate | 正确性 | Speedup | Score |
| --- | --- | ---: | ---: |
| `sigmoid_scale_sum_replay_v1` | pass | `1.0034x` | `60.03` |
| `sigmoid_scale_sum_replay_v2` | pass | `1.9980x` | `69.98` |
| `sigmoid_scale_sum_replay_v3` | pass | `1.8731x` | `68.73` |
| `sigmoid_scale_sum_replay_v4` | pass | `1.5451x` | `65.45` |

结论：replay provider 已能支撑确定性 CI/regression，并证明 provider 边界、prompt、candidate、submission 和 experiment record 链路可用。

### 7.4 `t1/fused_silu_and_mul` Pass@4

任务语义：

```text
silu(combined[..., :H]) * combined[..., H:]
shape: combined [4096, 8192], output [4096, 4096]
```

官方 Benchmark 结果：

| Candidate | 策略 | 正确性 | Speedup | Score |
| --- | --- | --- | ---: | ---: |
| `fused_silu_and_mul_v1` | torch / intrinsic reference | pass | `1.0027x` | `60.03` |
| `fused_silu_and_mul_v2` | Triton flattened tile | pass | `0.0033x` | `0.20` |
| `fused_silu_and_mul_v3` | Triton reduced UB pressure | pass | `0.0033x` | `0.20` |
| `fused_silu_and_mul_v4` | Triton chunked variant | pass | `0.0033x` | `0.20` |

结论：该任务正确性可达成，但当前 Triton flattened-output 思路性能很差。项目把它作为负向性能案例写入 Skill Library，用于提示后续 Code Agent 不应盲目替代高效 NPU intrinsic。

## 八、创新点

### 8.1 NPU-aware Operator Sketch

项目将算子生成拆分为语义规划和代码实现。Sketch 显式表达算子类别、并行轴、tile plan、memory plan、pipeline plan、boundary mask、accumulation dtype 和 performance knobs，使 LLM 不必直接面对复杂底层 API，从而降低 hallucination、shape 错误和 backend 语义错误。

### 8.2 实验驱动的 Skill Library

Skill Library 不是静态提示词集合，而是由真实 Benchmark 实验驱动更新的工程 playbook。它包含适用条件、shape/dtype 约束、Sketch 模板、tile 策略、访存策略、常见错误、profiling 解释和 bad-to-good 轨迹。

### 8.3 正确性与性能双闭环

系统将正确性和性能分成两个闭环。第一层通过编译日志、运行错误和数值误差修复候选；第二层只对正确性通过的候选做硬件 profiling 和性能搜索。这样避免为了性能牺牲正确性，也避免把 fallback 结果误判为自定义 kernel 成功。

### 8.4 Provider 可替换

项目明确把 LLM 作为 provider，而不是系统本身。`replay` 用于确定性回归，`openai` 用于 live LLM，未来可接入本地模型。所有 provider 共用同一 OpSpec、Sketch、Skill、Prompt、Benchmark 和结果导入流程，使模型比较更公平。

### 8.5 正负样本都写回

项目不仅记录成功加速案例，也记录性能负例。例如 `fused_silu_and_mul` 的 Triton 变体正确但慢，这类信息对后续生成同样重要，可以防止 Agent 重复走低价值方向。

## 九、AI 技术选择

项目采用 Agent/RAG/Prompt/Skill 的组合，不把大规模微调作为初赛主路线。原因是比赛周期短、NPU 领域高质量训练数据有限，且真实硬件验证比离线训练更关键。

当前 AI 技术方案：

- **Agent workflow**：Spec、Sketch、Skill Retriever、Code、Verify、Repair、Profiler/Search、Skill Writer。
- **RAG/Skill retrieval**：从本地 `skills/`、实验记录、Prompt 模板和后续外部文档检索上下文。
- **Prompt engineering**：用 OpSpec YAML、Sketch YAML 和技能摘要约束 Code Agent 输出。
- **Provider abstraction**：同一接口支持 replay、OpenAI、未来本地/open code model。
- **硬件反馈搜索**：基于真实 NPU 正确性和性能结果迭代，不依赖模型主观判断。

## 十、代码与复现说明

### 10.1 初始化 Benchmark

```bash
bash scripts/setup_benchmark_submodule.sh
```

### 10.2 扫描 Benchmark Registry

```bash
python scripts/scan_benchmark_cases.py \
  --output benchmarks/raw/akg_kernels_bench_lite_registry.yaml \
  --repo-root .
```

### 10.3 生成 OpSpec 和 Sketch

```bash
python scripts/extract_opspec_batch.py \
  --output-dir benchmarks/parsed \
  --repo-root .
```

### 10.4 通过 Provider 生成候选

```bash
python scripts/generate_candidate.py \
  --opspec benchmarks/parsed/t1_sigmoid_scale_sum.yaml \
  --provider replay \
  --backend triton_ascend \
  --pass-n 4 \
  --run-id replay-sigmoid-scale-sum-pass4 \
  --output-root outputs/generated
```

### 10.5 生成官方提交布局

```bash
python scripts/create_submission.py \
  --team sigmoid_scale_sum_v2 \
  --candidate sigmoid_scale_sum_v2 \
  --case t1/sigmoid_scale_sum=kernel_forge/candidates/sigmoid_scale_sum_v2.py
```

### 10.6 运行官方 Benchmark

```bash
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \
  outputs/submissions/sigmoid_scale_sum_pass4 \
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \
  --output outputs/results/sigmoid_scale_sum_pass4 \
  --warmup 10 \
  --iterations 100 \
  --num-trials 3
```

### 10.7 汇总 Pass@N

```bash
python scripts/summarize_passn.py \
  --results-dir outputs/results/sigmoid_scale_sum_pass4 \
  --case t1/sigmoid_scale_sum \
  --candidate sigmoid_scale_sum_v1 \
  --candidate sigmoid_scale_sum_v2 \
  --candidate sigmoid_scale_sum_v3 \
  --candidate sigmoid_scale_sum_v4 \
  --output experiments/reports/2026-07-07-sigmoid-scale-sum-pass4.yaml
```

### 10.8 本地测试

```bash
python -m pytest -q tests/test_agent_generation_workflow.py tests/test_fused_silu_and_mul_pass4.py
```

完整测试在初始化 AKG submodule 后运行：

```bash
python -m pytest -q
```

## 十一、提交包装方案

Step 3 需要向 GitLink 竞赛仓提交 PR，并邮件发送更新版项目书。若上游没有更严格布局，建议提交包采用：

```text
projects/<team-name>/SketchSkill-AKG/
  README.md
  PROJECT_README.md
  docs/
  kernel_forge/
  prompts/
  skills/
  scripts/
  experiments/
```

需要包含：

- 面向评审的项目/package `README.md`，以及保留原始开发说明的
  `PROJECT_README.md`。
- `docs/technical_design.md`。
- `docs/competition_alignment.md`。
- `docs/architecture.md`。
- `docs/project_workflow.md`。
- `docs/benchmark_spec.md`。
- `docs/dev_guide.md`。
- `kernel_forge/` 源码。
- `scripts/` 复现脚本。
- `prompts/` Prompt 模板。
- `skills/` Skill Library。
- `experiments/` 中简洁实验记录和报告。
- 本项目书完整版 Markdown 和导出的邮件附件版本。
- `docs/submission/step3_completion_audit.md`，用于列出本地完成项和
  PR/邮件外部动作。

不应包含：

- `outputs/`。
- 虚拟环境。
- raw logs、profiling dumps、编译产物、缓存目录。
- 远程服务器凭证或 API key。

## 十二、实施计划

| 阶段 | 当前状态 | 下一步 |
| --- | --- | --- |
| 环境与 Benchmark 复现 | 已完成基本闭环 | 继续记录不同 Ascend worker 差异 |
| OpSpec 与 Sketch | T1 非矩阵乘 4 个任务已支持 | 扩展 T2/T3 符号 shape parser |
| Candidate 与 Pass@N | 手动和 replay Pass@4 已跑通 | 接入 live provider 生成并验证 |
| Skill Library | 已有正负案例写回 | 增加 prompt-to-skill 自动总结 |
| Repair Agent | 规则和 prompt 初版存在 | 实现自动错误分类和修复循环 |
| Profiler/Search | 手动 tuning 案例存在 | 增加 profile import 与搜索记录 |
| 提交材料 | 基础项目书、PDF、Step 3 plan 已有 | 完成 PR 包、项目书完整版、邮件 |

## 十三、预期成果与评价指标

预期成果：

- SketchSkill-AKG 原型系统。
- Benchmark registry、OpSpec、Sketch、Prompt、Skill 和候选生成 pipeline。
- 官方 Benchmark 正确性和性能报告。
- Pass@1/Pass@4 汇总。
- 至少一个正向性能案例和一个负向性能案例。
- 可复现的远程 Ascend 实验记录。
- GitLink PR、项目书完整版和技术设计文档。

评价指标：

- 编译成功率。
- runtime 成功率。
- correctness pass rate。
- Pass@1 / Pass@4。
- `max_abs_diff` / `max_rel_diff`。
- baseline latency / solution latency。
- speedup / weighted score。
- provider、model、prompt version、retrieved skills 是否可追踪。
- 失败是否能归类并写回技能库。

## 十四、风险分析与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| LLM 直接生成代码不稳定 | 编译失败、API 幻觉 | 使用 OpSpec、Sketch、Skill 和 Prompt 约束；保留 replay regression provider |
| 正确性不足 | Pass@N 不达预期 | 多候选生成、backend probe、错误分类、Repair Agent |
| 性能不及 baseline | score 低 | 正确性优先；只对正确候选做 profiling；记录负例防止重复 |
| Triton-Ascend API 差异 | 跨 worker 不稳定 | 记录 CANN、torch_npu、triton、backend 版本；提供 bootstrap 脚本 |
| NPU 资源有限 | 搜索轮次不足 | 保持搜索空间小，优先 T1 子集和代表性算子 |
| T2/T3 parser 不完整 | 高阶任务覆盖不足 | 先稳定 T1；后续扩展符号 shape 和复杂输入构造 |
| PR/邮件材料不完整 | 影响初赛提交 | 用 Step 3 checklist 管理 PR 包、项目书、邮件和状态记录 |

## 十五、初赛 Step 3 提交计划

当前 Step 3 目标：

1. 完成 `docs/technical_design.md`。
2. 完成项目书完整版 Markdown。
3. 完成 PR/package README 和打包 checklist。
4. 更新项目书导出版本。
5. 向 `https://www.gitlink.org.cn/mindspore-ai/ccf-akg` 提交 PR。
6. 将更新版项目书发送至 `contact@public.mindspore.cn`。
7. 在 `docs/status.md` 记录 PR link、邮件日期和提交状态。

PR 标题建议：

```text
Initial submission: SketchSkill-AKG Ascend NPU skill-driven operator generation system
```

邮件标题建议：

```text
Initial Round + 算子炼金术师: SketchSkill-AKG updated project book
```

## 十六、参考文献与相关开源项目

[1] MindSpore AKG. Auto Kernel Generator (AKG) README.
[2] MindSpore AKG Agents. AKG Agents README_CN.
[3] Jinye Du et al. AKG Kernel Agent: A Multi-Agent Framework for Cross-Platform Kernel Synthesis. arXiv:2512.23424, 2025.
[4] Scaling Intelligence, Stanford. KernelBench: Can LLMs Write GPU Kernels? 2024/2025.
[5] KernelBench authors. KernelBench: Can LLMs Write Efficient GPU Kernels? arXiv:2502.10517, 2025.
[6] Zhongzhen Wen et al. AscendCraft: Automatic Ascend NPU Kernel Generation via DSL-Guided Transcompilation. arXiv:2601.22760, 2026.
[7] AscendOptimizer authors. AscendOptimizer: Episodic Agent for Ascend NPU Operator Optimization. arXiv:2603.23566, 2026.
[8] Tile-AI. TileLang and TileLang-Ascend documentation and repositories.
[9] Triton Project. Triton language and compiler documentation.
[10] LLVM MLIR Project. Linalg Dialect documentation.
[11] Huawei / MindSpore. Ascend C Custom Operator Development and CANN documentation.
[12] KrxGu. kernel-skills: skill library for coding agents to write, optimize and debug compute kernels.
[13] Kevin authors / Cognition. Kevin: Multi-Turn RL for Generating CUDA Kernels. arXiv:2507.11948, 2025.
[14] CUDA-L1 authors. Improving CUDA Optimization via Contrastive Reinforcement Learning. 2025.
