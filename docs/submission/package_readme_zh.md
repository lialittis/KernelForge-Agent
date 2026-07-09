# SketchSkill-AKG GitLink 初赛 Step 3 提交包

团队：算子炼金术师

本文档是 GitLink PR 包根目录的评审入口。项目原始开发 README 在打包时保留为 `PROJECT_README.md`，详细项目书、技术方案和复现说明位于 `docs/`。

## 项目概述

SketchSkill-AKG 是面向昇腾 910 NPU Benchmark 的技能驱动算子自动生成与硬件反馈优化系统。项目目标不是提交单个手写最优算子，而是构建可复用的 AI/Agent 原型：从 AKG Bench Lite 任务中抽取 OpSpec，生成 NPU-aware Operator Sketch，检索 Skill Library，通过 provider 边界生成 `ModelNew` 候选，使用官方 Benchmark 在真实 Ascend 环境中验证正确性和性能，再把有效经验写回技能库。

当前 Step 3 提交重点展示已经实现并可复现的确定性闭环，包括 Benchmark 元数据、13/13 Lite OpSpec/Sketch 覆盖、Skill Library、Prompt 模板、`replay` provider、OpenAI provider 边界、候选生成、官方提交布局生成、Benchmark 结果导入、Pass@N 报告和昇腾实测证据。当前结论是：AKG Bench Lite 的 OpSpec 覆盖已达到 13/13；当前可执行候选与 Pass@4 实验证据已覆盖优先验证子集；完整的实时 AI 生成对比仅受模型/API 配置限制。

## 目录导览

```text
.
├── README.md                         # 当前评审入口
├── PROJECT_README.md                 # 项目原始开发 README
├── docs/
│   ├── project_book_full_zh.md        # 中文项目书完整版
│   ├── technical_design.md            # 技术设计文档
│   ├── submission_package_readme.md   # 提交包和复现说明
│   ├── architecture.md                # 系统架构
│   ├── competition_alignment.md       # 赛题对齐
│   └── tasks/initial_submission_step3.md
├── kernel_forge/                      # OpSpec、provider、候选和工作流代码
├── prompts/                           # Code/Repair/Skill Writer prompt 模板
├── skills/                            # Operator-pattern Skill Library
├── scripts/                           # 扫描、生成、提交、导出和打包脚本
├── benchmarks/                        # Benchmark registry 和解析后的 OpSpec/Sketch
├── experiments/                       # 精简实验记录和 Pass@N 报告
└── tests/                             # 本地确定性测试
```

## 当前实现状态

- 固定 AKG Bench Lite benchmark 子模块和复现脚本。
- 扫描官方 13 个 benchmark case，已完成 13/13 Lite OpSpec/Sketch 解析，并通过 `scripts/validate_opspecs.py` 验证。
- 实现 elementwise、fused elementwise、rowwise reduction、rowwise softmax、normalization、matmul-like、MoE top-k softmax、layout/transpose、convolution 和 decode/attention 类 Sketch 模板。
- 建立按算子模式组织的 Skill Library 和 Code/Repair/Skill Writer prompt 模板。
- 实现确定性 `replay` provider，以及可替换的 OpenAI Responses API provider 边界。
- 实现 candidate 生成、官方提交目录生成、Benchmark 结果导入和 Pass@N 汇总。
- 已记录 GELU、`sigmoid_scale_sum`、`fused_silu_and_mul` 的 Ascend worker 实测证据。

## 复现入口

初始化 Benchmark：

```bash
bash scripts/setup_benchmark_submodule.sh
```

扫描 Benchmark registry：

```bash
python scripts/scan_benchmark_cases.py \
  --output benchmarks/raw/akg_kernels_bench_lite_registry.yaml \
  --repo-root .
```

生成 OpSpec 和 Sketch：

```bash
python scripts/extract_opspec_batch.py \
  --output-dir benchmarks/parsed \
  --repo-root .
```

运行确定性 provider smoke：

```bash
python scripts/generate_candidate.py \
  --opspec benchmarks/parsed/t1_sigmoid_scale_sum.yaml \
  --provider replay \
  --backend triton_ascend \
  --pass-n 1 \
  --run-id replay-provider-smoke \
  --output-root /tmp/kf-generated-smoke
```

运行重点本地测试：

```bash
python -m pytest -q tests/test_agent_generation_workflow.py tests/test_fused_silu_and_mul_pass4.py
```

在 Ascend worker 上运行官方 Benchmark 的命令示例见 `docs/submission_package_readme.md`。

## Benchmark 证据摘要

| 任务 | 批次 | Pass@1 | Pass@4 | 最佳 Speedup | 说明 |
| --- | --- | --- | --- | ---: | --- |
| `t1/sigmoid_scale_sum` | manual | true | true, 4/4 | `2.0279x` | 正向 rowwise reduction 案例 |
| `t1/sigmoid_scale_sum` | replay provider | true | true, 4/4 | `1.9980x` | provider pipeline 复现 |
| `t1/fused_silu_and_mul` | manual | true | true, 4/4 | `1.0027x` | Triton 变体正确但慢，作为性能负例写入技能库 |
| `t2/add_rmsnorm_cast` | manual | true | true, 4/4 | `2.0135x` | 正向 T2 normalization 案例 |
| `t3/layernorm_gated` | manual | true | true, 4/4 | `1.5137x` | 正向 T3 fp16 gated RMSNorm 案例 |
| `t1/gelu` | tuning case study | true | 未作为正式 Pass@4 | `0.6059x` | 数值稳定性和 backend 调试案例 |

## 关键文档

- `docs/project_book_full_zh.md`：提交用中文项目书完整版。
- `docs/technical_design.md`：技术设计、模块边界和算法流程。
- `docs/submission_package_readme.md`：提交包、复现命令、PR/邮件草稿和清单。
- `docs/submission/step3_completion_audit.md`：本地完成度、验证命令和剩余 PR/邮件动作。
- `docs/competition_alignment.md`：与赛题目标、评价方式和交付物的对应关系。
- `experiments/reports/`：Pass@N 和 Benchmark 结果摘要。

## 已知限制

- Lite benchmark 13 个 case 已完成 OpSpec/Sketch 覆盖；更大 Benchmark、动态 shape 和完整 AKG Agents full-mode 对比仍在后续工作中。
- Repair Agent 和 Profiler/Search Agent 已有接口与规则沉淀，但完整自动闭环仍需继续实现。
- 当前包不依赖 live LLM 运行；`openai` provider 已实现边界和单元测试，真实模型对比计划在 Step 3 后执行。
- GELU 和 fused elementwise 的部分 Triton-Ascend 候选正确但慢，已作为负向性能经验写入 Skill Library。
- 本包只包含已跟踪文件和 Step 3 allowlist 文件；新的本地实验循环需要先形成一致的候选、实验记录、测试和报告后再纳入正式 PR。
