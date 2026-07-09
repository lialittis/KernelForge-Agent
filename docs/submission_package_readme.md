# SketchSkill-AKG GitLink 初赛 Step 3 提交包说明

本文档用于准备提交到 GitLink 竞赛仓的 PR 包。若上游仓库没有指定更严格目录结构，建议使用以下 team-scoped 布局：

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

## 项目简介

SketchSkill-AKG 是面向昇腾 910 NPU Benchmark 的技能驱动算子自动生成与硬件反馈优化系统。系统以 AKG Bench Lite 为 Benchmark 基础，以 Triton-Ascend 为初始代码生成后端，采用 OpSpec、NPU-aware Operator Sketch、Skill Library、Prompt、provider、官方 Benchmark runner 和实验记录构成可复现闭环。

当前提交包重点展示：

- 面向评审的中文包根目录 `README.md`，原开发 README 保留为
  `PROJECT_README.md`。
- Benchmark registry 和 Lite 13/13 OpSpec/Sketch 解析。
- `scripts/validate_opspecs.py` 确定性 validator，以及 prompt/package/replay
  等 pre-AI infrastructure 测试。
- `replay` provider 和 `openai` provider 边界。
- Prompt 模板和 Skill Library。
- 官方 Ascend Benchmark 实验记录。
- Pass@N 报告和正负性能案例。

## 必含文件

- `README.md`
- `docs/project_book_full_zh.md`
- `docs/technical_design.md`
- `docs/submission/gitlink_pr_title.txt`
- `docs/submission/gitlink_pr_body.md`
- `docs/submission/package_readme_zh.md`
- `docs/submission/project_book_email_zh.md`
- `docs/submission/step3_completion_audit.md`
- `docs/competition_alignment.md`
- `docs/architecture.md`
- `docs/project_workflow.md`
- `docs/benchmark_spec.md`
- `docs/dev_guide.md`
- `docs/tasks/initial_submission_step3.md`
- `kernel_forge/`
- `scripts/`
- `prompts/`
- `skills/`
- `benchmarks/parsed/`
- `benchmarks/raw/akg_kernels_bench_lite_registry.yaml`
- `experiments/runs/*.yaml`
- `experiments/reports/*.yaml`

## 不应包含

- `outputs/`
- `.venv/`、`venv/`、`env/`
- `__pycache__/`
- `.pytest_cache/`
- raw logs
- profiling dumps
- 编译产物
- 下载缓存
- API key、SSH 密码、云资源凭证

## 复现命令

准备 GitLink package：

```bash
python scripts/prepare_gitlink_package.py \
  --team operator-alchemists \
  --output-root outputs/gitlink_package
```

生成目录：

```text
outputs/gitlink_package/projects/operator-alchemists/SketchSkill-AKG/
```

该脚本只复制 Git 已跟踪文件和明确列入 Step 3 的新文档草稿，覆盖源码、文档、Prompt、Skill、Benchmark 元数据、测试和简洁实验记录，并生成 `PACKAGE_MANIFEST.json`。脚本会把 `docs/submission/package_readme_zh.md` 安装为包根目录 `README.md`，并把项目原始 README 保留为 `PROJECT_README.md`。`outputs/`、缓存、日志、虚拟环境、凭证和无关未跟踪文件不会被复制。若后续要把新的本地实验循环纳入正式 PR，应先把对应候选、实验记录、测试和报告作为一致变更集纳入 Git 跟踪或打包 allowlist。

导出邮件附件用 Markdown 项目书：

```bash
python scripts/export_project_book.py \
  --output outputs/submission/project_book_full_zh.md
```

PR 创建后可把链接填入导出版本：

```bash
python scripts/export_project_book.py \
  --pr-link "<GitLink PR URL>" \
  --output outputs/submission/project_book_full_zh.md
```

导出文件默认合并 `docs/project_book_full_zh.md`、`docs/technical_design.md` 和 `docs/submission_package_readme.md`，生成一个独立 Markdown 附件，同时写出 `.metadata.json` 记录来源文件。

初始化 Benchmark：

```bash
bash scripts/setup_benchmark_submodule.sh
```

扫描 Benchmark：

```bash
python scripts/scan_benchmark_cases.py \
  --output benchmarks/raw/akg_kernels_bench_lite_registry.yaml \
  --repo-root .
```

生成 OpSpec：

```bash
python scripts/extract_opspec_batch.py \
  --output-dir benchmarks/parsed \
  --repo-root .
```

运行 provider 生成 smoke：

```bash
python scripts/generate_candidate.py \
  --opspec benchmarks/parsed/t1_sigmoid_scale_sum.yaml \
  --provider replay \
  --backend triton_ascend \
  --pass-n 1 \
  --run-id replay-provider-smoke \
  --output-root /tmp/kf-generated-smoke
```

运行重点单元测试：

```bash
python -m pytest -q tests/test_agent_generation_workflow.py tests/test_fused_silu_and_mul_pass4.py
```

在昇腾 worker 上运行官方 Benchmark 时使用：

```bash
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \
  outputs/submissions/sigmoid_scale_sum_pass4 \
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \
  --output outputs/results/sigmoid_scale_sum_pass4 \
  --warmup 10 \
  --iterations 100 \
  --num-trials 3
```

## 当前 Benchmark 结果摘要

| 任务 | 候选批次 | Pass@1 | Pass@4 | 最佳 Speedup | 说明 |
| --- | --- | --- | --- | ---: | --- |
| `t1/sigmoid_scale_sum` | manual | true | true, 4/4 | `2.0279x` | 正向 rowwise reduction 案例 |
| `t1/sigmoid_scale_sum` | replay provider | true | true, 4/4 | `1.9980x` | provider pipeline 复现 |
| `t1/fused_silu_and_mul` | manual | true | true, 4/4 | `1.0027x` | Triton 变体正确但慢，负向性能案例 |
| `t2/add_rmsnorm_cast` | manual | true | true, 4/4 | `2.0135x` | 正向 T2 normalization 案例 |
| `t3/layernorm_gated` | manual | true | true, 4/4 | `1.5137x` | 正向 T3 fp16 gated RMSNorm 案例 |
| `t1/gelu` | tuning case study | true | 未作为正式 Pass@4 | `0.6059x` | 数值和 backend 调试案例 |

总体提交结论：

```text
Lite benchmark OpSpec coverage: 13/13. Current executable candidates and
Pass@4 evidence cover a priority subset. Full live AI generation remains gated
only by model/API configuration.
```

## PR 文案草稿

可直接使用：

- 标题：`docs/submission/gitlink_pr_title.txt`
- 正文：`docs/submission/gitlink_pr_body.md`

标题：

```text
Initial submission: SketchSkill-AKG Ascend NPU skill-driven operator generation system
```

正文：

```md
## Team

算子炼金术师

## Summary

This PR submits SketchSkill-AKG, a skill-driven AI/Agent prototype for Ascend NPU operator generation and hardware-feedback optimization.

## Current Implementation

- Pinned AKG Bench Lite benchmark submodule.
- Benchmark registry and Lite OpSpec/Sketch coverage for all 13 official cases.
- Deterministic OpSpec/Sketch validation gate.
- Skill Library and prompt templates.
- Replay provider and OpenAI Responses provider boundary.
- Candidate generation, official submission materialization, result import, and Pass@N reporting.
- Ascend benchmark evidence for a priority executable subset.

## Reproduction

See `README.md`, `docs/technical_design.md`, and `docs/submission_package_readme.md`.

## Benchmark Evidence

- `t1/sigmoid_scale_sum` manual Pass@4: 4/4 passed, best speedup 2.0279x.
- `t1/sigmoid_scale_sum` replay provider Pass@4: 4/4 passed, best speedup 1.9980x.
- `t1/fused_silu_and_mul` Pass@4: 4/4 correctness passed, Triton variants recorded as performance-negative lessons.
- `t2/add_rmsnorm_cast` Pass@4: 4/4 passed, best speedup 2.0135x.
- `t3/layernorm_gated` Pass@4: 4/4 passed, best speedup 1.5137x.

## Known Limitations

- Lite benchmark OpSpec/Sketch coverage is complete; larger Benchmark suites,
  dynamic shapes, and AKG Agents full-mode comparison remain future work.
- Full Repair Agent and Profiler/Search Agent automation is still under development.
- Live OpenAI/provider benchmark comparison is planned after model/API configuration exists.
```

## 邮件草稿

可直接使用：`docs/submission/project_book_email_zh.md`

主题：

```text
Initial Round + 算子炼金术师: SketchSkill-AKG updated project book
```

正文：

```text
老师您好，

附件为算子炼金术师团队 SketchSkill-AKG 项目的更新版项目书。项目已完成 AKG Bench Lite 固定版本管理、Lite 13/13 OpSpec/Sketch 覆盖、Skill Library、Prompt、provider 边界、候选生成、官方 Benchmark 结果导入和 Pass@N 报告，并在昇腾环境上记录了 priority subset 的 Pass@4 实验结果。当前完整 live AI 生成对比仅受模型/API 配置阻塞。

GitLink PR 链接：<PR link 待填写>

谢谢。
```

## Step 3 Checklist

- [ ] 确认 `docs/project_book_full_zh.md` 内容完整。
- [ ] 确认 `docs/technical_design.md` 内容完整。
- [ ] 运行 `python scripts/export_project_book.py --output outputs/submission/project_book_full_zh.md`。
- [ ] 运行 `python scripts/prepare_gitlink_package.py --team operator-alchemists`。
- [ ] 创建 GitLink fork/branch。
- [ ] 将 `outputs/gitlink_package/projects/operator-alchemists/SketchSkill-AKG/` 复制到 GitLink fork 的目标目录。
- [ ] 检查提交包不含 `outputs/`、缓存、日志、密钥。
- [ ] 打开 GitLink PR。
- [ ] 使用 `docs/submission/gitlink_pr_title.txt` 和 `docs/submission/gitlink_pr_body.md` 填写 PR。
- [ ] 邮件发送更新项目书。
- [ ] 在 `docs/status.md` 记录 PR link、邮件日期和状态。
