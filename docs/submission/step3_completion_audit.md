# 初赛 Step 3 完成度审计

日期：2026-07-08

本文档用于在打开 GitLink PR 和发送邮件前，核对 `docs/tasks/initial_submission_step3.md` 中的要求是否已有可验证证据。

## 审计结论

本地提交材料已经准备完成，包括中文项目书完整版、技术设计文档、评审入口 README、PR 文案、邮件文案、项目书导出脚本和 GitLink package 打包脚本。根据当前工作约定，Codex 只负责本地准备，不主动打开 GitLink PR，也不发送外部邮件；这两个动作由用户手动执行。

## 交付项状态

| 要求 | 当前状态 | 证据 |
| --- | --- | --- |
| 改进版项目书 | 已完成本地 Markdown 版本 | `docs/project_book_full_zh.md` |
| 技术设计文档 | 已完成 | `docs/technical_design.md` |
| GitLink package README | 已完成，打包时安装为根目录 `README.md` | `docs/submission/package_readme_zh.md` |
| 原项目 README 保留 | 已完成，打包时保留为 `PROJECT_README.md` | `scripts/prepare_gitlink_package.py` |
| PR package 文件树 | 已生成到 `outputs/` | `outputs/gitlink_package/projects/operator-alchemists/SketchSkill-AKG/` |
| PR 标题和正文 | 已完成草稿 | `docs/submission/gitlink_pr_title.txt`、`docs/submission/gitlink_pr_body.md` |
| 邮件正文 | 已完成草稿，PR 链接待填写 | `docs/submission/project_book_email_zh.md` |
| 邮件附件 Markdown | 已导出本地版本 | `outputs/submission/project_book_full_zh.md` |
| GitLink PR | 待用户手动提交 | 需要在 GitLink fork 中创建 PR |
| 项目书邮件 | 待用户在 PR 链接生成后手动发送 | 目标邮箱：`contact@public.mindspore.cn` |
| 状态记录 | 已记录本地准备状态，PR 链接和邮件日期待补 | `docs/status.md` |

## 当前包内容原则

GitLink package 只复制 Git 已跟踪文件和明确列入 Step 3 allowlist 的新文档/脚本。这样可以防止 runtime output、缓存、凭证、日志和无关未跟踪实验文件进入 PR 包。若后续决定把新的实验循环纳入正式提交，应先把对应候选、实验记录、测试和报告作为一个一致的变更集纳入 allowlist 或 Git 跟踪，再重新打包。

## 已验证命令

```bash
python -m py_compile scripts/export_project_book.py scripts/prepare_gitlink_package.py
python scripts/export_project_book.py --output outputs/submission/project_book_full_zh.md
python scripts/prepare_gitlink_package.py --team operator-alchemists --output-root outputs/gitlink_package
python -m pytest -q tests/test_agent_generation_workflow.py tests/test_fused_silu_and_mul_pass4.py
git diff --check
```

包 hygiene 检查：

```bash
find outputs/gitlink_package/projects/operator-alchemists/SketchSkill-AKG \
  \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' \
  -o -name '.ruff_cache' -o -name '.venv' -o -name 'venv' -o -name 'env' \
  -o -name 'outputs' -o -name '*.pyc' -o -name '*.log' -o -name '*.tmp' \
  -o -name '.DS_Store' \) -print
```

预期输出为空。

## 剩余动作

1. 人工检查生成包根目录 `README.md`、`docs/project_book_full_zh.md` 和 `docs/technical_design.md`。
2. 将 `outputs/gitlink_package/projects/operator-alchemists/SketchSkill-AKG/` 放入 GitLink fork。
3. 使用 `docs/submission/gitlink_pr_title.txt` 和 `docs/submission/gitlink_pr_body.md` 打开 PR。
4. PR 创建后运行：

```bash
python scripts/export_project_book.py \
  --pr-link "<GitLink PR URL>" \
  --output outputs/submission/project_book_full_zh.md
```

5. 使用 `docs/submission/project_book_email_zh.md` 发送更新版项目书邮件。
6. 在 `docs/status.md` 记录 PR 链接、邮件日期和提交状态。
