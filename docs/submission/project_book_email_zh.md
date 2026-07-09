# 邮件标题

```text
Initial Round + 算子炼金术师: SketchSkill-AKG updated project book
```

# 邮件正文

```text
老师您好，

附件为算子炼金术师团队 SketchSkill-AKG 项目的更新版项目书。

本项目面向基于 AI/Agent 的 NPU 算子自动生成赛题，已从基础方案推进到可运行原型，当前实现包括：

1. AKG Bench Lite 固定版本管理和复现脚本；
2. Lite benchmark 13/13 OpSpec / NPU-aware Sketch 覆盖和 validator；
3. Skill Library、Prompt 模板和 provider 边界；
4. replay provider 与 OpenAI Responses provider 适配器；
5. candidate 生成、官方提交布局生成、Benchmark 结果导入和 Pass@N 报告；
6. 昇腾环境上的 priority subset Pass@4 实验记录。

当前提交结论为：AKG Bench Lite 的 OpSpec 覆盖已达到 13/13；当前可执行候选与 Pass@4 实验证据已覆盖优先验证子集；完整的实时 AI 生成对比仅受模型/API 配置限制。

当前 Benchmark 证据包括：

- t1/sigmoid_scale_sum 手动 Pass@4：4/4 通过，最佳 speedup 2.0279x；
- t1/sigmoid_scale_sum replay provider Pass@4：4/4 通过，最佳 speedup 1.9980x；
- t1/fused_silu_and_mul Pass@4：4/4 正确性通过，Triton 变体作为性能负例写入 Skill Library；
- t2/add_rmsnorm_cast Pass@4：4/4 通过，最佳 speedup 2.0135x；
- t3/layernorm_gated Pass@4：4/4 通过，最佳 speedup 1.5137x；
- GELU 调优案例：gelu_triton_v13 正确性通过，speedup 0.6059x，形成数值稳定性和 backend 调试经验。

GitLink PR 链接：<PR link 待填写>

附件：
- SketchSkill-AKG 项目书完整版

谢谢。

算子炼金术师团队
于天池、郑遣俊
```
