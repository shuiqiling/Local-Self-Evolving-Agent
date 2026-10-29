# 索引提炼候选 index-v1

用户要求教会模型从原文提炼记忆索引。规范保存于 `prompts/index_lora_spec.txt`。本任务与查询决策分开训练和验收，使用同一 Qwen3-4B 基底按角色切换适配器；没有增加第二个常驻4B模型，也没有启用未通过验收的 memory-v1。

数据：64条新编原文→索引训练样本、8条开发题、16条冻结测试题。覆盖对象与编号、多个主题、严格数值边界、否定与更正、多对象绑定、计划与实际观测、局部缺失、历史指令和分时状态。teacher输出均经过100字符上限、对象、字面编号/数值和主题覆盖检查。自动检查不能证明完整语义忠实，评测后仍需逐条核对。

初始化为已接受的 concise-v2 适配器，冻结官方 Qwen3-4B-Instruct-2507 基底，用CPU FP32继续训练最后8层的q/v/o/down低秩参数，共1,867,776个参数。64步，每16步保存检查点；编号、数值、否定、更正等目标token加权3倍。原始模型和已有适配器保留。

Ubuntu工作目录：`/home/ubuntu/AI-Native/embodied-agent/training/index-v1`。

- `lea-index-lora-v1.service`：训练。
- `lea-index-lora-v1-pipeline.service`：训练后转换、开发集对照、通过开发门槛后运行冻结测试，保存原始索引输出，等待语义复核。
- `lea-index-lora-eval.service`：临时 :18806 推理服务，对比未加载索引适配器的基底和index-v1，评测结束后停止。

索引提炼提示词保持相同，两个条件均使用100字符、200输出token上限。比较身份保留、编号/数值有无原文依据、主题覆盖、完整性与耗时；不将规则初筛分数称为准确率。没有自动部署步骤。

程序接线：Settings新增可选的 `LEA_INDEXER_ADAPTER_ID` 和 `LEA_INDEXER_ADAPTER_VERSION`。ModelClient按primary/indexer角色显式启用对应适配器，其他角色关闭；默认值不启用索引适配器。角色隔离测试和全套57项本地测试通过。生产服务代码和配置未替换。

查看进度：

```sh
systemctl --user status lea-index-lora-v1.service lea-index-lora-v1-pipeline.service
cat /home/ubuntu/AI-Native/embodied-agent/training/index-v1/training-report.json
cat /home/ubuntu/AI-Native/embodied-agent/training/index-v1/pipeline-status.json
```

本任务不会修复 memory-v1 已发现的具体搜索问题占位语或条件判断错误；这些仍需另外处理。当前上线版本继续为 concise-v2＋hybrid 流程。
