# 记忆系统 LoRA：memory-v1

用户已审核记忆规范并明确要求开始训练。批准内容保存为 `prompts/memory_lora_review.txt`，SHA256：`cd0314c08efb4f8c2101b8c489d87aed203bfbb4039eb857bd368940b42757ca`。

Ubuntu 目录：`/home/ubuntu/AI-Native/embodied-agent/training/memory-v1`。

用户指定成功后的名称为“水麒麟”。命名意图保存在 Ubuntu `identity.json` 和本地 WORK_STATUS.json；内部版本号仍为 memory-v1，便于追踪和回退。

本次从当前 `concise-v2` PEFT 适配器继续训练，生成一个独立版本。基底仍为官方 Qwen3-4B-Instruct-2507，CPU FP32 计算，冻结基底，最后八层 q/v/o/down，rank 8，1,867,776 个可训练参数。不合并、不覆盖原适配器。

训练数据为新编合成数据，不读取用户会话或此前评测答案：96 条训练样本（61 直接回答、32 读档、3 检索索引）、12 条开发题、26 条冻结测试题。各类别的训练样本一半使用实际 PRIMARY_SCOPED 提示词，一半使用简短的记忆规则；开发和测试均使用实际提示词。训练一遍全部 96 条，完整 assistant 决策损失，动作 token 权重 4，索引选择 token 权重 2；不是之前只侧重答案正文的训练。

启动时已校验数据哈希，测试集设为只读。训练只打开 train.jsonl，每 16 步保存适配器、优化器和 RNG 状态。以 8 CPU 线程、峰值学习率 0.00025 运行，进程内存上限 36G、禁止该服务使用 swap；原模型服务保持运行，训练期间会占用 CPU。

## 后台服务

- `lea-memory-lora-v1.service`：训练。
- `lea-memory-lora-v1-pipeline.service`：等待训练结束，转换 GGUF，启动独立 :18805 对照服务，依次进行开发集、冻结测试集及真实 Runtime 小规模对照；开发集动作/必要读档/索引选择退步时先停止，不消耗冻结测试集。
- `lea-memory-lora-eval.service`：临时的双适配器推理服务，评测管线退出时停止。

没有修改 :18800 线上 Agent、:18801 原推理服务、:18803 Embedding 服务，也不会自动启用候选。评测原始输出仍需按来源人工复核，关键词初筛不能当成准确率。

## 查看状态

```sh
systemctl --user status lea-memory-lora-v1.service lea-memory-lora-v1-pipeline.service
tail -n 10 /home/ubuntu/AI-Native/embodied-agent/training/memory-v1/training.log
cat /home/ubuntu/AI-Native/embodied-agent/training/memory-v1/pipeline-status.json
```

关键文件：`training-report.json`、`latest-checkpoint.json`、`pipeline-status.json`、`pipeline.log`、`evaluation-dev/report.json`、`evaluation-test/report.json`、`runtime-evaluation/report.json`。

数据清单保存在 `data/manifest.json`；训练集 SHA256 为 `b9a08afefec4ea90ea565fb67c762958c866b5467c9526d1d1eab44df042363a`。初始 concise-v2 PEFT 权重 SHA256 为 `48e105c01a479b393e690d88beb2bcb56b7c19cd7500a7635a45dfd44a13814e`。

若训练中断，先检查服务、错误和最近检查点，再使用 `train.py --root ... --resume` 恢复；不覆盖已完成的 adapter。不要重建或修改已经冻结的数据来匹配评测结果。
