# 水麒麟线上源码快照 · 2026-11-01

此目录是从 Ubuntu 只读取得的应用与学习器源码，包含 44 个运行模块和 `app/lea/` 应用包。源码文件保持采集时字节内容，校验值见 `source-manifest.json`。没有复制模型、LoRA 权重、数据库、原始对话、密钥、VPN/SSH 配置或运行日志。

## 当前职责

- `runtime/unified_auto_agent.py`：统一聊天、记忆、网页和工具开发入口。
- `runtime/continuity_state.py`：基于原始事件与真实回执切窗、按引用读取原文；不另调模型生成交接文档。
- `runtime/helper_evidence.py`、`helper_learning_filter.py`：独立辅助 LoRA 的证据选择、强制保留前提与回执、过滤无效学习。
- `runtime/behavior_targets.py`、`auto_evidence_training.py`：核对实际来源后转换训练样本；不训练未经证实的原因、影响和建议。
- `runtime/auto_memory_worker.py`、`light_lora_pipeline.py`、`auto_memory_train.py`：每 10 次真实切窗触发处理、CPU QLoRA、权重加载与切换。
- `runtime/kv_after_lora.py`：新 LoRA 成功加载后清理旧 KV，失败或跳过训练时不清理。
- `runtime/idle_state_input.py`、`shared_context.py`：独立状态入口与外部聊天共享上下文。
- `runtime/unified_tools.py`、`app/lea/web.py`、`app/lea/selfdev_runner.py`：网页搜索/分页/正文、工具编写/测试/登记及受限执行。

目录中保留部分运行时导入的历史兼容分支；文件存在不表示其策略当前启用。当前配置与测试边界以[公开进度摘要](../../../validation/shuiqilin-progress-20261101.json)为准。

## 不加载模型的程序检查

从仓库根目录运行：

```sh
python -m pip install -e './agent-runtime[test]'
python agent-runtime/deployment/shuiqilin/20261101/run_checks.py
```

检查覆盖证据来源、条件/回执保留、行为样本、自动训练、KV 清理和切窗。测试夹具采用受控开发案例与合成边界案例，不包括私人对话。公开测试中的“无效批次筛选”使用合成夹具；真机批次回放另见验收摘要。

程序检查不调用模型、不联网，不改变线上服务，也不能代替模型能力评估。

## 部署边界

这是可审核的现场快照，**不是通用的一键部署包**。源文件保留既有部署目录、端口及实验布局约定。训练还需要单独准备的官方模型、CPU NF4 依赖、PEFT、适配器转换脚本和已验证的 llama.cpp；它们不在此目录。

在另一台机器部署时，需要配置路径、服务、基础模型和适配器清单，并重新生成实现指纹。不要直接运行带 `install`、`promote` 或服务切换行为的脚本来覆盖现有部署。

已知问题包括：必要上下文仍可能超过工作预算；工具开发的跨批次纠错链尚未完整转成训练样本；新权重目前只有可加载检查，没有能力回归门槛。小说阅读器任务已经实际暴露前两类限制，尚未完成或发布。
