# 水麒麟线上源码快照 · 2026-11-02

此目录保存当前 Ubuntu 应用与自动学习器的 78 个源码文件，逐文件 SHA-256 见 [source-manifest.json](source-manifest.json)。采集时应用与学习器中同名副本已核对一致；源码字节未改写。已排除历史部署探针和未启用的实验版本。模型、LoRA 权重、运行数据库、私人对话、连接凭据及原始日志不在发布内容中。

## 当前实现

- `runtime/active_memory.py`：主动查询记忆，每批三条索引，支持下一批与按来源读取细节。
- `runtime/navigation_runtime.py`：辅助 LoRA 选择原文索引；原文完整存档。
- `runtime/active_context.py`、`state_view.py`：精简可选历史，保留当前要求、约束和真实回执；区分宿主观测与任务设备。
- `runtime/idle_sessions.py`：一次空闲活动连续决策，默认 300 秒预算；用户任务和训练优先，连续两步无进展停止。
- `runtime/idle_development.py`、`unified_tools.py`：编写提交、当前版本测试、本地发布、实际使用；新工具仅处理纯 Python JSON 输入输出。
- `runtime/execution_recovery.py`、`receipt_reporting.py`、`write_receipts.py`：有进展才续接，单次操作约束、真实写后读回及有界预算恢复。
- `runtime/continuity_state.py`：原始事件检查点，不另调模型写交接文档；按引用找回完整原文。
- `runtime/auto_memory_worker.py`、`auto_evidence_training.py`：每十次真实切窗处理一批，来源校验后进行轻量 CPU QLoRA；无合格样本跳过。

当前主模型为 Qwen3-30B-A3B Q4_K_M，主／辅助 LoRA 分别管理，共用基础权重。公开配置摘要及真机结果见[进度报告](../../../validation/shuiqilin-progress-20261102.json)。

## 可复核检查

从仓库根目录执行：

```sh
python -m pip install -e './agent-runtime[test]'
python agent-runtime/deployment/shuiqilin/20261102/run_checks.py
```

发布前结果为 **162 项通过、1 项明确的已知失败（strict xfail）**，不是 163 项通过。详见 [program-checks.json](program-checks.json)。检查不加载模型、不联网、不修改线上服务。Windows 上开发执行器的单元测试使用可信夹具执行；Ubuntu 的真实模型验收及前述 54 项检查使用实际受限子进程，二者不混为一谈。

旧切窗测试已按主动检索契约迁移：历史原文完整保存但不要求全部自动注入，当前请求不重复塞入摘要，状态采样可替换旧视图，三个只读记忆接口允许用于只读回答。保持旧断言的 2026-11-01 快照仍保留。

迁移过程中发现一个实际缺陷：用户说“第二种”时，当前指代匹配器可能未带回对应的上一条答复。该用例以严格 xfail 留在新测试中，未改线上源码或把失败当成通过。Python 3.13 的部分检查还出现 SQLite 连接关闭警告，已在检查报告中注明。

## 真实模型结果与边界

30B 在“反复处理标签、减少后续劳动”的目标下，自行提交 `process_labels`，自测 3/3、预先冻结的独立用例 4/4，通过后发布并实际调用。首次成功使用约 389.4 秒，跨两段空闲活动；之后又申请相同调用两次，被去重并按无进展停止，整轮约 562.1 秒。系统策略有工具复用提示，因此这不是无提示自主规划成功率，也没有证明开发成本已被省下的时间抵消。

第一轮输出格式要求含糊：算法输出正确，但对象包装与验收数组格式不符。模型只修改了自测，没有修复格式，发布被阻止。该失败保留在进度摘要中。

小说阅读器已登记并完成字符串分页、保存进度和续读的引导验收，但长 TXT 文件读取接口未完成，使用说明内容验收也未通过。新 LoRA 仍只有可加载检查，未建立完整的能力回归门槛。

## 部署边界

这是现场源码快照，不是一键部署包。文件保留原有目录、服务和模型路径约定，另机运行需配置模型、适配器、CPU NF4 依赖、转换工具、llama.cpp 及实现指纹。不要直接运行带安装、服务切换或权重提升行为的模块覆盖现有环境。旧 4B／原生 KV 链路继续暂停。
