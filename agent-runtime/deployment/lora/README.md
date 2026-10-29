# 凝练输出 LoRA

这是人工触发的风格训练试验，不是自动经验学习系统。基础模型为已校验的官方
`Qwen3-4B-Instruct-2507` Safetensors；CPU 训练采用 FP32 计算，冻结基础权重，
选用的 concise-v2 只训练最后 8 层的 `q_proj/v_proj/o_proj/down_proj`，rank 8、alpha 16，
共 1,867,776 个可训练参数。
导出的 F16 LoRA 通过 llama.cpp 挂载到当前 Q4_K_M 基底，不合并基础权重。

`build_data.py` 生成 96 条人工编写的合成训练样本及独立开发/测试集；
`heldout_extra.py` 将测试集后半替换为 12 条不同措辞和场景的题目。
训练只读取 `data/train.jsonl`，不使用之前的检索评测答案或用户聊天记录。
样本包括直接回答、条件/否定/更正、简短拒答、来源归属、工具调用与按需读档。
第一轮 32 次更新、累积 2 条样本，平均回答反而变长，被拒绝启用。
第二轮采用答案正文损失与工具调用回放，64 次更新、每次 1 条样本，实际接触 64 条训练样本；
峰值进程内存约 19.1 GiB。每个版本保存自身脚本、数据、检查点和报告。

```powershell
.venv\Scripts\python.exe deployment/lora/build_data.py
.venv\Scripts\python.exe deployment/lora/heldout_extra.py
.venv\Scripts\python.exe deployment/lora/build_probes.py
```

Ubuntu 工作目录为 `/home/ubuntu/AI-Native/embodied-agent/training/concise-v2`。
第二轮训练、日志及检查点由 `lea-style-lora-v2.service` 保存；退出时恢复本项目服务。
适配器转换后先以全局强度 0 加载，`evaluate.py` 用每请求的 `lora` 字段对照 0/1，
使用相同 Q4 权重、提示词、证据、200 字符上限与 384 token 上限。评测期间不修改检索参数。
开发集用于检查 12 条模板及工具控制题；最终测试选取 12 条新措辞题，
并加入此前 12 题的冻结检索上下文，避免重复评测模板题。
另外增加 6 条反向事实题，检查日期、位置、成功/失败和已知邮箱是否被训练样例带偏；总计 30 题。
程序的关键词检查只作初筛，最终接受还需逐条核对来源、条件、拒答及工具参数。

`manage_adapter.py prepare` 转换并禁用加载；`activate` 要求已有 `review.json`，
且开发/测试报告哈希与审阅记录一致。启用后只对一级角色附加 LoRA，交接及其他角色
显式使用基础模型。调用日志记录适配器版本，健康检查核对适配器是否实际加载。

Ubuntu 回退命令：

```sh
python3 training/concise-v2/manage_adapter.py --root /home/ubuntu/AI-Native/embodied-agent --version concise-v2 rollback
```

训练报告、评测原始输出、人工审阅和适配器清单均保留在 Ubuntu 工作目录及
`adapters/concise-v2/manifest.json`。本地 `reports/lora-v2-test.json` 和
`reports/lora-reviewed-results.json` 保存原始对照和逐条复核；完整权重证据包已续传至
`reports/lora-evidence.zip` 并解压到 `reports/lora/`，整包 SHA-256 与全部 35 个文件哈希验证通过。
合成小样本上的改进不能代替真实长期使用评估，也不代表整个项目的 LoRA/Wake-Sleep 学习闭环已完成。

当前以 `--lora-scaled ...:0` 加载候选；一级请求指定强度 1，其他角色指定强度 0。
此前单用 `--lora-init-without-apply` 时接口报告的默认强度仍为 1，加载验证已拒绝该配置，
日志中保留此失败。不能仅凭命令行选项推断适配器未应用。

最终 30 题正文平均 36.933 → 27.7 字符（减少 25%）；生成中位耗时 12.7365 → 12.372 秒。
原文复核两组均 28/30 完整正确、2/30 部分正确：MJ-502 的权限与实际状态混淆，以及澄岳仓库负责人串联错误仍存在。
关键词初筛不是准确率；同义拒答、未使用字段 `tool=none` 等会影响初筛结果，逐条结论见 `review.json`。
开发题中的 world.move 整数版本参数问题也未修复；本试验不认证一般工具规划能力。
真实模型、真实 Embedding 和完整 Runtime 的隔离记忆问答及计算器执行均已通过。
