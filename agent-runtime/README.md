# Local Evolving Agent：Ubuntu 运行原型

2026-10-29 已将原生 KV 压缩接入 Ubuntu 正式一级 Agent。当前模型为 **Qwen3-4B-Instruct-2507 Q4_K_M**，原始 BF16 基准保留；正式健康检查返回一级 LoRA `concise-v2`。检索使用 **Qwen3-Embedding-0.6B F16 + 关键词并行召回、轻量规则重排和必要上下文补齐**，4B 组织回答，需要更多细节时可主动读档。

## 正式切窗机制

一级原生服务维护每个任务的真实 KV、各层各头的绝对位置及历史打分。物理窗口达到 **3072 tokens** 后，每头保留 **614 tokens**：起始 64、最近 128、按 Fast KVzip 分数选择的中间 422。旧 token 不重新送入模型，不生成文字摘要再重算；新输入继续追加在真实位置上。

切窗与工具执行账本、原始事件存档、最新约束版本协调。后台将 KV 与位置/分数侧车保存为带 SHA-256 的检查点，用原子 `CURRENT` 指针提交；重复请求可重放已提交结果，无效生成回滚。服务重启后加载检查点。输出语法按动作和授权工具约束，包含非空最终回答及工具必填参数；执行层继续验证权限、路径和世界状态版本。

正式服务的分工如下：

| 服务 | 职责 |
|---|---|
| `lea-agent.service`，18800 | 工作台、任务、约束、事件、工具及窗口事务 |
| `lea-native-kv.service`，18804，仅回环 | 一级 Q4 + LoRA 推理、Fast KVzip、任务检查点 |
| `lea-model.service`，18801 | 保留的辅助角色与旧模型路径 |
| `lea-embedding.service`，18803 | Embedding 编码 |

正式服务保留此前已验证的 AVX-VNNI Q4_K Repack GEMV/GEMM 优化库，基于 llama.cpp `86ea01d05ec237f89b78b41c8c1ee0f908141ac7`。原生引擎使用生成 6 线程、预填充 16 线程、batch/ubatch 512、Flash Attention；打分服务使用单 CPU 线程。它是 CPU 上的移植验证，不能等同于上游 CUDA/BF16 的逐位实现。

## 实测与边界

完整字段、版本指纹和不同测试口径见 [公开验收摘要](validation/native-kv-20261029.json)。

| 检查 | 结果 |
|---|---|
| 原生服务隔离验证 | 8/8：请求重放、任务隔离、自动真实压缩、无效生成回滚、浏览器来源拦截、取消、重启恢复、损坏检查点拒绝 |
| Agent 隔离联调 | 7/7：实际写入、自动切窗、切窗后新文件读取、写入不重复、无生成摘要、手动切窗、双服务重启后续接 |
| 正式入口部署检查 | 8/8：实际写入、自动切窗、切窗后读取、不重复写入、无文字摘要、保留 KV 位级一致、旧 token 不重算、手动切窗无生成 |
| 三次正式满窗压缩 | 0.285 / 0.277 / 0.316 秒，仅压缩算子及导入；约 453.0 MB → 90.5 MB 的序列化 KV |
| 正式手动切窗 | 提交至作业完成约 3.66 秒，含队列、保存、状态更新和轮询；单次观测，不是 P95 |
| 当前源码本地回归 | `python -m pytest -q`，69 passed |

**五分之一保留存在明确的语义损失。** 先前十窗口连续试验的核心事实正确为 **34/50**，答案与解释都完整正确为 **32/50**，仅 **2/10** 个窗口五题全对，第二个窗口开始出现错误。结构性 KV 检查均通过，不能据此宣称回答准确率可靠。本次上线没有重做该完整语义评测。早期联调还暴露过重复读、工具别名、空回答与漏参数，已补上执行账本及输出约束；这些修复也不能证明所有未来调用都正确。

原始事件与执行账本保留在磁盘上，但被压缩遗漏的信息不会自动重新进入 KV。模型仍可能误解否定、更正、对象或计划。绝对位置也受基础模型训练上下文上限约束，不是无限长度历史。当前具身入口只有模拟世界；真实相机、IMU、导航、电机，以及经验自动提炼、自动参数固化尚未完成。

## 代码范围

- `lea/`：当前本地开发源码，包含额外的 ScopedMemory 和索引适配器候选功能；**不代表所有候选功能均已部署到 Ubuntu**。
- `deployment/kv_service/app/lea/`：从实际安装版本制作的四文件正式覆盖层；上线时没有以全部本地源码覆盖生产环境。
- `deployment/kv_service/baseline/lea/`：制作与审核覆盖层所用的旧安装源码。`patch_app.py` 可重新生成覆盖层。
- `deployment/kv_service/native_worker.cpp`、`backend.py`：已上线的持久原生引擎和事务式会话服务。
- `deployment/kv_direct/fastkvzip_chain/`：连续窗口试验和原生 KV 操作的可审核实现。
- `deployment/vnni/`：有固定源码指纹校验的算子补丁与测试代码。
- `deployment/lora/`、`memory_lora/`、`index_lora/`、`index_lora_v2/`：人工触发的训练和评测代码；文件存在不代表对应适配器正在正式服务中启用。
- `prompts/`、`plans/`：提示词审核稿和 KV 试验计划，包含历史版本。

模型权重、训练产物、检查点、实际运行数据、下载/登录辅助脚本及网络凭据不随源码提交。原始机器日志与完整验收材料保存在部署机器和本地工作目录，公开仓库提供去除连接信息的实测摘要。

## 本地开发与 Ubuntu 部署

```sh
cd agent-runtime
python -m venv .venv
# 激活虚拟环境后
python -m pip install -e '.[test]'
python -m pytest -q
```

普通开发入口为 `uvicorn lea.app:create_app --factory --host 127.0.0.1 --port 18800`；还需要配置实际模型与 Embedding 服务。原生 KV 模式由 `LEA_NATIVE_KV_URL` 显式启用，未启用时仍走旧模型路径。

`deployment/kv_service/` 的部署脚本针对已经准备好的 Ubuntu 目录和固定版本依赖，并非通用的一键安装程序。运行前需要模型、LoRA、已优化的 llama.cpp 库、Torch 环境，以及经校验的 Fast KVzip gate 与上游源码。先在隔离端口运行 `stage_backend_tests.py`、`stage_agent_tests.py`，检查报告后再执行 `activate.py`。它会校验原安装源码、等待无活动作业、备份、安装覆盖层与用户服务，并通过正式健康检查；随后运行 `production_smoke.py` 验证正式入口。

`rollback.py` 恢复此前应用代码和原始记录可见游标，保留上线后新增的消息、约束、执行账本及 KV 检查点；不会用旧数据库覆盖新记录。部署脚本里的路径需要按目标机器调整，不应在陌生环境直接运行。

上游参考：[FastKVzip](https://github.com/Janghyun1230/FastKVzip)、[Fast-KVzip gate](https://huggingface.co/Jang-Hyun/Fast-KVzip)、[llama.cpp](https://github.com/ggml-org/llama.cpp)。上游依赖和权重适用各自许可；本仓库未重新分发其权重或整套源码。
