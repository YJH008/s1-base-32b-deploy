# S1-Base-32B 部署文档与运维手册

> 磐石科学基础大模型 S1-Base-32B，基于 vLLM 双卡（A800）Docker 部署完整记录。

## 一、环境信息

| 项目 | 内容 |
|------|------|
| 模型 | 磐石科学基础大模型 S1-Base-32B（ScienceOne 团队） |
| 架构 | Qwen3ForCausalLM（基于 Qwen3-32B 训练） |
| 参数量 | 32B（bfloat16，权重约 62GB） |
| 服务器 | 10.33.19.172（主机名 smtg4003） |
| GPU | 8× NVIDIA A800-SXM4-80GB |
| 部署显卡 | GPU 6 + GPU 7（张量并行 TP=2） |
| 端口 | 宿主机 10090 → 容器 8000 |
| 镜像 | vllm/vllm-openai:latest |
| 模型路径 | /share/model/S1-Base-32B |

## 二、部署过程复盘

### 2.1 遇到的问题与解决

| 阶段 | 问题 | 根因 | 解决 |
|------|------|------|------|
| 单卡初试 | `ValueError: KV cache insufficient`（需 32GiB，仅剩 4.62GiB） | 单卡部署 32B 模型，权重占 61.06GiB，默认 max_seq_len=131072 需 32GiB KV cache | 改双卡 TP=2 |
| 双卡 0.95 显存 | `ValueError: Free memory less than desired utilization` | vector-svc 残留进程占 ~4.3GB，每卡实际空闲仅 74.92GiB，0.95需 75.29GiB | 降到 0.90 |
| 上下文扩展 | 默认 32K 不够 | — | 设 `--max-model-len 131072` |

### 2.2 关键发现

1. **模型原生上下文上限 131072（128K）**：`max_position_embeddings: 131072`，实测 130772 token 输入成功处理。
2. **KV cache 与显存关系**：权重每卡占 30.61GiB，剩余才可用于 KV cache。0.85 利用率下 KV cache 池为 252,624 tokens。
3. **排队不崩溃**：vLLM V1 调度器原生排队，实测 8 路×59488 token 并发（总需求 47.5万 token 远超池子）全部返回 200，无 OOM。
4. **工具调用**：S1-Base 无 tool-calling 微调，模型不会产出 `tool_calls`（实测返回空数组）。但必须加 `--enable-auto-tool-choice --tool-call-parser hermes`，否则 vLLM 遇到带 `tools` 的请求会直接报 400，导致 WorkBuddy 等框架无法接入。
5. **思考无法关闭**：S1-Base 的思考是训练固化的自定义格式（` thinking` 文本标记，token id 151667），`enable_thinking=false` 无效，无 `reasoning_content` 独立字段。

## 三、拉起命令（最终版）

```bash
docker run -d \
  --name s1-base-32b \
  --restart=always \
  --gpus '"device=6,7"' \
  --ipc=host \
  -p 10090:8000 \
  -v /share/model/S1-Base-32B:/models/S1-Base-32B:ro \
  -e VLLM_NO_USAGE_STATS=1 \
  vllm/vllm-openai:latest \
  --model /models/S1-Base-32B \
  --served-model-name S1-Base-32B \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

### 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| --restart=always | always | 容器退出自动重启（自愈） |
| --gpus | device=6,7 | 指定显卡 6、7 |
| --ipc | host | 共享内存，避免多进程 NCCL 报错 |
| -p | 10090:8000 | 端口映射 |
| --model | /models/S1-Base-32B | 容器内模型路径 |
| --served-model-name | S1-Base-32B | API 暴露的模型名 |
| --dtype | bfloat16 | 权重精度 |
| --tensor-parallel-size | 2 | 双卡张量并行 |
| --max-model-len | 131072 | 最大上下文 128K |
| --gpu-memory-utilization | 0.85 | 显存利用率（留 15% 防 OOM 安全垫） |
| --enable-auto-tool-choice | - | 允许 `tool_choice: auto`（WorkBuddy 等框架接入必需） |
| --tool-call-parser | hermes | 工具调用解析器（Qwen3 系推荐 hermes） |

## 四、启动 / 停止 / 重启

```bash
# 启动（首次或容器被删除后）
docker run -d --name s1-base-32b ...（上面完整命令）

# 停止
docker stop s1-base-32b

# 启动（已存在）
docker start s1-base-32b

# 重启
docker restart s1-base-32b

# 删除容器（会触发 restart 策略注意）
docker rm -f s1-base-32b

# 查看日志
docker logs -f s1-base-32b

# 查看容器状态
docker ps -a --filter name=s1-base-32b
```

## 五、参数调整指南

### 5.1 调整上下文长度

```bash
# 修改 --max-model-len 值后重建容器
--max-model-len 32768    # 32K
--max-model-len 65536    # 64K
--max-model-len 131072   # 128K（模型上限，推荐）
```

### 5.2 调整显存利用率（防 OOM 安全垫）

| gpu_memory_utilization | 每卡空闲 | 安全垫 | 适用场景 |
|------------------------|---------|--------|---------|
| 0.90 | ~1.86GB | 薄 | max-model-len ≤ 32K 低并发 |
| 0.85 | ~5.94GB | 厚 | 128K 高并发（当前推荐） |
| 0.80 | ~9.8GB | 更厚 | 极端突发场景 |

> 注意：不能设 0.95，因为 vector-svc 残留进程占 ~4.3GB，0.95 需 75.29GiB 超过实际空闲 74.92GiB 会启动失败。

### 5.3 并发与 KV cache 测算

KV cache 池（0.85 下）= 252,624 tokens，并发数 = 池 ÷ 每路 token 消耗：

| 每路上下文 | 理论并发 |
|-----------|---------|
| 128K 满负荷 | ~2 路 |
| 64K | ~3 路 |
| 32K | ~7 路 |
| 8K | ~30 路 |

## 六、验证与测试

```bash
# 健康检查
curl -s -o /dev/null -w '%{http_code}' http://10.33.19.172:10090/health
# 输出：200

# 模型列表
curl -s http://10.33.19.172:10090/v1/models

# 推理测试
curl -s http://10.33.19.172:10090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "S1-Base-32B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 512
  }'
```

## 七、测试记录

| 测试项 | 结果 |
|--------|------|
| 36,102 token 长输入 | ✅ 成功 |
| 126,092 token 长输入 | ✅ 成功 |
| 130,772 token 长输入 | ✅ 成功（接近 128K 上限） |
| 131,063 token | ⚠️ 400（超出 131072 上限，正常拒绝） |
| 8 路 × 59488 token 并发 | ✅ 全部 200，排队不崩溃 |

## 八、工具调用与 WorkBuddy 接入

### 8.1 为什么必须加工具调用参数

S1-Base-32B 定位为科学推理模型，**本身无 tool-calling 微调**，即使传入 `tools` 也不会真正产出 `tool_calls`（实测返回空数组）。

但如果不加 `--enable-auto-tool-choice --tool-call-parser hermes` 这两个参数，vLLM 遇到带 `tools` 的请求会**直接返回 400 错误**（`"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`）。

WorkBuddy 等 AI 框架发起请求时默认会携带 tool 定义，因此**不加参数会导致接入失败**；加上后请求正常返回 200，对话功能可用。

### 8.2 能力边界

| 场景 | 是否可用 |
|------|---------|
| 普通对话 / 问答 / 科学推理 | ✅ 正常 |
| 模型主动调用外部工具 | ❌ 不支持（退化文字回答） |

> 如需模型主动调用工具（如联网搜索、查天气），需改用原生支持 function calling 的模型（如 Qwen3-32B-Instruct）。

### 8.3 WorkBuddy 配置

模型地址：`http://10.33.19.172:10090/v1`
模型名：`S1-Base-32B`
接口格式：OpenAI 兼容（chat/completions）