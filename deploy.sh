#!/bin/bash
# S1-Base-32B 部署脚本（服务器 10.33.19.172 上执行）
# 显卡 6 + 7 双卡，端口 10090，上下文 128K

set -e

CONTAINER_NAME="s1-base-32b"
MODEL_PATH="/share/model/S1-Base-32B"
IMAGE="vllm/vllm-openai:latest"
HOST_PORT=10090

run_container() {
    docker run -d \
      --name "$CONTAINER_NAME" \
      --restart=always \
      --gpus '"device=6,7"' \
      --ipc=host \
      -p ${HOST_PORT}:8000 \
      -v ${MODEL_PATH}:/models/S1-Base-32B:ro \
      -e VLLM_NO_USAGE_STATS=1 \
      "$IMAGE" \
      --model /models/S1-Base-32B \
      --served-model-name S1-Base-32B \
      --dtype bfloat16 \
      --tensor-parallel-size 2 \
      --max-model-len 131072 \
      --gpu-memory-utilization 0.85
}

case "$1" in
    start)
        if docker ps --filter name="$CONTAINER_NAME" --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
            echo "容器已在运行"
        elif docker ps -a --filter name="$CONTAINER_NAME" --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
            docker start "$CONTAINER_NAME"
            echo "已启动已存在的容器"
        else
            run_container
            echo "已创建并启动新容器"
        fi
        ;;
    stop)
        docker stop "$CONTAINER_NAME"
        echo "已停止"
        ;;
    restart)
        docker restart "$CONTAINER_NAME"
        echo "已重启"
        ;;
    logs)
        docker logs -f "$CONTAINER_NAME"
        ;;
    status)
        docker ps -a --filter name="$CONTAINER_NAME"
        nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv -i 6,7
        ;;
    remove)
        docker rm -f "$CONTAINER_NAME"
        echo "已删除容器"
        ;;
    *)
        echo "用法: $0 {start|stop|restart|logs|status|remove}"
        exit 1
        ;;
esac