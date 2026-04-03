#!/bin/bash

# --- 配置区域 ---
LOCAL_DIR="/home/xqaqyn/nano-vllm/" 
REMOTE_HOST="nano-vllm-remote"
REMOTE_DIR="/root/autodl-tmp/nano-vllm"

# 定义排除列表（注意：末尾加 / 表示匹配目录）
EXCLUDES=(
    "--exclude=.git/"
    "--exclude=__pycache__/"
    "--exclude=.venv/"
    "--exclude=node_modules/"
    "--exclude=.DS_Store"
    "--exclude=.env"
    "--exclude=*.pyc"
)
# ----------------

echo "🔄 正在镜像同步到 $REMOTE_HOST..."

# 执行 rsync
# 使用 "${EXCLUDES[@]}" 确保数组中的每个元素被正确作为独立参数传递
rsync -avz --delete "${EXCLUDES[@]}" "$LOCAL_DIR" "$REMOTE_HOST:$REMOTE_DIR"

if [ $? -eq 0 ]; then
    echo "✅ 同步圆满完成！"
else
    echo "❌ 同步失败，请检查配置。"
fi