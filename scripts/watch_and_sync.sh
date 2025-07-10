#!/bin/bash

# GeneDiffusion 文件监控和自动同步脚本
# 监控文件变化并自动推送到GitHub

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
WATCH_INTERVAL=30  # 检查间隔（秒）
PROJECT_DIR="$(dirname "$0")/.."

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   GeneDiffusion 文件监控自动同步系统   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo -e "\n${YELLOW}监控目录:${NC} $PROJECT_DIR"
echo -e "${YELLOW}检查间隔:${NC} ${WATCH_INTERVAL}秒"
echo -e "${YELLOW}开始时间:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "\n${GREEN}按 Ctrl+C 停止监控${NC}\n"

# 进入项目目录
cd "$PROJECT_DIR" || exit 1

# 初始化上次检查时间
LAST_SYNC=$(date +%s)

# 主循环
while true; do
    # 检查是否有未提交的更改
    if ! git diff-index --quiet HEAD -- 2>/dev/null || [ -n "$(git ls-files --others --exclude-standard)" ]; then
        echo -e "\n${YELLOW}[$(date '+%H:%M:%S')] 检测到文件变化:${NC}"
        
        # 显示变化的文件
        git status --short
        
        # 添加所有更改
        git add .
        
        # 生成提交信息
        TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
        CHANGED_FILES=$(git diff --cached --name-only | wc -l)
        
        # 获取主要更改类型
        if git diff --cached --name-only | grep -q "^src/"; then
            CHANGE_TYPE="Update source code"
        elif git diff --cached --name-only | grep -q "^docs/"; then
            CHANGE_TYPE="Update documentation"
        elif git diff --cached --name-only | grep -q "^tests/"; then
            CHANGE_TYPE="Update tests"
        elif git diff --cached --name-only | grep -q "^configs/"; then
            CHANGE_TYPE="Update configuration"
        else
            CHANGE_TYPE="Update project files"
        fi
        
        COMMIT_MSG="${CHANGE_TYPE}: ${CHANGED_FILES} files changed at ${TIMESTAMP}"
        
        # 提交更改
        echo -e "${YELLOW}提交更改...${NC}"
        git commit -m "$COMMIT_MSG" > /dev/null 2>&1
        
        # 推送到GitHub
        echo -e "${YELLOW}推送到GitHub...${NC}"
        if git push origin main > /dev/null 2>&1; then
            echo -e "${GREEN}✓ 成功同步到GitHub！${NC}"
            echo -e "${GREEN}  提交: $COMMIT_MSG${NC}"
            LAST_SYNC=$(date +%s)
        else
            echo -e "${RED}✗ 推送失败！${NC}"
            echo -e "${RED}  请检查SSH密钥是否已添加到GitHub${NC}"
            echo -e "${RED}  使用以下命令查看详细错误:${NC}"
            echo -e "${RED}  git push origin main${NC}"
        fi
    else
        # 计算距离上次同步的时间
        CURRENT_TIME=$(date +%s)
        TIME_DIFF=$((CURRENT_TIME - LAST_SYNC))
        MINUTES=$((TIME_DIFF / 60))
        SECONDS=$((TIME_DIFF % 60))
        
        # 显示状态
        echo -ne "\r${GREEN}[$(date '+%H:%M:%S')] 监控中... 距上次同步: ${MINUTES}分${SECONDS}秒${NC}    "
    fi
    
    # 等待下次检查
    sleep $WATCH_INTERVAL
done 