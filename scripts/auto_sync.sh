#!/bin/bash

# GeneDiffusion 自动同步到GitHub脚本
# 用于自动提交和推送更改到GitHub

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 进入项目目录
cd "$(dirname "$0")/.." || exit 1

echo -e "${YELLOW}GeneDiffusion 自动同步脚本${NC}"
echo "================================="

# 检查是否有未提交的更改
if git diff-index --quiet HEAD --; then
    echo -e "${GREEN}✓ 没有需要同步的更改${NC}"
    exit 0
fi

# 显示更改的文件
echo -e "\n${YELLOW}检测到以下更改:${NC}"
git status --short

# 添加所有更改
echo -e "\n${YELLOW}添加更改到暂存区...${NC}"
git add .

# 生成提交信息
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
CHANGED_FILES=$(git diff --cached --name-only | wc -l)
COMMIT_MSG="Auto sync: ${CHANGED_FILES} files changed at ${TIMESTAMP}"

# 提交更改
echo -e "\n${YELLOW}提交更改...${NC}"
git commit -m "$COMMIT_MSG"

# 推送到GitHub
echo -e "\n${YELLOW}推送到GitHub...${NC}"
if git push origin main; then
    echo -e "\n${GREEN}✓ 成功同步到GitHub！${NC}"
    echo -e "提交信息: $COMMIT_MSG"
else
    echo -e "\n${RED}✗ 推送失败！请检查网络连接和SSH密钥配置${NC}"
    exit 1
fi

echo -e "\n${GREEN}同步完成！${NC}" 