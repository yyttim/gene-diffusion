#!/bin/bash
# 启动全量数据预处理 - 支持后台运行

PROJECT_DIR="/home/share/huadjyin/home/lishaoshuai/yyt/gene-diffusion"
cd $PROJECT_DIR

echo "GeneDiffusion 全量数据预处理启动器"
echo "===================================="
echo ""
echo "请选择运行方式:"
echo "1) 使用 nohup (推荐)"
echo "2) 使用 screen"
echo "3) 前台运行 (不推荐，SSH断开会终止)"
echo ""
read -p "请输入选择 (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "使用 nohup 启动..."
        echo "日志将保存到: logs/preprocessing/"
        echo ""
        
        # 确保脚本有执行权限
        chmod +x scripts/preprocess_full_data.sh
        
        # 使用nohup启动
        nohup bash scripts/preprocess_full_data.sh > /dev/null 2>&1 &
        PID=$!
        
        echo "预处理已在后台启动，PID: $PID"
        echo ""
        echo "监控进度命令:"
        echo "  bash scripts/monitor_preprocessing.sh"
        echo ""
        echo "查看进程:"
        echo "  ps -ef | grep preprocess"
        echo ""
        echo "停止进程:"
        echo "  kill $PID"
        ;;
        
    2)
        echo ""
        echo "使用 screen 启动..."
        
        # 检查screen是否安装
        if ! command -v screen &> /dev/null; then
            echo "错误: screen 未安装"
            echo "请先安装: sudo apt-get install screen"
            exit 1
        fi
        
        # 确保脚本有执行权限
        chmod +x scripts/preprocess_full_data.sh
        
        # 创建screen会话
        SESSION_NAME="gene_preprocess"
        screen -dmS $SESSION_NAME bash scripts/preprocess_full_data.sh
        
        echo "预处理已在 screen 会话 '$SESSION_NAME' 中启动"
        echo ""
        echo "查看会话:"
        echo "  screen -r $SESSION_NAME"
        echo ""
        echo "分离会话: Ctrl+A 然后按 D"
        echo ""
        echo "监控进度:"
        echo "  bash scripts/monitor_preprocessing.sh"
        echo ""
        echo "查看所有screen会话:"
        echo "  screen -ls"
        ;;
        
    3)
        echo ""
        echo "前台运行..."
        echo "警告: SSH断开将终止进程！"
        echo ""
        read -p "确定要继续吗？(y/n): " confirm
        
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            chmod +x scripts/preprocess_full_data.sh
            bash scripts/preprocess_full_data.sh
        else
            echo "已取消"
        fi
        ;;
        
    *)
        echo "无效选择"
        exit 1
        ;;
esac 