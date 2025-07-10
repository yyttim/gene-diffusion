#!/bin/bash
# GeneDiffusion任务管理脚本

PROJECT_DIR="/home/share/huadjyin/home/lishaoshuai/yyt/gene-diffusion"
LOG_DIR="${PROJECT_DIR}/logs"

function show_menu() {
    echo "===== GeneDiffusion 任务管理 ====="
    echo "1. 查看当前任务"
    echo "2. 查看任务详情"
    echo "3. 查看任务日志"
    echo "4. 终止任务"
    echo "5. 查看专用池资源"
    echo "6. 查看共享池资源"
    echo "7. 查看历史任务"
    echo "8. 提交测试任务"
    echo "9. 提交多节点任务"
    echo "0. 退出"
    echo "=========================="
}

function view_jobs() {
    echo "当前任务列表："
    djob -CO "JOB_ID:10 JOB_NAME:20 USER:16 JOB_STATE:10 REQ_CPU:8 REQ_GPU:8 TASK_EXEC_NODES"
}

function view_job_details() {
    read -p "请输入任务ID: " job_id
    djob -x -W $job_id
}

function view_job_logs() {
    read -p "请输入任务ID: " job_id
    echo "选择日志类型："
    echo "1. 标准输出 (stdout)"
    echo "2. 错误输出 (stderr)"
    echo "3. 两者都看"
    read -p "请选择 (1-3): " log_choice
    
    case $log_choice in
        1)
            tail -f ${LOG_DIR}/*${job_id}.out
            ;;
        2)
            tail -f ${LOG_DIR}/*${job_id}.err
            ;;
        3)
            tail -f ${LOG_DIR}/*${job_id}.*
            ;;
        *)
            echo "无效选择"
            ;;
    esac
}

function kill_job() {
    read -p "请输入要终止的任务ID: " job_id
    read -p "确认终止任务 $job_id? (y/n): " confirm
    if [ "$confirm" = "y" ]; then
        djob -T $job_id
        echo "任务 $job_id 已终止"
    fi
}

function view_pool_resources() {
    echo "专用池资源使用情况："
    dnode -W -g tmp_P24Z28400N0259
}

function view_shared_resources() {
    echo "共享池可用资源："
    dnode -W -g share | grep -v CLOSED | awk '{if($10!=0){print $1,$2,$3,$4,$8,$10}}'
}

function view_history() {
    echo "历史任务（最近20个）："
    djob -D | head -20
}

function submit_test_job() {
    echo "提交测试任务..."
    cd $PROJECT_DIR
    bash scripts/submit_job.sh
}

function submit_multinode_job() {
    echo "提交多节点任务..."
    cd $PROJECT_DIR
    bash scripts/submit_multinode_job.sh
}

# 主循环
while true; do
    show_menu
    read -p "请选择操作 (0-9): " choice
    
    case $choice in
        1) view_jobs ;;
        2) view_job_details ;;
        3) view_job_logs ;;
        4) kill_job ;;
        5) view_pool_resources ;;
        6) view_shared_resources ;;
        7) view_history ;;
        8) submit_test_job ;;
        9) submit_multinode_job ;;
        0) echo "退出任务管理"; exit 0 ;;
        *) echo "无效选择，请重试" ;;
    esac
    
    echo
    read -p "按回车继续..."
    clear
done 