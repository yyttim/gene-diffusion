# GeneDiffusion 全量数据预处理指南

## 当前状态

预处理已在后台启动并正在运行：
- **进程PID**: 2351204
- **开始时间**: 2025-07-10 14:29:41
- **预计耗时**: 1-2小时（处理234个物种）
- **输出目录**: `datasets/preprocessed/`

## 监控进度

### 1. 快速状态检查
```bash
./check_preprocessing_status.sh
```

### 2. 详细进度监控
```bash
bash scripts/monitor_preprocessing.sh
```

### 3. 实时日志查看
```bash
# 查看主日志
tail -f logs/preprocessing/preprocess_full_20250710_142941.log

# 查看进度日志（包含INFO信息）
tail -f logs/preprocessing/preprocess_full_20250710_142941.err
```

### 4. 查看已处理的物种
```bash
ls -la datasets/preprocessed/*.h5 | wc -l  # 统计数量
ls -lt datasets/preprocessed/*.h5 | head -10  # 最新处理的10个
```

## 进程管理

### 检查进程状态
```bash
ps -ef | grep 2351204  # 查看主进程
ps aux | grep preprocess | grep -v grep  # 查看所有预处理进程
```

### 停止预处理（如需要）
```bash
kill 2351204  # 优雅停止
kill -9 2351204  # 强制停止（仅在必要时）
```

### 重新启动（如果中断）
```bash
# 先删除锁文件
rm datasets/preprocessed/.preprocessing.lock

# 重新启动
./start_preprocessing.sh
# 选择 1 使用nohup
```

## 预处理完成后

1. **检查数据完整性**
   ```bash
   ls datasets/preprocessed/*.h5 | wc -l  # 应该有234个文件
   ```

2. **查看统计信息**
   ```bash
   cat datasets/preprocessed/dataset_stats.json | python -m json.tool
   ```

3. **开始训练**
   ```bash
   # 小模型测试
   bash scripts/train_with_preprocessed.sh
   ```

## 注意事项

- 预处理使用8个进程并行处理，CPU使用率会很高（正常现象）
- SSH断开不会影响后台进程（使用了nohup）
- 如果需要查看更多历史日志，可以查看`logs/preprocessing/nohup.out`
- 预处理数据将占用约50-100GB磁盘空间

## 故障排除

1. **进程意外停止**
   - 检查错误日志：`tail -100 logs/preprocessing/preprocess_full_*.err`
   - 检查系统资源：`df -h`（磁盘空间）、`free -h`（内存）

2. **处理速度很慢**
   - 正常速度约20-30秒/物种
   - 检查CPU使用：`top`
   - 检查IO等待：`iostat -x 1`

3. **锁文件问题**
   - 如果提示"预处理已在运行"但实际没有，删除锁文件：
     ```bash
     rm datasets/preprocessed/.preprocessing.lock
     ``` 