# GeneDiffusion GitHub自动同步指南

## 🚀 快速开始

### 1. 配置SSH密钥

首先，您需要将SSH公钥添加到GitHub账户：

1. 登录 [GitHub](https://github.com)
2. 进入 Settings → SSH and GPG keys
3. 点击 "New SSH key"
4. 添加您的SSH公钥（已在终端显示）
5. 点击 "Add SSH key"

### 2. 首次推送

配置好SSH密钥后，执行首次推送：

```bash
git push -u origin main
```

## 📁 自动同步工具

我们提供了两个自动同步脚本：

### 1. 手动同步脚本 (`auto_sync.sh`)

用于手动触发同步：

```bash
# 运行同步
./scripts/auto_sync.sh
```

**功能**：
- 检测所有文件变化
- 自动生成提交信息
- 推送到GitHub

### 2. 实时监控脚本 (`watch_and_sync.sh`)

用于持续监控文件变化并自动同步：

```bash
# 启动监控
./scripts/watch_and_sync.sh

# 在后台运行
nohup ./scripts/watch_and_sync.sh > sync.log 2>&1 &
```

**功能**：
- 每30秒检查一次文件变化
- 自动分类提交（源码/文档/测试/配置）
- 实时显示同步状态
- 彩色输出便于查看

## 🔧 配置选项

### 修改检查间隔

编辑 `scripts/watch_and_sync.sh`，修改 `WATCH_INTERVAL` 变量：

```bash
WATCH_INTERVAL=30  # 默认30秒，可改为其他值
```

### 自定义提交信息

脚本会根据修改的文件类型自动生成提交信息：
- `src/` → "Update source code"
- `docs/` → "Update documentation"
- `tests/` → "Update tests"
- `configs/` → "Update configuration"

## 📋 使用建议

1. **开发时使用监控脚本**：
   ```bash
   # 开始开发前启动监控
   ./scripts/watch_and_sync.sh
   ```

2. **重要更新使用手动提交**：
   ```bash
   # 手动提交以添加详细说明
   git add .
   git commit -m "feat: Add species-aware embedding improvements"
   git push
   ```

3. **查看同步日志**：
   ```bash
   # 如果在后台运行
   tail -f sync.log
   ```

## ⚠️ 注意事项

1. **SSH密钥**：确保SSH密钥已添加到GitHub账户
2. **网络连接**：需要稳定的网络连接
3. **文件大小**：大文件会被`.gitignore`自动忽略
4. **敏感信息**：不要提交包含密码或密钥的文件

## 🛠️ 故障排除

### 推送失败

如果看到"推送失败"错误：

1. 检查SSH密钥：
   ```bash
   ssh -T git@github.com
   ```

2. 检查远程仓库：
   ```bash
   git remote -v
   ```

3. 手动推送查看详细错误：
   ```bash
   git push origin main
   ```

### 停止监控

按 `Ctrl+C` 停止监控脚本，或者：

```bash
# 查找进程
ps aux | grep watch_and_sync

# 终止进程
kill <PID>
```

## 📝 最佳实践

1. **频繁小提交**：让每次提交保持小而专注
2. **有意义的提交信息**：重要更新使用手动提交
3. **定期拉取**：如果多人协作，定期执行 `git pull`
4. **检查状态**：使用 `git status` 查看当前状态

---

现在您的GeneDiffusion项目已配置自动同步功能！每次修改都会自动备份到GitHub。 🎉 