#!/usr/bin/env python
"""GeneDiffusion环境测试脚本"""

import sys
import platform

print("===== GeneDiffusion 环境测试 =====")
print(f"Python版本: {sys.version}")
print(f"系统架构: {platform.machine()}")
print(f"操作系统: {platform.system()}")
print()

# 测试核心依赖
print("测试核心依赖...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  - CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA版本: {torch.version.cuda}")
        print(f"  - GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError as e:
    print(f"✗ PyTorch导入失败: {e}")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy导入失败: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas导入失败: {e}")

# 测试深度学习工具
print("\n测试深度学习工具...")
try:
    import einops
    print(f"✓ Einops {einops.__version__}")
except ImportError as e:
    print(f"✗ Einops导入失败: {e}")

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers导入失败: {e}")

# 测试生物信息学工具
print("\n测试生物信息学工具...")
try:
    import Bio
    print(f"✓ BioPython {Bio.__version__}")
except ImportError as e:
    print(f"✗ BioPython导入失败: {e}")

try:
    import pysam
    print(f"✓ PySAM {pysam.__version__}")
except ImportError as e:
    print(f"✗ PySAM导入失败: {e}")

# 测试项目模块
print("\n测试项目模块...")
sys.path.insert(0, '/home/share/huadjyin/home/lishaoshuai/yyt/gene-diffusion/src')
try:
    import data_processor
    print("✓ data_processor模块")
except ImportError as e:
    print(f"✗ data_processor导入失败: {e}")

try:
    import models
    print("✓ models模块")
except ImportError as e:
    print(f"✗ models导入失败: {e}")

print("\n===== 环境测试完成 =====") 