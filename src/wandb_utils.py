"""
Wandb工具函数
"""

import logging
from typing import Dict, Any, Optional

try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def is_wandb_initialized() -> bool:
    """检查wandb是否已初始化"""
    if not WANDB_AVAILABLE or wandb is None:
        return False
    
    try:
        return hasattr(wandb, 'run') and getattr(wandb, 'run', None) is not None
    except:
        return False


def wandb_init(**kwargs) -> Optional[Any]:
    """安全地初始化wandb"""
    if not WANDB_AVAILABLE or wandb is None:
        logging.warning("Wandb未安装，跳过初始化")
        return None
    
    try:
        init_func = getattr(wandb, 'init', None)
        if init_func:
            return init_func(**kwargs)
        return None
    except Exception as e:
        logging.error(f"Wandb初始化失败: {e}")
        return None


def wandb_log(data: Dict[str, Any]) -> None:
    """安全地记录到wandb"""
    if not WANDB_AVAILABLE or wandb is None or not is_wandb_initialized():
        return
    
    try:
        log_func = getattr(wandb, 'log', None)
        if log_func:
            log_func(data)
    except Exception as e:
        logging.debug(f"Wandb记录失败: {e}")


def wandb_finish() -> None:
    """安全地结束wandb运行"""
    if not WANDB_AVAILABLE or wandb is None or not is_wandb_initialized():
        return
    
    try:
        finish_func = getattr(wandb, 'finish', None)
        if finish_func:
            finish_func()
    except Exception as e:
        logging.debug(f"Wandb结束失败: {e}") 