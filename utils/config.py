#!/usr/bin/env python3
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: config.py
@time: 2025/11/21 20:56
@desc: Configuration management with .env support
'''
import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_norm = CURRENT_DIR.replace('\\', '/')
SRC_DIR = _norm[:_norm.find('/src/') + 4] if '/src/' in _norm else CURRENT_DIR
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import os
from typing import Optional
try:
    from dotenv import load_dotenv
except ImportError:
    print("Please install python-dotenv package: pip install python-dotenv")

__all__ = ['load_env_file', 'get_api_key', 'get_config', 'mask_secret', 'remove_bom']


# Global variable to avoid reloading .env file
_env_loaded = False


def remove_bom(path: str):
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            content = f.read()

        # Write back without BOM
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"Warning: Could not process .env file: {e}")


def mask_secret(value: str, prefix: int = 4, suffix: int = 4) -> str:
    """
    将密钥做部分遮盖，默认保留前4后4位。
    """
    if not value:
        return ""
    n = len(value)
    if n <= prefix + suffix:
        if n <= 2:
            return "*" * n
        return value[0] + "*" * (n - 2) + value[-1]
    return value[:prefix] + "*" * (n - prefix - suffix) + value[-suffix:]

def load_env_file(env_path: Optional[str] = None, override: bool = False) -> bool:
    """
    Load .env file

    Args:
        env_path: Path to .env file, defaults to .env in project root
        override: Whether to override existing environment variables, defaults to False

    Returns:
        True if file was loaded successfully, False otherwise
    """
    global _env_loaded

    if env_path is None:
        # recursively search for .env file from current directory upwards
        search_dir = os.path.dirname(os.path.abspath(__file__))
        found = None
        while True:
            candidate = os.path.join(search_dir, ".env")
            if os.path.isfile(candidate):
                found = candidate
                break
            parent = os.path.dirname(search_dir)
            if parent == search_dir: # Reached filesystem root
                break
            search_dir = parent
        env_path = found

    if not env_path or not os.path.isfile(env_path):
        return False


    remove_bom(env_path)

    result = load_dotenv(env_path, override=override)

    if result:
        print(f"Successfully loaded .env file: {env_path}")
        print("DEEPSEEK_API_KEY:", mask_secret(os.getenv("DEEPSEEK_API_KEY")))

        _env_loaded = True
    else:
        print(f"Failed to load .env file or file is empty: {env_path}")

    return result



def get_api_key(env_var: str = "DEEPSEEK_API_KEY") -> Optional[str]:
    """
    Get API Key from environment variable
    
    Args:
        env_var: Environment variable name
        
    Returns:
        API Key if found, None otherwise
    """
    # Try to load .env file first (only once)
    if not _env_loaded:
        load_env_file()
    
    # Try to get API Key (priority: original name -> uppercase -> lowercase)
    api_key = os.getenv(env_var)
    if not api_key:
        api_key = os.getenv(env_var.upper())
    if not api_key:
        api_key = os.getenv(env_var.lower())
    
    if not api_key:
        print(
            f"Environment variable {env_var} not found. "
            f"Please check if .env file or system environment variable is properly set."
            , level="warning"
        )
    
    return api_key


def get_config(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get configuration value from environment variable
    
    Args:
        key: Configuration key name
        default: Default value
        
    Returns:
        Configuration value
    """
    # Try to load .env file first (only once)
    if not _env_loaded:
        load_env_file()
    
    return os.getenv(key, default)


load_env_file()