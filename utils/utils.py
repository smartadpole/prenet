#!/usr/bin/env python3
# encoding: utf-8
'''
@author: sunhao
@contact: smartadpole@163.com
@file: utils.py
@time: 2025/2/17 10:04
@desc: 
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

import time
from functools import wraps

__all__ = ['timeit']


# Decorator function to measure the time taken by a function
def timeit(time_len):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            wrapper.times.append(elapsed_time)
            if len(wrapper.times) % time_len == 0:
                average_time = sum(wrapper.times[-time_len:]) / time_len
                unit = "ms" if average_time < 1 else "s"
                time_value = average_time * 1000 if average_time < 1 else average_time
                if 1 == time_len:
                    print(f"{func.__name__}: {time_value:.1f} {unit}", level='TIME')
                else:
                    print(f"Average time for last {time_len} frames in {func.__name__}: {time_value:.1f} {unit}", level='TIME')

            return result

        wrapper.times = []
        return wrapper

    return decorator
