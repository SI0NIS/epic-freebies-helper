# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
from zoneinfo import ZoneInfo
from loguru import logger

def timezone_filter(record):
    record["time"] = record["time"].astimezone(ZoneInfo("Asia/Shanghai"))
    return record

def init_log(**sink_channel):
    # 简单的日志初始化，不再包含任何补丁逻辑
    log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
    logger.remove()
    # diagnose=False: never render local variable values inside tracebacks.
    # This keeps credentials (e.g. account tuples) out of CI logs. The setting
    # belongs on the sink — logger.catch() does not accept a diagnose argument.
    logger.add(sink=sys.stdout, level=log_level, filter=timezone_filter, diagnose=False)

    # 挂载其他日志输出
    if sink_channel.get("error"):
        logger.add(
            sink=sink_channel.get("error"),
            level="ERROR",
            rotation="5 MB",
            filter=timezone_filter,
            diagnose=False,
        )
    if sink_channel.get("runtime"):
        logger.add(
            sink=sink_channel.get("runtime"),
            level="TRACE",
            rotation="5 MB",
            filter=timezone_filter,
            diagnose=False,
        )
        
    return logger
