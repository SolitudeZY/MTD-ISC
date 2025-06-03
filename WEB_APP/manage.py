#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from pathlib import Path

# 当前文件路径：D:\Python Project\WEBSELF_2\WEB\manage.py
BASE_DIR = Path(__file__).resolve().parent  # 直接获取WEB目录路径
sys.path.append(str(BASE_DIR))  # 将WEB目录加入系统路径
sys.path.insert(0, str(BASE_DIR))  # 将WEB_APP目录加入路径
sys.path.insert(0, str(BASE_DIR / "APP_core"))  # 确保APP_core可导入


def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "APP_core.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
