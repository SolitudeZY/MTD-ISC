# mtd/apps.py
from django.apps import AppConfig


class MTDConfig(AppConfig):
    name = 'MTD'
    verbose_name = "恶意流量检测"

    def ready(self):
        from . import signals  # 使用相对导入
