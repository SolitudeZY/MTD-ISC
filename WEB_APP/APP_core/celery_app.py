# APP_core/celery_app.py
from celery import Celery
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'APP_core.settings')

app = Celery('APP_core')
app.config_from_object('django.conf:settings', namespace='CELERY')

# 显式指定包含tasks.py的app名称（如MTD）
app.autodiscover_tasks(lambda: ['MTD'])  # 修正此处

if __name__ == '__main__':
    app.start()
