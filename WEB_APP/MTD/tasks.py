# MTD/tasks.py
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
import os
from django.conf import settings
import time
import logging

logger = logging.getLogger(__name__)


@shared_task(soft_time_limit=60)  # 必须保留装饰器
def simulate_training(dataset_name):
    source_path = os.path.join(settings.MEDIA_ROOT, 'models', 'ema_0.9999_017000.pt')
    sanitized_name = dataset_name.replace('-', '_')
    target_path = os.path.join(settings.MEDIA_ROOT, 'models', f'ema_0.9999_{sanitized_name}.pt')

    try:
        if not os.path.exists(source_path):
            logger.error(f"源文件不存在: {source_path}")
            return {'status': 'error', 'message': '源文件不存在'}

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(source_path, 'rb') as src, open(target_path, 'wb') as dst:
            dst.write(src.read())
        logger.info(f"文件复制成功: {target_path}")

        return {
            'status': 'completed',
            'dataset_name': sanitized_name,
            'path': target_path
        }

    except SoftTimeLimitExceeded:
        return {'status': 'error', 'message': '任务超时'}
    except Exception as e:
        logger.error(f"任务失败: {str(e)}")
        return {'status': 'error', 'message': str(e)}
