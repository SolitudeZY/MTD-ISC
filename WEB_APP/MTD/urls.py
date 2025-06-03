from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    # path('', views.test, name='index'),  # 空字符串''：表示根URL路径，用于匹配用户访问的根URL（例如http://127.0.0.1:8000/）。

    path('model_management/', views.model_management, name='model_management'),  # 添加模型管理 model_management URL
    path('delete_model/<int:model_id>/', views.delete_model, name='delete_model'),  # 删除模型
    path('model_detail/<int:model_id>/', views.model_detail, name='model_detail'),  # 查看模型描述

    path('malicious_model_introduction/', views.malicious_model_introduction,
         name='malicious_model_introduction'),  # 恶意流量检测模型介绍
    path('data_augmentation_introduction/', views.data_augmentation_introduction,
         name='data_augmentation_introduction'),  # 数据增强模型介绍
    path('model_introduction/', views.model_introduction,
         name='model_introduction'),  # 模型介绍
    path('fullscreen-image/', views.fullscreen_image, name='fullscreen_image'),  # 全屏查看图片

    path('data_augmentation/', views.data_augmentation, name='data_augmentation'),  # 数据增强部分urls
    path('start_training/', views.start_training, name='start_training'),
    path('get_progress/', views.get_progress, name='get_progress'),
    path('download_model/<str:dataset_name>/', views.download_model, name='download_model'),
    path('download_source/', views.download_source_code, name='download_source'),
    path('sample_generation/', views.sample_generation, name='sample_generation'),  # 样本生成
    path('download_samples/', views.download_samples, name='download_samples'),

    path('model_detection/', views.model_detection, name='model_detection'),
    path('detection_results/<int:detection_id>/', views.detection_results, name='detection_results'),
    path('detection_records/', views.detection_records, name='detection_records'),  # 模型检测记录
    path('detection/delete/<int:pk>/', views.DetectionDeleteView.as_view(),  # 删除模型检测记录
         name='delete_detection_record'),

    path('visualization/', views.visualization, name='visualization'),  # 检测结果可视化
    path('dataset_model_distribution/', views.dataset_model_distribution, name='dataset_model_distribution'),

    path('attack_situation_awareness/', views.attack_situation_awareness, name='attack_situation_awareness'),  # 攻击态势感知

    path('personal_information/', views.personal_information, name='personal_information'),  # 个人信息
    path('dataset_management/', views.dataset_management, name='dataset_management'),  # 数据集管理

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
