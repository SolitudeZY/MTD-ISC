from django.urls import path
from . import views

app_name = 'linux_traffic'

urlpatterns = [
    path('', views.linux_traffic_capture, name='capture'),
    path('start/', views.start_linux_capture, name='start'),
    path('stop/', views.stop_linux_capture, name='stop'),
    path('status/', views.linux_capture_status, name='status'),
    path('download/<str:filename>/', views.download_linux_capture, name='download'),
    path('delete/<str:filename>/', views.delete_linux_capture, name='delete_linux_capture'),
    path('delete-all/', views.delete_all_linux_captures, name='delete_all_linux_captures'),
]