"""
URL configuration for WEB_APP project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# WEB_APP/APP_core/urls
from django.contrib import admin
from django.urls import path, include
from MTD.views import home  # 导入主页视图,忽略这里的报错，可正常运行

urlpatterns = [
    path("admin/", admin.site.urls),
    # path('', include('MTD.urls')), # 测试用
    path('', include('MTD.login.urls')),
    # path('', include('MTD.home.urls')),
    path('', include('MTD.urls')),
    path('home/', home, name='home'),  # 添加主页URL

]
