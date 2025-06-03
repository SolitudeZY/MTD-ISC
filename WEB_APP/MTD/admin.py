from django.contrib import admin
from .models import ModelManagement, DatasetManagement, DetectionHistory, UserInfo

admin.site.register(ModelManagement)
admin.site.register(DatasetManagement)
admin.site.register(DetectionHistory)
admin.site.register(UserInfo)