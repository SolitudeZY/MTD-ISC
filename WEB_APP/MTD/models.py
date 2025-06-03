import uuid

from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone
from django.conf import settings




# 数据增强管理
# class DataAugmentationTask(models.Model):
#     STATUS_CHOICES = (
#         ('pending', '待处理'),
#         ('running', '运行中'),
#         ('completed', '已完成'),
#         ('failed', '失败'),
#         ('stopped', '已终止'),
#     )
#
#     id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
#     user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
#     created_at = models.DateTimeField(auto_now_add=True)
#     status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
#     parameters = models.JSONField()
#     log_file = models.FileField(upload_to='task_logs/')
#     output_dir = models.CharField(max_length=255, blank=True)
#
#     def __str__(self):
#         return f"Task {self.id} - {self.status}"


class ModelManagement(models.Model):
    """模型管理"""
    MODEL_TYPE_CHOICES = (
        ('CNN', '卷积神经网络'),
        ('RNN', '循环神经网络'),
        ('LSTM', '长短期记忆网络'),
        ('TCN', '时序卷积网络'),
        ('BiLSTM', '双向长短期记忆网路'),
        ('BiTCN', '双向时序卷积网络'),
        ('Deep_Learning', '深度学习模型'),
        ('Machine_Learning', '机器学习模型'),
        ('Stacking_Learning_Model', '集成学习模型'),
        ('DMSE', '多堆叠集成模型'),  # deep-multi-stacking-ensemble
        ('OTHER', '其他，详情见描述'),
    )

    name = models.CharField(max_length=100, unique=True, verbose_name="模型名称")
    category = models.CharField(max_length=100, choices=MODEL_TYPE_CHOICES, default='OTHER', verbose_name="模型类别")
    upload_time = models.DateTimeField(default=timezone.now, verbose_name="上传时间")
    model_file = models.FileField(upload_to='models/', verbose_name="模型文件")
    description = models.TextField(blank=True, null=True, verbose_name="模型描述")

    class Meta:
        ordering = ['-upload_time']
        verbose_name = "模型管理"
        verbose_name_plural = "模型管理"

    def __str__(self):
        return f"{self.name} ({self.get_category_display()})"


class DatasetManagement(models.Model):
    """数据集管理"""
    DATASET_TYPE_CHOICES = (
        ('PCAP', '原始网络流量'),
        ('CSV', 'CSV格式'),
        ('GRAY', '灰度图格式'),
        ('RGB', 'RGB图格式'),
        ('OTHER', '其他'),
    )

    name = models.CharField(max_length=100, unique=True, verbose_name="数据集名称")
    category = models.CharField(max_length=10, choices=DATASET_TYPE_CHOICES, default='OTHER', verbose_name="数据集类别")
    upload_time = models.DateTimeField(default=timezone.now, verbose_name="上传时间")
    data_file = models.FileField(upload_to='datasets/', verbose_name="数据文件")
    size = models.PositiveIntegerField(verbose_name="数据量(条)")

    class Meta:
        ordering = ['-upload_time']
        verbose_name = "数据集管理"
        verbose_name_plural = "数据集管理"

    def __str__(self):
        return f"{self.name} ({self.get_category_display()}) - {self.size}条数据"

    def delete(self, *args, **kwargs):
        # 先删除数据库记录，再删除文件
        if self.data_file:
            storage = self.data_file.storage
            path = self.data_file.path
            super().delete(*args, **kwargs)
            storage.delete(path)
        else:
            super().delete(*args, **kwargs)


class DetectionHistory(models.Model):
    """模型检测历史"""
    model = models.ForeignKey(ModelManagement, on_delete=models.CASCADE, verbose_name="检测模型")
    dataset = models.ForeignKey(DatasetManagement, on_delete=models.CASCADE, verbose_name="检测数据集")
    detection_time = models.DateTimeField(auto_now_add=True, verbose_name="检测时间")
    is_malicious = models.BooleanField(default=False, verbose_name="是否含有恶意流量")
    accuracy = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True, verbose_name="检测准确率")
    FPR = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True, verbose_name="假阳率")
    TPR = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True, verbose_name="真阳率")
    F1_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True, verbose_name="F1分数")
    report = models.FileField(upload_to='detection_reports/', null=True, blank=True, verbose_name="检测报告")

    class Meta:
        ordering = ['-detection_time']
        verbose_name = "检测历史"
        verbose_name_plural = "检测历史记录"

    def __str__(self):
        return f"{self.model.name} - {self.dataset.name} - {self.detection_time.strftime('%Y-%m-%d %H:%M')}"


class UserInfo(AbstractUser):
    SEX_CHOICES = [
        ('F', 'Female'),
        ('M', 'Male'),
    ]
    phone = models.CharField(max_length=11, blank=True)
    sex = models.CharField(max_length=1, choices=SEX_CHOICES, blank=True)
    birth = models.DateField(blank=True, null=True)  # 修改为null=True，避免auto_now_add问题
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)

    def to_dict(self):
        """重写model_to_dict()方法转字典"""
        from datetime import datetime

        opts = self._meta
        data = {}
        for f in opts.concrete_fields:
            value = f.value_from_object(self)
            if isinstance(value, datetime):
                value = value.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(f, models.FileField):
                value = value.url if value else None
            data[f.name] = value
        return data


# 模型管理表
class Models_manage(models.Model):
    model_id = models.AutoField(primary_key=True)
    model_name = models.CharField(max_length=20, null=True)
    model_grouping = models.CharField(max_length=20, null=True)
    # 上传时间
    create_time = models.DateTimeField(auto_now_add=True, null=True)
    is_incremental_learning = models.CharField(max_length=20, null=True)
    is_multiple = models.CharField(max_length=20, null=True)

    class Meta:
        db_table = 'models_info'

    def to_dict(self):
        """重写model_to_dict()方法转字典"""
        from datetime import datetime

        opts = self._meta
        data = {}
        for f in opts.concrete_fields:
            value = f.value_from_object(self)
            if isinstance(value, datetime):
                value = value.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(f, models.FileField):
                value = value.url if value else None
            data[f.name] = value
        return data


class Malicious_models_manage(models.Model):
    Malicious_model_id = models.AutoField(primary_key=True)
    Malicious_model_name = models.CharField(max_length=20, null=True)
    Malicious_model_grouping = models.CharField(max_length=20, null=True)
    # 上传时间
    create_time = models.DateTimeField(auto_now_add=True, null=True)
    is_Bidirectional = models.CharField(max_length=20, null=True)
    is_feature = models.CharField(max_length=20, null=True)

    class Meta:
        db_table = 'Malicious_models_manage'

    def to_dict(self):
        """重写model_to_dict()方法转字典"""
        from datetime import datetime

        opts = self._meta
        data = {}
        for f in opts.concrete_fields:
            value = f.value_from_object(self)
            if isinstance(value, datetime):
                value = value.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(f, models.FileField):
                value = value.url if value else None
            data[f.name] = value
        return data


# 数据集管理
class Database_manage(models.Model):
    database_id = models.AutoField(primary_key=True)
    database_name = models.CharField(max_length=20, null=False)
    database_grouping = models.CharField(max_length=20, null=False)
    database_instances = models.CharField(max_length=20, null=False)
    database_features = models.CharField(max_length=20, null=False)
    create_time = models.DateTimeField(auto_now_add=True, null=False)

    class Meta:
        db_table = 'database_info'

    def to_dict(self):
        """重写model_to_dict()方法转字典"""
        from datetime import datetime

        opts = self._meta
        data = {}
        for f in opts.concrete_fields:
            value = f.value_from_object(self)
            if isinstance(value, datetime):
                value = value.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(f, models.FileField):
                value = value.url if value else None
            data[f.name] = value
        return data


# 数据集管理2
class database_manage2(models.Model):
    Database_id = models.AutoField(primary_key=True)
    Database_name = models.CharField(max_length=20, null=False)
    Database_number = models.CharField(max_length=20, null=False)
    Database_type = models.CharField(max_length=200, null=False)
    create_time = models.DateTimeField(auto_now_add=True, null=False)

    class Meta:
        db_table = 'database_manage2'

    def to_dict(self):
        """重写model_to_dict()方法转字典"""
        from datetime import datetime

        opts = self._meta
        data = {}
        for f in opts.concrete_fields:
            value = f.value_from_object(self)
            if isinstance(value, datetime):
                value = value.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(f, models.FileField):
                value = value.url if value else None
            data[f.name] = value
        return data


# 模型信息表
class model_info(models.Model):
    model_id = models.AutoField(primary_key=True)
    model_name = models.CharField(max_length=200, blank=False)
    model_info_url = models.URLField(max_length=200)
    upload_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'model_info'


# 测试数据集表
class Test_dataset(models.Model):
    test_id = models.AutoField(primary_key=True)
    test_name = models.CharField(max_length=40, blank=False)
    test_path = models.URLField(max_length=200)
    upload_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'Test_dataset'


# 真实程序表
class Execute_the_program(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=40, blank=False)
    path = models.URLField(max_length=200)

    class Meta:
        db_table = 'Execute_the_program'


# 执行结果表
class experimental_result(models.Model):
    id = models.AutoField(primary_key=True)
    # 模型名称
    tool_name = models.CharField(max_length=40, blank=False)
    # 数据集名称
    testcase_name = models.CharField(max_length=40, blank=False)
    # 指标
    indicator_a = models.CharField(max_length=40, blank=False)
    indicator_p = models.CharField(max_length=40, blank=False)
    indicator_r = models.CharField(max_length=40, blank=False)
    indicator_f = models.CharField(max_length=40, blank=False)

    class Meta:
        db_table = 'experimental_result'


#  态势预警
class Early_warning_database(models.Model):
    id = models.AutoField(primary_key=True)
    # 模型名称
    tool_name = models.CharField(max_length=40, blank=False)
    # 数据集名称
    testcase_name = models.CharField(max_length=40, blank=False)
    # 指标
    indicator_a = models.CharField(max_length=40, blank=False)
    indicator_p = models.CharField(max_length=40, blank=False)
    indicator_r = models.CharField(max_length=40, blank=False)
    indicator_f = models.CharField(max_length=40, blank=False)
    statu = models.CharField(max_length=40, blank=False)

    class Meta:
        db_table = 'Early_warning_database'


# 恶意流量检测结果表(赵英伟留下的中英混合代码 =_=)
class eyi_result(models.Model):
    id = models.AutoField(primary_key=True)
    # 模型名称
    models_name = models.CharField(max_length=40, blank=False)
    # 数据集名称
    database_name = models.CharField(max_length=40, blank=False)
    # 指标
    average_a = models.CharField(max_length=40, blank=False)
    average_p = models.CharField(max_length=40, blank=False)
    average_r = models.CharField(max_length=40, blank=False)
    average_f = models.CharField(max_length=40, blank=False)

    class Meta:
        db_table = 'eyi_result'


class malicious_traffic11(models.Model):
    id = models.AutoField(primary_key=True)
    # 模型名称
    name = models.CharField(max_length=200, blank=False)
    path = models.CharField(max_length=200, blank=False)


class malicious(models.Model):
    Mid = models.AutoField(primary_key=True)
    # 模型名称
    Mname = models.CharField(max_length=200, blank=False)
    Mpath = models.CharField(max_length=200, blank=False)

    class Meta:
        db_table = 'malicious'
