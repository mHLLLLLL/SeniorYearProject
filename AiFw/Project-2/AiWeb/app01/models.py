from django.db import models


# Create your models here.

# 创建数据库表，用于储存模型位置路径
class ModelPath(models.Model):
    name = models.CharField(verbose_name="weight_name", max_length=50)
    path = models.CharField(verbose_name="weight_path", max_length=100)


# 创建数据库表，用于储存上传文件的路径
class UploadPath(models.Model):
    name = models.CharField(verbose_name="upload_name", max_length=50)
    path = models.CharField(verbose_name="upload_path", max_length=100)


# 创建数据库表，用于储存处理后的文件路径
class OutputPath(models.Model):
    name = models.CharField(verbose_name="output_name", max_length=50)
    path = models.CharField(verbose_name="output_path", max_length=100)
