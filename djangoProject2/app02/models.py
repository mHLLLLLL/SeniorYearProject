from datetime import date

from django.db import models


# Create your models here.
# 创建数据库表单
# 用户表
class UserInfo(models.Model):
    name = models.CharField(verbose_name="姓名", max_length=16, default="zhangsan")
    password = models.CharField(verbose_name="密码", max_length=64, default="123")
    age = models.IntegerField(verbose_name="年龄", default=18)
    account = models.DecimalField(verbose_name="账户余额", max_digits=10, decimal_places=2, default=0)
    enroll = models.DateField(verbose_name="入职时间", default=date.today)
    # 性别
    gender_choise = (
        (1, "男"),
        (2, "女")
    )
    gender = models.SmallIntegerField(verbose_name="性别", choices=gender_choise)
    # 无约束
    # depart_id = models.BigIntegerField(verbose_name="部门ID")

    # 1.有约束
    #   - to，与那张表关联
    #   - to_field，表中的那一列关联
    # 2.django自动
    #   - 写的depart
    #   - 生成数据列 depart_id
    # 3.部门表被删除
    # ### 3.1 级联删除/联表删除
    depart = models.ForeignKey(to="Department", to_field="id", on_delete=models.CASCADE, verbose_name="部门名称")
    # ### 3.2 置空
    # depart = models.ForeignKey(to="Department", to_field="id", null=True, blank=True, on_delete=models.SET_NULL)


# 部门表
class Department(models.Model):
    title = models.CharField(verbose_name="标题", max_length=32)

    def __str__(self):
        return self.title


# 靓号管理
class PhoneNumber(models.Model):
    number = models.CharField(verbose_name="手机号", max_length=11)
    price = models.IntegerField(verbose_name="价格")
    level_choices = (
        (1, "一级"),
        (2, "二级"),
        (3, "三级"),
        (4, "四级")
    )
    status_choices = (
        (1, "已占用"),
        (2, "未占用")
    )
    level = models.SmallIntegerField(verbose_name="级别", choices=level_choices, default=1)
    status = models.SmallIntegerField(verbose_name="状态", choices=status_choices, default=2)


# 管理员
class Admin(models.Model):
    username = models.CharField(verbose_name="用户名", max_length=32)
    password = models.CharField(verbose_name="密码", max_length=64)


class City(models.Model):
    name = models.CharField(verbose_name="名称", max_length=32)
    count = models.IntegerField(verbose_name="人口")

    img = models.FileField(verbose_name="Logo", max_length=128, upload_to="city/")
