from django.db import models


# Create your models here.
class UserInfor(models.Model):
    username = models.CharField(max_length=30)
    password = models.CharField(max_length=30)
    age = models.IntegerField(default=99)


class Department(models.Model):
    title = models.CharField(max_length=16)
