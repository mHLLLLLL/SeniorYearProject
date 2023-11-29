"""djangoProject1 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
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
from django.conf import settings
from django.contrib import admin
from django.urls import path, re_path
from django.views.static import serve

from app02 import views
# url和函数的对应关系
urlpatterns = [
    re_path(r'^media/(?P<path>.*)$', serve, {"document_root": settings.MEDIA_ROOT}),
    # path("admin/", admin.site.urls),
    # 登录路由
    path("login/",views.Login),
    #主界面
    path("main/",views.main),
    #部门管理
    path("depart/",views.depart),
    path("depart/add/",views.depart_add),
    path("depart/delete/",views.depart_delete),
    path("depart/<int:nid>/edit/",views.depart_edit), #传递动态的值

    #用户管理
    path("user/",views.user),
    path("user/add/",views.user_add),
    path("user/delete/",views.user_delete),
    path("user/<int:nid>/edit/",views.user_edit),

    #靓号管理
    path("phonenumber/",views.number),
    path("phonenumber/add/",views.number_add),
    path("phonenumber/delete/",views.number_delete),
    path("phonenumber/<int:nid>/edit/",views.number_edit),

    #文件上传
    path("file/",views.load_file),
    path("file/upload/model_form",views.upload_model_form),
]
