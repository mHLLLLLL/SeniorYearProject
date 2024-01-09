from django.http import HttpResponse
from django.shortcuts import render, redirect  # 导入render模块

from polls.models import UserInfor


# Create your views here.

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def text(request):
    return render(request, 'text.html')


def tpl(request):
    name_1 = "value"
    name_2 = ["admin", "user", "guest"]
    name_3 = {"name": "admin",
              "age": 19}
    return render(request,
                  'tpl.html',
                  {'n1': name_1, 'n2': name_2, "n3": name_3}
                  )


def something(request):
    print(request.method)
    print(request.GET)
    return HttpResponse("返回内容")
    # return render(request, 'something.html', {"title": "标题"})
    # return redirect("http://www.baidu.com")


def login(request):
    if request.method == "GET":
        return render(request, 'login.html')
    elif request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        if username == "admin" and password == "123456":
            return HttpResponse("登录成功")
        else:
            return render(request, 'login.html', {"error_msg": "用户名或密码错误"})


def orm(request):
    # 增加数据
    # Department.objects.create(title="RNG")
    # UserInfor.objects.create(username='张三', password="123456", age=18)
    # UserInfor.objects.create(username="李四", password="123")
    # UserInfor.objects.create(username="王五", password="123")
    # UserInfor.objects.create(username="赵六", password="123")

    # 删除数据
    # UserInfor.objects.filter(id=2).delete()

    # 获取数据
    data_list = UserInfor.objects.all()
    # for object in data_list:
    #     print(object.username, object.password, object.age)

    # data_list = UserInfor.objects.filter(id=1).first()
    # print(data_list.username, data_list.password, data_list.age)

    ## 改
    # UserInfor.objects.filter(id=1).update(age=20)

    return render(request, "info_list.html", {"data_list": data_list})


def info_add(request):
    if request.method == "GET":
        return render(request, "info_add.html")
    elif request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        age = request.POST.get("age")
        UserInfor.objects.create(username=username, password=password, age=age)
        return redirect("/orm/")


def info_delete(request):
    nid = request.GET.get("nid")
    UserInfor.objects.filter(id=nid).delete()
    return HttpResponse("删除成功")
