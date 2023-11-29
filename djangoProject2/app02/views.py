from django.shortcuts import render, redirect, HttpResponse
from app02 import models
from django import forms
from django.core.validators import RegexValidator
from django.core.exceptions import ValidationError


# Create your views here.

# 写视图函数

# 登录视图函数
def Login(request):
    if request.method == "GET":
        return render(request, "login.html")
    else:
        # name = models.Admin.objects.
        if request.POST.get("username") == "admin" and request.POST.get("password") == "123456":
            return redirect('/main')
        else:
            return render(request, "login.html", {"message": "信息错误"})


# 主界面视图函数
def main(request):
    return HttpResponse("你好")


def depart(request):
    '''部门列表'''
    querset = models.Department.objects.all()
    return render(request, "depart.html", {"quersey": querset})


# 添加部门
def depart_add(request):
    if request.method == "GET":
        return render(request, "depart_add.html")
    title = request.POST.get("title")
    models.Department.objects.create(title=title)
    return redirect("/depart")


# 删除部门
def depart_delete(request):
    nid = request.GET.get('nid')
    models.Department.objects.filter(id=nid).delete()
    return redirect("/depart")


# 修改部门信息
def depart_edit(request, nid):
    if request.method == "GET":
        # value = request.GET.get('value')
        row_obj = models.Department.objects.filter(id=nid).first().title
        # print(row_obj)
        return render(request, "depart_edit.html", {'value': row_obj})
    else:
        new = request.POST.get('title')
        models.Department.objects.filter(id=nid).update(title=new)
        return redirect("/depart")


def user(request):
    '''用户管理'''
    users = models.UserInfo.objects.all()

    return render(request, "user.html", {"users": users})


class UserModelForm(forms.ModelForm):
    class Meta:
        model = models.UserInfo  # 不要加()
        fields = ["name", "password", "age", "enroll", "gender", "account", "depart"]
        widgets = {
            "password": forms.PasswordInput(attrs={"class": "form-control"})
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            field.widget.attrs = {"class": "form-control", "placeholder": field.label}


def user_add(request):
    """添加用户ModelForm版本"""
    if request.method == "GET":
        form = UserModelForm()
        return render(request, 'user_add.html', {"form": form})

    form = UserModelForm(data=request.POST)
    if form.is_valid():
        form.save()
        return redirect("/user")
    return render(request, 'user_add.html', {"form": form})


# 删除用户
def user_delete(request):
    nid = request.GET.get("nid")
    models.UserInfo.objects.filter(id=nid).delete()
    return redirect("/user")


# 编辑用户
def user_edit(request, nid):
    row_obj = models.UserInfo.objects.filter(id=nid).first()
    if request.method == "GET":
        form = UserModelForm(instance=row_obj)
        return render(request, 'user_edit.html', {"form": form})
    else:
        form = UserModelForm(data=request.POST, instance=row_obj)
        if form.is_valid():
            form.save()
            return redirect("/user")
        return render(request, 'user_edit.html', {"form": form})


# 展示phonenumber
def number(request):
    data_dic = {}
    value = request.GET.get('q', '')
    if value:
        data_dic["number__contains"] = value
    numbers = models.PhoneNumber.objects.filter(**data_dic)
    return render(request, "number.html", {"number": numbers, "dates": value})


class numberform(forms.ModelForm):
    number = forms.CharField(label="手机号",
                             validators=[RegexValidator(r'^1\d{10}+$', '号码必须以1开头且11位')])

    class Meta:
        model = models.PhoneNumber
        fields = ['number', 'price', 'level', 'status']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            field.widget.attrs = {"class": "form-control", "placeholder": field.label}

    def clean_number(self):
        # self.instance.pk
        txt_number = self.cleaned_data["number"]
        exists = models.PhoneNumber.objects.filter(number=txt_number).exists()
        if exists:
            raise ValidationError("手机号已存在")
        return txt_number


'''编辑'''


class edit(forms.ModelForm):
    number = forms.CharField(label="手机号",
                             validators=[RegexValidator(r'^1\d{10}+$', '号码必须以1开头且11位')])

    class Meta:
        model = models.PhoneNumber
        fields = ['number', 'price', 'level', 'status']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            field.widget.attrs = {"class": "form-control", "placeholder": field.label}

    def clean_number(self):
        self.instance.pk
        txt_number = self.cleaned_data["number"]
        exists = models.PhoneNumber.objects.exclude(id=self.instance.pk).filter(number=txt_number).exists()
        if exists:
            raise ValidationError("手机号已存在")
        return txt_number


# 添加靓号
def number_add(request):
    if request.method == "GET":
        form = numberform()
        return render(request, "number_add.html", {"form": form})
    else:
        form = numberform(data=request.POST)
        if form.is_valid():
            form.save()
            return redirect("/phonenumber/")
        return render(request, 'number_add.html', {"form": form})


# 删除靓号
def number_delete(request):
    nid = request.GET.get('nid')
    models.PhoneNumber.objects.filter(id=nid).delete()
    return redirect("/phonenumber")


# 编辑靓号
def number_edit(request, nid):
    row_obj = models.PhoneNumber.objects.filter(id=nid).first()
    if request.method == "GET":
        form = edit(instance=row_obj)
        return render(request, "number_edit.html", {"form": form})
    else:
        form = edit(data=request.POST, instance=row_obj)
        if form.is_valid():
            form.save()
            return redirect("/phonenumber/")
        return render(request, 'number_add.html', {"form": form})


# 文件上传
def load_file(request):
    if request.method == "GET":
        return render(request, "file.html")
    file_obj = request.FILES.get("avator")
    f = open(file_obj.name, mode='wb')
    for chunk in file_obj.chunks():
        f.write(chunk)

    f.close()
    return HttpResponse('1233')


# 添加用户基于ModelForm
# from django import forms
#
# class UserModelForm(forms.ModelForm):
#     class Meta:
#         model = models.UserInfo
#         fiedls = ["name","password","age","account"]
#
#
# def user_add(request):
#     form = UserModelForm()
#     return render(request,"user_add.html",{"form":form})
'''
#添加用户
def add(request):
    if request.method == "GET":
        return render(request,"add.html")
    else:
        user_name =request.POST.get("name")
        user_password = request.POST.get("password")
        user_age = request.POST.get("age")
        UserInfo.objects.create(name=user_name,password=user_password,age=user_age)
        return redirect(信息展示页的路由)


#删除用户
def delete(request):
    nid=request.GET.get("nid")
    UserInfo.objects.filter(id=nid).delte()
    return redirect('信息展示页路由')
'''


from app02.utils.Bootstrap import BootStrapModelForm

class upForm(BootStrapModelForm):
    name = forms.CharField(label="姓名")
    age = forms.IntegerField(label="年龄")
    img = forms.FileField(label="头像")

def uploadform(request):
   title = "上传文件"
   if request.method == "POST":
       form = upForm()
       return render(request,"upload_form.html",{"title":title,"form":form})
   form = upForm(data=request.POST,files=request.FILES)

   if form.is_valid():
         form.save()
         return HttpResponse("上传成功")

def upload_model_form(request):
    return render(request,"upload_form.html")
