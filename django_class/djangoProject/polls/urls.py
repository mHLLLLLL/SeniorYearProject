from django.urls import path
from . import views

urlpatterns = [


    path("index/", views.index, name="index"),
    path("text/", views.text, name="text"),
    path("tpl", views.tpl, name="tpl"),
    path("something/", views.something, name="something"),
    path("orm/", views.orm, name="orm"),

]