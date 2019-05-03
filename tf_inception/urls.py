"""TensorflowServer URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.views.generic import RedirectView
# from django.contrib import admin
from classify_image.views import classify, classify_api

urlpatterns = [
    url(r'^classify_image/classify/api/$', classify_api),
    url(r'^classify_image/classify/$', classify),
    url(r'^$', RedirectView.as_view(url='classify_image/classify/')) #redirecting all urls to /classify
    # url(r'^admin/', admin.site.urls),
]