from django.conf.urls import re_path, include, url
from rest_framework import routers
from rest_framework.documentation import include_docs_urls

from .views import WeatherView

urlpatterns = [
    url('weather/', WeatherView.as_view(), name="weathers"),
    url('docs/', include_docs_urls(title="drought-api"))
]