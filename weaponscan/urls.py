from django.contrib import admin
from django.urls import path, re_path
from django.conf import settings
from django.views.static import serve
from backend.api import api

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', api.urls),
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
]