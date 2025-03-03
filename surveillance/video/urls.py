from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import home, video_feed, check_new_faces

urlpatterns = [
    path("", home, name="home"),
    path("video_feed/<str:mode>/", video_feed, name="video_feed"),
    path("check_new_faces/", check_new_faces, name="check_new_faces"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
