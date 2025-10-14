from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

# Основні URL-и проекту
urlpatterns = [
    path('admin/', admin.site.urls),  # Адміністративний інтерфейс
    path('', include('summ_web.urls')),  # URL-и додатку сумаризації
]

# Обслуговування статичних файлів під час розробки
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])


