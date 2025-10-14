from django.urls import path
from django.conf.urls.i18n import i18n_patterns
from django.contrib.auth import views as auth_views
from . import views
from .forms import LoginForm

# URL-и додатку сумаризації
urlpatterns = [
    path('', views.chat_home, name='home'),  # Головна сторінка
    path('history/', views.history, name='history'),  # Історія сумаризацій
    path('register/', views.register, name='register'),  # Реєстрація
    path('login/', views.login_view, name='login'),  # Вхід
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),  # Вихід
    path('progress/<str:task_id>/', views.progress_stream, name='progress_stream'),  # Прогрес сумаризації
    path('summarize/<str:task_id>/', views.async_summarize, name='async_summarize'),  # Асинхронна сумаризація
    path('result/<str:task_id>/', views.get_result, name='get_result'),  # Отримання результату
]


