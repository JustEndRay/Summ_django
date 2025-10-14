from __future__ import annotations

import os
from pathlib import Path

# Базова директорія проекту
BASE_DIR = Path(__file__).resolve().parent.parent

# Секретний ключ Django (з змінної середовища або за замовчуванням)
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'dev-secret-key')
# Режим налагодження (з змінної середовища)
DEBUG = os.getenv('DJANGO_DEBUG', '1') == '1'
# Дозволені хости (з змінної середовища)
ALLOWED_HOSTS = os.getenv('DJANGO_ALLOWED_HOSTS', '127.0.0.1,localhost').split(',')

# Встановлені додатки Django
INSTALLED_APPS = [
    'django.contrib.admin',  # Адміністративний інтерфейс
    'django.contrib.auth',   # Система аутентифікації
    'django.contrib.contenttypes',  # Типи контенту
    'django.contrib.sessions',  # Сесії
    'django.contrib.messages',  # Повідомлення
    'django.contrib.staticfiles',  # Статичні файли
    'summ_web',  # Наш додаток сумаризації
]

# Проміжне програмне забезпечення (middleware)
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',  # Безпека
    'django.contrib.sessions.middleware.SessionMiddleware',  # Сесії
    'django.middleware.locale.LocaleMiddleware',  # Локалізація
    'django.middleware.common.CommonMiddleware',  # Загальні функції
    'django.middleware.csrf.CsrfViewMiddleware',  # CSRF захист
    'django.contrib.auth.middleware.AuthenticationMiddleware',  # Аутентифікація
    'django.contrib.messages.middleware.MessageMiddleware',  # Повідомлення
    'django.middleware.clickjacking.XFrameOptionsMiddleware',  # Захист від clickjacking
]

# Конфігурація URL-ів
ROOT_URLCONF = 'config.urls'

# Конфігурація шаблонів
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],  # Директорія шаблонів
        'APP_DIRS': True,  # Пошук шаблонів у додатках
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# WSGI додаток
WSGI_APPLICATION = 'config.wsgi.application'

# Конфігурація бази даних
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',  # SQLite для розробки
        'NAME': BASE_DIR / 'db.sqlite3',  # Шлях до файлу бази даних
    }
}

# Відключаємо валідатори паролів для розробки щоб спростити реєстрацію
AUTH_PASSWORD_VALIDATORS = []

# Мовні налаштування
LANGUAGE_CODE = 'en'  # Мова за замовчуванням
TIME_ZONE = 'UTC'  # Часовий пояс
USE_I18N = True  # Увімкнення інтернаціоналізації
USE_TZ = True  # Використання часових поясів

# Доступні мови
LANGUAGES = [
    ('en', 'English'),
    ('uk', 'Українська'),
]

# Шляхи до файлів локалізації
LOCALE_PATHS = [BASE_DIR / 'locale']

# Налаштування мови
LANGUAGE_COOKIE_NAME = 'django_language'
LANGUAGE_COOKIE_AGE = 86400  # 24 години
LANGUAGE_SESSION_KEY = 'django_language'

# Налаштування статичних файлів
STATIC_URL = '/static/'  # URL для статичних файлів
STATIC_ROOT = BASE_DIR / 'staticfiles'  # Директорія для збору статичних файлів
STATICFILES_DIRS = [BASE_DIR / 'static']  # Додаткові директорії статичних файлів

# Автоматичне поле за замовчуванням
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# URL для перенаправлення після входу/виходу
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/'


