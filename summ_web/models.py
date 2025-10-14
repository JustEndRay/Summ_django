from __future__ import annotations

from django.contrib.auth import get_user_model
from django.db import models


class SummaryRecord(models.Model):
    """Модель для збереження записів сумаризації"""
    user = models.ForeignKey(get_user_model(), null=True, blank=True, on_delete=models.SET_NULL, related_name='summaries')  # Користувач
    created_at = models.DateTimeField(auto_now_add=True)  # Дата створення
    input_type = models.CharField(max_length=32)  # Тип вводу: text|file
    input_preview = models.TextField(blank=True)  # Попередній перегляд: перші N символів або назва файлу
    short_summary = models.TextField()  # Коротке резюме
    extended_summary = models.TextField(blank=True)  # Розширене резюме
    facts = models.TextField(blank=True)  # Факти (розділені новими рядками)

    class Meta:
        ordering = ['-created_at']  # Сортування за датою створення (новіші спочатку)


