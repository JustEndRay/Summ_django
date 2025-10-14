from __future__ import annotations

from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import get_user_model


class TextSummarizeForm(forms.Form):
    """Форма для сумаризації тексту"""
    text = forms.CharField(widget=forms.Textarea(attrs={'rows': 6, 'placeholder': 'Enter text for summarization...'}), required=False)
    file = forms.FileField(required=False)  # Поле для завантаження файлу
    no_facts = forms.BooleanField(required=False, label='Do not show key facts', help_text='Disable extraction and display of key facts')

    def clean(self):
        """Валідація форми - потрібен або текст, або файл"""
        cleaned = super().clean()
        if not cleaned.get('text') and not cleaned.get('file'):
            raise forms.ValidationError('Please enter text or upload a file.')
        return cleaned


class RegisterForm(UserCreationForm):
    """Форма реєстрації користувача"""
    username = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Username'}))
    email = forms.EmailField(widget=forms.EmailInput(attrs={'placeholder': 'Email'}))
    password1 = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Password'}))
    password2 = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Confirm Password'}))
    
    class Meta:
        model = get_user_model()
        fields = ('username', 'email')


class LoginForm(AuthenticationForm):
    """Форма входу користувача"""
    username = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Username'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Password'}))


