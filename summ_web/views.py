from __future__ import annotations

from pathlib import Path
from typing import Tuple

from django.contrib import messages
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, StreamingHttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
from django.utils.translation import gettext as _
from django.utils import translation

from .forms import TextSummarizeForm, RegisterForm, LoginForm
from .models import SummaryRecord

# Ініціалізація та управління CUDA
import torch
import json
import time
import threading
import uuid

# Global progress tracking
progress_tracker = {}
progress_lock = threading.Lock()
import os

def update_progress(task_id: str, progress: int, message: str, eta_seconds: int = None):
    """Update progress for a task"""
    with progress_lock:
        progress_tracker[task_id] = {
            'progress': progress,
            'message': message,
            'eta_seconds': eta_seconds,
            'timestamp': time.time()
        }

def get_progress(task_id: str) -> dict:
    """Get current progress for a task"""
    with progress_lock:
        return progress_tracker.get(task_id, {'progress': 0, 'message': 'Starting...', 'eta_seconds': None})

def clear_progress(task_id: str):
    """Clear progress for a task"""
    with progress_lock:
        if task_id in progress_tracker:
            del progress_tracker[task_id]

def _fallback_language_detection(text: str) -> str:
    """Резервне визначення мови коли langdetect не працює"""
    # Визначення російської мови
    russian_words = ['и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'за', 'из', 'к', 'о', 'у', 'со', 'об', 'во', 'при', 'про', 'без', 'над', 'под']
    russian_word_count = sum(1 for word in text.lower().split() if word in russian_words)
    russian_ratio = russian_word_count / len(text.split()) if len(text.split()) > 0 else 0
    
    # Визначення англійської мови
    english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among']
    english_word_count = sum(1 for word in text.lower().split() if word in english_words)
    english_ratio = english_word_count / len(text.split()) if len(text.split()) > 0 else 0
    
    print(f"[DEBUG] Fallback detection - Russian: {russian_ratio:.3f}, English: {english_ratio:.3f}", flush=True)
    
    if russian_ratio > 0.1:
        return "ru"
    elif english_ratio > 0.1:
        return "en"
    else:
        return "en"  # За замовчуванням англійська

def smart_chunk(text: str, max_words: int = 180) -> list[str]:
    """Розумне розбиття на чанки за реченнями з збереженням семантичної цілісності"""
    import re
    
    # Спочатку очищаємо текст
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Розбиваємо за реченнями
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return [text] if text else []
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # Якщо додавання цього речення перевищить max_words і у нас є контент
        if current_word_count + sentence_words > max_words and current_chunk:
            # Зберігаємо поточний чанк
            chunks.append(" ".join(current_chunk))
            # Починаємо новий чанк
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            # Додаємо речення до поточного чанка
            current_chunk.append(sentence)
            current_word_count += sentence_words
    
    # Додаємо фінальний чанк якщо існує
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Фільтруємо дуже короткі чанки
    chunks = [chunk for chunk in chunks if len(chunk.split()) > 10]
    
    return chunks

def _split_into_paragraphs(text: str, language: str = "en") -> list[str]:
    """Розбиває текст на абзаци для ієрархічної сумаризації використовуючи розумне розбиття"""
    if language == "uk":
        return smart_chunk(text, max_words=300)  # Більші чанки для української з mT5
    else:
        return smart_chunk(text, max_words=200)  # Стандартні чанки для англійської

# Функція пост-обробки видалена для підвищення продуктивності

# Глобальне сховище результатів
results_storage = {}
results_lock = threading.Lock()

def async_summarize(request: HttpRequest, task_id: str) -> HttpResponse:
    """Запускає асинхронну сумаризацію та повертає ID завдання"""
    try:
        form = TextSummarizeForm(request.POST, request.FILES)
        if form.is_valid():
            text: str = form.cleaned_data.get('text') or ''
            uploaded = request.FILES.get('file')
            no_facts = form.cleaned_data.get('no_facts', False)
            
            if uploaded:
                # Save uploaded file
                tmp_dir = Path.cwd() / 'tmp_uploads'
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / uploaded.name
                with tmp_path.open('wb') as fh:
                    for chunk in uploaded.chunks():
                        fh.write(chunk)
                from summarizer_agent.io.loaders import load_any
                text, _ = load_any(tmp_path)
                
                # Auto-cleanup uploaded file after processing
                try:
                    tmp_path.unlink()
                    print(f"[INFO] Cleaned up uploaded file: {uploaded.name}", flush=True)
                except Exception as e:
                    print(f"[WARNING] Failed to cleanup file {uploaded.name}: {e}", flush=True)
            
            # Start summarization in background thread
            def summarize_thread():
                try:
                    print(f"[DEBUG] Starting summarization thread for task {task_id}", flush=True)
                    short_summary, facts, _ = _summarize_text(text, no_facts=no_facts, task_id=task_id)
                    
                    print(f"[DEBUG] Summarization completed for task {task_id}", flush=True)
                    print(f"[DEBUG] Short summary length: {len(short_summary)}", flush=True)
                    
                    # Store results (extended_summary removed)
                    with results_lock:
                        results_storage[task_id] = {
                            'short_summary': short_summary,
                            'facts': facts,
                            'completed': True
                        }
                        print(f"[DEBUG] Results stored for task {task_id}", flush=True)
                    
                    # Save to database if user is authenticated
                    if request.user.is_authenticated:
                        input_type = 'file' if uploaded else 'text'
                        input_preview = uploaded.name if uploaded else text.strip()[:120]
                        SummaryRecord.objects.create(
                            user=request.user,
                            input_type=input_type,
                            input_preview=input_preview,
                            short_summary=short_summary,
                            extended_summary="",  # Extended summary removed
                            facts="\n".join(facts),
                        )
                    
                except Exception as e:
                    update_progress(task_id, -1, f"Error: {str(e)}")
                    with results_lock:
                        results_storage[task_id] = {
                            'error': str(e),
                            'completed': True
                        }
            
            thread = threading.Thread(target=summarize_thread)
            thread.daemon = True
            thread.start()
            
            return HttpResponse(json.dumps({'task_id': task_id}), content_type='application/json')
        else:
            return HttpResponse(json.dumps({'error': 'Invalid form'}), content_type='application/json', status=400)
    except Exception as e:
        return HttpResponse(json.dumps({'error': str(e)}), content_type='application/json', status=500)

def get_result(request: HttpRequest, task_id: str) -> HttpResponse:
    """Get summarization result"""
    print(f"[DEBUG] get_result called for task {task_id}", flush=True)
    with results_lock:
        print(f"[DEBUG] Available tasks: {list(results_storage.keys())}", flush=True)
        if task_id in results_storage:
            result = results_storage[task_id]
            print(f"[DEBUG] Found result for task {task_id}, completed: {result.get('completed')}", flush=True)
            if result.get('completed'):
                # Clean up
                del results_storage[task_id]
                clear_progress(task_id)
                print(f"[DEBUG] Returning result for task {task_id}", flush=True)
                return HttpResponse(json.dumps(result), content_type='application/json')
            else:
                print(f"[DEBUG] Task {task_id} still processing", flush=True)
                return HttpResponse(json.dumps({'status': 'processing'}), content_type='application/json')
        else:
            print(f"[DEBUG] Task {task_id} not found", flush=True)
            return HttpResponse(json.dumps({'error': 'Task not found'}), content_type='application/json', status=404)

def progress_stream(request: HttpRequest, task_id: str) -> StreamingHttpResponse:
    """Stream progress updates via Server-Sent Events"""
    def event_stream():
        print(f"[DEBUG] SSE stream started for task {task_id}", flush=True)
        last_progress = -1
        while True:
            progress_data = get_progress(task_id)
            current_progress = progress_data['progress']
            
            # Only send update if progress changed
            if current_progress != last_progress:
                data = {
                    'progress': current_progress,
                    'message': progress_data['message'],
                    'eta_seconds': progress_data['eta_seconds']
                }
                print(f"[DEBUG] SSE sending progress {current_progress}% for task {task_id}", flush=True)
                yield f"data: {json.dumps(data)}\n\n"
                last_progress = current_progress
                
                # Stop streaming when complete
                if current_progress >= 100:
                    print(f"[DEBUG] SSE stream completed for task {task_id}", flush=True)
                    break
            
            time.sleep(0.5)  # Update every 500ms
    
    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    return response

# Set CUDA environment variables for better compatibility
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Global CUDA context lock and state
_cuda_lock = threading.Lock()
_cuda_initialized = False
_cuda_available = False

def _initialize_cuda():
    """Initialize CUDA with proper error handling"""
    global _cuda_initialized, _cuda_available
    
    with _cuda_lock:
        if _cuda_initialized:
            return _cuda_available
            
        try:
            print(f"[DEBUG] Checking CUDA availability...")
            print(f"[DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}")
            
            if not torch.cuda.is_available():
                print("[WARNING] CUDA not available")
                _cuda_available = False
                _cuda_initialized = True
                return False
            
            # Get device count
            device_count = torch.cuda.device_count()
            print(f"[DEBUG] Device count: {device_count}")
            
            if device_count == 0:
                print("[WARNING] No CUDA devices found")
                _cuda_available = False
                _cuda_initialized = True
                return False
            
            # Set current device
            print(f"[DEBUG] Setting current device to 0...")
            torch.cuda.set_device(0)
            
            # Test basic CUDA operations
            print(f"[DEBUG] Testing CUDA operations...")
            test_tensor = torch.tensor([1.0], device='cuda:0')
            result = test_tensor * 2
            del test_tensor, result
            
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            device_name = torch.cuda.get_device_name(0)
            device_props = torch.cuda.get_device_properties(0)
            
            print(f"[INFO] CUDA initialized successfully")
            print(f"[INFO] Device: {device_name}")
            print(f"[INFO] VRAM: {device_props.total_memory / 1024**3:.1f} GB")
            print(f"[INFO] CUDA Version: {torch.version.cuda}")
            
            _cuda_available = True
            _cuda_initialized = True
            return True
        except Exception as e:
            print(f"[WARNING] CUDA initialization failed: {e}")
            import traceback
            traceback.print_exc()
            _cuda_available = False
            _cuda_initialized = True
            return False

def _ensure_cuda():
    """Ensure CUDA is available, raise error if not"""
    # Reset CUDA state for each request to avoid context issues
    global _cuda_initialized, _cuda_available
    _cuda_initialized = False
    _cuda_available = False
    
    if not _initialize_cuda():
        raise RuntimeError("CUDA not available but GPU processing is required")

_summarizers = {}  # Cache for different language models


def _clean_summary_prefixes(text: str) -> str:
    """Remove Ukrainian prefixes and clean up summary text."""
    if not text:
        return text
    
    # Remove common Ukrainian prefixes
    prefixes_to_remove = [
        "=== Коротке резюме ===",
        "=== Розширене резюме ===",
        "=== Факти та цифри ===",
        "Коротке резюме:",
        "Розширене резюме:",
        "Факти та цифри:",
        "=== Short Summary ===",
        "=== Extended Summary ===",
        "=== Key Facts ===",
        "Short Summary:",
        "Extended Summary:",
        "Key Facts:",
    ]
    
    cleaned_text = text.strip()
    
    # Remove prefixes
    for prefix in prefixes_to_remove:
        if cleaned_text.startswith(prefix):
            cleaned_text = cleaned_text[len(prefix):].strip()
            break
    
    # Clean up common artifacts
    import re
    
    # Remove repeated phrases
    cleaned_text = re.sub(r'(.+?)\1+', r'\1', cleaned_text)
    
    # Remove excessive punctuation and symbols
    cleaned_text = re.sub(r'[^\w\s\.,!?;:\-\(\)\"\'«»—–]', ' ', cleaned_text)
    
    # Clean up multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Remove sentences that are just symbols or very short
    sentences = cleaned_text.split('.')
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10 and not re.match(r'^[^\w]*$', sentence):
            clean_sentences.append(sentence)
    
    cleaned_text = '. '.join(clean_sentences).strip()
    if cleaned_text and not cleaned_text.endswith('.'):
        cleaned_text += '.'
    
    return cleaned_text


def _reset_summarizers():
    """Reset all summarizers to force reload"""
    global _summarizers
    if _summarizers:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        _summarizers = {}


def _get_summarizer(model_name: str = "facebook/bart-large-cnn", device_arg: str = "auto"):
    """Get or create summarizer for specific model"""
    global _summarizers
    
    cache_key = f"{model_name}_{device_arg}"
    
    if cache_key not in _summarizers:
        try:
            print(f"[INFO] Creating new summarizer for model: {model_name} with device: {device_arg}", flush=True)
            from summarizer_agent.nlp.summarizer import TextSummarizer
            
            # Ensure CUDA is available
            print(f"[DEBUG] Ensuring CUDA availability", flush=True)
            _ensure_cuda()
            
            # Примусове перезавантаження через очищення кешу
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Якщо device_arg == "cpu", примусово використовуємо CPU
            if device_arg == "cpu":
                print("[INFO] Forcing CPU usage for stability", flush=True)
            
            _summarizers[cache_key] = TextSummarizer(model_name=model_name, device_arg=device_arg)
            print(f"[INFO] Summarizer created successfully on device: {device_arg}", flush=True)
        except ImportError as e:
            print(f"[ERROR] Import error in _get_summarizer: {e}", flush=True)
            raise ImportError(f"ML dependencies not installed: {e}. Please install requirements.txt")
        except Exception as e:
            print(f"[ERROR] Error creating summarizer: {e}", flush=True)
            # GPU required: do not fallback to CPU
            raise
    else:
        print(f"[INFO] Using cached summarizer for model: {model_name} with device: {device_arg}", flush=True)
    
    return _summarizers[cache_key]


def _summarize_text(raw_text: str, no_facts: bool = False, task_id: str = None) -> Tuple[str, list[str], str]:
    try:
        print("[INFO] Starting text summarization...", flush=True)
        import sys
        sys.stdout.flush()
        
        if task_id:
            update_progress(task_id, 5, "Initializing summarization...")
        
        # Ensure CUDA is available
        print("[INFO] Ensuring CUDA availability...", flush=True)
        _ensure_cuda()
        
        if task_id:
            update_progress(task_id, 10, "CUDA initialized, loading model...")
        
        from summarizer_agent.io.loaders import load_any
        from summarizer_agent.nlp.chunker import chunk_text
        from summarizer_agent.nlp.facts import extract_facts
        from summarizer_agent.post.assemble import assemble_summary
        
        # Enhanced language detection with Ukrainian priority
        detected_lang = "en"  # Default to English
        
        # First, check for Ukrainian characters
        ukrainian_chars = sum(1 for c in raw_text if ord(c) >= 0x0400 and ord(c) <= 0x04FF)
        total_chars = len([c for c in raw_text if c.isalpha()])
        
        # Ukrainian-specific characters and words
        ukrainian_indicators = ['і', 'ї', 'є', 'ґ', 'І', 'Ї', 'Є', 'Ґ']
        ukrainian_words = [
            'та', 'що', 'для', 'від', 'про', 'без', 'над', 'під', 'між', 'через', 'за', 'до', 'по', 'у', 'на', 'в', 'з', 'із', 'про', 'при', 'проти', 'замість', 'окрім', 'крім',
            'було', 'була', 'були', 'є', 'єсть', 'має', 'може', 'треба', 'потрібно', 'можна', 'необхідно',
            'як', 'так', 'теж', 'також', 'але', 'однак', 'тому', 'тому', 'тому', 'тому', 'тому', 'тому',
            'це', 'цей', 'ця', 'це', 'ці', 'той', 'та', 'те', 'ті', 'свій', 'своя', 'своє', 'свої',
            'мій', 'моя', 'моє', 'мої', 'твій', 'твоя', 'твоє', 'твої', 'наш', 'наша', 'наше', 'наші',
            'ваш', 'ваша', 'ваше', 'ваші', 'їхній', 'їхня', 'їхнє', 'їхні', 'її', 'його', 'їх',
            'хто', 'що', 'який', 'яка', 'яке', 'які', 'де', 'куди', 'звідки', 'коли', 'чому', 'як',
            'школа', 'років', 'працював', 'пропрацював', 'місто', 'країна', 'люди', 'робота', 'життя'
        ]
        
        # Count Ukrainian indicators
        ukrainian_indicator_count = sum(1 for c in raw_text if c in ukrainian_indicators)
        ukrainian_word_count = sum(1 for word in raw_text.lower().split() if word in ukrainian_words)
        
        # Calculate Ukrainian probability
        cyrillic_ratio = ukrainian_chars / total_chars if total_chars > 0 else 0
        indicator_ratio = ukrainian_indicator_count / len(raw_text) if len(raw_text) > 0 else 0
        word_ratio = ukrainian_word_count / len(raw_text.split()) if len(raw_text.split()) > 0 else 0
        
        print(f"[DEBUG] Language detection analysis:", flush=True)
        print(f"[DEBUG] Cyrillic chars: {ukrainian_chars}/{total_chars} ({cyrillic_ratio:.2f})", flush=True)
        print(f"[DEBUG] Ukrainian indicators: {ukrainian_indicator_count} ({indicator_ratio:.3f})", flush=True)
        print(f"[DEBUG] Ukrainian words: {ukrainian_word_count} ({word_ratio:.3f})", flush=True)
        
        # Determine if Ukrainian
        is_ukrainian = (
            cyrillic_ratio > 0.2 or  # At least 20% Cyrillic characters
            indicator_ratio > 0.01 or  # At least 1% Ukrainian-specific characters
            word_ratio > 0.1  # At least 10% Ukrainian words
        )
        
        # Always try langdetect first for better accuracy
        try:
            from langdetect import detect, detect_langs
            detected_lang = detect(raw_text)
            langs = detect_langs(raw_text)
            print(f"[INFO] langdetect result: {detected_lang} (confidence: {langs[0].prob:.3f})", flush=True)
            
            # If langdetect gives low confidence, use our detection
            if langs[0].prob < 0.7:
                print(f"[INFO] Low confidence ({langs[0].prob:.3f}), using our detection", flush=True)
                if is_ukrainian:
                    detected_lang = "uk"
                    print(f"[INFO] Our detection: Ukrainian (cyrillic: {cyrillic_ratio:.2f}, indicators: {indicator_ratio:.3f}, words: {word_ratio:.3f})", flush=True)
                else:
                    detected_lang = _fallback_language_detection(raw_text)
            else:
                # High confidence from langdetect, but double-check Ukrainian
                if detected_lang != "uk" and is_ukrainian:
                    print(f"[INFO] langdetect says {detected_lang}, but our detection suggests Ukrainian", flush=True)
                    print(f"[INFO] Ukrainian indicators: cyrillic={cyrillic_ratio:.2f}, indicators={indicator_ratio:.3f}, words={word_ratio:.3f}", flush=True)
                    # Override with Ukrainian if our detection is strong
                    if cyrillic_ratio > 0.5 or indicator_ratio > 0.02 or word_ratio > 0.05:
                        detected_lang = "uk"
                        print(f"[INFO] Overriding to Ukrainian based on strong indicators", flush=True)
                    
        except ImportError:
            print("[INFO] langdetect module not available, using our detection", flush=True)
            if is_ukrainian:
                detected_lang = "uk"
                print(f"[INFO] Our detection: Ukrainian (cyrillic: {cyrillic_ratio:.2f}, indicators: {indicator_ratio:.3f}, words: {word_ratio:.3f})", flush=True)
            else:
                detected_lang = _fallback_language_detection(raw_text)
        except Exception as e:
            print(f"[INFO] langdetect error ({e}), using our detection", flush=True)
            if is_ukrainian:
                detected_lang = "uk"
                print(f"[INFO] Our detection: Ukrainian (cyrillic: {cyrillic_ratio:.2f}, indicators: {indicator_ratio:.3f}, words: {word_ratio:.3f})", flush=True)
            else:
                detected_lang = _fallback_language_detection(raw_text)
        
        # Use appropriate model based on language
        if detected_lang == "uk":
            # Try different model names that might be in cache
            import os
            # Optimized fallback order for Ukrainian - mT5 first
            possible_models = [
                "csebuetnlp/mT5_multilingual_XLSum",  # First try - optimal for Ukrainian
                "facebook/bart-large-cnn"  # Final fallback
            ]
            
            model_name = None
            for model in possible_models:
                try:
                    # Test if model can be loaded
                    from transformers import AutoTokenizer
                    print(f"[DEBUG] Testing model: {model}", flush=True)
                    # Use fast tokenizer for mBART, slow for mT5
                    use_fast = "mbart" in model.lower() or "bart" in model.lower()
                    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=use_fast)
                    model_name = model
                    print(f"[INFO] Using {model} for Ukrainian text (successfully loaded)", flush=True)
                    break
                except Exception as e:
                    print(f"[DEBUG] Failed to load {model}: {str(e)[:200]}...", flush=True)
                    continue
            
            if not model_name:
                # Final fallback to BART
                model_name = "facebook/bart-large-cnn"
                print(f"[INFO] All models failed, using BART for Ukrainian text", flush=True)
        else:
            # Try mT5 first for English as well
            possible_models = [
                "csebuetnlp/mT5_multilingual_XLSum",  # First try - multilingual model
                "facebook/bart-large-cnn"  # Fallback - specialized for news summarization
            ]
            
            model_name = None
            for model in possible_models:
                try:
                    # Test if model can be loaded
                    from transformers import AutoTokenizer
                    print(f"[DEBUG] Testing model: {model}", flush=True)
                    # Use slow tokenizer for mT5, fast for BART
                    use_fast = "bart" in model.lower()
                    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=use_fast)
                    model_name = model
                    print(f"[INFO] Using {model} for English text (successfully loaded)", flush=True)
                    break
                except Exception as e:
                    print(f"[DEBUG] Failed to load {model}: {str(e)[:200]}...", flush=True)
                    continue
            
            if not model_name:
                # Final fallback to BART
                model_name = "facebook/bart-large-cnn"
                print(f"[INFO] All models failed, using BART for English text", flush=True)
        
        # Автоматичне визначення параметрів для великих текстів
        text_words = len(raw_text.split())
        is_large_text = text_words > 5000  # Вважаємо великим текст більше 5000 слів
        
        # Use GPU (already verified at module import)
        device_arg = "cuda:0"
        print(f"[INFO] Using GPU processing", flush=True)
        
        # Configure chunk size based on language and text size
        if detected_lang == "uk":
            # Larger chunks for Ukrainian with mBART (better context handling)
            if is_large_text:
                print(f"[INFO] Large Ukrainian text detected ({text_words} words) - using larger chunks for mBART", flush=True)
                chunk_max_tokens = 1024  # Збільшено для mBART
                overlap_tokens = 200
            else:
                print(f"[INFO] Ukrainian text size: {text_words} words - using optimized chunks for mBART", flush=True)
                chunk_max_tokens = 1024  # Збільшено для mBART
                overlap_tokens = 200
        else:
            # Standard chunks for English
            if is_large_text:
                print(f"[INFO] Large text detected ({text_words} words) - using GPU with smaller chunks", flush=True)
                chunk_max_tokens = 256  # Менші чанки для GPU
                overlap_tokens = 50
            else:
                print(f"[INFO] Text size: {text_words} words - using GPU with standard chunks", flush=True)
                chunk_max_tokens = 512
                overlap_tokens = 100
            
        if detected_lang == "uk":
            print(f"[INFO] Using mT5 model for Ukrainian text (optimized for multilingual summarization)", flush=True)
        else:
            print(f"[INFO] Using mT5 model for English text (multilingual model with conservative parameters)", flush=True)
        print(f"[INFO] Loading summarizer model: {model_name}...", flush=True)
        
        if task_id:
            update_progress(task_id, 20, f"Loading model: {model_name}...")
        summarizer = _get_summarizer(model_name=model_name, device_arg=device_arg)
        
        # Check if text is long enough for hierarchical summarization
        text_length = len(raw_text)
        # For Ukrainian, disable hierarchical summarization for shorter texts to preserve structure
        if detected_lang == "uk":
            is_long_text = text_length > 6000  # Збільшено поріг для української
        else:
            is_long_text = text_length > 4000
        
        if is_long_text:
            print(f"[INFO] Long text detected ({text_length} chars), using hierarchical summarization", flush=True)
            if task_id:
                update_progress(task_id, 30, "Splitting text into paragraphs for hierarchical summarization...")
            
            # Split into paragraphs first
            paragraphs = _split_into_paragraphs(raw_text, detected_lang)
            print(f"[INFO] Created {len(paragraphs)} paragraphs", flush=True)
            
            if task_id:
                update_progress(task_id, 40, f"Summarizing {len(paragraphs)} paragraphs...")
            
            # Summarize each paragraph
            paragraph_summaries = []
            for i, paragraph in enumerate(paragraphs):
                if task_id:
                    progress = 40 + int((i / len(paragraphs)) * 30)  # 40-70% range
                    update_progress(task_id, progress, f"Summarizing paragraph {i+1}/{len(paragraphs)}...")
                
                # Summarize paragraph
                para_chunks = chunk_text(paragraph, tokenizer=summarizer.tokenizer, max_tokens=chunk_max_tokens, overlap_tokens=overlap_tokens, device=device_arg)
                if para_chunks:
                    para_summary = summarizer.summarize_chunks(para_chunks, min_words=30, max_words=80, device=device_arg, progress_callback=None, language=detected_lang)
                    if para_summary:
                        paragraph_summaries.extend(para_summary)
            
            # Combine paragraph summaries for final summarization
            combined_text = " ".join(paragraph_summaries)
            print(f"[INFO] Combined {len(paragraph_summaries)} paragraph summaries", flush=True)
            
            if task_id:
                update_progress(task_id, 70, "Creating final summary from paragraph summaries...")
            
            # Final summarization of combined text
            final_chunks = chunk_text(combined_text, tokenizer=summarizer.tokenizer, max_tokens=chunk_max_tokens, overlap_tokens=overlap_tokens, device=device_arg)
            chunks = final_chunks
        else:
            if detected_lang == "uk":
                print(f"[INFO] Short Ukrainian text ({text_length} chars), using single-stage summarization to preserve structure", flush=True)
            else:
                print(f"[INFO] Standard text length ({text_length} chars), using regular chunking", flush=True)
            print(f"[INFO] Chunking text with max_tokens={chunk_max_tokens}...", flush=True)
            if task_id:
                update_progress(task_id, 30, "Splitting text into chunks...")
            chunks = chunk_text(raw_text, tokenizer=summarizer.tokenizer, max_tokens=chunk_max_tokens, overlap_tokens=overlap_tokens, device=device_arg)
            print(f"[INFO] Created {len(chunks)} chunks", flush=True)
            
            if task_id:
                update_progress(task_id, 40, f"Created {len(chunks)} chunks, starting summarization...")
        
        print("[INFO] Summarizing chunks...", flush=True)
        
        # Комбінований підхід до розрахунку довжини чанків
        text_words = len(raw_text.split())
        num_chunks = len(chunks)
        target_total_words = 400  # Базова цільова довжина фінального резюме
        
        # Базові значення
        base_min = 60
        base_max = 150
        
        # Фактори адаптації
        text_factor = min(1.3, max(0.8, text_words / 2000))  # Адаптація під розмір тексту
        chunk_factor = min(1.2, max(0.9, 5 / num_chunks))    # Адаптація під кількість чанків
        
        # Розрахунок адаптивних меж
        chunk_min_words = int(base_min * text_factor * chunk_factor)
        chunk_max_words = int(base_max * text_factor * chunk_factor)
        
        # Адаптуємо цільову довжину фінального резюме під реальні можливості чанків
        realistic_target_words = min(target_total_words, num_chunks * chunk_max_words)
        
        print(f"[INFO] Adaptive chunk sizing: {text_words} words -> {num_chunks} chunks", flush=True)
        print(f"[INFO] Adaptation factors: text={text_factor:.2f}, chunks={chunk_factor:.2f}", flush=True)
        print(f"[INFO] Chunk limits: min={chunk_min_words}, max={chunk_max_words} words", flush=True)
        print(f"[INFO] Final target: {realistic_target_words} words (from {target_total_words})", flush=True)
        
        # Progress callback for chunk summarization
        def progress_callback(progress, message):
            if task_id:
                update_progress(task_id, progress, message)
        
        chunk_summaries = summarizer.summarize_chunks(chunks, min_words=chunk_min_words, max_words=chunk_max_words, device=device_arg, progress_callback=progress_callback, language=detected_lang)
        print(f"[INFO] Summarized {len(chunk_summaries)} chunks", flush=True)
        
        if task_id:
            update_progress(task_id, 90, "Assembling final summary...")
        
        # Facts extraction disabled by default
        print("[INFO] Facts extraction disabled", flush=True)
        facts = []
        tables_note = ""
        
        print("[INFO] Assembling final summary...", flush=True)
        short_summary, extended_summary = assemble_summary(
            chunk_summaries, 
            facts, 
            tables_note, 
            target_min_words=100, 
            target_max_words=realistic_target_words,
            original_text=raw_text
        )
        
        print("[INFO] Summarization completed successfully!", flush=True)
        
        if task_id:
            update_progress(task_id, 100, "Summarization completed!")
        
        # Очищення префіксів з результатів
        short_summary = _clean_summary_prefixes(short_summary)
        
        # Post-processing removed for performance
        
        # Очищення CUDA кешу після кожної сумаризації для запобігання проблемам з пам'яттю
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        sys.stdout.flush()
        return short_summary, facts, ""
    except ImportError as e:
        print(f"[ERROR] Import error: {e}", flush=True)
        raise ImportError(f"ML dependencies not installed: {e}. Please install requirements.txt")
    except Exception as e:
        print(f"[ERROR] Summarization error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


@require_http_methods(["GET", "POST"])
@csrf_protect
def chat_home(request: HttpRequest) -> HttpResponse:
    print(f"[INFO] chat_home called with method: {request.method}", flush=True)
    result: dict = {}
    
    # Check if this is an AJAX request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    if request.method == 'POST':
        print("[INFO] Processing POST request", flush=True)
        form = TextSummarizeForm(request.POST, request.FILES)
        if form.is_valid():
            print("[INFO] Form is valid, starting processing", flush=True)
            text: str = form.cleaned_data.get('text') or ''
            uploaded = request.FILES.get('file')
            input_type = 'text'
            input_preview = ''
            try:
                if uploaded:
                    print(f"[INFO] Processing uploaded file: {uploaded.name}", flush=True)
                    # Зберігаємо тимчасово у MEDIA-подібну директорію в BASE_DIR/tmp
                    tmp_dir = Path.cwd() / 'tmp_uploads'
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    tmp_path = tmp_dir / uploaded.name
                    with tmp_path.open('wb') as fh:
                        for chunk in uploaded.chunks():
                            fh.write(chunk)
                    from summarizer_agent.io.loaders import load_any
                    text, tables_note = load_any(tmp_path)
                    input_type = 'file'
                    input_preview = uploaded.name
                    
                    # Auto-cleanup uploaded file after processing
                    try:
                        tmp_path.unlink()
                        print(f"[INFO] Cleaned up uploaded file: {uploaded.name}", flush=True)
                    except Exception as e:
                        print(f"[WARNING] Failed to cleanup file {uploaded.name}: {e}", flush=True)
                else:
                    input_preview = (text or '').strip()[:120]

                print(f"[INFO] About to call _summarize_text with text length: {len(text)}", flush=True)
                no_facts = form.cleaned_data.get('no_facts', False)
                
                # Generate task ID for progress tracking
                task_id = str(uuid.uuid4())
                update_progress(task_id, 0, "Starting summarization...")
                
                short_summary, facts, _ = _summarize_text(text, no_facts=no_facts, task_id=task_id)
                
                # Clean up progress tracking
                clear_progress(task_id)

                if request.user.is_authenticated:
                    SummaryRecord.objects.create(
                        user=request.user,
                        input_type=input_type,
                        input_preview=input_preview,
                        short_summary=short_summary,
                        extended_summary="",  # Extended summary removed
                        facts="\n".join(facts),
                    )

                result = {
                    'short_summary': short_summary,
                    'facts': facts if not no_facts else [],
                }
            except Exception as exc:
                print(f"[ERROR] Error during summarization: {exc}", flush=True)
                import traceback
                traceback.print_exc()
                import sys
                sys.stdout.flush()
                messages.error(request, f'Error: {exc}')
                result = {
                    'error': str(exc),
                    'debug_info': traceback.format_exc()
                }
    else:
        print("[INFO] Processing GET request", flush=True)
        form = TextSummarizeForm()

    print("[INFO] Rendering template", flush=True)
    from django.utils import translation
    import time
    return render(request, 'home.html', {
        'form': form, 
        'result': result,
        'LANGUAGE_CODE': translation.get_language(),
        'timestamp': int(time.time())
    })


@login_required
def history(request: HttpRequest) -> HttpResponse:
    records = SummaryRecord.objects.filter(user=request.user)
    import time
    return render(request, 'history.html', {
        'records': records,
        'timestamp': int(time.time())
    })


@require_http_methods(["GET", "POST"])
@csrf_protect
def register(request: HttpRequest) -> HttpResponse:
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = RegisterForm()
    import time
    return render(request, 'register.html', {
        'form': form,
        'timestamp': int(time.time())
    })


@require_http_methods(["GET", "POST"])
@csrf_protect
def login_view(request: HttpRequest) -> HttpResponse:
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
    else:
        form = LoginForm()
    import time
    return render(request, 'login.html', {
        'form': form,
        'timestamp': int(time.time())
    })




