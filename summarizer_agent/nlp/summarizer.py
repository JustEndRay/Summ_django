from __future__ import annotations

# Обгортка над попередньо натренованою моделлю сумаризації (Hugging Face)
from typing import List, Optional
import warnings
import logging
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Приглушуємо попередження transformers про послідовну обробку на GPU
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*pipelines sequentially on GPU.*")


def _mask_numbers(text: str) -> tuple[str, dict]:
    """
    Замінює всі роки та числа на маски, щоб модель не спотворювала їх.
    Повертає змінений текст та словник {mask: original_number}.
    """
    mapping = {}
    def repl(match):
        num = match.group(0)
        key = f"<NUM_{len(mapping)}>"
        mapping[key] = num
        return key

    masked = re.sub(r"\b\d{3,4}\b", repl, text)
    return masked, mapping


def _restore_numbers(text: str, mapping: dict) -> str:
    """Відновлює оригінальні числа замість масок."""
    for key, val in mapping.items():
        text = text.replace(key, val)
    return text


def _preserve_facts(original_text: str, summary: str) -> str:
    """
    Повертає summary, очищений від вигаданих дат і чисел,
    які не зустрічалися в оригінальному тексті.
    """
    orig_years = set(re.findall(r"\b(1\d{3}|20\d{2})\b", original_text))
    sum_years = re.findall(r"\b(1\d{3}|20\d{2})\b", summary)
    for y in sum_years:
        if y not in orig_years:
            summary = summary.replace(y, "")
    
    # Прибираємо з summary одиночні дати/роки, якщо вони відсутні в оригіналі
    orig_nums = set(re.findall(r"\b\d+\b", original_text))
    for n in re.findall(r"\b\d+\b", summary):
        if n not in orig_nums and len(n) > 2:
            summary = summary.replace(n, "")
    
    return re.sub(r'\s{2,}', ' ', summary).strip()


class TextSummarizer:
    """Ініціалізує пайплайн сумаризації і надає метод для списку чанків."""

    def __init__(self, model_name: str = "facebook/bart-large-cnn", device_arg: str = "auto") -> None:
        self.model_name = model_name
        print(f"[DEBUG] Initializing model: {model_name}", flush=True)
        
        # Обробляємо різні типи моделей
        if "mt5" in model_name.lower():
            # mT5 модель - використовуємо повільний токенайзер для уникнення проблем з SentencePiece
            print(f"[DEBUG] Loading mT5 tokenizer with use_fast=False", flush=True)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                self.model_type = "mt5"
                print(f"[INFO] mT5 model detected, using slow tokenizer", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed to load mT5 tokenizer: {e}", flush=True)
                # Fallback до BART
                print(f"[INFO] Falling back to BART tokenizer", flush=True)
                self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", use_fast=True)
                self.model_type = "bart"
        elif "mbart" in model_name.lower():
            # mBART модель - використовуємо швидкий токенайзер з мовними кодами
            print(f"[DEBUG] Loading mBART tokenizer with use_fast=True", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model_type = "mbart"
            
            # Встановлюємо мовні коди для mBART
            if "many-to-many-mmt" in model_name.lower():
                self.source_lang = "uk_XX"  # Українська для mBART
                self.target_lang = "uk_XX"  # Українська для mBART
                # Встановлюємо явні мовні коди в токенайзері
                self.tokenizer.src_lang = "uk_XX"
                self.tokenizer.tgt_lang = "uk_XX"
                print(f"[INFO] mBART many-to-many model detected, using Ukrainian language codes", flush=True)
            else:
                self.source_lang = None
                self.target_lang = None
                print(f"[INFO] mBART model detected, using fast tokenizer", flush=True)
        else:
            # BART модель - використовуємо швидкий токенайзер
            print(f"[DEBUG] Loading BART tokenizer with use_fast=True", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model_type = "bart"
            print(f"[INFO] BART model detected, using fast tokenizer", flush=True)
        
        # Встановлюємо мовно-специфічні конфігурації (якщо ще не встановлені)
        if not hasattr(self, 'source_lang'):
            self.source_lang = None
        if not hasattr(self, 'target_lang'):
            self.target_lang = None

        # Використовуємо GPU (CUDA вже перевірено при імпорті модуля)
        if device_arg == "cuda:0":
            # transformers pipeline приймає int індекс пристрою для CUDA; віддаємо перевагу цілому числу для CUDA
            device = 0
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            device_props = torch.cuda.get_device_properties(current_device)
            print(f"[INFO] Using GPU: {device_name}")
            print(f"[INFO] VRAM: {device_props.total_memory / 1024**3:.1f} GB")
        elif device_arg == "cpu":
            device = "cpu"
            print("[INFO] Using CPU")
        else:
            # За замовчуванням використовуємо GPU
            device = 0
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            device_props = torch.cuda.get_device_properties(current_device)
            print(f"[INFO] Using GPU: {device_name}")
            print(f"[INFO] VRAM: {device_props.total_memory / 1024**3:.1f} GB")

        # Завантаження моделі з оптимізаціями
        print(f"[DEBUG] Loading model: {model_name}", flush=True)
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print(f"[INFO] Model loaded successfully: {model_name}", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to load model {model_name}: {e}", flush=True)
            if "mt5" in model_name.lower():
                # Fallback до BART для mT5
                print(f"[INFO] Falling back to BART model", flush=True)
                model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
                self.model_name = "facebook/bart-large-cnn"
                self.model_type = "bart"
            else:
                raise
        
        if device != "cpu" and torch.cuda.is_available():
            # GPU оптимізації
            model.gradient_checkpointing_enable()  # Зменшує використання VRAM
            print(f"[INFO] GPU optimizations enabled: gradient checkpointing", flush=True)
        
        # Використовуємо пайплайн сумаризації для всіх моделей
        self.summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=self.tokenizer,
            framework="pt",
            device=device,
        )

    def summarize_chunks(self, chunks: List[str], min_words: int, max_words: int, device: str = "auto", progress_callback=None, language: str = "en") -> List[str]:
        """Повертає список резюме для кожного вхідного чанка.

        Замість жорстких обмежень max_length (які можуть обрізати речення),
        генеруємо за допомогою max_new_tokens/min_new_tokens і потім
        акуратно обрізаємо по межі речення у межах ліміту слів.
        """
        if not chunks:
            return []
        
        # Визначаємо чи використовується CPU
        is_cpu = device == "cpu" or (device == "auto" and (
            not hasattr(self.summarizer.model, 'device') or 
            str(self.summarizer.model.device) == 'cpu' or
            'cpu' in str(self.summarizer.model.device)
        ))
        
        # Евристика токенів: для дуже коротких лімітів зменшуємо стохастику і простір генерації
        very_short_word_threshold = 90
        if max_words <= very_short_word_threshold:
            # Більш консервативна генерація для зменшення галюцинацій
            min_new_tokens = max(6, int(min_words * 1.1))
            max_new_tokens = max(min_new_tokens + 16, int(max_words * 1.2))
            gen_kwargs = dict(
                early_stopping=True,
                no_repeat_ngram_size=3,
                num_beams=4 if is_cpu else 6,  # Менше beams для CPU
                do_sample=False,  # вимикаємо семплінг на дуже коротких виходах
                repetition_penalty=1.15,
            )
        else:
            # Покращений режим для кращої якості суммаризації
            min_new_tokens = max(12, int(min_words * 1.2))
            max_new_tokens = max(min_new_tokens + 24, int(max_words * 1.4))
            
            if is_cpu:
                # Оптимізовані параметри для CPU
                gen_kwargs = dict(
                    early_stopping=True,
                    no_repeat_ngram_size=2,  # Менше для швидкості
                    num_beams=3,  # Менше beams для CPU
                    temperature=0.7,  # Трохи вище для різноманітності
                    do_sample=True,
                    top_p=0.9,  # Вище для швидкості
                    top_k=50,  # Вище для швидкості
                    repetition_penalty=1.1,  # Менше для швидкості
                    length_penalty=1.0,
                )
            else:
                # Детерміновані параметри для mT5 (жорстке запобігання галюцинаціям)
                gen_kwargs = dict(
                    early_stopping=True,
                    num_beams=6,  # Збільшено для кращої якості
                    no_repeat_ngram_size=4,  # Збільшено для запобігання повторенню
                    repetition_penalty=1.7,  # Максимально збільшено проти повторень
                    length_penalty=1.0,  # Нейтральна довжина
                    do_sample=False,  # Детермінований вихід
                    temperature=0.6,  # Додано для детермінованого декодування
                )

        results = []
        
        # Batch обробка для уникнення попереджень transformers про логування
        batch_size = 4 if not is_cpu else 2  # Менші батчі для CPU
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for batch_idx, batch_start in enumerate(range(0, len(chunks), batch_size)):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            try:
                # Оновлюємо прогрес
                if progress_callback:
                    progress_percent = 50 + int((batch_idx / total_batches) * 40)  # 50-90% діапазон
                    progress_callback(progress_percent, f"Processing batch {batch_idx + 1}/{total_batches}...")
                
                # Очищення GPU кешу перед обробкою батчу
                if device != "cpu":
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Маскуємо числа перед обробкою та додаємо anchor-guided summarization
                masked_batch_chunks, num_maps = [], []
                for i, chunk in enumerate(batch_chunks):
                    # Маскуємо числа для запобігання галюцинаціям
                    masked_chunk, num_mapping = _mask_numbers(chunk)
                    masked_batch_chunks.append(masked_chunk)
                    num_maps.append(num_mapping)
                    
                    # Вставляємо ключові факти з оригіналу (anchor-guided summarization)
                    anchors = re.findall(r"\b(19\d{2}|20\d{2})\b", masked_chunk)
                    if anchors:
                        masked_chunk = masked_chunk + "\n\nKeep original years: " + ", ".join(anchors)
                    
                    chunk_tokens = self.tokenizer.encode(masked_chunk, add_special_tokens=False)
                    if len(chunk_tokens) > 1024:  # BART max input length
                        print(f"[WARNING] Chunk {batch_start + i + 1}/{len(chunks)} too long ({len(chunk_tokens)} tokens), truncating to 1024")
                        masked_chunk = self.tokenizer.decode(chunk_tokens[:1024], skip_special_tokens=True)
                    masked_batch_chunks[i] = masked_chunk
                
                # Use summarization pipeline with batch processing
                if self.model_type == "mt5":
                    # mT5 model - add language prefix for better results
                    prefixed_chunks = []
                    for chunk in masked_batch_chunks:
                        if language == "uk":
                            prefixed_chunk = f"summarize Ukrainian: {chunk}"
                        else:
                            prefixed_chunk = f"summarize English: {chunk}"
                        prefixed_chunks.append(prefixed_chunk)
                    
                    batch_results = self.summarizer(
                        prefixed_chunks,
                        min_new_tokens=min_new_tokens,
                        max_new_tokens=max_new_tokens,
                        **gen_kwargs,
                    )
                elif self.model_type == "mbart":
                    # mBART model - don't use language codes for summarization pipeline
                    # Language codes are not supported in summarization mode
                    batch_results = self.summarizer(
                        masked_batch_chunks,
                        min_new_tokens=min_new_tokens,
                        max_new_tokens=max_new_tokens,
                        **gen_kwargs,
                    )
                else:
                    # BART model - use standard summarization
                    batch_results = self.summarizer(
                        masked_batch_chunks,
                        min_new_tokens=min_new_tokens,
                        max_new_tokens=max_new_tokens,
                        **gen_kwargs,
                    )
                
                # Process batch results
                for i, result in enumerate(batch_results):
                    try:
                        raw = result["summary_text"].strip()
                        # Відновлюємо оригінальні числа замість масок
                        raw = _restore_numbers(raw, num_maps[i])
                        # Застосовуємо _preserve_facts для видалення вигаданих дат і чисел
                        raw = _preserve_facts(batch_chunks[i], raw)
                        results.append(_trim_to_sentence_window(raw, min_words=min_words, max_words=max_words))
                    except (KeyError, TypeError) as e:
                        print(f"[ERROR] Invalid result format for chunk {batch_start + i + 1}: {e}")
                        results.append("")
                        
            except (RuntimeError, IndexError) as e:
                if "CUDA" in str(e) and not is_cpu:
                    # GPU required; do not fallback to CPU, propagate error
                    raise
                elif "index out of range" in str(e).lower():
                    print(f"[WARNING] Token index error in batch {batch_start}-{batch_end}, processing individually")
                    # Fallback to individual processing for this batch
                    for i, chunk in enumerate(batch_chunks):
                        try:
                            # Маскуємо числа для запобігання галюцинаціям
                            masked_chunk, num_mapping = _mask_numbers(chunk)
                            
                            # Додаємо anchor-guided summarization
                            anchors = re.findall(r"\b(19\d{2}|20\d{2})\b", masked_chunk)
                            if anchors:
                                masked_chunk = masked_chunk + "\n\nKeep original years: " + ", ".join(anchors)
                            
                            chunk_tokens = self.tokenizer.encode(masked_chunk, add_special_tokens=False)
                            if len(chunk_tokens) > 512:  # Більш агресивне обрізання
                                masked_chunk = self.tokenizer.decode(chunk_tokens[:512], skip_special_tokens=True)
                            
                            # Використовуємо специфічну для моделі обробку
                            if self.model_type == "mt5":
                                # mT5 модель - додаємо мовний префікс
                                if language == "uk":
                                    prefixed_chunk = f"summarize Ukrainian: {masked_chunk}"
                                else:
                                    prefixed_chunk = f"summarize English: {masked_chunk}"
                                
                                raw = self.summarizer(
                                    prefixed_chunk,
                                    min_new_tokens=min_new_tokens,
                                    max_new_tokens=max_new_tokens,
                                    **gen_kwargs,
                                )[0]["summary_text"].strip()
                                # Відновлюємо оригінальні числа замість масок
                                raw = _restore_numbers(raw, num_mapping)
                                # Застосовуємо _preserve_facts для видалення вигаданих дат і чисел
                                raw = _preserve_facts(batch_chunks[i], raw)
                            elif self.model_type == "mbart":
                                # mBART model - don't use language codes for summarization pipeline
                                # Language codes are not supported in summarization mode
                                raw = self.summarizer(
                                    masked_chunk,
                                    min_new_tokens=min_new_tokens,
                                    max_new_tokens=max_new_tokens,
                                    **gen_kwargs,
                                )[0]["summary_text"].strip()
                                # Відновлюємо оригінальні числа замість масок
                                raw = _restore_numbers(raw, num_mapping)
                                # Застосовуємо _preserve_facts для видалення вигаданих дат і чисел
                                raw = _preserve_facts(batch_chunks[i], raw)
                            else:
                                # BART model - use standard summarization
                                raw = self.summarizer(
                                    masked_chunk,
                                    min_new_tokens=min_new_tokens,
                                    max_new_tokens=max_new_tokens,
                                    **gen_kwargs,
                                )[0]["summary_text"].strip()
                                # Відновлюємо оригінальні числа замість масок
                                raw = _restore_numbers(raw, num_mapping)
                                # Застосовуємо _preserve_facts для видалення вигаданих дат і чисел
                                raw = _preserve_facts(batch_chunks[i], raw)
                            results.append(_trim_to_sentence_window(raw, min_words=min_words, max_words=max_words))
                        except Exception as individual_error:
                            print(f"[ERROR] Individual chunk {batch_start + i + 1} failed: {individual_error}")
                            results.append("")
                else:
                    print(f"[ERROR] Batch processing error: {e}")
                    # Add empty results for failed batch
                    results.extend([""] * len(batch_chunks))
        return results


def _trim_to_sentence_window(text: str, min_words: int, max_words: int) -> str:
    """Робить обрізку за повними реченнями в межах word-ліміту.

    Логіка:
    1) Якщо текст уже влізає в max_words — повертаємо як є.
    2) Розбиваємо на речення за розділовими знаками [.?!…].
    3) Додаємо повні речення, поки сумарна кількість слів ≤ max_words.
    4) Якщо перше ж речення довше за max_words — повертаємо його цілим і додаємо «…».
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    # Грубе розбиття на речення з збереженням знаків
    import re
    sentences = re.split(r"(?<=[\.\!\?…])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    collected: list[str] = []
    total_words = 0
    for sent in sentences:
        sent_words = len(sent.split())
        if not collected and sent_words > max_words:
            # Перше речення вже довше за ліміт — повертаємо його повністю з «…»
            return sent.rstrip() + " …"
        if total_words + sent_words <= max_words:
            collected.append(sent)
            total_words += sent_words
        else:
            break

    if collected:
        return " ".join(collected).strip()

    # Fallback: якщо не вдалося — класичне відсікання по словах із еліпсисом
    slice_text = " ".join(words[:max_words]).rstrip()
    return slice_text + " …"


