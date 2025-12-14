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
    Повертає summary, очищений від вигаданих дат, чисел та метаданих,
    які не зустрічалися в оригінальному тексті.
    """
    # Видаляємо вигадані роки
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
    
    # Видаляємо вигадані метадані про тип тексту, автора тощо
    summary = _remove_hallucinated_metadata(original_text, summary)
    
    return re.sub(r'\s{2,}', ' ', summary).strip()


def _remove_hallucinated_metadata(original_text: str, summary: str) -> str:
    """
    Видаляє з резюме вигадані метадані, які не згадуються в оригінальному тексті.
    Наприклад: "філософ", "репортаж", "автор", "журналіст" тощо.
    """
    original_lower = original_text.lower()
    summary_lower = summary.lower()
    
    # Список типових галлюцинацій про тип тексту та автора
    hallucination_patterns = [
        # Українські типи текстів
        (r'\b(репортаж|репортер|журналіст|журналістика)\b', ['репортаж', 'репортер', 'журналіст', 'журналістика']),
        (r'\b(стаття|статті)\s+(написав|написала|автор)\b', ['стаття', 'написав', 'написала', 'автор']),
        (r'\b(філософ|філософія|філософський)\b', ['філософ', 'філософія', 'філософський']),
        (r'\b(автор|письменник|поет)\s+(?:цього|тексту|статті)\b', ['автор', 'письменник', 'поет']),
        (r'\b(це|цей)\s+(?:репортаж|стаття|нарис|есе)\b', ['репортаж', 'стаття', 'нарис', 'есе']),
        # Англійські еквіваленти - розширений список
        (r'\b(report|reporter|journalist|journalism|news|newspaper)\b', ['report', 'reporter', 'journalist', 'journalism', 'news', 'newspaper']),
        (r'\b(article|essay|piece|text)\s+(?:written|authored|created|produced)\s+(?:by|from)\b', ['article', 'essay', 'piece', 'text', 'written', 'authored', 'created', 'produced']),
        (r'\b(philosopher|philosophy|philosophical|thinker)\b', ['philosopher', 'philosophy', 'philosophical', 'thinker']),
        (r'\b(author|writer|poet|novelist|columnist)\s+(?:of\s+)?(?:this|the|a)\s+(?:text|article|essay|piece|report|story)\b', ['author', 'writer', 'poet', 'novelist', 'columnist']),
        (r'\b(this|the|a)\s+(?:report|article|essay|piece|story|text|document)\s+(?:is|was|discusses|explores|examines)\b', ['report', 'article', 'essay', 'piece', 'story', 'text', 'document']),
        (r'\b(according\s+to\s+the\s+author|the\s+author\s+(?:states|claims|argues|suggests))\b', ['author', 'states', 'claims', 'argues', 'suggests']),
        (r'\b(in\s+this\s+(?:article|essay|report|piece|text|story))\b', ['article', 'essay', 'report', 'piece', 'text', 'story']),
        (r'\b(the\s+(?:article|essay|report|piece|text|story)\s+(?:concludes|ends|finishes|summarizes))\b', ['article', 'essay', 'report', 'piece', 'text', 'story', 'concludes', 'ends', 'finishes', 'summarizes']),
        (r'\b(this\s+(?:piece|work|text|document)\s+(?:by|from))\b', ['piece', 'work', 'text', 'document']),
    ]
    
    # Перевіряємо кожен паттерн
    # Використовуємо зворотний порядок, щоб не змінювати індекси під час видалення
    matches_to_remove = []
    
    for pattern, keywords in hallucination_patterns:
        matches = list(re.finditer(pattern, summary_lower))
        for match in matches:
            # Перевіряємо, чи всі ключові слова присутні в оригіналі
            found_in_original = any(keyword in original_lower for keyword in keywords)
            
            if not found_in_original:
                matches_to_remove.append((match.start(), match.end(), pattern, keywords))
    
    # Видаляємо знайдені галлюцинації в зворотному порядку
    for start, end, pattern, keywords in sorted(matches_to_remove, reverse=True):
        # Знаходимо повне речення, що містить цю фразу
        sentence_start = summary.rfind('.', 0, start)
        sentence_start = sentence_start + 1 if sentence_start >= 0 else 0
        sentence_end = summary.find('.', end)
        sentence_end = sentence_end if sentence_end >= 0 else len(summary)
        
        # Перевіряємо, чи речення не містить іншої важливої інформації
        sentence = summary[sentence_start:sentence_end].strip()
        sentence_words = sentence.split()
        
        # Більш агресивне видалення: якщо речення коротке (до 20 слів) і містить галлюцинацію - видаляємо його повністю
        if len(sentence_words) <= 20:
            print(f"[INFO] Removing sentence with hallucination ({len(sentence_words)} words): {sentence[:100]}...", flush=True)
            summary = summary[:sentence_start].rstrip() + summary[sentence_end:].lstrip()
        else:
            # Інакше видаляємо тільки проблемну фразу та навколишні слова
            # Розширюємо контекст для видалення
            context_start = max(0, start - 10)
            context_end = min(len(summary), end + 10)
            print(f"[INFO] Removing hallucinated phrase with context: {summary[context_start:context_end]}", flush=True)
            summary = summary[:context_start].rstrip() + summary[context_end:].lstrip()
        
        # Оновлюємо summary_lower після змін
        summary_lower = summary.lower()
    
    return summary


def _verify_summary_against_original(original_text: str, summary: str) -> str:
    """
    Перевіряє відповідність резюме оригінальному тексту.
    Видаляє ТІЛЬКИ явні галлюцинації, зберігаючи весь валидний контент.
    """
    if not summary or not original_text:
        return summary
    
    # Спочатку застосовуємо перевірку на типові галлюцинації
    summary = _remove_common_hallucination_phrases(original_text, summary)
    
    # Розбиваємо на речення
    sentences = re.split(r'[.!?…]+', summary)
    original_lower = original_text.lower()
    
    # Визначаємо мову
    is_ukrainian = bool(re.search(r'[а-яєіїґ]', original_lower))
    
    # Витягуємо ключові слова з оригіналу (тільки важливі слова)
    if is_ukrainian:
        original_words = set(re.findall(r'\b[а-яєіїґ]{5,}\b', original_lower))
    else:
        original_words = set(re.findall(r'\b[a-z]{5,}\b', original_lower))
    
    # Розширений список стоп-слів
    stop_words = {
        'що', 'який', 'яка', 'яке', 'які', 'для', 'від', 'до', 'на', 'в', 'з', 'по', 'про', 'при', 'без', 'над', 'під',
        'між', 'через', 'після', 'перед', 'біля', 'коло', 'навколо', 'всередині', 'зовні', 'поза', 'окрім', 'крім',
        'замість', 'завдяки', 'внаслідок', 'згідно', 'відповідно', 'щодо', 'стосовно', 'відносно', 'порівняно',
        'все', 'всі', 'вся', 'кожен', 'кожна', 'кожне', 'кожні', 'якийсь', 'якась', 'якесь', 'якісь',
        'деякий', 'деяка', 'деяке', 'деякі', 'інший', 'інша', 'інше', 'інші', 'той', 'та', 'те', 'ті',
        'цей', 'ця', 'це', 'ці', 'сам', 'сама', 'саме', 'самі', 'весь', 'вся', 'все', 'всі',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
        'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'can', 'could', 'should', 'would', 'may', 'might', 'must', 'will', 'shall', 'not', 'no', 'yes'
    }
    original_words = {w for w in original_words if w not in stop_words}
    
    # Перевіряємо кожне речення - видаляємо ТІЛЬКИ явні галлюцинації
    valid_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence.split()) < 3:
            continue
        
        # Перевіряємо наявність явних галлюцинацій про автора/тип тексту
        sentence_lower = sentence.lower()
        has_hallucination = False
        
        # Перевірка на явні галлюцинації
        # Для англійської - більш агресивна перевірка
        if not is_ukrainian:
            hallucination_indicators = [
                r'\b(?:author|writer|philosopher|journalist|reporter|columnist|editor)\s+(?:states|claims|argues|suggests|concludes|notes|observes|says|writes|explains|describes|discusses)\b',
                r'\b(?:according\s+to|as\s+stated\s+by|as\s+mentioned\s+by|as\s+reported\s+by)\s+(?:the\s+)?(?:author|writer|philosopher|journalist|reporter)\b',
                r'\b(?:this|the)\s+(?:article|essay|report|piece|text|story|document|series|column|profile)\s+(?:is|was)\s+(?:written|authored|created|produced|published)\s+by\b',
                r'\b(?:in\s+this|this)\s+(?:article|essay|report|piece|text|story|document|series|column|profile)\b',
                r'\b(?:the\s+)?(?:article|essay|report|piece|text|story|document)\s+(?:concludes|ends|finishes|summarizes|discusses|explores|examines)\b',
                r'\b(?:imagine|suppose|consider|think\s+about|picture)\s+(?:that|if|a|an)\b',  # Заборонені вступні фрази
            ]
        else:
            hallucination_indicators = [
                r'\b(?:author|writer|philosopher|journalist|reporter)\s+(?:states|claims|argues|suggests|concludes|notes|observes|says|writes)\b',
                r'\b(?:according\s+to|as\s+stated\s+by|as\s+mentioned\s+by)\s+(?:the\s+)?(?:author|writer|philosopher|journalist)\b',
                r'\b(?:this|the)\s+(?:article|essay|report|piece|text|story|document)\s+(?:is|was)\s+(?:written|authored|created|produced)\s+by\b',
                r'\b(?:автор|письменник|філософ|журналіст|репортер)\s+(?:стверджує|заявляє|аргументує|припускає|завершує|зазначає|каже|пише)\b',
                r'\b(?:згідно|як\s+зазначено|як\s+згадано)\s+(?:з\s+)?(?:автором|письменником|філософом|журналістом)\b',
                r'\b(?:уявіть|припустімо|припустимо|припустіть|подумаємо)\s+(?:що|ніби)\b',  # Заборонені вступні фрази укр
            ]
        
        for pattern in hallucination_indicators:
            if re.search(pattern, sentence_lower):
                # Перевіряємо, чи ця фраза є в оригіналі
                if pattern not in original_lower and not any(keyword in original_lower for keyword in ['author', 'writer', 'автор', 'письменник']):
                    print(f"[INFO] Removing explicit hallucination: {sentence[:100]}...", flush=True)
                    has_hallucination = True
                    break
        
        if has_hallucination:
            continue

        # Додаткова агресивна перевірка на ролі/статуси (journalist/author/etc.)
        if not is_ukrainian:
            role_patterns = [
                r'\b(journalist|reporter|columnist|editor|author|writer|press|media)\b',
                r'\b(article|essay|report|story|piece|column|profile)\b',
                r'\baccording to\b',
                r'\bin this\b',
            ]
        else:
            role_patterns = [
                r'\b(журналіст|репортер|кореспондент|автор|письменник|оглядач)\b',
                r'\bстаття\b|\bрепортаж\b|\bнарис\b|\бесе\b',
                r'\bзгідно\b',
                r'\bу цьому\b',
            ]
        role_hallucination = False
        for rp in role_patterns:
            if re.search(rp, sentence_lower) and rp not in original_lower:
                role_hallucination = True
                break
        if role_hallucination:
            continue
        
        # Якщо речення коротке (менше 6 слів) - завжди залишаємо його
        if len(sentence.split()) < 6:
            valid_sentences.append(sentence)
            continue
        
        # Для довших речень перевіряємо наявність ключових слів
        if is_ukrainian:
            sentence_keywords = set(re.findall(r'\b[а-яєіїґ]{5,}\b', sentence_lower))
        else:
            sentence_keywords = set(re.findall(r'\b[a-z]{5,}\b', sentence_lower))
        
        sentence_keywords = {w for w in sentence_keywords if w not in stop_words}
        
        # Якщо є хоча б одне ключове слово з оригіналу - залишаємо речення
        if sentence_keywords and (sentence_keywords & original_words):
            valid_sentences.append(sentence)
            continue
        
        # Якщо немає ключових слів, але речення не містить явних галлюцинацій - залишаємо
        # (може бути перефразуванням)
        valid_sentences.append(sentence)
    
    # Збираємо речення назад
    if valid_sentences:
        result = '. '.join(valid_sentences) + '.' if not valid_sentences[-1].endswith('.') else '. '.join(valid_sentences)
        print(f"[INFO] Summary verification: {len(sentences)} sentences -> {len(valid_sentences)} valid sentences", flush=True)
        return result
    else:
        # Якщо всі речення видалені, повертаємо оригінал
        print(f"[WARNING] All sentences removed, returning original summary", flush=True)
        return summary


def _remove_common_hallucination_phrases(original_text: str, summary: str) -> str:
    """
    Видаляє ТІЛЬКИ явні фрази-галлюцинації про автора/тип тексту.
    """
    original_lower = original_text.lower()
    summary_lower = summary.lower()
    
    # Перевіряємо, чи в оригіналі є згадки про автора/тип тексту
    has_author_mention = any(word in original_lower for word in ['author', 'writer', 'автор', 'письменник', 'philosopher', 'філософ', 'journalist', 'журналіст'])
    has_text_type = any(word in original_lower for word in ['article', 'essay', 'report', 'стаття', 'репортаж', 'есе'])
    
    # Тільки явні фрази з автором/типом тексту, яких немає в оригіналі
    explicit_hallucinations = []
    
    if not has_author_mention:
        # Англійські фрази про автора
        explicit_hallucinations.extend([
            r'\b(?:according\s+to|as\s+stated\s+by|as\s+mentioned\s+by)\s+(?:the\s+)?(?:author|writer|philosopher|journalist|reporter)\b',
            r'\bthe\s+(?:author|writer|philosopher|journalist|reporter)\s+(?:states|claims|argues|suggests|concludes|notes|observes|says|writes)\s+(?:that|this|these)\b',
            r'\b(?:this|the)\s+(?:article|essay|report|piece|text|story|document)\s+(?:is|was)\s+(?:written|authored|created|produced)\s+by\s+(?:the\s+)?(?:author|writer|philosopher|journalist)\b',
        ])
        # Українські фрази про автора
        explicit_hallucinations.extend([
            r'\b(?:згідно|як\s+зазначено|як\s+згадано)\s+(?:з\s+)?(?:автором|письменником|філософом|журналістом|репортером)\b',
            r'\b(?:автор|письменник|філософ|журналіст|репортер)\s+(?:стверджує|заявляє|аргументує|припускає|завершує|зазначає|каже|пише)\s+(?:що|це|ці)\b',
        ])
    
    if not has_text_type:
        # Фрази про тип тексту
        explicit_hallucinations.extend([
            r'\b(?:this|the)\s+(?:article|essay|report|piece|text|story|document)\s+(?:is|was)\s+(?:a|an)\s+(?:report|article|essay|piece|story|document)\b',
            r'\b(?:це|цей)\s+(?:стаття|репортаж|нарис|есе|текст)\s+(?:є|був)\s+(?:статтею|репортажем|нарисом|есе|текстом)\b',
        ])
    
    # Видаляємо тільки явні галлюцинації
    removed_count = 0
    for pattern in explicit_hallucinations:
        matches = list(re.finditer(pattern, summary_lower))
        for match in reversed(matches):
            matched_text = summary[match.start():match.end()]
            # Додаткова перевірка - фраза точно не в оригіналі
            if matched_text.lower() not in original_lower:
                print(f"[INFO] Removing explicit hallucination: {matched_text}", flush=True)
                # Видаляємо тільки фразу, не все речення
                summary = summary[:match.start()].rstrip() + ' ' + summary[match.end():].lstrip()
                summary_lower = summary.lower()
                removed_count += 1
                if removed_count >= 5:  # Обмеження на кількість видалень
                    break
        if removed_count >= 5:
            break
    
    return summary


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
            # Найбільш консервативна генерація: мінімум свободи, максимум анти-галюцинацій
            min_new_tokens = max(6, int(min_words * 1.0))
            max_new_tokens = max(min_new_tokens + 8, int(max_words * 1.05))
        else:
            # Строгий режим навіть для довших текстів: жорстко обмежуємо вихід
            min_new_tokens = max(12, int(min_words * 1.0))
            max_new_tokens = max(min_new_tokens + 10, int(max_words * 1.05))
        
        # Єдиний ультра-консервативний профіль генерації для всіх мов/пристроїв
        gen_kwargs = dict(
            early_stopping=True,
            num_beams=6,              # Обмежений пошук, щоб не вигадувати
            no_repeat_ngram_size=8,   # Заборона повторів і вигадування фраз
            repetition_penalty=3.0,   # Дуже високий штраф за повтори/галюцинації
            length_penalty=3.0,       # Максимальне стиснення, щоб не «розписувати»
            do_sample=False,          # Детерміновано — жодного семплінгу
        )

        results = []
        
        print(f"[INFO] Starting summarization of {len(chunks)} chunks", flush=True)
        
        # Batch обробка для уникнення попереджень transformers про логування
        batch_size = 4 if not is_cpu else 2  # Менші батчі для CPU
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"[INFO] Processing {total_batches} batches with batch_size={batch_size}", flush=True)
        
        for batch_idx, batch_start in enumerate(range(0, len(chunks), batch_size)):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            print(f"[INFO] Processing batch {batch_idx + 1}/{total_batches}: chunks {batch_start + 1}-{batch_end} of {len(chunks)}", flush=True)
            
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
                    # mT5 model - add explicit summarization instructions for better compression
                    prefixed_chunks = []
                    for chunk in masked_batch_chunks:
                        if language == "uk":
                            # Детальні інструкції для запобігання галлюцинаціям
                            prefixed_chunk = f"Підсумуй текст нижче. Не вигадуй нову інформацію, не додавай імена, факти чи організації, яких немає в тексті. Не створюй власних висновків. Стисни лише те, що присутнє в оригіналі. Збережи всі ключові тезиси, але будь лаконічним. Підсумок має складатися з 3–5 чітких речень. Не створюй вступів або фраз на кшталт \"Уявіть, що...\", якщо таких частин немає в оригінальному тексті. Починай сумаризацію тільки з фактів, які реально присутні в тексті. Не додавай загальних міркувань про тему, навіть якщо вони логічні. Використовуй виключно те, що явно є в тексті.\n\n{chunk}"
                        else:
                            # EXTREMELY STRICT instructions for English to enforce extractive-only summaries
                            prefixed_chunk = f"RULE 1 — EXTRACTIVE ONLY: Use ONLY verbatim fragments from the input. No paraphrasing, no rewriting.\nRULE 2 — FORBIDDEN CONTENT: Do NOT add names of people or organizations; no articles/reports/letters/columns/interviews; no invented metadata; no introductions; no conclusions or interpretations; no guides/instructions/recommendations; no information not explicitly in the input.\nRULE 3 — TRACEABILITY: Every sentence MUST exist in the source text. If it is not in the source, DO NOT output it.\nRULE 4 — NO STYLE INSERTION: No journalistic structure, no headings, no added context.\nRULE 5 — OUTPUT FORMAT: Output is ONLY a shortened version of the input by removing text. No additions.\n\nINPUT TEXT:\n{chunk}\n\nSUMMARY (extractive-only, verbatim fragments):"
                        prefixed_chunks.append(prefixed_chunk)
                    
                    batch_results = self.summarizer(
                        prefixed_chunks,
                        min_new_tokens=min_new_tokens,
                        max_new_tokens=max_new_tokens,
                        **gen_kwargs,
                    )
                elif self.model_type == "mbart":
                    # mBART model - add instructions in the text itself since it doesn't support prefixes
                    # Language codes are not supported in summarization mode
                    instructed_chunks = []
                    for chunk in masked_batch_chunks:
                        if language == "uk":
                            # Детальні інструкції для запобігання галлюцинаціям
                            instructed_chunk = f"Підсумуй текст нижче. Не вигадуй нову інформацію, не додавай імена, факти чи організації, яких немає в тексті. Не створюй власних висновків. Стисни лише те, що присутнє в оригіналі. Збережи всі ключові тезиси, але будь лаконічним. Підсумок має складатися з 3–5 чітких речень. Не створюй вступів або фраз на кшталт \"Уявіть, що...\", якщо таких частин немає в оригінальному тексті. Починай сумаризацію тільки з фактів, які реально присутні в тексті. Не додавай загальних міркувань про тему, навіть якщо вони логічні. Використовуй виключно те, що явно є в тексті.\n\n{chunk}"
                        else:
                            # EXTREMELY STRICT instructions for English to enforce extractive-only summaries
                            instructed_chunk = f"RULE 1 — EXTRACTIVE ONLY: Use ONLY verbatim fragments from the input. No paraphrasing, no rewriting.\nRULE 2 — FORBIDDEN CONTENT: Do NOT add names of people or organizations; no articles/reports/letters/columns/interviews; no invented metadata; no introductions; no conclusions or interpretations; no guides/instructions/recommendations; no information not explicitly in the input.\nRULE 3 — TRACEABILITY: Every sentence MUST exist in the source text. If it is not in the source, DO NOT output it.\nRULE 4 — NO STYLE INSERTION: No journalistic structure, no headings, no added context.\nRULE 5 — OUTPUT FORMAT: Output is ONLY a shortened version of the input by removing text. No additions.\n\nINPUT TEXT:\n{chunk}\n\nSUMMARY (extractive-only, verbatim fragments):"
                        instructed_chunks.append(instructed_chunk)
                    
                    batch_results = self.summarizer(
                        instructed_chunks,
                        min_new_tokens=min_new_tokens,
                        max_new_tokens=max_new_tokens,
                        **gen_kwargs,
                    )
                else:
                    # BART model - add instructions in the text itself
                    instructed_chunks = []
                    for chunk in masked_batch_chunks:
                        if language == "uk":
                            # Детальні інструкції для запобігання галлюцинаціям
                            instructed_chunk = f"Підсумуй текст нижче. Не вигадуй нову інформацію, не додавай імена, факти чи організації, яких немає в тексті. Не створюй власних висновків. Стисни лише те, що присутнє в оригіналі. Збережи всі ключові тезиси, але будь лаконічним. Підсумок має складатися з 3–5 чітких речень. Не створюй вступів або фраз на кшталт \"Уявіть, що...\", якщо таких частин немає в оригінальному тексті. Починай сумаризацію тільки з фактів, які реально присутні в тексті. Не додавай загальних міркувань про тему, навіть якщо вони логічні. Використовуй виключно те, що явно є в тексті.\n\n{chunk}"
                        else:
                            # EXTREMELY STRICT instructions for English to prevent hallucinations
                            instructed_chunk = f"TASK: Extract and compress ONLY facts explicitly stated in the text below.\n\nSTRICT RULES - VIOLATION FORBIDDEN:\n1. DO NOT add names, people, organizations, locations, dates, or any entities not in the source.\n2. DO NOT add opinions, examples, explanations, analogies, or interpretations.\n3. DO NOT add introductions, conclusions, context, background, or structure.\n4. DO NOT fill gaps or guess missing information.\n5. DO NOT paraphrase creatively - use source words when possible.\n6. DO NOT add journalistic style, narrative, or commentary.\n\nREQUIREMENTS:\n- Use ONLY information explicitly present in the source text.\n- Every sentence must be directly traceable to the original.\n- Summary: 3-5 sentences containing ONLY existing facts.\n- Compress without adding anything.\n\nSOURCE TEXT:\n{chunk}\n\nSUMMARY (ONLY existing facts, no additions):"
                        instructed_chunks.append(instructed_chunk)
                    
                    batch_results = self.summarizer(
                        instructed_chunks,
                        min_new_tokens=min_new_tokens,
                        max_new_tokens=max_new_tokens,
                        **gen_kwargs,
                    )
                
                # Process batch results
                print(f"[INFO] Received {len(batch_results)} results for batch {batch_idx + 1}", flush=True)
                for i, result in enumerate(batch_results):
                    try:
                        chunk_idx = batch_start + i + 1
                        raw = result["summary_text"].strip()
                        print(f"[INFO] Chunk {chunk_idx}/{len(chunks)} summary length: {len(raw)} chars, {len(raw.split())} words", flush=True)
                        # Відновлюємо оригінальні числа замість масок
                        raw = _restore_numbers(raw, num_maps[i])
                        # Застосовуємо _preserve_facts для видалення вигаданих дат і чисел
                        raw = _preserve_facts(batch_chunks[i], raw)
                        # Перевіряємо відповідність оригіналу через ключові слова
                        raw = _verify_summary_against_original(batch_chunks[i], raw)
                        trimmed = _trim_to_sentence_window(raw, min_words=min_words, max_words=max_words)
                        results.append(trimmed)
                        print(f"[INFO] Chunk {chunk_idx}/{len(chunks)} final summary: {len(trimmed)} chars, {len(trimmed.split())} words", flush=True)
                    except (KeyError, TypeError) as e:
                        print(f"[ERROR] Invalid result format for chunk {batch_start + i + 1}: {e}", flush=True)
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
                                # mT5 модель - додаємо детальні інструкції для запобігання галлюцинаціям
                                if language == "uk":
                                    prefixed_chunk = f"Підсумуй текст нижче. Не вигадуй нову інформацію, не додавай імена, факти чи організації, яких немає в тексті. Не створюй власних висновків. Стисни лише те, що присутнє в оригіналі. Збережи всі ключові тезиси, але будь лаконічним. Підсумок має складатися з 3–5 чітких речень. Не створюй вступів або фраз на кшталт \"Уявіть, що...\", якщо таких частин немає в оригінальному тексті. Починай сумаризацію тільки з фактів, які реально присутні в тексті. Не додавай загальних міркувань про тему, навіть якщо вони логічні. Використовуй виключно те, що явно є в тексті.\n\n{masked_chunk}"
                                else:
                                    # EXTREMELY STRICT instructions for English to enforce extractive-only summaries
                                    prefixed_chunk = f"RULE 1 — EXTRACTIVE ONLY: Use ONLY verbatim fragments from the input. No paraphrasing, no rewriting.\nRULE 2 — FORBIDDEN CONTENT: Do NOT add names of people or organizations; no articles/reports/letters/columns/interviews; no invented metadata; no introductions; no conclusions or interpretations; no guides/instructions/recommendations; no information not explicitly in the input.\nRULE 3 — TRACEABILITY: Every sentence MUST exist in the source text. If it is not in the source, DO NOT output it.\nRULE 4 — NO STYLE INSERTION: No journalistic structure, no headings, no added context.\nRULE 5 — OUTPUT FORMAT: Output is ONLY a shortened version of the input by removing text. No additions.\n\nINPUT TEXT:\n{masked_chunk}\n\nSUMMARY (extractive-only, verbatim fragments):"
                                
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
                                # Перевіряємо відповідність оригіналу через ключові слова
                                raw = _verify_summary_against_original(batch_chunks[i], raw)
                            elif self.model_type == "mbart":
                                # mBART model - add detailed instructions to prevent hallucinations
                                if language == "uk":
                                    instructed_chunk = f"Підсумуй текст нижче. Не вигадуй нову інформацію, не додавай імена, факти чи організації, яких немає в тексті. Не створюй власних висновків. Стисни лише те, що присутнє в оригіналі. Збережи всі ключові тезиси, але будь лаконічним. Підсумок має складатися з 3–5 чітких речень. Не створюй вступів або фраз на кшталт \"Уявіть, що...\", якщо таких частин немає в оригінальному тексті. Починай сумаризацію тільки з фактів, які реально присутні в тексті. Не додавай загальних міркувань про тему, навіть якщо вони логічні. Використовуй виключно те, що явно є в тексті.\n\n{masked_chunk}"
                                else:
                                    # EXTREMELY STRICT instructions for English to enforce extractive-only summaries
                                    instructed_chunk = f"RULE 1 — EXTRACTIVE ONLY: Use ONLY verbatim fragments from the input. No paraphrasing, no rewriting.\nRULE 2 — FORBIDDEN CONTENT: Do NOT add names of people or organizations; no articles/reports/letters/columns/interviews; no invented metadata; no introductions; no conclusions or interpretations; no guides/instructions/recommendations; no information not explicitly in the input.\nRULE 3 — TRACEABILITY: Every sentence MUST exist in the source text. If it is not in the source, DO NOT output it.\nRULE 4 — NO STYLE INSERTION: No journalistic structure, no headings, no added context.\nRULE 5 — OUTPUT FORMAT: Output is ONLY a shortened version of the input by removing text. No additions.\n\nINPUT TEXT:\n{masked_chunk}\n\nSUMMARY (extractive-only, verbatim fragments):"
                                
                                raw = self.summarizer(
                                    instructed_chunk,
                                    min_new_tokens=min_new_tokens,
                                    max_new_tokens=max_new_tokens,
                                    **gen_kwargs,
                                )[0]["summary_text"].strip()
                                # Відновлюємо оригінальні числа замість масок
                                raw = _restore_numbers(raw, num_mapping)
                                # Застосовуємо _preserve_facts для видалення вигаданих дат і чисел
                                raw = _preserve_facts(batch_chunks[i], raw)
                                # Перевіряємо відповідність оригіналу через ключові слова
                                raw = _verify_summary_against_original(batch_chunks[i], raw)
                            else:
                                # BART model - add detailed instructions to prevent hallucinations
                                if language == "uk":
                                    instructed_chunk = f"Підсумуй текст нижче. Не вигадуй нову інформацію, не додавай імена, факти чи організації, яких немає в тексті. Не створюй власних висновків. Стисни лише те, що присутнє в оригіналі. Збережи всі ключові тезиси, але будь лаконічним. Підсумок має складатися з 3–5 чітких речень. Не створюй вступів або фраз на кшталт \"Уявіть, що...\", якщо таких частин немає в оригінальному тексті. Починай сумаризацію тільки з фактів, які реально присутні в тексті. Не додавай загальних міркувань про тему, навіть якщо вони логічні. Використовуй виключно те, що явно є в тексті.\n\n{masked_chunk}"
                                else:
                                    # EXTREMELY STRICT instructions for English to prevent hallucinations
                                    instructed_chunk = f"TASK: Extract and compress ONLY facts explicitly stated in the text below.\n\nSTRICT RULES - VIOLATION FORBIDDEN:\n1. DO NOT add names, people, organizations, locations, dates, or any entities not in the source.\n2. DO NOT add opinions, examples, explanations, analogies, or interpretations.\n3. DO NOT add introductions, conclusions, context, background, or structure.\n4. DO NOT fill gaps or guess missing information.\n5. DO NOT paraphrase creatively - use source words when possible.\n6. DO NOT add journalistic style, narrative, or commentary.\n\nREQUIREMENTS:\n- Use ONLY information explicitly present in the source text.\n- Every sentence must be directly traceable to the original.\n- Summary: 3-5 sentences containing ONLY existing facts.\n- Compress without adding anything.\n\nSOURCE TEXT:\n{masked_chunk}\n\nSUMMARY (ONLY existing facts, no additions):"
                                
                                raw = self.summarizer(
                                    instructed_chunk,
                                    min_new_tokens=min_new_tokens,
                                    max_new_tokens=max_new_tokens,
                                    **gen_kwargs,
                                )[0]["summary_text"].strip()
                                # Відновлюємо оригінальні числа замість масок
                                raw = _restore_numbers(raw, num_mapping)
                                # Застосовуємо _preserve_facts для видалення вигаданих дат і чисел
                                raw = _preserve_facts(batch_chunks[i], raw)
                                # Перевіряємо відповідність оригіналу через ключові слова
                                raw = _verify_summary_against_original(batch_chunks[i], raw)
                            results.append(_trim_to_sentence_window(raw, min_words=min_words, max_words=max_words))
                        except Exception as individual_error:
                            print(f"[ERROR] Individual chunk {batch_start + i + 1} failed: {individual_error}")
                            results.append("")
                else:
                    print(f"[ERROR] Batch processing error: {e}")
                    # Add empty results for failed batch
                    results.extend([""] * len(batch_chunks))
        
        print(f"[INFO] Summarization completed: {len(results)} results from {len(chunks)} chunks", flush=True)
        print(f"[INFO] Results summary lengths: {[len(r.split()) for r in results]} words", flush=True)
        
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


