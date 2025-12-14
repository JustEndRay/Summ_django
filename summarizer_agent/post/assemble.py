from __future__ import annotations

# Покращений постпроцесинг: збірка резюме з ранжуванням та дедуплікацією
from typing import List, Tuple, Set
import re
from collections import Counter


def assemble_summary(
    chunk_summaries: List[str],
    facts: List[str],
    tables_note: str,
    target_min_words: int,
    target_max_words: int,
    original_text: str,
    only_facts: bool = False,
) -> Tuple[str, str]:
    """Формує коротке та розширене резюме з врахуванням фактів і приміток."""
    if only_facts:
        facts_text = "\n".join(facts) if facts else "Фактів не знайдено."
        return facts_text, facts_text

    # Обробляємо резюме чанків (тепер з індексами чанків)
    processed_summaries_with_chunks = _process_chunk_summaries(chunk_summaries, original_text)
    
    # Ранжуємо речення за важливістю з урахуванням позиції чанків
    ranked_sentences = _rank_sentences(processed_summaries_with_chunks, original_text, len(chunk_summaries))
    
    # Формуємо коротке резюме
    short_summary = _create_short_summary(
        ranked_sentences,
        target_min_words,
        target_max_words,
        original_text=original_text,
    )
    
    # Формуємо розширене резюме (використовуємо тільки речення без індексів)
    processed_summaries_only = [sentence for sentence, _ in processed_summaries_with_chunks]
    extended_summary = _create_extended_summary(processed_summaries_only, facts, tables_note, target_max_words)

    return short_summary, extended_summary


def _process_chunk_summaries(chunk_summaries: List[str], original_text: str) -> List[Tuple[str, int]]:
    """Обробляє резюме чанків: видаляє дублікати та покращує якість.
    
    Повертає список кортежів (речення, індекс_чанка) для відстеження позиції.
    """
    processed = []
    seen_sentences: Set[str] = set()
    
    for chunk_idx, summary in enumerate(chunk_summaries):
        if not summary.strip():
            continue
            
        # Розбиваємо на речення
        sentences = _split_into_sentences(summary)
        
        for sentence in sentences:
            # Нормалізуємо речення для порівняння
            cleaned = _cleanup_sentence(sentence)
            if _looks_like_boilerplate(cleaned):
                continue
            normalized = _normalize_sentence(cleaned)
            
            # Перевіряємо на дублікати і базову схожість з оригіналом
            # Зменшуємо поріг схожості для кращого покриття
            if (
                normalized not in seen_sentences
                and len(cleaned.split()) >= 3
                and _has_sufficient_overlap(cleaned, original_text, min_jaccard=0.1)
            ):
                seen_sentences.add(normalized)
                processed.append((cleaned, chunk_idx))  # Зберігаємо індекс чанка
    
    return processed


def _rank_sentences(sentences_with_chunks: List[Tuple[str, int]], original_text: str, total_chunks: int) -> List[Tuple[str, float]]:
    """Ранжує речення за важливістю на основі різних факторів.
    
    Враховує позицію чанка для кращого покриття всього тексту.
    """
    if not sentences_with_chunks:
        return []
    
    # Витягуємо ключові слова з оригінального тексту
    key_words = _extract_keywords(original_text)
    
    ranked = []
    for sentence, chunk_idx in sentences_with_chunks:
        # Обчислюємо позицію чанка (0.0 = початок, 1.0 = кінець)
        chunk_position = chunk_idx / max(1, total_chunks - 1) if total_chunks > 1 else 0.5
        
        score = _calculate_sentence_score(sentence, key_words, original_text, chunk_position)
        ranked.append((sentence, score))
    
    # Сортуємо за важливістю
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _calculate_sentence_score(sentence: str, key_words: Set[str], original_text: str, chunk_position: float = 0.5) -> float:
    """Обчислює важливість речення на основі різних факторів.
    
    Args:
        sentence: Речення для оцінки
        key_words: Набір ключових слів з оригінального тексту
        original_text: Оригінальний текст
        chunk_position: Позиція чанка (0.0 = початок, 1.0 = кінець)
    """
    score = 0.0
    words = sentence.lower().split()
    
    # Фактор 1: Кількість ключових слів (збільшено вагу)
    key_word_count = sum(1 for word in words if word in key_words)
    score += key_word_count * 2.0
    
    # Фактор 2: Позиція чанка (легкий бонус для початку, але не занадто великий)
    # Початок трохи важливіший, але не настільки, щоб ігнорувати середину та кінець
    if chunk_position < 0.3:  # Перші 30% тексту
        position_bonus = 0.5
    elif chunk_position < 0.7:  # Середина тексту
        position_bonus = 0.3  # Невеликий бонус для рівномірного покриття
    else:  # Останні 30% тексту
        position_bonus = 0.4  # Бонус для висновків
    score += position_bonus
    
    # Фактор 3: Довжина речення (оптимальна довжина)
    word_count = len(words)
    if 10 <= word_count <= 25:
        score += 1.0
    elif 5 <= word_count <= 35:
        score += 0.5
    else:
        score -= 0.5
    
    # Фактор 4: Наявність чисел та дат
    if re.search(r'\d+', sentence):
        score += 1.0
    
    # Фактор 5: Наявність імен власних
    if re.search(r'\b[А-ЯЄІЇҐA-Z][а-яєіїґa-z]+\b', sentence):
        score += 0.5
    
    return score


def _extract_keywords(text: str, top_n: int = 20) -> Set[str]:
    """Витягує ключові слова з тексту."""
    # Простий підхід: слова, що часто зустрічаються
    # Підтримуємо як українські, так і англійські тексти
    words = re.findall(r'\b[а-яєіїґa-z]{3,}\b', text.lower())
    word_freq = Counter(words)
    
    # Видаляємо стоп-слова (українські та англійські)
    stop_words = {
        # Українські стоп-слова
        'що', 'який', 'яка', 'яке', 'які', 'для', 'від', 'до', 'на', 'в', 'з', 'по', 'про', 'при', 'без', 'над', 'під', 'між', 'через', 'після', 'перед', 'під', 'над', 'біля', 'коло', 'навколо', 'всередині', 'зовні', 'поза', 'окрім', 'крім', 'замість', 'завдяки', 'через', 'внаслідок', 'згідно', 'відповідно', 'щодо', 'стосовно', 'відносно', 'порівняно', 'на відміну', 'на противагу', 'навпаки', 'натомість', 'зате', 'але', 'однак', 'проте', 'втім', 'все', 'всі', 'вся', 'все', 'кожен', 'кожна', 'кожне', 'кожні', 'будь', 'будь', 'будь', 'будь', 'якийсь', 'якась', 'якесь', 'якісь', 'деякий', 'деяка', 'деяке', 'деякі', 'інший', 'інша', 'інше', 'інші', 'той', 'та', 'те', 'ті', 'цей', 'ця', 'це', 'ці', 'сам', 'сама', 'саме', 'самі', 'весь', 'вся', 'все', 'всі', 'ніякий', 'ніяка', 'ніяке', 'ніякі', 'жоден', 'жодна', 'жодне', 'жодні',
        # Англійські стоп-слова
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'under', 'over', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'not', 'no', 'yes', 'all', 'any', 'some', 'many', 'much', 'few', 'little', 'more', 'most', 'other', 'another', 'each', 'every', 'both', 'either', 'neither', 'one', 'two', 'first', 'second', 'last', 'next', 'same', 'different', 'such', 'so', 'very', 'too', 'also', 'just', 'only', 'even', 'still', 'yet', 'already', 'again', 'here', 'there', 'where', 'when', 'why', 'how', 'what', 'who', 'which', 'whose', 'whom'
    }
    
    filtered_words = {word for word, freq in word_freq.most_common(top_n) if word not in stop_words and freq > 1}
    return filtered_words


def _create_short_summary(
    ranked_sentences: List[Tuple[str, float]],
    min_words: int,
    max_words: int,
    *,
    original_text: str,
) -> str:
    """Створює коротке резюме з акцентом на цілісність при малих лімітах.

    Політика для very-short режиму (≈ ≤ 90 слів ≈ 450 символів):
    - Не зшивати кілька речень з різних частин, якщо це не сусідні речення.
    - Віддати перевагу ОДНОМУ найкращому самодостатньому реченню, яке зустрічається в оригіналі
      (майже дослівно) або має високу лексичну схожість із ним.
    - Якщо немає зрозумілого самодостатнього речення в межах ліміту — повернути перше провідне
      (lead) речення, яке вміщується, без штучних доповнень.
    - Конкатенацію застосовувати лише для більшого ліміту.
    - Для кращого покриття вибирати речення з різних частин тексту (початок, середина, кінець).
    """
    # Для великих бюджетів формуємо з ранжованих речень для кращого покриття
    if max_words >= 120:
        # Використовуємо ранжовані речення з усіх чанків замість тільки оригінальних
        collected: List[str] = []
        total = 0
        seen_normalized = set()
        
        for sentence, score in ranked_sentences:
            sentence_clean = _cleanup_sentence(sentence)
            if _looks_like_boilerplate(sentence_clean):
                continue
            
            sentence_norm = _normalize_sentence(sentence_clean)
            if sentence_norm in seen_normalized:
                continue
            
            w = len(sentence_clean.split())
            if w > max_words:
                continue
            if total + w <= max_words:
                collected.append(sentence_clean)
                seen_normalized.add(sentence_norm)
                total += w
            if total >= min_words and len(collected) >= 2:  # Мінімум 2 речення для покриття
                break
        
        if collected:
            return " ".join(collected)
        
        # Fallback до оригінальних речень, якщо ранжовані не підійшли
        collected: List[str] = []
        total = 0
        for s in _best_original_sentences(original_text):
            if _looks_like_boilerplate(s):
                continue
            w = len(s.split())
            if w > max_words:
                continue
            if total + w <= max_words:
                collected.append(s)
                total += w
            if total >= min_words:
                break
        if collected:
            return " ".join(collected)

    if not ranked_sentences:
        # Фолбек: формуємо з оригінальних речень у межах бюджету
        collected: List[str] = []
        total = 0
        for s in _best_original_sentences(original_text):
            if _looks_like_boilerplate(s):
                continue
            w = len(s.split())
            if w > max_words:
                continue
            if total + w <= max_words:
                collected.append(s)
                total += w
            if total >= min_words:
                break
        return " ".join(collected)

    # Порог для "дуже короткого" резюме за кількістю слів
    very_short_word_threshold = 50

    # Передобчислення: набір нормалізованих речень оригіналу для перевірки "verbatim"/похідності
    original_sentences = _split_into_sentences(original_text)
    normalized_original = {_normalize_sentence(s): s for s in original_sentences}

    def is_self_contained(candidate: str) -> bool:
        words = candidate.split()
        return 8 <= len(words) <= max_words

    def appears_verbatim_or_close(candidate: str) -> bool:
        norm = _normalize_sentence(candidate)
        if norm in normalized_original:
            return True
        # Лексична схожість Jaccard як проста евристика
        cand_set = set(w.lower() for w in re.findall(r"\w+", candidate))
        for orig_norm, orig_txt in normalized_original.items():
            orig_set = set(w.lower() for w in re.findall(r"\w+", orig_txt))
            if not orig_set:
                continue
            jacc = len(cand_set & orig_set) / max(1, len(cand_set | orig_set))
            if jacc >= 0.7:
                return True
        return False

    if max_words <= very_short_word_threshold:
        # Вибираємо одне найкраще самодостатнє речення, бажано наявне у джерелі
        for sentence, score in ranked_sentences:
            if len(sentence.split()) > max_words:
                continue
            if is_self_contained(sentence) and appears_verbatim_or_close(sentence):
                return sentence.strip()
        # Якщо не знайдено з високою схожістю — візьмемо перше лід-орієнтоване, що влізає
        for sentence, score in ranked_sentences:
            if len(sentence.split()) <= max_words and is_self_contained(sentence):
                return sentence.strip()
        # Фолбек: найкоротше з топ-N, що вміщується
        for sentence, _ in ranked_sentences[:5]:
            words = sentence.split()
            if len(words) <= max_words:
                return sentence.strip()
        # Останній фолбек — жорстка обрізка першого речення з рангу
        first = ranked_sentences[0][0]
        words = first.split()
        return " ".join(words[:max_words]).rstrip() + "…"

    # Звичайний (не ультракороткий) режим: можна обережно конкатенувати
    # Додаємо логіку для рівномірного покриття всього тексту
    selected_sentences: List[str] = []
    current_words = 0
    seen_normalized: Set[str] = set()  # Відстежуємо нормалізовані речення для уникнення дублікатів
    
    for sentence, score in ranked_sentences:
        sentence = _cleanup_sentence(sentence)
        sentence_words = len(sentence.split())
        if sentence_words > max_words:
            continue
        
        # Перевіряємо на дублікати
        sentence_normalized = _normalize_sentence(sentence)
        if sentence_normalized in seen_normalized:
            continue
        
        # Перевіряємо, чи можемо додати речення
        if current_words + sentence_words <= max_words:
            selected_sentences.append(sentence)
            seen_normalized.add(sentence_normalized)
            current_words += sentence_words
            # Продовжуємо поки не вичерпано бюджет або немає речень
        else:
            # Якщо не вміщається, завершуємо, якщо вже досягнули мінімуму
            if current_words >= min_words:
                break
            # Інакше пропустимо це речення і спробуємо наступне коротше
            else:
                continue

    # Якщо нічого не додали, повертаємо найкоротше речення, що вміщається
    if not selected_sentences:
        for sentence, _ in sorted(ranked_sentences, key=lambda x: len(x[0].split())):
            sentence = _cleanup_sentence(sentence)
            if len(sentence.split()) <= max_words:
                return sentence.strip()
        # Фолбек: обрізка першого
        first = _cleanup_sentence(ranked_sentences[0][0])
        words = first.split()
        return " ".join(words[:max_words]).rstrip() + "…"

    # Якщо залишився істотний бюджет, дозаповнимо з оригінальних речень
    if current_words < max_words * 0.8:
        # Уникати повторів
        seen_norm = {_normalize_sentence(s) for s in selected_sentences}
        for orig_sentence in _best_original_sentences(original_text):
            if _looks_like_boilerplate(orig_sentence):
                continue
            if _normalize_sentence(orig_sentence) in seen_norm:
                continue
            if not _has_sufficient_overlap(orig_sentence, original_text, min_jaccard=0.1):
                continue
            w = len(orig_sentence.split())
            if w > max_words:
                continue
            if current_words + w <= max_words:
                selected_sentences.append(orig_sentence)
                seen_norm.add(_normalize_sentence(orig_sentence))
                current_words += w
            if current_words >= max_words:
                break

    return " ".join(selected_sentences)


def _create_extended_summary(summaries: List[str], facts: List[str], tables_note: str, max_words: int) -> str:
    """Створює розширене резюме з урахуванням фактів і приміток."""
    parts: List[str] = []

    if tables_note:
        parts.append(tables_note)

    # Додаємо резюме з покращеною структурою
    if summaries:
        combined_summaries = " ".join(summaries)
        if len(combined_summaries.split()) <= max_words - 20:  # Залишаємо місце для фактів
            parts.append(combined_summaries)
        else:
            # Обрізаємо до max_words з урахуванням фактів
            words = combined_summaries.split()
            available_words = max_words - 20 if facts else max_words
            parts.append(" ".join(words[:available_words]) + "...")

    # Додаємо факти як структурований список
    if facts:
        bullet_facts = [f"• {f}" for f in facts if f.strip()]
        if bullet_facts:
            parts.append("Ключові факти:\n" + "\n".join(bullet_facts))

    return "\n\n".join([p for p in parts if p and p.strip()])


def _normalize_sentence(sentence: str) -> str:
    """Нормалізує речення для порівняння."""
    # Видаляємо зайві пробіли та приводимо до нижнього регістру
    normalized = re.sub(r'\s+', ' ', sentence.strip().lower())
    # Видаляємо пунктуацію на кінці
    normalized = re.sub(r'[.!?…]+$', '', normalized)
    return normalized


def _split_into_sentences(text: str) -> List[str]:
    """Розбиває текст на речення, зберігаючи кінцеву пунктуацію."""
    # Розбиваємо із захопленням роздільника, потім склеюємо назад
    # Підтримуємо як українські, так і англійські тексти
    parts = re.split(r'(\s*[.!?…]+)(?=\s+[А-ЯЄІЇҐA-Z]|\s*$)', text)
    sentences: List[str] = []
    i = 0
    while i < len(parts):
        chunk = parts[i].strip()
        if not chunk:
            i += 1
            continue
        punct = ""
        if i + 1 < len(parts) and re.fullmatch(r'\s*[.!?…]+', parts[i + 1] or ""):
            punct = parts[i + 1].strip()
            i += 2
        else:
            i += 1
        sentence = (chunk + (" " + punct if punct and not chunk.endswith(punct) else punct)).strip()
        sentences.append(sentence)
    return [s for s in sentences if s]


def _cleanup_sentence(sentence: str) -> str:
    """Видаляє кліше та зайві вступи/висновки, зберігаючи зміст речення."""
    s = sentence.strip()
    # Прибираємо типові «новинні» кліше
    boilerplate_patterns = [
        r'^(У\s+підсумку|Загалом|В\s+цілому|Таким\s+чином|Отже)[:,\s]+',
        r'(?:-|\u2014)\s*повідомляють\s+ЗМІ\.?$',
        r'(?:-|\u2014)\s*як\s+повідомляють\s+ЗМІ\.?$',
        r'\s*\((?:фото|відео)[:\)]\)\s*$',
    ]
    for pat in boilerplate_patterns:
        s = re.sub(pat, '', s, flags=re.IGNORECASE)
    # Очищаємо зайві пробіли
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _has_sufficient_overlap(candidate: str, original_text: str, *, min_jaccard: float) -> bool:
    cand_set = set(w.lower() for w in re.findall(r"\w+", candidate))
    if not cand_set:
        return False
    best = 0.0
    for orig in _split_into_sentences(original_text):
        orig_set = set(w.lower() for w in re.findall(r"\w+", orig))
        if not orig_set:
            continue
        j = len(cand_set & orig_set) / max(1, len(cand_set | orig_set))
        if j > best:
            best = j
            if best >= min_jaccard:
                return True
    return best >= min_jaccard


def _looks_like_boilerplate(sentence: str) -> bool:
    patterns = [
        r'про\s+це\s+повідомила?\s+(?:його\s+)?прес-?служба\.?$',
        r'повідомили?\s+у\s+прес-?службі\.?$',
        r'пише\s+видання\s+[^.]+\.?$',
        r'як\s+повідомляють\s+ЗМІ\.?$',
    ]
    for pat in patterns:
        if re.search(pat, sentence, flags=re.IGNORECASE):
            return True
    return False


def _best_original_sentences(original_text: str) -> List[str]:
    candidates = []
    for s in _split_into_sentences(original_text):
        s_clean = _cleanup_sentence(s)
        if len(s_clean.split()) < 6:
            continue
        if _looks_like_boilerplate(s_clean):
            continue
        candidates.append(s_clean)
    # Евристика: довжина у межах 10..30 слів краща; речення на початку тексту — пріоритет
    scored: List[Tuple[str, float]] = []
    total_len = max(1, len(original_text))
    for s in candidates:
        w = len(s.split())
        length_bonus = 1.0 if 10 <= w <= 30 else 0.5
        pos = original_text.find(s)
        pos_bonus = 1.0 - (max(0, pos) / total_len)
        scored.append((s, length_bonus + 0.8 * pos_bonus))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored]


