from __future__ import annotations

# Розбиття тексту на чанки за кількістю токенів токенайзера
from typing import List


def chunk_text(
    text: str,
    tokenizer,
    max_tokens: int = 512,  # Безпечний ліміт для BART (модель має максимум 1024)
    overlap_tokens: int = 100,  # Оптимально для GPU з 16 ГБ
    device: str = "auto",  # Для автоматичного визначення розміру чанків
) -> List[str]:
    """Розбиває довгий текст на перекриті чанки за токенами з семантичним розбиттям.

    - max_tokens: розмір чанка
    - overlap_tokens: перекриття між сусідніми чанками для збереження контексту
    - device: пристрій для автоматичного визначення оптимальних параметрів
    """
    if not text:
        return []

    # Автоматичне визначення оптимальних параметрів для CPU/GPU
    if device == "cpu" or (device == "auto" and max_tokens <= 512):
        max_tokens = min(512, max_tokens)  # Безпечний ліміт для стабільності
        overlap_tokens = min(100, overlap_tokens)  # Консервативне перекриття
    
    # Розбиваємо текст на речення з збереженням знаків пунктуації
    sentences = _split_into_sentences_with_punctuation(text)
    
    if not sentences:
        return [text]
    
    # Якщо весь текст поміщається в один чанк
    if len(tokenizer.encode(text, add_special_tokens=False)) <= max_tokens:
        return [text]
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        
        # Якщо одне речення занадто довге, розбиваємо його
        if len(sentence_tokens) > max_tokens:
            # Зберігаємо поточний чанк
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_tokens = 0
            
            # Розбиваємо довге речення на частини
            long_sentence_chunks = _split_long_sentence(sentence, tokenizer, max_tokens)
            chunks.extend(long_sentence_chunks)
            continue
        
        # Перевіряємо, чи поміститься речення в поточний чанк
        test_tokens = current_tokens + len(sentence_tokens)
        
        if test_tokens <= max_tokens:
            # Додаємо речення до поточного чанка
            current_chunk += " " + sentence if current_chunk else sentence
            current_tokens = test_tokens
        else:
            # Зберігаємо поточний чанк і починаємо новий
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            current_chunk = sentence
            current_tokens = len(sentence_tokens)
    
    # Додаємо останній чанк
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Додаємо перекриття між чанками (15% від попереднього чанка)
    if len(chunks) > 1:
        overlapped_chunks = []
        overlap_ratio = 0.15  # 15% перекриття
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Обчислюємо розмір перекриття
                prev_tokens = tokenizer.encode(chunks[i-1], add_special_tokens=False)
                overlap_size = max(20, int(len(prev_tokens) * overlap_ratio))  # Мінімум 20 токенів
                
                if len(prev_tokens) > overlap_size:
                    # Витягуємо перекриття з попереднього чанка
                    overlap_text = tokenizer.decode(prev_tokens[-overlap_size:], skip_special_tokens=True)
                    
                    # Перевіряємо, чи перекриття не обрізає речення посередині
                    overlap_text = _ensure_sentence_boundary(overlap_text, chunks[i-1])
                    
                    # Додаємо перекриття до поточного чанка
                    overlapped_chunk = overlap_text + " " + chunk
                    overlapped_chunks.append(overlapped_chunk)
                else:
                    overlapped_chunks.append(chunk)
        
        return overlapped_chunks
    
    return chunks


def _split_into_sentences_with_punctuation(text: str) -> List[str]:
    """Розбиває текст на речення з збереженням знаків пунктуації."""
    import re
    
    # Очищаем текст от лишних пробелов
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Розбиваємо на речення з збереженням знаків пунктуації
    # Ищем конец предложения, за которым следует пробел или конец строки
    sentence_pattern = r'([.!?…]+)(?:\s+|$)'
    parts = re.split(sentence_pattern, text)
    
    sentences = []
    current_sentence = ""
    
    for i, part in enumerate(parts):
        if re.match(sentence_pattern, part):
            # Это знак пунктуации
            current_sentence += part
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            current_sentence = ""
        else:
            # Это текст
            current_sentence += part
    
    # Добавляем последнее предложение, если оно есть
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    return [s for s in sentences if s.strip()]


def _split_long_sentence(sentence: str, tokenizer, max_tokens: int) -> List[str]:
    """Розбиває довге речення на частини за словами."""
    words = sentence.split()
    chunks = []
    current_chunk = ""
    
    for word in words:
        test_chunk = current_chunk + " " + word if current_chunk else word
        test_tokens = tokenizer.encode(test_chunk, add_special_tokens=False)
        
        if len(test_tokens) <= max_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = word
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def _ensure_sentence_boundary(overlap_text: str, original_chunk: str) -> str:
    """Перевіряє, чи перекриття не обрізає речення посередині."""
    import re
    
    # Якщо перекриття закінчується знаком пунктуації - все добре
    if re.search(r'[.!?…]\s*$', overlap_text.strip()):
        return overlap_text
    
    # Якщо ні - знаходимо найближчий знак пунктуації в оригінальному чанку
    # Шукаємо позицію перекриття в оригінальному чанку
    overlap_start = original_chunk.find(overlap_text)
    if overlap_start == -1:
        return overlap_text
    
    # Шукаємо попередній знак пунктуації
    before_overlap = original_chunk[:overlap_start]
    sentence_end_match = re.search(r'([.!?…])\s*$', before_overlap)
    
    if sentence_end_match:
        # Знаходимо початок цього речення
        sentence_start = before_overlap.rfind(sentence_end_match.group(1)) + 1
        if sentence_start > 0:
            # Додаємо повне речення до перекриття
            full_sentence = original_chunk[sentence_start:overlap_start + len(overlap_text)]
            return full_sentence
    
    return overlap_text


def _split_into_sentences(text: str) -> List[str]:
    """Розбиває текст на речення з урахуванням української та англійської мов."""
    import re
    # Простий паттерн для розбиття на речення
    sentence_endings = r'[.!?…]+'
    sentences = re.split(sentence_endings, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


