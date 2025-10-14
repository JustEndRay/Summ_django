from __future__ import annotations

# Покращений модуль виділення фактів з використанням NER та розширених regex
import re
from typing import List, Dict, Set
import spacy
from collections import Counter


NUMBER_PATTERN = re.compile(r"(?<!\w)(?:\d{1,3}(?:[\s,]\d{3})*|\d+)(?:[.,]\d+)?%?")
DATE_PATTERN = re.compile(r"\b(\d{1,2}[\./-]\d{1,2}[\./-]\d{2,4}|\d{4})\b")
MONEY_PATTERN = re.compile(r"(?:\d+[.,]?\d*)\s*(?:млрд|млн|тис|доларів|гривень|євро|рублів)")
PERCENTAGE_PATTERN = re.compile(r"\d+[.,]?\d*%")
AGE_PATTERN = re.compile(r"\b(\d+)\s*(?:років|рік|року)\b")


def extract_facts(text: str) -> List[str]:
    """Повертає список коротких фактів: числа, дати/роки, сутності з NER."""
    if not text:
        return []

    facts: List[str] = []

    # Числа та відсотки
    numbers = set(NUMBER_PATTERN.findall(text))
    if numbers:
        facts.append("Числа: " + ", ".join(sorted(numbers, key=lambda x: float(x.replace(',', '.').replace('%', '')) if x.replace(',', '.').replace('%', '').replace('.', '').isdigit() else 0)))

    # Дати/роки
    dates = set(DATE_PATTERN.findall(text))
    if dates:
        facts.append("Дати: " + ", ".join(sorted(dates)))

    # Грошові суми
    money_amounts = set(MONEY_PATTERN.findall(text))
    if money_amounts:
        facts.append("Грошові суми: " + ", ".join(sorted(money_amounts)))

    # Відсотки
    percentages = set(PERCENTAGE_PATTERN.findall(text))
    if percentages:
        facts.append("Відсотки: " + ", ".join(sorted(percentages)))

    # Вік
    ages = set(AGE_PATTERN.findall(text))
    if ages:
        facts.append("Вік: " + ", ".join(sorted(ages, key=int)) + " років")

    # NER для витягнення сутностей
    entities = _extract_entities_with_spacy(text)
    if entities:
        facts.append("Сутності: " + ", ".join(entities))

    return facts


def _extract_entities_with_spacy(text: str) -> List[str]:
    """Витягує сутності за допомогою spaCy NER."""
    try:
        # Спробуємо завантажити українську модель, якщо доступна
        try:
            nlp = spacy.load("uk_core_news_sm")
        except (OSError, IOError):
            # Якщо українська модель недоступна, спробуємо англійську
            try:
                nlp = spacy.load("en_core_web_sm")
            except (OSError, IOError):
                # Якщо жодна модель недоступна, використовуємо fallback
                return _fallback_entity_extraction(text)
        
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART"]:
                entities.append(ent.text)
        
        # Додаємо капіталізовані фрази як fallback
        capitalized_phrases = re.findall(r"\b([А-ЯЄІЇҐ][а-яєіїґ]+(?:\s+[А-ЯЄІЇҐ][а-яєіїґ]+){0,3})\b", text)
        entities.extend(capitalized_phrases[:10])  # Обмежуємо кількість
        
        # Видаляємо дублікати та сортуємо
        unique_entities = list(set(entities))
        return sorted(unique_entities)[:20]  # Обмежуємо до 20 сутностей
        
    except Exception as e:
        # Fallback до простого regex підходу
        print(f"[WARNING] spaCy error: {e}, using fallback")
        return _fallback_entity_extraction(text)


def _fallback_entity_extraction(text: str) -> List[str]:
    """Fallback метод для витягнення сутностей без spaCy."""
    # Простий regex підхід для української мови
    capitalized_phrases = re.findall(r"\b([А-ЯЄІЇҐ][а-яєіїґ]+(?:\s+[А-ЯЄІЇҐ][а-яєіїґ]+){0,3})\b", text)
    return sorted(set(capitalized_phrases))[:20]


