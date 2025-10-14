import argparse
from pathlib import Path
from typing import Optional

# Імпорти підмодулів пайплайна
from summarizer_agent.io.loaders import load_any
from summarizer_agent.nlp.chunker import chunk_text
from summarizer_agent.nlp.summarizer import TextSummarizer
from summarizer_agent.nlp.facts import extract_facts
from summarizer_agent.post.assemble import assemble_summary
from summarizer_agent.export.exporters import export_txt, export_pdf


def run_cli() -> None:
    # CLI інтерфейс: параметри запуску агента
    parser = argparse.ArgumentParser(description="ШІ-агент для автоматизованого резюмування текстів")
    # Дозволяємо шлях із пробілами (Windows): збираємо всі позиційні токени та з'єднуємо
    parser.add_argument("input", nargs="+", help="Шлях до вхідного файлу (.txt, .pdf, .docx, .csv, .xlsx, .jpg, .png)")
    parser.add_argument("--model", type=str, default="facebook/bart-large-cnn", help="Назва моделі Hugging Face для сумаризації (BART-large за замовчуванням)")
    parser.add_argument("--interactive", action="store_true", help="Напівінтерактивний режим: уточнити/перегенерувати/експортувати/тільки факти")
    # Сумісність: прапор не потрібен, формат визначається автоматично
    parser.add_argument("--docx", action="store_true", help="Сумісність: не потрібен. Формат файлу визначається автоматично.")
    parser.add_argument("--device", type=str, default="auto", help="Пристрій: auto|cpu|cuda|cuda:0")
    # Параметри довжини тепер адаптивні, ці значення використовуються як фолбек
    parser.add_argument("--min_words", type=int, default=0, help="Мінімальна кількість слів (0 = авто)")
    parser.add_argument("--max_words", type=int, default=0, help="Максимальна кількість слів (0 = авто)")
    parser.add_argument("--pdf", action="store_true", help="Експортувати також у PDF поруч із TXT")
    parser.add_argument("--ocr", action="store_true", help="Примусово виконати OCR для зображень (потрібен встановлений Tesseract)")
    # Нові прапори для вбудованих об'єктів у PDF/DOCX
    parser.add_argument("--parse_docx_tables", action="store_true", help="Витягувати таблиці з DOCX як markdown")
    parser.add_argument("--ocr_docx_images", action="store_true", help="OCR для зображень усередині DOCX")
    parser.add_argument("--parse_pdf_tables", action="store_true", help="Витягувати таблиці з PDF за допомогою Camelot/Tabula")
    parser.add_argument("--ocr_pdf_images", action="store_true", help="OCR зображень усередині PDF (PyMuPDF+pytesseract)")
    parser.add_argument("--chunk_tokens", type=int, default=512, help="Розмір чанка у токенах з урахуванням токенайзера")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="Перекриття між чанками у токенах")
    parser.add_argument("--large_text", action="store_true", help="Оптимізація для великих текстів (автоматично використовує CPU та збільшені чанки)")
    parser.add_argument("--no_facts", action="store_true", help="Не витягувати та не показувати ключові факти")
    parser.add_argument("--force_gpu", action="store_true", help="Примусово використовувати GPU навіть для великих текстів (може спричинити помилки пам'яті)")
    args = parser.parse_args()

    # Відновлення шляху з позиційних токенів
    input_path = Path(" ".join(args.input))
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Завантаження сирого контенту з файлу будь-якого підтримуваного типу
    text, tables_note = load_any(
        input_path,
        force_ocr=args.ocr,
        parse_docx_tables=args.parse_docx_tables,
        ocr_docx_images=args.ocr_docx_images,
        parse_pdf_tables=args.parse_pdf_tables,
        ocr_pdf_images=args.ocr_pdf_images,
    )

    if not text or len(text.strip()) == 0:
        raise ValueError("No textual content extracted from input.")

    # Автоматичне визначення пристрою для великих текстів
    device_arg = args.device
    chunk_tokens = args.chunk_tokens
    chunk_overlap = args.chunk_overlap
    
    if args.large_text and not args.force_gpu:
        print("[INFO] Large text mode enabled - optimizing for CPU processing")
        device_arg = "cpu"  # Примусово використовуємо CPU
        chunk_tokens = min(512, args.chunk_tokens)  # Безпечний ліміт для стабільності
        chunk_overlap = min(100, args.chunk_overlap)  # Консервативне перекриття
        print(f"[INFO] Using CPU with chunk_tokens={chunk_tokens}, chunk_overlap={chunk_overlap}")
    elif args.force_gpu:
        print("[INFO] Force GPU mode enabled - using GPU with optimizations")
        device_arg = "auto"  # Використовуємо GPU з оптимізаціями
        chunk_tokens = min(256, args.chunk_tokens)  # Менші чанки для GPU
        chunk_overlap = min(50, args.chunk_overlap)  # Менше перекриття
        print(f"[INFO] Using GPU with chunk_tokens={chunk_tokens}, chunk_overlap={chunk_overlap}")

    # Ініціалізація узагальнювача тексту (модель + токенайзер)
    summarizer = TextSummarizer(model_name=args.model, device_arg=device_arg)

    # Розбиття довгого тексту на перекриті чанки за кількістю токенів
    chunks = chunk_text(text, tokenizer=summarizer.tokenizer, max_tokens=chunk_tokens, overlap_tokens=chunk_overlap, device=device_arg)

    # Адаптивна довжина: ~20% від кількості слів у вихідному документі (мін. 60, макс. 600)
    orig_words = max(1, len(text.split()))
    auto_target = max(60, min(600, int(orig_words * 0.2)))
    target_min = args.min_words or int(auto_target * 0.7)
    target_max = args.max_words or int(auto_target * 1.2)

    # Комбінований підхід до розрахунку довжини чанків
    num_chunks = len(chunks)
    
    # Базові значення
    base_min = 60
    base_max = 150
    
    # Фактори адаптації
    text_factor = min(1.3, max(0.8, orig_words / 2000))  # Адаптація під розмір тексту
    chunk_factor = min(1.2, max(0.9, 5 / num_chunks))    # Адаптація під кількість чанків
    
    # Розрахунок адаптивних меж
    per_chunk_min = int(base_min * text_factor * chunk_factor)
    per_chunk_max = int(base_max * text_factor * chunk_factor)
    
    print(f"[INFO] Adaptive chunk sizing: {orig_words} words -> {num_chunks} chunks")
    print(f"[INFO] Adaptation factors: text={text_factor:.2f}, chunks={chunk_factor:.2f}")
    print(f"[INFO] Chunk limits: min={per_chunk_min}, max={per_chunk_max} words")
    chunk_summaries = summarizer.summarize_chunks(chunks, min_words=per_chunk_min, max_words=per_chunk_max, device=device_arg)

    # Витягнення простих фактів/чисел із оригіналу та проміжних резюме (якщо не відключено)
    facts = [] if args.no_facts else extract_facts(text + "\n\n" + "\n".join(chunk_summaries))

    # Збирання фінального короткого та розширеного резюме
    final_summary, extended_summary = assemble_summary(
        original_text=text,
        chunk_summaries=chunk_summaries,
        facts=facts,
        tables_note=tables_note,
        target_min_words=target_min,
        target_max_words=target_max,
        only_facts=False,
    )

    # У напівінтерактивному режимі дозволяємо швидкі дії
    if args.interactive:
        print("\nПопередній перегляд короткого резюме:\n")
        print(final_summary[:800] + ("..." if len(final_summary) > 800 else ""))
        while True:
            print("\nВиберіть дію: [s] зберегти, [r] перегенерувати з новими параметрами, [f] лише факти, [q] вийти без збереження")
            choice = input("> ").strip().lower()
            if choice == "s":
                break
            elif choice == "f":
                only_facts_summary, only_facts_extended = assemble_summary(
                    original_text=text,
                    chunk_summaries=chunk_summaries,
                    facts=facts,
                    tables_note=tables_note,
                    target_min_words=args.min_words,
                    target_max_words=args.max_words,
                    only_facts=True,
                )
                final_summary, extended_summary = only_facts_summary, only_facts_extended
                print("\n(Режим тільки фактів застосовано)")
            elif choice == "r":
                try:
                    new_min = int(input("Нова мін. кількість слів: ").strip() or args.min_words)
                    new_max = int(input("Нова макс. кількість слів: ").strip() or args.max_words)
                except ValueError:
                    print("Некоректне число. Спробуйте ще раз.")
                    continue
                # перерахунок довжин на чанк з адаптивним підходом
                # Базові значення
                base_min = 60
                base_max = 150
                
                # Фактори адаптації
                text_factor = min(1.3, max(0.8, orig_words / 2000))
                chunk_factor = min(1.2, max(0.9, 5 / len(chunks)))
                
                # Розрахунок адаптивних меж
                adaptive_min = int(base_min * text_factor * chunk_factor)
                adaptive_max = int(base_max * text_factor * chunk_factor)
                
                chunk_summaries = summarizer.summarize_chunks(
                    chunks,
                    min_words=max(adaptive_min, new_min // max(1, len(chunks))),
                    max_words=max(adaptive_max, new_max // max(1, len(chunks))),
                )
                facts = extract_facts(text + "\n\n" + "\n".join(chunk_summaries))
                final_summary, extended_summary = assemble_summary(
                    original_text=text,
                    chunk_summaries=chunk_summaries,
                    facts=facts,
                    tables_note=tables_note,
                    target_min_words=new_min,
                    target_max_words=new_max,
                    only_facts=False,
                )
                print("\nПерегенеровано. Попередній перегляд:\n")
                print(final_summary[:800] + ("..." if len(final_summary) > 800 else ""))
            elif choice == "q":
                print("Вихід без збереження.")
                return
            else:
                print("Невідома команда. Спробуйте ще раз.")

    # Експорт результатів у файли
    output_base = input_path.with_suffix("")
    txt_path = output_base.with_name(output_base.name + "_summary.txt")
    export_txt(txt_path, final_summary, extended_summary, facts)

    if args.pdf:
        pdf_path = output_base.with_name(output_base.name + "_summary.pdf")
        export_pdf(pdf_path, final_summary, extended_summary, facts)

    print(str(txt_path))
    if args.pdf:
        print(str(pdf_path))


if __name__ == "__main__":
    run_cli()


