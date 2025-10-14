from __future__ import annotations

# Експортери результатів у TXT та PDF
from pathlib import Path
from typing import Optional


def export_txt(path: Path, short_summary: str, extended_summary: str, facts: list[str]) -> None:
    """Записує у .txt коротке й розширене резюме та секцію фактів."""
    lines = [
        "=== Коротке резюме ===",
        short_summary.strip(),
        "",
        "=== Розширене резюме ===",
        extended_summary.strip(),
        "",
    ]
    
    # Додаємо факти тільки якщо вони є
    if facts:
        lines.extend([
            "=== Факти та цифри ===",
            "\n".join(facts),
            "",
        ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def export_pdf(path: Path, short_summary: str, extended_summary: str, facts: list[str]) -> None:
    """Генерує PDF зі структурованими секціями за допомогою reportlab.

    Використовує вбудування Unicode-шрифту (TTF), щоб коректно відображати кирилицю та інші скрипти.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab import rl_config
    except Exception as exc:
        raise RuntimeError("reportlab is required to export PDF.") from exc

    # Пошук придатного Unicode-шрифту на системі
    font_name = _register_unicode_font_if_available(
        [
            # Популярні кросплатформені шрифти
            "DejaVuSans.ttf",
            "Arial.ttf",
            "SegoeUI.ttf",
            "NotoSans-Regular.ttf",
            "LiberationSans-Regular.ttf",
        ]
    ) or "Helvetica"

    doc = SimpleDocTemplate(str(path), pagesize=A4)
    base_styles = getSampleStyleSheet()

    # Копіюємо стилі й примушуємо використовувати Unicode-шрифт
    heading_style = ParagraphStyle(
        name="Heading2Unicode",
        parent=base_styles["Heading2"],
        fontName=font_name,
    )
    body_style = ParagraphStyle(
        name="BodyUnicode",
        parent=base_styles["BodyText"],
        fontName=font_name,
        leading=14,
    )

    flow = []

    def add_section(title: str, body: str) -> None:
        flow.append(Paragraph(f"<b>{_escape_xml(title)}</b>", heading_style))
        flow.append(Spacer(1, 8))
        for para in body.split("\n\n"):
            flow.append(Paragraph(_escape_xml(para).replace("\n", "<br/>"), body_style))
            flow.append(Spacer(1, 8))

    # Без заголовків: просто секції контенту
    if short_summary.strip():
        add_section("", short_summary.strip())
    if extended_summary.strip():
        add_section("", extended_summary.strip())
    if facts:
        add_section("", "\n".join(f"- {f}" for f in facts))

    doc.build(flow)


def _register_unicode_font_if_available(candidate_files: list[str]) -> Optional[str]:
    """Намагається знайти й зареєструвати TTF шрифт із підтримкою Unicode.

    Повертає ім'я шрифту для використання у стилях або None, якщо нічого не знайдено.
    Перебирає загальні директорії шрифтів на Windows/Linux/macOS.
    """
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import os
    except Exception:
        return None

    search_dirs = [
        # Windows
        r"C:\\Windows\\Fonts",
        # Linux
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        # macOS
        "/System/Library/Fonts",
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
    ]

    for directory in search_dirs:
        if not directory or not Path(directory).exists():
            continue
        for filename in candidate_files:
            fpath = Path(directory) / filename
            if fpath.exists():
                font_name = f"_Embedded_{fpath.stem}"
                try:
                    pdfmetrics.registerFont(TTFont(font_name, str(fpath)))
                    return font_name
                except Exception:
                    continue
    return None


def _escape_xml(text: str) -> str:
    """Escape XML-значення для Paragraph (ReportLab підтримує mini-HTML)."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


