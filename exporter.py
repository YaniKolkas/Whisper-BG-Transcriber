"""
Модул за експортиране на транскрипции в различни формати: TXT, Word (.docx), SRT (субтитри) и JSON.
"""

from datetime import datetime
import io
import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def format_timestamp(seconds: float, srt_format: bool = False) -> str:
    """Форматира секунди във времеви маркер HH:MM:SS или HH:MM:SS,mmm за SRT."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)

    if srt_format:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def export_txt(segments: list, include_timestamps: bool = False) -> str:
    """Генерира текстов файл (TXT)."""
    lines = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        if include_timestamps:
            t_start = format_timestamp(seg.get("start", 0))
            t_end = format_timestamp(seg.get("end", 0))
            lines.append(f"[{t_start} - {t_end}] {text}")
        else:
            lines.append(text)
    
    if include_timestamps:
        return "\n".join(lines)
    
    # За чист текст обединяваме в плавни абзаци
    return "\n\n".join(lines)


def export_srt(segments: list) -> str:
    """Генерира субтитри във формат SRT."""
    srt_entries = []
    idx = 1
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start_str = format_timestamp(seg.get("start", 0), srt_format=True)
        end_str = format_timestamp(seg.get("end", 0), srt_format=True)
        srt_entries.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
        idx += 1
    return "\n".join(srt_entries)


def set_cell_border(cell, **kwargs):
    """Помощна функция за рамки на клетки в Word таблица."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = 'w:{}'.format(edge)
            element = OxmlElement(tag)
            element.set(qn('w:val'), edge_data.get('val', 'single'))
            element.set(qn('w:sz'), str(edge_data.get('sz', 4)))
            element.set(qn('w:space'), '0')
            element.set(qn('w:color'), edge_data.get('color', 'CCCCCC'))
            tcBorders.append(element)
    tcPr.append(tcBorders)


def export_docx(segments: list, metadata: dict = None, include_timestamps: bool = False) -> io.BytesIO:
    """Генерира форматиран Microsoft Word (.docx) документ."""
    if metadata is None:
        metadata = {}

    doc = docx.Document()

    # Задаване на полета на страницата (margins)
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    # Заглавие на документа
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    title_run = title_p.add_run("Транскрипция на аудиозапис")
    title_run.font.name = "Calibri"
    title_run.font.size = Pt(20)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(30, 41, 59)  # Тъмно синьо-сиво

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(14)
    sub_run = sub_p.add_run("OpenAI Whisper • Български език")
    sub_run.font.name = "Calibri"
    sub_run.font.size = Pt(11)
    sub_run.font.color.rgb = RGBColor(100, 116, 139)

    # Информационна кутия (Таблица с метаданни)
    table = doc.add_table(rows=4, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    meta_items = [
        ("Файл:", metadata.get("filename", "Неизвестен")),
        ("Продължителност:", metadata.get("duration_formatted", "-")),
        ("Модел:", f"Whisper {metadata.get('model_name', 'small').capitalize()}"),
        ("Дата на обработка:", metadata.get("created_at", datetime.now().strftime("%d.%m.%Y %H:%M"))),
    ]

    for i, (label, val) in enumerate(meta_items):
        cell_lbl = table.rows[i].cells[0]
        cell_val = table.rows[i].cells[1]
        
        cell_lbl.width = Inches(1.8)
        cell_val.width = Inches(4.8)

        p_lbl = cell_lbl.paragraphs[0]
        p_lbl.paragraph_format.space_before = Pt(2)
        p_lbl.paragraph_format.space_after = Pt(2)
        r_lbl = p_lbl.add_run(label)
        r_lbl.font.name = "Calibri"
        r_lbl.font.size = Pt(9.5)
        r_lbl.font.bold = True
        r_lbl.font.color.rgb = RGBColor(71, 85, 105)

        p_val = cell_val.paragraphs[0]
        p_val.paragraph_format.space_before = Pt(2)
        p_val.paragraph_format.space_after = Pt(2)
        r_val = p_val.add_run(str(val))
        r_val.font.name = "Calibri"
        r_val.font.size = Pt(9.5)
        r_val.font.color.rgb = RGBColor(30, 41, 59)

    # Разделителна линия
    div_p = doc.add_paragraph()
    div_p.paragraph_format.space_before = Pt(12)
    div_p.paragraph_format.space_after = Pt(12)
    div_run = div_p.add_run("―" * 48)
    div_run.font.color.rgb = RGBColor(226, 232, 240)

    # Заглавие на съдържанието
    sec_head = doc.add_paragraph()
    sec_head.paragraph_format.space_after = Pt(8)
    r_head = sec_head.add_run("Текст на разговора")
    r_head.font.name = "Calibri"
    r_head.font.size = Pt(13)
    r_head.font.bold = True
    r_head.font.color.rgb = RGBColor(15, 23, 42)

    # Добавяне на транскрибирания текст
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.25

        if include_timestamps:
            t_start = format_timestamp(seg.get("start", 0))
            t_end = format_timestamp(seg.get("end", 0))
            ts_run = p.add_run(f"[{t_start} - {t_end}]  ")
            ts_run.font.name = "Calibri"
            ts_run.font.size = Pt(9)
            ts_run.font.bold = True
            ts_run.font.color.rgb = RGBColor(99, 102, 241)  # Индиго/лилав акцент за време

        text_run = p.add_run(text)
        text_run.font.name = "Calibri"
        text_run.font.size = Pt(11)
        text_run.font.color.rgb = RGBColor(30, 41, 59)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer
