"""
Команден ред (CLI) инструмент за Whisper транскрибиране на български език.
Пример:
    python cli.py --input recording.mp3 --model small --format docx
    python cli.py --input speech.wav --model medium --format txt --timestamps
"""

import argparse
import sys
import io
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import transcriber
import exporter


def main():
    parser = argparse.ArgumentParser(
        description="Транскрибиране на аудиозаписи на български език с OpenAI Whisper."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Път до входния аудио или видео файл"
    )
    parser.add_argument(
        "--model", "-m", default="large-v3-turbo", choices=["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"],
        help="Whisper модел (препоръчителни: large-v3-turbo за качество, или small за скорост)"
    )
    parser.add_argument(
        "--format", "-f", default="docx", choices=["txt", "docx", "srt", "all"],
        help="Формат на изходния файл (по подразбиране: docx)"
    )
    parser.add_argument(
        "--timestamps", "-t", action="store_true",
        help="Включване на времеви маркери в текста"
    )
    parser.add_argument(
        "--no-vad", action="store_true",
        help="Изключване на VAD филтъра за тишина"
    )
    parser.add_argument(
        "--prompt", "-p", default=transcriber.DEFAULT_BG_PROMPT,
        help="Първоначален контекст/подсказка на български език"
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Грешка: Файлът '{input_path}' не съществува!", file=sys.stderr)
        sys.exit(1)

    print(f"--- Whisper BG Transcriber ---")
    print(f"Файл: {input_path.name}")
    print(f"Модел: {args.model}")
    print(f"Език: Български (bg)")
    print(f"Формат: {args.format}")
    print("Стартиране на обработката...")

    def on_progress(p):
        if p.get("latest_text"):
            print(f"[{p.get('progress')}%] {p.get('latest_text')}")

    try:
        result = transcriber.transcribe_audio(
            file_path=str(input_path),
            model_name=args.model,
            language="bg",
            vad_filter=not args.no_vad,
            initial_prompt=args.prompt,
            progress_callback=on_progress,
        )
    except Exception as e:
        print(f"\nГрешка при транскрибиране: {e}", file=sys.stderr)
        sys.exit(1)

    segments = result["segments"]
    base_name = input_path.stem
    output_dir = input_path.parent

    metadata = {
        "filename": input_path.name,
        "duration_formatted": f"{int(result['duration'] // 60)}:{int(result['duration'] % 60):02d}",
        "model_name": args.model,
    }

    formats_to_save = [args.format] if args.format != "all" else ["txt", "docx", "srt"]

    for fmt in formats_to_save:
        out_file = output_dir / f"{base_name}_transcription.{fmt}"
        if fmt == "txt":
            content = exporter.export_txt(segments, include_timestamps=args.timestamps)
            out_file.write_text(content, encoding="utf-8")
        elif fmt == "docx":
            buf = exporter.export_docx(segments, metadata=metadata, include_timestamps=args.timestamps)
            out_file.write_bytes(buf.getvalue())
        elif fmt == "srt":
            content = exporter.export_srt(segments)
            out_file.write_text(content, encoding="utf-8")
        print(f"Запазен файл: {out_file.resolve()}")

    print("\nГотово!")


if __name__ == "__main__":
    main()
