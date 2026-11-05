"""
Скрипт за автоматично компилиране на Whisper BG Transcriber в самостоятелно .EXE чрез PyInstaller.
"""

import sys
import os
import io
import shutil
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import PyInstaller.__main__
import PyInstaller.building.utils
import subprocess

# Предотвратяване на WinError 32 при отворена папка dist/WhisperTranscriber във File Explorer:
# изчистваме съдържанието на папката, без да трием самата коренова папка, която веднага след това се пресъздава.
def _safe_make_clean_directory(path):
    p = Path(path)
    if p.exists():
        for child in list(p.iterdir()):
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            except Exception:
                pass
    else:
        p.mkdir(parents=True, exist_ok=True)

PyInstaller.building.utils._make_clean_directory = _safe_make_clean_directory
try:
    import PyInstaller.building.api
    PyInstaller.building.api._make_clean_directory = _safe_make_clean_directory
except Exception:
    pass

# Спиране на стария процес, ако е пуснат, за да не заключва .dll файлове
if sys.platform == "win32":
    try:
        subprocess.run(["taskkill", "/F", "/IM", "WhisperTranscriber.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
for f in ["test_bggpt7b_diff.py", "test_fishing_diff.py", "test_bonus_chunk.py", "test_qwen_fixes.py", "test_bggpt_fixes.py"]:
    (BASE_DIR / f).unlink(missing_ok=True)

print("=" * 60)
print("  Компилиране на Whisper BG Transcriber в самостоятелно .EXE")
print("=" * 60)

# Аргументи за PyInstaller
pyinstaller_args = [
    str(BASE_DIR / "main.py"),
    "--name=WhisperTranscriber",
    "--onedir",
    "--noconfirm",
    # Събиране на всички бинарни файлове и DLL-и за ключовите библиотеки
    "--collect-all=faster_whisper",
    "--collect-all=ctranslate2",
    "--collect-all=onnxruntime",
    "--collect-all=onnxruntime_genai",
    "--collect-all=onnx_asr",
    "--collect-all=av",
    "--collect-all=scipy",
    "--collect-all=docx",
    "--collect-all=uvicorn",
    "--collect-all=fastapi",
    "--collect-all=starlette",
    "--collect-all=yt_dlp",
    # Добавяне на статични папки
    f"--add-data={BASE_DIR / 'templates'}{os.pathsep}templates",
    f"--add-data={BASE_DIR / 'static'}{os.pathsep}static",
    # Директории за изход
    f"--distpath={BASE_DIR / 'dist'}",
    f"--workpath={BASE_DIR / 'build'}",
]

dist_folder = BASE_DIR / "dist" / "WhisperTranscriber"
dist_models = dist_folder / "models"
models_backup = BASE_DIR / "dist" / "_models_preserve_tmp"
dist_debug = dist_folder / "debug_diff"
debug_backup = BASE_DIR / "dist" / "_debug_preserve_tmp"

# Запазване на вече изтеглените модели и debug_diff в dist/WhisperTranscriber/ (за да не се изтрият при компилация)
if dist_models.exists():
    print("Временно запазване на съществуващата папка models/...")
    if models_backup.exists():
        shutil.rmtree(models_backup, ignore_errors=True)
    shutil.move(str(dist_models), str(models_backup))

if dist_debug.exists():
    if debug_backup.exists():
        shutil.rmtree(debug_backup, ignore_errors=True)
    shutil.move(str(dist_debug), str(debug_backup))

try:
    print("Стартиране на PyInstaller...")
    PyInstaller.__main__.run(pyinstaller_args)
finally:
    dist_folder.mkdir(parents=True, exist_ok=True)
    if models_backup.exists():
        print("Възстановяване на запазените модели в dist/WhisperTranscriber/models...")
        if dist_models.exists():
            shutil.rmtree(dist_models, ignore_errors=True)
        shutil.move(str(models_backup), str(dist_models))

    if debug_backup.exists():
        if dist_debug.exists():
            shutil.rmtree(dist_debug, ignore_errors=True)
        shutil.move(str(debug_backup), str(dist_debug))

if (dist_folder / "WhisperTranscriber.exe").exists():
    root_models = BASE_DIR / "models"
    if root_models.exists():
        print("Копиране / синхронизиране на локалните модели в дистрибуционната папка...")
        shutil.copytree(root_models, dist_models, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".locks", "*.tmp"))

    print("\n" + "=" * 60)
    print("УСПЕШНО КОМПИЛИРАНЕ!")
    print(f"Готовото приложение се намира в:")
    print(f"{dist_folder.resolve()}")
    print("=" * 60)
else:
    print("\n[Грешка] Папката с готовата програма не беше намерена!")
