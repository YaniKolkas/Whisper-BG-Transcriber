"""
FastAPI сървър и REST API за Whisper транскрибатор на български език.
"""

import os
import uuid
import threading
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import transcriber
import exporter

app = FastAPI(title="Whisper BG Transcriber", version="1.0.0")

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import sys

if getattr(sys, "frozen", False):
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", sys.executable)).resolve()
    if not (BUNDLE_DIR / "templates").exists():
        BUNDLE_DIR = Path(sys.executable).resolve().parent
    EXE_DIR = Path(sys.executable).resolve().parent
else:
    BUNDLE_DIR = Path(__file__).resolve().parent
    EXE_DIR = BUNDLE_DIR

STATIC_DIR = BUNDLE_DIR / "static"
TEMPLATES_DIR = BUNDLE_DIR / "templates"
UPLOADS_DIR = EXE_DIR / "uploads"
DEBUG_DIFF_DIR = EXE_DIR / "debug_diff"

UPLOADS_DIR.mkdir(exist_ok=True)
DEBUG_DIFF_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Монтиране на статични файлове
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Памет за задачите (Jobs)
JOBS = {}


class TranscribeRequest(BaseModel):
    job_id: str
    model_name: str = "large-v3"
    language: str = "bg"
    vad_filter: bool = True
    include_timestamps: bool = False
    condition_on_previous_text: bool = False
    audio_preprocessing: bool = True
    expanded_beam_search: bool = True
    apply_postprocessing: bool = True
    postprocess_mode: str = "local"  # "local", "qwen", "gemini", "none"
    local_ai_model: str = "zealod/BgGPT-Gemma-2-2.6B-IT-v1.0-int4-onnx-directml"
    gemini_api_key_source: str = "input"  # "input", "env"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.8-flash"
    custom_prompt: Optional[str] = None
    engine: str = "auto"  # "auto", "directml", "faster_whisper", "cpu"
    debug_mode: bool = True


class TestGeminiRequest(BaseModel):
    key_source: str = "input"
    api_key: Optional[str] = None
    model_name: str = "gemini-3.8-flash"


class GeminiModelsRequest(BaseModel):
    key_source: str = "input"
    api_key: Optional[str] = None


class YouTubeRequest(BaseModel):
    url: str



def format_seconds(seconds: float) -> str:
    """Форматира секунди в четим вид мм:сс или чч:мм:сс."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def save_debug_diff_files(job: dict, result: dict) -> dict:
    """
    Записва междинните и финалните текстове в папка debug_diff/ за директно сравнение
    в KDiff3, Meld или WinMerge (по 1 сегмент на ред за точно подравняване ред по ред).
    """
    try:
        DEBUG_DIFF_DIR.mkdir(exist_ok=True)
        original_stem = Path(job.get("original_filename", "audio")).stem
        safe_stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in original_stem).strip("_")[:40] or "audio"
        ai_tag = str(result.get("ai_model_used") or result.get("postprocess_mode") or "model")
        safe_ai = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in ai_tag).strip("_")[:35] or "model"
        ts_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        run_dir = DEBUG_DIFF_DIR / f"{ts_str}_{safe_stem}_{safe_ai}"
        run_dir.mkdir(parents=True, exist_ok=True)

        whisper_raw = result.get("whisper_raw_text") or ""
        pre_ai = result.get("pre_ai_text") or whisper_raw
        final_lines = result.get("final_lines_text") or result.get("full_text") or ""

        file_whisper = run_dir / "1_whisper_model.txt"
        file_rules = run_dir / "2_after_rules_before_ai.txt"
        file_final = run_dir / "3_final_after_model.txt"

        file_whisper.write_text(whisper_raw + "\n", encoding="utf-8")
        file_rules.write_text(pre_ai + "\n", encoding="utf-8")
        file_final.write_text(final_lines + "\n", encoding="utf-8")

        # Запис и като LATEST_* в главната папка debug_diff/ за бърз Refresh (F5) в KDiff3 / Meld
        latest_whisper = DEBUG_DIFF_DIR / "LATEST_1_whisper_model.txt"
        latest_rules = DEBUG_DIFF_DIR / "LATEST_2_after_rules_before_ai.txt"
        latest_final = DEBUG_DIFF_DIR / "LATEST_3_final_after_model.txt"

        latest_whisper.write_text(whisper_raw + "\n", encoding="utf-8")
        latest_rules.write_text(pre_ai + "\n", encoding="utf-8")
        latest_final.write_text(final_lines + "\n", encoding="utf-8")

        return {
            "folder": str(run_dir.resolve()),
            "root_debug_folder": str(DEBUG_DIFF_DIR.resolve()),
            "whisper_file": str(file_whisper.resolve()),
            "rules_file": str(file_rules.resolve()),
            "final_file": str(file_final.resolve()),
            "latest_whisper_file": str(latest_whisper.resolve()),
            "latest_final_file": str(latest_final.resolve()),
        }
    except Exception as e:
        print(f"Предупреждение при запис на debug_diff файлове: {e}")
        return {}


def run_transcription_worker(
    job_id: str,
    model_name: str,
    language: str,
    vad_filter: bool,
    prompt: Optional[str],
    condition_on_previous_text: bool = False,
    audio_preprocessing: bool = True,
    expanded_beam_search: bool = True,
    apply_postprocessing: bool = True,
    postprocess_mode: str = "local",
    local_ai_model: str = "zealod/BgGPT-Gemma-2-2.6B-IT-v1.0-int4-onnx-directml",
    gemini_api_key_source: str = "input",
    gemini_api_key: Optional[str] = None,
    gemini_model: str = "gemini-3.8-flash",
    engine: str = "auto",
    debug_mode: bool = True,
):
    """Фонова нишка за изпълнение на транскрипцията."""
    job = JOBS.get(job_id)
    if not job:
        return

    def update_progress(data: dict):
        if job_id in JOBS:
            JOBS[job_id].update(data)

    try:
        import gemini_corrector
        effective_gemini_key = gemini_corrector.get_effective_api_key(gemini_api_key_source, gemini_api_key)

        file_path = job["file_path"]
        result = transcriber.transcribe_audio(
            file_path=file_path,
            model_name=model_name,
            language=language,
            vad_filter=vad_filter,
            condition_on_previous_text=condition_on_previous_text,
            audio_preprocessing=audio_preprocessing,
            expanded_beam_search=expanded_beam_search,
            apply_postprocessing=apply_postprocessing,
            postprocess_mode=postprocess_mode,
            local_ai_model=local_ai_model,
            gemini_api_key=effective_gemini_key,
            gemini_model=gemini_model,
            initial_prompt=prompt if prompt else transcriber.DEFAULT_BG_PROMPT,
            progress_callback=update_progress,
            engine=engine,
        )
        debug_info = save_debug_diff_files(job, result)
        result["debug_mode"] = debug_mode
        result["debug_files"] = debug_info
        JOBS[job_id].update(result)
        JOBS[job_id]["status"] = "completed"
    except Exception as e:
        import traceback
        traceback.print_exc()
        if job_id in JOBS:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)
            JOBS[job_id]["message"] = f"Грешка при обработка: {str(e)}"


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Връща главната уеб страница."""
    index_file = TEMPLATES_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Whisper BG Transcriber</h1><p>index.html липсва.</p>")


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Качва аудио/видео файл и връща идентификатор на задачата."""
    allowed_exts = {
        ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma",
        ".mp4", ".mkv", ".webm", ".avi", ".mov"
    }
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдържан файлов формат '{ext}'. Поддържани: MP3, WAV, M4A, OGG, FLAC, MP4, MKV, WEBM."
        )

    job_id = str(uuid.uuid4())
    safe_filename = f"{job_id}{ext}"
    saved_path = UPLOADS_DIR / safe_filename

    # Записване на файла на диска
    with open(saved_path, "wb") as f:
        content = await file.read()
        f.write(content)

    duration = transcriber.get_audio_duration(str(saved_path))

    JOBS[job_id] = {
        "job_id": job_id,
        "original_filename": file.filename,
        "file_path": str(saved_path),
        "file_size": len(content),
        "duration": duration,
        "duration_formatted": format_seconds(duration),
        "created_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "status": "ready",
        "progress": 0,
        "message": "Файлът е качен успешно. Готов за транскрипция.",
        "segments": [],
        "full_text": "",
        "model_name": "small",
    }

    return {
        "job_id": job_id,
        "filename": file.filename,
        "duration": duration,
        "duration_formatted": format_seconds(duration),
        "file_size_mb": round(len(content) / (1024 * 1024), 2),
    }


@app.post("/api/youtube")
async def fetch_youtube_audio(req: YouTubeRequest):
    """Извлича аудио потока директно от YouTube (или друга платформа чрез yt-dlp) и създава задача."""
    import asyncio
    url = (req.url or "").strip()
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(
            status_code=400,
            detail="Моля, въведете валиден YouTube URL адрес (започващ с https://)."
        )

    job_id = str(uuid.uuid4())
    outtmpl = str(UPLOADS_DIR / f"{job_id}.%(ext)s")

    def _download():
        import yt_dlp
        ydl_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio[ext=mp3]/bestaudio/best",
            "outtmpl": outtmpl,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_path = ydl.prepare_filename(info)
            return info, downloaded_path

    try:
        info, downloaded_path = await asyncio.to_thread(_download)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Грешка при извличане от YouTube: {str(e)}"
        )

    saved_path = Path(downloaded_path)
    if not saved_path.exists():
        matches = list(UPLOADS_DIR.glob(f"{job_id}.*"))
        if matches:
            saved_path = matches[0]
        else:
            raise HTTPException(status_code=500, detail="Сваленият аудио файл не беше намерен на диска.")

    file_size = saved_path.stat().st_size
    duration = float(info.get("duration") or 0.0)
    if duration <= 0:
        duration = transcriber.get_audio_duration(str(saved_path))

    title = (info.get("title") or "YouTube_Audio").strip()
    channel = (info.get("uploader") or info.get("channel") or "").strip()
    display_name = f"{title}{saved_path.suffix}"

    JOBS[job_id] = {
        "job_id": job_id,
        "original_filename": display_name,
        "source_url": url,
        "channel": channel,
        "file_path": str(saved_path),
        "file_size": file_size,
        "duration": duration,
        "duration_formatted": format_seconds(duration),
        "created_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "status": "ready",
        "progress": 0,
        "message": "Аудиото от YouTube е извлечено успешно. Готово за транскрипция.",
        "segments": [],
        "full_text": "",
        "model_name": "large-v3-turbo",
    }

    return {
        "job_id": job_id,
        "filename": f"📺 {title}" + (f" ({channel})" if channel else ""),
        "raw_title": title,
        "duration": duration,
        "duration_formatted": format_seconds(duration),
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "source_url": url,
    }


@app.post("/api/transcribe")
async def start_transcription(req: TranscribeRequest):
    """Стартира процес на транскрибиране."""
    job = JOBS.get(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задачата не е намерена.")

    if job["status"] in ("loading_model", "transcribing"):
        return {"status": job["status"], "message": "Транскрипцията вече се изпълнява."}

    job["status"] = "loading_model"
    job["progress"] = 5
    job["model_name"] = req.model_name
    job["include_timestamps"] = req.include_timestamps
    job["debug_mode"] = req.debug_mode

    # Стартиране във фонова нишка
    t = threading.Thread(
        target=run_transcription_worker,
        args=(
            req.job_id,
            req.model_name,
            req.language,
            req.vad_filter,
            req.custom_prompt,
            req.condition_on_previous_text,
            req.audio_preprocessing,
            req.expanded_beam_search,
            req.apply_postprocessing,
            req.postprocess_mode,
            req.local_ai_model,
            req.gemini_api_key_source,
            req.gemini_api_key,
            req.gemini_model,
            req.engine,
            req.debug_mode,
        ),
        daemon=True,
    )
    t.start()

    return {"status": "started", "job_id": req.job_id}


@app.get("/api/system-info")
async def get_system_info():
    """Връща разпознатия хардуер за ускорение (NVIDIA CUDA, AMD Radeon DirectML или CPU)."""
    return transcriber.get_hardware_info()


@app.post("/api/gemini/models")
async def get_gemini_models(req: GeminiModelsRequest):
    """Извлича актуалния списък с налични модели за дадения API ключ чрез ModelService.ListModels."""
    import gemini_corrector
    key = gemini_corrector.get_effective_api_key(req.key_source, req.api_key)
    if not key:
        return {
            "success": False,
            "error": "Липсва въведен API ключ или не е зададена системна променлива GEMINI_API_KEY."
        }
    return gemini_corrector.fetch_available_models(key)


@app.post("/api/test-gemini")
async def test_gemini(req: TestGeminiRequest):
    """Тества валидността на Gemini API ключ или системната променлива GEMINI_API_KEY за конкретен модел."""
    import gemini_corrector
    key = gemini_corrector.get_effective_api_key(req.key_source, req.api_key)
    if not key:
        return {
            "success": False,
            "error": "Липсва въведен API ключ или не е зададена системна променлива GEMINI_API_KEY."
        }
    return gemini_corrector.test_gemini_connection(key, req.model_name)


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Връща текущия статус и междинни сегменти."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задачата не е намерена.")

    return {
        "job_id": job["job_id"],
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "latest_text": job.get("latest_text", ""),
        "segments": job.get("segments", []),
        "full_text": job.get("full_text", ""),
        "whisper_raw_text": job.get("whisper_raw_text", ""),
        "pre_ai_text": job.get("pre_ai_text", ""),
        "final_lines_text": job.get("final_lines_text", ""),
        "postprocess_mode": job.get("postprocess_mode", "local"),
        "ai_model_used": job.get("ai_model_used", ""),
        "debug_mode": job.get("debug_mode", True),
        "debug_files": job.get("debug_files", {}),
        "duration": job.get("duration", 0),
        "duration_formatted": job.get("duration_formatted", ""),
        "elapsed_seconds": job.get("elapsed_seconds", 0),
        "language": job.get("language", "bg"),
        "device_type": job.get("device_type"),
        "device_name": job.get("device_name"),
        "device_label": job.get("device_label"),
        "badge_class": job.get("badge_class"),
        "engine_used": job.get("engine_used"),
        "error": job.get("error"),
    }


def _find_diff_tool() -> Optional[tuple[str, str]]:
    """Открива инсталиран Meld, KDiff3 или WinMerge в Windows."""
    import shutil
    local_app = os.environ.get("LOCALAPPDATA", "")
    prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    prog_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    user_profile = os.environ.get("USERPROFILE", "")

    candidates = [
        ("Meld", shutil.which("meld") or ""),
        ("Meld", os.path.join(prog_files, "Meld", "Meld.exe")),
        ("Meld", os.path.join(prog_x86, "Meld", "Meld.exe")),
        ("Meld", os.path.join(local_app, "Programs", "Meld", "Meld.exe")),
        ("Meld", os.path.join(user_profile, "scoop", "apps", "meld", "current", "Meld.exe")),
        ("KDiff3", shutil.which("kdiff3") or ""),
        ("KDiff3", os.path.join(prog_files, "KDiff3", "kdiff3.exe")),
        ("KDiff3", os.path.join(prog_files, "KDiff3", "bin", "kdiff3.exe")),
        ("KDiff3", os.path.join(prog_x86, "KDiff3", "kdiff3.exe")),
        ("KDiff3", os.path.join(local_app, "Programs", "KDiff3", "kdiff3.exe")),
        ("KDiff3", os.path.join(user_profile, "scoop", "apps", "kdiff3", "current", "kdiff3.exe")),
        ("WinMerge", shutil.which("WinMergeU") or ""),
        ("WinMerge", os.path.join(prog_files, "WinMerge", "WinMergeU.exe")),
        ("WinMerge", os.path.join(prog_x86, "WinMerge", "WinMergeU.exe")),
        ("WinMerge", os.path.join(local_app, "Programs", "WinMerge", "WinMergeU.exe")),
    ]
    for name, path in candidates:
        if path and os.path.isfile(path):
            return name, path
    return None


@app.post("/api/debug/launch-diff/{job_id}")
async def launch_diff_tool(job_id: str, mode: str = "2way"):
    """
    Стартира автоматично KDiff3, Meld или WinMerge със записаните дебъг файлове:
    - mode="2way" (по подразбиране): сравнява 1_whisper_model.txt и 3_final_after_model.txt
    - mode="3way": сравнява 1_whisper_model.txt, 2_after_rules_before_ai.txt и 3_final_after_model.txt
    """
    import subprocess
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задачата не е намерена.")

    debug_files = job.get("debug_files") or {}
    f_whisper = debug_files.get("whisper_file")
    f_rules = debug_files.get("rules_file")
    f_final = debug_files.get("final_file")
    folder = debug_files.get("folder") or str(DEBUG_DIFF_DIR.resolve())

    if not f_whisper or not f_final or not os.path.isfile(f_whisper):
        debug_files = save_debug_diff_files(job, job)
        job["debug_files"] = debug_files
        f_whisper = debug_files.get("whisper_file")
        f_rules = debug_files.get("rules_file")
        f_final = debug_files.get("final_file")
        folder = debug_files.get("folder") or str(DEBUG_DIFF_DIR.resolve())

    found = _find_diff_tool()
    if found:
        tool_name, tool_path = found
        if mode == "3way" and f_rules and os.path.isfile(f_rules):
            cmd = [tool_path, f_whisper, f_rules, f_final]
        else:
            cmd = [tool_path, f_whisper, f_final]
        subprocess.Popen(cmd)
        return {
            "launched": True,
            "tool": tool_name,
            "folder": folder,
            "message": f"Стартиран е {tool_name} със сравнение на файловете!",
        }
    else:
        # Ако няма KDiff3/Meld в стандартните папки, отваряме директно папката в Windows Explorer
        try:
            if os.path.isdir(folder):
                os.startfile(folder)
        except Exception:
            pass
        return {
            "launched": False,
            "tool": None,
            "folder": folder,
            "message": f"Отворена е папката с файловете ({folder}). Можете да ги провлачите в KDiff3 или Meld.",
        }


@app.post("/api/debug/open-folder/{job_id}")
async def open_debug_folder(job_id: str):
    """Отваря папката debug_diff за конкретната транскрипция във File Explorer."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задачата не е намерена.")

    debug_files = job.get("debug_files") or {}
    folder = debug_files.get("folder")
    if not folder or not os.path.isdir(folder):
        debug_files = save_debug_diff_files(job, job)
        job["debug_files"] = debug_files
        folder = debug_files.get("folder") or str(DEBUG_DIFF_DIR.resolve())

    try:
        os.startfile(folder)
        return {"opened": True, "folder": folder}
    except Exception as e:
        return {"opened": False, "folder": folder, "error": str(e)}


@app.get("/api/audio/{job_id}")
async def stream_audio(job_id: str):
    """Предоставя каченото аудио за прослушване в браузъра."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задачата не е намерена.")

    file_path = Path(job["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Аудио файлът не е намерен.")

    def iterfile():
        with open(file_path, "rb") as f:
            while chunk := f.read(64 * 1024):
                yield chunk

    # Определяне на подходящ content-type
    ext = file_path.suffix.lower()
    content_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".mp4": "video/mp4",
        ".webm": "audio/webm",
        ".opus": "audio/ogg",
    }
    media_type = content_types.get(ext, "application/octet-stream")

    return StreamingResponse(iterfile(), media_type=media_type)


@app.get("/api/download/{job_id}/{export_format}")
async def download_file(job_id: str, export_format: str, timestamps: bool = False):
    """Генерира и сваля файл в избран формат: txt, docx, srt или дебъг файлове."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задачата не е намерена.")

    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Транскрипцията все още не е завършила.")

    segments = job.get("segments", [])
    original_stem = Path(job.get("original_filename", "transcription")).stem
    safe_base_name = "".join(c for c in original_stem if c.isalnum() or c in (" ", "-", "_")).strip() or "transcription"

    metadata = {
        "filename": job.get("original_filename", "audio"),
        "duration_formatted": job.get("duration_formatted", "-"),
        "model_name": job.get("model_name", "small"),
        "created_at": job.get("created_at", ""),
    }

    fmt = export_format.lower()
    if fmt == "txt":
        content = exporter.export_txt(segments, include_timestamps=timestamps)
        filename = f"{safe_base_name}.txt"
        return StreamingResponse(
            iter([content.encode("utf-8")]),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    elif fmt == "debug_whisper":
        content = (job.get("whisper_raw_text") or "") + "\n"
        filename = f"1_whisper_model_{safe_base_name}.txt"
        return StreamingResponse(
            iter([content.encode("utf-8")]),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    elif fmt == "debug_rules":
        content = (job.get("pre_ai_text") or "") + "\n"
        filename = f"2_after_rules_{safe_base_name}.txt"
        return StreamingResponse(
            iter([content.encode("utf-8")]),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    elif fmt == "debug_final":
        content = (job.get("final_lines_text") or job.get("full_text") or "") + "\n"
        filename = f"3_final_after_model_{safe_base_name}.txt"
        return StreamingResponse(
            iter([content.encode("utf-8")]),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    elif fmt == "docx":
        docx_buffer = exporter.export_docx(segments, metadata=metadata, include_timestamps=timestamps)
        filename = f"{safe_base_name}.docx"
        return StreamingResponse(
            docx_buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    elif fmt == "srt":
        content = exporter.export_srt(segments)
        filename = f"{safe_base_name}.srt"
        return StreamingResponse(
            iter([content.encode("utf-8")]),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    else:
        raise HTTPException(status_code=400, detail="Поддържаните формати за сваляне са: txt, docx, srt, debug_whisper, debug_rules, debug_final.")
