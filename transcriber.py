"""
Модул за зареждане на Faster-Whisper и DirectML енджин за транскрибиране на аудио/видео файлове на български език.
Поддържа:
- NVIDIA CUDA (чрез CTranslate2)
- AMD Radeon & Intel GPU (чрез Microsoft DirectML / ONNX Runtime)
- CPU мултинишков режим (Fallback)
"""

import os
import sys
import re
import time
import io
import gc
import subprocess
from typing import Dict, Any, Callable, Optional, List, Tuple

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import av
import numpy as np
from faster_whisper import WhisperModel
import faster_whisper.audio
import bg_corrector

# Пач за съвместимост между по-нови версии на PyAV и faster-whisper
def _safe_decode_audio(input_file, sampling_rate=16000, split_stereo=False):
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono" if not split_stereo else "stereo",
        rate=sampling_rate,
    )
    raw_buffer = io.BytesIO()
    dtype = None

    try:
        container = av.open(input_file, mode="r", metadata_errors="ignore")
    except TypeError:
        container = av.open(input_file, mode="r")

    with container:
        frames = container.decode(audio=0)
        frames = faster_whisper.audio._ignore_invalid_frames(frames)
        frames = faster_whisper.audio._group_frames(frames, 500000)
        frames = faster_whisper.audio._resample_frames(frames, resampler)

        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)

    del resampler
    gc.collect()

    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)
    audio = audio.astype(np.float32) / 32768.0

    if split_stereo:
        return audio[0::2], audio[1::2]
    return audio

faster_whisper.audio.decode_audio = _safe_decode_audio

# Изключване на предупреждения за HF Hub symlinks в Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def get_models_dir() -> str:
    """Връща пътя до локалната папка с модели."""
    env_dir = os.environ.get("WHISPER_MODELS_DIR")
    if env_dir and os.path.exists(env_dir):
        return env_dir
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    m_dir = os.path.join(base, "models")
    os.makedirs(m_dir, exist_ok=True)
    return m_dir

# Насочване на кеша на Hugging Face към папката models
_local_models_dir = get_models_dir()
os.environ.setdefault("HF_HUB_CACHE", _local_models_dir)
os.environ.setdefault("HF_HOME", _local_models_dir)

# Глобален кеш за моделите
_LOADED_MODELS: Dict[str, WhisperModel] = {}
_LOADED_ONNX_MODELS: Dict[str, Any] = {}


def unload_whisper_models() -> None:
    """Освобождава заредените Whisper модели от GPU/RAM паметта преди зареждане на голям 7B LLM."""
    global _LOADED_MODELS, _LOADED_ONNX_MODELS
    if _LOADED_MODELS or _LOADED_ONNX_MODELS:
        _LOADED_MODELS.clear()
        _LOADED_ONNX_MODELS.clear()
        import gc
        gc.collect()

# Препоръчителен български подсказващ контекст (initial prompt)
DEFAULT_BG_PROMPT = (
    "Това е аудиозапис на книжовен български език с правилен правопис, "
    "пълен и кратък член, препинателни знаци и запетаи."
)


def get_windows_gpu_names() -> List[str]:
    """Извлича имената на наличните видеокарти в Windows през CIM/WMI."""
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
        ]
        output = subprocess.check_output(cmd, text=True, timeout=4).strip().splitlines()
        return [n.strip() for n in output if n.strip()]
    except Exception:
        return []


def get_hardware_info() -> Dict[str, Any]:
    """
    Разпознава активния наличен хардуер:
    1. NVIDIA CUDA (най-висока производителност за GeForce/RTX)
    2. Microsoft DirectML (за AMD Radeon RX 5000/6000/7000, Intel Arc, Intel Iris Xe)
    3. CPU (многонишков режим за процесор)
    """
    # 1. Проверка за NVIDIA CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return {
                "device_type": "cuda",
                "has_gpu": True,
                "device_name": gpu_name,
                "label": f"GPU: {gpu_name} (CUDA ⚡)",
                "short_label": f"GPU: {gpu_name}",
                "badge_class": "badge-cuda",
                "engine": "faster_whisper",
            }
    except Exception:
        pass

    # 2. Проверка за Microsoft DirectML (AMD Radeon / Intel / DirectX 12)
    try:
        import onnxruntime as ort
        if "DmlExecutionProvider" in ort.get_available_providers():
            names = get_windows_gpu_names()
            gpu_name = "DirectX 12 GPU"
            # Приоритет на дискретна видеокарта
            for n in names:
                if any(b in n.lower() for b in ["radeon", "geforce", "rtx", "gtx", "arc"]):
                    gpu_name = n
                    break
            else:
                if names:
                    gpu_name = names[0]

            return {
                "device_type": "directml",
                "has_gpu": True,
                "device_name": gpu_name,
                "label": f"GPU: {gpu_name} (DirectML ⚡)",
                "short_label": f"GPU: {gpu_name}",
                "badge_class": "badge-directml",
                "engine": "directml",
            }
    except Exception:
        pass

    # 3. CPU режим
    cpu_threads = min(12, os.cpu_count() or 4)
    return {
        "device_type": "cpu",
        "has_gpu": False,
        "device_name": f"CPU ({cpu_threads} нишки)",
        "label": f"Процесор: CPU ({cpu_threads} нишки)",
        "short_label": f"CPU ({cpu_threads} нишки)",
        "badge_class": "badge-cpu",
        "engine": "faster_whisper",
    }


def get_audio_duration(file_path: str) -> float:
    """Извлича точната продължителност на аудио/видео файла в секунди чрез PyAV."""
    try:
        with av.open(file_path) as container:
            if container.duration is not None:
                return float(container.duration / av.time_base)
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if audio_streams and audio_streams[0].duration is not None:
                stream = audio_streams[0]
                return float(stream.duration * stream.time_base)
    except Exception as e:
        print(f"Грешка при четене на продължителност с av: {e}")
    return 0.0


def get_faster_whisper_model(model_name: str = "small") -> WhisperModel:
    """Зарежда Faster-Whisper CTranslate2 модел."""
    global _LOADED_MODELS
    
    valid_models = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]
    if model_name not in valid_models:
        model_name = "large-v3"

    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    device = "cpu"
    compute_type = "int8"
    cpu_threads = min(12, os.cpu_count() or 4)

    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
    except Exception:
        pass

    models_dir = get_models_dir()
    print(f"Зареждане на Whisper модел '{model_name}' от '{models_dir}' на {device.upper()} (тип: {compute_type}, нишки: {cpu_threads})...")
    
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        download_root=models_dir,
    )
    
    _LOADED_MODELS[model_name] = model
    return model


def get_onnx_directml_model(model_name: str = "large-v3-turbo", vad_filter: bool = True) -> Any:
    """Зарежда Whisper ONNX модел оптимизиран за DirectML (AMD Radeon / Intel / DirectX 12)."""
    global _LOADED_ONNX_MODELS
    import onnx_asr

    mapping = {
        "large-v3-turbo": "onnx-community/whisper-large-v3-turbo",
        "large-v3": "onnx-community/whisper-large-v3-ONNX",
        "medium": "onnx-community/whisper-medium-ONNX",
        "small": "onnx-community/whisper-small",
        "base": "onnx-community/whisper-base",
        "tiny": "onnx-community/whisper-tiny",
    }
    model_id = mapping.get(model_name, "onnx-community/whisper-large-v3-turbo")
    cache_key = f"{model_id}_vad_{vad_filter}"
    if cache_key in _LOADED_ONNX_MODELS:
        return _LOADED_ONNX_MODELS[cache_key]

    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    print(f"Зареждане на DirectML ONNX Whisper модел '{model_id}'...")
    base_model = onnx_asr.load_model(model_id, providers=providers)
    if vad_filter:
        vad = onnx_asr.load_vad("silero", providers=providers)
        model = base_model.with_vad(vad)
    else:
        model = base_model.with_timestamps()

    _LOADED_ONNX_MODELS[cache_key] = model
    return model


def preprocess_audio(audio: np.ndarray, sampling_rate: int = 16000) -> np.ndarray:
    """Предварителна акустична обработка за максимална отчетливост."""
    if audio is None or len(audio) == 0:
        return audio

    data = audio.copy().astype(np.float32)
    data = data - np.mean(data)

    try:
        from scipy import signal
        b, a = signal.butter(2, 75.0, btype="highpass", fs=sampling_rate)
        data = signal.filtfilt(b, a, data).astype(np.float32)
    except Exception as e:
        print(f"Забележка при филтриране: {e}")

    rms = np.sqrt(np.mean(data ** 2))
    target_rms = 0.12
    if rms > 1e-4:
        gain = min(target_rms / rms, 5.0)
        data = data * gain

    peak = np.max(np.abs(data))
    if peak > 0.95:
        data = data * (0.95 / peak)

    return data


def transcribe_audio_directml(
    audio_data: np.ndarray,
    model_name: str = "large-v3-turbo",
    language: str = "bg",
    vad_filter: bool = True,
    total_duration: float = 0.0,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    postprocess_mode: str = "local",
    start_time: float = 0.0,
    hardware_label: str = "DirectML GPU",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Изпълнява транскрипция чрез DirectML (AMD Radeon / Intel / DirectX)."""
    model = get_onnx_directml_model(model_name, vad_filter=vad_filter)
    lang_code = language if language != "auto" else "bg"

    collected_segments = []
    full_text_parts = []

    if vad_filter:
        results_iter = model.recognize(audio_data, language=lang_code)
        for i, seg in enumerate(results_iter):
            raw_text = seg.text.strip()
            if not raw_text:
                continue
            rules_text = bg_corrector.post_process_bulgarian_text(raw_text)
            cleaned_text = rules_text if postprocess_mode == "local" else raw_text
            seg_dict = {
                "id": i,
                "start": round(float(seg.start), 2),
                "end": round(float(seg.end), 2),
                "raw_text": raw_text,
                "pre_ai_text": rules_text,
                "text": cleaned_text,
            }
            collected_segments.append(seg_dict)
            full_text_parts.append(cleaned_text)

            if total_duration > 0:
                calc_pct = min(95, 10 + int((float(seg.end) / total_duration) * 85))
            else:
                calc_pct = min(92, 10 + len(collected_segments) * 3)

            if progress_callback:
                progress_callback({
                    "status": "transcribing",
                    "progress": calc_pct,
                    "message": f"Транскрибиране ({hardware_label} • {calc_pct}%): {cleaned_text[:35]}...",
                    "latest_text": cleaned_text,
                    "segments": collected_segments,
                    "duration": total_duration,
                    "elapsed_seconds": round(time.time() - start_time, 1),
                })
    else:
        # Нарязване на 30-секундни сегменти
        sr = 16000
        chunk_len = sr * 30
        num_chunks = int(np.ceil(len(audio_data) / chunk_len))
        for i in range(num_chunks):
            chunk = audio_data[i * chunk_len : (i + 1) * chunk_len]
            if len(chunk) < sr * 0.4:
                continue
            chunk_start = i * 30.0
            chunk_end = min(total_duration, (i + 1) * 30.0)
            res = model.recognize(chunk, language=lang_code)
            raw_text = res.text.strip() if hasattr(res, "text") else str(res).strip()
            if not raw_text:
                continue
            rules_text = bg_corrector.post_process_bulgarian_text(raw_text)
            cleaned_text = rules_text if postprocess_mode == "local" else raw_text
            seg_dict = {
                "id": i,
                "start": round(chunk_start, 2),
                "end": round(chunk_end, 2),
                "raw_text": raw_text,
                "pre_ai_text": rules_text,
                "text": cleaned_text,
            }
            collected_segments.append(seg_dict)
            full_text_parts.append(cleaned_text)

            if total_duration > 0:
                calc_pct = min(95, 10 + int((chunk_end / total_duration) * 85))
            else:
                calc_pct = min(92, 10 + len(collected_segments) * 4)

            if progress_callback:
                progress_callback({
                    "status": "transcribing",
                    "progress": calc_pct,
                    "message": f"Транскрибиране ({hardware_label} • {calc_pct}%): {cleaned_text[:35]}...",
                    "latest_text": cleaned_text,
                    "segments": collected_segments,
                    "duration": total_duration,
                    "elapsed_seconds": round(time.time() - start_time, 1),
                })

    return collected_segments, full_text_parts


def join_whisper_segments_for_llm(parts: List[str]) -> str:
    """
    Сглобява времевите сегменти от Whisper в единен свързан текст за LLM коректора,
    като премахва фалшиви точки и фалшиви главни букви, породени от паузи по средата на изречението.
    """
    cleaned = [p.strip() for p in parts if p and p.strip()]
    if not cleaned:
        return ""

    for i in range(1, len(cleaned)):
        prev = cleaned[i - 1]
        curr = cleaned[i]

        # 1. Ако предходният сегмент завършва с единична точка, а текущият започва с малка буква,
        # точката е фалшива пауза по средата на изречението.
        if prev.endswith(".") and not prev.endswith("..") and curr and curr[0].islower():
            cleaned[i - 1] = prev[:-1].rstrip()
            prev = cleaned[i - 1]

        # 2. Ако предходният сегмент НЕ завършва с препинателен знак за край на изречение (.!?…:),
        # премахваме само фалшива единична точка в края на текущата изолирана дума
        if prev and prev[-1] not in ".!?…:":
            if re.match(r"^[А-ЯA-Z][а-яa-z]+\.$", curr):
                curr = curr[:-1]
            cleaned[i] = curr

    return " ".join(cleaned).strip()


def _apply_corrected_text_to_segments(
    segments: List[Dict[str, Any]], corrected_text: str
) -> List[Dict[str, Any]]:
    """
    Разпределя коригирания от AI текст обратно по времевите сегменти чрез difflib подравняване,
    така че границите на всеки сегмент/ред да останат 100% синхронизирани за KDiff3 / Meld и субтитри.
    """
    if not segments or not corrected_text or not corrected_text.strip():
        return segments

    new_words = corrected_text.strip().split()
    if not new_words:
        return segments

    if len(segments) == 1:
        segments[0]["text"] = corrected_text.strip()
        return segments

    import difflib

    def _norm(word: str) -> str:
        return word.lower().strip(".,!?;:\"'()[]{}«»„“”-—–…")

    all_orig_words: List[str] = []
    word_to_seg: List[int] = []
    for seg_idx, seg in enumerate(segments):
        seg_words = seg.get("text", "").split()
        if not seg_words:
            seg_words = [""]
        for w in seg_words:
            all_orig_words.append(w)
            word_to_seg.append(seg_idx)

    orig_norm = [_norm(w) for w in all_orig_words]
    new_norm = [_norm(w) for w in new_words]

    matcher = difflib.SequenceMatcher(None, orig_norm, new_norm, autojunk=False)
    seg_buckets: List[List[str]] = [[] for _ in range(len(segments))]

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if j1 == j2:
            continue
        if tag == "equal":
            for offset in range(j2 - j1):
                orig_idx = min(i1 + offset, len(word_to_seg) - 1)
                seg_buckets[word_to_seg[orig_idx]].append(new_words[j1 + offset])
        elif tag == "replace":
            orig_span = max(1, i2 - i1)
            new_span = j2 - j1
            for offset in range(new_span):
                mapped_orig = i1 + int((offset / new_span) * orig_span)
                mapped_orig = min(mapped_orig, len(word_to_seg) - 1)
                seg_buckets[word_to_seg[mapped_orig]].append(new_words[j1 + offset])
        elif tag == "insert":
            target_orig = min(i1, len(word_to_seg) - 1) if i1 < len(word_to_seg) else max(0, len(word_to_seg) - 1)
            if i1 > 0 and i1 <= len(word_to_seg):
                target_orig = i1 - 1
            target_seg = word_to_seg[target_orig]
            for j in range(j1, j2):
                seg_buckets[target_seg].append(new_words[j])

    for i, seg in enumerate(segments):
        if seg_buckets[i]:
            seg["text"] = " ".join(seg_buckets[i])

    return segments


def transcribe_audio(
    file_path: str,
    model_name: str = "large-v3",
    language: str = "bg",
    vad_filter: bool = True,
    condition_on_previous_text: bool = False,
    audio_preprocessing: bool = True,
    expanded_beam_search: bool = True,
    temperature: float = 0.0,
    apply_postprocessing: bool = True,
    postprocess_mode: str = "local",
    local_ai_model: str = "zealod/BgGPT-Gemma-2-2.6B-IT-v1.0-int4-onnx-directml",
    gemini_api_key: Optional[str] = None,
    gemini_model: str = "gemini-3.8-flash",
    initial_prompt: Optional[str] = DEFAULT_BG_PROMPT,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    engine: str = "auto",
) -> Dict[str, Any]:
    """
    Главен метод за транскрипция на аудио/видео с автоматичен избор на най-бързия хардуерен енджин.
    """
    start_time = time.time()
    total_duration = get_audio_duration(file_path)
    hardware_info = get_hardware_info()

    # Избор на енджин
    effective_engine = engine
    if effective_engine == "auto":
        effective_engine = hardware_info["engine"]

    if progress_callback:
        progress_callback({
            "status": "loading_model",
            "progress": 5,
            "message": f"Зареждане на {model_name} на {hardware_info['label']}...",
            "duration": total_duration,
            "segments": [],
        })

    # Декодиране на аудиото
    audio_data = faster_whisper.audio.decode_audio(file_path, sampling_rate=16000)
    if audio_preprocessing:
        audio_data = preprocess_audio(audio_data, sampling_rate=16000)

    if not apply_postprocessing:
        postprocess_mode = "none"

    collected_segments = []
    full_text_parts = []
    used_engine = effective_engine
    detected_lang = language if language != "auto" else "bg"
    lang_probability = 99.0

    # Освобождаваме евентуално зареден LLM модел от предходна транскрипция, за да има пълна GPU памет за Whisper
    try:
        import qwen_corrector
        qwen_corrector.unload_local_ai_models()
    except Exception:
        pass

    # Опит за DirectML изпълнение
    if effective_engine == "directml":
        try:
            collected_segments, full_text_parts = transcribe_audio_directml(
                audio_data=audio_data,
                model_name=model_name,
                language=language,
                vad_filter=vad_filter,
                total_duration=total_duration,
                progress_callback=progress_callback,
                postprocess_mode=postprocess_mode,
                start_time=start_time,
                hardware_label=hardware_info["short_label"],
            )
            used_engine = "directml"
        except Exception as e:
            print(f"DirectML възникна проблем ({e}). Автоматичен fallback към Faster-Whisper CPU...")
            effective_engine = "faster_whisper"
            cpu_threads = min(12, os.cpu_count() or 4)
            hardware_info = {
                "device_type": "cpu",
                "has_gpu": False,
                "device_name": f"CPU ({cpu_threads} нишки) [Fallback]",
                "label": f"Процесор: CPU ({cpu_threads} нишки) • Fallback",
                "short_label": f"CPU ({cpu_threads} нишки)",
                "badge_class": "badge-cpu",
                "engine": "faster_whisper",
                "is_fallback": True,
                "fallback_reason": str(e),
            }
            if progress_callback:
                progress_callback({
                    "status": "transcribing",
                    "progress": 8,
                    "message": f"⚠️ DirectML не успя. Превключено към Процесор ({cpu_threads} нишки)...",
                    "duration": total_duration,
                    "segments": [],
                    "device_label": hardware_info["label"],
                    "device_name": hardware_info["device_name"],
                    "badge_class": hardware_info["badge_class"],
                })

    # Изпълнение с Faster-Whisper (CUDA или CPU)
    if effective_engine == "faster_whisper":
        model = get_faster_whisper_model(model_name)
        beam_size = 8 if expanded_beam_search else 5
        patience = 2.0 if expanded_beam_search else 1.0

        if progress_callback:
            progress_callback({
                "status": "transcribing",
                "progress": 10,
                "message": f"Транскрибирането започна ({hardware_info['label']})...",
                "duration": total_duration,
                "segments": [],
            })

        segments_generator, info = model.transcribe(
            audio_data,
            language=language if language != "auto" else None,
            task="transcribe",
            beam_size=beam_size,
            best_of=beam_size,
            patience=patience,
            repetition_penalty=1.1,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500),
            initial_prompt=initial_prompt if initial_prompt else None,
            condition_on_previous_text=condition_on_previous_text,
            temperature=temperature,
            compression_ratio_threshold=2.4,
            hallucination_silence_threshold=2.0,
        )

        detected_lang = info.language
        lang_probability = round(info.language_probability * 100, 1)

        for seg in segments_generator:
            raw_text = seg.text.strip()
            if not raw_text:
                continue
            rules_text = bg_corrector.post_process_bulgarian_text(raw_text)
            cleaned_text = rules_text if postprocess_mode == "local" else raw_text
            seg_dict = {
                "id": seg.id,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "raw_text": raw_text,
                "pre_ai_text": rules_text,
                "text": cleaned_text,
            }
            collected_segments.append(seg_dict)
            full_text_parts.append(cleaned_text)

            if total_duration > 0:
                calc_pct = min(95, 10 + int((seg.end / total_duration) * 85))
            else:
                calc_pct = min(92, 10 + len(collected_segments) * 3)

            if progress_callback:
                progress_callback({
                    "status": "transcribing",
                    "progress": calc_pct,
                    "message": f"Транскрибиране ({hardware_info['short_label']} • {calc_pct}%): {cleaned_text[:35]}...",
                    "latest_text": cleaned_text,
                    "segments": collected_segments,
                    "duration": total_duration,
                    "elapsed_seconds": round(time.time() - start_time, 1),
                })
        used_engine = "faster_whisper"

    full_text = join_whisper_segments_for_llm(full_text_parts) if postprocess_mode in ("qwen", "gemini") else " ".join(full_text_parts).strip()
    ai_model_used = "none"
    if postprocess_mode == "local":
        ai_model_used = "local_rules"

    # Пост-процесинг с локален AI (BgGPT / Qwen DirectML GPU)
    if postprocess_mode == "qwen" and full_text:
        try:
            # Освобождаваме Whisper от GPU паметта преди зареждане на локалния LLM (особено важно за 7B моделите)
            unload_whisper_models()
            import qwen_corrector
            meta = qwen_corrector.get_model_meta(local_ai_model)
            ai_model_used = meta["short_name"]
            if progress_callback:
                progress_callback({
                    "status": "postprocessing",
                    "progress": 96,
                    "message": f"Прилагане на локален AI коректор {meta['short_name']} (DirectML ⚡)...",
                    "segments": collected_segments,
                    "duration": total_duration,
                })
            qwen_result = qwen_corrector.correct_text_with_qwen(
                full_text,
                model_id=local_ai_model,
                progress_callback=progress_callback,
                total_duration=total_duration,
            )
            if qwen_result and qwen_result.strip():
                full_text = qwen_result.strip()
                collected_segments = _apply_corrected_text_to_segments(collected_segments, full_text)
        except Exception as e:
            print(f"Предупреждение при DirectML AI редакция: {e}")

    # Пост-процесинг с Google Gemini, ако е избран
    if postprocess_mode == "gemini" and full_text and gemini_api_key:
        ai_model_used = gemini_model
        if progress_callback:
            progress_callback({
                "status": "postprocessing",
                "progress": 96,
                "message": f"Прилагане на интелигентна редакция с Google Gemini ({gemini_model})...",
                "segments": collected_segments,
                "duration": total_duration,
            })
        try:
            import gemini_corrector
            gemini_result = gemini_corrector.correct_text_with_gemini(
                full_text,
                api_key=gemini_api_key,
                model_name=gemini_model
            )
            if gemini_result and gemini_result.strip():
                full_text = gemini_result.strip()
                collected_segments = _apply_corrected_text_to_segments(collected_segments, full_text)
        except Exception as e:
            print(f"Предупреждение при Gemini редакция: {e}")

    elapsed = round(time.time() - start_time, 2)

    whisper_raw_lines = "\n".join(s.get("raw_text", s.get("text", "")) for s in collected_segments)
    pre_ai_lines = "\n".join(s.get("pre_ai_text", s.get("text", "")) for s in collected_segments)
    final_lines = "\n".join(s.get("text", "") for s in collected_segments)

    return {
        "text": full_text,
        "full_text": full_text,
        "whisper_raw_text": whisper_raw_lines,
        "pre_ai_text": pre_ai_lines,
        "final_lines_text": final_lines,
        "postprocess_mode": postprocess_mode,
        "ai_model_used": ai_model_used,
        "segments": collected_segments,
        "language": detected_lang,
        "language_probability": lang_probability,
        "duration": total_duration,
        "duration_formatted": format_seconds(total_duration),
        "elapsed_seconds": elapsed,
        "device_type": hardware_info["device_type"],
        "device_name": hardware_info["device_name"],
        "device_label": hardware_info["label"],
        "badge_class": hardware_info["badge_class"],
        "engine_used": used_engine,
    }


def format_seconds(seconds: float) -> str:
    """Форматира секунди в четим вид мм:сс или чч:мм:сс."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
