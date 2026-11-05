"""
Модул за локална езикова корекция чрез BgGPT (INSAIT) и Qwen 2.5 върху Microsoft DirectML ONNX Runtime GenAI.
Поддържа всички 6 конвертирани модела BgGPT (2.6B и 7B в INT4, HYBRID и INT8) + Qwen 2.5 7B INT4.
Моделите се свалят автоматично от Hugging Face при първо поискване и се кешират в папка models/.
"""

import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

import bg_corrector
import transcriber

_LOADED_MODELS: Dict[str, Dict[str, Any]] = {}

DEFAULT_LOCAL_AI_MODEL = "zealod/BgGPT-Gemma-2-2.6B-IT-v1.0-int4-onnx-directml"

LOCAL_AI_MODELS: Dict[str, Dict[str, str]] = {
    "zealod/BgGPT-Gemma-2-2.6B-IT-v1.0-int4-onnx-directml": {
        "id": "zealod/BgGPT-Gemma-2-2.6B-IT-v1.0-int4-onnx-directml",
        "label": "BgGPT Gemma-2 2.6B • INT4 (2.51 GB • Препоръчителен и най-бърз ⚡)",
        "short_name": "BgGPT 2.6B (INT4)",
        "template": "gemma2",
    },
    "zealod/BgGPT-Gemma-2-2.6B-IT-v1.0-hybrid-onnx-directml": {
        "id": "zealod/BgGPT-Gemma-2-2.6B-IT-v1.0-hybrid-onnx-directml",
        "label": "BgGPT Gemma-2 2.6B • HYBRID INT4+FP16 (3.34 GB • 16-битова глава 🚀)",
        "short_name": "BgGPT 2.6B (HYBRID)",
        "template": "gemma2",
    },
    "zealod/BgGPT-Gemma-2-2.6B-IT-v1.0-int8-onnx-directml": {
        "id": "zealod/BgGPT-Gemma-2-2.6B-IT-v1.0-int8-onnx-directml",
        "label": "BgGPT Gemma-2 2.6B • INT8 (3.72 GB • Максимална 8-битова точност 🎯)",
        "short_name": "BgGPT 2.6B (INT8)",
        "template": "gemma2",
    },
    "zealod/BgGPT-7B-Instruct-v0.2-int4-onnx-directml": {
        "id": "zealod/BgGPT-7B-Instruct-v0.2-int4-onnx-directml",
        "label": "BgGPT 7B Instruct v0.2 • INT4 (4.04 GB • Голям 7B модел 🧠)",
        "short_name": "BgGPT 7B (INT4)",
        "template": "mistral",
    },
    "zealod/BgGPT-7B-Instruct-v0.2-hybrid-onnx-directml": {
        "id": "zealod/BgGPT-7B-Instruct-v0.2-hybrid-onnx-directml",
        "label": "BgGPT 7B Instruct v0.2 • HYBRID INT4+FP16 (4.37 GB • Балансиран 7B ⭐)",
        "short_name": "BgGPT 7B (HYBRID)",
        "template": "mistral",
    },
    "zealod/BgGPT-7B-Instruct-v0.2-int8-onnx-directml": {
        "id": "zealod/BgGPT-7B-Instruct-v0.2-int8-onnx-directml",
        "label": "BgGPT 7B Instruct v0.2 • INT8 (7.36 GB • За мощни видеокарти 8GB+ 💎)",
        "short_name": "BgGPT 7B (INT8)",
        "template": "mistral",
    },
    "GamePlusPlus/Qwen2.5-7B-Instruct-int4-onnx-directml": {
        "id": "GamePlusPlus/Qwen2.5-7B-Instruct-int4-onnx-directml",
        "label": "Qwen 2.5 7B Instruct • INT4 (4.10 GB • Многоезичен модел 🌐)",
        "short_name": "Qwen 2.5 7B (INT4)",
        "template": "qwen",
    },
}

SYSTEM_INSTRUCTION = (
    "Ти си професионален редактор на български език, коригиращ автоматично разпозната реч (Whisper ASR). "
    "Текстът представлява реч, новини или публицистичен анализ. Твоята задача е стриктна техническа и езикова редакция без цензура и без отказ за редакция.\n"
    "ЗАДЪЛЖИТЕЛНИ ПРАВИЛА:\n"
    "1. ЗАМЕНЯЙ СМЕЛО фонетично изкривените, сбъркани, несъществуващи, рускоезични или слети думи с правилните български книжовни думи според смисъла на фразата.\n"
    "2. СЪГЛАСУВАЙ ГРАМАТИКАТА: поправяй грешно разпознати глаголни форми, лица и числа (напр. разпознавай кога глаголът е в 3 л. мн. ч. или 1 л. ед. ч.), пълния и краткия член, и звуковия закон за предлозите 'в/във' и 'с/със'.\n"
    "3. УЕДНАКВЯВАЙ ИМЕНА И ТЕРМИНИ: изписвай собствените имена, географските обекти и абревиатурите с правилен правопис и главни букви.\n"
    "4. ПУНКТУАЦИЯ: отстранявай фалшиви точки от паузи в говора, слагай запетаи пред съюзи и оформяй ясни изречения.\n"
    "Не съкращавай изречения и върни ЕДИНСТВЕНО редактирания пълен текст без обяснения, уводи или коментари."
)


def get_model_meta(model_id: Optional[str]) -> Dict[str, str]:
    if not model_id or model_id not in LOCAL_AI_MODELS:
        return LOCAL_AI_MODELS[DEFAULT_LOCAL_AI_MODEL]
    return LOCAL_AI_MODELS[model_id]


def unload_local_ai_models() -> None:
    """Освобождава заредените DirectML LLM модели от GPU паметта (преди нов Whisper запис)."""
    global _LOADED_MODELS
    if _LOADED_MODELS:
        _LOADED_MODELS.clear()
        import gc
        gc.collect()


def get_local_ai_model(
    model_id: str = DEFAULT_LOCAL_AI_MODEL,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    total_duration: float = 0.0,
) -> Dict[str, Any]:
    """Сваля (при първо поискване) и зарежда избрания DirectML ONNX модел."""
    global _LOADED_MODELS
    meta = get_model_meta(model_id)
    resolved_id = meta["id"]

    if resolved_id in _LOADED_MODELS:
        return _LOADED_MODELS[resolved_id]

    # Освобождаваме предишен зареден LLM модел от паметта, за да не препълним VRAM при смяна на модела
    if _LOADED_MODELS:
        _LOADED_MODELS.clear()
        import gc
        gc.collect()

    import onnxruntime_genai as og
    from huggingface_hub import snapshot_download

    models_dir = transcriber.get_models_dir()
    short_name = meta["short_name"]

    if progress_callback:
        progress_callback({
            "status": "postprocessing",
            "progress": 95,
            "message": f"Проверка / сваляне на {short_name} от Hugging Face (при първо пускане)...",
            "duration": total_duration,
        })

    print(f"Подготовка на локален AI модел {short_name} ({resolved_id}) в {models_dir}...")
    model_path = snapshot_download(repo_id=resolved_id, cache_dir=models_dir)

    if progress_callback:
        progress_callback({
            "status": "postprocessing",
            "progress": 96,
            "message": f"Зареждане на {short_name} върху DirectML GPU...",
            "duration": total_duration,
        })

    t0 = time.time()
    model = og.Model(model_path)
    tok = og.Tokenizer(model)
    stream = tok.create_stream()
    print(f"{short_name} зареден успешно върху DirectML за {time.time() - t0:.2f} сек.")

    resources = {
        "model": model,
        "tokenizer": tok,
        "stream": stream,
        "path": model_path,
        "meta": meta,
    }
    _LOADED_MODELS[resolved_id] = resources
    return resources


_BG_STOPWORDS = {
    "това", "което", "като", "много", "защото", "когато", "обаче", "също",
    "само", "вече", "може", "така", "един", "една", "едно", "тези", "този",
    "тази", "нещо", "няма", "тука", "след", "през", "пред", "около",
    "между", "всички", "всичко", "всеки", "всяка", "които", "която", "който",
    "дори", "още", "просто", "малко", "доста", "понеже", "затова", "точно",
    "разбира", "всъщност", "примерно", "наистина", "винаги", "никога", "някой",
    "някоя", "някакъв", "някаква", "други", "друга", "друго", "него", "нея",
    "изключително", "особено", "място", "различни", "толкова", "слагам", "сега",
    "част", "интересна", "интересно", "интересни", "започвам", "започваме",
    "отново", "напиши", "както", "често", "продължаваме", "популярните",
    "популярна", "популярен", "качествени", "първо", "второ", "трето",
    "четвърто", "пето", "трябва", "където", "докато", "заради", "няколко",
    "видео", "видеото", "видеа", "канал", "коментарите", "коментар", "началото",
    "момента", "година", "годината", "месеци", "време", "начин", "неща", "хора",
}


def _apply_self_corpus_consistency(text: str) -> str:
    """
    Самообучаващ се контекстен филтър (БЕЗ статичен речник):
    Използва статистиката на думите от САМИЯ текущ транскрипт, за да поправи местата, където Whisper е:
    1) Изгълтал началната гласна 'и-' след 'най-' или 'по-' (напр. 'най-диотскато' при налично 'идиотска' в същия запис,
       или 'най-сползваните' при налично 'използване' в същия запис).
    2) Разцепил представка от честа за записа дума (напр. 'при мамката' при многократно срещано 'примамка'/'примамки' в същия запис).
    """
    if not text:
        return text

    # Извличаме всички думи (и корените след дефис като 'по-идиотска' -> 'идиотска') от самия документ
    raw_tokens = re.findall(r"[А-Яа-я]{4,}", text.lower())
    word_counts: Dict[str, int] = {}
    stem5_set = set()
    stem6_counts: Dict[str, int] = {}

    for tok in raw_tokens:
        word_counts[tok] = word_counts.get(tok, 0) + 1
        if len(tok) >= 5:
            stem5_set.add(tok[:5])
        if len(tok) >= 6:
            s6 = tok[:6]
            stem6_counts[s6] = stem6_counts.get(s6, 0) + 1

    # 1. Възстановяване на изгълтано 'и-' / 'из-' след 'най-' и 'по-', ако пълният корен съществува в същия запис
    def _fix_superlative_elision(m: re.Match) -> str:
        prefix = m.group(1)
        rest = m.group(2)
        rest_low = rest.lower()
        if len(rest_low) >= 4 and rest_low[:5] not in stem5_set:
            cand_i = "и" + rest_low
            if cand_i[:5] in stem5_set:
                return f"{prefix}-и{rest}"
            if rest_low.startswith("с"):
                cand_iz = "из" + rest_low[1:]
                if cand_iz[:5] in stem5_set:
                    return f"{prefix}-из{rest[1:]}"
        return m.group(0)

    text = re.sub(r"\b(най|по|Най|По)-([а-я]{4,})\b", _fix_superlative_elision, text)

    # 2. Сливане на фалшиво разцепени от Whisper представки ('при мамката' -> 'примамката'),
    # когато слетият 6-буквен корен се среща като цяла дума в същия запис.
    def _fix_split_prefix(m: re.Match) -> str:
        prep = m.group(1)
        noun = m.group(2)
        merged_low = (prep + noun).lower()
        if len(merged_low) >= 6:
            s6 = merged_low[:6]
            if stem6_counts.get(s6, 0) >= 2:
                return prep + noun
            # Ако има 1 сгрешена съгласна в корена (напр. 'при малка' при чест корен 'примам' >= 3 пъти в записа)
            if len(merged_low) >= 7:
                for known_word, cnt in word_counts.items():
                    if cnt >= 3 and len(known_word) == len(merged_low) and known_word[:5] == merged_low[:5] and known_word[-2:] == merged_low[-2:]:
                        diffs = sum(1 for a, b in zip(known_word, merged_low) if a != b)
                        if diffs == 1:
                            return (prep[0] + known_word[1:]) if prep[0].isupper() else known_word
        return m.group(0)

    text = re.sub(
        r"\b(при|по|на|за|от|из|под|над|раз|пре|про|съ|се)\s+([а-я]{4,})\b",
        _fix_split_prefix,
        text,
        flags=re.IGNORECASE,
    )

    return text


def _extract_global_vocabulary(text: str, max_terms: int = 45) -> str:
    """
    Извлича динамично от целия транскрипт най-често срещаните ключови термини, имена и абревиатури (~35-45 думи).
    Така всеки отделен пасаж получава глобалния речник на целия запис само с ~50 токена,
    без да претоварва DirectML буфера и Windows TDR таймера (0x887A0006).
    """
    words = re.findall(r"[А-Яа-яA-Za-z]{3,}", text)
    if not words:
        return ""

    freq: Dict[str, int] = {}
    proper_forms: Dict[str, str] = {}
    for w in words:
        low = w.lower()
        if low in _BG_STOPWORDS or len(low) < 4:
            continue
        freq[low] = freq.get(low, 0) + 1
        if w[0].isupper() and low not in proper_forms:
            proper_forms[low] = w

    ranked = sorted(
        [k for k, cnt in freq.items() if cnt >= 2 or k in proper_forms],
        key=lambda k: (freq[k], len(k)),
        reverse=True,
    )[:max_terms]

    display_terms = [proper_forms.get(k, k) for k in ranked]
    return ", ".join(display_terms)


def _build_prompt(template_type: str, text: str, global_vocab: str = "") -> str:
    vocab_hint = (
        f"\nКлючови думи, термини и имена от записа (използвай ги, за да разпознаеш изкривените от ASR думи): {global_vocab}\n"
        if global_vocab
        else ""
    )
    user_body = (
        f"{SYSTEM_INSTRUCTION}{vocab_hint}\n\n"
        f"Поправи всички фонетични, лексикални и пунктуационни грешки в следния текст:\n{text}"
    )
    if template_type == "gemma2":
        return (
            f"<bos><start_of_turn>user\n"
            f"{user_body}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    elif template_type == "mistral":
        return f"<s>[INST] {user_body} [/INST]"
    else:
        return (
            f"<|im_start|>system\n{SYSTEM_INSTRUCTION}{vocab_hint}<|im_end|>\n"
            f"<|im_start|>user\nПоправи всички фонетични, лексикални и пунктуационни грешки от разпознаването на реч в следния текст:\n{text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


def _split_into_chunks(text: str, max_words: int = 95) -> List[str]:
    """Разбива дългия текст на смислови блокове по изречения (до max_words думи), за да не се претоварва контекстът на LLM."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+|\n+", text) if s.strip()]
    if not sentences:
        sentences = [text.strip()]

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sent in sentences:
        words = sent.split()
        # Ако едно отделно изречение без точки е твърде дълго (над max_words), го разделяме по запетаи/брой думи
        if len(words) > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i : i + max_words]))
            continue

        if current_word_count + len(words) <= max_words:
            current_chunk.append(sent)
            current_word_count += len(words)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_word_count = len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def _sanitize_llm_output(corrected: str, original_clean: str) -> str:
    """Изчиства евентуални мета-коментари от LLM и гарантира, че няма изпуснати изречения."""
    if not corrected:
        return original_clean

    # Премахване на типични мета-уводи или заключения от чат модели
    corrected = re.sub(
        r"^(Ето\s+(го\s+)?коригирания[т]?\s+текст[^:]*:|Това\s+е\s+коригирания[т]?\s+текст[^:]*:|Коригиран\s+текст\s*:)\s*",
        "",
        corrected,
        flags=re.IGNORECASE,
    ).strip()
    corrected = re.sub(
        r"\s*(Надявам\s+се\s+този\s+текст[^.!]*[.!]|Ако\s+имате\s+нужда\s+от\s+допълнителн[^.!]*[.!]|Не\s+се\s+колебай[те]*\s+да\s+попита[ш|те][^.!] *[.!]).*$",
        "",
        corrected,
        flags=re.IGNORECASE,
    ).strip()
    corrected = corrected.replace("**", "").strip()

    orig_words = original_clean.split()
    new_words = corrected.split()
    # Защитен праг само при екстремно съкращаване (под 35% от думите)
    if len(orig_words) >= 15 and len(new_words) < len(orig_words) * 0.35:
        print(f"[DEBUG FALLBACK]: Оригинал {len(orig_words)} думи, Модел върна само {len(new_words)} думи. Връщане на оригинала.")
        return original_clean

    return corrected


def correct_chunk_local_ai(
    chunk: str,
    resources: Dict[str, Any],
    global_vocab: str = "",
) -> str:
    """
    Коригира смислов пасаж (~95 думи) с подаден глобален речник от целия запис.
    Динамично оразмерява KV-кеша според дължината на пасажа и незабавно освобождава VRAM паметта (del generator).
    """
    import onnxruntime_genai as og
    import gc

    model = resources["model"]
    tok = resources["tokenizer"]
    stream = resources["stream"]
    template_type = resources["meta"]["template"]

    prompt = _build_prompt(template_type, chunk, global_vocab=global_vocab)
    tokens = tok.encode(prompt)
    input_len = len(tokens)
    expected_out_tokens = max(50, len(tok.encode(chunk)))
    max_gen = max(100, int(expected_out_tokens * 1.35) + 32)
    # Точно оразмеряване на KV-кеша само за текущия пасаж (~350 токена = ~180MB вместо 1GB VRAM)
    total_max_len = input_len + max_gen

    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=total_max_len,
        do_sample=False,
        repetition_penalty=1.05,
    )

    generator = og.Generator(model, params)
    generator.append_tokens(tokens)

    output_pieces = []
    try:
        while not generator.is_done():
            generator.generate_next_token()
            tok_id = generator.get_next_tokens()[0]
            piece = stream.decode(tok_id)
            if piece in ("<end_of_turn>", "<eos>", "</s>", "<|im_end|>", "<|endoftext|>", "[/INST]"):
                break
            output_pieces.append(piece)
    finally:
        # Критично важно за DirectML: незабавно освобождаване на KV-кеша от VRAM след всеки пасаж
        del generator
        gc.collect()

    corrected = "".join(output_pieces).strip()
    corrected = re.sub(r"<\|im_end\|>|<\|endoftext\|>|<end_of_turn>|<eos>|</s>|\[/INST\]", "", corrected).strip()
    corrected = _sanitize_llm_output(corrected, chunk)
    return corrected


def correct_text_with_qwen(
    text: str,
    model_id: str = DEFAULT_LOCAL_AI_MODEL,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    total_duration: float = 0.0,
) -> str:
    """
    Главен метод за редакция на български текст чрез избран локален DirectML модел (BgGPT / Qwen).
    Извлича глобалния речник на целия запис и обработва текста на прецизни, безопасни за GPU TDR пасажи (~95 думи).
    """
    if not text or not text.strip():
        return text

    try:
        resources = get_local_ai_model(
            model_id=model_id,
            progress_callback=progress_callback,
            total_duration=total_duration,
        )
        short_name = resources["meta"]["short_name"]
        # Първо прилагаме самообучаващо се уеднаквяване спрямо речника на самия запис (без статични правила)
        text = _apply_self_corpus_consistency(text)
        global_vocab = _extract_global_vocabulary(text)
        chunks = _split_into_chunks(text, max_words=95)

        total_chunks = len(chunks)
        corrected_chunks = []
        for idx, chunk in enumerate(chunks, 1):
            if progress_callback:
                pct = min(99, 96 + int((idx / total_chunks) * 3))
                progress_callback({
                    "status": "postprocessing",
                    "progress": pct,
                    "message": f"Локален AI коректор {short_name} (DirectML ⚡ • част {idx}/{total_chunks})...",
                    "duration": total_duration,
                })

            try:
                res_chunk = correct_chunk_local_ai(
                    chunk,
                    resources,
                    global_vocab=global_vocab,
                )
                corrected_chunks.append(res_chunk)
            except Exception as chunk_err:
                print(f"DirectML предупреждение на пасаж {idx}/{total_chunks} ({chunk_err}). Презареждане на GPU сесията...")
                unload_local_ai_models()
                resources = get_local_ai_model(model_id=model_id)
                # Повторен опит с разделяне на пасажа на две по-малки части
                sub_chunks = _split_into_chunks(chunk, max_words=45)
                for sub in sub_chunks:
                    try:
                        corrected_chunks.append(correct_chunk_local_ai(sub, resources, global_vocab=global_vocab))
                    except Exception:
                        corrected_chunks.append(sub)
                        unload_local_ai_models()
                        resources = get_local_ai_model(model_id=model_id)

        result_text = " ".join(corrected_chunks).strip()
        return result_text if result_text else text

    except Exception as e:
        import traceback
        print(f"DirectML AI корекция ({model_id}) срещна грешка ({e}):")
        traceback.print_exc()
        unload_local_ai_models()
        return text
