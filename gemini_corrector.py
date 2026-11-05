"""
Модул за интеграция с Google Gemini API за интелигентен облачен пост-процесинг на български език.
Поддържа подаване на API ключ директно или през системна променлива GEMINI_API_KEY.
Включва актуалните модели (gemini-3.8-flash, gemini-3.5-flash-lite, gemini-4-argon)
и динамично откриване и извличане на модели чрез ModelService.ListModels.
"""

import os
import json
import requests
from typing import Optional, Dict, Any, List

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

SYSTEM_INSTRUCTION = (
    "Ти си експертен редактор на български текст, получен от автоматично разпознаване на реч (ASR). "
    "Получаваш целия транскрибиран текст наведнъж. Използвай глобалния контекст на целия текст, за да:\n"
    "1. Възстановиш думи, имена и абревиатури, които са записани фонетично грешно, разцепени на срички, слети или изписани на латиница, като се водиш от смисъла на изречението и от останалите места в текста, където същите имена или термини се споменават.\n"
    "2. Замениш фонетично близки, но безсмислени в контекста думи и русизми с правилните български книжовни думи, които пасват логически и граматически на изказа.\n"
    "3. Поправиш пунктуацията, главните букви, запетаите, звуковия закон за 'в/във' и 'с/със' и пълния/краткия член.\n"
    "Запази всички изречения от началото до края в първоначалния им ред, без да съкращаваш, без да преразказваш и без да пропускаш абзаци. "
    "Върни ЕДИНСТВЕНО коригирания пълен текст без никакви уводни думи, обяснения или коментари."
)


def get_effective_api_key(key_source: str = "input", manual_key: Optional[str] = None) -> Optional[str]:
    """Връща активния API ключ според избрания източник."""
    if key_source == "env":
        return os.environ.get("GEMINI_API_KEY", "").strip() or None
    if manual_key and manual_key.strip():
        return manual_key.strip()
    return os.environ.get("GEMINI_API_KEY", "").strip() or None


def format_model_label(model_id: str, display_name: str) -> str:
    """Генерира четлив и ясен етикет за падащото меню."""
    m = model_id.lower()
    disp = display_name.strip() if display_name else model_id
    if "3.8-flash" in m:
        return f"{disp} ({model_id}) • Препоръчителен • Бърз"
    if "3.5-flash-lite" in m or "3.5-flash" in m:
        return f"{disp} ({model_id}) • Свръхбърз"
    if "4-argon" in m or "gemini-4" in m:
        return f"{disp} ({model_id}) • Gemini 4 • Прецизен"
    if "flash" in m:
        return f"{disp} ({model_id}) • Flash"
    if "pro" in m:
        return f"{disp} ({model_id}) • Pro"
    return f"{disp} ({model_id})"


def model_priority_score(item: Dict[str, Any]) -> int:
    """Определя реда на подреждане на моделите."""
    m = item.get("id", "").lower()
    if "3.8-flash" in m:
        return 0
    if "3.5-flash-lite" in m:
        return 1
    if "3.5-flash" in m:
        return 2
    if "4-argon" in m:
        return 3
    if "2.5-flash" in m:
        return 4
    if "2.0-flash" in m:
        return 5
    if "flash" in m:
        return 6
    if "pro" in m:
        return 7
    return 10


def fetch_available_models(api_key: str) -> Dict[str, Any]:
    """
    Извиква ModelService.ListModels, за да извлече точния списък с валидни модели за този API ключ.
    """
    if not api_key or not api_key.strip():
        return {
            "success": False,
            "error": "Моля, първо въведете Gemini API ключ или настройте променливата GEMINI_API_KEY."
        }

    clean_key = api_key.strip()
    collected_models: List[Dict[str, Any]] = []
    seen_ids = set()
    last_error = ""
    discovered_version = "v1beta"

    # Опитваме първо v1beta, след това v1
    for version in ("v1beta", "v1"):
        url = f"https://generativelanguage.googleapis.com/{version}/models?key={clean_key}"
        try:
            resp = requests.get(url, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("models", []):
                    methods = m.get("supportedGenerationMethods", [])
                    # Трябва да поддържа generateContent
                    if "generateContent" in methods:
                        raw_name = m.get("name", "")
                        model_id = raw_name.replace("models/", "").strip()
                        if model_id and model_id not in seen_ids:
                            seen_ids.add(model_id)
                            disp = m.get("displayName", model_id)
                            desc = m.get("description", "")
                            collected_models.append({
                                "id": model_id,
                                "name": disp,
                                "label": format_model_label(model_id, disp),
                                "description": desc,
                                "version": version
                            })
                if collected_models:
                    discovered_version = version
                    break
            elif resp.status_code in (400, 403):
                try:
                    err_json = resp.json()
                    msg = err_json.get("error", {}).get("message", resp.text[:200])
                except Exception:
                    msg = resp.text[:200]
                return {
                    "success": False,
                    "error": f"Грешка от Google ({resp.status_code}): {msg}"
                }
            elif resp.status_code == 404:
                last_error = f"Грешка 404: Услугата за модели не бе открита за версия {version}."
            else:
                last_error = f"Грешка {resp.status_code}: {resp.text[:200]}"
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Времето за връзка с Google изтече (Timeout). Проверете вашата интернет връзка."
            }
        except Exception as e:
            last_error = f"Грешка при комуникация: {str(e)}"

    if collected_models:
        collected_models.sort(key=model_priority_score)
        default_model = collected_models[0]["id"]
        return {
            "success": True,
            "version": discovered_version,
            "models": collected_models,
            "model_ids": [m["id"] for m in collected_models],
            "default_model": default_model,
            "count": len(collected_models),
        }

    return {
        "success": False,
        "error": last_error or "Не бяха намерени активни модели за генериране на съдържание за този ключ."
    }


def test_gemini_connection(api_key: str, model_name: str = DEFAULT_GEMINI_MODEL) -> Dict[str, Any]:
    """
    Тества валидността на ключа и работоспособността на конкретно посочения модел.
    """
    if not api_key or not api_key.strip():
        return {"success": False, "error": "Липсва Gemini API ключ."}

    clean_key = api_key.strip()
    target_model = model_name.strip() if model_name else DEFAULT_GEMINI_MODEL

    # Пробваме посочения модел през v1beta и v1
    last_error = ""
    for version in ("v1beta", "v1"):
        url = f"https://generativelanguage.googleapis.com/{version}/models/{target_model}:generateContent?key={clean_key}"
        payload = {
            "contents": [{"parts": [{"text": "Тест за връзка. Отговори само с една дума: ОК"}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 10}
        }
        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=12
            )
            if resp.status_code == 200:
                return {
                    "success": True,
                    "message": f"Връзката е отлична! Моделът '{target_model}' отговори успешно.",
                    "active_model": target_model,
                }
            elif resp.status_code == 404:
                err_msg = ""
                try:
                    err_msg = resp.json().get("error", {}).get("message", "")
                except Exception:
                    pass
                last_error = f"Моделът '{target_model}' не бе намерен за API {version} (Грешка 404). {err_msg}".strip()
            elif resp.status_code in (400, 403):
                try:
                    err_data = resp.json()
                    msg = err_data.get("error", {}).get("message", resp.text[:200])
                except Exception:
                    msg = resp.text[:200]
                return {
                    "success": False,
                    "error": f"Грешка от Google API ({resp.status_code}): {msg}"
                }
            else:
                last_error = f"Грешка от Google ({resp.status_code}): {resp.text[:200]}"
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Времето за изчакване към Google изтече (Timeout). Проверете интернет връзката."}
        except Exception as e:
            last_error = str(e)

    return {
        "success": False,
        "error": f"{last_error}. Препоръка: Натиснете бутона 'Провери и изтегли наличните модели', за да изберете актуален модел.",
        "failed_model": target_model,
    }


def correct_text_with_gemini(
    text: str,
    api_key: str,
    model_name: str = DEFAULT_GEMINI_MODEL
) -> str:
    """
    Изпраща транскрибирания текст към Google Gemini за цялостна корекция.
    Ако посоченият модел върне 404, автоматично се опитва с резервни модели от акаунта.
    """
    if not text or not text.strip():
        return text

    if not api_key or not api_key.strip():
        raise ValueError("Липсва Gemini API ключ.")

    clean_key = api_key.strip()
    target_model = model_name.strip() if model_name else DEFAULT_GEMINI_MODEL

    candidates = [target_model]
    for fallback in (DEFAULT_GEMINI_MODEL, "gemini-3.5-flash-lite", "gemini-4-argon"):
        if fallback not in candidates:
            candidates.append(fallback)

    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "contents": [
            {"parts": [{"text": text}]}
        ],
        "generationConfig": {
            "temperature": 0.2,
        }
    }

    last_error = ""
    for candidate in candidates:
        for version in ("v1beta", "v1"):
            url = f"https://generativelanguage.googleapis.com/{version}/models/{candidate}:generateContent?key={clean_key}"
            try:
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=60
                )
                if response.status_code == 200:
                    data = response.json()
                    res_candidates = data.get("candidates", [])
                    if res_candidates:
                        parts = res_candidates[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"].strip()
                    return text
                elif response.status_code == 404:
                    last_error = f"404 за {candidate} ({version})"
                    continue
                else:
                    last_error = f"Грешка {response.status_code}: {response.text[:200]}"
            except Exception as e:
                last_error = str(e)

    # Fallback към открит модел от ListModels
    info = fetch_available_models(clean_key)
    if info.get("success") and info.get("models"):
        first_m = info["models"][0]
        discovered_id = first_m["id"] if isinstance(first_m, dict) else str(first_m)
        ver = info.get("version", "v1beta")
        url = f"https://generativelanguage.googleapis.com/{ver}/models/{discovered_id}:generateContent?key={clean_key}"
        try:
            resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                res_candidates = data.get("candidates", [])
                if res_candidates:
                    parts = res_candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
        except Exception as e:
            last_error = str(e)

    raise RuntimeError(f"Неуспешна заявка към Gemini. Последна грешка: {last_error}")
