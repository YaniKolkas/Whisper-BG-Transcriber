"""
Модул за 100% локална българска езикова корекция (Post-processing) без външни API ключове и без интернет.
Включва:
1. Строг звуков закон за предлозите "в/във" и "с/със"
2. Граматично членуване след предлози (пълен -> кратък член)
3. Синтактично членуване на подлог в началото на изречения (кратък -> пълен член за съществителни и прилагателни)
4. Пунктуационен енджин за автоматично поставяне на запетаи пред подчинителни и противопоставителни съюзи
5. Изчистване на Whisper артефакти, паразитни интервали и защита на десетични числа
"""

import re
from typing import List, Dict, Any

# Списък с предлози в българския език (след които пълен член е 100% граматически недопустим)
PREPOSITIONS = {
    "в", "във", "с", "със", "на", "от", "за", "при", "към", "през", "по",
    "зад", "под", "над", "между", "сред", "около", "край", "пред", "до",
    "без", "след", "против", "чрез", "вместо", "относно", "покрай", "срещу",
    "помежду", "спрямо", "върху", "въпреки", "поради", "из", "откъм", "вследствие",
    "съгласно", "посредством", "наред"
}

# Думи, които прекратяват действието на предложната група
PREP_TERMINATORS = {
    "и", "или", "но", "а", "ала", "обаче", "че", "защото", "понеже", "ако",
    "когато", "като", "докато", "е", "са", "беше", "бяха", "ще", "се", "да",
    "съм", "си", "сме", "сте", "има", "няма", "стана", "започна"
}

# Думи, чийто корен завършва на -ът / -ят (НЕ са членувани форми)
NON_MEMBER_ROOTS = {
    "път", "кът", "кмет", "плет", "смет", "ръст", "лист", "мост", "храст",
    "текст", "тест", "пръст", "вест", "чест", "свят", "брат", "зет", "вят"
}

# Чести съществителни от м.р., които при говор се чуват с кратък член, но като подлог изискват пълен член (-ът / -ят)
COMMON_MASCULINE_SUBJECTS = {
    "филма": "филмът",
    "закона": "законът",
    "човека": "човекът",
    "града": "градът",
    "народа": "народът",
    "света": "светът",
    "пазара": "пазарът",
    "проекта": "проектът",
    "плана": "планът",
    "доклада": "докладът",
    "проблема": "проблемът",
    "случая": "случаят",
    "отбора": "отборът",
    "мача": "мачът",
    "модела": "моделът",
    "въпроса": "въпросът",
    "сайта": "сайтът",
    "клиента": "клиентът",
    "договора": "договорът",
    "шофьора": "шофьорът",
    "лекаря": "лекарят",
    "учителя": "учителят",
    "директора": "директорът",
    "автора": "авторът",
    "собственика": "собственикът",
    "министъра": "министърът",
    "парламента": "парламентът",
    "президента": "президентът",
    "резултата": "резултатът",
    "процеса": "процесът",
    "факта": "фактът",
    "отговора": "отговорът",
    "живота": "животът",
    "бизнеса": "бизнесът",
    "номера": "номерът",
    "компютъра": "компютърът",
    "телефона": "телефонът",
    "екрана": "екранът",
    "влака": "влакът",
    "самолета": "самолетът",
    "автобуса": "автобусът",
    "магазина": "магазинът",
    "деня": "денят",
    "пътя": "пътят",
}

# Изключения за думи от женски род, завършващи на -ия (за да не се бъркат с мъжки прилагателни)
FEMININE_IA_EXCLUSIONS = {
    "българия", "софия", "франция", "русия", "гърция", "турция", "италия", "испания",
    "англия", "германия", "австрия", "австралия", "азия", "африка", "сесия", "комисия",
    "полиция", "партия", "информация", "реакция", "ситуация", "традиция", "история",
    "теория", "категория", "стратегия", "енергия", "химия", "биология", "магия",
    "поезия", "армия", "линия", "дискусия", "организация", "позиция", "функция",
    "операция", "комбинация", "провинция", "стихия", "орисия", "австрия", "европа"
}

# Подчинителни и относителни съюзи, изискващи запетая пред тях
CONJUNCTIONS_REQUIRING_COMMA = [
    "че", "защото", "тъй като", "понеже", "за да", "без да",
    "въпреки че", "макар че", "като че ли", "сякаш", "след като",
    "преди да", "докато", "но", "а", "ала", "обаче",
    "който", "която", "което", "които",
    "чийто", "чиято", "чието", "чиито",
    "какъвто", "каквато", "каквото", "каквито",
    "колкото", "отколкото"
]

# Чести русизми и паразитни изрази от Whisper моделите (използва се само в режим "Базов речник")
WHISPER_REPLACEMENTS = {
    r"\bчто\b": "какво",
    r"\bЧто\b": "Какво",
    r"\bето\b": "това",
    r"\bЕто\b": "Това",
    r"\bсъщо така и\b": "също така",
}


def fix_v_vv_s_ss(text: str) -> str:
    """
    Прилага строгия звуков закон за 'в/във' и 'с/със':
    - 'във' задължително само пред думи с 'в' или 'ф'. В останалите случаи е 'в'.
    - 'със' задължително само пред думи със 'с' или 'з'. В останалите случаи е 'с'.
    """
    # 1. Поправка на предлог В / ВЪВ
    def replace_v(match):
        prep = match.group(1)
        space = match.group(2)
        next_word = match.group(3)
        clean_word = next_word.lstrip("\"'«„(").lower()
        
        is_upper = prep[0].isupper()
        starts_vf = clean_word.startswith("в") or clean_word.startswith("ф")
        
        if starts_vf:
            new_prep = "Във" if is_upper else "във"
        else:
            new_prep = "В" if is_upper else "в"
        return f"{new_prep}{space}{next_word}"

    text = re.sub(r"\b([Вв][ъЪ][вВ]|[Вв])(\s+)([А-Яа-яA-Za-z0-9\"'«„(]+)", replace_v, text)

    # 2. Поправка на предлог С / СЪС
    def replace_s(match):
        prep = match.group(1)
        space = match.group(2)
        next_word = match.group(3)
        clean_word = next_word.lstrip("\"'«„(").lower()
        
        is_upper = prep[0].isupper()
        starts_sz = clean_word.startswith("с") or clean_word.startswith("з")
        
        if starts_sz:
            new_prep = "Със" if is_upper else "със"
        else:
            new_prep = "С" if is_upper else "с"
        return f"{new_prep}{space}{next_word}"

    text = re.sub(r"\b([Сс][ъЪ][сС]|[Сс])(\s+)([А-Яа-яA-Za-z0-9\"'«„(]+)", replace_s, text)
    return text


def fix_preposition_definite_article(text: str) -> str:
    """
    Поправя пълния член на кратък член след предлог с ограничена област на действие (предотвратява теч на предлога).
    Пример: 'в градът' -> 'в града', 'до големият мост' -> 'до големия мост'
    """
    tokens = re.split(r"(\s+|[.,!?:;\"'«»„“()—–-])", text)
    
    after_prep = False
    words_after_prep = 0
    
    for i in range(len(tokens)):
        token = tokens[i]
        lower_token = token.lower()
        
        if lower_token in PREPOSITIONS:
            after_prep = True
            words_after_prep = 0
            continue
            
        # Препинателен знак прекъсва действието на предлога
        if re.match(r"^[.,!?:;—–]$", token) or lower_token in PREP_TERMINATORS:
            after_prep = False
            words_after_prep = 0
            continue
            
        if after_prep and re.match(r"^[А-Яа-я]+$", token):
            words_after_prep += 1
            modified = False
            
            # Проверка дали думата завършва на пълен член (-ът, -ят, -ият, -тият)
            if lower_token not in NON_MEMBER_ROOTS and len(lower_token) > 3:
                # 1. Прилагателни/числителни с пълен член: -ият -> -ия
                if lower_token.endswith("ият"):
                    tokens[i] = token[:-1]
                    modified = True
                # 2. -тият -> -тия
                elif lower_token.endswith("тият"):
                    tokens[i] = token[:-1]
                    modified = True
                # 3. Съществителни: -ът -> -а
                elif lower_token.endswith("ът"):
                    tokens[i] = token[:-2] + ("А" if token[-1].isupper() else "а")
                    modified = True
                # 4. Съществителни: -ят -> -я
                elif lower_token.endswith("ят"):
                    tokens[i] = token[:-2] + ("Я" if token[-1].isupper() else "я")
                    modified = True

            # Ако думата вече е била членувана или е съществително след модификация,
            # предложната група приключва (един предлог управлява само едно членуване).
            if modified or lower_token.endswith(("а", "я", "та", "то", "те")) or words_after_prep >= 3:
                after_prep = False
                words_after_prep = 0
                    
    return "".join(tokens)


def fix_subject_definite_article_start(text: str) -> str:
    """
    Поправя кратък член на пълен член за подлог в началото на изречения:
    1. Известни мъжки съществителни (напр. 'Филма беше...' -> 'Филмът беше...')
    2. Прилагателни имена в началото на изречение (напр. 'Новия закон...' -> 'Новият закон...')
    """
    # 1. Корекция на съществителни подлози в началото на изречение
    def replace_masculine_noun(m):
        prefix = m.group(1)
        word = m.group(2)
        rest = m.group(3)
        lower_word = word.lower()
        if lower_word in COMMON_MASCULINE_SUBJECTS:
            correct_word = COMMON_MASCULINE_SUBJECTS[lower_word]
            if word[0].isupper():
                correct_word = correct_word.capitalize()
            return f"{prefix}{correct_word}{rest}"
        return m.group(0)

    text = re.sub(
        r"(^|[.!?]\s+)([А-Я][а-я]+)(\s+)",
        replace_masculine_noun,
        text
    )

    # 2. Корекция на членувани прилагателни на -ия в началото на изречение (Новия -> Новият)
    def replace_adjective_start(m):
        prefix = m.group(1)
        word = m.group(2)
        rest = m.group(3)
        lower_word = word.lower()
        
        # Проверяваме дали не е изключение от женски род (България, Русия, история и т.н.)
        if lower_word not in FEMININE_IA_EXCLUSIONS and len(lower_word) >= 4:
            if lower_word.endswith("ия"):
                return f"{prefix}{word}т{rest}"
        return m.group(0)

    text = re.sub(
        r"(^|[.!?]\s+)([А-Я][а-я]+ия)(\s+[а-яА-Я]+)",
        replace_adjective_start,
        text
    )

    return text


def fix_punctuation_commas(text: str) -> str:
    """
    Автоматично поставя запетаи пред подчинителни съюзи и относителни местоимения,
    ако пред тях липсва такава.
    """
    for conj in sorted(CONJUNCTIONS_REQUIRING_COMMA, key=len, reverse=True):
        pattern = rf"(?<=[^\s.,!?:;—–«\"(])(\s+)({re.escape(conj)}\b)"
        # Не слагаме запетая, ако предходната дума е съюз (напр. "и че", "а че", "или че")
        text = re.sub(rf"(?<!\bи)(?<!\bа)(?<!\bно)(?<!\bили){pattern}", r",\1\2", text, flags=re.IGNORECASE)
    
    return text


def clean_spaces_and_formatting(text: str) -> str:
    """
    Оправя интервалите около препинателните знаци, предпазва десетични числа (12.5, 3,14), съкращения (т.е., т.н.) и изчиства русизми.
    """
    # Запазваме стандартните български съкращения преди обработка на точките
    text = re.sub(r"\b[Тт]\.\s*[Ее]\.", lambda m: "__ABBR_TE_UPPER__" if m.group(0)[0].isupper() else "__abbr_te__", text)
    text = re.sub(r"\b[Тт]\.\s*[Нн]\.", "__abbr_tn__", text)

    # Премахване на паразитни интервали преди запетаи и точки: "дума , друга" -> "дума, друга"
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)
    
    # Добавяне на интервал след запетая или точка, БЕЗ да се чупят десетични числа като 3.14 или 12,5:
    text = re.sub(r"(?<!\d)([.,!?:;])(?!\d|\s)([А-Яа-яA-Za-z0-9])", r"\1 \2", text)
    
    # Премахване на двойни запетаи или двойни интервали
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Главна буква след точка, удивителна или въпросителна
    def cap_after_punct(m):
        p = m.group(1)
        s = m.group(2)
        letter = m.group(3)
        return f"{p}{s}{letter.upper()}"
        
    text = re.sub(r"([.!?])(\s+)([а-я])", cap_after_punct, text)

    # Възстановяване на съкращенията т.е. и т.н.
    text = text.replace("__ABBR_TE_UPPER__", "Т.е.").replace("__abbr_te__", "т.е.").replace("__abbr_tn__", "т.н.")
    
    # Замяна на Whisper русизми
    for pattern, repl in WHISPER_REPLACEMENTS.items():
        text = re.sub(pattern, repl, text)
        
    return text.strip()


def post_process_bulgarian_text(text: str) -> str:
    """
    Главен конвейер за 100% локална българска граматична и пунктуационна корекция.
    """
    if not text or not text.strip():
        return text

    # 1. Поправка на предлозите 'в/във' и 'с/със'
    text = fix_v_vv_s_ss(text)

    # 2. Поправка на членуването след предлози (пълен -> кратък член)
    text = fix_preposition_definite_article(text)

    # 3. Интелигентно членуване на подлог в началото на изречения (кратък -> пълен член)
    text = fix_subject_definite_article_start(text)

    # 4. Поставяне на липсващи запетаи пред подчинителни съюзи и местоимения
    text = fix_punctuation_commas(text)

    # 5. Оформяне на интервали, регистър и изчистване на русизми
    text = clean_spaces_and_formatting(text)

    return text


def post_process_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Прилага българския пост-процесинг върху всеки сегмент от Whisper поотделно,
    като запазва времевите интервали.
    """
    processed = []
    for seg in segments:
        new_seg = dict(seg)
        raw_text = seg.get("text", "")
        new_seg["text"] = post_process_bulgarian_text(raw_text)
        processed.append(new_seg)
    return processed
