# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import bg_corrector

tests = [
    ("Той отиде в Варна с влак.", "Той отиде във Варна с влак."),
    ("В София сме с Стоян.", "В София сме със Стоян."),
    ("Във Пловдив вали с сняг.", "В Пловдив вали със сняг."),
    ("Във Франция бях със Стоян.", "Във Франция бях със Стоян."),
    ("Влязоха в магазинът.", "Влязоха в магазина."),
    ("Говорихме с докторът.", "Говорихме с доктора."),
    ("Отиде до големият мост.", "Отиде до големия мост."),
    ("Филма беше много хубав.", "Филмът беше много хубав."),
    ("Новия закон влезе в сила.", "Новият закон влезе в сила."),
    ("Той влезе в магазина и филмът започна.", "Той влезе в магазина и филмът започна."),
    ("Каза че ще закъснее.", "Каза, че ще закъснее."),
    ("Каза и че ще дойде.", "Каза и че ще дойде."),
    ("Не знам защото не пита.", "Не знам, защото не пита."),
    ("Човекът ,който видях , беше висок.", "Човекът, който видях, беше висок."),
    ("Лихвата е 12.5% или 3,14 лв.", "Лихвата е 12.5% или 3,14 лв."),
]

print("Running corrector tests...")
all_passed = True
for inp, expected in tests:
    res = bg_corrector.post_process_bulgarian_text(inp)
    if res != expected:
        print(f"FAIL:\n  Input:    {inp}\n  Got:      {res}\n  Expected: {expected}")
        all_passed = False
    else:
        print(f"PASS: {inp} -> {res}")

if all_passed:
    print("\nALL 15 TESTS PASSED SUCCESSFULLY!")
else:
    print("\nSOME TESTS FAILED!")
