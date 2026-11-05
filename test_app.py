import sys
import wave
import struct
import math
import numpy as np
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import transcriber
import exporter
import bg_corrector

def create_dummy_wav(filepath: str, duration_sec: float = 2.0):
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    with wave.open(filepath, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for i in range(num_samples):
            # 440 Hz тон
            val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            data = struct.pack("<h", val)
            wav_file.writeframesraw(data)

def test_pipeline():
    test_wav = "test_audio.wav"
    create_dummy_wav(test_wav, duration_sec=2.5)
    print(f"1. Създаден тестов аудио файл: {test_wav}")

    # Проверка на av продължителност
    dur = transcriber.get_audio_duration(test_wav)
    print(f"2. Изчислена продължителност: {dur:.2f} сек.")
    assert dur > 2.0, "Грешка при продължителност!"

    # Проверка на decode_audio (PyAV съвместимост)
    import faster_whisper.audio
    audio_arr = faster_whisper.audio.decode_audio(test_wav)
    print(f"2.1. Успешно декодирано аудио с PyAV: дължина {len(audio_arr)} семпъла")
    assert len(audio_arr) > 0, "Грешка при декодиране на аудио!"

    # Проверка на аудио нормализацията и филтрирането (Pre-processing)
    preprocessed_audio = transcriber.preprocess_audio(audio_arr, sampling_rate=16000)
    print(f"2.2. Успешна аудио нормализация и филтриране: пик = {np.max(np.abs(preprocessed_audio)):.2f}")
    assert len(preprocessed_audio) == len(audio_arr), "Грешка при дължина след нормализация!"

    # Проверка на езиковия коректор (Ниво 3)
    sample_raw = "Той отиде в Варна с влак и влезе в магазинът за да купи хляб. Филма беше супер защото имаше екшън."
    corrected = bg_corrector.post_process_bulgarian_text(sample_raw)
    print(f"2.3. Тест на езиков коректор (Ниво 3):\n  Оригинал: {sample_raw}\n  Коригиран: {corrected}")
    assert "във Варна" in corrected, "Грешка при 'в/във'!"
    assert "в магазина" in corrected, "Грешка при членуване след предлог!"
    assert "Филмът беше" in corrected, "Грешка при пълен член на подлог!"
    assert ", за да" in corrected, "Грешка при запетая пред 'за да'!"
    assert ", защото" in corrected, "Грешка при запетая пред 'защото'!"

    # Проверка на пълния транскрибиращ процес
    print("2.4. Тестване на пълната транскрипция с модел...")
    res = transcriber.transcribe_audio(test_wav, model_name="tiny", language="bg", vad_filter=False, apply_postprocessing=True)
    print(f"2.5. Транскрипцията завърши със статус: {res.get('status')}")

    # Проверка на експортьора
    mock_segments = [
        {"id": 0, "start": 0.0, "end": 2.5, "text": "Здравейте, това е успешен тест на български език."}
    ]
    txt_out = exporter.export_txt(mock_segments, include_timestamps=True)
    print("3. Експорт на TXT с времеви маркери:\n", txt_out)

    srt_out = exporter.export_srt(mock_segments)
    print("4. Експорт на SRT субтитри:\n", srt_out)

    docx_buf = exporter.export_docx(
        mock_segments,
        metadata={"filename": "test.wav", "duration_formatted": "00:02", "model_name": "small"},
        include_timestamps=True
    )
    assert docx_buf.getbuffer().nbytes > 1000, "DOCX генерирането е неуспешно!"
    print(f"5. Експорт на Word (.docx): Успешен! ({docx_buf.getbuffer().nbytes} байта)")

    # Почистване
    Path(test_wav).unlink(missing_ok=True)
    print("\nВсички модулни тестове преминаха успешно!")

if __name__ == "__main__":
    test_pipeline()
