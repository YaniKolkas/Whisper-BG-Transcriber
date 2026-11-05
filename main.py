"""
Главен стартиращ файл (Entrypoint) за Whisper BG Transcriber Desktop .EXE.
"""

import sys
import os
import io
import time
import socket
import threading
import webbrowser

# Настройване на UTF-8 конзола за Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import uvicorn
from app import app


def find_available_port(start_port: int = 8000, max_attempts: int = 20) -> int:
    """Намира първия свободен TCP порт, за да не крашва, ако 8000 вече се ползва."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port


def open_browser(port: int = 8000):
    """Изчаква 1.5 секунди и отваря браузъра на локалния адрес."""
    time.sleep(1.5)
    try:
        webbrowser.open(f"http://127.0.0.1:{port}")
    except Exception as e:
        print(f"Забележка при отваряне на браузър: {e}")


def main():
    port = find_available_port(8000)

    print("=" * 60)
    print("       Whisper BG Transcriber • OpenAI Whisper Desktop")
    print("=" * 60)
    print()
    print(f"Сървърът стартира на адрес: http://127.0.0.1:{port}")
    print("Браузърът ще се отвори автоматично след малко...")
    print("За да спрете програмата, просто затворете този прозорец.")
    print("-" * 60)

    # Стартиране на отварянето на браузъра във фонова нишка
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    # Стартиране на Uvicorn уеб сървъра
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",  # без излишен спам в конзолата
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n" + "!" * 60)
        print("ВЪЗНИКНА ГРЕШКА ПРИ СТАРТИРАНЕ:")
        print("!" * 60)
        traceback.print_exc()
        print("\n" + "-" * 60)
        try:
            input("Натиснете Enter, за да затворите прозореца...")
        except Exception:
            pass
