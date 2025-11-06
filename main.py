#!/usr/bin/env python3
"""
Main entry point for the embedding similarity experiment web application.
"""

from app import create_app


if __name__ == '__main__':
    app = create_app()

    print("\n" + "="*60)
    print("Эксперимент с эмбеддингами - Веб-сервер")
    print("="*60)
    print("\nСервер запускается...")
    print("После запуска откройте браузер по адресу:")
    print("\n  http://127.0.0.1:5000\n")
    print("="*60 + "\n")

    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        use_reloader=False
    )
