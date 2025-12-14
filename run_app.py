import subprocess
import sys

def main():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "dialog_pro.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
    except Exception as e:
        print("Ошибка при запуске Streamlit:", e)
        input("Нажмите Enter для выхода...")

if __name__ == "__main__":
    main()

