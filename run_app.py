import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "streamlit", "run", "dialog_pro.py"],
    check=True,
)
