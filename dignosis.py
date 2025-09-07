import sys
import platform
import subprocess

def check_python_version():
    ver = sys.version_info
    if not (3 <= ver.major and 9 <= ver.minor <= 12):
        return f"FAIL: Python {ver.major}.{ver.minor} unsupported. Need 3.9-3.12."
    return "PASS: Version OK."

def check_architecture():
    arch = platform.architecture()[0]
    if arch != '64bit':
        return f"FAIL: {arch} unsupported. Need 64-bit Python."
    return "PASS: 64-bit OK."

def check_pip_version():
    try:
        pip_ver = subprocess.check_output([sys.executable, '-m', 'pip', '--version']).decode().split()[1]
        major = int(pip_ver.split('.')[0])
        if major < 20:
            return "WARN: Old pip. Run: python -m pip install --upgrade pip"
        return "PASS: Pip OK."
    except:
        return "FAIL: Pip not found."

def try_import():
    try:
        import mediapipe
        return "PASS: MediaPipe installed."
    except ImportError:
        return "FAIL: Not installed. Run: pip install mediapipe"

print("MediaPipe Setup Check:")
print(check_python_version())
print(check_architecture())
print(check_pip_version())
print(try_import())
print("\nIf fails, fix as above then rerun script.")