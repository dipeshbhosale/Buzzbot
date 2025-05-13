import os
import sys
import subprocess

def install_pytesseract():
    """Install pytesseract package if not already installed."""
    try:
        import pytesseract
        return True
    except ImportError:
        print("Installing pytesseract...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytesseract"])
        return True
    except Exception as e:
        print(f"Error installing pytesseract: {e}")
        return False

def configure_tesseract():
    """Configure Tesseract path and verify installation."""
    try:
        import pytesseract
        tesseract_path = os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract OCR version {version} configured successfully")
            return True
        return False
    except Exception as e:
        print(f"Error configuring Tesseract: {e}")
        return False

def verify_tesseract():
    """Verify Tesseract installation and get installation help."""
    if install_pytesseract() and configure_tesseract():
        return True, None
    tesseract_path = os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR")
    return False, f"""Please complete OCR setup:
1. Install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
2. Add Tesseract to system PATH
3. Default install location: {tesseract_path}
4. Restart your application after installation"""
