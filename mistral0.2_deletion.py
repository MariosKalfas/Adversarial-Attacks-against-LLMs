import shutil
import os
from pathlib import Path

# Διαδρομή Cache (Windows)
home = Path.home()
cache_dir = home / ".cache" / "huggingface" / "hub" / "models--mistralai--Mistral-7B-Instruct-v0.3"

print(f"🔍 Έλεγχος για φάκελο Mistral: {cache_dir}")

if cache_dir.exists():
    try:
        print("⏳ Διαγραφή Mistral σε εξέλιξη...")
        shutil.rmtree(cache_dir)
        print("✅ ΕΠΙΤΥΧΙΑ! Ο χώρος ελευθερώθηκε.")
    except Exception as e:
        print(f"❌ ΣΦΑΛΜΑ: {e}")
else:
    print("⚠️ Ο φάκελος δεν βρέθηκε. Προχωράμε.")