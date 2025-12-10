
import sys
from pathlib import Path

print("Testing raw import torch...")
import torch
print("Raw import torch success")

print("Modifying sys.path...")
src_path = str(Path(__file__).parent / 'src')
sys.path.insert(0, src_path)
print(f"sys.path[0]: {sys.path[0]}")

print("Importing app_utils...")
try:
    import app_utils
    print("Import app_utils success")
except Exception as e:
    print(f"Import app_utils failed: {e}")
except SystemExit:
    print("SystemExit caught")
