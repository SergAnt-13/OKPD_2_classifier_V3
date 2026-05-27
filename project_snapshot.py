# project_snapshot.py
import os
from pathlib import Path

EXCLUDE_DIRS = {'.git', '__pycache__', '.venv', 'venv', '.idea', '.pytest_cache',
                'artifacts', 'runs', 'logs', 'node_modules'}
EXCLUDE_FILES = {'.DS_Store'}

def print_tree(root: Path, prefix: str = ""):
    entries = sorted(root.iterdir(), key=lambda e: (e.is_file(), e.name))
    for i, entry in enumerate(entries):
        if entry.name in EXCLUDE_DIRS or entry.name.endswith(".pyc"):
            continue
        if entry.is_dir():
            connector = "├── " if i < len(entries) - 1 else "└── "
            print(f"{prefix}{connector}{entry.name}/")
            print_tree(entry, prefix + ("│   " if i < len(entries) - 1 else "    "))
        else:
            connector = "├── " if i < len(entries) - 1 else "└── "
            print(f"{prefix}{connector}{entry.name}")

if __name__ == "__main__":
    root = Path(".")
    print(f"Структура проекта: {root.resolve().name}\n")
    print_tree(root)