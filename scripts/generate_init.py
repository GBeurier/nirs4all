"""
Script to automatically generate __init__.py files for each subpackage under nirs4all
Scans for class definitions and writes import statements in __init__.py
"""
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), '..', 'nirs4all')

CLASS_REGEX = re.compile(r'^class\s+(\w+)')

for dirpath, dirnames, filenames in os.walk(ROOT):
    # skip hidden dirs and __pycache__
    if any(part.startswith('.') or part == '__pycache__' for part in dirpath.split(os.sep)):
        continue
    py_files = [f for f in filenames if f.endswith('.py') and f != '__init__.py']
    imports = []
    for fname in py_files:
        path = os.path.join(dirpath, fname)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                m = CLASS_REGEX.match(line)
                if m:
                    class_name = m.group(1)
                    module = os.path.splitext(fname)[0]
                    imports.append(f'from .{module} import {class_name}')
    # write __init__.py
    init_path = os.path.join(dirpath, '__init__.py')
    with open(init_path, 'w', encoding='utf-8') as f:
        if imports:
            f.write("# Automatically generated init file\n")
            for imp in sorted(imports):
                f.write(imp + '\n')
        else:
            f.write("# Package init\n")
"}EOF
