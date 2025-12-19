# GitHub Copilot Instructions for nirs4all Workspace

# nirs4all is a python library for NIRS data analysis. It manage ML and DL pipelines for classification and regression tasks using scikit-learn, tensorflow, and pytorch.
# It saves pipelines in json and yaml, handles a db of predictions in json, handles multiple datasets and provides many matplotlib visualizations.

# You avoid overengineering, keep it simple and pragmatic, you write clean, neat, maintainable, readable, production-ready code
# You prefer using existing libraries and tools over reinventing the wheel.
# As nirs4all is not in version 1 yet, you remove deprecated functions, backward compatibility code or dead code. When reaching version 1, the code should be as clean as possible.

# When writing code, you follow best practices for Python, sklearn, tensorflow, React, and FastAPI development and adhere to PEP 8 guidelines.
# You write code that is compatible with Python 3.11 and above and write modular, reusable, and testable code.
# You use Google Style Docstrings

# Nirs4all project has a .venv with all required libraries and dependencies. You work within this environment.

# There are examples that serve as simple integration tests and simple documentation.
# You can launch them with run.ps1 and run.sh scripts.
# After each refactoring, features implementation or signature change, maintain examples, docs and tests up to date.
