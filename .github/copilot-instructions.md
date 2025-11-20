# GitHub Copilot Instructions for nirs4all Workspace

# nirs4all is a python library for NIRS data analysis. It manage ML and DL pipelines for classification and regression tasks using scikit-learn, tensorflow, and pytorch.
# It saves pipelines in json and yaml, handles a db of predictions in json, handles multiple datasets and provides many matplotlib visualizations.

# Avoid overengineering, keep it simple and pragmatic, I want clean, neat, maintenable and readable code
# prefer using existing libraries and tools rather than reinventing the wheel
# I don't want deprecated functions or backward compatibility dead code, simply remove them.

# When writing code, follow best practices for Python, sklearn, tensorflow, React, and FastAPI development.
# Write modular, reusable, and testable code.
# Use Google Style Docstrings

# Use the .venv in nirs4all when launching scripts or commands.

# examples Q1 to QN are my personal integration tests with separated environments and datasets.
# They are not unit tests but a simple way to verify that nirs4all works as expected in real-world scenarios
# is to launch > .\run.ps1 -l   in the example folder. It generates a log.txt file with all outputs. If no traceback in the log, everything is fine and running.
# So after massive changes, and after having launched unit tests, you can launch run.ps1 -l in examples to verify everything is fine.
# run.ps1 can be used also with name to run one file only > .\run.ps1 -n example1.py -l
