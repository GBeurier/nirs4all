"""Regenerate all annotated user-guide figures.

Run from the repository root: python examples/_internal/generate_guide_figures.py
See render_user_lessons.py for the deterministic drawing source and teaching data.
"""

from render_user_lessons import main

if __name__ == "__main__":
    main()
