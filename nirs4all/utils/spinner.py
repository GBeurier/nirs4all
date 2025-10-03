"""
Simple spinner utility for showing loading progress during long operations.
"""
import threading
import time
import sys
from contextlib import contextmanager
from typing import Optional

spinner_chars = "|/-\\"
nice_spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
emoji_spinner_chars = "🌑🌒🌓🌔🌕🌖🌗🌘"

class Spinner:
    """A simple rotating spinner for showing loading progress."""

    def __init__(self, message: str = "Loading", chars: str = "|/-\\", speed: float = 0.1):
        """
        Initialize spinner.

        Args:
            message: Message to display before spinner
            chars: Characters to cycle through for animation
            speed: Time between character changes (seconds)
        """
        self.message = message
        self.chars = chars
        self.speed = speed
        self.idx = 0
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def _spin(self):
        """Internal method to handle spinning animation."""
        while self.running:
            char = self.chars[self.idx % len(self.chars)]
            sys.stdout.write(f'\r{self.message} {char}')
            sys.stdout.flush()
            self.idx += 1
            time.sleep(self.speed)

    def start(self):
        """Start the spinner."""
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._spin)
            self._thread.daemon = True
            self._thread.start()

    def stop(self):
        """Stop the spinner and clear the line."""
        if self.running:
            self.running = False
            if self._thread:
                self._thread.join()
            # Clear the spinner line and add newline for next output
            sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r\n')
            sys.stdout.flush()


@contextmanager
def spinner_context(message: str = "Processing", chars: str = nice_spinner_chars, speed: float = 0.1):
    """
    Context manager for easy spinner usage.

    Usage:
        with spinner_context("Training model"):
            # Long running operation
            time.sleep(5)

    Args:
        message: Message to display before spinner
        chars: Characters to cycle through for animation
        speed: Time between character changes (seconds)
    """
    s = Spinner(message, chars, speed)
    try:
        s.start()
        yield s
    finally:
        s.stop()