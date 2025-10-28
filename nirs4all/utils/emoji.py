"""Emoji utility module for handling emoji display based on environment."""
import os

# Check if emojis should be disabled (set DISABLE_EMOJI=1 to disable)
DISABLE_EMOJI = os.environ.get('DISABLE_EMOJI', '0') == '1'


def emoji(char: str, fallback: str = "") -> str:
    """Return emoji character or fallback based on DISABLE_EMOJI setting.

    Args:
        char: The emoji character
        fallback: Fallback string when emojis are disabled (default: empty string)

    Returns:
        The emoji character or fallback
    """
    return fallback if DISABLE_EMOJI else char


# Common emojis as constants
ROCKET = emoji("🚀 ", "")
TROPHY = emoji("🏆 ", "")
MEDAL_GOLD = emoji("🥇 ", "")
MEDAL_SILVER = emoji("🥈 ", "")
MEDAL_BRONZE = emoji("🥉 ", "")
FLAG = emoji("🏁 ", "")
CHECK = emoji("✅ ", "")
CROSS = emoji("❌ ", "")
DISK = emoji("💾 ", "")
CHART = emoji("📊 ", "")
DIAMOND = emoji("🔷 ", "")
SMALL_DIAMOND = emoji("🔹", "")
TARGET = emoji("🎯 ", "")
SEARCH = emoji("🔍 ", "")
REFRESH = emoji("🔄 ", "")
WARNING = emoji("⚠️  ", "WARNING: ")
INFO = emoji("ℹ️  ", "INFO: ")
FOLDER = emoji("🗂️  ", "")
WEIGHT_LIFT = emoji("🏋️  ", "")
ALERT = emoji("🚨 ", "ALERT: ")
PIN = emoji("📍 ", "")
CLIPBOARD = emoji("📋 ", "")
WRENCH = emoji("🔧 ", "")
BULB = emoji("💡 ", "")
KEY = emoji("🔑 ", "")
PLAY = emoji("▶ ", ">")
BRAIN = emoji("🧠 ", "")
MICROSCOPE = emoji("🔬 ", "")
PALETTE = emoji("🎨 ", "")
CHECKMARK = emoji("✓ ", "*")

# Arrow indicators for metrics
ARROW_UP = emoji("↑", "^")
ARROW_DOWN = emoji("↓", "v")

