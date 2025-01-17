#!/usr/bin/env python3
# audiblez - A program to convert e-books into audiobooks using
# Kokoro-82M model for high-quality text-to-speech synthesis.
# by Claudio Santini 2025 - https://claudio.uk

import sys
from gui import start_gui
from engine import cli_main

if __name__ == '__main__':
    if len(sys.argv) > 1:
        cli_main(sys.argv)
    else:
        start_gui()