# VeraCore.py
import sys
if "--pick-file" in sys.argv:
    from vera_core.app.file_picker import run_picker
    run_picker()
    sys.exit(0)

import multiprocessing
multiprocessing.freeze_support()

import sys
from vera_core.app import main

if "--app" not in sys.argv:
    sys.argv.append("--app")   # force desktop window mode
main()