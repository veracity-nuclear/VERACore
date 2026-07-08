# VeraCore.py
import sys
if "--pick-file" in sys.argv:
    from vera_core.app.file_picker import run_picker
    i = sys.argv.index("--pick-file")
    run_picker(sys.argv[i + 1:])
    sys.exit(0)

import multiprocessing
multiprocessing.freeze_support()

import sys
from vera_core.app import main

if "--app" not in sys.argv:
    sys.argv.append("--app")
main()