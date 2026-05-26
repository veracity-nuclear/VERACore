# launcher.py
import multiprocessing
multiprocessing.freeze_support()

import sys
from vera_core.app import main

if "--app" not in sys.argv:
    sys.argv.append("--app")   # force desktop window mode
main()