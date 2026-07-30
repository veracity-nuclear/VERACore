import multiprocessing
import sys


def main() -> None:
    multiprocessing.freeze_support()

    if "--pick-file" in sys.argv:
        from vera_core.app.file_picker import run_picker

        index = sys.argv.index("--pick-file")
        run_picker(sys.argv[index + 1 :])
        return

    from vera_core.app import main as run_app

    if "--app" not in sys.argv:
        sys.argv.append("--app")

    run_app()


if __name__ == "__main__":
    main()
