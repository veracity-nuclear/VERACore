import asyncio, sys

def _picker_cmd(*args):
    if getattr(sys, "frozen", False):
        return [sys.executable, "--pick-file", *args]
    return [sys.executable, "-m", "vera_core.app.file_picker", *args]

async def launch_picker(*args, timeout=120):
    """Run the picker subprocess, return the chosen path or ''."""
    proc = await asyncio.create_subprocess_exec(
        *_picker_cmd(*args), stdout=asyncio.subprocess.PIPE
    )
    out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return out.decode().strip()