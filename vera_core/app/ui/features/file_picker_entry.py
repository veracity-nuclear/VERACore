import asyncio, sys

def _picker_cmd(*args):
    if getattr(sys, "frozen", False):
        return [sys.executable, "--pick-file", *args]
    return [sys.executable, "-m", "vera_core.app.file_picker", *args]

async def launch_picker(*args):
    """Run the picker subprocess and return the chosen path (or '')."""
    proc = await asyncio.create_subprocess_exec(
        *_picker_cmd(*args), stdout=asyncio.subprocess.PIPE
    )
    try:
        out, _ = await proc.communicate()
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise
    return out.decode().strip()