"""Launch the initialization diagnostic through a Jupyter kernelspec.

``jupyter run`` has a hard-coded ten-second silence timeout.  This lightweight
wrapper starts emitting output before importing JAX, then executes the actual
diagnostic, whose configuration comes from ``PMG_INIT_SENSITIVITY_*`` variables.
"""

import runpy
import threading
import time


stop = threading.Event()


def emit_heartbeat() -> None:
    while not stop.wait(5.0):
        print(f"heartbeat_monotonic_seconds={time.monotonic():.1f}", flush=True)


thread = threading.Thread(target=emit_heartbeat, daemon=True)
thread.start()
print("stage=import_initialization_sensitivity", flush=True)
try:
    runpy.run_module(
        "poor_man_gplvm.diagnostics.initialization_sensitivity",
        run_name="__main__",
    )
finally:
    stop.set()
    thread.join(timeout=5.0)
