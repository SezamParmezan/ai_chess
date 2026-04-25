import sys
import os
import time
import subprocess
import pytest

from typing import Generator

#backend and file system from app/
ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "app"))
os.chdir(os.path.join(ROOT_DIR, "app"))

#fixture to deploy internal server and test site with it
@pytest.fixture(scope="session")
def server() -> Generator[subprocess.Popen, None, None]:
    proc = subprocess.Popen(
        ["uvicorn", "api.main:chess", "--host", "127.0.0.1", "--port", "8000"],
        cwd=os.path.join(ROOT_DIR, "app")
    )

    # Wait until the port is open
    import socket
    for _ in range(30):
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    else:
        proc.terminate()
        raise RuntimeError("Server did not start in time")

    yield proc
    proc.terminate()