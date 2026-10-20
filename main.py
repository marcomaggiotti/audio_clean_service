"""Entrypoint: `python main.py` or `uvicorn main:app`."""

import logging

from app.config import get_settings
from app.main import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    # One process only: the job queue is in-process and the job list is a local file.
    uvicorn.run(app, host=settings.host, port=settings.port, proxy_headers=True, forwarded_allow_ips="*")
