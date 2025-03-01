import os

port = int(os.getenv("PORT", 8080))
bind = f"0.0.0.0:{port}"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 0