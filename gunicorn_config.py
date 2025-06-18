import multiprocessing

workers = 1  # or 2 for free tier safety
timeout = 120  # allow up to 2 minutes
worker_class = 'sync'
bind = '0.0.0.0:8000'


