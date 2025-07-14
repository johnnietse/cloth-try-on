# import multiprocessing

# # workers = 1  # or 2 for free tier safety
# # timeout = 120  # allow up to 2 minutes
# # worker_class = 'sync'
# # bind = '0.0.0.0:8000'


# workers = 4
# worker_class = "sync"
# worker_tmp_dir = "/dev/shm"
# bind = "0.0.0.0:10000"
# timeout = 120
# keepalive = 5


workers = 4
worker_class = "sync"
worker_tmp_dir = "/dev/shm"
bind = "0.0.0.0:10000"
timeout = 120
keepalive = 5
errorlog = "-"  # Log to stdout
accesslog = "-"  # Log to stdout
