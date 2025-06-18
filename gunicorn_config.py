# import multiprocessing
#
# # Server socket
# bind = "0.0.0.0:5000"
#
# # Worker processes
# workers = multiprocessing.cpu_count() * 2 + 1
# threads = 2
# worker_class = "gthread"
#
# # Timeout
# timeout = 120
# keepalive = 5
#
# # Logging
# accesslog = "-"
# errorlog = "-"
# loglevel = "info"
#
# # Security
# limit_request_line = 4094
# limit_request_fields = 100
# limit_request_field_size = 8190


import multiprocessing

workers = 1  # or 2 for free tier safety
timeout = 360  # allow up to 2 minutes
worker_class = 'sync'
bind = '0.0.0.0:8000'