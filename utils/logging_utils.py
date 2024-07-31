import time
from contextlib import contextmanager
import inspect

@contextmanager
def log_info():
    # Retrieve the name of the calling function/method
    frame = inspect.currentframe().f_back
    method_name = frame.f_code.co_name if frame else "Unknown"
    
    start_time = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"Running method {method_name}")
    
    try:
        yield
    finally:
        end_time = time.time()
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Elapsed time: {end_time - start_time:.2f} seconds")