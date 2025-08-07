import time
import psutil
import functools
from memory_profiler import memory_usage

def profile(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # process = psutil.Process()
        
        # tracemalloc.start()
        io_before = psutil.disk_io_counters()
        start_time = time.perf_counter()
        start_cpu = time.process_time()

        peak, result = memory_usage((func, args, kwargs), max_usage=True, retval=True, include_children=True)

        end_time = time.perf_counter()
        end_cpu = time.process_time()
        io_after = psutil.disk_io_counters()
        # current, peak = tracemalloc.get_traced_memory()
        # tracemalloc.stop()

        total_time = end_time - start_time
        cpu_time = end_cpu - start_cpu
        io_read = io_after.read_bytes - io_before.read_bytes
        io_read_time = io_after.read_time - io_before.read_time
        # io_write = io_after.write_bytes - io_before.write_bytes

        metrics = {
            "total_time_s": total_time,
            "cpu_time_s": cpu_time,
            "memory_peak_GB": peak / (1024),
            "io_read_MB": io_read / (1024*1024),
            "io_read_time_s": io_read_time / 1000
        }
        
        return result, metrics

    return wrapper
