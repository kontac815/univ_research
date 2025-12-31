import time
import pynvml
import wandb
import threading

def gpu_monitor(interval=1):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU0

    while True:
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        wandb.log({
            "gpu/memory_used_MB": mem.used / 1024**2,
            "gpu/utilization_%": util.gpu,
            "gpu/temp_C": temp,
        })
        time.sleep(interval)

def start_gpu_monitor():
    t = threading.Thread(target=gpu_monitor, daemon=True)
    t.start()
