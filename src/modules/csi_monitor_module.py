# csi_monitor_module.py

import subprocess

def start_serial_monitor(esp_idf_path, serial_port, baud_rate):
    command = [
        "python",
        f"{esp_idf_path}/tools/idf_monitor.py",
        "-p", serial_port,
        "-b", baud_rate
    ]
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors='ignore'
    )

def start_ping(ip, capture_duration):
    return subprocess.Popen(
        ["ping", "-w", str(capture_duration-1), ip],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
