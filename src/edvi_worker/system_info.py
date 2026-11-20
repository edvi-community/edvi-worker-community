from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass



@dataclass
class SystemInfo:
    os: str
    os_version: str
    architecture: str
    processor: str
    memory_gb: float
    logical_cpus: int
    hostname: str
    accelerator: str
    runtime_hint: str


def _run(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _accelerator(system: str, machine: str) -> tuple[str, str]:
    # Apple Silicon Macs can use Apple's MLX runtime.
    if system == "Darwin" and machine.lower() in {"arm64", "aarch64"}:
        return "Apple Silicon GPU", "MLX"

    # NVIDIA detection works on Windows/Linux when NVIDIA's nvidia-smi tool is installed.
    if system in {"Windows", "Linux"} and shutil.which("nvidia-smi"):
        name = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        if name:
            first_gpu = name.splitlines()[0].strip()
            return first_gpu, "CUDA"

    # Linux AMD detection: best-effort only. ROCm availability is a separate question.
    if system == "Linux" and shutil.which("lspci"):
        pci = _run(["lspci"])
        gpu_lines = [line for line in pci.splitlines() if "VGA" in line or "3D controller" in line]
        amd = next((line for line in gpu_lines if "AMD" in line or "Advanced Micro Devices" in line), "")
        if amd:
            return amd.split(":", 2)[-1].strip(), "ROCm/CPU (verify ROCm support)"

    return "Not detected", "CPU"



def _memory_gb(system: str) -> float:
    try:
        if system == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / (1024 ** 3), 1)

        if system == "Darwin":
            raw = _run(["sysctl", "-n", "hw.memsize"])
            if raw.isdigit():
                return round(int(raw) / (1024 ** 3), 1)

        if system == "Linux":
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 1)
    except (OSError, ValueError, AttributeError):
        pass
    return 0.0


def collect_system_info() -> SystemInfo:
    system = platform.system()
    machine = platform.machine()
    processor = platform.processor() or machine or "Unknown"
    accelerator, runtime = _accelerator(system, machine)

    friendly_os = {
        "Darwin": "macOS",
        "Windows": "Windows",
        "Linux": "Linux",
    }.get(system, system or "Unknown")

    return SystemInfo(
        os=friendly_os,
        os_version=platform.release(),
        architecture=machine or "Unknown",
        processor=processor,
        memory_gb=_memory_gb(system),
        logical_cpus=os.cpu_count() or 1,
        hostname=platform.node() or "Unknown",
        accelerator=accelerator,
        runtime_hint=runtime,
    )


def info_as_json(info: SystemInfo) -> str:
    return json.dumps(asdict(info), indent=2)
