import ctypes
import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import mock_open, patch

from edvi_worker import system_info as info


class HardwareTests(unittest.TestCase):
    def test_apple_silicon(self):
        for architecture in ("arm64", "aarch64"):
            with self.subTest(architecture=architecture):
                self.assertEqual(info._accelerator("Darwin", architecture),
                                 ("Apple Silicon GPU", "MLX"))

    @patch.object(info.shutil, "which", return_value=None)
    def test_missing_gpu_tools(self, which):
        for system in ("Darwin", "Windows", "Linux"):
            with self.subTest(system=system):
                self.assertEqual(info._accelerator(system, "x86_64"),
                                 ("Not detected", "CPU"))

    @patch.object(info.shutil, "which", return_value="nvidia-smi")
    @patch.object(info, "_run", return_value="NVIDIA RTX 4090\nNVIDIA RTX 3090")
    def test_nvidia_on_windows_and_linux(self, run, which):
        for system in ("Windows", "Linux"):
            self.assertEqual(info._accelerator(system, "x86_64"),
                             ("NVIDIA RTX 4090", "CUDA"))

    @patch.object(info.shutil, "which", side_effect=lambda name: name if name == "lspci" else None)
    @patch.object(info, "_run", return_value="03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Radeon RX 6800")
    def test_linux_amd(self, run, which):
        accelerator, runtime = info._accelerator("Linux", "x86_64")
        self.assertIn("Radeon RX 6800", accelerator)
        self.assertEqual(runtime, "ROCm/CPU (verify ROCm support)")

    @patch.object(info.shutil, "which", return_value="tool")
    @patch.object(info, "_run", return_value="")
    def test_failed_gpu_queries_fall_back_to_cpu(self, run, which):
        self.assertEqual(info._accelerator("Linux", "x86_64"), ("Not detected", "CPU"))

    def test_memory_sources_and_failures(self):
        with patch.object(info, "_run", return_value=str(16 * 1024 ** 3)):
            self.assertEqual(info._memory_gb("Darwin"), 16.0)
        with patch("builtins.open", mock_open(read_data="MemTotal:       8388608 kB\n")):
            self.assertEqual(info._memory_gb("Linux"), 8.0)
        with patch("builtins.open", side_effect=OSError):
            self.assertEqual(info._memory_gb("Linux"), 0.0)
        with patch.object(info, "_run", return_value="invalid"):
            self.assertEqual(info._memory_gb("Darwin"), 0.0)

    def test_windows_memory(self):
        def populate(pointer):
            pointer._obj.ullTotalPhys = 32 * 1024 ** 3
            return 1

        windows_api = SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=populate))
        with patch.object(ctypes, "windll", windows_api, create=True):
            self.assertEqual(info._memory_gb("Windows"), 32.0)

    def test_command_failures(self):
        for failure in (OSError(), subprocess.TimeoutExpired("tool", 3)):
            with patch.object(info.subprocess, "run", side_effect=failure):
                self.assertEqual(info._run(["tool"]), "")

    def test_collection_and_json(self):
        for system, friendly in (("Darwin", "macOS"), ("Windows", "Windows"), ("Linux", "Linux")):
            with self.subTest(system=system), \
                 patch.object(info.platform, "system", return_value=system), \
                 patch.object(info.platform, "machine", return_value="x86_64"), \
                 patch.object(info.platform, "processor", return_value=""), \
                 patch.object(info.platform, "release", return_value="test-release"), \
                 patch.object(info.platform, "node", return_value="test-host"), \
                 patch.object(info.os, "cpu_count", return_value=None), \
                 patch.object(info, "_memory_gb", return_value=16.0), \
                 patch.object(info, "_accelerator", return_value=("Not detected", "CPU")):
                result = json.loads(info.info_as_json(info.collect_system_info()))
                self.assertEqual(result, dict(os=friendly, os_version="test-release",
                    architecture="x86_64", processor="x86_64", memory_gb=16.0,
                    logical_cpus=1, hostname="test-host", accelerator="Not detected",
                    runtime_hint="CPU"))
