"""
Energy of a block of code: GPU power from NVML (sampled in a thread) and CPU
package energy from Intel RAPL when the OS lets us read it.

    with EnergyMeter() as em:
        for frame in frames:
            run(frame)
    print(em.report(n_frames=len(frames)))

GPU energy is the integral of board power over the block MINUS the idle
power measured just before it (so it is the energy the workload added, not
the card's floor).  The GPU must not be shared while measuring (no CARLA).
RAPL (/sys/class/powercap/intel-rapl:0/energy_uj) is root-only on most
recent kernels; without it the CPU side is reported as CPU-seconds only.
"""

import os
import threading
import time

RAPL = '/sys/class/powercap/intel-rapl:0/energy_uj'
RAPL_MAX = '/sys/class/powercap/intel-rapl:0/max_energy_range_uj'


def _rapl_read():
    try:
        with open(RAPL) as f:
            return int(f.read())
    except (OSError, ValueError):
        return None


class EnergyMeter:
    def __init__(self, gpu_index=0, period_s=0.02, idle_s=2.0):
        self.period, self.idle_s = period_s, idle_s
        self.nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(self.gpu_name, bytes):
                self.gpu_name = self.gpu_name.decode()
        except Exception:                       # no NVIDIA GPU / no pynvml
            self.gpu_name = None
        self.samples = []

    def _power_w(self):
        return self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def _mem_mb(self):
        return self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used / 2 ** 20

    def _loop(self):
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), self._power_w(), self._mem_mb()))
            time.sleep(self.period)

    def __enter__(self):
        self.idle_w = self.idle_mem = None
        if self.nvml is not None:
            idle = []
            t_end = time.perf_counter() + self.idle_s
            while time.perf_counter() < t_end:
                idle.append((self._power_w(), self._mem_mb()))
                time.sleep(self.period)
            self.idle_w = sum(p for p, _ in idle) / len(idle)
            self.idle_mem = sum(m for _, m in idle) / len(idle)
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        self.rapl0 = _rapl_read()
        self.cpu0 = time.process_time()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.t0
        self.cpu_s = time.process_time() - self.cpu0
        r1 = _rapl_read()
        self.cpu_j = None
        if self.rapl0 is not None and r1 is not None:
            d = r1 - self.rapl0
            if d < 0:                             # counter wrapped
                try:
                    d += int(open(RAPL_MAX).read())
                except (OSError, ValueError):
                    d = None
            self.cpu_j = None if d is None else d / 1e6
        self.gpu_j = self.gpu_peak_mem = None
        if self.nvml is not None:
            self._stop.set()
            self._thread.join()
            s = self.samples
            if len(s) >= 2:
                e = sum((t1 - t0) * 0.5 * (p0 + p1) for (t0, p0, _), (t1, p1, _) in zip(s, s[1:]))
                self.gpu_j = max(e - self.idle_w * (s[-1][0] - s[0][0]), 0.0)
                self.gpu_peak_mem = max(m for _, _, m in s) - self.idle_mem
        return False

    def report(self, n_frames):
        n = max(n_frames, 1)
        out = dict(frames=n_frames, wall_s=round(self.wall, 3),
                   ms_per_frame=round(1000 * self.wall / n, 2),
                   cpu_s_per_frame=round(self.cpu_s / n, 4), gpu=self.gpu_name)
        if self.gpu_j is not None:
            out.update(gpu_idle_w=round(self.idle_w, 1),
                       gpu_j_per_frame=round(self.gpu_j / n, 4),
                       gpu_extra_mem_mb=round(self.gpu_peak_mem, 0))
        if self.cpu_j is not None:
            out.update(cpu_pkg_j_per_frame=round(self.cpu_j / n, 4))
        else:
            out.update(cpu_pkg_j_per_frame=None,
                       note='RAPL not readable (run as root for CPU joules)'
                       if os.path.exists(RAPL) else 'no RAPL on this machine')
        return out
