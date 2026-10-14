# PROGRESS

Running log. A fresh session should be able to continue from `PLAN.md` + this file.

## Current state

- **Phase 0 complete** (2026-10-14). Waiting for approval to start Phase 1.
- Next: Phase 1 (graph I/O, two-pass streaming converter, transforms, generators, gstats, download.sh).

## Measured machine facts (Phase 0)

From `scripts/env_check.sh` and `build/gmicro --hello` (Metal API):

| Item | Value |
|---|---|
| Chip | Apple M2 |
| CPU | 8 cores = 4 performance + 4 efficiency |
| Memory | 8 GB unified |
| Page size | 16384 bytes |
| GPU | Apple M2, 10 cores (system_profiler; core count is not exposed by the Metal API) |
| Metal families | Apple8: yes, Metal3: yes; unified memory: yes |
| maxThreadsPerThreadgroup | 1024 (pipeline maxTotalThreadsPerThreadgroup = 1024 for vector_add) |
| maxThreadgroupMemoryLength | 32768 bytes |
| threadExecutionWidth (simdgroup) | 32 (asserted for every pipeline in `MetalContext::pipeline`) |
| maxBufferLength | 4096 MiB |
| recommendedMaxWorkingSetSize | 5461 MiB |
| macOS / SDK | 26.2 / Command Line Tools SDK 13.3 |
| Compiler | Apple clang 14.0.3; `-march=native` accepted |
| Offline `metal` compiler | not available, as expected; shaders compiled at runtime |
| libomp | Homebrew 23.1.2 (already installed) |
| Free disk at start | 17 GiB |

vector_add on 10M floats: verified (0 mismatches); first-run GPU time 1.49 ms
(80.5 GB/s effective for 120 MB of traffic); end-to-end 18.7 ms (includes first
command-buffer setup). OpenMP: 8 threads by default.

## Log

### 2026-10-14 · Phase 0: environment and scaffold

**Done**
- `PLAN.md` (the plan verbatim), `.gitignore`, `Makefile`, `README.md` skeleton, this file, `git init`.
- `scripts/env_check.sh`: chip, P/E cores, memory, page size, GPU cores, Low Power Mode, power source, disk, toolchain, libomp, Python, venv.
- `scripts/setup_python.sh`: `.venv` with numpy 2.5.4, pandas 3.0.6, matplotlib 3.11.2, networkx 3.7, scipy 1.18.1 (306 MB).
- `src/metal/context.{hpp,mm}`: device and queue, device-info print, runtime compile of `common.h` + `<file>.metal` (fast math off, MSL 3.0), per-file library cache, pipeline cache keyed by function name, assertion that simdgroup width is 32, shared-storage buffer helpers (zero-size requests become 4-byte buffers so bindings stay valid), `dispatch1D`, `gpuTimeMs`.
- `src/shaders/common.h`, `src/shaders/micro.metal` (`vector_add`).
- `build/gmicro --hello` (Metal hello world plus device properties) and `--omp` (OpenMP hello world).
- `tests/test_main.cpp`: assert-based runner. Tests: OpenMP team size, and vector_add at n = 0, 1, 255, 256, 257, 100003.

**Results**
- `make` clean rebuild: 0 warnings. `make test`: 2/2 pass. vector_add verified.

**Decisions**
- **`-mmacosx-version-min=26.0`** in compile and link flags. Homebrew libomp is built for macOS 26, while the CLT SDK is 13.3. Without this flag, every link prints `ld: warning: dylib ... was built for newer macOS version (26.0) than being linked (13.3)`, which would violate the zero-warnings rule.
- **Python 3.13 for `.venv`, not the default `python3` (3.14).** Homebrew python@3.14 here has a broken `pyexpat`: `Symbol not found: _XML_SetAllocTrackerActivationThreshold` in `/usr/lib/libexpat.1.dylib`. That breaks `plistlib` → `platform.mac_ver()` → pip's truststore, so `venv`/`ensurepip` fails. `setup_python.sh` now picks the first interpreter that passes a pyexpat check, and `$PYTHON` overrides it. (Upstream fix: `brew reinstall python@3.14 expat`. Not done, because it changes the user's system.)
- Tests are compiled as Objective-C++ so they can drive Metal directly.
- Shader directory is baked in at build time (`GGPU_SHADER_DIR`, absolute path) and can be overridden by an environment variable of the same name.
- The project lives in its own git repo (`git init` here), nested inside the home-directory repo.

**Open issues / notes for benchmarking**
- At Phase 0, **Low Power Mode was ON and the machine was on battery**. It must be plugged in with LPM off before Phase 2 timings; `env_check.sh` reports both.
