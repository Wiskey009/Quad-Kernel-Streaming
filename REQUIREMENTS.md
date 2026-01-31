# 📋 System Requirements

To successfully build and run the **Quad Kernel Streaming Pipeline**, ensure your system meets the following specifications.

## 🖥️ Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU** | x86_64 Dual Core | Modern Quad Core+ | AVX2 support required for Audio/Video DSP optimizations. |
| **RAM** | 4 GB | 16 GB+ | Compilation (esp. Rust and Ada link) can be memory intensive. |
| **Disk** | 2 GB Free Space | 10 GB SSD | For build artifacts and toolchains. |
| **GPU** | Integrated Graphics | NVIDIA GTX/RTX | **NVIDIA GPU** required for *hardware* video encoding (NVENC). The system runs in stub mode without it. |

---

## 🛠️ Software Prerequisites

### Operating System
- **Primary**: Windows 10 / 11 (64-bit)
- **Supported**: Linux (requires adaptation of CMake paths and system libs)

### Build Tools & Toolchains

You need the following tools installed and accessible in your system `PATH`.

#### 1. General Build Tools
- **CMake** (v3.20 or newer)
  - Download: [cmake.org](https://cmake.org/download/)
- **Git**
  - Download: [git-scm.com](https://git-scm.com/download/win)
- **MinGW-w64 (via MSYS2)**
  - Provides `gcc`, `g++`, `make`.
  - Environment: UCRT64 recommended.

#### 2. Ada / SPARK
- **Alire** (Ada Package Manager)
  - Download: [alire.ada.dev](https://alire.ada.dev/)
  - **Critical**: Must include the GNAT toolchain (automatically managed by Alire).

#### 3. Rust / WASM
- **Rustup & Cargo**
  - Download: [rustup.rs](https://rustup.rs/)
- **GNU Target for Windows**
  - Essential for linking Rust static libs with MinGW C/C++.
  - Install command: `rustup target add x86_64-pc-windows-gnu`

#### 4. Libraries (Optional for full mode)
- **NVIDIA Video Codec SDK** (Read-only headers included in stubs)
- **libopus** (Stubbed by default, install for real audio encoding)

---

## 🧩 Dependency Graph

```mermaid
graph TD
    A[Quad Kernel System] --> B[C Video Kernel]
    A --> C[C++ Audio Kernel]
    A --> D[Ada Math Kernel]
    A --> E[Rust WASM Bridge]
    
    B --> F[MinGW / GCC]
    C --> F
    C --> G[libopus (Optional)]
    D --> H[Alire / GNAT]
    E --> I[Cargo / Rustc]
```
