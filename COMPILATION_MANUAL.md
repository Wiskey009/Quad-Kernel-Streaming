# 🛠️ Quad Kernel Compilation Manual

This manual provides a step-by-step guide to compiling the **Quad Kernel Streaming Pipeline**. Since this is a polyglot system (C, C++, Ada, Rust), the build order is critical.

---

## ⚠️ Important Note on Toolchains

This project links code from **3 different compilers**:
1. `gcc` (System/Video - C)
2. `g++` (Audio - C++)
3. `gnat` (Math - Ada)
4. `rustc` (Bridge - Rust)

**Crucial**: To avoid ABI (Application Binary Interface) mismatches, all components must target the **GNU ABI** on Windows (`x86_64-w64-mingw32` / `x86_64-pc-windows-gnu`). **Do not mix MSVC and GNU targets.**

---

## 🚀 Step 1: Initialize Environment

Ensure you have cloned the repository and entered the directory:

```bash
git clone https://github.com/YOUR_USER/quad-kernel-streaming.git
cd quad-kernel-streaming
```

---

## 🧮 Step 2: Build Ada Math Kernel

The Ada kernel provides mathematical precision and validation. We use **Alire** to manage it.

1. Navigate to the Ada directory:
   ```bash
   cd quad_kernel_ada_math
   ```

2. Build the static library:
   ```bash
   alr build
   ```

3. **Verify Output**:
   Check that `libquad_math.a` exists in `quad_kernel_ada_math/lib/`.
   
   > *Note: If Alire asks to install a toolchain (gnat_native), verify YES.*

4. Return to root:
   ```bash
   cd ..
   ```

---

## 🦀 Step 3: Build Rust WASM Bridge

The Rust kernel handles WebSocket communication. We must build it as a static library using the **GNU target**.

1. Navigate to the Rust directory:
   ```bash
   cd quad_kernel_rust_wasm
   ```

2. Add the GNU target (if not already added):
   ```bash
   rustup target add x86_64-pc-windows-gnu
   ```

3. Build the static library:
   ```bash
   cargo build --release --target x86_64-pc-windows-gnu
   ```

4. **Verify Output**:
   Check for `libquad_kernel_rust_wasm.a` in `quad_kernel_rust_wasm/target/x86_64-pc-windows-gnu/release/`.

5. Return to root:
   ```bash
   cd ..
   ```

---

## 🎥 Step 4: Build Core System (C/C++) & Link

This step compiles the Video (C) and Audio (C++) kernels and links everything together using CMake.

1. Create a build directory:
   ```bash
   mkdir build_final
   cd build_final
   ```

2. **Configure CMake**:
   We explicitly tell CMake to use the **GCC/G++** compilers (often synonymous with the ones Alire uses, or your system MinGW).
   
   ```bash
   cmake .. -G "MinGW Makefiles"
   ```
   
   *Tip: If you encounter errors about missing Ada libs, ensure the paths in `CMakeLists.txt` point to your Alire toolchain location.*

3. **Compile and Link**:
   ```bash
   cmake --build .
   ```

---

## ✅ Step 5: Run the System

If compilation is successful, you will see `quad_kernel_system.exe` in the build directory.

Run it:
```bash
./quad_kernel_system.exe
```

### Expected Output
```text
=== QUAD KERNEL STREAMING SYSTEM v1.0 ===
[BRIDGE] Initializing Quad Kernel IPC Bridge...
[NVENC] Initializing encoder on device 0 for 3840x2160
[VIDEO KERNEL] Initialized successfully (4K H.265).
[AUDIO KERNEL] Initialized successfully (stub mode).
[MATH KERNEL] Initialized Precision Validation Engine.
[SYSTEM] Pipeline ready.
...
```

---

## 🔧 Troubleshooting

### **Error: `undefined reference to 'opus_...'`**
- **Cause**: The system tried to link against real libopus but it's not installed.
- **Fix**: Ensure the C++ audio adapter is using the STUB implementation (default in this repo) or install libopus and update CMakeLists.txt.

### **Error: `cannot find -lgnat`**
- **Cause**: The linker cannot find the Ada runtime libraries.
- **Fix**: Open `CMakeLists.txt` and verify the `link_directories` path points to your actual Alire GNAT installation (e.g., inside `%LOCALAPPDATA%\alire\cache\toolchains\...`).

### **Error: `undefined reference to '...rust_panic...'`**
- **Cause**: You likely built Rust with the MSVC target (default on Windows) instead of GNU.
- **Fix**: Re-run Step 3 ensuring `--target x86_64-pc-windows-gnu` is used.
