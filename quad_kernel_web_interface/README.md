# 🌐 Quad Kernel Web Interface

> *"The dashboard where the magic happens."*

This is the **High-Performance Frontend** for the Quad Kernel Streaming System. It connects to the Rust/C++ backend via **WebAssembly (WASM)**, allowing direct memory access and zero-copy rendering potential directly in the browser.

## 🏗️ Architecture

The interface is built with a latency-first approach, bypassing traditional REST APIs for direct kernel interoperability.

```mermaid
graph LR
    User[Streamer] --> React[React 18 UI]
    React --> Vite[Vite Dev Server]
    React -- Bindings --> WASM[Quad Kernel (Rust)]
    WASM -- FFI --> Cpp[Audio Engine]
    WASM -- FFI --> C[Video Engine]
    WASM --> WebGL[Browser Canvas]
```

### Tech Stack
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite (configured for `SharedArrayBuffer` support)
- **State Management**: Zustand
- **Kernel Bridge**: `wasm-bindgen` (Rust -> JS)

## 🚀 Getting Started

Ensure you have the WASM packet compiled (done automatically by the build pipeline).

```bash
# 1. Install dependencies
npm install

# 2. Start the 4K Dashboard
npm run dev
```

Open `http://localhost:5173` to verify Kernel connectivity.

## 🧩 Current Status
- ✅ **Infrastructure**: Vite + React scaffolding complete.
- ✅ **WASM Bridge**: `quad_kernel_rust_wasm` bindings imported successfully.
- ✅ **Connectivity**: Handshake with Rust kernel verified.
- 🚧 **Rendering**: WebGL video surface initialized (in progress).

## ⚠️ Requirements
- **Browser**: Chrome/Edge 91+ or Firefox 90+ (Required for `SharedArrayBuffer`).
- **GPU**: Hardware acceleration enabled for WebGL rendering.

---
*Powered by Quad Kernel Polyglot Architecture.*