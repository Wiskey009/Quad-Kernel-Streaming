import { useState, useEffect } from 'react'
import init, { WasmBuffer } from './wasm/pkg/quad_kernel_rust_wasm'

function App() {
    const [status, setStatus] = useState('Idle')
    const [kernelReady, setKernelReady] = useState(false)

    const initializeKernel = async () => {
        try {
            setStatus('Loading WASM module...')
            // Initialize the WASM module
            await init()

            setStatus('WASM Loaded! Initializing test...')

            // Test creating a Rust object from JS (Proof of Life)
            const data = new Uint8Array([1, 2, 3, 4])
            const buffer = new WasmBuffer(data)

            setStatus(`✅ Kernel Active! Created WasmBuffer with ${buffer.data.length} bytes.`)
            setKernelReady(true)
            console.log("Rust Kernel connected successfully!")

        } catch (err) {
            console.error(err)
            setStatus(`❌ Error: ${err}`)
        }
    }

    return (
        <div style={{
            backgroundColor: '#0a0a0a',
            color: '#ffffff',
            minHeight: '100vh',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            fontFamily: 'Inter, system-ui, sans-serif'
        }}>
            <h1>🔷 Quad Kernel Streaming Interface</h1>
            <div style={{
                padding: '2rem',
                border: '1px solid #333',
                borderRadius: '8px',
                backgroundColor: '#111',
                textAlign: 'center',
                minWidth: '400px'
            }}>
                <p style={{ marginBottom: '1rem' }}>System Status: <br />
                    <span style={{
                        color: kernelReady ? '#00ff00' : '#ffff00',
                        fontWeight: 'bold',
                        fontSize: '1.2rem'
                    }}>
                        {status}
                    </span>
                </p>

                {!kernelReady && (
                    <button
                        onClick={initializeKernel}
                        style={{
                            padding: '10px 20px',
                            backgroundColor: '#0078D6',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '1rem'
                        }}
                    >
                        🚀 Connect to Kernels
                    </button>
                )}

                {kernelReady && (
                    <div style={{ marginTop: '20px', padding: '10px', background: '#222', borderRadius: '4px' }}>
                        Ready for 4K Pipeline configuration.
                    </div>
                )}
            </div>
        </div>
    )
}

export default App
