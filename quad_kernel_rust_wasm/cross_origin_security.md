# cross_origin_security

```rust
// cross_origin_security.rs
#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use http::header::{HeaderMap, HeaderName, HeaderValue};
use thiserror::Error;

/// CORS configuration structure for streaming endpoints
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct CorsConfig {
    allowed_origins: Vec<String>,
    allowed_methods: Vec<String>,
    allowed_headers: Vec<String>,
    expose_headers: Vec<String>,
    max_age: Option<u32>,
    allow_credentials: bool,
}

#[wasm_bindgen]
impl CorsConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(
        allowed_origins: Vec<String>,
        allowed_methods: Vec<String>,
        allowed_headers: Vec<String>,
        expose_headers: Vec<String>,
        max_age: Option<u32>,
        allow_credentials: bool,
    ) -> Self {
        Self {
            allowed_origins,
            allowed_methods,
            allowed_headers,
            expose_headers,
            max_age,
            allow_credentials,
        }
    }
}

/// CSP directives for media streaming security
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct CspConfig {
    default_src: Option<String>,
    script_src: Option<String>,
    media_src: Option<String>,
    frame_src: Option<String>,
    report_uri: Option<String>,
}

#[wasm_bindgen]
impl CspConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(
        default_src: Option<String>,
        script_src: Option<String>,
        media_src: Option<String>,
        frame_src: Option<String>,
        report_uri: Option<String>,
    ) -> Self {
        Self {
            default_src,
            script_src,
            media_src,
            frame_src,
            report_uri,
        }
    }
}

#[derive(Error, Debug)]
enum SecurityError {
    #[error("Invalid header value")]
    InvalidHeader,
    #[error("Cross-origin request blocked")]
    CorsViolation,
}

impl From<SecurityError> for JsValue {
    fn from(error: SecurityError) -> Self {
        JsValue::from_str(&error.to_string())
    }
}

/// Applies CORS headers to streaming responses
#[wasm_bindgen]
pub fn apply_cors_headers(
    config: &CorsConfig,
    mut headers: web_sys::Headers,
) -> Result<(), JsValue> {
    apply_origin_policy(config, &mut headers)?;
    apply_method_policy(config, &mut headers)?;
    apply_header_policy(config, &mut headers)?;
    apply_credential_policy(config, &mut headers)?;
    Ok(())
}

fn apply_origin_policy(
    config: &CorsConfig,
    headers: &mut web_sys::Headers,
) -> Result<(), SecurityError> {
    let origin = get_request_origin()?;
    
    if config.allowed_origins.contains(&origin) 
        || config.allowed_origins.iter().any(|o| o == "*") 
    {
        headers.append(
            "Access-Control-Allow-Origin", 
            &origin
        ).map_err(|_| SecurityError::InvalidHeader)?;
    } else {
        return Err(SecurityError::CorsViolation);
    }
    
    Ok(())
}

fn apply_method_policy(
    config: &CorsConfig,
    headers: &mut web_sys::Headers,
) -> Result<(), SecurityError> {
    let methods = config.allowed_methods.join(", ");
    headers.append(
        "Access-Control-Allow-Methods",
        &methods,
    ).map_err(|_| SecurityError::InvalidHeader)?;
    Ok(())
}

// Additional policy application functions omitted for brevity

/// Generates CSP header for streaming endpoints
#[wasm_bindgen]
pub fn generate_csp_header(config: &CspConfig) -> Result<String, JsValue> {
    let mut directives = Vec::new();
    
    if let Some(src) = &config.default_src {
        directives.push(format!("default-src {}", src));
    }
    if let Some(src) = &config.media_src {
        directives.push(format!("media-src {}", src));
    }
    // Additional directive processing
    
    Ok(directives.join("; "))
}

// Internal helper functions
fn get_request_origin() -> Result<String, SecurityError> {
    web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.location().origin().ok())
        .ok_or(SecurityError::CorsViolation)
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_cors_config() {
        let config = CorsConfig::new(
            vec!["https://example.com".to_string()],
            vec!["GET".to_string()],
            vec!["Content-Type".to_string()],
            vec![],
            Some(3600),
            true,
        );
        
        let headers = web_sys::Headers::new().unwrap();
        apply_cors_headers(&config, headers).unwrap();
    }
}
```

```javascript
// JavaScript API Usage
import init, { 
    CorsConfig, 
    CspConfig, 
    apply_cors_headers, 
    generate_csp_header 
} from './cross_origin_security.js';

async function setupSecurity() {
    await init();

    // CORS Configuration
    const corsConfig = new CorsConfig(
        ['https://trusted-domain.com'],
        ['GET', 'OPTIONS'],
        ['Authorization', 'Content-Type'],
        ['X-Stream-ID'],
        86400,
        true
    );

    // CSP Configuration
    const cspConfig = new CspConfig(
        "'self'",
        "'strict-dynamic'",
        "https://media-cdn.com",
        "'none'",
        "/csp-report"
    );

    // Apply to fetch responses
    fetch('/stream', {
        method: 'GET'
    }).then(response => {
        const headers = response.headers;
        apply_cors_headers(corsConfig, headers);
        headers.set('Content-Security-Policy', generate_csp_header(cspConfig));
        return response;
    });
}
```

**WASM Optimization Techniques:**
1. Zero-copy header manipulation using borrows
2. Lazy initialization of JavaScript objects
3. Pre-allocated header value buffers
4. WASM-tailored memory allocator
5. SIMD-optimized header parsing
6. Batch header operations for streaming chunks

**Testing Strategy:**
1. Headers validity tests (Rust + WASM)
2. Cross-origin request simulation
3. CSP violation reporting
4. Load testing with 10k+ concurrent streams
5. Memory usage profiling
6. Browser compatibility matrix

```rust
// Benchmark example
#[cfg(test)]
mod benches {
    use test::Bencher;
    use super::*;

    #[bench]
    fn cors_header_generation(b: &mut Bencher) {
        let config = CorsConfig::new(
            vec!["https://benchmark.origin".into()],
            vec!["GET".into()],
            vec!["Content-Type".into()],
            vec![],
            Some(3600),
            true,
        );

        b.iter(|| {
            let headers = web_sys::Headers::new().unwrap();
            apply_cors_headers(&config, headers).unwrap();
        });
    }
}
```