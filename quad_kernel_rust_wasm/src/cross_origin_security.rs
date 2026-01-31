use thiserror::Error;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct CorsConfig {
    pub(crate) allowed_origins: Vec<String>,
    pub(crate) allowed_methods: Vec<String>,
    pub(crate) allowed_headers: Vec<String>,
    pub(crate) expose_headers: Vec<String>,
    pub(crate) max_age: Option<u32>,
    pub(crate) allow_credentials: bool,
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

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct CspConfig {
    pub(crate) default_src: Option<String>,
    pub(crate) media_src: Option<String>,
}

#[wasm_bindgen]
impl CspConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(default_src: Option<String>, media_src: Option<String>) -> Self {
        Self {
            default_src,
            media_src,
        }
    }
}

#[derive(Error, Debug)]
pub enum SecurityError {
    #[error("Invalid header value")]
    InvalidHeader,
    #[error("Cross-origin request blocked: {0}")]
    CORSBlocked(String),
    #[error("Security policy violation: {0}")]
    PolicyViolation(String),
    #[error("Cors violation")]
    CorsViolation,
}

impl From<SecurityError> for JsValue {
    fn from(error: SecurityError) -> Self {
        JsValue::from_str(&error.to_string())
    }
}

#[wasm_bindgen]
pub fn apply_cors_headers(config: &CorsConfig, headers: &web_sys::Headers) -> Result<(), JsValue> {
    let window =
        web_sys::window().ok_or_else(|| SecurityError::PolicyViolation("No window".to_string()))?;
    let origin = window
        .location()
        .origin()
        .map_err(|e| SecurityError::PolicyViolation(format!("{:?}", e)))?;

    if config.allowed_origins.contains(&origin) || config.allowed_origins.iter().any(|o| o == "*") {
        headers
            .append("Access-Control-Allow-Origin", &origin)
            .map_err(|_| SecurityError::InvalidHeader)?;
    } else {
        return Err(SecurityError::CORSBlocked(origin).into());
    }

    let methods = config.allowed_methods.join(", ");
    headers
        .append("Access-Control-Allow-Methods", &methods)
        .map_err(|_| SecurityError::InvalidHeader)?;

    Ok(())
}

#[wasm_bindgen]
pub fn generate_csp_header(config: &CspConfig) -> Result<String, JsValue> {
    let mut directives = Vec::new();
    if let Some(src) = &config.default_src {
        directives.push(format!("default-src {}", src));
    }
    if let Some(src) = &config.media_src {
        directives.push(format!("media-src {}", src));
    }
    Ok(directives.join("; "))
}
