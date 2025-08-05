use axum::{
    extract::{Json as ExtractJson, State},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use axum::http::StatusCode;
use once_cell::sync::Lazy;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, time::Duration};
use tower_http::cors::CorsLayer;

static CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(45))
        .build()
        .expect("client")
});

#[derive(Clone)]
struct App {
    base: String,
    model: String,
}

#[derive(Deserialize)]
struct PromptRequest {
    prompt: String,
}

#[derive(Serialize)]
struct PromptBody {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct OllamaResponse {
    response: String,
}

async fn generate_handler(
    State(app): State<App>,
    ExtractJson(req): ExtractJson<PromptRequest>,
) -> Result<Json<OllamaResponse>, Response> {
    let body = PromptBody {
        model: app.model.clone(),
        prompt: req.prompt,
        stream: false,
    };

    let res = CLIENT
        .post(format!("{}/api/generate", app.base))
        .json(&body)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Failed contacting Ollama API: {e}")).into_response())?;

    if !res.status().is_success() {
        let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let text = res.text().await.unwrap_or_default();
        return Err((status, format!("Ollama error: {}", text)).into_response());
    }

    let api: OllamaResponse = res
        .json()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Failed to parse Ollama response: {e}")).into_response())?;

    Ok(Json(api))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app_state = App {
        base: std::env::var("OLLAMA_BASE").unwrap_or_else(|_| "http://127.0.0.1:11434".into()),
        model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "mistral".into()),
    };

    let app = Router::new()
        .route("/generate", post(generate_handler))
        .with_state(app_state)
        .layer(CorsLayer::permissive());

    let addr = SocketAddr::from(([127, 0, 0, 1], 3001));
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind failed");
    axum::serve(listener, app).await.expect("server failed");
}


