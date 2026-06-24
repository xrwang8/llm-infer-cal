use std::{net::SocketAddr, path::PathBuf};

#[tokio::main]
async fn main() {
    let addr = std::env::var("LLM_INFER_CAL_WEB_ADDR")
        .ok()
        .and_then(|value| value.parse::<SocketAddr>().ok())
        .unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 8080)));

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind web API listener");
    let app = std::env::var_os("LLM_INFER_CAL_STATIC_DIR")
        .map(PathBuf::from)
        .map(llm_infer_cal_web::app_with_static)
        .unwrap_or_else(llm_infer_cal_web::app);

    println!("llm-infer-cal web listening on http://{addr}");
    axum::serve(listener, app)
        .await
        .expect("run web API");
}
