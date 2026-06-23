use std::net::SocketAddr;

#[tokio::main]
async fn main() {
    let addr = std::env::var("LLM_INFER_CAL_WEB_ADDR")
        .ok()
        .and_then(|value| value.parse::<SocketAddr>().ok())
        .unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 8080)));

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind web API listener");
    println!("llm-infer-cal web API listening on http://{addr}");
    axum::serve(listener, llm_infer_cal_web::app())
        .await
        .expect("run web API");
}
