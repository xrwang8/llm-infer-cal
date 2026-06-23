use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use llm_infer_cal_core::core::cache::{ArtifactCache, CacheKey};
use llm_infer_cal_core::model_source::base::{ModelArtifact, SiblingFile};
use serde_json::json;

fn artifact(sha: Option<&str>) -> ModelArtifact {
    ModelArtifact {
        source: "huggingface".to_string(),
        model_id: "deepseek-ai/DeepSeek-V4-Flash".to_string(),
        commit_sha: sha.map(str::to_string),
        config: json!({"model_type": "deepseek_v4", "hidden_size": 4096}),
        siblings: vec![
            SiblingFile {
                filename: "model-00001-of-00002.safetensors".to_string(),
                size: Some(100),
            },
            SiblingFile {
                filename: "config.json".to_string(),
                size: Some(10),
            },
        ],
    }
}

fn temp_cache_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "llm-infer-cal-rust-cache-{}-{name}-{nanos}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    dir
}

#[test]
fn cache_key_string_matches_python() {
    assert_eq!(
        CacheKey::new("huggingface", "foo/bar", Some("abc")).to_string(),
        "huggingface::foo/bar::abc"
    );
    assert_eq!(
        CacheKey::new("huggingface", "foo/bar", None).to_string(),
        "huggingface::foo/bar::HEAD"
    );
}

#[test]
fn set_then_get_and_bypass_match_python() {
    let dir = temp_cache_dir("basic");
    let cache = ArtifactCache::new(Some(&dir), 7 * 24 * 60 * 60).unwrap();
    let key = CacheKey::new("huggingface", "foo/bar", Some("abc"));

    cache.set(&key, &artifact(Some("abc"))).unwrap();
    let got = cache.get(&key, false).unwrap().unwrap();
    assert_eq!(got.model_id, "deepseek-ai/DeepSeek-V4-Flash");
    assert_eq!(got.siblings[0].filename, "model-00001-of-00002.safetensors");
    assert!(cache.get(&key, true).unwrap().is_none());

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn misses_and_invalidate_match_python() {
    let dir = temp_cache_dir("invalidate");
    let cache = ArtifactCache::new(Some(&dir), 7 * 24 * 60 * 60).unwrap();
    let key = CacheKey::new("huggingface", "foo/bar", Some("abc"));

    assert!(cache.get(&key, false).unwrap().is_none());
    cache.set(&key, &artifact(Some("abc"))).unwrap();
    assert!(cache.invalidate(&key).unwrap());
    assert!(cache.get(&key, false).unwrap().is_none());
    assert!(!cache.invalidate(&key).unwrap());

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn commit_sha_mismatch_and_none_sha_never_hit() {
    let dir = temp_cache_dir("sha");
    let cache = ArtifactCache::new(Some(&dir), 7 * 24 * 60 * 60).unwrap();
    let old_key = CacheKey::new("huggingface", "deepseek/V4", Some("abc"));
    let new_key = CacheKey::new("huggingface", "deepseek/V4", Some("def"));
    let none_key = CacheKey::new("huggingface", "deepseek/V4", None);

    cache.set(&old_key, &artifact(Some("abc"))).unwrap();
    assert!(cache.get(&old_key, false).unwrap().is_some());
    assert!(cache.get(&new_key, false).unwrap().is_none());
    assert!(cache.get(&none_key, false).unwrap().is_none());
    cache.set(&none_key, &artifact(None)).unwrap();
    assert!(cache.get(&none_key, false).unwrap().is_none());

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn zero_ttl_never_stores_observable_entry() {
    let dir = temp_cache_dir("ttl");
    let cache = ArtifactCache::new(Some(&dir), 0).unwrap();
    let key = CacheKey::new("huggingface", "foo/bar", Some("abc"));

    cache.set(&key, &artifact(Some("abc"))).unwrap();
    assert!(cache.get(&key, false).unwrap().is_none());

    let _ = fs::remove_dir_all(dir);
}
