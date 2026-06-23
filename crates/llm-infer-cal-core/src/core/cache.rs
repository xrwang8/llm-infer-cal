use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::model_source::base::ModelArtifact;

const DEFAULT_TTL_SECONDS: u64 = 7 * 24 * 60 * 60;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheKey {
    pub source: String,
    pub model_id: String,
    pub commit_sha: Option<String>,
}

impl CacheKey {
    pub fn new(source: &str, model_id: &str, commit_sha: Option<&str>) -> Self {
        Self {
            source: source.to_string(),
            model_id: model_id.to_string(),
            commit_sha: commit_sha.map(str::to_string),
        }
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}::{}::{}",
            self.source,
            self.model_id,
            self.commit_sha.as_deref().unwrap_or("HEAD")
        )
    }
}

#[derive(Clone, Debug)]
pub struct ArtifactCache {
    cache_dir: PathBuf,
    ttl_seconds: u64,
}

impl ArtifactCache {
    pub fn new(cache_dir: Option<&Path>, ttl_seconds: u64) -> io::Result<Self> {
        let cache_dir = cache_dir
            .map(Path::to_path_buf)
            .unwrap_or_else(default_cache_dir);
        fs::create_dir_all(&cache_dir)?;
        Ok(Self {
            cache_dir,
            ttl_seconds,
        })
    }

    pub fn with_default_ttl(cache_dir: Option<&Path>) -> io::Result<Self> {
        Self::new(cache_dir, DEFAULT_TTL_SECONDS)
    }

    pub fn get(&self, key: &CacheKey, bypass: bool) -> io::Result<Option<ModelArtifact>> {
        if bypass || key.commit_sha.is_none() {
            return Ok(None);
        }

        let path = self.path_for_key(key);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path)?;
        let entry: CacheEntry = match serde_json::from_slice(&bytes) {
            Ok(entry) => entry,
            Err(_) => return Ok(None),
        };
        if entry.expires_at_epoch_seconds <= now_epoch_seconds() {
            let _ = fs::remove_file(path);
            return Ok(None);
        }
        Ok(Some(entry.artifact))
    }

    pub fn set(&self, key: &CacheKey, artifact: &ModelArtifact) -> io::Result<()> {
        if key.commit_sha.is_none() || self.ttl_seconds == 0 {
            return Ok(());
        }

        let entry = CacheEntry {
            expires_at_epoch_seconds: now_epoch_seconds().saturating_add(self.ttl_seconds),
            artifact: artifact.clone(),
        };
        let bytes = serde_json::to_vec(&entry).map_err(io::Error::other)?;
        fs::write(self.path_for_key(key), bytes)
    }

    pub fn invalidate(&self, key: &CacheKey) -> io::Result<bool> {
        let path = self.path_for_key(key);
        if !path.exists() {
            return Ok(false);
        }
        fs::remove_file(path)?;
        Ok(true)
    }

    pub fn clear(&self) -> io::Result<()> {
        if self.cache_dir.exists() {
            for entry in fs::read_dir(&self.cache_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    fs::remove_file(path)?;
                }
            }
        }
        Ok(())
    }

    fn path_for_key(&self, key: &CacheKey) -> PathBuf {
        let mut hasher = DefaultHasher::new();
        key.to_string().hash(&mut hasher);
        self.cache_dir
            .join(format!("{:016x}.json", hasher.finish()))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CacheEntry {
    expires_at_epoch_seconds: u64,
    artifact: ModelArtifact,
}

fn default_cache_dir() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("llm-infer-cal")
}

fn now_epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}
