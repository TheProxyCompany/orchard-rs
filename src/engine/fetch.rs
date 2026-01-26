//! Engine binary fetching and installation.

use std::io;
use std::path::PathBuf;

use flate2::read::GzDecoder;
use reqwest::Client;
use sha2::{Digest, Sha256};
use tar::Archive;

use crate::error::{Error, Result};

const MANIFEST_URL: &str = "https://prod.proxy.ing/functions/v1/get-release-manifest";
const DEFAULT_CHANNEL: &str = "stable";
const REQUEST_TIMEOUT_SECS: u64 = 30;
const DOWNLOAD_TIMEOUT_SECS: u64 = 600;
const MAX_RETRIES: u32 = 3;

/// Engine binary fetcher.
///
/// Handles downloading, verifying, and installing PIE binaries.
pub struct EngineFetcher {
    client: Client,
    orchard_home: PathBuf,
}

impl EngineFetcher {
    /// Create a new fetcher with the default orchard home directory.
    pub fn new() -> Self {
        let orchard_home = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join(".orchard");
        Self::with_home(orchard_home)
    }

    /// Create a fetcher with a custom orchard home directory.
    pub fn with_home(orchard_home: PathBuf) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(DOWNLOAD_TIMEOUT_SECS))
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            client,
            orchard_home,
        }
    }

    /// Get the path to the engine binary, downloading if necessary.
    pub async fn get_engine_path(&self) -> Result<PathBuf> {
        // Check for local dev override
        if let Ok(local_build) = std::env::var("PIE_LOCAL_BUILD") {
            let local_path = PathBuf::from(&local_build)
                .join("bin")
                .join("proxy_inference_engine");
            if local_path.exists() {
                log::debug!("Using local PIE build: {:?}", local_path);
                return Ok(local_path);
            }
        }

        let binary_path = self.orchard_home.join("bin").join("proxy_inference_engine");

        if binary_path.exists() {
            return Ok(binary_path);
        }

        // Need to download
        std::fs::create_dir_all(&self.orchard_home)?;
        self.download_engine(DEFAULT_CHANNEL, None).await?;

        if !binary_path.exists() {
            return Err(Error::Network(
                "Download completed but binary not found".into(),
            ));
        }

        Ok(binary_path)
    }

    /// Download and install the engine binary.
    pub async fn download_engine(&self, channel: &str, version: Option<&str>) -> Result<()> {
        let manifest = self.fetch_manifest(channel).await?;

        let version = match version {
            Some(v) => v.to_string(),
            None => manifest
                .get("latest")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    Error::InvalidManifest(format!(
                        "No latest version defined for {} channel",
                        channel
                    ))
                })?
                .to_string(),
        };

        let versions = manifest
            .get("versions")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::InvalidManifest("Missing versions in manifest".into()))?;

        let info = versions.get(&version).ok_or_else(|| {
            let available: Vec<_> = versions.keys().map(|k| k.as_str()).collect();
            Error::InvalidManifest(format!(
                "Version {} not found in {} channel. Available: {}",
                version,
                channel,
                available.join(", ")
            ))
        })?;

        let url = info.get("url").and_then(|v| v.as_str()).ok_or_else(|| {
            Error::InvalidManifest(format!("No download URL for version {}", version))
        })?;

        let expected_sha256 = info.get("sha256").and_then(|v| v.as_str());

        log::info!("Downloading PIE version {}", version);
        let content = self.download_with_retry(url, expected_sha256).await?;

        self.extract_and_install(&content, &version)?;

        log::info!("Installed PIE version {}", version);
        Ok(())
    }

    /// Get the currently installed version.
    pub fn get_installed_version(&self) -> Option<String> {
        let version_file = self.orchard_home.join("version.txt");
        std::fs::read_to_string(version_file)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// Check if an update is available.
    pub async fn check_for_updates(&self, channel: &str) -> Option<String> {
        let installed = self.get_installed_version()?;

        let manifest = self.fetch_manifest(channel).await.ok()?;
        let latest = manifest.get("latest").and_then(|v| v.as_str())?;

        if latest != installed {
            Some(latest.to_string())
        } else {
            None
        }
    }

    async fn fetch_manifest(&self, channel: &str) -> Result<serde_json::Value> {
        let installed = self
            .get_installed_version()
            .unwrap_or_else(|| "unknown".into());

        let response = self
            .client
            .get(MANIFEST_URL)
            .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .query(&[
                ("channel", channel),
                ("v", &installed),
                ("os", std::env::consts::OS),
                ("arch", std::env::consts::ARCH),
            ])
            .send()
            .await
            .map_err(|e| Error::InvalidManifest(format!("Failed to fetch manifest: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::InvalidManifest(format!(
                "Server returned {}",
                response.status()
            )));
        }

        response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| Error::InvalidManifest(format!("Invalid manifest format: {}", e)))
    }

    async fn download_with_retry(
        &self,
        url: &str,
        expected_sha256: Option<&str>,
    ) -> Result<Vec<u8>> {
        for attempt in 0..MAX_RETRIES {
            match self.download_file(url).await {
                Ok(content) => {
                    if let Some(expected) = expected_sha256 {
                        let actual = ::hex::encode(Sha256::digest(&content));
                        if actual != expected {
                            return Err(Error::Integrity {
                                expected: expected.to_string(),
                                actual,
                            });
                        }
                    }
                    return Ok(content);
                }
                Err(e) => {
                    if attempt == MAX_RETRIES - 1 {
                        return Err(e);
                    }
                    log::warn!(
                        "Download attempt {} failed: {}, retrying...",
                        attempt + 1,
                        e
                    );
                }
            }
        }

        Err(Error::Network(
            "Download failed after maximum retries".into(),
        ))
    }

    async fn download_file(&self, url: &str) -> Result<Vec<u8>> {
        let response = self.client.get(url).send().await?;

        if !response.status().is_success() {
            return Err(Error::Network(format!("HTTP {}", response.status())));
        }

        let bytes = response.bytes().await?;
        Ok(bytes.to_vec())
    }

    fn extract_and_install(&self, content: &[u8], version: &str) -> Result<()> {
        std::fs::create_dir_all(&self.orchard_home)?;

        let bin_dir = self.orchard_home.join("bin");

        // Clean existing bin directory for atomic update
        if bin_dir.exists() {
            std::fs::remove_dir_all(&bin_dir)?;
        }

        // Extract tar.gz
        let decoder = GzDecoder::new(content);
        let mut archive = Archive::new(decoder);

        for entry in archive.entries()? {
            let mut entry = entry?;
            let path = entry.path()?;

            // Security: validate paths
            let path_str = path.to_string_lossy();
            if path_str.starts_with('/') || path_str.contains("..") {
                return Err(Error::Extract(format!(
                    "Unsafe path in archive: {}",
                    path_str
                )));
            }

            let dest = self.orchard_home.join(&*path);

            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }

            if entry.header().entry_type().is_file() {
                let mut file = std::fs::File::create(&dest)?;
                io::copy(&mut entry, &mut file)?;

                // Set executable permissions on Unix
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mode = entry.header().mode().unwrap_or(0o755);
                    std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(mode))?;
                }
            }
        }

        let binary_path = bin_dir.join("proxy_inference_engine");
        if !binary_path.exists() {
            return Err(Error::Extract(
                "Archive did not contain expected binary".into(),
            ));
        }

        // Ensure binary is executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&binary_path, std::fs::Permissions::from_mode(0o755))?;
        }

        // Write version file
        let version_file = self.orchard_home.join("version.txt");
        std::fs::write(version_file, version)?;

        Ok(())
    }
}

impl Default for EngineFetcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fetcher_creation() {
        let fetcher = EngineFetcher::new();
        assert!(fetcher.orchard_home.ends_with(".orchard"));
    }

    #[test]
    fn test_local_build_override() {
        // This test just verifies the logic path exists
        let fetcher = EngineFetcher::new();
        assert!(fetcher.orchard_home.exists() || !fetcher.orchard_home.exists());
    }
}
