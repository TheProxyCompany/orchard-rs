//! Deterministic local Q8 conversion of the pinned MoshiRAG voice checkpoint.
//! Small/norm tensors remain F32. Source assets are immutable and unchanged;
//! derived GGUFs are atomically published under Orchard's HF cache namespace.
use crate::{Error, Result};
use candle_core::{
    quantized::{gguf_file, GgmlDType, QTensor},
    safetensors::MmapedSafetensors,
    Device,
};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::BufWriter,
    path::{Path, PathBuf},
};

pub(super) fn q8_voice(source: &Path) -> Result<PathBuf> {
    let identity = format!(
        "moshi-rag-q8_0-v1:{}:{}",
        source.display(),
        fs::metadata(source)?.len()
    );
    let key = format!("{:x}", Sha256::digest(identity.as_bytes()));
    let root = hf_hub::Cache::from_env().path().join("orchard-conversions");
    fs::create_dir_all(&root)?;
    let target = root.join(format!("{key}.gguf"));
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(root.join(format!("{key}.lock")))?;
    lock.lock()?;
    if target.is_file() {
        let mut file = File::open(&target)?;
        let header = gguf_file::Content::read(&mut file).map_err(native_error)?;
        let source_matches = matches!(header.metadata.get("orchard.source"), Some(gguf_file::Value::String(value)) if value == &identity);
        let file_size = file.metadata()?.len();
        let complete = !header.tensor_infos.is_empty()
            && header.tensor_infos.values().all(|info| {
                let elements = info.shape.elem_count();
                let block = info.ggml_dtype.block_size();
                if elements % block != 0 {
                    return false;
                }
                let bytes = (elements / block).checked_mul(info.ggml_dtype.type_size());
                bytes
                    .and_then(|bytes| {
                        header
                            .tensor_data_offset
                            .checked_add(info.offset)?
                            .checked_add(bytes as u64)
                    })
                    .is_some_and(|end| end <= file_size)
            });
        if !source_matches || !complete {
            return Err(Error::Other(format!(
                "Derived MoshiRAG checkpoint has stale provenance or truncated tensors: {}",
                target.display()
            )));
        }
        return Ok(target);
    }
    let source_weights = unsafe { MmapedSafetensors::new(source) }.map_err(native_error)?;
    let mut tensors = source_weights.tensors();
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let mut quantized = Vec::with_capacity(tensors.len());
    for (name, view) in tensors {
        let shape = view.shape();
        let dtype = if shape.len() >= 2 && shape.last().is_some_and(|n| n % 32 == 0) {
            GgmlDType::Q8_0
        } else {
            GgmlDType::F32
        };
        let tensor = source_weights
            .load(&name, &Device::Cpu)
            .map_err(native_error)?;
        quantized.push((
            name,
            QTensor::quantize(&tensor, dtype).map_err(native_error)?,
        ));
    }
    let temporary = root.join(format!("{key}.{}.part", std::process::id()));
    let result = (|| -> Result<()> {
        let mut writer = BufWriter::new(File::create(&temporary)?);
        let provenance = gguf_file::Value::String(identity);
        let refs: Vec<_> = quantized
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        gguf_file::write(&mut writer, &[("orchard.source", &provenance)], &refs)
            .map_err(native_error)?;
        use std::io::Write;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        fs::rename(&temporary, &target)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result?;
    Ok(target)
}
fn native_error(error: candle_core::Error) -> Error {
    Error::Other(format!("MoshiRAG Q8: {error}"))
}
