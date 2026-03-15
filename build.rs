use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const REQUIRED_PROFILE_FILES: &[&str] = &[
    "chat_template.jinja",
    "capabilities.yaml",
    "control_tokens.json",
];
const SHARED_TEMPLATE_FILES: &[&str] = &["tool_macros.jinja"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=profiles");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let profiles_dir = manifest_dir.join("profiles");
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let generated_path = out_dir.join("embedded_profiles.rs");

    let generated = build_embedded_profiles_source(&profiles_dir)?;
    fs::write(generated_path, generated)?;

    Ok(())
}

fn build_embedded_profiles_source(
    profiles_dir: &Path,
) -> Result<String, Box<dyn std::error::Error>> {
    let profile_names = discover_profiles(profiles_dir)?;
    let mut generated = String::new();

    generated.push_str("#[derive(Clone, Copy)]\n");
    generated.push_str("pub struct EmbeddedProfile {\n");
    generated.push_str("    pub model_type: &'static str,\n");
    generated.push_str("    pub chat_template: &'static str,\n");
    generated.push_str("    pub capabilities: &'static str,\n");
    generated.push_str("    pub control_tokens: &'static str,\n");
    generated.push_str("}\n\n");

    for profile_name in &profile_names {
        let profile_dir = profiles_dir.join(profile_name);
        let const_prefix = profile_const_prefix(profile_name);

        for required_file in REQUIRED_PROFILE_FILES {
            let file_path = profile_dir.join(required_file);
            let content = fs::read_to_string(&file_path)?;
            generated.push_str(&format!(
                "static {}: &str = {};\n",
                profile_file_const_name(&const_prefix, required_file),
                rust_string_literal(&content)
            ));
        }
        generated.push('\n');
    }

    for shared_file in SHARED_TEMPLATE_FILES {
        let file_path = profiles_dir.join(shared_file);
        let content = fs::read_to_string(&file_path)?;
        generated.push_str(&format!(
            "static {}: &str = {};\n",
            shared_file_const_name(shared_file),
            rust_string_literal(&content)
        ));
    }

    generated.push_str(
        "\npub fn find_embedded_profile(model_type: &str) -> Option<EmbeddedProfile> {\n",
    );
    generated.push_str("    match model_type {\n");
    for profile_name in &profile_names {
        let const_prefix = profile_const_prefix(profile_name);
        generated.push_str(&format!(
            "        {:?} => Some(EmbeddedProfile {{ model_type: {:?}, chat_template: {}, capabilities: {}, control_tokens: {} }}),\n",
            profile_name,
            profile_name,
            profile_file_const_name(&const_prefix, "chat_template.jinja"),
            profile_file_const_name(&const_prefix, "capabilities.yaml"),
            profile_file_const_name(&const_prefix, "control_tokens.json"),
        ));
    }
    generated.push_str("        _ => None,\n");
    generated.push_str("    }\n");
    generated.push_str("}\n\n");

    generated.push_str("pub fn load_shared_template(name: &str) -> Option<&'static str> {\n");
    generated.push_str("    match name {\n");
    for shared_file in SHARED_TEMPLATE_FILES {
        generated.push_str(&format!(
            "        {:?} => Some({}),\n",
            shared_file,
            shared_file_const_name(shared_file)
        ));
    }
    generated.push_str("        _ => None,\n");
    generated.push_str("    }\n");
    generated.push_str("}\n");

    Ok(generated)
}

fn discover_profiles(profiles_dir: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut profile_names = Vec::new();

    for entry in fs::read_dir(profiles_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name == "_example" || file_name.starts_with('.') {
            continue;
        }

        for required_file in REQUIRED_PROFILE_FILES {
            let required_path = path.join(required_file);
            if !required_path.is_file() {
                return Err(format!(
                    "Profile '{}' is missing required file '{}'",
                    file_name, required_file
                )
                .into());
            }
        }

        profile_names.push(file_name.to_string());
    }

    profile_names.sort();
    Ok(profile_names)
}

fn profile_const_prefix(profile_name: &str) -> String {
    let mut prefix = String::from("PROFILE_");
    for ch in profile_name.chars() {
        if ch.is_ascii_alphanumeric() {
            prefix.push(ch.to_ascii_uppercase());
        } else {
            prefix.push('_');
        }
    }
    prefix
}

fn profile_file_const_name(const_prefix: &str, file_name: &str) -> String {
    format!("{}_{}", const_prefix, file_const_suffix(file_name))
}

fn shared_file_const_name(file_name: &str) -> String {
    format!("SHARED_{}", file_const_suffix(file_name))
}

fn file_const_suffix(file_name: &str) -> String {
    let mut suffix = String::new();
    for ch in file_name.chars() {
        if ch.is_ascii_alphanumeric() {
            suffix.push(ch.to_ascii_uppercase());
        } else {
            suffix.push('_');
        }
    }
    suffix
}

fn rust_string_literal(content: &str) -> String {
    format!("{content:?}")
}
