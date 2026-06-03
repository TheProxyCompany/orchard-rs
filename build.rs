use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const REQUIRED_PROFILE_FILES: &[&str] = &[
    "capabilities.yaml",
    "control_tokens.json",
    "generation.yaml",
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
    generated.push_str("    pub profile_name: &'static str,\n");
    generated.push_str("    pub capabilities: &'static str,\n");
    generated.push_str("    pub control_tokens: &'static str,\n");
    generated.push_str("    pub generation: &'static str,\n");
    generated.push_str("}\n\n");

    for (profile_name, _, templates) in &profile_names {
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
        for (_, template_file) in templates {
            let file_path = profile_dir.join(template_file);
            let content = fs::read_to_string(&file_path)?;
            generated.push_str(&format!(
                "static {}: &str = {};\n",
                profile_file_const_name(&const_prefix, template_file),
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
    for (profile_name, model_types, _) in &profile_names {
        let const_prefix = profile_const_prefix(profile_name);
        for model_type in model_types {
            generated.push_str(&format!(
                "        {:?} => Some(EmbeddedProfile {{ model_type: {:?}, profile_name: {:?}, capabilities: {}, control_tokens: {}, generation: {} }}),\n",
                model_type,
                model_type,
                profile_name,
                profile_file_const_name(&const_prefix, "capabilities.yaml"),
                profile_file_const_name(&const_prefix, "control_tokens.json"),
                profile_file_const_name(&const_prefix, "generation.yaml"),
            ));
        }
    }
    generated.push_str("        _ => None,\n");
    generated.push_str("    }\n");
    generated.push_str("}\n\n");

    generated.push_str(
        "pub fn load_profile_template(profile_name: &str, template_type: &str) -> Option<&'static str> {\n",
    );
    generated.push_str("    match (profile_name, template_type) {\n");
    for (profile_name, _, templates) in &profile_names {
        let const_prefix = profile_const_prefix(profile_name);
        for (template_type, template_file) in templates {
            generated.push_str(&format!(
                "        ({:?}, {:?}) => Some({}),\n",
                profile_name,
                template_type,
                profile_file_const_name(&const_prefix, template_file)
            ));
        }
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

fn discover_profiles(
    profiles_dir: &Path,
) -> io::Result<Vec<(String, Vec<String>, Vec<(String, String)>)>> {
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
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Profile '{}' is missing required file '{}'",
                        file_name, required_file
                    ),
                ));
            }
        }
        let templates = discover_templates(&path)?;

        let control_tokens_path = path.join("control_tokens.json");
        let control_tokens_text = fs::read_to_string(control_tokens_path)?;
        let control_tokens: serde_json::Value =
            serde_json::from_str(&control_tokens_text).map_err(io::Error::other)?;
        let mut model_types = control_tokens
            .get("model_types")
            .and_then(serde_json::Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .filter(|values| !values.is_empty())
            .unwrap_or_else(|| vec![file_name.to_string()]);
        if !model_types.iter().any(|model_type| model_type == file_name) {
            model_types.push(file_name.to_string());
        }
        model_types.sort();
        model_types.dedup();
        profile_names.push((file_name.to_string(), model_types, templates));
    }

    profile_names.sort();
    Ok(profile_names)
}

fn discover_templates(profile_dir: &Path) -> io::Result<Vec<(String, String)>> {
    let templates_dir = profile_dir.join("templates");
    let mut templates = Vec::new();
    if profile_dir.join("chat_template.jinja").is_file() {
        templates.push(("default".to_string(), "chat_template.jinja".to_string()));
    }
    if templates_dir.is_dir() {
        for entry in fs::read_dir(&templates_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|name| name.to_str()) != Some("jinja") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|name| name.to_str()) else {
                continue;
            };
            if templates
                .iter()
                .any(|(template_type, _)| template_type == stem)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Profile '{}' defines duplicate template '{}'",
                        profile_dir.display(),
                        stem
                    ),
                ));
            }
            templates.push((stem.to_string(), format!("templates/{stem}.jinja")));
        }
    }
    if templates.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Profile '{}' has no templates", profile_dir.display()),
        ));
    }
    templates.sort();
    Ok(templates)
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
