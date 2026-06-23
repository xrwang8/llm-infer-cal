use std::collections::HashMap;

use llm_infer_cal_core::common::i18n::{
    detect_locale_from_env_values, get_locale, set_locale, t, t_with,
};
use llm_infer_cal_core::engine_compat::loader::{find_match, load_matrix};

#[test]
fn engine_matrix_loads_and_matches_deepseek_v4() {
    let matrix = load_matrix().unwrap();
    assert_eq!(matrix.schema_version, 2);
    assert!(!matrix.entries.is_empty());

    let vllm = find_match("vllm", "deepseek_v4", None, Some(&matrix)).unwrap();
    assert_eq!(vllm.engine, "vllm");
    assert_eq!(vllm.support, "full");
    assert_eq!(vllm.verification_level, "cited");
    assert!(vllm.sources.iter().any(|source| source
        .url
        .as_deref()
        .unwrap_or("")
        .contains("v0.19.0")));
    assert!(vllm.caveats_en.iter().any(|caveat| caveat.contains("H800")));

    let sglang = find_match("sglang", "deepseek_v4", None, Some(&matrix)).unwrap();
    assert_eq!(sglang.support, "unverified");
    assert_eq!(sglang.verification_level, "unverified");
}

#[test]
fn engine_find_match_respects_version_and_case_like_rust_contract() {
    let current = find_match("VLLM", "DEEPSEEK_V4", Some("0.19.0"), None).unwrap();
    let older = find_match("vllm", "deepseek_v4", Some("0.18.0"), None);
    let invalid = find_match("vllm", "deepseek_v4", Some("not-a-version"), None).unwrap();
    let unknown = find_match("vllm", "brand_new_model_type_2030", None, None);

    assert_eq!(current.engine, "vllm");
    assert!(older.is_none());
    assert_eq!(invalid.matches_model_type, "deepseek_v4");
    assert!(unknown.is_none());
}

#[test]
fn engine_find_match_returns_highest_lower_bound_when_version_is_absent() {
    let matrix = load_matrix().unwrap();
    let entry = find_match("vllm", "llama", None, Some(&matrix)).unwrap();

    assert_eq!(entry.engine, "vllm");
    assert_eq!(entry.matches_model_type, "llama");
    assert_eq!(entry.support, "full");
}

#[test]
fn i18n_translates_unknown_keys_and_templates_like_rust_contract() {
    let original = get_locale();

    set_locale("en");
    assert_eq!(t("section.architecture"), "Architecture");
    assert_eq!(t("this.key.does.not.exist"), "this.key.does.not.exist");

    let mut args = HashMap::new();
    args.insert("variant", "CSA_HCA".to_string());
    args.insert("heads", "64".to_string());
    args.insert("kv_heads", "1".to_string());
    args.insert("head_dim", "512".to_string());
    let en = t_with("arch.attn_summary", &args);
    assert!(en.contains("CSA_HCA"));
    assert!(en.contains("heads=64"));

    set_locale("zh");
    assert_eq!(t("section.architecture"), "架构");
    assert_eq!(t("section.weights"), "权重");
    assert_eq!(t("section.kv_cache"), "单请求 KV Cache（BF16/FP16）");
    let zh = t_with("arch.attn_summary", &args);
    assert!(zh.contains("CSA_HCA"));
    assert!(zh.contains('（'));

    set_locale(original.as_str());
}

#[test]
fn i18n_detects_locale_from_env_order_like_rust_contract() {
    assert_eq!(
        detect_locale_from_env_values([
            ("LC_ALL", Some("zh_TW.UTF-8")),
            ("LC_MESSAGES", None),
            ("LANG", Some("en_US.UTF-8")),
        ]),
        "zh"
    );
    assert_eq!(
        detect_locale_from_env_values([
            ("LC_ALL", None),
            ("LC_MESSAGES", None),
            ("LANG", Some("zh_CN.UTF-8")),
        ]),
        "zh"
    );
    assert_eq!(
        detect_locale_from_env_values([
            ("LC_ALL", None),
            ("LC_MESSAGES", None),
            ("LANG", Some("en_US.UTF-8")),
        ]),
        "en"
    );
    assert_eq!(
        detect_locale_from_env_values([("LC_ALL", None), ("LC_MESSAGES", None), ("LANG", None),]),
        "en"
    );
}
