use llm_infer_cal_core::output::labels::{AnnotatedValue, Label};

#[test]
fn all_primary_labels_exist() {
    let labels = Label::all();
    let values: Vec<&'static str> = labels.iter().map(|label| label.as_str()).collect();

    for expected in [
        "verified",
        "inferred",
        "estimated",
        "cited",
        "unverified",
        "unknown",
        "llm-opinion",
    ] {
        assert!(values.contains(&expected), "missing label {expected}");
    }
    assert_eq!(values.len(), 7);
}

#[test]
fn annotated_value_preserves_data() {
    let value = AnnotatedValue::new(160_300_000_000_u64, Label::Verified, Some("HF siblings"));

    assert_eq!(value.value, 160_300_000_000);
    assert_eq!(value.label, Label::Verified);
    assert_eq!(value.source.as_deref(), Some("HF siblings"));
}

#[test]
fn render_tag_uses_bracket_format() {
    let value = AnnotatedValue::new(4.52_f64, Label::Inferred, None);

    assert_eq!(value.render_tag(), "[inferred]");
}

#[test]
fn label_displays_like_rust_contract_str_enum() {
    assert_eq!(Label::Verified.as_str(), "verified");
    assert_eq!(Label::Cited.to_string(), "cited");
}
