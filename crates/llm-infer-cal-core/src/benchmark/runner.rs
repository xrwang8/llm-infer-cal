use serde::Deserialize;

use crate::core::evaluator::{EvaluationReport, Evaluator};

const DATASET_YAML: &str = include_str!("../../../../src/llm_cal/benchmark/dataset.yaml");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    Pass,
    Fail,
    Skip,
}

impl Status {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::Skip => "SKIP",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(untagged)]
pub enum ExpectedValue {
    String(String),
    Int(i64),
    Bool(bool),
}

impl std::fmt::Display for ExpectedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpectedValue::String(value) => f.write_str(value),
            ExpectedValue::Int(value) => write!(f, "{value}"),
            ExpectedValue::Bool(true) => f.write_str("True"),
            ExpectedValue::Bool(false) => f.write_str("False"),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct Expectation {
    pub field: String,
    #[serde(default)]
    pub expected: Option<ExpectedValue>,
    #[serde(default, deserialize_with = "deserialize_optional_u64")]
    pub expected_min: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_optional_u64")]
    pub expected_max: Option<u64>,
    pub source: String,
}

impl Expectation {
    pub fn expected(field: &str, expected: ExpectedValue, source: &str) -> Self {
        Self {
            field: field.to_string(),
            expected: Some(expected),
            expected_min: None,
            expected_max: None,
            source: source.to_string(),
        }
    }

    pub fn range(
        field: &str,
        expected_min: Option<u64>,
        expected_max: Option<u64>,
        source: &str,
    ) -> Self {
        Self {
            field: field.to_string(),
            expected: None,
            expected_min,
            expected_max,
            source: source.to_string(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct BenchmarkEntry {
    pub name: String,
    pub model_id: String,
    pub gpu: String,
    #[serde(default = "default_engine")]
    pub engine: String,
    #[serde(default)]
    pub expectations: Vec<Expectation>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct BenchmarkDataset {
    pub schema_version: u64,
    pub entries: Vec<BenchmarkEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckResult {
    pub entry_name: String,
    pub field: String,
    pub status: Status,
    pub predicted: String,
    pub expected: String,
    pub source: String,
    pub note: Option<String>,
}

impl CheckResult {
    pub fn new(
        entry_name: &str,
        field: &str,
        status: Status,
        predicted: &str,
        expected: &str,
        source: &str,
    ) -> Self {
        Self {
            entry_name: entry_name.to_string(),
            field: field.to_string(),
            status,
            predicted: predicted.to_string(),
            expected: expected.to_string(),
            source: source.to_string(),
            note: None,
        }
    }

    fn with_note(mut self, note: String) -> Self {
        self.note = Some(note);
        self
    }
}

pub fn load_dataset() -> Result<BenchmarkDataset, serde_yaml::Error> {
    serde_yaml::from_str(DATASET_YAML)
}

pub fn run_all(evaluator: &Evaluator, dataset: &BenchmarkDataset) -> Vec<CheckResult> {
    let mut results = Vec::new();

    for entry in &dataset.entries {
        let report = evaluator.evaluate(
            &entry.model_id,
            &entry.gpu,
            &entry.engine,
            Default::default(),
        );
        match report {
            Ok(report) => {
                results.extend(
                    entry
                        .expectations
                        .iter()
                        .map(|expectation| check_one(&entry.name, &report, expectation)),
                );
            }
            Err(error) => {
                let note = format!("{error}");
                results.extend(entry.expectations.iter().map(|expectation| {
                    CheckResult::new(
                        &entry.name,
                        &expectation.field,
                        Status::Skip,
                        "(evaluation failed)",
                        &fmt_expected(expectation),
                        &expectation.source,
                    )
                    .with_note(note.clone())
                }));
            }
        }
    }

    results
}

pub fn check_one(
    entry_name: &str,
    report: &EvaluationReport,
    expectation: &Expectation,
) -> CheckResult {
    let (predicted, status) = evaluate_field(report, expectation);
    CheckResult::new(
        entry_name,
        &expectation.field,
        status,
        &predicted,
        &fmt_expected(expectation),
        &expectation.source,
    )
}

pub fn evaluate_field(report: &EvaluationReport, expectation: &Expectation) -> (String, Status) {
    match expectation.field.as_str() {
        "attention_variant" => {
            let actual = report
                .profile
                .attention
                .as_ref()
                .map(|attention| attention.variant.as_str())
                .unwrap_or("(none)");
            (
                actual.to_string(),
                pass_if_string(actual, expectation.expected.as_ref()),
            )
        }
        "quantization" => {
            let actual = report.weight.quantization_guess.value.as_str();
            (
                actual.to_string(),
                pass_if_string(actual, expectation.expected.as_ref()),
            )
        }
        "is_moe" => {
            let actual = report.profile.is_moe();
            (
                actual.to_string(),
                pass_if_bool(actual, expectation.expected.as_ref()),
            )
        }
        "weight_bytes" => {
            let actual = report.weight.total_bytes.value;
            let low = expectation.expected_min.unwrap_or(0);
            let high = expectation.expected_max.unwrap_or(1_u64 << 62);
            (
                fmt_u64(actual),
                if (low..=high).contains(&actual) {
                    Status::Pass
                } else {
                    Status::Fail
                },
            )
        }
        "fleet_prod_gpus" => {
            let Some(prod) = prod_option(report) else {
                return fleet_missing(report);
            };
            let passed = expectation
                .expected
                .as_ref()
                .and_then(expected_as_i64)
                .is_some_and(|expected| prod.gpu_count as i64 == expected);
            (
                prod.gpu_count.to_string(),
                if passed { Status::Pass } else { Status::Fail },
            )
        }
        "fleet_prod_gpus_at_most" => {
            let Some(prod) = prod_option(report) else {
                return fleet_missing(report);
            };
            let expected = expectation
                .expected
                .as_ref()
                .and_then(expected_as_i64)
                .unwrap_or(0);
            (
                format!("{} (max {expected})", prod.gpu_count),
                if prod.gpu_count as i64 <= expected {
                    Status::Pass
                } else {
                    Status::Fail
                },
            )
        }
        _ => ("(unknown field)".to_string(), Status::Skip),
    }
}

pub fn fmt_expected(expectation: &Expectation) -> String {
    if let Some(expected) = &expectation.expected {
        return expected.to_string();
    }

    if expectation.expected_min.is_some() || expectation.expected_max.is_some() {
        let low = expectation
            .expected_min
            .map(fmt_u64)
            .unwrap_or_else(|| "-∞".to_string());
        let high = expectation
            .expected_max
            .map(fmt_u64)
            .unwrap_or_else(|| "+∞".to_string());
        return format!("[{low}, {high}]");
    }

    "(unspecified)".to_string()
}

pub fn exit_code_from(results: &[CheckResult]) -> i32 {
    if results.iter().any(|result| result.status == Status::Fail) {
        1
    } else {
        0
    }
}

fn pass_if_string(actual: &str, expected: Option<&ExpectedValue>) -> Status {
    if matches!(expected, Some(ExpectedValue::String(expected)) if expected == actual) {
        Status::Pass
    } else {
        Status::Fail
    }
}

fn pass_if_bool(actual: bool, expected: Option<&ExpectedValue>) -> Status {
    if matches!(expected, Some(ExpectedValue::Bool(expected)) if *expected == actual) {
        Status::Pass
    } else {
        Status::Fail
    }
}

fn expected_as_i64(expected: &ExpectedValue) -> Option<i64> {
    match expected {
        ExpectedValue::Int(value) => Some(*value),
        ExpectedValue::String(value) => value.parse().ok(),
        ExpectedValue::Bool(_) => None,
    }
}

fn prod_option(report: &EvaluationReport) -> Option<&crate::fleet::planner::FleetOption> {
    report
        .fleet
        .as_ref()?
        .options
        .iter()
        .find(|option| option.tier == "prod")
}

fn fleet_missing(report: &EvaluationReport) -> (String, Status) {
    if report.fleet.is_none() {
        ("(no fleet)".to_string(), Status::Skip)
    } else {
        ("(no prod tier)".to_string(), Status::Skip)
    }
}

fn fmt_u64(value: u64) -> String {
    let text = value.to_string();
    let mut out = String::with_capacity(text.len() + text.len() / 3);
    let first_group = text.len() % 3;
    for (idx, ch) in text.chars().enumerate() {
        if idx > 0 && (idx == first_group || (idx > first_group && (idx - first_group) % 3 == 0)) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn default_engine() -> String {
    "vllm".to_string()
}

fn deserialize_optional_u64<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_yaml::Value>::deserialize(deserializer)?;
    let Some(value) = value else {
        return Ok(None);
    };
    match value {
        serde_yaml::Value::Number(number) => number
            .as_u64()
            .ok_or_else(|| {
                serde::de::Error::custom(format!("expected non-negative integer, got {number}"))
            })
            .map(Some),
        serde_yaml::Value::String(text) => text
            .replace('_', "")
            .parse::<u64>()
            .map(Some)
            .map_err(serde::de::Error::custom),
        other => Err(serde::de::Error::custom(format!(
            "expected integer or numeric string, got {other:?}"
        ))),
    }
}
