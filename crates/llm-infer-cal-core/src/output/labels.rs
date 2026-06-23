use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Label {
    Verified,
    Inferred,
    Estimated,
    Cited,
    Unverified,
    Unknown,
    LlmOpinion,
}

impl Label {
    pub const fn all() -> [Label; 7] {
        [
            Label::Verified,
            Label::Inferred,
            Label::Estimated,
            Label::Cited,
            Label::Unverified,
            Label::Unknown,
            Label::LlmOpinion,
        ]
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Label::Verified => "verified",
            Label::Inferred => "inferred",
            Label::Estimated => "estimated",
            Label::Cited => "cited",
            Label::Unverified => "unverified",
            Label::Unknown => "unknown",
            Label::LlmOpinion => "llm-opinion",
        }
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnnotatedValue<T> {
    pub value: T,
    pub label: Label,
    pub source: Option<String>,
}

impl<T> AnnotatedValue<T> {
    pub fn new(value: T, label: Label, source: Option<&str>) -> Self {
        Self {
            value,
            label,
            source: source.map(str::to_string),
        }
    }

    pub fn render_tag(&self) -> String {
        format!("[{}]", self.label.as_str())
    }
}
