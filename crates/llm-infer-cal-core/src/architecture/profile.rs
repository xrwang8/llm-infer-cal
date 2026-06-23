#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Family {
    Transformer,
    StateSpace,
    Unknown,
}

impl Family {
    pub const fn as_str(self) -> &'static str {
        match self {
            Family::Transformer => "transformer",
            Family::StateSpace => "state_space",
            Family::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Confidence {
    High,
    Medium,
    Low,
}

impl Confidence {
    pub const fn as_str(self) -> &'static str {
        match self {
            Confidence::High => "high",
            Confidence::Medium => "medium",
            Confidence::Low => "low",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionVariant {
    Mha,
    Gqa,
    Mqa,
    Mla,
    Nsa,
    CsaHca,
}

impl AttentionVariant {
    pub const fn as_str(self) -> &'static str {
        match self {
            AttentionVariant::Mha => "MHA",
            AttentionVariant::Gqa => "GQA",
            AttentionVariant::Mqa => "MQA",
            AttentionVariant::Mla => "MLA",
            AttentionVariant::Nsa => "NSA",
            AttentionVariant::CsaHca => "CSA_HCA",
        }
    }

    pub const fn is_sparse(self) -> bool {
        matches!(self, AttentionVariant::Nsa | AttentionVariant::CsaHca)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttentionTraits {
    pub variant: AttentionVariant,
    pub num_heads: u64,
    pub num_kv_heads: u64,
    pub head_dim: u64,
    pub q_lora_rank: Option<u64>,
    pub kv_lora_rank: Option<u64>,
    pub qk_rope_head_dim: Option<u64>,
    pub compress_ratios: Option<Vec<u64>>,
    pub nsa_topk: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MoeTraits {
    pub num_routed_experts: u64,
    pub num_shared_experts: u64,
    pub num_experts_per_tok: u64,
    pub moe_intermediate_size: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PositionTraits {
    pub rope_type: Option<String>,
    pub rope_theta: Option<f64>,
    pub rope_scaling_factor: Option<f64>,
    pub max_position_embeddings: Option<u64>,
}

impl Default for PositionTraits {
    fn default() -> Self {
        Self {
            rope_type: Some("rope".to_string()),
            rope_theta: None,
            rope_scaling_factor: None,
            max_position_embeddings: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArchitectureProfile {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub family: Family,
    pub num_hidden_layers: u64,
    pub hidden_size: u64,
    pub vocab_size: u64,
    pub confidence: Confidence,
    pub attention: Option<AttentionTraits>,
    pub moe: Option<MoeTraits>,
    pub position: Option<PositionTraits>,
    pub sliding_window: Option<u64>,
    pub intermediate_size: Option<u64>,
    pub tie_word_embeddings: bool,
    pub auxiliary: HashMap<String, Value>,
}

impl ArchitectureProfile {
    pub fn is_moe(&self) -> bool {
        self.moe.is_some()
    }

    pub fn is_sparse_attention(&self) -> bool {
        self.attention
            .as_ref()
            .is_some_and(|attention| attention.variant.is_sparse())
    }
}

impl Default for ArchitectureProfile {
    fn default() -> Self {
        Self {
            model_type: String::new(),
            architectures: Vec::new(),
            family: Family::Unknown,
            num_hidden_layers: 0,
            hidden_size: 0,
            vocab_size: 0,
            confidence: Confidence::Low,
            attention: None,
            moe: None,
            position: None,
            sliding_window: None,
            intermediate_size: None,
            tie_word_embeddings: false,
            auxiliary: HashMap::new(),
        }
    }
}
use std::collections::HashMap;

use serde_json::Value;
