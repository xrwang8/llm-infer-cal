#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "net/http"
require "set"
require "time"
require "uri"

ROOT = File.expand_path("..", __dir__)
MANIFEST_PATH = File.join(ROOT, "crates/llm-infer-cal-core/data/builtin_model_manifest.json")

VLLM_MODELS_URL = "https://recipes.vllm.ai/models.json"
SGLANG_INDEX_URL = "https://docs.sglang.io/llms.txt"
VLLM_PROVIDER_NAMES = ["Qwen", "DeepSeek", "Moonshot AI", "GLM (Z-AI)"].freeze
SGLANG_VENDOR_PATHS = ["/Qwen/", "/DeepSeek/", "/Moonshotai/", "/GLM/"].freeze
FAMILY_PAGE_IDS = [
  "Qwen/Qwen2.5-VL",
  "Qwen/Qwen3",
  "Qwen/Qwen3-Coder",
  "Qwen/Qwen3-Next",
  "Qwen/Qwen3-VL",
  "Qwen/Qwen3.5",
  "Qwen/Qwen3.6",
  "deepseek-ai/DeepSeek-V4",
  "zai-org/Glyph-FP8"
].freeze

PROVIDER_BY_PREFIX = {
  "Qwen/" => "Qwen",
  "deepseek-ai/" => "deepseek-ai",
  "moonshotai/" => "moonshotai",
  "zai-org/" => "zai-org",
  "ZhipuAI/" => "ZhipuAI",
  "nvidia/" => "nvidia"
}.freeze

ALIAS_MAP = {
  "zai-org/GLM-5.2" => ["ZhipuAI/GLM-5.2"],
  "zai-org/glm-5.2" => ["ZhipuAI/GLM-5.2"],
  "Qwen/qwen25-vl" => ["Qwen/Qwen2.5-VL-72B-Instruct"],
  "Qwen/qwen3-coder" => ["Qwen/Qwen3-Coder-480B-A35B-Instruct"],
  "Qwen/qwen3-next" => ["Qwen/Qwen3-Next-80B-A3B-Instruct"],
  "deepseek-ai/deepseek-v4" => ["deepseek-ai/DeepSeek-V4-Flash"]
}.freeze

def get_text(url)
  uri = URI(url)
  response = Net::HTTP.get_response(uri)
  raise "HTTP #{response.code} for #{url}" unless response.is_a?(Net::HTTPSuccess)

  response.body
end

def get_json(url)
  JSON.parse(get_text(url))
end

def provider_for(id)
  PROVIDER_BY_PREFIX.each do |prefix, provider|
    return provider if id.start_with?(prefix)
  end
  id.split("/", 2).first
end

def register(entries, id, mentioned_by:, provider: nil, aliases: [], vllm_recipe_json: nil, sglang_pages: [])
  return if id.nil? || id.empty?
  return if FAMILY_PAGE_IDS.include?(id)
  return if id.end_with?("-") || id.include?("...")
  return if id.end_with?("-NVFP")
  return if id == "nvidia/aime25"

  canonical = ALIAS_MAP.fetch(id, [id]).first
  entry = entries[canonical] ||= {
    "id" => canonical,
    "aliases" => [],
    "provider" => provider || provider_for(canonical),
    "preferred_source" => canonical.start_with?("ZhipuAI/") ? "modelscope" : "huggingface",
    "mentioned_by" => [],
    "vllm_recipe_json" => nil,
    "sglang_pages" => []
  }
  (aliases + ([id] unless id == canonical).to_a).each do |candidate|
    next if candidate == canonical
    entry["aliases"] << candidate unless entry["aliases"].include?(candidate)
  end
  Array(mentioned_by).each do |source|
    entry["mentioned_by"] << source unless entry["mentioned_by"].include?(source)
  end
  entry["vllm_recipe_json"] ||= vllm_recipe_json if vllm_recipe_json
  Array(sglang_pages).each do |page|
    entry["sglang_pages"] << page unless entry["sglang_pages"].include?(page)
  end
end

def extract_repo_ids(text)
  text.scan(%r{\b(?:Qwen|deepseek-ai|moonshotai|zai-org|ZhipuAI|nvidia)/[A-Za-z0-9_.-]+})
      .uniq
      .reject { |id| id.end_with?("-") || id.include?("...") }
end

def qwen3_variants
  variants = []
  {
    "235B-A22B" => true,
    "30B-A3B" => true,
    "32B" => false,
    "14B" => false,
    "8B" => false,
    "4B" => true,
    "1.7B" => false,
    "0.6B" => false
  }.each do |base, has_thinking|
    ["", "-FP8"].each do |quant|
      if has_thinking
        variants << "Qwen/Qwen3-#{base}#{quant}"
        variants << "Qwen/Qwen3-#{base}-Instruct-2507#{quant}"
        variants << "Qwen/Qwen3-#{base}-Thinking-2507#{quant}"
      else
        variants << "Qwen/Qwen3-#{base}#{quant}"
      end
    end
  end
  variants
end

def extra_sglang_ids_for(page_url)
  case page_url
  when /Qwen3\.md$/
    qwen3_variants
  when /Qwen2\.5-VL\.md$/
    [
      "Qwen/Qwen2.5-VL-72B-Instruct",
      "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
      "Qwen/Qwen2.5-VL-7B-Instruct",
      "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    ]
  when /Qwen3-Coder\.md$/
    [
      "Qwen/Qwen3-Coder-30B-A3B-Instruct",
      "Qwen/Qwen3-Coder-480B-A35B-Instruct",
      "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
      "nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4"
    ]
  when /Qwen3-Next\.md$/
    [
      "Qwen/Qwen3-Next-80B-A3B-Instruct",
      "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
      "Qwen/Qwen3-Next-80B-A3B-Thinking",
      "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4"
    ]
  when /Qwen3-VL\.md$/
    [
      "Qwen/Qwen3-VL-235B-A22B-Instruct",
      "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
      "Qwen/Qwen3-VL-235B-A22B-Thinking",
      "nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4"
    ]
  when /Qwen3\.5\.md$/
    [
      "Qwen/Qwen3.5-0.8B",
      "Qwen/Qwen3.5-2B",
      "Qwen/Qwen3.5-4B",
      "Qwen/Qwen3.5-9B",
      "Qwen/Qwen3.5-27B",
      "Qwen/Qwen3.5-27B-FP8",
      "Qwen/Qwen3.5-27B-GPTQ-Int4",
      "Qwen/Qwen3.5-35B-A3B",
      "Qwen/Qwen3.5-35B-A3B-FP8",
      "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
      "Qwen/Qwen3.5-122B-A10B",
      "Qwen/Qwen3.5-122B-A10B-FP8",
      "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
      "Qwen/Qwen3.5-397B-A17B",
      "Qwen/Qwen3.5-397B-A17B-FP8",
      "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
      "nvidia/Qwen3.5-397B-A17B-NVFP4"
    ]
  when /Qwen3\.6\.md$/
    [
      "Qwen/Qwen3.6-27B",
      "Qwen/Qwen3.6-27B-FP8",
      "Qwen/Qwen3.6-35B-A3B",
      "Qwen/Qwen3.6-35B-A3B-FP8",
      "nvidia/Qwen3.6-35B-A3B-NVFP4"
    ]
  when /DeepSeek-V4\.md$/
    [
      "deepseek-ai/DeepSeek-V4-Flash",
      "deepseek-ai/DeepSeek-V4-Pro",
      "nvidia/DeepSeek-V4-Flash-NVFP4",
      "nvidia/DeepSeek-V4-Pro-NVFP4"
    ]
  else
    []
  end
end

entries = {}

vllm_models = get_json(VLLM_MODELS_URL)
vllm_models.each do |model|
  next unless VLLM_PROVIDER_NAMES.include?(model["provider"])
  id = model["hf_id"]
  register(
    entries,
    id,
    provider: model["provider"],
    mentioned_by: "vllm",
    vllm_recipe_json: URI.join("https://recipes.vllm.ai", model["json"]).to_s
  )
end

sglang_index = get_text(SGLANG_INDEX_URL)
sglang_pages = sglang_index.scan(/\((https:\/\/docs\.sglang\.io\/cookbook\/(?:autoregressive|diffusion)\/[^)]+\.md)\)/)
                           .flatten
                           .select { |url| SGLANG_VENDOR_PATHS.any? { |path| url.include?(path) } }
                           .uniq

sglang_pages.each do |page_url|
  text = get_text(page_url)
  (extract_repo_ids(text) + extra_sglang_ids_for(page_url)).uniq.each do |id|
    register(entries, id, mentioned_by: "sglang", sglang_pages: [page_url])
  end
end

models = entries.values.sort_by { |entry| entry["id"].downcase }
manifest = {
  "schema_version" => 1,
  "captured_at" => Time.now.utc.iso8601,
  "sources" => [
    { "kind" => "vllm_models_json", "url" => VLLM_MODELS_URL },
    { "kind" => "sglang_llms_index", "url" => SGLANG_INDEX_URL }
  ],
  "models" => models
}

File.write(MANIFEST_PATH, "#{JSON.pretty_generate(manifest)}\n")
warn "wrote #{MANIFEST_PATH}: #{models.length} models"
