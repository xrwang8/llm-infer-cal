#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "net/http"
require "thread"
require "time"
require "uri"

ROOT = File.expand_path("..", __dir__)
MANIFEST_PATH = File.join(ROOT, "crates/llm-infer-cal-core/data/builtin_model_manifest.json")
CATALOG_PATH = File.join(ROOT, "crates/llm-infer-cal-core/data/builtin_models.json")

HF_ENDPOINT = ENV.fetch("HF_ENDPOINT", "https://huggingface.co").delete_suffix("/")
MODELSCOPE_ENDPOINT = ENV.fetch("MODELSCOPE_ENDPOINT", "https://www.modelscope.cn").delete_suffix("/")
WORKERS = Integer(ENV.fetch("BUILTIN_CATALOG_WORKERS", "12"))
TIMEOUT_S = Integer(ENV.fetch("BUILTIN_CATALOG_TIMEOUT_S", "30"))

def get_json(url, params: {}, headers: {}, limit: 4)
  uri = URI(url)
  uri.query = URI.encode_www_form(params) unless params.empty?

  http = Net::HTTP.new(uri.host, uri.port)
  http.use_ssl = uri.scheme == "https"
  http.open_timeout = [TIMEOUT_S, 10].min
  http.read_timeout = TIMEOUT_S

  request = Net::HTTP::Get.new(uri)
  headers.each { |key, value| request[key] = value }
  response = http.request(request)

  if response.is_a?(Net::HTTPRedirection) && response["location"] && limit.positive?
    return get_json(URI.join(uri, response["location"]).to_s, headers: headers, limit: limit - 1)
  end

  unless response.is_a?(Net::HTTPSuccess)
    raise "HTTP #{response.code} for #{uri}"
  end

  parsed = JSON.parse(response.body)
  raise "expected JSON object from #{uri}" unless parsed.is_a?(Hash)

  parsed
end

def fetch_hf(model)
  id = model.fetch("id")
  headers = {}
  headers["Authorization"] = "Bearer #{ENV["HF_TOKEN"]}" if ENV["HF_TOKEN"]&.length&.positive?

  info = get_json("#{HF_ENDPOINT}/api/models/#{id}", params: { "blobs" => "true" }, headers: headers)
  sha = info["sha"].to_s
  revision = sha.empty? ? "main" : sha
  config = fetch_hf_config(id, revision, info, headers)
  siblings = hf_safetensors(info)

  catalog_entry(model, "huggingface", sha.empty? ? nil : sha, config, aggregate_safetensors(siblings))
end

def fetch_hf_config(id, revision, info, headers)
  get_json("#{HF_ENDPOINT}/#{id}/resolve/#{revision}/config.json", headers: headers)
rescue StandardError => config_error
  begin
    config = get_json("#{HF_ENDPOINT}/#{id}/resolve/#{revision}/model_index.json", headers: headers)
    config["_builtin_config_source"] = "model_index.json"
    config
  rescue StandardError
    api_config = info["config"]
    if api_config.is_a?(Hash)
      api_config.merge("_builtin_config_source" => "huggingface_api_config")
    else
      raise config_error
    end
  end
end

def hf_safetensors(info)
  Array(info["siblings"]).map do |sibling|
    filename = sibling["rfilename"] || sibling["filename"]
    next unless filename&.end_with?(".safetensors")

    size = numeric_value(sibling["size"]) || numeric_value(sibling.dig("lfs", "size"))
    { "filename" => filename, "size" => size }
  end.compact
end

def fetch_modelscope(model)
  id = model.fetch("id")
  headers = {}
  if ENV["MODELSCOPE_API_TOKEN"]&.length&.positive?
    headers["Authorization"] = "Bearer #{ENV["MODELSCOPE_API_TOKEN"]}"
  end

  sha = modelscope_latest_sha(id, headers)
  revision = sha || "master"
  files = get_json(
    "#{MODELSCOPE_ENDPOINT}/api/v1/models/#{id}/repo/files",
    params: { "Recursive" => "true", "Revision" => revision },
    headers: headers
  )
  config = get_json(
    "#{MODELSCOPE_ENDPOINT}/api/v1/models/#{id}/repo",
    params: { "FilePath" => "config.json", "Revision" => revision },
    headers: headers
  )

  catalog_entry(model, "modelscope", revision, config, aggregate_safetensors(modelscope_safetensors(files)))
end

def modelscope_latest_sha(id, headers)
  info = get_json("#{MODELSCOPE_ENDPOINT}/api/v1/models/#{id}", headers: headers)
  data = info["Data"]
  return nil unless data.is_a?(Hash)

  %w[LatestSha latest_sha Revision Sha].map { |key| data[key].to_s }.find { |value| !value.empty? }
rescue StandardError
  nil
end

def modelscope_safetensors(payload)
  data = payload["Data"]
  files =
    if data.is_a?(Hash) && data["Files"].is_a?(Array)
      data["Files"]
    elsif data.is_a?(Array)
      data
    else
      []
    end

  files.map do |file|
    next unless file.is_a?(Hash)
    next if file["Type"].to_s == "tree"

    filename = file["Path"] || file["Name"]
    next unless filename&.end_with?(".safetensors")

    { "filename" => filename, "size" => numeric_value(file["Size"] || file["size"]) }
  end.compact
end

def numeric_value(value)
  case value
  when Integer
    value
  when String
    Integer(value, exception: false)
  end
end

def aggregate_safetensors(siblings)
  total = siblings.map { |sibling| sibling["size"] }.compact.sum
  return siblings if total.zero?

  [{ "filename" => "model.safetensors", "size" => total }]
end

def catalog_entry(model, source, commit_sha, config, siblings)
  {
    "id" => model.fetch("id"),
    "aliases" => Array(model["aliases"]),
    "provider" => model["provider"],
    "preferred_source" => model["preferred_source"],
    "mentioned_by" => Array(model["mentioned_by"]),
    "vllm_recipe_json" => model["vllm_recipe_json"],
    "sglang_pages" => Array(model["sglang_pages"]),
    "snapshot_status" => "ok",
    "captured_from" => source,
    "commit_sha" => commit_sha,
    "config" => config,
    "siblings" => siblings
  }
end

def recipe_only_entry(model, error)
  recipe = fetch_recipe_metadata(model)
  config = {
    "model_type" => recipe.dig("model", "architecture") || "unknown",
    "architectures" => recipe_architectures(recipe),
    "max_position_embeddings" => recipe.dig("model", "context_length"),
    "_builtin_config_source" => recipe.empty? ? "recipe_reference" : "vllm_recipe_json",
    "_builtin_fetch_status" => "recipe_only",
    "_builtin_fetch_error" => error.to_s[0, 500],
    "_builtin_manifest_id" => model.fetch("id"),
    "_builtin_recipe" => compact_hash({
      "parameter_count" => recipe.dig("model", "parameter_count"),
      "architecture" => recipe.dig("model", "architecture"),
      "min_vllm_version" => recipe.dig("model", "min_vllm_version"),
      "context_length" => recipe.dig("model", "context_length"),
      "base_args" => recipe.dig("model", "base_args"),
      "base_env" => recipe.dig("model", "base_env"),
      "variants" => recipe["variants"],
      "recommended_command" => recipe["recommended_command"]
    })
  }.compact

  {
    "id" => model.fetch("id"),
    "aliases" => Array(model["aliases"]),
    "provider" => model["provider"],
    "preferred_source" => model["preferred_source"],
    "mentioned_by" => Array(model["mentioned_by"]),
    "vllm_recipe_json" => model["vllm_recipe_json"],
    "sglang_pages" => Array(model["sglang_pages"]),
    "snapshot_status" => "recipe_only",
    "snapshot_error" => error.to_s[0, 500],
    "commit_sha" => nil,
    "config" => config,
    "siblings" => []
  }
end

def fetch_recipe_metadata(model)
  url = model["vllm_recipe_json"]
  return {} unless url && !url.empty?

  get_json(url)
rescue StandardError
  {}
end

def recipe_architectures(recipe)
  guide = recipe["guide"].to_s
  if guide.include?("Glm4vForConditionalGeneration")
    ["Glm4vForConditionalGeneration"]
  else
    []
  end
end

def compact_hash(hash)
  hash.reject { |_key, value| value.nil? || (value.respond_to?(:empty?) && value.empty?) }
end

def stub_entry(model, error)
  {
    "id" => model.fetch("id"),
    "aliases" => Array(model["aliases"]),
    "provider" => model["provider"],
    "preferred_source" => model["preferred_source"],
    "mentioned_by" => Array(model["mentioned_by"]),
    "vllm_recipe_json" => model["vllm_recipe_json"],
    "sglang_pages" => Array(model["sglang_pages"]),
    "snapshot_status" => "unavailable",
    "snapshot_error" => error.to_s[0, 500],
    "commit_sha" => nil,
    "config" => {
      "model_type" => "unknown",
      "architectures" => [],
      "_builtin_fetch_status" => "unavailable",
      "_builtin_fetch_error" => error.to_s[0, 500],
      "_builtin_manifest_id" => model.fetch("id")
    },
    "siblings" => []
  }
end

def fetch_model(model)
  if model["preferred_source"] == "modelscope"
    fetch_modelscope(model)
  else
    fetch_hf(model)
  end
rescue StandardError => error
  if model["vllm_recipe_json"] || !Array(model["sglang_pages"]).empty?
    recipe_only_entry(model, error)
  else
    stub_entry(model, error)
  end
end

manifest = JSON.parse(File.read(MANIFEST_PATH))
models = manifest.fetch("models")
queue = Queue.new
models.each { |model| queue << model }

results = []
mutex = Mutex.new
threads = WORKERS.times.map do
  Thread.new do
    loop do
      model = queue.pop(true)
      entry = fetch_model(model)
      mutex.synchronize do
        results << entry
        status = entry["snapshot_status"] == "ok" ? "ok" : "stub"
        warn format("[%3d/%3d] %-4s %s", results.length, models.length, status, model["id"])
      end
    rescue ThreadError
      break
    end
  end
end
threads.each(&:join)

ordered = results.sort_by { |entry| models.index { |model| model["id"] == entry["id"] } || results.length }
catalog = {
  "catalog_version" => 1,
  "captured_at" => Time.now.utc.iso8601,
  "manifest_sources" => manifest["sources"],
  "models" => ordered
}

File.write(CATALOG_PATH, "#{JSON.pretty_generate(catalog)}\n")
ok = ordered.count { |entry| entry["snapshot_status"] == "ok" }
stub = ordered.length - ok
warn "wrote #{CATALOG_PATH}: #{ordered.length} models, #{ok} ok, #{stub} stubs"
