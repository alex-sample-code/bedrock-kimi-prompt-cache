# Bedrock Mantle API - Kimi K2.5 Prompt Cache Test

Test prompt caching (vLLM prefix caching) on AWS Bedrock Mantle API with Moonshot AI's Kimi K2.5 model.

## Background

AWS Bedrock Mantle uses vLLM as the underlying inference engine, which supports **Automatic Prefix Caching (APC)**. The `cache_salt` parameter (a vLLM feature for multi-tenant prefix cache isolation) can be passed via the OpenAI SDK's `extra_body` to enable/control prefix caching behavior.

## Key Findings

| Finding | Detail |
|---|---|
| **cache_salt works** | `prompt_tokens_details.cached_tokens` returns non-zero values when cache hits |
| **Hit rate is unstable** | ~33-60% hit rate in testing |
| **No significant latency improvement** | Cache hit vs miss shows similar response times |
| **Root cause** | Bedrock Mantle is serverless — requests may route to different vLLM instances, and prefix cache is instance-level (in-memory KV cache), not globally shared |

## Prerequisites

- Python 3.8+
- AWS credentials configured (`aws configure`)
- Kimi K2.5 model access enabled in your Bedrock region

## Install Dependencies

```bash
pip install openai aws-bedrock-token-generator boto3
```

## Usage

```bash
# Default: us-east-1
python bedrock_kimi_prompt_cache_test.py

# Specify region
AWS_REGION=us-west-2 python bedrock_kimi_prompt_cache_test.py
```

## How It Works

1. Generates a short-term Bedrock API key using `aws-bedrock-token-generator`
2. Connects to Bedrock Mantle endpoint (`https://bedrock-mantle.{region}.api.aws/v1`) via OpenAI SDK
3. Runs 3 tests:
   - **Test 1**: Baseline — multiple requests without `cache_salt`
   - **Test 2**: With `cache_salt` — same system prompt, different user questions
   - **Test 3**: Rapid-fire — 5 identical requests to measure cache stability

## How to Enable Prompt Cache

```python
from openai import OpenAI

client = OpenAI(api_key=bedrock_api_key, base_url="https://bedrock-mantle.us-east-1.api.aws/v1")

response = client.chat.completions.create(
    model="moonshotai.kimi-k2.5",
    messages=messages,
    extra_body={"cache_salt": "default-tenant"},
)

# Check cache hit
if response.usage.prompt_tokens_details:
    print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens}")
```

## Sample Output

```
prompt_tokens_details: PromptTokensDetails(audio_tokens=0, cached_tokens=512)
```

When `cached_tokens > 0`, prefix cache was hit successfully.

## References

- [Bedrock Mantle Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html)
- [Kimi K2.5 Model Card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-moonshot-ai-kimi-k2-5.html)
- [vLLM Cache Salting RFC](https://github.com/vllm-project/vllm/issues/16016)

## License

MIT
