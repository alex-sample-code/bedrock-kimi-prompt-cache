#!/usr/bin/env python3
"""
Bedrock Mantle API - Kimi K2.5 Prompt Cache 测试 Demo
=====================================================

目标：通过 Bedrock Mantle (OpenAI compatible) API 调用 Kimi K2.5，
     验证 prompt cache (vLLM prefix caching) 功能，使用 cache_salt 参数。

背景：
- Bedrock Mantle 底层使用 vLLM 作为推理引擎
- vLLM 支持 Automatic Prefix Caching (APC)
- cache_salt 是 vLLM 的参数，用于多租户间 prefix cache 隔离
- 通过 OpenAI SDK 的 extra_body 传递 cache_salt

测试方法：
1. 构造一段较长的共享 system prompt（确保超过 prefix cache 最小 token 阈值）
2. 发送多轮请求，前缀相同，仅改变最后的 user 问题
3. 观察响应时间差异（cache hit 时应该更快）和 usage 中的 cache 相关字段
"""

import os
import sys
import time
import json
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = "moonshotai.kimi-k2.5"
BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws/v1"
CACHE_SALT = "default-tenant"  # vLLM prefix cache 隔离标识

# ============================================================
# 生成 Bedrock API Key (短期 token)
# ============================================================
def get_bedrock_api_key():
    """使用 aws-bedrock-token-generator 生成短期 API key"""
    try:
        from aws_bedrock_token_generator import BedrockTokenGenerator
        import boto3

        session = boto3.Session(region_name=REGION)
        credentials = session.get_credentials().get_frozen_credentials()
        generator = BedrockTokenGenerator()
        token = generator.get_token(credentials=credentials, region=REGION)
        print(f"✅ 成功生成 Bedrock API Key (短期 token)")
        return token
    except Exception as e:
        print(f"❌ 生成 API Key 失败: {e}")
        print("   请确保已配置 AWS credentials (aws configure)")
        sys.exit(1)


# ============================================================
# 构造测试数据
# ============================================================

# 较长的 system prompt，确保有足够 token 触发 prefix caching
# vLLM prefix caching 按 block（通常 16 tokens）匹配，prompt 越长 cache 收益越大
LONG_SYSTEM_PROMPT = """你是一位资深的 AWS 架构师助手。你精通以下 AWS 服务，并能为客户提供最佳实践建议：

1. 计算服务：EC2, Lambda, ECS, EKS, Fargate, Batch, App Runner
2. 存储服务：S3, EBS, EFS, FSx, Storage Gateway, Glacier
3. 数据库服务：RDS, Aurora, DynamoDB, ElastiCache, Neptune, Redshift, MemoryDB
4. 网络服务：VPC, ELB, CloudFront, Route 53, API Gateway, Direct Connect, Transit Gateway, PrivateLink
5. 安全服务：IAM, KMS, Secrets Manager, WAF, Shield, GuardDuty, Security Hub, Macie
6. AI/ML 服务：SageMaker, Bedrock, Rekognition, Comprehend, Textract, Transcribe, Polly, Lex
7. 容器与编排：ECR, ECS, EKS, App Mesh, Cloud Map
8. 监控与运维：CloudWatch, CloudTrail, Config, Systems Manager, X-Ray, EventBridge
9. 数据分析：Athena, Glue, EMR, Kinesis, QuickSight, Lake Formation, OpenSearch
10. 开发者工具：CodeCommit, CodeBuild, CodeDeploy, CodePipeline, CDK, CloudFormation, SAM

你需要根据客户的具体需求，提供详细的架构方案，包括：
- 服务选型理由
- 架构图描述
- 成本估算
- 安全最佳实践
- 高可用和容灾设计
- 性能优化建议

请用中文回答，语言简洁专业。对于每个建议，请说明适用场景和潜在风险。
在回答复杂问题时，先给出总体方案概览，再分模块详细说明。

以下是一些额外的上下文信息，帮助你更好地理解客户场景：

AWS Well-Architected Framework 六大支柱：
1. 卓越运营（Operational Excellence）：运行和监控系统，持续改进流程和程序
2. 安全性（Security）：保护信息和系统，风险评估和缓解策略
3. 可靠性（Reliability）：确保工作负载正确一致地执行预期功能
4. 性能效率（Performance Efficiency）：高效使用计算资源满足系统需求
5. 成本优化（Cost Optimization）：以最低价格运行系统交付业务价值
6. 可持续性（Sustainability）：最大限度减少工作负载的环境影响

请在你的每个建议中都考虑这六个支柱的影响。
"""

# 不同的用户问题，共享相同的 system prompt（前缀相同）
USER_QUESTIONS = [
    "我需要设计一个电商网站的架构，日均 UV 100万，峰值 QPS 5000。请给出方案概览。",
    "针对上述电商网站，数据库层应该如何设计？读写分离、缓存策略、分库分表方案？",
    "这个电商架构的安全层面需要考虑什么？WAF 规则、DDoS 防护、数据加密方案？",
]


# ============================================================
# 测试函数
# ============================================================

def test_without_cache_salt(client):
    """测试 1：不使用 cache_salt（基准测试）"""
    print("\n" + "=" * 70)
    print("测试 1：无 cache_salt（基准）")
    print("=" * 70)

    results = []
    for i, question in enumerate(USER_QUESTIONS):
        messages = [
            {"role": "system", "content": LONG_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        start = time.time()
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=200,
            )
            elapsed = time.time() - start

            usage = response.usage
            result = {
                "question": i + 1,
                "elapsed_sec": round(elapsed, 3),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                # 检查是否有 cache 相关字段
                "usage_raw": str(usage),
            }
            results.append(result)

            print(f"\n  问题 {i+1}: {question[:40]}...")
            print(f"  ⏱️  响应时间: {elapsed:.3f}s")
            print(f"  📊 Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}")
            print(f"  📋 Usage 完整: {usage}")

            # 检查 usage 中是否有 cache 相关的额外字段
            if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                print(f"  🔍 Prompt Token Details: {usage.prompt_tokens_details}")
            if hasattr(usage, 'cache_creation_input_tokens'):
                print(f"  💾 Cache Creation Tokens: {usage.cache_creation_input_tokens}")
            if hasattr(usage, 'cache_read_input_tokens'):
                print(f"  📖 Cache Read Tokens: {usage.cache_read_input_tokens}")

        except Exception as e:
            elapsed = time.time() - start
            print(f"\n  问题 {i+1}: ❌ 错误 - {e}")
            results.append({"question": i + 1, "error": str(e), "elapsed_sec": round(elapsed, 3)})

        time.sleep(1)  # 短暂等待避免限流

    return results


def test_with_cache_salt(client, salt=CACHE_SALT):
    """测试 2：使用 cache_salt（验证 prefix cache）"""
    print("\n" + "=" * 70)
    print(f"测试 2：使用 cache_salt=\"{salt}\"")
    print("=" * 70)

    results = []
    for i, question in enumerate(USER_QUESTIONS):
        messages = [
            {"role": "system", "content": LONG_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        start = time.time()
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=200,
                extra_body={"cache_salt": salt},
            )
            elapsed = time.time() - start

            usage = response.usage
            result = {
                "question": i + 1,
                "elapsed_sec": round(elapsed, 3),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cache_salt": salt,
                "usage_raw": str(usage),
            }
            results.append(result)

            print(f"\n  问题 {i+1}: {question[:40]}...")
            print(f"  ⏱️  响应时间: {elapsed:.3f}s")
            print(f"  📊 Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}")
            print(f"  📋 Usage 完整: {usage}")

            if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                print(f"  🔍 Prompt Token Details: {usage.prompt_tokens_details}")
            if hasattr(usage, 'cache_creation_input_tokens'):
                print(f"  💾 Cache Creation Tokens: {usage.cache_creation_input_tokens}")
            if hasattr(usage, 'cache_read_input_tokens'):
                print(f"  📖 Cache Read Tokens: {usage.cache_read_input_tokens}")

        except Exception as e:
            elapsed = time.time() - start
            print(f"\n  问题 {i+1}: ❌ 错误 - {e}")
            results.append({"question": i + 1, "error": str(e), "elapsed_sec": round(elapsed, 3)})

        time.sleep(1)

    return results


def test_rapid_fire_same_prompt(client, salt=CACHE_SALT, rounds=5):
    """测试 3：快速连续发送完全相同的请求，观察 cache 命中稳定性"""
    print("\n" + "=" * 70)
    print(f"测试 3：连续 {rounds} 次完全相同请求（cache_salt=\"{salt}\"）")
    print("=" * 70)

    messages = [
        {"role": "system", "content": LONG_SYSTEM_PROMPT},
        {"role": "user", "content": "简要说明 EKS 和 ECS 的区别，各适用什么场景？"},
    ]

    results = []
    for i in range(rounds):
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=150,
                extra_body={"cache_salt": salt},
            )
            elapsed = time.time() - start
            usage = response.usage

            print(f"\n  第 {i+1}/{rounds} 次:")
            print(f"  ⏱️  响应时间: {elapsed:.3f}s")
            print(f"  📊 Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}")
            print(f"  📋 Usage: {usage}")

            if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                print(f"  🔍 Details: {usage.prompt_tokens_details}")

            results.append({
                "round": i + 1,
                "elapsed_sec": round(elapsed, 3),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "usage_raw": str(usage),
            })

        except Exception as e:
            elapsed = time.time() - start
            print(f"\n  第 {i+1}/{rounds} 次: ❌ {e}")
            results.append({"round": i + 1, "error": str(e), "elapsed_sec": round(elapsed, 3)})

        time.sleep(0.5)  # 短间隔，给 cache 生效时间

    return results


def print_summary(no_cache_results, cache_results, rapid_results):
    """打印测试结果摘要"""
    print("\n" + "=" * 70)
    print("📊 测试结果摘要")
    print("=" * 70)

    def avg_time(results):
        times = [r["elapsed_sec"] for r in results if "error" not in r]
        return sum(times) / len(times) if times else 0

    print(f"\n  无 cache_salt 平均响应时间: {avg_time(no_cache_results):.3f}s")
    print(f"  有 cache_salt 平均响应时间: {avg_time(cache_results):.3f}s")

    if rapid_results:
        rapid_times = [r["elapsed_sec"] for r in rapid_results if "error" not in r]
        if len(rapid_times) >= 2:
            print(f"\n  连续请求 - 第1次: {rapid_times[0]:.3f}s")
            print(f"  连续请求 - 后续平均: {sum(rapid_times[1:]) / len(rapid_times[1:]):.3f}s")
            speedup = rapid_times[0] / (sum(rapid_times[1:]) / len(rapid_times[1:])) if rapid_times[1:] else 0
            if speedup > 1.1:
                print(f"  🚀 后续请求加速比: {speedup:.2f}x（可能存在 cache 命中）")
            else:
                print(f"  ⚠️  加速比: {speedup:.2f}x（cache 效果不明显）")

    print("\n" + "=" * 70)
    print("📝 分析结论")
    print("=" * 70)
    print("""
  1. 如果 usage 中出现 prompt_tokens_details.cached_tokens > 0，说明 cache 生效
  2. 如果后续请求响应时间明显低于首次，说明 prefix cache 可能在工作
  3. cache_salt 参数是 vLLM prefix cache 的隔离机制：
     - 相同 salt → 共享 cache
     - 不同 salt → cache 隔离
  4. Kimi K2.5 在 Bedrock 上的 cache 命中率可能不稳定（已知问题）
     因为 Bedrock Mantle 是 serverless，底层可能路由到不同实例
""")


# ============================================================
# 主入口
# ============================================================
def main():
    print("🚀 Bedrock Mantle API - Kimi K2.5 Prompt Cache 测试")
    print(f"   Region: {REGION}")
    print(f"   Model:  {MODEL_ID}")
    print(f"   Base URL: {BASE_URL}")

    # 生成 API Key
    api_key = get_bedrock_api_key()

    # 创建 OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )

    # 先验证连通性
    print("\n🔗 验证 API 连通性...")
    try:
        test_resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "说 'hello'"}],
            max_tokens=10,
        )
        print(f"   ✅ 连通! 响应: {test_resp.choices[0].message.content}")
    except Exception as e:
        print(f"   ❌ 连接失败: {e}")
        print("   请检查：1) AWS credentials 2) 模型是否已在该 region 可用 3) API Key 是否有效")
        sys.exit(1)

    # 运行测试
    no_cache = test_without_cache_salt(client)
    with_cache = test_with_cache_salt(client)
    rapid = test_rapid_fire_same_prompt(client)

    # 输出摘要
    print_summary(no_cache, with_cache, rapid)

    # 保存完整结果到文件
    output = {
        "config": {
            "region": REGION,
            "model": MODEL_ID,
            "base_url": BASE_URL,
            "cache_salt": CACHE_SALT,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
        "test_1_no_cache": no_cache,
        "test_2_with_cache": with_cache,
        "test_3_rapid_fire": rapid,
    }
    outfile = os.path.join(os.path.dirname(__file__), "bedrock_kimi_cache_results.json")
    with open(outfile, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 完整结果已保存: {outfile}")


if __name__ == "__main__":
    main()
