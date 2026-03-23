#!/usr/bin/env python3
"""
Bedrock Tool Use 对比测试: Kimi K2.5 vs Claude
================================================
场景: 独立站(DTC)卖家助手 Agent，10个tools，~10k input tokens
目标: 验证 Kimi K2.5 在 tool use 场景下 stopReason 是否正确返回 tool_use
"""

import os
import sys
import time
import json
import boto3
from openai import OpenAI

REGION = os.environ.get("AWS_REGION", "us-east-1")
KIMI_MODEL = "moonshotai.kimi-k2.5"
CLAUDE_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"
KIMI_BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws/v1"
MAX_TOKENS = 1024

# ============================================================
# API Key 生成
# ============================================================
def get_bedrock_api_key():
    try:
        from aws_bedrock_token_generator import BedrockTokenGenerator
        session = boto3.Session(region_name=REGION)
        credentials = session.get_credentials().get_frozen_credentials()
        generator = BedrockTokenGenerator()
        token = generator.get_token(credentials=credentials, region=REGION)
        print("✅ Bedrock API Key 生成成功")
        return token
    except Exception as e:
        print(f"❌ API Key 生成失败: {e}")
        sys.exit(1)

# ============================================================
# 10 个 Tools 定义 (OpenAI format)
# ============================================================
TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "list_products",
            "description": "列出店铺中的商品列表，支持按分类、状态、关键词筛选，支持分页",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "商品分类筛选，如：电子产品、服装、家居、美妆、食品、运动户外",
                        "enum": ["电子产品", "服装", "家居", "美妆", "食品", "运动户外"]
                    },
                    "status": {
                        "type": "string",
                        "description": "商品状态筛选",
                        "enum": ["active", "draft", "archived", "out_of_stock"]
                    },
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词，匹配商品名称或描述"
                    },
                    "page": {
                        "type": "integer",
                        "description": "页码，从1开始，默认1"
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "每页数量，默认20，最大100"
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "排序字段",
                        "enum": ["created_at", "updated_at", "price", "inventory", "sales"]
                    },
                    "sort_order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "排序方向，默认desc"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_detail",
            "description": "根据SKU获取商品的完整详细信息，包括名称、描述、价格、库存、图片、变体等",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "商品SKU编号，如 SKU-001"
                    }
                },
                "required": ["sku"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_product",
            "description": "创建一个新商品，需要提供名称、描述、价格、库存等基本信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "商品名称，长度2-200字符"
                    },
                    "description": {
                        "type": "string",
                        "description": "商品详细描述，支持HTML格式，建议200字以上"
                    },
                    "price": {
                        "type": "number",
                        "description": "商品售价（元），必须大于0"
                    },
                    "compare_at_price": {
                        "type": "number",
                        "description": "原价/划线价（元），用于显示折扣，必须大于price"
                    },
                    "cost_price": {
                        "type": "number",
                        "description": "成本价（元），用于利润计算"
                    },
                    "inventory": {
                        "type": "integer",
                        "description": "初始库存数量"
                    },
                    "category": {
                        "type": "string",
                        "description": "商品分类"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "商品标签列表，用于搜索和筛选"
                    },
                    "images": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "商品图片URL列表，第一张为主图"
                    },
                    "weight": {
                        "type": "number",
                        "description": "商品重量(kg)，用于运费计算"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "draft"],
                        "description": "商品状态，active直接上架，draft为草稿"
                    }
                },
                "required": ["name", "price", "inventory", "category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_product",
            "description": "更新已有商品的信息，如名称、描述、分类、标签等（不含价格和库存，请用专用接口）",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "要更新的商品SKU"
                    },
                    "name": {
                        "type": "string",
                        "description": "新的商品名称"
                    },
                    "description": {
                        "type": "string",
                        "description": "新的商品描述"
                    },
                    "category": {
                        "type": "string",
                        "description": "新的商品分类"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "新的标签列表（覆盖原有标签）"
                    },
                    "weight": {
                        "type": "number",
                        "description": "新的商品重量(kg)"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "draft", "archived"],
                        "description": "新的商品状态"
                    }
                },
                "required": ["sku"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_product",
            "description": "删除指定商品，删除后不可恢复。如有未完成订单关联该商品，将拒绝删除",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "要删除的商品SKU"
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "确认删除，必须为true才能执行"
                    }
                },
                "required": ["sku", "confirm"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_inventory",
            "description": "更新商品库存数量，支持增加、减少或直接设置库存",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "商品SKU"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "库存数量，根据operation类型含义不同"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "set"],
                        "description": "操作类型: add增加、subtract减少、set直接设置"
                    },
                    "reason": {
                        "type": "string",
                        "description": "库存变更原因，如：采购入库、盘点调整、退货入库"
                    },
                    "warehouse": {
                        "type": "string",
                        "description": "仓库标识，默认主仓库"
                    }
                },
                "required": ["sku", "quantity", "operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_product_price",
            "description": "设置商品价格，支持设置售价、原价（划线价）和成本价",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "商品SKU"
                    },
                    "price": {
                        "type": "number",
                        "description": "新售价（元）"
                    },
                    "compare_at_price": {
                        "type": "number",
                        "description": "新原价/划线价（元），设为null清除"
                    },
                    "cost_price": {
                        "type": "number",
                        "description": "新成本价（元）"
                    },
                    "schedule_start": {
                        "type": "string",
                        "description": "定时生效时间，ISO 8601格式，如2026-04-01T00:00:00+08:00"
                    },
                    "schedule_end": {
                        "type": "string",
                        "description": "定时结束时间（价格恢复原值），ISO 8601格式"
                    }
                },
                "required": ["sku", "price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_product_images",
            "description": "为商品添加图片，支持批量上传，可指定排序位置",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "商品SKU"
                    },
                    "images": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "图片URL地址"
                                },
                                "alt_text": {
                                    "type": "string",
                                    "description": "图片ALT文本，用于SEO"
                                },
                                "position": {
                                    "type": "integer",
                                    "description": "排序位置，0为主图"
                                }
                            },
                            "required": ["url"]
                        },
                        "description": "图片列表"
                    },
                    "replace_existing": {
                        "type": "boolean",
                        "description": "是否替换现有图片，默认false(追加)"
                    }
                },
                "required": ["sku", "images"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_list",
            "description": "获取订单列表，支持按日期范围、状态、金额等条件筛选",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "开始日期，格式YYYY-MM-DD"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "结束日期，格式YYYY-MM-DD"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "paid", "shipped", "delivered", "cancelled", "refunded"],
                        "description": "订单状态筛选"
                    },
                    "min_amount": {
                        "type": "number",
                        "description": "最小订单金额"
                    },
                    "max_amount": {
                        "type": "number",
                        "description": "最大订单金额"
                    },
                    "customer_email": {
                        "type": "string",
                        "description": "客户邮箱筛选"
                    },
                    "page": {
                        "type": "integer",
                        "description": "页码"
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "每页数量"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sales_analytics",
            "description": "获取销售数据分析报表，支持按时间范围和多维度聚合",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "开始日期，格式YYYY-MM-DD"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "结束日期，格式YYYY-MM-DD"
                    },
                    "granularity": {
                        "type": "string",
                        "enum": ["daily", "weekly", "monthly"],
                        "description": "时间粒度"
                    },
                    "dimensions": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["category", "product", "channel", "region"]
                        },
                        "description": "分析维度"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["revenue", "orders", "units_sold", "avg_order_value", "conversion_rate", "refund_rate"]
                        },
                        "description": "指标列表"
                    }
                },
                "required": ["start_date", "end_date"]
            }
        }
    }
]


# ============================================================
# Tools - Bedrock Converse format (for Claude)
# ============================================================
def convert_to_converse_tools(openai_tools):
    """Convert OpenAI function tools to Bedrock Converse toolConfig format"""
    converse_tools = []
    for t in openai_tools:
        func = t["function"]
        tool_spec = {
            "toolSpec": {
                "name": func["name"],
                "description": func["description"],
                "inputSchema": {
                    "json": func["parameters"]
                }
            }
        }
        converse_tools.append(tool_spec)
    return {"tools": converse_tools}

TOOLS_CONVERSE = convert_to_converse_tools(TOOLS_OPENAI)

# ============================================================
# System Prompt - 独立站卖家助手 + 商品目录 (~10k tokens)
# ============================================================
SYSTEM_PROMPT = """# 独立站卖家助手 - ShopMaster AI

你是 ShopMaster AI，一个专业的独立站（DTC电商）卖家助手。你的职责是帮助卖家高效管理店铺运营，包括商品管理、库存管理、订单查询和数据分析。

## 你的能力
1. **商品管理**: 创建、查看、更新、删除商品信息
2. **库存管理**: 实时更新库存数量，支持入库、出库、盘点
3. **价格管理**: 设置商品售价、促销价、成本价
4. **图片管理**: 添加和管理商品图片
5. **订单查询**: 查看订单列表和详情
6. **数据分析**: 销售额、订单量、转化率等多维度分析

## 操作规范
- 在执行任何修改操作前，先确认用户的意图
- 删除商品需要用户明确确认
- 价格修改时注意检查是否合理（不应低于成本价）
- 库存操作需要记录变更原因
- 批量操作时逐一确认

## 店铺信息
- 店铺名称: TechStyle 数码生活馆
- 主营品类: 电子产品、智能家居、时尚配件
- 目标市场: 中国大陆、东南亚
- 币种: 人民币 (CNY)
- 仓库: 深圳主仓、义乌分仓

## 分类体系
1. **电子产品** - 手机配件、电脑周边、音频设备、充电设备、智能穿戴
2. **智能家居** - 智能灯具、智能安防、智能清洁、环境监测、智能控制
3. **时尚配件** - 手机壳、数码包袋、桌面摆件、创意礼品
4. **运动户外** - 运动耳机、运动手环、户外装备、骑行配件
5. **生活日用** - 收纳整理、厨房小工具、个人护理、旅行用品

## 当前商品目录

### 电子产品
| SKU | 商品名称 | 售价(元) | 原价(元) | 库存 | 状态 | 月销量 |
|-----|---------|---------|---------|------|------|--------|
| SKU-001 | 65W氮化镓快充充电器（三口） | 99.00 | 159.00 | 2340 | active | 1856 |
| SKU-002 | USB-C转HDMI 4K拓展坞 | 189.00 | 299.00 | 567 | active | 423 |
| SKU-003 | 无线机械键盘RGB版 | 329.00 | 459.00 | 189 | active | 267 |
| SKU-004 | 2TB移动固态硬盘 USB3.2 | 599.00 | 799.00 | 445 | active | 312 |
| SKU-005 | 主动降噪蓝牙耳机 ANC Pro | 459.00 | 699.00 | 723 | active | 534 |
| SKU-006 | 电竞鼠标 16000DPI 轻量化 | 199.00 | 299.00 | 1120 | active | 689 |
| SKU-007 | 磁吸无线充电板 15W | 79.00 | 129.00 | 3200 | active | 2100 |

### 智能家居
| SKU | 商品名称 | 售价(元) | 原价(元) | 库存 | 状态 | 月销量 |
|-----|---------|---------|---------|------|------|--------|
| SKU-008 | WiFi智能LED灯泡 RGBW 可调色温 | 39.00 | 69.00 | 5600 | active | 3400 |
| SKU-009 | 智能门铃摄像头 1080P 双向语音 | 299.00 | 499.00 | 234 | active | 178 |
| SKU-010 | 扫地机器人 激光导航 自动集尘 | 1999.00 | 2999.00 | 89 | active | 67 |
| SKU-011 | 智能温湿度传感器 WiFi版 | 49.00 | 79.00 | 4300 | active | 1890 |
| SKU-012 | WiFi智能插座 远程控制 定时开关 | 29.00 | 49.00 | 8900 | active | 5600 |
| SKU-013 | 空气质量检测仪 PM2.5/甲醛/TVOC | 259.00 | 399.00 | 345 | active | 198 |

### 时尚配件
| SKU | 商品名称 | 售价(元) | 原价(元) | 库存 | 状态 | 月销量 |
|-----|---------|---------|---------|------|------|--------|
| SKU-014 | iPhone 16 Pro 液态硅胶手机壳 | 49.00 | 89.00 | 6700 | active | 4500 |
| SKU-015 | 多功能数码收纳包 防水面料 | 69.00 | 119.00 | 1200 | active | 780 |
| SKU-016 | 实木桌面显示器增高架 带收纳 | 159.00 | 249.00 | 456 | active | 234 |
| SKU-017 | 创意宇航员手机支架 | 39.00 | 69.00 | 3400 | active | 2100 |
| SKU-018 | LED氛围灯 USB供电 16色遥控 | 59.00 | 99.00 | 2800 | active | 1670 |

### 运动户外
| SKU | 商品名称 | 售价(元) | 原价(元) | 库存 | 状态 | 月销量 |
|-----|---------|---------|---------|------|------|--------|
| SKU-019 | 骨传导运动蓝牙耳机 IPX8防水 | 359.00 | 549.00 | 567 | active | 345 |
| SKU-020 | 智能运动手环 心率血氧监测 | 149.00 | 249.00 | 2300 | active | 1560 |
| SKU-021 | 户外便携蓝牙音箱 20W IPX7 | 199.00 | 329.00 | 890 | active | 456 |
| SKU-022 | 自行车尾灯 智能刹车感应 USB充电 | 59.00 | 99.00 | 1800 | active | 920 |
| SKU-023 | 运动腰包 弹力防水 6.7寸手机适配 | 39.00 | 69.00 | 4500 | active | 2800 |

### 生活日用
| SKU | 商品名称 | 售价(元) | 原价(元) | 库存 | 状态 | 月销量 |
|-----|---------|---------|---------|------|------|--------|
| SKU-024 | 桌面线材收纳盒 大容量 散热孔设计 | 45.00 | 79.00 | 3200 | active | 1900 |
| SKU-025 | 便携榨汁杯 无线充电 350ml | 89.00 | 149.00 | 1560 | active | 980 |
| SKU-026 | 旅行分装瓶套装 硅胶材质 过安检 | 29.00 | 49.00 | 7800 | active | 4200 |
| SKU-027 | 电动牙刷 声波震动 5种模式 IPX7 | 129.00 | 199.00 | 2100 | active | 1340 |
| SKU-028 | 多功能厨房电子秤 0.1g精度 | 35.00 | 59.00 | 4500 | active | 2670 |
| SKU-029 | 记忆棉护颈U型枕 磁吸扣设计 | 69.00 | 109.00 | 1800 | active | 1120 |
| SKU-030 | USB充电暖手宝 双面发热 10000mAh | 59.00 | 99.00 | 890 | draft | 0 |

## 近期运营数据概览
- 本月GMV: ¥2,847,560（同比+23%）
- 本月订单数: 18,934单
- 平均客单价: ¥150.4
- 退货率: 3.2%
- 复购率: 28%
- 热销TOP3: SKU-012(WiFi智能插座), SKU-014(手机壳), SKU-008(智能灯泡)
- 库存预警: SKU-010(扫地机器人)库存<100，SKU-009(智能门铃)库存<250

## 促销日历
- 3月25日-31日: 春季焕新季，全场满200减30
- 4月1日-7日: 清明小长假，户外品类专场8折
- 4月15日-20日: 会员日，会员专享额外9折

## 物流合作方
- 国内: 顺丰速运(默认)、中通快递(经济)、京东物流(大件)
- 跨境: 云途物流、燕文物流、4PX递四方
- 深圳主仓发货时效: 当日15:00前下单当日发出
- 义乌分仓发货时效: 当日12:00前下单当日发出

## 注意事项
1. 所有价格修改操作请先与用户确认，避免误操作
2. 库存变更需要记录原因，便于后续审计
3. 商品删除操作不可逆，需要用户二次确认
4. 批量操作建议先预览再执行
5. 涉及促销价格的修改，注意不要与当前促销活动冲突
6. 跨境商品需要注意合规信息（品名、材质、电池等申报要素）
"""


# ============================================================
# 测试用例
# ============================================================
TEST_CASES = [
    {
        "name": "list_products",
        "user_message": "帮我查看一下店铺里所有的商品",
        "expected_tool": "list_products"
    },
    {
        "name": "set_price",
        "user_message": "把 SKU-001 的价格从 99 改成 79",
        "expected_tool": "set_product_price"
    },
    {
        "name": "create_product",
        "user_message": "创建一个新商品：无线蓝牙耳机，价格 199，库存 500，分类是电子产品",
        "expected_tool": "create_product"
    },
    {
        "name": "update_inventory",
        "user_message": "SKU-015 的库存补充 200 个",
        "expected_tool": "update_inventory"
    },
    {
        "name": "sales_analytics",
        "user_message": "看看上周的销售数据，按品类维度分析一下",
        "expected_tool": "get_sales_analytics"
    }
]


# ============================================================
# 测试 Kimi K2.5 (OpenAI compatible API)
# ============================================================
def test_kimi(client, test_case):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": test_case["user_message"]}
    ]
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=KIMI_MODEL,
            messages=messages,
            tools=TOOLS_OPENAI,
            tool_choice="auto",
            max_tokens=MAX_TOKENS,
        )
        elapsed = time.time() - start
        choice = response.choices[0]

        # Extract tool calls if any
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments[:200]
                })

        content = choice.message.content or ""
        return {
            "model": "Kimi K2.5",
            "test": test_case["name"],
            "finish_reason": choice.finish_reason,
            "has_tool_use": len(tool_calls) > 0,
            "tool_calls": tool_calls,
            "content_preview": content[:200],
            "elapsed_sec": round(elapsed, 3),
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "error": None
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "model": "Kimi K2.5",
            "test": test_case["name"],
            "finish_reason": "ERROR",
            "has_tool_use": False,
            "tool_calls": [],
            "content_preview": "",
            "elapsed_sec": round(elapsed, 3),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": str(e)
        }


# ============================================================
# 测试 Claude (Bedrock Converse API)
# ============================================================
def test_claude(bedrock_client, test_case):
    messages = [
        {"role": "user", "content": [{"text": test_case["user_message"]}]}
    ]
    system_prompt = [{"text": SYSTEM_PROMPT}]

    start = time.time()
    try:
        response = bedrock_client.converse(
            modelId=CLAUDE_MODEL,
            messages=messages,
            system=system_prompt,
            toolConfig=TOOLS_CONVERSE,
            inferenceConfig={"maxTokens": MAX_TOKENS}
        )
        elapsed = time.time() - start

        stop_reason = response.get("stopReason", "unknown")
        output = response.get("output", {}).get("message", {})
        content_blocks = output.get("content", [])

        tool_calls = []
        text_content = ""
        for block in content_blocks:
            if "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append({
                    "name": tu["name"],
                    "arguments": json.dumps(tu.get("input", {}), ensure_ascii=False)[:200]
                })
            if "text" in block:
                text_content += block["text"]

        usage = response.get("usage", {})
        return {
            "model": "Claude Sonnet 4",
            "test": test_case["name"],
            "finish_reason": stop_reason,
            "has_tool_use": len(tool_calls) > 0,
            "tool_calls": tool_calls,
            "content_preview": text_content[:200],
            "elapsed_sec": round(elapsed, 3),
            "prompt_tokens": usage.get("inputTokens", 0),
            "completion_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("inputTokens", 0) + usage.get("outputTokens", 0),
            "error": None
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "model": "Claude Sonnet 4",
            "test": test_case["name"],
            "finish_reason": "ERROR",
            "has_tool_use": False,
            "tool_calls": [],
            "content_preview": "",
            "elapsed_sec": round(elapsed, 3),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": str(e)
        }


# ============================================================
# 结果展示
# ============================================================
def print_comparison_table(results):
    print("\n" + "=" * 100)
    print("📊 Tool Use 对比测试结果")
    print("=" * 100)

    # Group by test case
    tests = {}
    for r in results:
        key = r["test"]
        if key not in tests:
            tests[key] = {}
        tests[key][r["model"]] = r

    header = f"{'测试场景':<18} {'模型':<16} {'stop_reason':<14} {'tool_use?':<10} {'调用的tool':<22} {'耗时(s)':<10} {'input_tk':<10} {'output_tk':<10}"
    print(header)
    print("-" * 100)

    for test_name, models in tests.items():
        for model_name, r in models.items():
            tool_name = r["tool_calls"][0]["name"] if r["tool_calls"] else "-"
            has_tool = "✅" if r["has_tool_use"] else "❌"
            error_mark = " ⚠️" if r["error"] else ""
            print(f"{r['test']:<18} {r['model']:<16} {r['finish_reason']:<14} {has_tool:<10} {tool_name:<22} {r['elapsed_sec']:<10} {r['prompt_tokens']:<10} {r['completion_tokens']:<10}{error_mark}")
        print("-" * 100)

    # Summary
    kimi_results = [r for r in results if r["model"] == "Kimi K2.5"]
    claude_results = [r for r in results if r["model"] == "Claude Sonnet 4"]

    kimi_tool_count = sum(1 for r in kimi_results if r["has_tool_use"])
    claude_tool_count = sum(1 for r in claude_results if r["has_tool_use"])

    print(f"\n📈 汇总:")
    print(f"  Kimi K2.5:      {kimi_tool_count}/{len(kimi_results)} 次正确返回 tool_use")
    print(f"  Claude Sonnet 4: {claude_tool_count}/{len(claude_results)} 次正确返回 tool_use")

    kimi_errors = [r for r in kimi_results if r["error"]]
    claude_errors = [r for r in claude_results if r["error"]]
    if kimi_errors:
        print(f"\n  ⚠️  Kimi 错误: {len(kimi_errors)} 次")
        for e in kimi_errors:
            print(f"      - {e['test']}: {e['error'][:100]}")
    if claude_errors:
        print(f"\n  ⚠️  Claude 错误: {len(claude_errors)} 次")
        for e in claude_errors:
            print(f"      - {e['test']}: {e['error'][:100]}")

    # Print content when Kimi didn't return tool_use
    kimi_no_tool = [r for r in kimi_results if not r["has_tool_use"] and not r["error"]]
    if kimi_no_tool:
        print(f"\n🔍 Kimi 未返回 tool_use 的用例详情:")
        for r in kimi_no_tool:
            print(f"  [{r['test']}] finish_reason={r['finish_reason']}")
            print(f"    content: '{r['content_preview'][:150]}'")


# ============================================================
# Main
# ============================================================
def main():
    print("🚀 Bedrock Tool Use 对比测试: Kimi K2.5 vs Claude Sonnet 4")
    print(f"   Region: {REGION}")
    print(f"   Kimi Model: {KIMI_MODEL}")
    print(f"   Claude Model: {CLAUDE_MODEL}")
    print(f"   Max Tokens: {MAX_TOKENS}")

    # Init clients
    print("\n📡 初始化客户端...")
    api_key = get_bedrock_api_key()
    kimi_client = OpenAI(api_key=api_key, base_url=KIMI_BASE_URL)
    bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)

    # Verify connectivity
    print("\n🔗 验证 Kimi K2.5 连通性...")
    try:
        resp = kimi_client.chat.completions.create(
            model=KIMI_MODEL,
            messages=[{"role": "user", "content": "说 hello"}],
            max_tokens=10
        )
        print(f"   ✅ Kimi OK: {resp.choices[0].message.content}")
    except Exception as e:
        print(f"   ❌ Kimi 连接失败: {e}")

    print("\n🔗 验证 Claude 连通性...")
    try:
        resp = bedrock_client.converse(
            modelId=CLAUDE_MODEL,
            messages=[{"role": "user", "content": [{"text": "说 hello"}]}],
            inferenceConfig={"maxTokens": 10}
        )
        text = resp["output"]["message"]["content"][0]["text"]
        print(f"   ✅ Claude OK: {text}")
    except Exception as e:
        print(f"   ❌ Claude 连接失败: {e}")

    # Run tests
    all_results = []
    for i, tc in enumerate(TEST_CASES):
        print(f"\n{'='*60}")
        print(f"测试 {i+1}/{len(TEST_CASES)}: {tc['name']}")
        print(f"用户消息: {tc['user_message']}")
        print(f"期望 tool: {tc['expected_tool']}")
        print(f"{'='*60}")

        # Test Kimi
        print(f"\n  🌙 测试 Kimi K2.5...")
        kimi_result = test_kimi(kimi_client, tc)
        kimi_result["expected_tool"] = tc["expected_tool"]
        all_results.append(kimi_result)
        print(f"     finish_reason: {kimi_result['finish_reason']}")
        print(f"     has_tool_use: {kimi_result['has_tool_use']}")
        if kimi_result["tool_calls"]:
            for t in kimi_result["tool_calls"]:
                print(f"     tool: {t['name']}")
        if kimi_result["content_preview"]:
            print(f"     content: {kimi_result['content_preview'][:100]}")
        if kimi_result["error"]:
            print(f"     ❌ error: {kimi_result['error'][:200]}")

        time.sleep(1)

        # Test Claude
        print(f"\n  🤖 测试 Claude Sonnet 4...")
        claude_result = test_claude(bedrock_client, tc)
        claude_result["expected_tool"] = tc["expected_tool"]
        all_results.append(claude_result)
        print(f"     finish_reason: {claude_result['finish_reason']}")
        print(f"     has_tool_use: {claude_result['has_tool_use']}")
        if claude_result["tool_calls"]:
            for t in claude_result["tool_calls"]:
                print(f"     tool: {t['name']}")
        if claude_result["content_preview"]:
            print(f"     content: {claude_result['content_preview'][:100]}")
        if claude_result["error"]:
            print(f"     ❌ error: {claude_result['error'][:200]}")

        time.sleep(1)

    # Print comparison
    print_comparison_table(all_results)

    # Save results
    output = {
        "config": {
            "region": REGION,
            "kimi_model": KIMI_MODEL,
            "claude_model": CLAUDE_MODEL,
            "max_tokens": MAX_TOKENS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "system_prompt_length": len(SYSTEM_PROMPT),
        },
        "results": all_results
    }
    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bedrock_tool_use_results.json")
    with open(outfile, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 完整结果已保存: {outfile}")


if __name__ == "__main__":
    main()
