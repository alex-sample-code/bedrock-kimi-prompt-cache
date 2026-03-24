#!/usr/bin/env python3
"""
复现 Kimi K2.5 Tool Use 失败: end_turn + 空 content
=====================================================
目标: input tokens ~10k, 多轮对话, 复杂指令, 复现 stopReason=end_turn 且 content 为空
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

def get_bedrock_api_key():
    from aws_bedrock_token_generator import BedrockTokenGenerator
    session = boto3.Session(region_name=REGION)
    credentials = session.get_credentials().get_frozen_credentials()
    generator = BedrockTokenGenerator()
    return generator.get_token(credentials=credentials, region=REGION)


# ============================================================
# 10 个 Tools (OpenAI format) - 更详细的参数定义，增加 token 消耗
# ============================================================
TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "list_products",
            "description": "列出店铺中的商品列表。支持多种筛选条件组合查询，包括按分类、状态、价格区间、库存范围、上架时间、标签等维度筛选。返回结果包含商品基本信息列表和分页元数据。注意：默认按更新时间倒序排列，每页最多返回50条记录。如果需要导出全量商品数据，请使用 export_products 接口。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "商品分类筛选。支持的分类包括：电子产品(electronics)、智能家居(smart_home)、时尚配件(fashion)、运动户外(sports)、生活日用(daily)、母婴用品(baby)、食品保健(food)、图书文具(books)。可传中文或英文标识。", "enum": ["电子产品", "智能家居", "时尚配件", "运动户外", "生活日用", "母婴用品", "食品保健", "图书文具"]},
                    "status": {"type": "string", "description": "商品状态筛选。active=已上架在售, draft=草稿未发布, archived=已下架归档, out_of_stock=售罄(库存为0自动标记), pending_review=审核中, rejected=审核驳回", "enum": ["active", "draft", "archived", "out_of_stock", "pending_review", "rejected"]},
                    "keyword": {"type": "string", "description": "搜索关键词，支持模糊匹配商品名称、SKU编号、商品描述、标签。使用空格分隔多个关键词表示AND关系。"},
                    "min_price": {"type": "number", "description": "最低价格筛选（含），单位：元"},
                    "max_price": {"type": "number", "description": "最高价格筛选（含），单位：元"},
                    "min_inventory": {"type": "integer", "description": "最低库存数量筛选"},
                    "max_inventory": {"type": "integer", "description": "最高库存数量筛选"},
                    "created_after": {"type": "string", "description": "创建时间起始筛选，ISO 8601格式"},
                    "created_before": {"type": "string", "description": "创建时间截止筛选，ISO 8601格式"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "标签筛选，返回包含任一指定标签的商品"},
                    "has_images": {"type": "boolean", "description": "是否有图片筛选"},
                    "page": {"type": "integer", "description": "页码，从1开始，默认1"},
                    "page_size": {"type": "integer", "description": "每页数量，默认20，最大50"},
                    "sort_by": {"type": "string", "enum": ["created_at", "updated_at", "price", "inventory", "sales", "name", "views"], "description": "排序字段"},
                    "sort_order": {"type": "string", "enum": ["asc", "desc"], "description": "排序方向，默认desc"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_detail",
            "description": "根据SKU编号获取商品的完整详细信息。返回数据包括：基本信息(名称/描述/分类/标签)、价格信息(售价/原价/成本价/历史价格)、库存信息(各仓库库存/安全库存/预留库存)、媒体资源(图片/视频URL)、变体信息(颜色/尺码/规格)、SEO信息(meta标题/描述)、物流信息(重量/尺寸/运费模板)、统计数据(浏览量/收藏数/转化率)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "商品SKU编号，格式为SKU-XXX（如SKU-001）。也支持传入商品ID(纯数字)进行查询。"},
                    "include_variants": {"type": "boolean", "description": "是否包含变体信息，默认true"},
                    "include_stats": {"type": "boolean", "description": "是否包含统计数据（浏览量、转化率等），默认true"},
                    "include_history": {"type": "boolean", "description": "是否包含价格变更历史，默认false"}
                },
                "required": ["sku"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_product",
            "description": "创建一个新商品并添加到店铺。必填字段为名称、价格、库存数量和分类。其他字段如描述、图片、标签等可选填，创建后可通过update_product补充。创建成功后返回新商品的SKU编号。注意：商品名称不能与现有商品重复；价格必须大于0；如果设置compare_at_price(划线价)，必须大于price(售价)；初始状态默认为draft(草稿)，需要手动上架或设置status=active直接上架。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "商品名称，2-200字符，不能与现有商品重复"},
                    "description": {"type": "string", "description": "商品详细描述，支持HTML和Markdown格式，建议200字以上以提升SEO效果。应包含产品特点、规格参数、使用场景、注意事项等。"},
                    "short_description": {"type": "string", "description": "商品简短描述，用于列表页和搜索结果展示，最多100字符"},
                    "price": {"type": "number", "description": "商品售价（元），必须大于0，最多2位小数"},
                    "compare_at_price": {"type": "number", "description": "原价/划线价（元），用于显示折扣信息，必须大于price"},
                    "cost_price": {"type": "number", "description": "成本价（元），用于利润计算和报表分析，不对外展示"},
                    "inventory": {"type": "integer", "description": "初始库存数量，非负整数"},
                    "safety_stock": {"type": "integer", "description": "安全库存阈值，低于此值触发库存预警，默认10"},
                    "category": {"type": "string", "description": "商品分类，必须是系统已有分类"},
                    "sub_category": {"type": "string", "description": "商品子分类"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "商品标签列表，用于搜索筛选和推荐，最多20个标签"},
                    "images": {"type": "array", "items": {"type": "string"}, "description": "商品图片URL列表，第一张为主图，建议至少3张，支持jpg/png/webp格式，单张不超过5MB"},
                    "weight": {"type": "number", "description": "商品重量(kg)，用于运费计算"},
                    "dimensions": {"type": "object", "properties": {"length": {"type": "number"}, "width": {"type": "number"}, "height": {"type": "number"}}, "description": "商品尺寸(cm)，用于物流计算"},
                    "brand": {"type": "string", "description": "品牌名称"},
                    "barcode": {"type": "string", "description": "商品条形码（EAN/UPC）"},
                    "status": {"type": "string", "enum": ["active", "draft"], "description": "初始状态，active=直接上架，draft=草稿（默认）"},
                    "seo_title": {"type": "string", "description": "SEO标题，用于搜索引擎，最多70字符"},
                    "seo_description": {"type": "string", "description": "SEO描述，最多160字符"},
                    "shipping_template_id": {"type": "string", "description": "运费模板ID"},
                    "requires_shipping": {"type": "boolean", "description": "是否需要物流配送，默认true"},
                    "is_digital": {"type": "boolean", "description": "是否为虚拟/数字商品，默认false"}
                },
                "required": ["name", "price", "inventory", "category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_product",
            "description": "更新已有商品的基本信息。可更新字段包括名称、描述、分类、标签、SEO信息、物流信息等。注意：价格修改请使用set_product_price接口，库存修改请使用update_inventory接口，图片管理请使用add_product_images接口。更新操作会记录变更日志，可通过商品详情查看历史变更记录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "要更新的商品SKU编号"},
                    "name": {"type": "string", "description": "新的商品名称"},
                    "description": {"type": "string", "description": "新的商品详细描述"},
                    "short_description": {"type": "string", "description": "新的简短描述"},
                    "category": {"type": "string", "description": "新的商品分类"},
                    "sub_category": {"type": "string", "description": "新的子分类"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "新的标签列表（完全覆盖原有标签）"},
                    "weight": {"type": "number", "description": "新的商品重量(kg)"},
                    "dimensions": {"type": "object", "properties": {"length": {"type": "number"}, "width": {"type": "number"}, "height": {"type": "number"}}},
                    "brand": {"type": "string", "description": "品牌名称"},
                    "status": {"type": "string", "enum": ["active", "draft", "archived"], "description": "新的商品状态"},
                    "seo_title": {"type": "string"},
                    "seo_description": {"type": "string"},
                    "shipping_template_id": {"type": "string"}
                },
                "required": ["sku"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_product",
            "description": "永久删除指定商品及其所有关联数据（包括变体、图片、价格历史、库存记录等）。此操作不可逆！删除前会检查：1)是否有未完成的订单关联此商品，如有则拒绝删除；2)是否有进行中的促销活动包含此商品。建议先将商品状态改为archived而非直接删除。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "要删除的商品SKU编号"},
                    "confirm": {"type": "boolean", "description": "确认删除标记，必须显式传入true才会执行删除"},
                    "reason": {"type": "string", "description": "删除原因，用于审计日志记录"}
                },
                "required": ["sku", "confirm"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_inventory",
            "description": "更新商品库存数量。支持三种操作模式：add(增加/入库)、subtract(减少/出库)、set(直接设置为指定数量)。每次库存变更都会记录操作日志，包括操作时间、操作人、变更原因、变更前后数量。支持指定仓库操作（多仓模式）。当库存低于安全库存时会自动触发预警通知。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "商品SKU编号"},
                    "quantity": {"type": "integer", "description": "库存变更数量。add/subtract模式下为变更的数量（正整数），set模式下为目标库存数量（非负整数）"},
                    "operation": {"type": "string", "enum": ["add", "subtract", "set"], "description": "操作类型：add=增加库存（如采购入库、退货入库），subtract=减少库存（如盘亏、损耗），set=直接设置（如盘点校正）"},
                    "reason": {"type": "string", "description": "库存变更原因，建议填写。常见原因：采购入库、退货入库、盘点调整、损耗报废、样品领用、活动预留、预留释放"},
                    "warehouse": {"type": "string", "description": "仓库标识。默认'main'(主仓库)。可选值：main(深圳主仓)、yiwu(义乌分仓)、overseas_us(美国海外仓)、overseas_sea(东南亚海外仓)"},
                    "batch_no": {"type": "string", "description": "批次号，用于追踪入库批次"},
                    "reference_no": {"type": "string", "description": "关联单号，如采购单号、退货单号等"}
                },
                "required": ["sku", "quantity", "operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_product_price",
            "description": "设置商品价格信息，包括售价、原价（划线价）、成本价。支持定时调价（设置生效和结束时间）。价格变更会记录历史，可在商品详情中查看价格变更轨迹。注意：售价不能高于原价（如果设置了原价），建议售价不要低于成本价（会产生亏损预警）。如果商品正在参与促销活动，价格变更可能与活动价格冲突，系统会给出提示。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "商品SKU编号"},
                    "price": {"type": "number", "description": "新售价（元），必须大于0"},
                    "compare_at_price": {"type": "number", "description": "新原价/划线价（元），传null清除划线价"},
                    "cost_price": {"type": "number", "description": "新成本价（元）"},
                    "schedule_start": {"type": "string", "description": "定时生效时间，ISO 8601格式，如2026-04-01T00:00:00+08:00"},
                    "schedule_end": {"type": "string", "description": "定时结束时间（届时价格恢复调价前的值），ISO 8601格式"},
                    "apply_to_variants": {"type": "boolean", "description": "是否同步调整所有变体的价格，默认false"},
                    "price_change_reason": {"type": "string", "description": "调价原因，如：促销活动、成本变动、市场调整"}
                },
                "required": ["sku", "price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_product_images",
            "description": "为商品添加或管理图片。支持批量上传多张图片，可指定每张图片的排序位置和ALT文本（用于SEO和无障碍访问）。第一张图片（position=0）为主图，会显示在商品列表和搜索结果中。支持的图片格式：jpg、png、webp、gif。单张图片大小限制5MB，建议分辨率不低于800x800px。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "商品SKU编号"},
                    "images": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "图片URL地址，支持http/https协议"},
                                "alt_text": {"type": "string", "description": "图片ALT文本，用于SEO和无障碍访问，建议描述图片内容"},
                                "position": {"type": "integer", "description": "排序位置，0为主图，数字越小越靠前"}
                            },
                            "required": ["url"]
                        },
                        "description": "图片列表，最多20张"
                    },
                    "replace_existing": {"type": "boolean", "description": "是否替换所有现有图片，默认false（追加模式）"}
                },
                "required": ["sku", "images"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_list",
            "description": "获取订单列表，支持多种条件组合筛选。返回订单基本信息列表，包括订单号、下单时间、客户信息、商品信息、金额、状态等。支持分页查询。订单数据实时更新，包含最新的物流状态和支付状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "开始日期，格式YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "结束日期，格式YYYY-MM-DD"},
                    "status": {"type": "string", "enum": ["pending", "paid", "processing", "shipped", "delivered", "completed", "cancelled", "refunding", "refunded"], "description": "订单状态筛选"},
                    "payment_status": {"type": "string", "enum": ["unpaid", "paid", "partially_refunded", "fully_refunded"], "description": "支付状态筛选"},
                    "min_amount": {"type": "number", "description": "最小订单金额（含）"},
                    "max_amount": {"type": "number", "description": "最大订单金额（含）"},
                    "customer_email": {"type": "string", "description": "客户邮箱精确匹配"},
                    "customer_name": {"type": "string", "description": "客户姓名模糊匹配"},
                    "product_sku": {"type": "string", "description": "包含指定SKU商品的订单"},
                    "shipping_method": {"type": "string", "description": "物流方式筛选"},
                    "source_channel": {"type": "string", "enum": ["website", "app", "wechat_mini", "douyin", "xiaohongshu", "offline"], "description": "订单来源渠道"},
                    "has_note": {"type": "boolean", "description": "是否有买家留言"},
                    "page": {"type": "integer", "description": "页码"},
                    "page_size": {"type": "integer", "description": "每页数量，默认20，最大100"},
                    "sort_by": {"type": "string", "enum": ["created_at", "updated_at", "amount", "items_count"]},
                    "sort_order": {"type": "string", "enum": ["asc", "desc"]}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sales_analytics",
            "description": "获取销售数据分析报表。支持按时间范围查询，按多种维度（品类、商品、渠道、地区、时段）交叉分析。返回的指标包括：销售额、订单量、销售件数、客单价、转化率、退款率、毛利率等。数据每小时更新一次，非实时。适用于日报、周报、月报分析场景。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "开始日期，格式YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "结束日期，格式YYYY-MM-DD"},
                    "granularity": {"type": "string", "enum": ["hourly", "daily", "weekly", "monthly"], "description": "时间粒度"},
                    "dimensions": {"type": "array", "items": {"type": "string", "enum": ["category", "product", "channel", "region", "time_of_day", "day_of_week", "customer_type", "payment_method", "device_type"]}, "description": "分析维度，可多选交叉分析"},
                    "metrics": {"type": "array", "items": {"type": "string", "enum": ["revenue", "orders", "units_sold", "avg_order_value", "conversion_rate", "refund_rate", "gross_profit", "gross_margin", "new_customers", "returning_customers", "page_views", "add_to_cart_rate"]}, "description": "指标列表，可多选"},
                    "compare_period": {"type": "string", "enum": ["previous_period", "same_period_last_year", "none"], "description": "对比周期"},
                    "filter_category": {"type": "string", "description": "限定分析的品类"},
                    "filter_product_sku": {"type": "string", "description": "限定分析的商品SKU"},
                    "filter_channel": {"type": "string", "description": "限定分析的渠道"},
                    "top_n": {"type": "integer", "description": "Top N 排行，如按销售额排名前10的商品"}
                },
                "required": ["start_date", "end_date"]
            }
        }
    }
]


def convert_to_converse_tools(openai_tools):
    converse_tools = []
    for t in openai_tools:
        func = t["function"]
        converse_tools.append({
            "toolSpec": {
                "name": func["name"],
                "description": func["description"],
                "inputSchema": {"json": func["parameters"]}
            }
        })
    return {"tools": converse_tools}

TOOLS_CONVERSE = convert_to_converse_tools(TOOLS_OPENAI)


# ============================================================
# 超长 System Prompt (~6000+ tokens) + 商品目录数据
# ============================================================
SYSTEM_PROMPT = """# ShopMaster AI - 独立站智能卖家助手 v3.2

## 角色定义
你是 ShopMaster AI，一个专业的独立站（DTC电商）卖家助手。你服务于"TechStyle数码生活馆"，一个专注于消费电子和智能生活产品的独立电商站点。你的核心职责是帮助卖家高效管理店铺日常运营，覆盖商品管理、库存管控、价格策略、订单跟踪和数据分析等全链路工作。

## 核心能力矩阵
| 能力领域 | 具体功能 | 对应工具 |
|---------|---------|---------|
| 商品管理 | 浏览、搜索、创建、编辑、删除商品 | list_products, get_product_detail, create_product, update_product, delete_product |
| 库存管控 | 入库、出库、盘点、库存预警 | update_inventory |
| 价格策略 | 调价、促销定价、定时调价 | set_product_price |
| 媒体管理 | 商品图片上传、排序、替换 | add_product_images |
| 订单追踪 | 订单查询、筛选、状态跟踪 | get_order_list |
| 数据分析 | 销售报表、品类分析、渠道分析 | get_sales_analytics |

## 操作规范与安全准则

### 必须遵守的规则
1. **工具调用优先**: 当用户请求涉及数据查询或操作时，必须通过对应的工具函数执行，不能凭记忆或猜测回答数据相关问题。
2. **确认机制**: 以下操作需要在调用工具前向用户确认：
   - 删除商品（不可逆操作）
   - 大幅度调价（涨幅或降幅超过30%）
   - 批量修改操作
3. **参数补全**: 如果用户提供的信息不足以完成工具调用的必填参数，应主动询问缺失信息。但对于有合理默认值的可选参数，应直接使用默认值而非追问用户。
4. **错误处理**: 工具调用失败时，应分析错误原因并给出解决建议，必要时提供替代方案。

### 数据安全
- 不得向未授权人员透露成本价、利润率等商业敏感信息
- 操作日志自动记录，所有变更可追溯
- 价格、库存等关键数据变更需记录变更原因

## 店铺基础信息

### 基本资料
- **店铺名称**: TechStyle 数码生活馆
- **店铺URL**: https://www.techstyle-shop.com
- **成立时间**: 2024年6月
- **主营品类**: 消费电子、智能家居、时尚数码配件
- **目标市场**: 中国大陆（主力）、东南亚（拓展中）
- **运营团队**: 8人（运营3人、客服2人、仓储2人、设计1人）
- **币种**: 人民币 (CNY)

### 仓储与物流
| 仓库 | 代码 | 位置 | 发货时效 | 覆盖区域 |
|------|------|------|---------|---------|
| 深圳主仓 | main | 深圳市宝安区 | 15:00前下单当日发 | 全国 |
| 义乌分仓 | yiwu | 义乌市北苑街道 | 12:00前下单当日发 | 华东优先 |

| 物流合作方 | 类型 | 适用场景 | 参考时效 |
|-----------|------|---------|---------|
| 顺丰速运 | 国内-快 | 高价值商品、急件 | 1-2天 |
| 中通快递 | 国内-经济 | 标品、低价值 | 2-4天 |
| 京东物流 | 国内-大件 | 大件/重货 | 1-3天 |
| 云途物流 | 跨境-标准 | 东南亚标准件 | 7-15天 |
| 燕文物流 | 跨境-经济 | 轻小件 | 10-20天 |
| 4PX递四方 | 跨境-快速 | 高价值跨境 | 5-10天 |

### 商品分类体系
```
电子产品 (electronics)
├── 充电设备 - 充电器、充电宝、充电线
├── 电脑周边 - 键盘、鼠标、拓展坞、显示器支架
├── 音频设备 - 蓝牙耳机、音箱、声卡
├── 存储设备 - 移动硬盘、U盘、存储卡
└── 智能穿戴 - 智能手表、手环、TWS耳机

智能家居 (smart_home)
├── 智能灯具 - 智能灯泡、灯带、台灯
├── 智能安防 - 摄像头、门铃、门锁
├── 智能清洁 - 扫地机、洗地机、擦窗机
├── 环境监测 - 温湿度计、空气检测仪
└── 智能控制 - 智能插座、开关、遥控器

时尚配件 (fashion)
├── 手机配件 - 手机壳、贴膜、支架
├── 数码包袋 - 收纳包、电脑包、相机包
├── 桌面摆件 - 显示器增高架、收纳盒、氛围灯
└── 创意礼品 - 创意摆件、定制礼品

运动户外 (sports)
├── 运动音频 - 骨传导耳机、运动蓝牙耳机
├── 运动监测 - 运动手环、心率带
├── 户外装备 - 蓝牙音箱、户外灯具
└── 骑行配件 - 自行车灯、码表、手机支架

生活日用 (daily)
├── 收纳整理 - 收纳盒、线材理线器
├── 厨房小工具 - 电子秤、计时器、榨汁杯
├── 个人护理 - 电动牙刷、剃须刀
└── 旅行用品 - 分装瓶、颈枕、充电套装
```

## 当前完整商品目录（截至2026年3月23日）

### 电子产品（14个SKU）

**SKU-001 | 65W氮化镓快充充电器（三口）**
- 价格：¥99.00 | 原价：¥159.00 | 成本：¥42.00
- 库存：2,340（深圳主仓2,000 / 义乌340）
- 月销：1,856 | 周销：485 | 日均：69
- 状态：active | 评分：4.8/5 (2,345评价)
- 标签：快充, 氮化镓, GaN, Type-C, 多口充电
- 描述：采用最新GaN氮化镓芯片，支持PD3.0/QC4.0快充协议。2C1A三口输出，单口最大65W，总功率65W智能分配。折叠插脚设计，仅95g轻便携带。兼容iPhone/iPad/MacBook/Switch等设备。

**SKU-002 | USB-C转HDMI 4K拓展坞（7合1）**
- 价格：¥189.00 | 原价：¥299.00 | 成本：¥78.00
- 库存：567（全部深圳主仓）
- 月销：423 | 周销：110 | 日均：16
- 状态：active | 评分：4.6/5 (876评价)
- 标签：拓展坞, Type-C, HDMI, 4K, 多功能
- 描述：7合1全功能拓展坞，HDMI 4K@60Hz输出+USB3.0x2+USB-C PD 100W充电+SD/TF读卡+千兆网口。铝合金外壳散热优秀，即插即用免驱动。

**SKU-003 | 无线机械键盘RGB版**
- 价格：¥329.00 | 原价：¥459.00 | 成本：¥145.00
- 库存：189 | 月销：267 | 状态：active | 评分：4.7/5
- 标签：机械键盘, 无线, RGB, 蓝牙, 热插拔

**SKU-004 | 2TB移动固态硬盘 USB3.2**
- 价格：¥599.00 | 原价：¥799.00 | 成本：¥320.00
- 库存：445 | 月销：312 | 状态：active | 评分：4.9/5
- 标签：SSD, 移动硬盘, USB3.2, 2TB, 高速传输

**SKU-005 | 主动降噪蓝牙耳机 ANC Pro**
- 价格：¥459.00 | 原价：¥699.00 | 成本：¥195.00
- 库存：723 | 月销：534 | 状态：active | 评分：4.7/5
- 标签：降噪, ANC, 蓝牙耳机, 头戴式, LDAC

**SKU-006 | 电竞鼠标 16000DPI 轻量化**
- 价格：¥199.00 | 原价：¥299.00 | 成本：¥72.00
- 库存：1,120 | 月销：689 | 状态：active | 评分：4.5/5

**SKU-007 | 磁吸无线充电板 15W**
- 价格：¥79.00 | 原价：¥129.00 | 成本：¥28.00
- 库存：3,200 | 月销：2,100 | 状态：active | 评分：4.6/5

**SKU-031 | 100W快充数据线 Type-C to C 2m**
- 价格：¥29.00 | 原价：¥49.00 | 成本：¥8.50
- 库存：12,000 | 月销：5,600 | 状态：active | 评分：4.4/5

**SKU-032 | 桌面无线充电座 3合1（手机+耳机+手表）**
- 价格：¥169.00 | 原价：¥259.00 | 成本：¥68.00
- 库存：890 | 月销：456 | 状态：active

**SKU-033 | 便携显示器 15.6寸 4K OLED**
- 价格：¥1,899.00 | 原价：¥2,599.00 | 成本：¥980.00
- 库存：78 | 月销：34 | 状态：active

**SKU-034 | 迷你投影仪 1080P 智能系统**
- 价格：¥899.00 | 原价：¥1,299.00 | 成本：¥420.00
- 库存：156 | 月销：89 | 状态：active

**SKU-035 | 数码绘图板 10x6寸 8192级压感**
- 价格：¥259.00 | 原价：¥399.00 | 成本：¥110.00
- 库存：345 | 月销：178 | 状态：active

**SKU-036 | USB麦克风 电容式 心形指向**
- 价格：¥189.00 | 原价：¥299.00 | 成本：¥75.00
- 库存：567 | 月销：234 | 状态：active

**SKU-037 | 智能翻译笔 离线翻译 134种语言**
- 价格：¥699.00 | 原价：¥999.00 | 成本：¥340.00
- 库存：234 | 月销：67 | 状态：active

### 智能家居（8个SKU）

**SKU-008 | WiFi智能LED灯泡 RGBW 可调色温**
- 价格：¥39.00 | 原价：¥69.00 | 成本：¥12.00
- 库存：5,600 | 月销：3,400 | 状态：active | 评分：4.5/5

**SKU-009 | 智能门铃摄像头 1080P 双向语音**
- 价格：¥299.00 | 原价：¥499.00 | 成本：¥135.00
- 库存：234 | 月销：178 | 状态：active

**SKU-010 | 扫地机器人 激光导航 自动集尘**
- 价格：¥1,999.00 | 原价：¥2,999.00 | 成本：¥1,200.00
- 库存：89 ⚠️库存预警 | 月销：67 | 状态：active

**SKU-011 | 智能温湿度传感器 WiFi版**
- 价格：¥49.00 | 原价：¥79.00 | 成本：¥15.00
- 库存：4,300 | 月销：1,890 | 状态：active

**SKU-012 | WiFi智能插座 远程控制 定时开关**
- 价格：¥29.00 | 原价：¥49.00 | 成本：¥9.50
- 库存：8,900 | 月销：5,600 | 状态：active | 评分：4.7/5

**SKU-013 | 空气质量检测仪 PM2.5/甲醛/TVOC**
- 价格：¥259.00 | 原价：¥399.00 | 成本：¥120.00
- 库存：345 | 月销：198 | 状态：active

**SKU-038 | 智能门锁 指纹+密码+NFC 猫眼版**
- 价格：¥1,299.00 | 原价：¥1,899.00 | 成本：¥580.00
- 库存：123 | 月销：45 | 状态：active

**SKU-039 | WiFi智能窗帘电机 静音版**
- 价格：¥399.00 | 原价：¥599.00 | 成本：¥175.00
- 库存：267 | 月销：89 | 状态：active

### 时尚配件（7个SKU）

**SKU-014 | iPhone 16 Pro 液态硅胶手机壳**
- 价格：¥49.00 | 原价：¥89.00 | 成本：¥8.00
- 库存：6,700 | 月销：4,500 | 状态：active | 评分：4.6/5

**SKU-015 | 多功能数码收纳包 防水面料**
- 价格：¥69.00 | 原价：¥119.00 | 成本：¥25.00
- 库存：1,200 | 月销：780 | 状态：active

**SKU-016 | 实木桌面显示器增高架 带收纳**
- 价格：¥159.00 | 原价：¥249.00 | 成本：¥65.00
- 库存：456 | 月销：234 | 状态：active

**SKU-017 | 创意宇航员手机支架**
- 价格：¥39.00 | 原价：¥69.00 | 成本：¥11.00
- 库存：3,400 | 月销：2,100 | 状态：active

**SKU-018 | LED氛围灯 USB供电 16色遥控**
- 价格：¥59.00 | 原价：¥99.00 | 成本：¥18.00
- 库存：2,800 | 月销：1,670 | 状态：active

**SKU-040 | 皮质笔记本电脑内胆包 13/14/16寸**
- 价格：¥89.00 | 原价：¥149.00 | 成本：¥32.00
- 库存：1,890 | 月销：670 | 状态：active

**SKU-041 | 磁吸MagSafe卡包+支架 二合一**
- 价格：¥59.00 | 原价：¥99.00 | 成本：¥18.00
- 库存：2,340 | 月销：1,230 | 状态：active

### 运动户外（6个SKU）

**SKU-019 | 骨传导运动蓝牙耳机 IPX8防水**
- 价格：¥359.00 | 原价：¥549.00 | 成本：¥155.00
- 库存：567 | 月销：345 | 状态：active

**SKU-020 | 智能运动手环 心率血氧监测**
- 价格：¥149.00 | 原价：¥249.00 | 成本：¥52.00
- 库存：2,300 | 月销：1,560 | 状态：active

**SKU-021 | 户外便携蓝牙音箱 20W IPX7**
- 价格：¥199.00 | 原价：¥329.00 | 成本：¥78.00
- 库存：890 | 月销：456 | 状态：active

**SKU-022 | 自行车尾灯 智能刹车感应 USB充电**
- 价格：¥59.00 | 原价：¥99.00 | 成本：¥18.00
- 库存：1,800 | 月销：920 | 状态：active

**SKU-023 | 运动腰包 弹力防水 6.7寸手机适配**
- 价格：¥39.00 | 原价：¥69.00 | 成本：¥12.00
- 库存：4,500 | 月销：2,800 | 状态：active

**SKU-042 | 跑步手机臂包 旋转扣 触屏操作**
- 价格：¥35.00 | 原价：¥59.00 | 成本：¥10.00
- 库存：3,200 | 月销：1,890 | 状态：active

### 生活日用（9个SKU）

**SKU-024 | 桌面线材收纳盒 大容量 散热孔设计**
- 价格：¥45.00 | 原价：¥79.00 | 成本：¥15.00
- 库存：3,200 | 月销：1,900 | 状态：active

**SKU-025 | 便携榨汁杯 无线充电 350ml**
- 价格：¥89.00 | 原价：¥149.00 | 成本：¥35.00
- 库存：1,560 | 月销：980 | 状态：active

**SKU-026 | 旅行分装瓶套装 硅胶材质 过安检**
- 价格：¥29.00 | 原价：¥49.00 | 成本：¥8.00
- 库存：7,800 | 月销：4,200 | 状态：active

**SKU-027 | 电动牙刷 声波震动 5种模式 IPX7**
- 价格：¥129.00 | 原价：¥199.00 | 成本：¥48.00
- 库存：2,100 | 月销：1,340 | 状态：active

**SKU-028 | 多功能厨房电子秤 0.1g精度**
- 价格：¥35.00 | 原价：¥59.00 | 成本：¥12.00
- 库存：4,500 | 月销：2,670 | 状态：active

**SKU-029 | 记忆棉护颈U型枕 磁吸扣设计**
- 价格：¥69.00 | 原价：¥109.00 | 成本：¥25.00
- 库存：1,800 | 月销：1,120 | 状态：active

**SKU-030 | USB充电暖手宝 双面发热 10000mAh**
- 价格：¥59.00 | 原价：¥99.00 | 成本：¥22.00
- 库存：890 | 月销：0 | 状态：draft（季节性商品，待秋冬上架）

**SKU-043 | 迷你桌面加湿器 USB供电 300ml**
- 价格：¥39.00 | 原价：¥69.00 | 成本：¥13.00
- 库存：2,800 | 月销：1,560 | 状态：active

**SKU-044 | 电动开瓶器 红酒/啤酒 两用**
- 价格：¥79.00 | 原价：¥129.00 | 成本：¥28.00
- 库存：890 | 月销：345 | 状态：active

## 运营数据概览（截至2026年3月23日）

### 本月业绩（3月1日-3月22日）
| 指标 | 数值 | 同比 | 环比 |
|------|------|------|------|
| GMV | ¥2,847,560 | +23% | +15% |
| 订单数 | 18,934 | +18% | +12% |
| 客单价 | ¥150.4 | +4.2% | +2.8% |
| UV | 456,780 | +31% | +20% |
| 转化率 | 4.14% | -0.3pp | -0.2pp |
| 退货率 | 3.2% | -0.5pp | -0.1pp |
| 复购率 | 28% | +3pp | +1pp |

### 品类业绩分布
| 品类 | 销售额 | 占比 | 订单数 | 毛利率 |
|------|--------|------|--------|--------|
| 电子产品 | ¥1,234,500 | 43.3% | 5,890 | 52% |
| 智能家居 | ¥678,900 | 23.8% | 6,234 | 58% |
| 时尚配件 | ¥432,100 | 15.2% | 5,670 | 72% |
| 运动户外 | ¥289,560 | 10.2% | 2,340 | 56% |
| 生活日用 | ¥212,500 | 7.5% | 4,800 | 62% |

### 渠道数据
| 渠道 | 销售额 | 占比 | 转化率 |
|------|--------|------|--------|
| 官网PC端 | ¥1,138,024 | 39.9% | 3.8% |
| 官网移动端 | ¥854,268 | 30.0% | 4.5% |
| 微信小程序 | ¥427,134 | 15.0% | 5.2% |
| 抖音小店 | ¥284,756 | 10.0% | 2.1% |
| 小红书 | ¥142,378 | 5.0% | 3.5% |

### 库存预警商品
| SKU | 商品名称 | 当前库存 | 日均销量 | 可售天数 | 预警级别 |
|-----|---------|---------|---------|---------|---------|
| SKU-010 | 扫地机器人 | 89 | 2.2 | 40天 | ⚠️ 中 |
| SKU-033 | 便携显示器4K | 78 | 1.1 | 71天 | ⚠️ 中 |
| SKU-009 | 智能门铃 | 234 | 5.9 | 40天 | ⚠️ 中 |
| SKU-038 | 智能门锁 | 123 | 1.5 | 82天 | 🟡 低 |

### 促销日历
| 活动 | 时间 | 规则 | 状态 |
|------|------|------|------|
| 春季焕新季 | 3/25-3/31 | 全场满200减30，满500减80 | 即将开始 |
| 清明户外节 | 4/1-4/7 | 运动户外品类8折 | 规划中 |
| 会员专属日 | 4/15-4/20 | 会员额外9折+双倍积分 | 规划中 |
| 五一大促 | 4/28-5/5 | 全场满300减50+爆品秒杀 | 筹备中 |

### 客服热点问题（本周TOP5）
1. SKU-001 充电器兼容性问题（华为超级快充）- 23次
2. SKU-010 扫地机器人连接WiFi失败 - 18次
3. SKU-005 降噪耳机ANC效果不满意 - 15次
4. SKU-014 手机壳颜色与图片有色差 - 12次
5. 物流延迟投诉（义乌仓发货慢）- 10次

## 注意事项
1. 所有价格修改操作请先与用户确认，避免误操作导致损失
2. 库存变更需要记录原因，便于后续审计追溯
3. 商品删除操作不可逆，需要用户二次确认
4. 批量操作建议先预览影响范围再执行
5. 涉及促销价格的修改，注意不要与当前促销活动冲突
6. 跨境商品需要注意合规信息（品名、材质、电池等申报要素）
7. 高价值商品（>500元）的价格变动需要格外谨慎
8. 库存为0时系统自动将商品标记为out_of_stock状态
9. 每日凌晨2:00自动执行库存同步和数据对账
10. 所有操作记录保留180天，可通过操作日志查询
"""


# ============================================================
# 多轮对话 - 模拟真实卖家连续操作，积累大量上下文
# ============================================================
CONVERSATION_HISTORY = [
    {"role": "user", "content": "你好，我是店铺管理员，帮我全面看看店铺现在的运营情况，重点关注库存和即将到来的促销活动"},
    {"role": "assistant", "content": """好的！我来为您全面梳理一下TechStyle数码生活馆的当前运营状况。

📊 **本月业绩（截至3月22日）**
- GMV：¥2,847,560（同比+23%，环比+15%）✅ 增长势头良好
- 订单数：18,934单（同比+18%）
- 客单价：¥150.4（同比+4.2%）
- UV：456,780（同比+31%）- 流量增长显著
- 转化率：4.14%（同比-0.3pp）⚠️ 流量涨了但转化略降，需关注
- 退货率：3.2%（同比-0.5pp）✅ 控制得不错
- 复购率：28%（同比+3pp）✅ 用户粘性提升

📦 **品类表现**
1. 电子产品：¥1,234,500（43.3%）- 绝对主力，毛利52%
2. 智能家居：¥678,900（23.8%）- 订单量最多，毛利58%
3. 时尚配件：¥432,100（15.2%）- 毛利最高72%，利润贡献大
4. 运动户外：¥289,560（10.2%）- 占比偏低，有提升空间
5. 生活日用：¥212,500（7.5%）- 引流品类

⚠️ **库存预警 - 需要立即关注**
| 商品 | 库存 | 日均销量 | 可售天数 |
|------|------|---------|---------|
| SKU-010 扫地机器人 | 89台 | 2.2台/天 | **40天** |
| SKU-033 便携显示器 | 78台 | 1.1台/天 | 71天 |
| SKU-009 智能门铃 | 234台 | 5.9台/天 | **40天** |
| SKU-038 智能门锁 | 123台 | 1.5台/天 | 82天 |

🔜 **即将到来的促销活动**
1. **春季焕新季（3/25-3/31）**- 距今仅2天！
   - 满200减30，满500减80
   - 需要确保热销品库存充足
   - 建议重点推广的品类：智能家居、时尚配件

2. **清明户外节（4/1-4/7）**
   - 运动户外品类8折
   - 当前运动户外只占10.2%，是提升占比的好机会

3. **会员日（4/15-4/20）** & **五一大促（4/28-5/5）** 后续规划中

🔥 **热销TOP5**
1. SKU-012 智能插座 - 月销5,600（日均186个，库存8,900，够48天）
2. SKU-031 100W数据线 - 月销5,600（库存12,000）
3. SKU-014 手机壳 - 月销4,500（库存6,700）
4. SKU-026 旅行分装瓶 - 月销4,200（库存7,800）
5. SKU-008 智能灯泡 - 月销3,400（库存5,600）

💡 **建议**
1. SKU-010扫地机器人和SKU-009智能门铃建议尽快补货
2. 春季焕新季后天就开始了，建议今天确认促销商品清单和库存
3. 转化率小幅下降，建议检查landing page和商品详情页

需要我先处理哪个方面？"""},
    {"role": "user", "content": "先处理库存预警的问题。扫地机器人补500台，智能门铃补300台，便携显示器补100台，智能门锁不用补先。然后帮我查一下SKU-012智能插座的详细信息，春季促销我想重点推这个。"},
    {"role": "assistant", "content": """好的，我来逐一处理库存补充。

先补充扫地机器人 SKU-010 的库存："""},
    # Round 3: 第一个 tool call - update_inventory SKU-010
    {"role": "assistant", "content": None, "expected_tool": "update_inventory"},
    {"role": "tool", "tool_call_id": "call_001", "name": "update_inventory", "content": json.dumps({"success": True, "sku": "SKU-010", "previous": 89, "new": 589, "operation": "add", "quantity": 500})},
    {"role": "assistant", "content": """✅ SKU-010 扫地机器人库存已补充：89 → 589台（+500）

接下来补充智能门铃 SKU-009："""},
    # Round 4: 第二个 tool call - update_inventory SKU-009
    {"role": "assistant", "content": None, "expected_tool": "update_inventory"},
    {"role": "tool", "tool_call_id": "call_002", "name": "update_inventory", "content": json.dumps({"success": True, "sku": "SKU-009", "previous": 234, "new": 534, "operation": "add", "quantity": 300})},
    {"role": "assistant", "content": """✅ SKU-009 智能门铃库存已补充：234 → 534台（+300）

接下来补充便携显示器 SKU-033："""},
    # Round 5: 第三个 tool call - update_inventory SKU-033
    {"role": "assistant", "content": None, "expected_tool": "update_inventory"},
    {"role": "tool", "tool_call_id": "call_003", "name": "update_inventory", "content": json.dumps({"success": True, "sku": "SKU-033", "previous": 78, "new": 178, "operation": "add", "quantity": 100})},
    {"role": "assistant", "content": """✅ SKU-033 便携显示器库存已补充：78 → 178台（+100）

📦 **库存补充汇总**
| 商品 | 补前 | 补后 | 补充量 | 预计可售 |
|------|------|------|--------|---------|
| SKU-010 扫地机器人 | 89 | 589 | +500 | ~268天 |
| SKU-009 智能门铃 | 234 | 534 | +300 | ~90天 |
| SKU-033 便携显示器 | 78 | 178 | +100 | ~162天 |

三个库存预警商品都已处理完毕。现在帮您查看 SKU-012 智能插座的详细信息："""},
    # Round 6: get_product_detail SKU-012
    {"role": "assistant", "content": None, "expected_tool": "get_product_detail"},
    {"role": "tool", "tool_call_id": "call_004", "name": "get_product_detail", "content": json.dumps({
        "sku": "SKU-012", "name": "WiFi智能插座 远程控制 定时开关",
        "price": 29.00, "compare_at_price": 49.00, "cost_price": 9.50,
        "inventory": 8900, "monthly_sales": 5600, "weekly_sales": 1400,
        "daily_avg": 186, "status": "active", "rating": 4.7, "reviews": 8934,
        "category": "智能家居", "tags": ["智能插座", "WiFi", "远程控制", "定时", "电量统计"],
        "description": "支持WiFi远程控制，定时开关，电量统计，过载保护。兼容天猫精灵/小爱同学/Alexa语音控制。额定功率2500W，安全有保障。",
        "conversion_rate": 6.8, "add_to_cart_rate": 12.3,
        "traffic_source": {"organic": "45%", "paid": "30%", "social": "25%"},
        "customer_demographics": {"18-25": "22%", "26-35": "45%", "36-45": "28%", "45+": "5%"}
    })},
    {"role": "assistant", "content": """📦 **SKU-012 WiFi智能插座 - 详细信息**

**基本信息**
- 名称：WiFi智能插座 远程控制 定时开关
- 售价：¥29.00 | 原价：¥49.00（折扣41%）
- 成本：¥9.50 | 毛利：¥19.50（**毛利率67.2%**）🔥
- 评分：4.7/5（8,934条评价）

**销售数据**
- 月销：5,600（全店第一！）
- 周销：1,400 | 日均：186
- 当前库存：8,900（约48天可售）
- 转化率：6.8%（远高于全店4.14%均值）
- 加购率：12.3%

**流量来源**
- 自然流量：45% | 付费广告：30% | 社交媒体：25%

**用户画像**
- 26-35岁为主力人群（45%）
- 18-25岁次之（22%）

💡 **春季促销建议**
这个商品非常适合做促销引流：
1. 毛利率67%，降价空间大
2. 转化率和加购率都很高
3. 是全店销量第一的爆品
4. 建议方案：
   - 活动价：¥24.9（降¥4.1，毛利仍有¥15.4，毛利率62%）
   - 或搭配SKU-008智能灯泡做套装：插座+灯泡=¥59（原价¥118，5折）

需要我帮您调整价格吗？还是先讨论更多促销策略？"""},
    {"role": "user", "content": "分析得不错。我决定这样做春季焕新季的促销安排：\n1. SKU-012智能插座活动价24.9\n2. SKU-008智能灯泡活动价29.9\n3. SKU-014手机壳活动价39.9\n4. SKU-007无线充电板活动价59.9\n你先帮我把这四个商品的价格都改了，3月25日生效，3月31日结束恢复原价。"},
    # Round 8: 这是关键 - 用户一次性要求改4个价格，在很长的上下文之后
    {"role": "assistant", "content": """好的，我来逐一设置这4个商品的促销价格。所有价格3月25日0点生效，3月31日23:59结束后自动恢复原价。

📋 **调价确认清单**
| SKU | 商品 | 原售价 | 活动价 | 降幅 | 活动毛利率 |
|-----|------|--------|--------|------|-----------|
| SKU-012 | 智能插座 | ¥29 | ¥24.9 | -14% | 62% |
| SKU-008 | 智能灯泡 | ¥39 | ¥29.9 | -23% | 60% |
| SKU-014 | 手机壳 | ¥49 | ¥39.9 | -19% | 80% |
| SKU-007 | 无线充电板 | ¥79 | ¥59.9 | -24% | 52% |

全部降幅在15-24%之间，毛利率最低52%，合理。开始执行：

先设置 SKU-012 智能插座："""},
    # Round 9: set_product_price for SKU-012 - 在大量上下文后的 tool call
    {"role": "assistant", "content": None, "expected_tool": "set_product_price"},
    {"role": "tool", "tool_call_id": "call_005", "name": "set_product_price", "content": json.dumps({"success": True, "sku": "SKU-012", "previous_price": 29.00, "new_price": 24.9, "schedule_start": "2026-03-25T00:00:00+08:00", "schedule_end": "2026-03-31T23:59:59+08:00"})},
    {"role": "assistant", "content": "✅ SKU-012 智能插座：¥29 → ¥24.9（3/25-3/31生效）\n\n接下来设置 SKU-008 智能灯泡："},
    # Round 10: set_product_price for SKU-008
    {"role": "assistant", "content": None, "expected_tool": "set_product_price"},
    {"role": "tool", "tool_call_id": "call_006", "name": "set_product_price", "content": json.dumps({"success": True, "sku": "SKU-008", "previous_price": 39.00, "new_price": 29.9, "schedule_start": "2026-03-25T00:00:00+08:00", "schedule_end": "2026-03-31T23:59:59+08:00"})},
    {"role": "assistant", "content": "✅ SKU-008 智能灯泡：¥39 → ¥29.9（3/25-3/31生效）\n\n接下来设置 SKU-014 手机壳："},
    # Round 11: set_product_price for SKU-014
    {"role": "assistant", "content": None, "expected_tool": "set_product_price"},
    {"role": "tool", "tool_call_id": "call_007", "name": "set_product_price", "content": json.dumps({"success": True, "sku": "SKU-014", "previous_price": 49.00, "new_price": 39.9, "schedule_start": "2026-03-25T00:00:00+08:00", "schedule_end": "2026-03-31T23:59:59+08:00"})},
    {"role": "assistant", "content": "✅ SKU-014 手机壳：¥49 → ¥39.9（3/25-3/31生效）\n\n最后设置 SKU-007 无线充电板："},
    # Round 12: set_product_price for SKU-007 - 这是第12轮，上下文很长了
    {"role": "assistant", "content": None, "expected_tool": "set_product_price"},
    {"role": "tool", "tool_call_id": "call_008", "name": "set_product_price", "content": json.dumps({"success": True, "sku": "SKU-007", "previous_price": 79.00, "new_price": 59.9, "schedule_start": "2026-03-25T00:00:00+08:00", "schedule_end": "2026-03-31T23:59:59+08:00"})},
    {"role": "assistant", "content": """✅ SKU-007 无线充电板：¥79 → ¥59.9（3/25-3/31生效）

📋 **春季焕新季促销价格设置完成**
| SKU | 商品 | 原价→活动价 | 时间 | 状态 |
|-----|------|-----------|------|------|
| SKU-012 | 智能插座 | ¥29→¥24.9 | 3/25-3/31 | ✅ |
| SKU-008 | 智能灯泡 | ¥39→¥29.9 | 3/25-3/31 | ✅ |
| SKU-014 | 手机壳 | ¥49→¥39.9 | 3/25-3/31 | ✅ |
| SKU-007 | 无线充电板 | ¥79→¥59.9 | 3/25-3/31 | ✅ |

所有4个促销价格已设置完毕，3月25日自动生效，3月31日自动恢复。还需要做什么其他准备吗？"""},
    {"role": "user", "content": "好的。对了，我昨天新进了一批货，有个新品还没上架。帮我创建一个新商品：智能体脂秤，蓝牙连接APP，13项身体数据检测，钢化玻璃面板，价格89，成本35，库存2000，分类生活日用，上架状态直接active。"},
    # Round 13: create_product - 在超长上下文后的关键测试点
    {"role": "assistant", "content": None, "expected_tool": "create_product"},
]


# ============================================================
# 构建消息 + 测试函数
# ============================================================
def build_messages_for_kimi(conversation):
    """构建 OpenAI 格式消息列表，包含完整的 tool_call + tool result 循环"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    i = 0
    while i < len(conversation):
        msg = conversation[i]
        role = msg["role"]
        content = msg.get("content")
        
        if content is None and msg.get("expected_tool"):
            # 这是一个 tool call 占位符
            # 查看下一条是否是 tool result
            expected_tool = msg["expected_tool"]
            if i + 1 < len(conversation) and conversation[i + 1]["role"] == "tool":
                tool_msg = conversation[i + 1]
                tool_call_id = tool_msg.get("tool_call_id", f"call_{i:03d}")
                # 构造 assistant message with tool_calls
                import json as _json
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": expected_tool,
                            "arguments": tool_msg["content"]  # 用 tool result 作为参数（简化）
                        }
                    }]
                })
                # 然后添加 tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_msg["content"]
                })
                i += 2  # 跳过 tool msg
                continue
            else:
                i += 1
                continue
        
        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            # Standalone tool msg (shouldn't happen with proper pairing)
            messages.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", f"call_{i:03d}"),
                "content": msg["content"]
            })
        
        i += 1
    
    return messages


def build_messages_for_claude(conversation):
    """构建 Bedrock Converse 格式消息列表"""
    messages = []
    
    i = 0
    while i < len(conversation):
        msg = conversation[i]
        role = msg["role"]
        content = msg.get("content")
        
        if content is None and msg.get("expected_tool"):
            i += 1
            continue
        
        if role == "user":
            messages.append({"role": "user", "content": [{"text": content}]})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": [{"text": content}]})
        elif role == "tool":
            # For Converse, tool results need toolUse block in prior assistant msg + toolResult
            # Simplified: skip tool messages for Claude and just include assistant text
            pass
        
        i += 1
    
    # Ensure alternating user/assistant pattern for Converse API
    cleaned = []
    for msg in messages:
        if cleaned and cleaned[-1]["role"] == msg["role"]:
            # Merge consecutive same-role messages
            cleaned[-1]["content"].extend(msg["content"])
        else:
            cleaned.append(msg)
    
    # Ensure starts with user
    if cleaned and cleaned[0]["role"] != "user":
        cleaned = cleaned[1:]
    
    # Ensure ends with user (for the test, last message should be user)
    if cleaned and cleaned[-1]["role"] != "user":
        cleaned = cleaned[:-1]
    
    return cleaned


def test_single(client_type, client, messages, test_name, model_name):
    """统一测试函数"""
    start = time.time()
    try:
        if client_type == "kimi":
            response = client.chat.completions.create(
                model=KIMI_MODEL,
                messages=messages,
                tools=TOOLS_OPENAI,
                tool_choice="auto",
                max_tokens=MAX_TOKENS,
            )
            elapsed = time.time() - start
            choice = response.choices[0]
            
            tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls.append({
                        "name": tc.function.name,
                        "arguments": tc.function.arguments[:300]
                    })
            
            content = choice.message.content or ""
            return {
                "model": model_name,
                "test": test_name,
                "finish_reason": choice.finish_reason,
                "has_tool_use": len(tool_calls) > 0,
                "tool_calls": tool_calls,
                "content_preview": content[:400],
                "content_is_empty": content.strip() == "",
                "elapsed_sec": round(elapsed, 3),
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "error": None
            }
        else:
            # Claude Converse
            system_prompt = [{"text": SYSTEM_PROMPT}]
            response = client.converse(
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
                        "arguments": json.dumps(tu.get("input", {}), ensure_ascii=False)[:300]
                    })
                if "text" in block:
                    text_content += block["text"]
            
            usage = response.get("usage", {})
            return {
                "model": model_name,
                "test": test_name,
                "finish_reason": stop_reason,
                "has_tool_use": len(tool_calls) > 0,
                "tool_calls": tool_calls,
                "content_preview": text_content[:400],
                "content_is_empty": text_content.strip() == "",
                "elapsed_sec": round(elapsed, 3),
                "prompt_tokens": usage.get("inputTokens", 0),
                "completion_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("inputTokens", 0) + usage.get("outputTokens", 0),
                "error": None
            }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "model": model_name,
            "test": test_name,
            "finish_reason": "ERROR",
            "has_tool_use": False,
            "tool_calls": [],
            "content_preview": "",
            "content_is_empty": True,
            "elapsed_sec": round(elapsed, 3),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": str(e)[:500]
        }


# ============================================================
# 找出所有需要测试的检查点
# ============================================================
def find_checkpoints(conversation):
    checkpoints = []
    for i, msg in enumerate(conversation):
        if msg.get("content") is None and msg.get("expected_tool"):
            checkpoints.append({
                "index": i,
                "expected_tool": msg["expected_tool"],
            })
    return checkpoints


def main():
    print("🚀 复现 Kimi K2.5 Tool Use 失败测试")
    print(f"   Kimi: {KIMI_MODEL}")
    print(f"   Claude: {CLAUDE_MODEL}")
    print(f"   目标: 复现 stopReason=end_turn + content 为空")
    print(f"   策略: 超长 system prompt + 大量多轮对话历史 → input tokens ~10k+")
    
    # Init
    api_key = get_bedrock_api_key()
    kimi_client = OpenAI(api_key=api_key, base_url=KIMI_BASE_URL)
    bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)
    
    checkpoints = find_checkpoints(CONVERSATION_HISTORY)
    print(f"\n   对话总消息数: {len(CONVERSATION_HISTORY)}")
    print(f"   检查点数量: {len(checkpoints)}")
    
    # 只测试最后几个检查点（上下文最长的）
    # 以及全部检查点，观察 token 增长和行为变化
    all_results = []
    
    for cp_idx, cp in enumerate(checkpoints):
        idx = cp["index"]
        expected = cp["expected_tool"]
        
        # Build messages up to this checkpoint
        conv_slice = CONVERSATION_HISTORY[:idx]
        kimi_messages = build_messages_for_kimi(conv_slice)
        claude_messages = build_messages_for_claude(conv_slice)
        
        test_name = f"cp{cp_idx+1}_{expected}"
        
        print(f"\n{'━'*80}")
        print(f"检查点 {cp_idx+1}/{len(checkpoints)}: {test_name}")
        print(f"  Kimi 消息数: {len(kimi_messages)} | Claude 消息数: {len(claude_messages)}")
        print(f"  期望 tool: {expected}")
        
        # Test Kimi
        print(f"\n  🌙 Kimi K2.5...")
        kimi_r = test_single("kimi", kimi_client, kimi_messages, test_name, "Kimi K2.5")
        all_results.append(kimi_r)
        
        icon = "✅" if kimi_r["has_tool_use"] else "❌"
        tool_name = kimi_r["tool_calls"][0]["name"] if kimi_r["tool_calls"] else "-"
        empty_flag = " 🔴EMPTY_CONTENT" if kimi_r["content_is_empty"] and not kimi_r["has_tool_use"] else ""
        print(f"     {icon} finish={kimi_r['finish_reason']}, tool={tool_name}, "
              f"tokens={kimi_r['prompt_tokens']}+{kimi_r['completion_tokens']}, "
              f"time={kimi_r['elapsed_sec']}s{empty_flag}")
        if not kimi_r["has_tool_use"]:
            if kimi_r["content_is_empty"]:
                print(f"     🔴🔴🔴 REPRODUCED! content is empty + no tool_use 🔴🔴🔴")
            else:
                print(f"     content: {kimi_r['content_preview'][:200]}")
        if kimi_r["error"]:
            print(f"     ❌ error: {kimi_r['error'][:200]}")
        
        time.sleep(1)
        
        # Test Claude
        if claude_messages and claude_messages[-1]["role"] == "user":
            print(f"\n  🤖 Claude Sonnet 4...")
            claude_r = test_single("claude", bedrock_client, claude_messages, test_name, "Claude Sonnet 4")
            all_results.append(claude_r)
            
            icon = "✅" if claude_r["has_tool_use"] else "❌"
            tool_name = claude_r["tool_calls"][0]["name"] if claude_r["tool_calls"] else "-"
            print(f"     {icon} finish={claude_r['finish_reason']}, tool={tool_name}, "
                  f"tokens={claude_r['prompt_tokens']}+{claude_r['completion_tokens']}, "
                  f"time={claude_r['elapsed_sec']}s")
            if claude_r["error"]:
                print(f"     ❌ error: {claude_r['error'][:200]}")
        else:
            print(f"\n  🤖 Claude: skipped (messages don't end with user)")
        
        time.sleep(1)
    
    # Summary
    print(f"\n{'━'*100}")
    print(f"📊 复现测试结果汇总")
    print(f"{'━'*100}")
    
    kimi_results = [r for r in all_results if r["model"] == "Kimi K2.5"]
    claude_results = [r for r in all_results if r["model"] == "Claude Sonnet 4"]
    
    print(f"\n{'检查点':<12} {'模型':<16} {'finish':<14} {'tool?':<8} {'empty?':<8} {'tool名':<22} {'prompt_tk':<10} {'耗时':<8}")
    print("-" * 100)
    for r in all_results:
        tool_name = r["tool_calls"][0]["name"] if r["tool_calls"] else "-"
        has_tool = "✅" if r["has_tool_use"] else "❌"
        empty = "🔴YES" if r["content_is_empty"] and not r["has_tool_use"] else "no"
        print(f"{r['test']:<12} {r['model']:<16} {r['finish_reason']:<14} {has_tool:<8} {empty:<8} {tool_name:<22} {r['prompt_tokens']:<10} {r['elapsed_sec']:<8}")
    
    # Kimi stats
    kimi_tool_ok = sum(1 for r in kimi_results if r["has_tool_use"])
    kimi_empty = sum(1 for r in kimi_results if r["content_is_empty"] and not r["has_tool_use"])
    print(f"\n📈 Kimi K2.5 汇总: {kimi_tool_ok}/{len(kimi_results)} 正确 tool_use, {kimi_empty} 次空 content")
    
    if kimi_empty > 0:
        print(f"\n🔴🔴🔴 成功复现 {kimi_empty} 次 'end_turn + 空 content' 问题！ 🔴🔴🔴")
    
    # Save
    output = {
        "config": {
            "region": REGION, "kimi_model": KIMI_MODEL, "claude_model": CLAUDE_MODEL,
            "max_tokens": MAX_TOKENS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "system_prompt_chars": len(SYSTEM_PROMPT),
            "conversation_messages": len(CONVERSATION_HISTORY),
        },
        "results": all_results
    }
    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bedrock_tool_use_reproduce_results.json")
    with open(outfile, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {outfile}")


if __name__ == "__main__":
    main()
