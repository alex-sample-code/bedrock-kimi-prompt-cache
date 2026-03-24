#!/usr/bin/env python3
"""
Bedrock Tool Use Bug 复现: Kimi K2.5 返回 stop + 空 content
=============================================================

问题描述:
  Kimi K2.5 在 Bedrock Mantle 上，多轮对话包含 tool_call → tool_result 循环时，
  模型概率性返回 finish_reason=stop 且 content 为空字符串（无 tool_use、无文字）。

复现条件:
  - Input tokens ~9.5k-10k+
  - 多轮对话中包含 assistant(tool_calls) → tool(result) 的历史记录
  - 用户消息明确要求执行某个操作（应触发 tool call）

对比:
  Claude Sonnet 4 在相同场景下 100% 正确返回 tool_use。

运行:
  pip install openai boto3 aws-bedrock-token-generator
  python3 bedrock_tool_use_test.py
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
    """生成 Bedrock Mantle API Key"""
    from aws_bedrock_token_generator import BedrockTokenGenerator
    session = boto3.Session(region_name=REGION)
    credentials = session.get_credentials().get_frozen_credentials()
    generator = BedrockTokenGenerator()
    return generator.get_token(credentials=credentials, region=REGION)


# ============================================================
# 10 个 Tools (OpenAI function calling format)
# ============================================================
TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "list_products",
            "description": "列出店铺中的商品列表。支持多种筛选条件组合查询，包括按分类、状态、价格区间、库存范围、上架时间、标签等维度筛选。返回结果包含商品基本信息列表和分页元数据。注意：默认按更新时间倒序排列，每页最多返回50条记录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "商品分类筛选", "enum": ["电子产品", "智能家居", "时尚配件", "运动户外", "生活日用"]},
                    "status": {"type": "string", "description": "商品状态", "enum": ["active", "draft", "archived", "out_of_stock"]},
                    "keyword": {"type": "string", "description": "搜索关键词，匹配名称、SKU、描述"},
                    "min_price": {"type": "number", "description": "最低价格（元）"},
                    "max_price": {"type": "number", "description": "最高价格（元）"},
                    "page": {"type": "integer", "description": "页码，默认1"},
                    "page_size": {"type": "integer", "description": "每页数量，默认20，最大50"},
                    "sort_by": {"type": "string", "enum": ["created_at", "updated_at", "price", "inventory", "sales"]},
                    "sort_order": {"type": "string", "enum": ["asc", "desc"]}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_detail",
            "description": "根据SKU获取商品完整信息，包括基本信息、价格、库存、媒体、变体、统计数据等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "商品SKU编号，如SKU-001"},
                    "include_variants": {"type": "boolean", "description": "是否包含变体信息，默认true"},
                    "include_stats": {"type": "boolean", "description": "是否包含统计数据，默认true"}
                },
                "required": ["sku"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_product",
            "description": "创建新商品。必填：名称、价格、库存、分类。其他可选。创建后返回新SKU。名称不能重复，价格>0，设置compare_at_price时必须>price。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "商品名称，2-200字符"},
                    "description": {"type": "string", "description": "详细描述，支持HTML/Markdown"},
                    "price": {"type": "number", "description": "售价（元），>0"},
                    "compare_at_price": {"type": "number", "description": "划线价（元），必须>price"},
                    "cost_price": {"type": "number", "description": "成本价（元）"},
                    "inventory": {"type": "integer", "description": "初始库存"},
                    "category": {"type": "string", "description": "分类"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "标签列表"},
                    "images": {"type": "array", "items": {"type": "string"}, "description": "图片URL列表"},
                    "weight": {"type": "number", "description": "重量(kg)"},
                    "status": {"type": "string", "enum": ["active", "draft"], "description": "状态，默认draft"}
                },
                "required": ["name", "price", "inventory", "category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_product",
            "description": "更新商品基本信息（名称、描述、分类、标签等）。价格用set_product_price，库存用update_inventory。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "商品SKU"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "category": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "status": {"type": "string", "enum": ["active", "draft", "archived"]}
                },
                "required": ["sku"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_product",
            "description": "永久删除商品及关联数据，不可逆。有未完成订单时拒绝删除。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "商品SKU"},
                    "confirm": {"type": "boolean", "description": "必须为true"}
                },
                "required": ["sku", "confirm"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_inventory",
            "description": "更新库存。支持add(入库)、subtract(出库)、set(盘点设置)。每次变更记录操作日志。低于安全库存自动预警。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "商品SKU"},
                    "quantity": {"type": "integer", "description": "变更数量"},
                    "operation": {"type": "string", "enum": ["add", "subtract", "set"], "description": "操作类型"},
                    "reason": {"type": "string", "description": "变更原因"},
                    "warehouse": {"type": "string", "description": "仓库，默认main"}
                },
                "required": ["sku", "quantity", "operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_product_price",
            "description": "设置商品价格，支持售价、划线价、成本价。支持定时调价（设生效/结束时间）。价格变更记录历史。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "商品SKU"},
                    "price": {"type": "number", "description": "新售价（元）"},
                    "compare_at_price": {"type": "number", "description": "划线价（元）"},
                    "cost_price": {"type": "number", "description": "成本价（元）"},
                    "schedule_start": {"type": "string", "description": "定时生效，ISO 8601"},
                    "schedule_end": {"type": "string", "description": "定时结束，ISO 8601"}
                },
                "required": ["sku", "price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_product_images",
            "description": "为商品添加图片，支持批量，可指定排序位置和ALT文本。position=0为主图。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "商品SKU"},
                    "images": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "alt_text": {"type": "string"},
                                "position": {"type": "integer"}
                            },
                            "required": ["url"]
                        }
                    },
                    "replace_existing": {"type": "boolean", "description": "是否替换现有图片，默认false"}
                },
                "required": ["sku", "images"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_list",
            "description": "获取订单列表，支持按日期、状态、金额、客户、商品SKU等筛选。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD"},
                    "status": {"type": "string", "enum": ["pending", "paid", "shipped", "delivered", "cancelled", "refunded"]},
                    "min_amount": {"type": "number"},
                    "max_amount": {"type": "number"},
                    "customer_email": {"type": "string"},
                    "product_sku": {"type": "string"},
                    "page": {"type": "integer"},
                    "page_size": {"type": "integer"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sales_analytics",
            "description": "获取销售分析报表，支持按品类、商品、渠道、地区等维度交叉分析。指标包括销售额、订单量、客单价、转化率、退款率等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD"},
                    "granularity": {"type": "string", "enum": ["daily", "weekly", "monthly"]},
                    "dimensions": {"type": "array", "items": {"type": "string", "enum": ["category", "product", "channel", "region"]}},
                    "metrics": {"type": "array", "items": {"type": "string", "enum": ["revenue", "orders", "units_sold", "avg_order_value", "conversion_rate", "refund_rate"]}},
                    "top_n": {"type": "integer"}
                },
                "required": ["start_date", "end_date"]
            }
        }
    }
]


def convert_to_converse_tools(openai_tools):
    """转换为 Bedrock Converse toolConfig 格式"""
    return {"tools": [
        {"toolSpec": {"name": t["function"]["name"], "description": t["function"]["description"],
                       "inputSchema": {"json": t["function"]["parameters"]}}}
        for t in openai_tools
    ]}

TOOLS_CONVERSE = convert_to_converse_tools(TOOLS_OPENAI)

# ============================================================
# System Prompt - 独立站卖家助手 + 完整商品目录 (~5k tokens)
# ============================================================
SYSTEM_PROMPT = """# ShopMaster AI - 独立站智能卖家助手

## 角色
你是 ShopMaster AI，独立站（DTC电商）卖家助手，服务于"TechStyle数码生活馆"。职责：商品管理、库存管控、价格策略、订单跟踪、数据分析。

## 操作规范
1. 用户请求涉及数据操作时，必须调用对应工具函数执行
2. 删除商品需用户确认，大幅调价(>30%)需提示
3. 可选参数有合理默认值时直接使用，不追问

## 店铺信息
- 店铺：TechStyle 数码生活馆 | 主营：消费电子、智能家居
- 仓库：深圳主仓(15:00前当日发) / 义乌分仓(12:00前当日发)
- 物流：顺丰(快)、中通(经济)、京东(大件)、云途/燕文/4PX(跨境)

## 分类体系
- 电子产品：充电设备、电脑周边、音频设备、存储设备、智能穿戴
- 智能家居：智能灯具、智能安防、智能清洁、环境监测、智能控制
- 时尚配件：手机配件、数码包袋、桌面摆件、创意礼品
- 运动户外：运动音频、运动监测、户外装备、骑行配件
- 生活日用：收纳整理、厨房小工具、个人护理、旅行用品

## 当前商品目录（44个SKU）

### 电子产品（14个）
| SKU | 名称 | 售价 | 原价 | 成本 | 库存 | 月销 | 状态 |
|-----|------|------|------|------|------|------|------|
| SKU-001 | 65W氮化镓快充充电器（三口） | ¥99 | ¥159 | ¥42 | 2,340 | 1,856 | active |
| SKU-002 | USB-C转HDMI 4K拓展坞（7合1） | ¥189 | ¥299 | ¥78 | 567 | 423 | active |
| SKU-003 | 无线机械键盘RGB版 | ¥329 | ¥459 | ¥145 | 189 | 267 | active |
| SKU-004 | 2TB移动固态硬盘 USB3.2 | ¥599 | ¥799 | ¥320 | 445 | 312 | active |
| SKU-005 | 主动降噪蓝牙耳机 ANC Pro | ¥459 | ¥699 | ¥195 | 723 | 534 | active |
| SKU-006 | 电竞鼠标 16000DPI 轻量化 | ¥199 | ¥299 | ¥72 | 1,120 | 689 | active |
| SKU-007 | 磁吸无线充电板 15W | ¥79 | ¥129 | ¥28 | 3,200 | 2,100 | active |
| SKU-031 | 100W快充数据线 Type-C 2m | ¥29 | ¥49 | ¥8.5 | 12,000 | 5,600 | active |
| SKU-032 | 桌面无线充电座 3合1 | ¥169 | ¥259 | ¥68 | 890 | 456 | active |
| SKU-033 | 便携显示器 15.6寸 4K OLED | ¥1,899 | ¥2,599 | ¥980 | 78 | 34 | active |
| SKU-034 | 迷你投影仪 1080P 智能系统 | ¥899 | ¥1,299 | ¥420 | 156 | 89 | active |
| SKU-035 | 数码绘图板 10x6寸 8192级压感 | ¥259 | ¥399 | ¥110 | 345 | 178 | active |
| SKU-036 | USB麦克风 电容式 心形指向 | ¥189 | ¥299 | ¥75 | 567 | 234 | active |
| SKU-037 | 智能翻译笔 134种语言 | ¥699 | ¥999 | ¥340 | 234 | 67 | active |

### 智能家居（8个）
| SKU | 名称 | 售价 | 原价 | 成本 | 库存 | 月销 | 状态 |
|-----|------|------|------|------|------|------|------|
| SKU-008 | WiFi智能LED灯泡 RGBW | ¥39 | ¥69 | ¥12 | 5,600 | 3,400 | active |
| SKU-009 | 智能门铃摄像头 1080P | ¥299 | ¥499 | ¥135 | 234 | 178 | active |
| SKU-010 | 扫地机器人 激光导航 自动集尘 | ¥1,999 | ¥2,999 | ¥1,200 | 89 | 67 | active |
| SKU-011 | 智能温湿度传感器 WiFi | ¥49 | ¥79 | ¥15 | 4,300 | 1,890 | active |
| SKU-012 | WiFi智能插座 远程控制 | ¥29 | ¥49 | ¥9.5 | 8,900 | 5,600 | active |
| SKU-013 | 空气质量检测仪 PM2.5/甲醛 | ¥259 | ¥399 | ¥120 | 345 | 198 | active |
| SKU-038 | 智能门锁 指纹+密码+NFC | ¥1,299 | ¥1,899 | ¥580 | 123 | 45 | active |
| SKU-039 | WiFi智能窗帘电机 | ¥399 | ¥599 | ¥175 | 267 | 89 | active |

### 时尚配件（7个）
| SKU | 名称 | 售价 | 原价 | 成本 | 库存 | 月销 | 状态 |
|-----|------|------|------|------|------|------|------|
| SKU-014 | iPhone 16 Pro 液态硅胶壳 | ¥49 | ¥89 | ¥8 | 6,700 | 4,500 | active |
| SKU-015 | 多功能数码收纳包 防水 | ¥69 | ¥119 | ¥25 | 1,200 | 780 | active |
| SKU-016 | 实木显示器增高架 带收纳 | ¥159 | ¥249 | ¥65 | 456 | 234 | active |
| SKU-017 | 创意宇航员手机支架 | ¥39 | ¥69 | ¥11 | 3,400 | 2,100 | active |
| SKU-018 | LED氛围灯 16色遥控 | ¥59 | ¥99 | ¥18 | 2,800 | 1,670 | active |
| SKU-040 | 皮质笔记本内胆包 | ¥89 | ¥149 | ¥32 | 1,890 | 670 | active |
| SKU-041 | 磁吸MagSafe卡包+支架 | ¥59 | ¥99 | ¥18 | 2,340 | 1,230 | active |

### 运动户外（6个）
| SKU | 名称 | 售价 | 原价 | 成本 | 库存 | 月销 | 状态 |
|-----|------|------|------|------|------|------|------|
| SKU-019 | 骨传导运动耳机 IPX8 | ¥359 | ¥549 | ¥155 | 567 | 345 | active |
| SKU-020 | 智能运动手环 心率血氧 | ¥149 | ¥249 | ¥52 | 2,300 | 1,560 | active |
| SKU-021 | 户外蓝牙音箱 20W IPX7 | ¥199 | ¥329 | ¥78 | 890 | 456 | active |
| SKU-022 | 自行车尾灯 智能刹车感应 | ¥59 | ¥99 | ¥18 | 1,800 | 920 | active |
| SKU-023 | 运动腰包 弹力防水 | ¥39 | ¥69 | ¥12 | 4,500 | 2,800 | active |
| SKU-042 | 跑步手机臂包 旋转扣 | ¥35 | ¥59 | ¥10 | 3,200 | 1,890 | active |

### 生活日用（9个）
| SKU | 名称 | 售价 | 原价 | 成本 | 库存 | 月销 | 状态 |
|-----|------|------|------|------|------|------|------|
| SKU-024 | 桌面线材收纳盒 | ¥45 | ¥79 | ¥15 | 3,200 | 1,900 | active |
| SKU-025 | 便携榨汁杯 无线充电 | ¥89 | ¥149 | ¥35 | 1,560 | 980 | active |
| SKU-026 | 旅行分装瓶套装 硅胶 | ¥29 | ¥49 | ¥8 | 7,800 | 4,200 | active |
| SKU-027 | 电动牙刷 声波 5模式 | ¥129 | ¥199 | ¥48 | 2,100 | 1,340 | active |
| SKU-028 | 厨房电子秤 0.1g精度 | ¥35 | ¥59 | ¥12 | 4,500 | 2,670 | active |
| SKU-029 | 记忆棉护颈U型枕 | ¥69 | ¥109 | ¥25 | 1,800 | 1,120 | active |
| SKU-030 | USB暖手宝 10000mAh | ¥59 | ¥99 | ¥22 | 890 | 0 | draft |
| SKU-043 | 迷你桌面加湿器 300ml | ¥39 | ¥69 | ¥13 | 2,800 | 1,560 | active |
| SKU-044 | 电动开瓶器 红酒/啤酒 | ¥79 | ¥129 | ¥28 | 890 | 345 | active |


## 商品详细描述补充

### SKU-001 65W氮化镓快充充电器
采用第三代GaN氮化镓芯片，体积比传统充电器小60%。2个USB-C口+1个USB-A口，单口最大65W输出，支持PD3.0/QC4.0/PPS/AFC/FCP等全协议。折叠插脚设计，净重仅95g。兼容iPhone 16/15系列、iPad Pro、MacBook Air/Pro、三星Galaxy系列、华为Mate/P系列、Switch/Steam Deck等设备。内置智能温控芯片，过充/过流/过热/短路四重保护。通过FCC/CE/CCC认证，质保18个月。包装含：充电器x1、Type-C线1.5mx1、说明书x1。

### SKU-002 USB-C转HDMI 4K拓展坞（7合1）
全铝合金CNC一体成型外壳，重量仅68g。接口配置：HDMI 4K@60Hz x1、USB 3.0 x2（5Gbps）、USB-C PD直通充电100W x1、SD卡槽x1、TF卡槽x1、千兆RJ45网口x1。支持Mac/Windows/iPad OS/ChromeOS，即插即用免驱动。HDMI支持HDR10输出，兼容4K显示器和投影仪。网口支持Wake-on-LAN和PXE引导。散热孔+导热硅胶垫设计，长时间使用不掉速。附赠防尘收纳袋。

### SKU-005 主动降噪蓝牙耳机 ANC Pro
头戴式设计，40mm复合振膜单元，频响范围20Hz-40kHz。支持ANC主动降噪（-35dB）、通透模式、游戏低延迟模式（60ms）。蓝牙5.3，支持LDAC/AAC/SBC编码，多点连接（同时连2台设备）。可折叠旋转设计，蛋白皮耳罩+记忆海绵，佩戴舒适。续航：ANC开40小时、ANC关60小时，快充10分钟可听3小时。CVC8.0双麦通话降噪，内置环境声检测传感器。APP可自定义EQ、降噪等级、触控手势。随机附赠：3.5mm音频线、USB-C充电线、飞机转接头、收纳硬壳包。

### SKU-010 扫地机器人 激光导航 自动集尘
LDS激光雷达导航，SLAM算法实时建图，精度达2cm。4000Pa大吸力，3档可调。自动集尘底座，2.5L大容量尘袋，约60天免倒。250ml电控水箱，3档出水量微调，Y字形湿拖路线。支持多层地图记忆（最多4层楼），虚拟墙/禁区设置。APP远程控制，支持天猫精灵/小爱同学语音控制。5200mAh电池，续航约180分钟（标准模式），断点续扫。越障能力2cm，防跌落传感器x4。主刷+边刷双重清扫系统，配HEPA滤网。噪音：标准模式55dB、强力模式65dB。尺寸：350x350x98mm，净重3.6kg（含集尘座6.2kg）。

### SKU-033 便携显示器 15.6寸 4K OLED
15.6英寸4K UHD（3840x2160）OLED面板，100% DCI-P3色域，100000:1对比度，0.1ms响应时间。支持HDR10，峰值亮度500nit。USB-C一线直连（支持DP Alt Mode），兼容笔记本/手机/Switch/PS5/Xbox。VESA 75x75支架孔，可壁挂。内置双1W立体声扬声器。Mini HDMI + USB-C x2 接口配置，附赠保护套、USB-C线x2、Mini HDMI线x1、支架。重量680g、厚度5.2mm（最薄处）。色准ΔE<1，适合设计师和摄影师外出校色使用。

## 运营策略与竞品分析

### 定价策略
- 引流品（<50元）：SKU-012/008/026/028/031 —— 高转化、低客单，吸引新客
- 利润品（50-200元）：SKU-001/005/015/020/025 —— 主力利润来源，毛利率55-67%
- 形象品（>500元）：SKU-004/010/033/037/038 —— 提升品牌调性，转化率低但毛利绝对值高
- 新品孵化期策略：首月亏损推广→第2月平本→第3月盈利

### 各品类竞争态势
1. **充电类（SKU-001/007/031/032）**：红海市场，竞争激烈。Anker/倍思/小米为主要竞品。我们优势：性价比+独立站专属售后。
2. **智能家居（SKU-008-013/038-039）**：增长快，米家生态强势但独立站有差异化空间。我们优势：不绑定特定生态，兼容多平台。
3. **手机配件（SKU-014/017/041）**：SKU少但走量大。季节性弱，全年稳定。核心竞争力：设计感+材质。
4. **运动户外（SKU-019-023/042）**：季节性强（春夏旺季），客单价较高。拓展方向：新增露营品类。

### 用户画像
| 维度 | 特征 |
|------|------|
| 年龄 | 25-35岁为主(52%)、18-24(18%)、36-45(22%)、45+(8%) |
| 性别 | 男性58%、女性42% |
| 地域 | 一线城市32%、新一线28%、二线24%、三线及以下16% |
| 消费偏好 | 追求性价比(45%)、品质优先(35%)、新品尝鲜(20%) |
| 复购周期 | 首次→二次平均45天 | 活跃用户平均30天复购 |
| 客单价分布 | <50元(35%)、50-150(38%)、150-500(20%)、>500(7%) |

### 流量来源分析
| 渠道 | UV占比 | 转化率 | ROI | 趋势 |
|------|--------|--------|-----|------|
| 自然搜索(SEO) | 28% | 3.2% | - | ↑ 持续优化中 |
| 直接访问 | 15% | 5.8% | - | → 品牌认知稳定 |
| 付费搜索(SEM) | 22% | 3.5% | 4.2 | → 维持 |
| 信息流广告 | 18% | 2.1% | 2.8 | ↓ 需优化素材 |
| 社交媒体 | 12% | 4.5% | 3.5 | ↑ 小红书增长快 |
| EDM邮件 | 5% | 6.2% | 8.1 | ↑ 高ROI渠道 |

### 退货原因TOP5（本月）
1. 与描述不符/色差（32%）—— 主要是SKU-014手机壳
2. 质量问题/瑕疵（24%）—— 需加强QC
3. 不想要了/冲动消费（18%）
4. 尺寸/规格不合适（15%）
5. 物流损坏（11%）—— 改善包装

### 客服SOP摘要
- 响应时效：首次响应<2分钟，解决时效<24小时
- 退换货政策：7天无理由，15天质量问题换新，1年保修
- 补偿权限：<10元优惠券客服直接发，10-50元需组长审批，>50元需经理审批
- 差评处理流程：24小时内联系→了解原因→提供方案→跟进满意度→更新FAQ

## 技术架构
- 前端：Next.js 14 + TailwindCSS，部署在 Vercel
- 后端：Node.js + PostgreSQL + Redis，部署在 AWS（ap-southeast-1）
- 支付：Stripe(国际) + 微信支付/支付宝(国内)
- CDN：CloudFront，图片存储 S3
- 监控：CloudWatch + Sentry，日志 CloudWatch Logs
- 数据分析：自建 BI（Metabase）+ Google Analytics 4

## 合规要求
- 跨境电商：海关申报要素（品名、材质、用途、品牌）
- 含电池商品（SKU-001/004/007/020/022/025/030/031/032）需要UN38.3报告和MSDS
- 3C认证：充电器类必须有CCC认证
- EMC/RF：蓝牙/WiFi设备需要SRRC型号核准


## 本月每日GMV趋势
| 日期 | GMV(¥) | 订单数 | UV | 转化率 | 备注 |
|------|--------|--------|-----|--------|------|
| 3/1 | 98,450 | 652 | 16,230 | 4.02% | |
| 3/2 | 112,300 | 745 | 18,450 | 4.04% | 周六 |
| 3/3 | 125,670 | 834 | 20,100 | 4.15% | 周日 |
| 3/4 | 95,230 | 631 | 15,890 | 3.97% | |
| 3/5 | 102,450 | 678 | 16,780 | 4.04% | |
| 3/6 | 108,900 | 722 | 17,340 | 4.16% | |
| 3/7 | 115,200 | 764 | 18,670 | 4.09% | |
| 3/8 | 156,780 | 1,042 | 24,560 | 4.24% | 妇女节 |
| 3/9 | 142,300 | 945 | 22,340 | 4.23% | |
| 3/10 | 98,670 | 654 | 16,450 | 3.98% | |
| 3/11 | 105,340 | 699 | 17,120 | 4.08% | |
| 3/12 | 112,560 | 747 | 18,230 | 4.10% | |
| 3/13 | 118,900 | 789 | 19,120 | 4.13% | |
| 3/14 | 135,670 | 900 | 21,340 | 4.22% | 周末 |
| 3/15 | 148,230 | 984 | 23,100 | 4.26% | 周末 |
| 3/16 | 102,340 | 679 | 16,890 | 4.02% | |
| 3/17 | 108,900 | 722 | 17,560 | 4.11% | |
| 3/18 | 115,670 | 767 | 18,900 | 4.06% | |
| 3/19 | 122,340 | 812 | 19,670 | 4.13% | |
| 3/20 | 128,900 | 856 | 20,890 | 4.10% | |
| 3/21 | 145,670 | 967 | 23,120 | 4.18% | 周末 |
| 3/22 | 152,340 | 1,012 | 24,560 | 4.12% | 周末 |

## 各SKU本周销售排行（3/16-3/22）
| 排名 | SKU | 商品名称 | 周销量 | 周销售额 | 环比 |
|------|-----|---------|--------|---------|------|
| 1 | SKU-012 | WiFi智能插座 | 1,420 | ¥41,180 | +8% |
| 2 | SKU-031 | 100W快充数据线 | 1,380 | ¥40,020 | +5% |
| 3 | SKU-014 | iPhone手机壳 | 1,125 | ¥55,125 | +3% |
| 4 | SKU-026 | 旅行分装瓶 | 1,050 | ¥30,450 | +12% |
| 5 | SKU-008 | 智能LED灯泡 | 890 | ¥34,710 | +6% |
| 6 | SKU-023 | 运动腰包 | 720 | ¥28,080 | +15% |
| 7 | SKU-028 | 厨房电子秤 | 680 | ¥23,800 | +4% |
| 8 | SKU-017 | 宇航员手机支架 | 560 | ¥21,840 | -2% |
| 9 | SKU-007 | 磁吸无线充电板 | 540 | ¥42,660 | +7% |
| 10 | SKU-042 | 跑步手机臂包 | 490 | ¥17,150 | +18% |
| 11 | SKU-020 | 智能运动手环 | 420 | ¥62,580 | +9% |
| 12 | SKU-011 | 温湿度传感器 | 480 | ¥23,520 | +3% |
| 13 | SKU-001 | 65W快充充电器 | 465 | ¥46,035 | +6% |
| 14 | SKU-041 | MagSafe卡包 | 340 | ¥20,060 | +11% |
| 15 | SKU-018 | LED氛围灯 | 420 | ¥24,780 | -1% |
| 16 | SKU-043 | 桌面加湿器 | 410 | ¥15,990 | +8% |
| 17 | SKU-024 | 线材收纳盒 | 480 | ¥21,600 | +2% |
| 18 | SKU-006 | 电竞鼠标 | 175 | ¥34,825 | +4% |
| 19 | SKU-027 | 电动牙刷 | 340 | ¥43,860 | +7% |
| 20 | SKU-029 | 护颈U型枕 | 290 | ¥20,010 | +5% |

## 客户评价精选（近7天）
- SKU-001: "充电速度很快，三个口同时充也没问题，出差必带" ★★★★★
- SKU-005: "降噪效果超出预期，音质也不错，就是夹头稍紧" ★★★★☆
- SKU-010: "地图规划很智能，自动集尘太方便了，猫毛克星" ★★★★★
- SKU-012: "29块钱太值了，APP操作很方便，买了5个全屋用" ★★★★★
- SKU-014: "手感好，不发黄，但颜色比图片深一点" ★★★★☆
- SKU-020: "心率监测准确，续航14天没问题，性价比之王" ★★★★★
- SKU-033: "OLED屏幕太震撼了，设计用完全够，就是价格贵" ★★★★☆
- SKU-038: "指纹识别很快，NFC解锁也方便，安装服务到位" ★★★★★


## 供应链管理
### 核心供应商
| 供应商 | 品类 | 合作方式 | 账期 | 备注 |
|--------|------|---------|------|------|
| 深圳华强电子 | 充电器/数据线 | OEM定制 | 月结30天 | SKU-001/007/031/032 |
| 东莞智联科技 | 蓝牙音频 | ODM贴牌 | 月结45天 | SKU-005/019/021 |
| 佛山美家智能 | 智能家居 | 代理分销 | 款到发货 | SKU-008-013/038-039 |
| 义乌创意工坊 | 配件/礼品 | 现货采购 | 现结 | SKU-014/017/018 |
| 南京存储科技 | 存储设备 | 授权分销 | 月结30天 | SKU-004 |
| 苏州机器人科技 | 扫地机器人 | 独家代理 | 月结60天 | SKU-010 |

### 采购周期
- 电子产品：下单后7-15天到货（定制款15-25天）
- 智能家居：现货3-5天，定制15-20天
- 配件类：现货1-3天
- 大件/高价值：需提前30天预订

### 质检流程
1. 来料检验（IQC）：外观、功能、安全性抽检，AQL 2.5
2. 过程检验（IPQC）：贴标、包装、配件齐全性
3. 出货检验（OQC）：随机抽样功能测试，批次追溯码
4. 客诉追溯：批次号→供应商→质量改进闭环

## 售后处理规范
### 退换货流程
1. 客户申请退换货（APP/客服）
2. 客服审核（1小时内）→ 同意/拒绝/需补充信息
3. 客户寄回（提供退货地址和运费说明）
4. 仓库验收（1个工作日）→ 质检结果记录
5. 退款/换货处理（验收通过后24小时内）
6. 回访客户满意度

### 常见问题处理预案
| 问题类型 | 处理方式 | 补偿标准 |
|---------|---------|---------|
| 发错货 | 补发正确商品+上门取件 | 赠10元优惠券 |
| 商品损坏 | 换新或退款 | 视情况赠优惠券 |
| 物流延迟>3天 | 主动联系+道歉 | 赠5元优惠券 |
| 质量投诉 | 退换货+记录供应商问题 | 赠15元优惠券 |
| 差评回复 | 24小时内联系+解决方案 | 个案处理 |

## 运营概览
- 本月GMV: ¥2,847,560 (+23% YoY) | 订单: 18,934 | 客单价: ¥150.4
- 转化率: 4.14% | 退货率: 3.2% | 复购率: 28%
- 热销TOP3: SKU-012(5,600) > SKU-031(5,600) > SKU-014(4,500)
- 库存预警: SKU-010(89台/40天) SKU-033(78台/71天) SKU-009(234台/40天)
- 近期促销: 3/25-31春季焕新季(满200-30) | 4/1-7清明户外节(运动8折)
"""


# ============================================================
# 多轮对话历史 - 包含 tool_call → tool_result 循环
# 模拟: 卖家先问运营情况 → 补库存(3个SKU) → 查商品详情 → 调促销价
# ============================================================
CONVERSATION = [
    # --- Round 1: 用户问运营情况 ---
    {"role": "user", "content": "你好，我是店铺管理员，帮我看看现在店铺的整体运营情况，重点关注库存预警和即将到来的促销活动"},
    {"role": "assistant", "content": """好的！TechStyle数码生活馆运营概览：

📊 本月业绩（截至3月22日）
- GMV：¥2,847,560（+23%）| 订单：18,934 | 客单价：¥150.4
- 转化率：4.14%（略降-0.3pp）| 退货率：3.2% | 复购率：28%

⚠️ 库存预警
- SKU-010 扫地机器人 89台（仅40天库存）
- SKU-033 便携显示器 78台
- SKU-009 智能门铃 234台（40天库存）

🔜 促销日历
- 春季焕新季 3/25-31：满200减30，满500减80（后天开始！）
- 清明户外节 4/1-7：运动户外8折

🏆 热销TOP3：SKU-012智能插座(5,600) / SKU-031数据线(5,600) / SKU-014手机壳(4,500)

需要先处理哪方面？"""},

    # --- Round 2: 用户要求批量补库存 + 查商品详情 ---
    {"role": "user", "content": "先处理库存。扫地机器人补500台，智能门铃补300台，便携显示器补100台。然后帮我查一下SKU-012智能插座的详情，春季促销我想重点推。"},
    {"role": "assistant", "content": "好的，先补库存。处理扫地机器人 SKU-010："},

    # --- Tool call 1: update_inventory SKU-010 ---
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_001", "type": "function", "function": {"name": "update_inventory", "arguments": "{\"sku\":\"SKU-010\",\"quantity\":500,\"operation\":\"add\",\"reason\":\"采购入库\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_001", "content": "{\"success\":true,\"sku\":\"SKU-010\",\"previous\":89,\"new\":589,\"operation\":\"add\",\"quantity\":500}"},
    {"role": "assistant", "content": "✅ SKU-010 扫地机器人：89 → 589台（+500）\n\n接下来补智能门铃 SKU-009："},

    # --- Tool call 2: update_inventory SKU-009 ---
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_002", "type": "function", "function": {"name": "update_inventory", "arguments": "{\"sku\":\"SKU-009\",\"quantity\":300,\"operation\":\"add\",\"reason\":\"采购入库\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_002", "content": "{\"success\":true,\"sku\":\"SKU-009\",\"previous\":234,\"new\":534,\"operation\":\"add\",\"quantity\":300}"},
    {"role": "assistant", "content": "✅ SKU-009 智能门铃：234 → 534台（+300）\n\n最后补便携显示器 SKU-033："},

    # --- Tool call 3: update_inventory SKU-033 ---
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_003", "type": "function", "function": {"name": "update_inventory", "arguments": "{\"sku\":\"SKU-033\",\"quantity\":100,\"operation\":\"add\",\"reason\":\"采购入库\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_003", "content": "{\"success\":true,\"sku\":\"SKU-033\",\"previous\":78,\"new\":178,\"operation\":\"add\",\"quantity\":100}"},
    {"role": "assistant", "content": """✅ SKU-033 便携显示器：78 → 178台（+100）

📦 库存补充完成：
| 商品 | 补前→补后 | 预计可售 |
|------|---------|---------|
| SKU-010 扫地机器人 | 89→589 | ~268天 |
| SKU-009 智能门铃 | 234→534 | ~90天 |
| SKU-033 便携显示器 | 78→178 | ~162天 |

现在查看 SKU-012 智能插座详情："""},

    # --- Tool call 4: get_product_detail SKU-012 ---
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_004", "type": "function", "function": {"name": "get_product_detail", "arguments": "{\"sku\":\"SKU-012\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_004", "content": "{\"sku\":\"SKU-012\",\"name\":\"WiFi智能插座 远程控制 定时开关\",\"price\":29.00,\"compare_at_price\":49.00,\"cost_price\":9.50,\"inventory\":8900,\"monthly_sales\":5600,\"daily_avg\":186,\"rating\":4.7,\"reviews\":8934,\"conversion_rate\":6.8,\"add_to_cart_rate\":12.3}"},
    {"role": "assistant", "content": """📦 SKU-012 WiFi智能插座详情：
- 售价：¥29 | 原价：¥49 | 成本：¥9.50 | 毛利率67%
- 月销：5,600（全店第一）| 日均186 | 库存8,900（48天）
- 评分：4.7/5（8,934评价）| 转化率6.8% | 加购率12.3%

💡 非常适合做促销引流：毛利率高(67%)，转化好，销量第一。
建议活动价¥24.9（毛利仍有¥15.4，毛利率62%）。需要调价吗？"""},

    # --- Round 5: 用户决定促销方案，要求改4个商品价格 ---
    {"role": "user", "content": "好，春季焕新季促销安排如下：\n1. SKU-012智能插座 活动价24.9\n2. SKU-008智能灯泡 活动价29.9\n3. SKU-014手机壳 活动价39.9\n4. SKU-007无线充电板 活动价59.9\n全部3月25日生效，3月31日结束恢复原价。帮我改。"},
    {"role": "assistant", "content": """好的，4个商品促销价确认：
| SKU | 商品 | 原售价→活动价 | 活动毛利率 |
|-----|------|-------------|-----------|
| SKU-012 | 智能插座 | ¥29→¥24.9 | 62% |
| SKU-008 | 智能灯泡 | ¥39→¥29.9 | 60% |
| SKU-014 | 手机壳 | ¥49→¥39.9 | 80% |
| SKU-007 | 无线充电板 | ¥79→¥59.9 | 52% |

开始设置，先改 SKU-012："""},

    # --- Tool call 5: set_product_price SKU-012 ---
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_005", "type": "function", "function": {"name": "set_product_price", "arguments": "{\"sku\":\"SKU-012\",\"price\":24.9,\"schedule_start\":\"2026-03-25T00:00:00+08:00\",\"schedule_end\":\"2026-03-31T23:59:59+08:00\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_005", "content": "{\"success\":true,\"sku\":\"SKU-012\",\"previous_price\":29.00,\"new_price\":24.9}"},
    {"role": "assistant", "content": "✅ SKU-012 智能插座：¥29→¥24.9（3/25-3/31）\n\n设置 SKU-008："},

    # --- Tool call 6: set_product_price SKU-008 ---
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_006", "type": "function", "function": {"name": "set_product_price", "arguments": "{\"sku\":\"SKU-008\",\"price\":29.9,\"schedule_start\":\"2026-03-25T00:00:00+08:00\",\"schedule_end\":\"2026-03-31T23:59:59+08:00\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_006", "content": "{\"success\":true,\"sku\":\"SKU-008\",\"previous_price\":39.00,\"new_price\":29.9}"},
    {"role": "assistant", "content": "✅ SKU-008 智能灯泡：¥39→¥29.9（3/25-3/31）\n\n设置 SKU-014："},

    # --- Tool call 7: set_product_price SKU-014 ---
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_007", "type": "function", "function": {"name": "set_product_price", "arguments": "{\"sku\":\"SKU-014\",\"price\":39.9,\"schedule_start\":\"2026-03-25T00:00:00+08:00\",\"schedule_end\":\"2026-03-31T23:59:59+08:00\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_007", "content": "{\"success\":true,\"sku\":\"SKU-014\",\"previous_price\":49.00,\"new_price\":39.9}"},
    {"role": "assistant", "content": "✅ SKU-014 手机壳：¥49→¥39.9（3/25-3/31）\n\n最后设置 SKU-007："},

    # --- Tool call 8: set_product_price SKU-007 ---
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_008", "type": "function", "function": {"name": "set_product_price", "arguments": "{\"sku\":\"SKU-007\",\"price\":59.9,\"schedule_start\":\"2026-03-25T00:00:00+08:00\",\"schedule_end\":\"2026-03-31T23:59:59+08:00\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_008", "content": "{\"success\":true,\"sku\":\"SKU-007\",\"previous_price\":79.00,\"new_price\":59.9}"},
    {"role": "assistant", "content": """✅ SKU-007 无线充电板：¥79→¥59.9（3/25-3/31）

📋 春季焕新季促销价格全部设置完成！
| SKU | 商品 | 活动价 | 时间 | 状态 |
|-----|------|--------|------|------|
| SKU-012 | 智能插座 | ¥24.9 | 3/25-3/31 | ✅ |
| SKU-008 | 智能灯泡 | ¥29.9 | 3/25-3/31 | ✅ |
| SKU-014 | 手机壳 | ¥39.9 | 3/25-3/31 | ✅ |
| SKU-007 | 无线充电板 | ¥59.9 | 3/25-3/31 | ✅ |

还需要做什么？"""},

    # --- Round 8: 最终测试 - 在超长上下文后要求创建新商品 (应触发 create_product) ---
    {"role": "user", "content": "对了，我昨天进了一批新货。帮我创建一个新商品：智能体脂秤，蓝牙连APP，13项身体数据检测，钢化玻璃面板，价格89，成本35，库存2000，分类生活日用，直接上架。"},
]

# 最后一条 user message 期望触发 create_product
EXPECTED_TOOL = "create_product"


# ============================================================
# 构建消息列表
# ============================================================
def build_kimi_messages():
    """构建 OpenAI 格式消息（含 system prompt + 完整 tool call 历史）"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in CONVERSATION:
        if msg.get("tool_calls"):
            # assistant with tool_calls (no text content)
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": msg["tool_calls"]
            })
        elif msg["role"] == "tool":
            messages.append({
                "role": "tool",
                "tool_call_id": msg["tool_call_id"],
                "content": msg["content"]
            })
        else:
            messages.append({"role": msg["role"], "content": msg["content"]})
    return messages


def build_claude_messages():
    """构建 Bedrock Converse 格式消息（简化：跳过 tool 交互，只保留文本轮次）"""
    messages = []
    for msg in CONVERSATION:
        if msg.get("tool_calls") or msg["role"] == "tool":
            continue
        if msg["content"] is None:
            continue
        role = msg["role"]
        entry = {"role": role, "content": [{"text": msg["content"]}]}
        # Merge consecutive same-role
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(entry["content"])
        else:
            messages.append(entry)
    # Ensure starts with user
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return messages


# ============================================================
# 测试函数
# ============================================================
def test_kimi(client, messages, run_id):
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=KIMI_MODEL, messages=messages,
            tools=TOOLS_OPENAI, tool_choice="auto", max_tokens=MAX_TOKENS,
        )
        elapsed = time.time() - start
        choice = response.choices[0]
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({"name": tc.function.name, "arguments": tc.function.arguments[:300]})
        content = choice.message.content or ""
        return {
            "run": run_id, "model": "Kimi K2.5",
            "finish_reason": choice.finish_reason,
            "has_tool_use": len(tool_calls) > 0,
            "tool_calls": tool_calls,
            "content": content[:500],
            "content_is_empty": content.strip() == "",
            "elapsed_sec": round(elapsed, 3),
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
    except Exception as e:
        return {
            "run": run_id, "model": "Kimi K2.5",
            "finish_reason": "ERROR", "has_tool_use": False, "tool_calls": [],
            "content": "", "content_is_empty": True,
            "elapsed_sec": round(time.time() - start, 3),
            "prompt_tokens": 0, "completion_tokens": 0,
            "error": str(e)[:300]
        }


def test_claude(client, messages, run_id):
    start = time.time()
    try:
        response = client.converse(
            modelId=CLAUDE_MODEL, messages=messages,
            system=[{"text": SYSTEM_PROMPT}],
            toolConfig=TOOLS_CONVERSE,
            inferenceConfig={"maxTokens": MAX_TOKENS}
        )
        elapsed = time.time() - start
        stop = response.get("stopReason", "unknown")
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        tool_calls, text = [], ""
        for b in blocks:
            if "toolUse" in b:
                tool_calls.append({"name": b["toolUse"]["name"],
                                    "arguments": json.dumps(b["toolUse"].get("input", {}), ensure_ascii=False)[:300]})
            if "text" in b:
                text += b["text"]
        usage = response.get("usage", {})
        return {
            "run": run_id, "model": "Claude Sonnet 4",
            "finish_reason": stop,
            "has_tool_use": len(tool_calls) > 0,
            "tool_calls": tool_calls,
            "content": text[:500],
            "content_is_empty": text.strip() == "",
            "elapsed_sec": round(elapsed, 3),
            "prompt_tokens": usage.get("inputTokens", 0),
            "completion_tokens": usage.get("outputTokens", 0),
        }
    except Exception as e:
        return {
            "run": run_id, "model": "Claude Sonnet 4",
            "finish_reason": "ERROR", "has_tool_use": False, "tool_calls": [],
            "content": "", "content_is_empty": True,
            "elapsed_sec": round(time.time() - start, 3),
            "prompt_tokens": 0, "completion_tokens": 0,
            "error": str(e)[:300]
        }


# ============================================================
# Main - 跑多次以验证概率性复现
# ============================================================
RUNS = 5  # 跑5次观察概率

def main():
    print("=" * 80)
    print("🔴 Kimi K2.5 Tool Use Bug 复现测试")
    print("=" * 80)
    print(f"模型: {KIMI_MODEL} vs {CLAUDE_MODEL}")
    print(f"场景: 独立站卖家助手，多轮对话含8轮 tool_call→tool_result")
    print(f"最终指令: 创建新商品（期望触发 {EXPECTED_TOOL}）")
    print(f"测试轮数: {RUNS} 次（验证概率性）")

    api_key = get_bedrock_api_key()
    kimi_client = OpenAI(api_key=api_key, base_url=KIMI_BASE_URL)
    bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)

    kimi_messages = build_kimi_messages()
    claude_messages = build_claude_messages()

    print(f"\nKimi 消息数: {len(kimi_messages)} | Claude 消息数: {len(claude_messages)}")

    all_results = []
    kimi_empty_count = 0
    kimi_no_tool_count = 0

    for run in range(1, RUNS + 1):
        print(f"\n{'─'*60}")
        print(f"Run {run}/{RUNS}")

        # Kimi
        kr = test_kimi(kimi_client, kimi_messages, run)
        all_results.append(kr)
        tool_name = kr["tool_calls"][0]["name"] if kr["tool_calls"] else "-"
        icon = "✅" if kr["has_tool_use"] else "❌"
        empty_flag = ""
        if kr["content_is_empty"] and not kr["has_tool_use"]:
            empty_flag = " 🔴 EMPTY CONTENT"
            kimi_empty_count += 1
        if not kr["has_tool_use"]:
            kimi_no_tool_count += 1

        print(f"  Kimi:   {icon} finish={kr['finish_reason']}, tool={tool_name}, "
              f"tokens={kr['prompt_tokens']}in+{kr['completion_tokens']}out, "
              f"{kr['elapsed_sec']}s{empty_flag}")
        if not kr["has_tool_use"] and kr["content"] and not kr["content_is_empty"]:
            print(f"          content: {kr['content'][:150]}...")
        if kr.get("error"):
            print(f"          error: {kr['error'][:150]}")

        time.sleep(1)

        # Claude
        cr = test_claude(bedrock_client, claude_messages, run)
        all_results.append(cr)
        tool_name = cr["tool_calls"][0]["name"] if cr["tool_calls"] else "-"
        icon = "✅" if cr["has_tool_use"] else "❌"
        print(f"  Claude: {icon} finish={cr['finish_reason']}, tool={tool_name}, "
              f"tokens={cr['prompt_tokens']}in+{cr['completion_tokens']}out, "
              f"{cr['elapsed_sec']}s")
        if cr.get("error"):
            print(f"          error: {cr['error'][:150]}")

        time.sleep(1)

    # Summary
    print(f"\n{'='*80}")
    print(f"📊 结果汇总 ({RUNS} 次)")
    print(f"{'='*80}")

    kimi_results = [r for r in all_results if r["model"] == "Kimi K2.5"]
    claude_results = [r for r in all_results if r["model"] == "Claude Sonnet 4"]

    print(f"\n{'Run':<6} {'模型':<16} {'finish_reason':<14} {'tool_use':<10} {'空content':<10} {'tool':<20} {'prompt_tk':<10}")
    print("-" * 80)
    for r in all_results:
        tool = r["tool_calls"][0]["name"] if r["tool_calls"] else "-"
        has = "✅" if r["has_tool_use"] else "❌"
        empty = "🔴 YES" if r["content_is_empty"] and not r["has_tool_use"] else "no"
        print(f"{r['run']:<6} {r['model']:<16} {r['finish_reason']:<14} {has:<10} {empty:<10} {tool:<20} {r['prompt_tokens']:<10}")

    kimi_ok = sum(1 for r in kimi_results if r["has_tool_use"])
    claude_ok = sum(1 for r in claude_results if r["has_tool_use"])

    print(f"\n📈 Kimi K2.5:      {kimi_ok}/{RUNS} 次正确 tool_use | {kimi_empty_count} 次 stop+空content | {kimi_no_tool_count} 次未调 tool")
    print(f"📈 Claude Sonnet4: {claude_ok}/{RUNS} 次正确 tool_use")

    if kimi_empty_count > 0:
        print(f"\n🔴🔴🔴 复现成功: {kimi_empty_count}/{RUNS} 次出现 'stop + 空 content' 🔴🔴🔴")
    elif kimi_no_tool_count > 0:
        print(f"\n⚠️  未复现空content，但有 {kimi_no_tool_count} 次未调tool（返回了文字）")
    else:
        print(f"\n✅ 本次 {RUNS} 轮全部正确（问题具有概率性，可增加 RUNS 重试）")

    # Save
    output = {
        "config": {
            "region": REGION, "kimi_model": KIMI_MODEL, "claude_model": CLAUDE_MODEL,
            "max_tokens": MAX_TOKENS, "runs": RUNS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "expected_tool": EXPECTED_TOOL,
        },
        "results": all_results
    }
    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bedrock_tool_use_results.json")
    with open(outfile, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果: {outfile}")


if __name__ == "__main__":
    main()
