  Paste this content, then press Ctrl+D:

  # Resilient Trade Manager v1

  Resilient Trade Manager 是一个本地运行的 Bybit 实时监控与分析面板，支持图表、进阶分析和可选交易控制。支持实时数据
  流、多币种分析与高级图表覆盖层。

  ## 亮点

  - 实时与同步模式（REST + WebSocket）
  - 进阶分析（绩效、风险、maker 或 taker、稳定度）
  - 多币种历史记录筛选与导出
  - 持仓与活动订单显示，含 PnL 提示
  - 图表覆盖层（盈亏平衡、强平、活动订单、成交）
  - Tick 与 5 秒流式图表（来自实时或公开成交）
  - 交易控制（图表买卖、仓位控制）
  - 智能止盈止损（追价限价 + 拆分止损）
  - 中英双语界面
  - Resilient Lab 505 信息

  ## 快速开始

  1. 安装依赖：
     pip install -r requirements.txt

  2. 添加 Bybit 密钥到 .env（API 功能可选）：
     BYBIT_API_KEY=your_key
     BYBIT_API_SECRET=your_secret
     BYBIT_CATEGORY=linear

  3. 启动服务：
     python resilient_tm.py

  4. 打开浏览器：
     http://127.0.0.1:8010

  ## 说明

  - 交易功能需要拥有执行权限的 API Key。
  - 默认本地运行，不会上传数据。
  - Tick 与 5 秒图表依赖实时成交或公开成交流。

  ---

  Resilient Trade Manager 面板
  ============================

  包含同步与实时模式的交易分析与监控面板，提供绩效与风险指标以及实时持仓跟踪。

  功能
  ----
  - 同步模式：通过 Bybit REST 获取历史成交并分析
  - 实时模式：通过 WebSocket 监控成交、持仓与订单
  - 进阶分析：Sharpe、Calmar、Sortino、VaR、CVaR、maker 或 taker
  - 持仓监控：实时持仓与统计
  - 成交分析：按小时、按日、按币种统计

  快速启动
  --------
  1) 安装依赖：
     pip install -r requirements.txt

  2) 运行服务：
     python resilient_tm.py --data-root "/path/to/metrics-root"

  3) 打开：
     http://127.0.0.1:8010/

  模式
  ----
  1. 文件模式（默认）：读取本地指标文件
     - 读取：<data-root>/runtime/metrics.json
     - 适用于写入指标文件的产品

  2. 同步模式：历史数据分析
     - 用法：TRADE_OPTIMIZER_MODE=sync python resilient_tm.py

  3. 实时模式：WebSocket 实时监控
     - 用法：TRADE_OPTIMIZER_MODE=realtime python resilient_tm.py

  4. Bybit API 模式（兼容）：简单 REST 拉取
     - 用法：TRADE_OPTIMIZER_MODE=bybit_api python resilient_tm.py

  环境变量
  --------
  同步、实时或 bybit_api 模式必填：
  - BYBIT_API_KEY
  - BYBIT_API_SECRET

  可选：
  - BYBIT_CATEGORY：linear（默认）| spot | inverse
  - BYBIT_API_URL：https://api.bybit.com（默认）或测试网
  - TRADE_OPTIMIZER_MODE：file | sync | realtime | bybit_api
  - TRADE_OPTIMIZER_DATA_ROOT：数据目录路径（可替代 --data-root）

  说明
  ----
  - 文件模式需要有可访问的指标目录。如果没有写入方产品的数据根目录，请使用同步或实时模式。