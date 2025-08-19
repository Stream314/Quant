from bigmodule import M

# <aistudiograph>

# @param(id="m5", name="initialize")
# 交易引擎：初始化函数，只执行一次
def m5_initialize_bigquant_run(context):
    from bigtrader.finance.commission import PerOrder

    # 系统已经设置了默认的交易手续费和滑点，要修改手续费可使用如下函数
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))

# @param(id="m5", name="before_trading_start")
# 交易引擎：每个单位时间开盘前调用一次。
def m5_before_trading_start_bigquant_run(context, data):
    # 盘前处理，订阅行情等
    pass

# @param(id="m5", name="handle_tick")
# 交易引擎：tick数据处理函数，每个tick执行一次
def m5_handle_tick_bigquant_run(context, tick):
    pass

# @param(id="m5", name="handle_data")
def m5_handle_data_bigquant_run(context, data):
    import pandas as pd

    # 下一个交易日不是调仓日，则不生成信号
    if not context.rebalance_period.is_signal_date(data.current_dt.date()):
        return

    # 从传入的数据 context.data 中读取今天的信号数据
    today_df = context.data[context.data["date"] == data.current_dt.strftime("%Y-%m-%d")]
    target_instruments = set(today_df["instrument"])

    # 获取当前已持有股票
    holding_instruments = set(context.get_account_positions().keys())

    # 卖出不在目标持有列表中的股票
    for instrument in holding_instruments - target_instruments:
        context.order_target_percent(instrument, 0)
        
    # 买入目标持有列表中的股票
    for i, x in today_df.iterrows():
        # 处理 null 或者 decimal.Decimal 类型等
        position = 0.0 if pd.isnull(x.position) else float(x.position)
        context.order_target_percent(x.instrument, position)

# @param(id="m5", name="handle_trade")
# 交易引擎：成交回报处理函数，每个成交发生时执行一次
def m5_handle_trade_bigquant_run(context, trade):
    pass

# @param(id="m5", name="handle_order")
# 交易引擎：委托回报处理函数，每个委托变化时执行一次
def m5_handle_order_bigquant_run(context, order):
    pass

# @param(id="m5", name="after_trading")
# 交易引擎：盘后处理函数，每日盘后执行一次
def m5_after_trading_bigquant_run(context, data):
    pass

# @module(position="-436,-868", comment="""使用基本信息对股票池过滤""")
m1 = M.cn_stock_basic_selector.v8(
    exchanges=["""上交所""", """深交所"""],
    list_sectors=["""主板"""],
    indexes=["""中证500""", """上证指数""", """创业板指""", """深证成指""", """上证50""", """科创50""", """沪深300""", """中证1000""", """中证100""", """深证100"""],
    st_statuses=["""正常"""],
    margin_tradings=["""两融标的""", """非两融标的"""],
    sw2021_industries=["""农林牧渔""", """采掘""", """基础化工""", """钢铁""", """有色金属""", """建筑建材""", """机械设备""", """电子""", """汽车""", """交运设备""", """信息设备""", """家用电器""", """食品饮料""", """纺织服饰""", """轻工制造""", """医药生物""", """公用事业""", """交通运输""", """房地产""", """金融服务""", """商贸零售""", """社会服务""", """信息服务""", """银行""", """非银金融""", """综合""", """建筑材料""", """建筑装饰""", """电力设备""", """国防军工""", """计算机""", """传媒""", """通信""", """煤炭""", """石油石化""", """环保""", """美容护理"""],
    drop_suspended=True,
    m_cached=False,
    m_name="""m1"""
)

# @module(position="-406,-773", comment="""因子特征""")
m2 = M.input_features_dai.v30(
    input_1=m1.data,
    mode="""表达式""",
    expr="""-- DAI SQL 算子/函数: https://bigquant.com/wiki/doc/dai-PLSbc1SbZX#h-%E5%87%BD%E6%95%B0
-- 数据&字段: 数据文档 https://bigquant.com/data/home
-- 数据使用: 表名.字段名, 对于没有指定表名的列，会从 expr_tables 推断

float_market_cap  AS score
 
""",
    expr_filters="""-- DAI SQL 算子/函数: https://bigquant.com/wiki/doc/dai-PLSbc1SbZX#h-%E5%87%BD%E6%95%B0
-- 数据&字段: 数据文档 https://bigquant.com/data/home
list_days > 365
pe_ttm > 0
st_status = 0 

""",
    expr_tables="""cn_stock_prefactors_community""",
    extra_fields="""date, instrument""",
    order_by="""date, instrument""",
    expr_drop_na=True,
    extract_data=False,
    m_name="""m2"""
)

# @module(position="-368,-673", comment="""持股数量、打分到仓位""")
m3 = M.score_to_position.v4(
    input_1=m2.data,
    score_field="""score""",
    hold_count=3,
    position_expr="""-- DAI SQL 算子/函数: https://bigquant.com/wiki/doc/dai-PLSbc1SbZX#h-%E5%87%BD%E6%95%B0
-- 在这里输入表达式, 每行一个表达式, 输出仓位字段必须命名为 position, 模块会进一步做归一化
-- 排序倒数: 1 / score_rank AS position
-- 对数下降: 1 / log2(score_rank + 1) AS position
-- TODO 拟合、最优化 ..

-- 等权重分配
1 AS position
""",
    total_position=1,
    extract_data=False,
    m_name="""m3"""
)

# @module(position="-358,-573", comment="""抽取预测数据""")
m4 = M.extract_data_dai.v17(
    sql=m3.data,
    start_date="""2020-01-01""",
    start_date_bound_to_trading_date=True,
    end_date="""2024-08-14""",
    end_date_bound_to_trading_date=True,
    before_start_days=90,
    debug=False,
    m_name="""m4"""
)

# @module(position="-325,-473", comment="""交易，日线，设置初始化函数和K线处理函数，以及初始资金、基准等""")
m5 = M.bigtrader.v34(
    data=m4.data,
    start_date="""""",
    end_date="""""",
    initialize=m5_initialize_bigquant_run,
    before_trading_start=m5_before_trading_start_bigquant_run,
    handle_tick=m5_handle_tick_bigquant_run,
    handle_data=m5_handle_data_bigquant_run,
    handle_trade=m5_handle_trade_bigquant_run,
    handle_order=m5_handle_order_bigquant_run,
    after_trading=m5_after_trading_bigquant_run,
    capital_base=500000,
    frequency="""daily""",
    product_type="""股票""",
    rebalance_period_type="""交易日""",
    rebalance_period_days="""5""",
    rebalance_period_roll_forward=True,
    backtest_engine_mode="""标准模式""",
    before_start_days=0,
    volume_limit=1,
    order_price_field_buy="""open""",
    order_price_field_sell="""open""",
    benchmark="""沪深300指数""",
    plot_charts=True,
    debug=False,
    backtest_only=False,
    m_name="""m5"""
)
# </aistudiograph>