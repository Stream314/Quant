from bigmodule import M
import dai


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


# 提取股票数据
stock_sql = """
SELECT
    date,
    instrument,
    float_market_cap
FROM cn_stock_prefactors_community
WHERE list_days > 365
AND pe_ttm > 0
-- 非st
AND st_status = 0 
-- 主板
AND list_sector = 1
-- 非停牌
AND suspended = 0
-- 非北证50
AND is_bz50 = 0
"""
stock_data = dai.query(stock_sql, filters={"date": ["2019-12-01", "2024-08-14"]}).df()
stock_data_cleaned = stock_data.dropna()

# 仓位分配
#设定持仓股票数量
count_num = 3
filtered_df = stock_data_cleaned.sort_values(by=['date', 'float_market_cap'], ascending=[True, True]).groupby('date').head(count_num)
#设置仓位
filtered_df['position'] = 1 / count_num
filtered_df['score_rank'] = filtered_df.groupby('date')['float_market_cap'].rank(ascending=True).astype(int)
#重命名
# 重命名 'float_market_type' 为 'score'
filtered_df = filtered_df.rename(columns={'float_market_cap': 'score'})

stock_data = dai.DataSource.write_bdb(filtered_df)

# 设置回测起始时间和终止时间
start_date = '2020-01-01'
end_date = '2024-08-14'


# @module(position="-325,-473", comment="""交易，日线，设置初始化函数和K线处理函数，以及初始资金、基准等""", comment_collapsed=False)
m5 = M.bigtrader.v30(
    data=stock_data,
    start_date=start_date,
    end_date=end_date,
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