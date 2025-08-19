from bigmodule import M

# <aistudiograph>

# @param(id="m1", name="initialize")
# 交易引擎：初始化函数，只执行一次
def m1_initialize_bigquant_run(context):
    # 加载预测数据
    context.all_data = context.data
    print(len(context.all_data))


# @param(id="m1", name="before_trading_start")
# 交易引擎：每个单位时间开盘前调用一次。
def m1_before_trading_start_bigquant_run(context, data):
    # 盘前处理，订阅行情等
    pass

# @param(id="m1", name="handle_tick")
# 交易引擎：tick数据处理函数，每个tick执行一次
def m1_handle_tick_bigquant_run(context, tick):
    pass

# @param(id="m1", name="handle_data")
# 交易引擎：bar数据处理函数，每个时间单位执行一次
def m1_handle_data_bigquant_run(context, data):
    
    import pandas as pd 
    from bigtrader.constant  import Direction 
    from bigtrader.constant  import OrderType

    today = data.current_dt.strftime('%Y-%m-%d') # 当前交易日期
    all_data = context.all_data[context.all_data.date == today]
    all_data = pd.DataFrame(all_data)
    if len(all_data) == 0:#过滤没有指标的数据
        return
    
    price = all_data['close'].iloc[0]
    high_line = all_data['bolling_up'].iloc[0]
    low_line = all_data['bolling_down'].iloc[0]
    instrument = context.future_symbol(context.instruments[0]) # 交易标的
    long_position = context.get_account_position(instrument, direction=Direction.LONG).avail_qty#多头持仓
    short_position = context.get_account_position(instrument, direction=Direction.SHORT).avail_qty#空头持仓
    curr_position = short_position + long_position#总持仓
    if short_position > 0:
        if price > high_line:
            context.buy_close(instrument, short_position, price, order_type=OrderType.MARKET)
            context.buy_open(instrument, 4, price, order_type=OrderType.MARKET)
            print(today,'先平空再开多')
    elif long_position > 0:
        if price < low_line:
            context.sell_close(instrument, long_position, price, order_type=OrderType.MARKET)
            context.sell_open(instrument, 4, price, order_type=OrderType.MARKET)
            print(today,'先平多再开空',curr_position)
    elif curr_position==0:
        if price > high_line:
            context.buy_open(instrument, 4, price, order_type=OrderType.MARKET)
            print('空仓开多')
        elif price<low_line:
            context.sell_open(instrument, 4, price, order_type=OrderType.MARKET)
            print('空仓开空')

# @param(id="m1", name="handle_trade")
# 交易引擎：成交回报处理函数，每个成交发生时执行一次
def m1_handle_trade_bigquant_run(context, trade):
    pass

# @param(id="m1", name="handle_order")
# 交易引擎：委托回报处理函数，每个委托变化时执行一次
def m1_handle_order_bigquant_run(context, order):
    pass

# @param(id="m1", name="after_trading")
# 交易引擎：盘后处理函数，每日盘后执行一次
def m1_after_trading_bigquant_run(context, data):
    pass

# @module(position="-166,-926", comment="""因子特征""")
m2 = M.input_features_dai.v30(
    mode="""表达式""",
    expr="""-- DAI SQL 算子/函数: https://bigquant.com/wiki/doc/dai-PLSbc1SbZX#h-%E5%87%BD%E6%95%B0
-- 数据&字段: 数据文档 https://bigquant.com/data/home
-- 数据使用: 表名.字段名, 对于没有指定表名的列，会从 expr_tables 推断
close
m_avg(close,20) as ma 
m_stddev(close,20) as _std
ma+2*_std as bolling_up 
ma-2*_std as bolling_down


""",
    expr_filters="""-- DAI SQL 算子/函数: https://bigquant.com/wiki/doc/dai-PLSbc1SbZX#h-%E5%87%BD%E6%95%B0
-- 数据&字段: 数据文档 https://bigquant.com/data/home
instrument in ('jm2201.DCE')
""",
    expr_tables="""cn_future_bar1d""",
    extra_fields="""date, instrument""",
    order_by="""date, instrument""",
    expr_drop_na=True,
    sql="""""",
    extract_data=False,
    m_name="""m2"""
)

# @module(position="-166,-833", comment="""抽取预测数据""")
m4 = M.extract_data_dai.v17(
    sql=m2.data,
    start_date="""2021-02-17""",
    start_date_bound_to_trading_date=True,
    end_date="""2021-12-31""",
    end_date_bound_to_trading_date=True,
    before_start_days=90,
    debug=False,
    m_name="""m4"""
)

# @module(position="-170,-722", comment="""""", comment_collapsed=True)
m1 = M.bigtrader.v34(
    data=m4.data,
    start_date="""""",
    end_date="""""",
    initialize=m1_initialize_bigquant_run,
    before_trading_start=m1_before_trading_start_bigquant_run,
    handle_tick=m1_handle_tick_bigquant_run,
    handle_data=m1_handle_data_bigquant_run,
    handle_trade=m1_handle_trade_bigquant_run,
    handle_order=m1_handle_order_bigquant_run,
    after_trading=m1_after_trading_bigquant_run,
    capital_base=1000000,
    frequency="""daily""",
    product_type="""期货""",
    rebalance_period_type="""交易日""",
    rebalance_period_days="""1""",
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
    m_name="""m1"""
)
# </aistudiograph>