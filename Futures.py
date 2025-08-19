from bigmodule import M

# <aistudiograph>

# @param(id="m1", name="initialize")
# 交易引擎：初始化函数，只执行一次
def m1_initialize_bigquant_run(context):
    # 加载预测数据
    context.signal_data = context.data # 信号数据
    context.option_data = context.options['data'].read()  # 主力合约映射关系表
    context.lots = 5 # 下单手数

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
    dominant_contract = context.option_data[context.option_data['date'] == today].dominant.values[0] # 主力合约

    signal_data = context.signal_data[context.signal_data.date == today] # 当天信号指标
    signal_data = pd.DataFrame(signal_data) 

    if len(signal_data) == 0:#过滤没有指标的数据
        return
    
    price = signal_data['close'].iloc[0]
    high_line = signal_data['bolling_up'].iloc[0]
    low_line = signal_data['bolling_down'].iloc[0]
    
    # 获取账户持仓
    account_pos = context.get_account_positions()
    
    if len(account_pos) == 0:
        cur_hold_ins = dominant_contract # 当天无仓位的情况下，默认设置
    else:
        cur_hold_ins = list(context.get_account_positions().keys())[0]  

    long_position = context.get_account_position(cur_hold_ins, direction=Direction.LONG).avail_qty #多头持仓
    short_position = context.get_account_position(cur_hold_ins, direction=Direction.SHORT).avail_qty #空头持仓
    curr_position = short_position + long_position # 总持仓
    
#     if today == '2023-04-17':
#         print('===========:',today,  price, long_position, short_position, curr_position, '主力合约标的:', dominant_contract, '当前持仓:', cur_hold_ins) 
    
    # 先进行移仓换月
    if dominant_contract != cur_hold_ins:
        # 移仓换月
        if short_position > 0:
            context.buy_close(cur_hold_ins, short_position, price, order_type=OrderType.MARKET)
            context.sell_open(dominant_contract, short_position , price, order_type=OrderType.MARKET)
            print(today, '空头仓位的情形下移仓换月!')

        elif long_position > 0:
            context.sell_close(cur_hold_ins, long_position, price, order_type=OrderType.MARKET)
            context.buy_open(dominant_contract, long_position, price, order_type=OrderType.MARKET)
            print(today, '多头仓位的情形下移仓换月!')

    # 再进行交易逻辑判断 
    if short_position > 0:
        if price > high_line:
            # 空头情况下，价格突破上轨
            context.buy_close(dominant_contract, short_position, price, order_type=OrderType.MARKET)
            context.buy_open(dominant_contract, context.lots, price, order_type=OrderType.MARKET)
            print(today,'先平空再开多', dominant_contract)

    elif long_position > 0:
        if price < low_line:
            # 多仓情况下，价格突破下轨
            context.sell_close(dominant_contract, long_position, price, order_type=OrderType.MARKET)
            context.sell_open(dominant_contract, context.lots, price, order_type=OrderType.MARKET)
            print(today,'先平多再开空', dominant_contract)

    elif curr_position == 0:
        if price > high_line:
            # 无仓位情形下，价格突破上轨
            context.buy_open(dominant_contract, context.lots, price, order_type=OrderType.MARKET)
            print(today, '空仓开多', dominant_contract)

        elif price < low_line:
            # 无仓位情形下，价格突破下轨 
            context.sell_open(dominant_contract, context.lots, price, order_type=OrderType.MARKET)
            print(today, '空仓开空', dominant_contract) 

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

# @module(position="-156,-1180", comment="""因子特征""")
m2 = M.input_features_dai.v30(
    mode="""表达式""",
    expr="""-- DAI SQL 算子/函数: https://bigquant.com/wiki/doc/dai-PLSbc1SbZX#h-%E5%87%BD%E6%95%B0
-- 数据&字段: 数据文档 https://bigquant.com/data/home
-- 数据使用: 表名.字段名, 对于没有指定表名的列，会从 expr_tables 推断
close
-- 中轨
m_avg(close,60) as ma 

-- 带宽
m_stddev(close,60) as _std

-- 上轨
ma+2*_std as bolling_up 
-- 下轨
ma-2*_std as bolling_down

""",
    expr_filters="""-- DAI SQL 算子/函数: https://bigquant.com/wiki/doc/dai-PLSbc1SbZX#h-%E5%87%BD%E6%95%B0
-- 数据&字段: 数据文档 https://bigquant.com/data/home
-- 抽取焦煤期货相关的数据
instrument like 'jm%'""",
    expr_tables="""cn_future_bar1d""",
    extra_fields="""date, instrument""",
    order_by="""date, instrument""",
    expr_drop_na=True,
    sql="""""",
    extract_data=False,
    m_name="""m2"""
)

# @module(position="-160,-1050", comment="""抽取预测数据""")
m4 = M.extract_data_dai.v17(
    sql=m2.data,
    start_date="""2019-01-01""",
    start_date_bound_to_trading_date=True,
    end_date="""2024-07-31""",
    end_date_bound_to_trading_date=True,
    before_start_days=90,
    debug=False,
    m_name="""m4"""
)

# @module(position="171,-1148", comment="""因子特征""", comment_collapsed=True)
m3 = M.input_features_dai.v30(
    mode="""表达式""",
    expr="""-- DAI SQL 算子/函数: https://bigquant.com/wiki/doc/dai-PLSbc1SbZX#h-%E5%87%BD%E6%95%B0
-- 数据&字段: 数据文档 https://bigquant.com/data/home
-- 数据使用: 表名.字段名, 对于没有指定表名的列，会从 expr_tables 推断
dominant""",
    expr_filters="""-- DAI SQL 算子/函数: https://bigquant.com/wiki/doc/dai-PLSbc1SbZX#h-%E5%87%BD%E6%95%B0
-- 数据&字段: 数据文档 https://bigquant.com/data/home
-- instrument in ('jm0000.DCE')

instrument like 'jm%'""",
    expr_tables="""cn_future_dominant""",
    extra_fields="""date, instrument""",
    order_by="""date, instrument""",
    expr_drop_na=True,
    sql="""""",
    extract_data=False,
    m_name="""m3"""
)

# @module(position="167,-1043", comment="""抽取预测数据""", comment_collapsed=True)
m5 = M.extract_data_dai.v17(
    sql=m3.data,
    start_date="""2019-01-01""",
    start_date_bound_to_trading_date=True,
    end_date="""2024-09-11""",
    end_date_bound_to_trading_date=True,
    before_start_days=90,
    debug=False,
    m_name="""m5"""
)

# @module(position="-63,-930", comment="""""", comment_collapsed=True)
m1 = M.bigtrader.v30(
    data=m4.data,
    options_data=m5.data,
    start_date="""""",
    end_date="""""",
    initialize=m1_initialize_bigquant_run,
    before_trading_start=m1_before_trading_start_bigquant_run,
    handle_tick=m1_handle_tick_bigquant_run,
    handle_data=m1_handle_data_bigquant_run,
    handle_trade=m1_handle_trade_bigquant_run,
    handle_order=m1_handle_order_bigquant_run,
    after_trading=m1_after_trading_bigquant_run,
    capital_base=200000,
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