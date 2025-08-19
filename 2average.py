from bigmodule import M

# <aistudiograph>

# @param(id="m3", name="initialize")
# 交易引擎：初始化函数，只执行一次
def m3_initialize_bigquant_run(context):
    from bigtrader.finance.commission import PerOrder
    # 系统已经设置了默认的交易手续费和滑点，要修改手续费可使用如下函数
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
    

# @param(id="m3", name="handle_data")
def m3_handle_data_bigquant_run(context, data):
    import pandas as pd

    # 下一个交易日不是调仓日，则不生成信号
    if not context.rebalance_period.is_signal_date(data.current_dt.date()):
        return

    # 从传入的数据 context.data 中读取今天的信号数据
    today_df = context.data[context.data["date"] == data.current_dt.strftime("%Y-%m-%d")]
    # 无当天数据返回
    if len(today_df)==0:
        return
    
    # 账户持仓
    positions = context.get_account_positions()
    
    # 持股数量
    for stock in context.instruments:
        if stock in positions.keys():
            hold_num = positions[stock].current_qty
        else:
            hold_num = 0 
        
            
        # 交易信号
        try:
            entry_condi  = today_df[today_df['instrument'] == stock].entry_condi.values[0]
            exit_condi  = today_df[today_df['instrument'] == stock].exit_condi.values[0]
        except :
            continue
        
        weight = 1 / len(context.instruments)  
        
        # 满足开仓条件
        if entry_condi:
            # 如果没有仓位
            if hold_num == 0:
                context.order_target_percent(stock, weight)
            # 如果持有仓位 
            elif hold_num > 0:
                pass 

        # 满仓平仓条件
        if exit_condi: 
            # 如果没有仓位
            if hold_num == 0:
                pass 
            # 如果有仓位
            elif hold_num > 0:
                context.order_target_percent(stock, 0)

# @module(position="-243,-1187", comment="""""", comment_collapsed=True)
m1 = M.input_features_dai.v30(
    mode="""表达式""",
    expr="""m_avg(close,5) as short_ma
m_avg(close,80) as long_ma

-- 进场条件
m_lag(short_ma, 1) <= m_lag(long_ma, 1) and short_ma > long_ma  as entry_condi 
-- 出场条件 
m_lag(short_ma, 1) >= m_lag(long_ma, 1) and short_ma < long_ma as exit_condi 
 """,
    expr_filters="""instrument in ('000002.SZ', '600519.SH')""",
    expr_tables="""cn_stock_prefactors_community""",
    extra_fields="""date,instrument""",
    order_by="""date, instrument""",
    expr_drop_na=False,
    extract_data=False,
    m_name="""m1"""
)

# @module(position="-242,-1099", comment="""""", comment_collapsed=True)
m4 = M.extract_data_dai.v17(
    sql=m1.data,
    start_date="""2015-01-01""",
    start_date_bound_to_trading_date=False,
    end_date="""2024-08-26""",
    end_date_bound_to_trading_date=False,
    before_start_days=60,
    debug=False,
    m_name="""m4"""
)

# @module(position="-266,-1010", comment="""交易，日线，设置初始化函数和K线处理函数，以及初始资金、基准等""", comment_collapsed=True)
m3 = M.bigtrader.v34(
    data=m4.data,
    start_date="""""",
    end_date="""""",
    initialize=m3_initialize_bigquant_run,
    handle_data=m3_handle_data_bigquant_run,
    capital_base=1000000,
    frequency="""daily""",
    product_type="""股票""",
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
    m_name="""m3"""
)
# </aistudiograph>