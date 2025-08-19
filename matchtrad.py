from bigmodule import M
import pandas as pd

# <aistudiograph>

# @param(id="m1", name="initialize")
# 交易引擎：初始化函数，只执行一次
def m1_initialize_bigquant_run(context):
    from bigtrader.finance.commission import PerOrder

    # 系统已经设置了默认的交易手续费和滑点，要修改手续费可使用如下函数
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
   
     
    stocklist = context.instruments
    
    prices_df = pd.pivot_table(context.data, values='close', index=['date'], columns=['instrument'])
    prices_df.fillna(method='ffill',inplace=True)
    context.x = prices_df[stocklist[0]] # 股票1
    context.y = prices_df[stocklist[1]] # 股票2
     
# @param(id="m1", name="before_trading_start")
def m1_before_trading_start_bigquant_run(context,data):    
    pass
# @param(id="m1", name="handle_tick")
# 交易引擎：tick数据处理函数，每个tick执行一次
def m1_handle_tick_bigquant_run(context, tick):
    pass

# @param(id="m1", name="handle_data")
# 回测引擎：每日数据处理函数，每天执行一次
def m1_handle_data_bigquant_run(context, data):
 
    # 线性回归两个股票的股价 y=ax+b
    from pyfinance import ols
    import numpy as np 
    y = context.y 
    x = context.x 
    model = ols.OLS(y=y, x=x)

    def zscore(series):
        return (series - series.mean()) / np.std(series)
    
    # 计算 y-a*x 序列的zscore值序列
    zscore_calcu = zscore(y-model.beta*x)
    context.zscore = zscore_calcu
    
    today = data.current_dt.strftime("%Y-%m-%d")
    zscore_today = context.zscore.loc[today]
    
    #获取股票的列表
    stocklist=context.instruments
    # 转换成回测引擎所需要的symbol格式
    symbol_1 = context.symbol(stocklist[0]) 
    symbol_2 = context.symbol(stocklist[1])  

    # 持仓
    cur_position_1 = context.portfolio.positions[symbol_1].amount
    cur_position_2 = context.portfolio.positions[symbol_2].amount
       
    # 交易逻辑
    # 如果zesore大于上轨（>1），则价差会向下回归均值，因此需要买入股票x，卖出股票y
    if zscore_today > 1 and cur_position_1 == 0 and data.can_trade(symbol_1) and data.can_trade(symbol_2):  
        context.order_target_percent(symbol_2, 0)
        context.order_target_percent(symbol_1, 1)
#         print(today, '全仓买入：',stocklist[0], '卖出全部:', stocklist[1])
        
    # 如果zesore小于下轨（<-1），则价差会向上回归均值，因此需要买入股票y，卖出股票x
    elif zscore_today < -1 and cur_position_2 == 0 and data.can_trade(symbol_1) and data.can_trade(symbol_2):  
        context.order_target_percent(symbol_1, 0)  
        context.order_target_percent(symbol_2, 1)
#         print(today, '全仓买入：',stocklist[1],  '卖出全部:', stocklist[0])
 

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

# @module(position="-341,-815", comment="""因子特征""")
m2 = M.input_features_dai.v30(
    mode="""表达式""",
    expr="""-- DAI SQL 算子/函数: https://bigquant.com/wiki/doc/dai-PLSbc1SbZX#h-%E5%87%BD%E6%95%B0
-- 数据&字段: 数据文档 https://bigquant.com/data/home
-- 数据使用: 表名.字段名, 对于没有指定表名的列，会从 expr_tables 推断

close/adjust_factor as close
-- 使用 float 类型。默认是高精度 decimal.Decimal, 不能和float直接相乘""",
    expr_filters="""-- DAI SQL 算子/函数: https://bigquant.com/wiki/doc/dai-PLSbc1SbZX#h-%E5%87%BD%E6%95%B0
-- 数据&字段: 数据文档 https://bigquant.com/data/home

instrument in ('601328.SH','601998.SH')

""",
    expr_tables="""cn_stock_prefactors_community""",
    extra_fields="""date, instrument""",
    order_by="""date, instrument""",
    expr_drop_na=True,
    sql="""-- 使用DAI SQL获取数据，构建因子等，如下是一个例子作为参考
-- DAI SQL 语法: https://bigquant.com/wiki/doc/dai-PLSbc1SbZX#h-sql%E5%85%A5%E9%97%A8%E6%95%99%E7%A8%8B

SELECT

    -- 在这里输入因子表达式
    -- DAI SQL 算子/函数: https://bigquant.com/wiki/doc/dai-PLSbc1SbZX#h-%E5%87%BD%E6%95%B0
    -- 数据&字段: 数据文档 https://bigquant.com/data/home

    c_rank(volume) AS rank_volume,
    close / m_lag(close, 1) as return_0,

    -- 日期和股票代码
    date, instrument
FROM
    -- 预计算因子 cn_stock_factors https://bigquant.com/data/datasources/cn_stock_factors
    cn_stock_factors
WHERE
    -- WHERE 过滤，在窗口等计算算子之前执行
    -- 剔除ST股票
    st_status = 0
QUALIFY
    -- QUALIFY 过滤，在窗口等计算算子之后执行，比如 m_lag(close, 3) AS close_3，对于 close_3 的过滤需要放到这里
    -- 去掉有空值的行
    COLUMNS(*) IS NOT NULL
-- 按日期和股票代码排序，从小到大
ORDER BY date, instrument
""",
    extract_data=False,
    m_name="""m2"""
)

# @module(position="-341,-709", comment="""抽取预测数据""")
m4 = M.extract_data_dai.v17(
    sql=m2.data,
    start_date="""2019-01-01""",
    start_date_bound_to_trading_date=True,
    end_date="""2024-09-04""",
    end_date_bound_to_trading_date=True,
    before_start_days=90,
    debug=False,
    m_name="""m4"""
)

# @module(position="-340,-596", comment="""""", comment_collapsed=True)
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
    m_cached=False,
    m_name="""m1"""
)
# </aistudiograph>