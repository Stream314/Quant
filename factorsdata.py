params = {'group_num':10, 'factor_field':'ma_amount_60', 'instruments':'全市场', 'factor_direction':1, 'benchmark':'中证500', 'data_process':True} # instruments支持选项：沪深300、中证500、中证1000、全市场；benchmark支持的选项：沪深300、中证500、中证1000

sql = """
SELECT
    date, 
    instrument, 
    m_AVG(amount, 60) AS ma_amount_60
FROM
    cn_stock_bar1d
ORDER BY
    date, instrument;
"""

start_date = '2018-01-01'
end_date =  '2024-01-01'
factor_data = dai.query(sql, filters={"date": [start_date, end_date]}).df()

# 因子数据处理
factor_data.dropna(subset=[params['factor_field']], inplace=True)
factor_data = factor_data[['instrument', 'date', params['factor_field']]]