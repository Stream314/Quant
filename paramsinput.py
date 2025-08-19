perf_df = alpha_instance.whole_perf  

perf_dict = {}
perf_dict['factor_direction'] = params['factor_direction']
perf_dict['IC'] =  np.nanmean(alpha_instance.ic['g_ic'])

for i in perf_df.index:
    df = perf_df.iloc[i]
    flag = df['portfolio'] 

    for c in perf_df.columns:
        perf_dict['%s_%s'%(flag, c)] = df[c]

perf_dict.update(alpha_instance.params)

for key, value in perf_dict.items():
    if isinstance(value, (int, float)):
        if np.isinf(value) or np.isnan(value):
            perf_dict[key] = None