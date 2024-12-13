def linear_regression(df):
    X=df[FEATURE_IN_LINEAR]
    y=df['dur']
    model = LinearRegression()
    model.fit(X,y)
    print("Coefficient:",model.coef_)
    print("Intercept:",model.intercept_)

    y_pred = model.predict(X)
    print("MSE:", mean_squared_error(y, y_pred))
    print("R-squared:", r2_score(y, y_pred))

def run_delta(df):
    X_baseline = df[['f']]
    y=df['dur']
    baseline_model = LinearRegression()
    baseline_model.fit(X_baseline,y)
    y_pred_baseline = baseline_model.predict(X_baseline)
    mse_baseline = mean_squared_error(y, y_pred_baseline)
    r2_baseline = r2_score(y,y_pred_baseline)

    print("Baseline Model:")
    print("  Coefficients:", baseline_model.coef_)
    print("  Intercept:", baseline_model.intercept_)
    print("  MSE:", mse_baseline)
    print("  R-squared:", r2_baseline)

    X_target_model = df[['f','Surprisal']]
    target_model = LinearRegression()
    target_model.fit(X_target_model,y)
    y_pred_target_model = target_model.predict(X_target_model)
    mse_target = mean_squared_error(y, y_pred_target_model)
    r2_target = r2_score(y,y_pred_target_model)
    print("Target Model:")
    print("  Coefficients:", target_model.coef_)
    print("  Intercept:", target_model.intercept_)
    print("  MSE:", mse_target)
    print("  R-squared:", r2_target)

    print("\nComparison:")
    print(f"MSE Baseline Model: {mse_baseline:.2f}, Target Model: {mse_target:.2f}")
    print(f"R-squared Baseline Model: {r2_baseline:.2f}, New Model: {r2_target:.2f}")

df = pd.read_excel("sentences.xlsx", sheet_name='word')
grouped_words = df.groupby("SN")["WORD"].apply(list).tolist()
if not os.path.exists("sentences_prediction.xlsx"):
    sentences_prediction(grouped_words)

if not os.path.exists("experiments_with_surprisal.xlsx"):
    df = pd.read_excel("experiments.xlsx")
    dur_sum_df = df.groupby(['id','sn','wn'],as_index=False).agg({
        'dur':'sum',
        'f':'first',
        'l':'first',
        'i':'first',
        'fl':'mean',
        'ao':'mean',
        'o':'mean',
    })
    dur_sum_df.to_excel("experiments_sum_dur.xlsx", index = False)
    df_with_surprisal = pd.read_excel("sentences_prediction.xlsx")

    df_a=dur_sum_df
    df_b=df_with_surprisal

    df_b=df_b.rename(columns={"SN":"sn","NW":"wn"})
    df_a = df_a.merge(df_b[['sn', 'wn', 'WORD', 'Surprisal']], on=['sn', 'wn'], how='left')
    df_a.to_excel("experiments_with_surprisal.xlsx", index=False)
if not os.path.exists("experiments_aver_dur.xlsx"):
    df = pd.read_excel("experiments_with_surprisal.xlsx")
    df_aver_dur = df.groupby(['sn', 'wn'], as_index=False).agg({
        'dur': 'mean',  # 计算 dur 的均值
        'f': 'first',
        'l': 'first',
        'i': 'first',
        'WORD': 'first',
        'Surprisal': 'first',
        'fl': 'mean',
        'ao': 'mean',
        'o': 'mean'
    })
    df_aver_dur.to_excel("experiments_aver_dur.xlsx", index=False)

df_linear = pd.read_excel("experiments_aver_dur.xlsx")
df_linear['l'] = 1 / df_linear['l']
missing_columns = [col for col in FEATURE_IN_LINEAR if col not in df_linear.columns]
if missing_columns:
    raise ValueError(f"The following columns are missing from the data: {missing_columns}")
df_linear.dropna(inplace=True)

linear_regression(df_linear)

run_delta(df_linear)