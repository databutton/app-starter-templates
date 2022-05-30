import numpy as np
from sklearn.linear_model import LinearRegression


def predict_linear(dg):
    X = np.array([1, 2, 3, 4, 5])
    g = []
    for i in range(0, len(dg.index)):
        f = dg.iloc[i][::-1]
        n = len(f)
        y = f[n - 5 :]
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)
        z = [[6]]
        est = model.predict(z)
        g.append(est[0])

    dg.insert(0, "January 2018 Estimated", g)

    return dg


def assign_to_products(df, dg):
    estimated = []
    for cat in dg.index:
        dh = df[df["Sub-Category"] == cat]
        total_months = [np.sum(dh.iloc[a, 4:]) for a in range(0, len(dh.index))]
        total_months = total_months / np.sum(total_months)
        estimated = np.append(estimated, dg["January 2018 Estimated"].loc[cat] * total_months)
    df.insert(4, "January 2018 Estimated", estimated)

    return df
