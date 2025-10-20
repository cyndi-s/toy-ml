import argparse, pandas as pd
from sklearn.linear_model import LinearRegression
p=argparse.ArgumentParser(); p.add_argument("--data",default="data.csv"); p.add_argument("--out",default="model.pkl"); a=p.parse_args()
df=pd.read_csv(a.data); X=df[['x']].values; y=df['y'].values
m=LinearRegression().fit(X,y); print("coef:",m.coef_[0],"intercept:",m.intercept_)
