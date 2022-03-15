import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import json
import lib.views


def generate_views(raw_data):
    df = pd.read_csv(raw_data, encoding='windows-1252')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df = df.set_index(df['Order Date'])
    df = df.sort_index(ascending=False)
    df = df.drop(['Order Date'], axis=1)

    stats = ['Quantity', 'Sales', 'Profit', 'Discount']
    for stat in stats:
        dQ = lib.views.make_pivot_table(df, stat)
        dQ.to_csv('./data/generated/'+stat+'.csv')

    return 1



if __name__ == "__main__":
    print('Generating data views fra raw data')
    raw_data = './data/raw/superstore.csv'
    generate_views(raw_data)


