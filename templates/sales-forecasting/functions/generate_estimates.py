import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import lib.estimate



def generate_estimates():
    stats = ['Quantity', 'Sales', 'Profit', 'Discount']
    for stat in stats:
        print('Generating predictions for ' + stat)
        df = pd.read_csv('./data/generated/'+stat+'.csv')
        dg = df.groupby('Sub-Category').sum()
        dg =lib.estimate.predict_linear(dg)
        df = lib.estimate.assign_to_products(df, dg)
    
        df.to_csv('./data/predicted/'+stat+'-estimated.csv')
        
    return 1


if __name__ == "__main__":
    print('Estimating stats for next month pr product')
    generate_estimates()

    