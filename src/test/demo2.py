import sys
import pandas as pd
import json
import ast
# import os,sys
# sys.path.append("F:\\论文\\kaggle\\predict_future_sales\\src\\test")

def func(a):
    #data = a
    print(a)
    data = [{'name': 'vikash', 'age': 27}, {'name': 'Satyam', 'age': 14}]
    df = pd.DataFrame.from_dict(data, orient='columns')
    print(df)
    data = pd.DataFrame
    #a = eval(a)
    #a = ast.literal_eval(a)
    #a = json.loads(a)
    print(a)
    print(type(a))
    # print(type(df))


if __name__ == '__main__':
    a = []
    for i in range(1, len(sys.argv)):
        a.append((sys.argv[i]))

    print(a[0])
    func(a[0])



