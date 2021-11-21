import sys
import pandas as pd
import json
import ast
def func(a):
    #data = a
    #data = [{'name': 'vikash', 'age': 27}, {'name': 'Satyam', 'age': 14}]
    #df = pd.DataFrame.from_dict(data, orient='columns')
    print(a)
    #a = eval(a)
    a = ast.literal_eval(a)
    #a = json.loads(a)
    print(a)
    print(type(a))


if __name__ == '__main__':
    a = []
    for i in range(1, len(sys.argv)):
        a.append((sys.argv[i]))

    func(a[0])



