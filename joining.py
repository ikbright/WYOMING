import pandas as pd

files_list = ['data1.txt', 'data2.txt', 'data3.txt', 'data4.txt']

df_list = []
for file in files_list:
    df_list.append(pd.read_csv(file, header=None, sep=' '))

big_df = pd.concat(df_list, ignore_index=True)
print(big_df.shape)
