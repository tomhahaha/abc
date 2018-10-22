import pandas as pd

part1=pd.DataFrame(pd.read_csv('../data/output/data-finally.csv'))
part2=pd.read_csv('../data/output/ws_res/part-00001','\t')
part1.columns=['A','B','C']
part1=part1[['A','B']]
part2.columns=['A','B']
data=part2.merge(part1,on='A')[['B_y','B_x']]
data.to_csv('../data/output/part-00001-效果.csv')