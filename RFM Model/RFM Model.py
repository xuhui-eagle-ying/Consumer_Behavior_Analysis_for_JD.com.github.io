import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font='SimHei',style='darkgrid')

# 1、获取数据
data = pd.read_excel('Online_Retail.xlsx')

# 2、数据清洗
data = data[(data['Quan']>0)&(data['ItemPrice']>0)].dropna()

# 3、数据处理
#根据数量*单价计算每笔交易的金额
data['Sales'] = data['Quan'] * data['ItemPrice']
#将交易日期处理为日期数据类型
data['TransactionDate'] = pd.to_datetime(data.TransactionDate,format='%Y/%m/%d')
#将交易日期到现在的距离
data['DateDiff'] = datetime.now() - data['TransactionDate']
#从时间中获取天数
data['DateDiff'] = data['DateDiff'].dt.days

# 4、建立模型
#统计每个用户距离现在有多久没有消费了，即找出最小的最近消费距离
R_Agg = data.groupby(by=['UserID'],as_index=False)['DateDiff'].agg('min')
#统计每个用户交易的总次数，即对订单数计数
F_Agg = data.groupby(by=['UserID'],as_index=False)['TransactionNo'].agg('count')
#统计每个用户交易的总额，即对每次的交易金额求和
M_Agg = data.groupby(by=['UserID'],as_index=False)['Sales'].agg('sum')
#把统计结果关联起来，整合为一个数据框
#因为关联列名UserID都一样，因此可以省略on条件
aggData = R_Agg.merge(F_Agg).merge(M_Agg)
#修改列名
aggData.columns = ['UserID','RecencyAgg','FrequencyAgg','MonetaryAgg']

#根据用户最近消费距离列，按照从小到大的顺序，
#求出排在0%、20%、40%、60%、80%、100%的数据点
bins = aggData.RecencyAgg.quantile(q=[0,0.2,0.4,0.6,0.8,1],interpolation='nearest')
#cut函数分段区间默认为左开放、右封闭，为了避免分段数值的最小值不在分段区间内，
#所以将bins参数列表最小值设为0。
bins[0] = 0
#用户越久没有消费，R分值越小，因此对应的得分需要从大到小排列，
#以标签值的方式进行赋值
rLabels = [5,4,3,2,1]
#根据百分位数对数据进行分段
R_S = pd.cut(aggData.RecencyAgg,bins,labels=rLabels)

#根据用户消费频次列，按照从小到大的顺序，
#求出排在0%、20%、40%、60%、80%、100%的数据点
bins = aggData.FrequencyAgg.quantile(q=[0,0.2,0.4,0.6,0.8,1],interpolation='nearest')
#cut函数分段区间默认为左开放、右封闭，为了避免分段数值的最小值不在分段区间内，
#所以将bins参数列表最小值设为0。
bins[0] = 0
#因为用户消费频次越高，得分越高，
#因此对应的得分需要从小到大排列，以标签值的方式进行赋值
fLabels = [1,2,3,4,5]
#根据百分位数对数据进行分段
F_S = pd.cut(aggData.FrequencyAgg,bins,labels=fLabels)

#根据用户最近消费距离列，按照从小到大的顺序，
#求出排在0%、20%、40%、60%、80%、100%的数据点
bins = aggData.MonetaryAgg.quantile(q=[0,0.2,0.4,0.6,0.8,1],interpolation='nearest')
#cut函数分段区间默认为左开放、右封闭，为了避免分段数值的最小值不在分段区间内，
#所以将bins参数列表最小值设为0。
bins[0] = 0
#因为用户消费总额越高，得分越高，
#因此对应的得分需要从小到大排列，以标签值的方式进行赋值
mLabels = [1,2,3,4,5]
#根据百分位数对数据进行分段
M_S = pd.cut(aggData.MonetaryAgg,bins,labels=mLabels)

#将R、F、M值的得分这三列，分别增加到aggData数据框中
aggData['R_S'] = R_S
aggData['F_S'] = F_S
aggData['M_S'] = M_S

#因为R_S、F_S、M_S得分值采用标签赋值方式获取，
#所以要先转为数值型再计算出RFM得分值
aggData['RFM'] = 100*R_S.astype(int) + 10*F_S.astype(int) + 1*M_S.astype(int)
#根据RFM这一列，按照从小到大的顺序，分为八等份，因此，需要求出排在0%、20%、40%、60%、80%、100%的数据点
bins = aggData.RFM.quantile(q=[0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1],interpolation='nearest')
#cut函数分段区间默认为左开放、右封闭，为了避免分段数值的最小值不在分段区间内，
#所以将bins参数列表最小值设为0。
bins[0] = 0
#RFM值越大，得分越高，故对应得分需从小到大排列，以标签值的方式赋值
rfmLabels = [1,2,3,4,5,6,7,8]
#根据百分位数对数据进行分段
aggData['level'] = pd.cut(aggData.RFM,bins,labels=rfmLabels)

def user_tag(x):
    if x == 1:
        category = '潜在客户'
    elif x == 2:
        category = '一般价值客户'
    elif x == 3:
        category = '重点发展客户'
    elif x == 4:
        category = '一般发展客户'
    elif x == 5:
        category = '重点保持客户'
    elif x == 6:
        category = '一般保持客户'
    elif x == 7:
        category = '重点挽留客户'
    else:
        category = '高价值客户'
    return category

aggData['User_tag'] = aggData['level'].apply(user_tag)
aggData = aggData.sort_values(by='level',ascending=False).reset_index(inplace=False,drop=True)
print(aggData)
print()
#aggData = aggData[['UserID','User_tag']]

# 5、结果统计
#对level进行分组，按照客户ID进行计数统计
user_tag = aggData.groupby(by='User_tag').size()
print(user_tag)

# 6、将统计结果导出到Excel中
#声明一个读写对象
writer = pd.ExcelWriter('用户价值分层.xlsx',engine='xlsxwriter')
#分别将表aggData和user_tag写入Excel中的Sheet1、Sheet2
aggData.to_excel(writer,sheet_name='用户分层')
user_tag.to_excel(writer,sheet_name='各类型用户数量')
#保存读写的内容
writer.save()

# 7、数据可视化
#绘制直方图对比各类型用户数量
plt.figure(figsize=(10,4),dpi=80)
user_tag.sort_values(ascending=True,inplace=True)
plt.title(label='各类型用户数量图',fontsize=22,color='white',backgroundcolor='#334f65', pad=20)
s = plt.barh(user_tag.index,user_tag.values,height=0.8,color=plt.cm.coolwarm_r(np.linspace(0,1,len(user_tag))))
for rect in s:
    width = rect.get_width()
    plt.text(width+20,rect.get_y()+rect.get_height()/2,str(width),ha='center')

plt.grid(axis='y')
plt.savefig('各类型用户数量图.jpg',dpi=1000,transparent=True,pad_inches=0,bbox_inches='tight')
plt.show()

#绘制饼图对比各类型用户比例
plt.figure(figsize=(10,6),dpi=80)
plt.title(label='各类型用户比例图',fontsize=22,color='white',backgroundcolor='#334f65',pad=20)
patches,l_text,p_text = plt.pie(user_tag.values,labels=user_tag.index,colors=plt.cm.coolwarm_r(np.linspace(0,1,len(user_tag))),autopct='%.2f%%',startangle=370)
plt.legend(bbox_to_anchor=(1.5,1.0))
plt.savefig('用户类型比例图.jpg',dpi=1000,transparent=False,pad_inches=0,bbox_inches='tight')
plt.show()
