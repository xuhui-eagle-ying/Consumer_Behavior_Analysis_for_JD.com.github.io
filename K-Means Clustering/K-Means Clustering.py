import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings('ignore')
plt.rc('font', family='SimHei', size=18)    #显示中文标签
import seaborn as sns
sns.set(font='SimHei',style='darkgrid')
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics

# 1、获取数据
data = pd.read_excel('Online_Retail.xlsx')

# 2、数据清洗
data = data[(data['Quan']>0)&(data['ItemPrice']>0)].dropna()
#data.info()    #总览数据，查看有无空值

# 3、选择我们需要的字段
data['Sales'] = data['Quan'] * data['ItemPrice']
data = data[['UserID','TransactionDate','Sales']]

# 4、特征构建

# (1) 提取付款日期数据

# 提取每个用户最近（最大）的购买日期
data_r = data.groupby('UserID')['TransactionDate'].max().reset_index()
# 与当前日期相减，取得最近一次购买距当前的天数(假设当前日期是'2020-12-10')
data_r['recency'] = data_r['TransactionDate'].apply(lambda x:(pd.to_datetime('2020-12-10')-x).days)
# 两个日期相减，得到的数据类型是timedelta类型，要进行数值计算，需要提取出天数数字。
data_r.drop('TransactionDate',axis = 1,inplace = True)
#data.head()

# (2) 提取购买次数数据
# 分组聚合，得到每个用户发生于不同日期的购买次数
data_f = data.groupby('UserID')['TransactionDate'].count().reset_index()
# 修改列名
data_f.rename({'TransactionDate':'frequence'},axis = 1,inplace = True)
#data_f.head()

# (3) 提取购买金额数据
data_m = data.groupby('UserID')['Sales'].sum().reset_index()
data_m['money'] = data_m['Sales']/data_f['frequence']
data_m.drop('Sales',axis = 1,inplace = True)
#data_m.head()

# 所以现在我们已经有了包含recency、frequence、money的3个DataFrame表了，下面合并三个表：
data_rf = pd.merge(data_r,data_f,on = 'UserID',how = 'inner')
data_rfm = pd.merge(data_rf,data_m, on = 'UserID',how = 'inner')
#data_rfm.head()

# 5、查看数据分布特征

# 数据的分布特征会影响算法结果，所以有必要先了解数据的大致分布
plt.figure(figsize = (6,4))
plt.rc('font', family='SimHei', size=18)
#axes = plt.gca()
#axes.set_xlim([xmin,xmax])
#sns.set(style = 'darkgrid')
sns.countplot(data_rfm['frequence'])
plt.show()

plt.rc('font', family='SimHei', size=18)
#axes = plt.gca()
#axes.set_xlim([-500,1000])
sns.distplot(data_rfm['frequence'])
plt.title('frequence的分布直方图',fontsize = 15)
plt.show()

plt.rc('font', family='SimHei', size=18)
#axes = plt.gca()
#axes.set_xlim([xmin,xmax])
sns.distplot(data_rfm['recency'])
plt.title('recency的分布直方图',fontsize = 15)
plt.show()

plt.rc('font', family='SimHei', size=18)
#axes = plt.gca()
#axes.set_xlim([-1000,5000])
sns.distplot(data_rfm['money'],color = 'g')
plt.title('money的分布直方图',fontsize = 15)
plt.show()

# 5、数据处理和模型构造

# 首先，对数据进行标准化处理，这是为了消除量纲的影响
data_rfm_s = data_rfm.copy()
min_max_scaler = preprocessing.MinMaxScaler()
data_rfm_s = min_max_scaler.fit_transform(data_rfm[['recency','frequence','money']])
inertia = []
ch_score = []
ss_score = []
for k in range(2,9):
    model = KMeans(n_clusters = k, init = 'k-means++',max_iter = 500)
    model.fit(data_rfm_s)
    pre = model.predict(data_rfm_s)
    ch = metrics.calinski_harabasz_score(data_rfm_s,pre)
    ss = metrics.silhouette_score(data_rfm_s,pre)
    inertia.append(model.inertia_)
    ch_score.append(ch)
    ss_score.append(ss)
print(ch_score,ss_score,inertia)

# 画图可以更加直观看出三个指标的变化
score = pd.Series([ch_score,ss_score,inertia],index = ['ch_score','ss_score','inertia'])
aa = score.index.tolist()
plt.figure(figsize = (15,6))
j = 1
for i in aa:
    plt.subplot(1,3,j)
    plt.plot(list(range(2,9)),score[i])
    plt.xlabel('k的数目',fontsize = 13)
    plt.ylabel(f'{i}值',fontsize = 13)
    plt.title(f'{i}值变化趋势',fontsize = 15)
    j+=1
plt.subplots_adjust(wspace = 0.3)

# 根据上图中3个指标综合判断，选择k=5时(选其他数字一共改3处代码)，各指标变化出现明显的拐点，聚类效果相对较优，所以分成四类比较好
model = KMeans(n_clusters = 5, init = 'k-means++',max_iter = 500)
model.fit(data_rfm_s)
ppre = model.predict(data_rfm_s)
ppre = pd.DataFrame(ppre)
data = pd.concat([data_rfm,ppre],axis = 1)
data.rename({0:u'cluster'},axis = 1,inplace = True)
data.head()

# 可以看出，每个用户都有了对应的类别，并且可以查看每个类别的中心：
labels = model.labels_   # 取得各样本的类别数据
labels = pd.DataFrame(labels,columns = ['类别'])
result = pd.concat([pd.DataFrame(model.cluster_centers_),labels['类别'].value_counts().sort_index()],axis = 1)
result.columns = ['recency','frequence','money','cluster']
print(result)

aa = ['recency', 'frequence', 'money']
plt.figure(figsize = (18,4))    # 设置绘图区的大小：18 X 4
for k in range(5):
    j = 1
    for i in aa:
        plt.subplot(1,3,j)    # 设置1行3列的图片区，也就是画出来1行，3张图，j表示画在哪个位置。
        a = data[i]
        sns.kdeplot(data[data['cluster']==k][i])    # sns就是seaborn库的缩写，import seaborn as sns
        plt.title(i,fontsize = 14)                  # 图片标题是i，字号大小是14
        j+=1
    plt.show()
        
# 6、将统计结果导出到Excel中
#声明一个读写对象
writer = pd.ExcelWriter('用户价值分层 K-Means聚类.xlsx',engine='xlsxwriter')
#分别将表aggData和user_tag写入Excel中的Sheet1、Sheet2
data.to_excel(writer,sheet_name='用户分层结果')
result.to_excel(writer,sheet_name='K-Means各层中心点')
#保存读写的内容
writer.save()
