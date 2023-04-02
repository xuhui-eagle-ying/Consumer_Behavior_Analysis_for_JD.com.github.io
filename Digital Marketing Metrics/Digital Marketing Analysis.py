#一、导入数据
#导入常用的第三方库
import pymysql
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pyecharts import Funnel    # 从pyecharts包中导出创建漏斗图的函数

#导入编码，方便后期能显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#连接Mysql数据库
conn = pymysql.connect(host='###.#.###.###',port=####,user='###',passwd='########',db='#########',charset='utf8')
TB_user = pd.read_sql('select * from TB_user',conn)

#二、清洗数据
#构建重列名字典
dict={'user_id':'用户ID','item_id':'商品ID','behavior_type':'用户行为类型','user_geohash':'地理位置','item_category':'商品种类','time':'日期'}
data = TB_user.copy()

#列名重命名
data.rename(columns=dict,inplace=True)
"""
#查看数据的缺失值情况
data.isnull().sum()

#发现地理位置的缺失值比较严重，但无法填充，也无法删除，此处仅查看缺失的比例
print('地理位置缺失比例:{:.2%}'.format(data['地理位置'].isnull().sum()/data.shape[0]))
"""
#将日期列的数据格式由字符转换为时间
data['日期'] = pd.to_datetime(data['日期'])

#对数据进行时间排序
data = data.sort_values(by='日期',ascending=True).reset_index(drop=True)

#三、用户行为分析及数据可视化
#定义图像处理函数
def open():
    plt.ion()  # 打开交互模式
    mngr = plt.get_current_fig_manager()  # 获取当前figure manager

def close():
    plt.show()
    plt.pause(5)  # 该句显示图片5秒
    plt.ioff()  # 显示完后一定要配合使用plt.ioff()关闭交互模式，否则可能出奇怪的问题
    plt.clf()  # 清空图片
    plt.close()  # 清空窗口
    
#1、PV,UV随时间的分析
#日PV、UV值
pv_daily = data.groupby('日期',as_index=True)['用户ID'].count()
pv_daily = pv_daily.rename(columns={'用户ID':'PV（日期）访问量'})
uv_daily = data.groupby('日期',as_index=True)['用户ID'].apply(lambda x:x.drop_duplicates().count())
uv_daily = uv_daily.rename(columns={'用户ID':'UV（日期）访问量'})

#建立PV和UV随时间的关系图
open()
fig,axes = plt.subplots(2,1,sharex=True)
pv_daily.plot(x='date',y='pv',ls='dashed',lw=3,marker='o',color='r',ax=axes[0]).set_title('京东商城2020年11月18日～12月18日每日PV流量统计图')
uv_daily.plot(x='date',y='uv',lw=3,marker='^',color='b',ax=axes[1]).set_title('京东商城2020年11月18日～12月18日每日UV流量统计图')
fig.tight_layout()
fig = plt.gcf()
fig.savefig('日PV、UV流量图.jpg', dpi=1000, transparent=True, pad_inches=0, bbox_inches='tight')
close()

#2、付费率随时间的变化
open()
data_user_buy = data.groupby(['日期','用户ID','用户行为类型'],as_index=False)['商品ID'].count().rename(columns={'商品ID':'行为次数'})
data_user_buy = data_user_buy.groupby('日期').apply(lambda x:x[data_user_buy['用户行为类型']=='4']['用户ID'].count()/len(x['用户ID'].unique()))
data_user_buy = data_user_buy.plot(lw=3,color='g')
plt.title('京东商城2020年11月18日～12月18日用户付费率统计图')
fig.tight_layout()
fig = plt.gcf()
fig.savefig('用户付费情况统计图.jpg', dpi=1000, transparent=True, pad_inches=0, bbox_inches='tight')
close()

#3、复购率
#计算复购率
data_buy = data[data['用户行为类型']=='4'].groupby(['用户ID','商品ID'])['f1'].apply(lambda x:len(x.unique())).rename(columns={'f1':'购买次数'})
data_rebuy = data_buy-1
data_rebuy = data_rebuy[data_rebuy!=0]
data_rebuy_count = data_rebuy.value_counts()
print('复购率:%.2f%%\n'%(sum(data_rebuy)/sum(data_buy)*100))
#print(data_rebuy_count)

#复购次数和时间间隔统计
data_day_buy = data[data['用户行为类型']=='4'].groupby(['用户ID','日期'],as_index=False)['用户行为类型'].count()
data_day_buy = data_day_buy.groupby('用户ID',as_index=True)['日期'].apply(lambda x:x.sort_values().diff(1).dropna())
data_day_buy = data_day_buy.map(lambda x:x.days)
data_day_count = data_day_buy.value_counts()
#print(data_day_count)

open()
fig,axes = plt.subplots(1,2)
ax1 = data_rebuy_count.plot(figsize=(20,5),kind='bar',ax=axes[0])
ax2 = data_day_count.plot(figsize=(20,5),kind='bar',ax=axes[1])
ax1.set_title('复购次数分布',fontsize=15)
ax2.set_title('复购时间间隔分布',fontsize=15)
fig = plt.gcf()
fig.savefig('复购情况统计图.jpg')
close()

#4、漏斗分析-显示不同用户行为的总体数量
data_user_count = data.groupby('用户行为类型').count()
pv_all = data_user_count['用户ID']
click = pv_all.iloc[0]; print('点击量:%s'%click)
favor = pv_all.iloc[1]; print('收藏量:%s'%favor)
cart = pv_all.iloc[2]; print('加购物车量:%s'%cart)
buy = pv_all.iloc[3]; print('购买量:%s\n'%buy)

#绘制漏斗图
click_rate = 100
cart_rate = cart / click * 100
favor_rate = favor / click * 100
buy_rate = buy / click * 100

attrs_keys = ['点击','加购物车','收藏','支付']
attrs_values = [click_rate,cart_rate,favor_rate,buy_rate]
funnel = Funnel("总体转化率",width=800, height=400, title_pos='center')
funnel.add("商品交易环节",attrs_keys,attrs_values,is_label_show=True,label_formatter='{b} {c}%',label_pos="outside",
           legend_orient='vertical',legend_pos='left',is_legend_show=True)

funnel.render('总体转化率.html')

#5、建立RFM模型（由于缺少金额M数据，此处仅分析时间R和频度F）
#每位用户的最近购买时间（假设今天为2020年12月19日）
recent_buy_time = data[data['用户行为类型']=='4'].groupby('用户ID')['日期']
recent_buy_time = recent_buy_time.apply(lambda x:datetime(2020,12,19)-x.sort_values().iloc[-1])
recent_buy_time = recent_buy_time.reset_index().rename(columns={'日期':'距离时间'})
recent_buy_time['距离时间'] = recent_buy_time['距离时间'].map(lambda x:x.days)

#每位用户最近一个月交易次数统计
buy_freq = data[data['用户行为类型']=='4'].groupby('用户ID')['日期'].count()
buy_freq = buy_freq.reset_index().rename(columns={'日期':'购买频次'})

#将R、F合并
rfm=pd.merge(recent_buy_time,buy_freq,left_on='用户ID',right_on='用户ID',how='outer')

#给R、F打分score
rfm['距离时间标签']=pd.qcut(rfm['距离时间'],2,labels=['2','1'])
rfm['购买频次标签']=pd.qcut(rfm['购买频次'],2,labels=['1','2'])

#得分拼接
rfm['用户标签'] = rfm['距离时间标签'].str.cat(rfm['购买频次标签'])

#根据RFM分类
rfm_dict = {'22':'价值用户','12':'保持用户','21':'发展用户','11':'挽留用户'}
rfm['用户标签分类'] = rfm['用户标签'].map(rfm_dict)
#print(rfm)

#计算各用户类型的比例
user_total = rfm.groupby('用户标签',as_index=True)['用户ID'].count().rename(columns={'用户ID':'总数'})
user_rate = rfm.groupby('用户标签',as_index=True)['用户ID'].apply(lambda x:x.count()/sum(user_total)).rename(columns={'用户ID':'比例'})
user_category = pd.DataFrame({'总数':user_total,'比例':user_rate})
user_category = user_category.reset_index(drop=False)
user_category['用户标签分类'] = user_category['用户标签'].map(rfm_dict)
print(user_category)

#绘制饼图
open()
x = user_category['总数']
labels = user_category['用户标签分类']
plt.pie(x,labels=labels,autopct='%.2f%%',shadow=False,radius=1.0,labeldistance=1.1)
plt.title("各类型用户比例图",loc="center")
fig = plt.gcf()
fig.savefig('用户类型比例图.jpg', dpi=1000, transparent=True, pad_inches=0, bbox_inches='tight')
close()
