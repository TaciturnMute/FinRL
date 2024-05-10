import mplfinance as mpf
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torchvision import transforms
import sys
sys.path.append('/mnt/')
from finrl.data.photo_indicator import calculate_dmi
from matplotlib.ticker import MultipleLocator,FormatStrFormatter


def figure1_plot(data: pd.DataFrame):

    """输入140个日期的数据，最后20天绘制蜡烛图和成交量的柱状图，前120天绘制滑动平均线。返回图片和图片类型，图片类型用于裁剪处理"""

    mav_5 = data.iloc[:140]['Close'].rolling(5).mean()
    mav_20 = data.iloc[:140]['Close'].rolling(20).mean()
    mav_60 = data.iloc[:140]['Close'].rolling(60).mean()
    mav_120 = data.iloc[:140]['Close'].rolling(120).mean()
    mav_time_range = data.iloc[120:140].index

    mc=mpf.make_marketcolors(up='red',down='blue',
                            edge= {'up':'red','down':'blue'},
                            wick={'up':'red','down':'blue'},
                            volume={'up':'red','down':'blue'},  
                            )
    s = mpf.make_mpf_style(
                        marketcolors=mc,
                        facecolor='white',
                        base_mpf_style= 'default', 
    )
    fig = mpf.figure(style=s,figsize=(5.0,10))
    gs = gridspec.GridSpec(4, 1, figure=fig)  # 定义一个3行1列的网格
    ax1 = fig.add_subplot(gs[0:3, 0])  # 前两行分配给 ax1
    ax2 = fig.add_subplot(gs[3, 0])    # 第三行分配给 ax2
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    maximum = max(max(mav_5[mav_time_range]),max(mav_20[mav_time_range]),max(mav_60[mav_time_range]),max(mav_120[mav_time_range]),max(data['Close'][mav_time_range]))
    minimum = min(min(mav_5[mav_time_range]),min(mav_20[mav_time_range]),min(mav_60[mav_time_range]),min(mav_120[mav_time_range]),min(data['Close'][mav_time_range]))
    diff = maximum - minimum

    gap_candidates = [0.05, 0.1, 0.2, 0.4, 0.5, 1, 2, 5,10, 15]
    best_ticks_num = 4
    best_ticks_num_gap_loss = 100
    best_ticks_gap = None
    for gap in gap_candidates:
        if abs(int(diff / gap) - best_ticks_num) < best_ticks_num_gap_loss:
            best_ticks_gap = gap
            best_ticks_num_gap_loss = int(diff / gap) - best_ticks_num
    ax1.yaxis.set_major_locator(MultipleLocator(best_ticks_gap))
          
    ax2.set_xticks([])
    ax2.set_xticklabels([])
    ax2.set_yticks([])
    plt.subplots_adjust(hspace=0.2)

    apds = [
        mpf.make_addplot(mav_5[mav_time_range],ax=ax1,type='line',color='blue'),
        mpf.make_addplot(mav_20[mav_time_range],ax=ax1,type='line',color='orange'),
        mpf.make_addplot(mav_60[mav_time_range],ax=ax1,type='line',color='green'),
        mpf.make_addplot(mav_120[mav_time_range],ax=ax1,type='line',color='red'),
        ]

    mpf.plot(
        data.iloc[120:140],
        type='candle', 
        ax=ax1, 
        volume=ax2, 
        scale_width_adjustment=dict(volume=0.8, candle=1.2, lines=1),    
        addplot=apds,
        )
    
    if maximum<10:
        type_ = 0
    elif maximum < 100 and 10 <= maximum:
        type_ = 1
    else:
        type_ = 2
    return fig,type_


def figure1_preprocess(path,type_):

    """裁剪处理第一种类型图片，返回处理后的图片和其tensor形式"""

    image_original = Image.open(path)

    if type_ == 0:
        image = image_original.crop((16,100,500,910))

    elif type_ == 1:
        image = image_original.crop((15,100,500,910))
        
    else:
        image = image_original.crop((8,100,500,912))

    image = image.resize((240,400))   # 宽度比例，高度比例

    transform = transforms.ToTensor()

    tensor_image = transform(image)

    return image,tensor_image


def figure2_plot(data_with_dmi):

    """生成第二种类型图片"""

    DI_plus = data_with_dmi['+DI']
    DI_minus = data_with_dmi['-DI']
    ADX = data_with_dmi['ADX']
    # time_range = data_with_dmi.index[130:150]

    fig = plt.figure(figsize=(4,8))
    ax1 = fig.add_subplot()
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_ylim(0,100)
    ax1.yaxis.grid(True, color='lightgray', linestyle='-', linewidth=2)
    ax1.yaxis.set_major_locator(MultipleLocator(20))
    ax1.plot(DI_plus,color='green')
    ax1.plot(DI_minus,color='orange')
    ax1.plot(ADX,color='blue')

    return fig

def figure2_preprocess(path):

    """裁剪处理"""

    image_original = Image.open(path)

    image = image_original.crop((0,67,400,733))   # 接收四个位置参数，构成一个元组。分别代表左上右下。

    image = image.resize((240,400))   # 宽度比例，高度比例

    transform = transforms.ToTensor()

    tensor_image = transform(image)

    return image,tensor_image


def figure3_plot(data):

    """生成第三种类型图片"""

    fig = plt.figure(figsize=(4,8))
    ax1 = fig.add_subplot()
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_ylim(0,100)
    ax1.yaxis.grid(True, color='lightgray', linestyle='-', linewidth=2)
    ax1.yaxis.set_major_locator(MultipleLocator(20))  # 快速%K线
    ax1.plot(data['k_fast'],color='green',linewidth=2)  # 慢速%D线。
    ax1.plot(data['d_slow'],color='red',linewidth=2)
    ax1.axhline(20,color='orange',linewidth=3)
    ax1.axhline(80,color='blue',linewidth=3)
    return fig

def figure3_preprocess(path):

    """裁剪处理"""

    image_original = Image.open(path)

    image = image_original.crop((0,67,400,733))   

    image = image.resize((240,400))   

    transform = transforms.ToTensor()

    tensor_image = transform(image)

    return image,tensor_image