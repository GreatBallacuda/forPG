import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba.analyse
from wordcloud import WordCloud, ImageColorGenerator
import os
import copy
if not os.path.exists('./results/'): os.makedirs('./results/')


resultData = './results/processed.csv'
sentiThresh = 0.8
livePlot = False

## plot(bin) multi-dimentional data
def multiPlot(valueLists,xList,subXList=None,ylim=None,labels=None,totalWidth = 0.8,saveFile = None,livePlot = False,**legendKW):
    if len(valueLists) == 1:
        pass
    elif len(valueLists) != len(subXList):
        raise Exception('value list and subXList not matching.')
    fig,ax = plt.subplots()
    x = np.arange(len(valueLists[0]),dtype = 'float')
    totalWidth, n = totalWidth, len(valueLists)
    width = totalWidth / n
    for i in range(len(valueLists)):
        ax.bar(x+i*width, valueLists[i], width=width, label=subXList[i] if subXList is not None else None)
    ax.set_xticks(x+(totalWidth-width)/2)
    ax.set_xticklabels(xList)
    handles, sublabels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, sublabels, **legendKW)
    # ax.legend(**legendKW)
    # ax.legend()
    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    if saveFile is not None:
        # ax.savefig(labels[0]+'-'+labels[1]+'.png')
        fig.savefig(saveFile, bbox_extra_artists=(lgd,), bbox_inches='tight')
    if livePlot:
        plt.show()

## wrap function: use masks to generate multi-dimentional data, and plot
def maskPlot(xList,xMasksList,yMask,livePlot=False,subXList=None,ylim=None,labels=None,totalWidth = 0.8,saveFile = None,**legendKW):
    y = []
    for i in xMasksList:
        suby = []
        for j in i:
            if j[j].shape[0] != 0:
                suby.append(j[j&yMask].shape[0]/j[j].shape[0])
            else:
                suby.append(0)
        y.append(suby)
    multiPlot(
        valueLists=y,
        xList=xList,subXList=subXList,
        ylim=ylim,labels=labels,livePlot = livePlot,
        totalWidth = totalWidth,saveFile = saveFile,**legendKW)

## tool to generate formatted-mask array for maskPlot function.
def masksListGen(mainMasks,subMasks):
    masksList = []
    for j in subMasks:
        subList = []
        for i in mainMasks:
            subList.append(i&j)
        masksList.append(subList)
    return masksList

## tool to generate cross-mask list
def crossMask(maskA,maskList):
    return(list(map(lambda x:x&maskA,maskList)))


## main:
DFprocessed = pd.read_csv(resultData,index_col = 0)

## generate mask array to show the statistics, it's a very convinient and fast way.
Mask = {}
for col in ('ONLINE_STORE', 'BRAND', 'YEAR','MONTH'):
    Mask[col] = {}
    if DFprocessed[col].dtype == 'object':
        DFprocessed[col] = DFprocessed[col].str.lower()
    #     values = list(DFprocessed[col].unique())
    # else:
    values = list(DFprocessed[col].unique())
    list.sort(values)
    for item in values:
        Mask[col][item] = DFprocessed[col] == item

## delete wrong annotation
del Mask['YEAR'][1900]

Mask['SENTIMENT'] = {}
Mask['SENTIMENT']['positive'] = DFprocessed['SENTIMENT'] > sentiThresh
Mask['SENTIMENT']['negative'] = DFprocessed['SENTIMENT'] < 1 - sentiThresh
Mask['SENTIMENT']['neutral'] = (DFprocessed['SENTIMENT'] > 1 - sentiThresh)&(DFprocessed['SENTIMENT'] < sentiThresh)


## analyze:
## Brand-Year_Positive Review Rate
maskPlot(xList=list(Mask['BRAND'].keys()),yMask=Mask['SENTIMENT']['positive'],
    xMasksList=masksListGen(Mask['BRAND'].values(),Mask['YEAR'].values()),
    subXList = list(Mask['YEAR'].keys()),
    labels = ('Brand','Positive Review Rate'),
    ylim = (0.6,0.9),bbox_to_anchor=(1, 1),
    saveFile = './results/Brand_Year-Positive_Review_Rate.png'
    )

## Year-Brand_Positive Review Rate
maskPlot(xList=list(Mask['YEAR'].keys()),yMask=Mask['SENTIMENT']['positive'],
    xMasksList=masksListGen(Mask['YEAR'].values(),Mask['BRAND'].values()),
    subXList = list(Mask['BRAND'].keys()),
    labels = ('YEAR','Positive Review Rate'),
    ylim = (0.6,0.9),bbox_to_anchor=(1, 1),
    saveFile = './results/Year-Brand_Positive-Review-Rate.png'
    )

##Brand-Store_Positive Review Rate
maskPlot(xList=list(Mask['BRAND'].keys()),yMask=Mask['SENTIMENT']['positive'],
    xMasksList=masksListGen(Mask['BRAND'].values(),Mask['ONLINE_STORE'].values()),
    subXList = list(Mask['ONLINE_STORE'].keys()),
    labels = ('Brand','Positive Review Rate'),
    ylim = (0.5,1),bbox_to_anchor=(1, 1),
    saveFile = './results/Brand_Store-Positive_Review_Rate.png'
    )

## focus on pampers:
##pampers-Store-Year_Positive Review Rate
maskPlot(xList=list(Mask['ONLINE_STORE'].keys()),yMask=Mask['SENTIMENT']['positive'],
    xMasksList=masksListGen(crossMask(Mask['BRAND']['pampers'],Mask['ONLINE_STORE'].values()),Mask['YEAR'].values()),
    subXList = list(Mask['YEAR'].keys()),
    labels = ('STORE','Positive Review Rate'),
    ylim = (0.4,1),bbox_to_anchor=(1, 1),
    saveFile = './results/pampers_Store_Year-Positive_Review_Rate.png'
    )

##pampers-Year-Store_Positive Review Rate
maskPlot(xList=list(Mask['YEAR'].keys()),yMask=Mask['SENTIMENT']['positive'],
    xMasksList=masksListGen(Mask['YEAR'].values(),crossMask(Mask['BRAND']['pampers'],Mask['ONLINE_STORE'].values())),
    subXList = list(Mask['ONLINE_STORE'].keys()),
    labels = ('Year','Positive Review Rate'),
    ylim = (0.4,1),bbox_to_anchor=(1, 1),
    saveFile = './results/pampers_Year_Store-Positive_Review_Rate.png'
    )


## deeper analyze: check main aspects of neg-reviews.
NegativeRV = DFprocessed.loc[Mask['SENTIMENT']['negative'],'REVIEW_TEXT_CN']
NegativeRV.to_csv('negative.csv')
stopWordsFile = '../stopwords-master/中文停用词表.txt'
targetFile = 'negative.csv'
jieba.analyse.set_stop_words(stopWordsFile)
textRead = open(targetFile).read()
jieba.suggest_freq('红屁屁', True)
jieba.suggest_freq('红屁股', True)
keywords = jieba.analyse.extract_tags(textRead, topK=25,withWeight=True)


wc = WordCloud(font_path='simhei.ttf',background_color='white', max_words=25)

cloudSource = dict(zip([i[0] for i in keywords], [i[1] for i in keywords]))
wc.generate_from_frequencies(cloudSource)
plt.imshow(wc)
plt.axis("off")
if livePlot:
    plt.show()
wc.to_file('./results/negtiveReviewCloud.png')


## deeper analyze: check how many neg-reviews are about 'fake product'.
Mask['SENTIMENT']['fake'] = np.array([0]*DFprocessed.shape[0],dtype = bool)
tempMask = np.zeros(NegativeRV.shape[0],dtype = bool)
testStr = '正品|假货'
for i in range(NegativeRV.shape[0]):
    r = re.search(testStr,NegativeRV.iloc[i])
    if r is not None:
        tempMask[i] = True

fakeRateOfNegative = tempMask[tempMask].shape[0]/tempMask.shape[0]
negMask = Mask['SENTIMENT']['negative'].copy()
negMask[negMask] = tempMask
Mask['SENTIMENT']['fake'][negMask] = True

##Brand-Year_Fake_Rate
maskPlot(xList=list(Mask['BRAND'].keys()),yMask=Mask['SENTIMENT']['fake'],
    xMasksList=masksListGen(Mask['BRAND'].values(),Mask['YEAR'].values()),
    subXList = list(Mask['YEAR'].keys()),
    labels = ('BRAND','Fake Rate'),
    ylim = (0,0.1),bbox_to_anchor=(1, 1),
    saveFile = './results/Brand_Year-Fake_Rate.png'
    )

##Brand-Store_Fake_Rate
maskPlot(xList=list(Mask['BRAND'].keys()),yMask=Mask['SENTIMENT']['fake'],
    xMasksList=masksListGen(Mask['BRAND'].values(),Mask['ONLINE_STORE'].values()),
    subXList = list(Mask['ONLINE_STORE'].keys()),
    labels = ('BRAND','Fake Rate'),
    ylim = (0,0.35),bbox_to_anchor=(1, 1),
    saveFile = './results/Brand-Store_Fake-Rate.png'
    )

##pampers-Year-Store_Fake Rate
maskPlot(xList=list(Mask['YEAR'].keys()),yMask=Mask['SENTIMENT']['fake'],
    xMasksList=masksListGen(Mask['YEAR'].values(),crossMask(Mask['BRAND']['pampers'],Mask['ONLINE_STORE'].values())),
    subXList = list(Mask['ONLINE_STORE'].keys()),
    labels = ('YEAR','Fake Rate'),
    ylim = (0,0.1),bbox_to_anchor=(1, 1),
    saveFile = './results/pampers_Year_Store-Fake_Rate.png'
    )


# generate fake issue excluded mask dict for further analysis:
fakeExMask = copy.deepcopy(Mask)
usingPart = copy.deepcopy(~Mask['SENTIMENT']['fake'])

for i in fakeExMask.keys():
    for j in fakeExMask[i].keys():
        fakeExMask[i][j] = fakeExMask[i][j][usingPart]

## Fake_issure_excluded-Brand-Year_Positive Review Rate
maskPlot(xList=list(fakeExMask['BRAND'].keys()),yMask=fakeExMask['SENTIMENT']['positive'],
    xMasksList=masksListGen(fakeExMask['BRAND'].values(),fakeExMask['YEAR'].values()),
    subXList = list(fakeExMask['YEAR'].keys()),
    labels = ('Brand','Positive Review Rate'),
    ylim = (0.6,0.9),bbox_to_anchor=(1, 1),
    saveFile = './results/Fake_issure_excluded_Brand_Year-Positive_Review_Rate.png'
    )






