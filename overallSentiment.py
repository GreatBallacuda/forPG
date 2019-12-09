import json
import numpy as np
import codecs
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizer import Tokenizer
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, get_all_attributes
import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import math

locals().update(get_all_attributes(keras.layers))
set_gelu('tanh')

# pre-trained model: albert_tiny_zh_google
ModelDir = 'models/albert_tiny_zh_google'


maxlen = 128

testPredLen = 1000
trainTestRatio = 0.8
sentiThresh = 0.8

config_path = os.path.join(ModelDir,'albert_config_tiny_g.json')
checkpoint_path = os.path.join(ModelDir,'albert_model.ckpt')
dict_path = os.path.join(ModelDir,'vocab.txt')
fineTuneWeights = 'bert4keras/sentiment/best_model.weights'
test_data = load_data('bert4keras/sentiment/sentiment.test.data')
targetData = 'datasets/ds-nlp-interview-question_v2.xlsx'

def load_data(filename):
    D = []
    with codecs.open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D

tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator:
    """数据生成器
    """
    def __init__(self, data, predictOnly = False,batch_size=32):
        self.data = data
        self.predictOnly = predictOnly
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        if not self.predictOnly:
            for i in idxs:
                text, label = self.data[i]
                token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
                if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        else:
            for i in idxs:
                text = self.data[i]
                token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                # batch_labels.append([label])
                if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    # batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids]
                    batch_token_ids, batch_segment_ids = [], []
    def forfit(self):
        while True:
            for d in self.__iter__(True):
                yield d


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total

class Evaluator(keras.callbacks.Callback):
    def __init__(self,valid_generator,test_generator,weightsSavePath):
        self.best_val_acc = 0.
        self.valid_generator = valid_generator
        self.test_generator = test_generator
        self.weightsSavePath = weightsSavePath
    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(self.valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.weights')
            model.save_weights(self.weightsSavePath)
        test_acc = evaluate(self.test_generator)
        print(u'val_acc: %05f, best_val_acc: %05f, test_acc: %05f\n' %
              (val_acc, self.best_val_acc, test_acc))


# Load Pre-Trained model
bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    albert=True,
    return_keras_model=False,
)
output = Dropout(rate=0.1)(bert.model.output)
output = Dense(units=2,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()
AdamLR = extend_with_piecewise_linear_lr(Adam)

model.compile(
    loss='sparse_categorical_crossentropy',
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=AdamLR(learning_rate=1e-4,
                     lr_schedule={1000: 1, 2000: 0.1}),
    metrics=['accuracy'],
)

# Load FineTune-1st weights
model.load_weights(fineTuneWeights)


# Load dataset
DFraw = pd.read_excel(targetData)

# ust fintTune-1st model to make testPred(default 1000 rows)
Review = DFraw.iloc[0:testPredLen,0].values
test_generator = data_generator(Review,predictOnly = True)
preds = []
for x in test_generator:
    # possibillity for positive
    y_pred = model.predict(x)[:,1]
    preds.extend(y_pred)

DFresult = pd.DataFrame(columns = ['REVIEW','SENTIMENT'])
DFresult = DFresult.assign(REVIEW = Review, SENTIMENT = preds)
DFresult.to_csv('testPredResult.csv')

# dict for fineTune to make code cleaner.
DKft2 = {
    'negative':{
        'polarity':-1,
        'testPredFile':'negResult.csv',
        'manualAnnotatedFile':'ADJnegResult.csv',
        'accBeforeFT2': None,
        'length' : None,
        'DFadj' : None,
        'FT2testFile': 'fineTune2_negTest.txt',
    },
    'positive':{
        'polarity':1,
        'testPredFile':'posResult.csv',
        'manualAnnotatedFile':'ADJposResult.csv',
        'accBeforeFT2': None,
        'length' : None,
        'DFadj' : None,
        'FT2testFile': 'fineTune2_posTest.txt',
    }
}

# copy result for manually annotation.
for pola in DKft2:
    DFfocus = DFresult[DFresult['SENTIMENT']*DKft2[pola]['polarity']>0.5*DKft2[pola]['polarity']]
    DFfocus.to_csv(DKft2[pola]['testPredFile'])
    DKft2[pola]['length'] = DFfocus.shape[0]
    shutil.copy(DKft2[pola]['testPredFile'],DKft2[pola]['manualAnnotatedFile'])

# for the ballance of negative data and positive data, use the shorter part as input lengh of each pole.
ballanceL = min(DKft2['negative']['length'],DKft2['positive']['length'])

# here, mannually annotate output data.
# .
# method/rule: if the preds data is wrong(eg. one is negative emotion while preds result exceeds 0.5), add a negative sign to the result.
# ...
# ....
# time ticking ...
# .....
# ..
# .

# NOTICE: ONLY AFTER manually annotated data, go on
for pola in DKft2:
    DKft2[pola]['DFadj'] = pd.read_csv(DKft2[pola]['manualAnnotatedFile'],index_col = 0)
    DKft2[pola]['DFadj'] = DKft2[pola]['DFadj'].iloc[:ballanceL,:]
    errMask = DKft2[pola]['DFadj']['SENTIMENT']<0
    DKft2[pola]['accBeforeFT2'] = 1- errMask[errMask].shape[0]/errMask.shape[0]
    print('Accuracy of {} Reviews:{:.4f}'.format(pola,DKft2[pola]['accBeforeFT2']))
    DKft2[pola]['DFadj'].loc[errMask,'SENTIMENT'] = 1 if DKft2[pola]['polarity'] < 0 else 0
    DKft2[pola]['DFadj'].loc[~errMask,'SENTIMENT'] = 0 if DKft2[pola]['polarity'] < 0 else 1
    DKft2[pola]['DFadj'] = DKft2[pola]['DFadj'].astype({'SENTIMENT':int})
    DKft2[pola]['DFadj'].loc[int(ballanceL*trainTestRatio):,:].to_csv(DKft2[pola]['FT2testFile'],header = None, index = None, sep='\t')


# Generate Train and test/validate data. (for the reason of limited time, I use valid_data = test_data here)
DF_FT2_Train = pd.concat([DKft2['negative']['DFadj'].iloc[:int(ballanceL*trainTestRatio),:],
    DKft2['positive']['DFadj'].iloc[:int(ballanceL*trainTestRatio),:]],
    ignore_index=True)
DF_FT2_Test = pd.concat([DKft2['negative']['DFadj'].iloc[int(ballanceL*trainTestRatio):,:],
    DKft2['positive']['DFadj'].iloc[int(ballanceL*trainTestRatio):,:]],
    ignore_index=True)
DF_FT2_Train.to_csv('FT2_Train.txt',header = None, index = None, sep='\t')
DF_FT2_Test.to_csv('FT2_Test.txt',header = None, index = None, sep='\t')

# Load
train_data = load_data('FT2_Train.txt')
valid_data = load_data('FT2_Test.txt')
test_data = load_data('FT2_Test.txt')
train_generator = data_generator(train_data)
valid_generator = data_generator(valid_data)
test_generator = data_generator(test_data)
evaluator = Evaluator(valid_generator = valid_generator,test_generator = test_generator,weightsSavePath='FT2.weights')
# train model
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=50,
                    callbacks=[evaluator])

#Test result:
model.load_weights('FT2.weights')
print(u'final test acc: %05f\n' % (evaluate(test_generator)))


# Pred the whole data set:
finalReview = DFraw.iloc[:,0].values
finalTest_generator = data_generator(finalReview,predictOnly = True,batch_size=32)

preds = []
for x in finalTest_generator:
    # possibillity for positive
    y_pred = model.predict(x)[:,1]
    preds.extend(y_pred)
    print('Prediction Progress:{:.2%}'.format(len(preds)/finalReview.shape[0]),end = '\r')

# Save reulsts and analyse:
DFprocessed = DFraw.copy()
DFprocessed['SENTIMENT'] = preds
DFprocessed.to_csv('processed.csv')

# use mask array to show the statistics, it's a very convinient and fast way.
Mask = {}
for col in ('ONLINE_STORE', 'BRAND', 'YEAR','MONTH'):
    Mask[col] = {}
    # string data:
    if DFprocessed[col].dtype == 'object':
        DFprocessed[col] = DFprocessed[col].str.lower()
        values = list(DFprocessed[col].unique())
    # numerical data:
    else:
        values = list(DFprocessed[col].unique())
        list.sort(values)
    for item in values:
        print(item)
        Mask[col][item] = DFprocessed[col] == item

# delete wrong annotation
del Mask['YEAR'][1900]

Mask['SENTIMENT'] = {}
Mask['SENTIMENT']['positive'] = DFprocessed['SENTIMENT'] > sentiThresh
Mask['SENTIMENT']['negative'] = DFprocessed['SENTIMENT'] < 1 - sentiThresh
Mask['SENTIMENT']['neutral'] = (DFprocessed['SENTIMENT'] > 1 - sentiThresh)&(DFprocessed['SENTIMENT'] < sentiThresh)


# define a plot function that accepts masks as arguments. Make the analysis fast, clear and elegant.
def pltmask(xList,xMasks,yMask,labels = None,width = 0.2,saveFile=True):
    y = []
    for i in xMasks:
        y.append(i[i&yMask].shape[0]/i[i].shape[0])
        print(y)
    plt.bar(xList,y,width=width)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    if saveFile:
        plt.savefig(labels[0]+'-'+labels[1]+'.png')
    plt.show()


# some shallow/apparent analysis:
pltmask(Mask['BRAND'].keys(),Mask['BRAND'].values(),Mask['SENTIMENT']['positive'],labels= ['brand','positive'])

pltmask(Mask['BRAND'].keys(),Mask['BRAND'].values(),Mask['SENTIMENT']['negative'],labels= ['brand','negative'])

pltmask(Mask['ONLINE_STORE'].keys(),Mask['ONLINE_STORE'].values(),Mask['SENTIMENT']['negative'],labels= ['ONLINE_STORE','negative'])

pltmask(Mask['YEAR'].keys(),Mask['YEAR'].values(),Mask['SENTIMENT']['negative']*Mask['BRAND']['pampers'],labels= ['YEAR','pampers-negative'])

pltmask(Mask['MONTH'].keys(),Mask['MONTH'].values(),Mask['SENTIMENT']['negative']*Mask['BRAND']['pampers'],labels= ['MONTH','pampers-negative'])

pltmask(Mask['MONTH'].keys(),Mask['MONTH'].values(),Mask['BRAND']['pampers'],labels= ['MONTH','pampers-reviewNumber'])






# shutil.copy('ADJposResultBK.csv','ADJposResult.csv')
# shutil.copy('ADJnegResultBK.csv','ADJnegResult.csv')

