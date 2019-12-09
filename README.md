## NOTICE:
for the sake of privacy of your datasets, please copy it to datasets/ dir

# env requirement: 
- Python3 
- Tesorflow 1.13+ 
- Keras 2.3+

# run:
overallSentiment.py

# Main methods: DNN-AlBERT(google) + FineTune twice.
- FineTune 1st
    source: https://github.com/bojone/bert4keras.git 
    ACC: 0.86
- FineTUne 2nd: 
    Datasets: based on the predicted result by fineTune1 model, Manually adjust 1000 reviews's annotations.
    ACC: 0.92

# Next:
- [ ] still much to read from the result data ..
- [ ] Aspect-Based Sentiment Analysis
