# NOTICE:
for the sake of the privacy of your datasets, I deleted from repo. Please copy it to "datasets" dir.

# Highlight:
- AlBERT works really good. Especially after good fine-tuning.
- Using Numpy mask array is a good and fast way to analyse cross-section data.

# Env requirement: 
- Python3 
- Tesorflow 1.13+ 
- Keras 2.3+

# Run:
overallSentiment.py

# Main methods: DNN-ALBERT_tiny_zh_google + FineTune twice.
- FineTune 1st
    - source: https://github.com/bojone/bert4keras.git
    - ACC: 0.86 (for the test dataset)
- FineTune 2nd: 
    - Datasets: based on the predicted result by fineTune1 model, manually adjusted 1000 reviews's annotations.
    - ACC: 0.92 (for the test dataset)

# Next:
For the 1st time of doing a NLP project, I've paied too much time in ABSA tech review which leads to no time for further analysis of processed data. So:
And for my carelessness, I over writed my manually-annotated data once. So I annotated the data sets twice. 
- [ ] still much (can be easy)to read from the result data .. eg.:
    - keyword analysis: check what is the most Positive/Negative aspect that cosumers hold.
    - exclude the "fake product" case in negative reviews, then analyse.
- [ ] Aspect-Based Sentiment Analysis
