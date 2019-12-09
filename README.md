# NOTICE:
for the sake of the privacy of your datasets, please copy it to datasets/ dir

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
- [ ] still much (can be easy)to read from the result data .. (keyword analysis, eg.)
- [ ] Aspect-Based Sentiment Analysis
