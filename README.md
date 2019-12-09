# NOTICE:
for the sake of privacy of your datasets, please copy it to datasets/ dir

# Env requirement: 
- Python3 
- Tesorflow 1.13+ 
- Keras 2.3+

# Run:
overallSentiment.py

# Main methods: DNN-AlBERT(google) + FineTune twice.
- FineTune 1st
    - source: https://github.com/bojone/bert4keras.git
    - ACC: 0.86 (for the test dataset)
- FineTUne 2nd: 
    - Datasets: based on the predicted result by fineTune1 model, manually adjusted 1000 reviews's annotations.
    - ACC: 0.92 (for the test dataset)

# Next:
- [ ] still much to read from the result data ..
- [ ] Aspect-Based Sentiment Analysis
