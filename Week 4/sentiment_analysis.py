import matplotlib.pyplot as plt
import pandas as pd 

twitter_data = pd.read_csv('results_olympics.csv')

print(twitter_data.corr())

plt.scatter(twitter_data.retwc, twitter_data.polarity)
plt.scatter(twitter_data.retwc, twitter_data.subjectivity)

twitter_data_subjective = twitter_data[twitter_data['subjectivity']>0.5]

print(twitter_data_subjective.corr())