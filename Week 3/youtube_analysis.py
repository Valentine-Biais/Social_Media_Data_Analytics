import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm

twitter_data = pd.read_csv('twitter_result.csv', error_bad_lines=False, sep=';')
print(twitter_data)

plt.figure()
friendsList = [x for x in twitter_data.friends if str(x) != 'nan']
hist1, edges1 = np.histogram(friendsList)
plt.bar(edges1[:-1], hist1, width=edges1[1:]-edges1[:-1])

print(twitter_data.corr())

plt.scatter(twitter_data.followers, twitter_data.retwc)

y = [x for x in twitter_data.retwc if str(x) != 'nan']
X = [x for x in twitter_data.followers if str(x) != 'nan']
X = sm.add_constant(X)

lr_model = sm.OLS(y,X).fit()

print(lr_model.summary())

X_prime = np.linspace(X.followers.min(), X.followers.max(), 100)
X_prime = sm.add_constant(X_prime)

y_hat = lr_model.predict(X_prime)
plt.scatter(X.followers, y)
plt.xlabel('Followers')
plt.ylabel('Retwc')
plt.plot(X_prime[:,1], y_hat)