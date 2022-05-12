import sys
import math
import pandas as pd
import numpy as np

'''
Create probabilistic model.   (Write your own code to do this.)   
• Compute the prior probability for each class, 1 (spam) and 0 (not-spam) in 
the training data. As described in part 1, P(1) should be about 0.4.    
  
• For each of the 57 features, compute the mean and standard deviation in the 
training set of the values given each class.    If any of the features has zero standard 
deviation, assign it a “minimal” standard deviation (e.g., 0.0001) to avoid a divide-by-
zero error in Gaussian Naïve Bayes.    
'''

def naive_bayes_classifier(data, mu_spam, mu_not, sig_spam, sig_not):
    #P(xi|cj) = N(xi; mu^i,cj, sig^i,cj)
    #N(xi; mu^i,cj, sig^i,cj) = 1/sqrt(2pi) * e ^ -[(x-mu)^2/2sig^2]

    #Feature probabilities given class:
    feat_prob_1 = np.zeros(shape=data.shape)
    feat_prob_0 = np.zeros(shape=data.shape)
    #for spam
    for i in range(len(data)):
        for j in range(data.shape[1]):
            feat_prob_1[i][j] = (1 / math.sqrt(2 * math.pi)) * np.exp(-(math.pow((data[i][j] - mu_spam[j]), 2)  / (2 * math.pow(sig_spam[j], 2)) ))
    #for not spam
    for i in range(len(data)):
        for j in range(data.shape[1]):
            feat_prob_0[i][j] = (1 / math.sqrt(2 * math.pi)) * np.exp(-(math.pow((data[i][j] - mu_not[j]), 2)  / (2 * math.pow(sig_not[j], 2)) ))

    #class prediction:
    prior_1 = .4
    prior_0 = .6
    prediction_1 = np.zeros(len(data))
    prediction_0 = np.zeros(len(data))
    for i in range(len(data)):
        total_1 = math.log(prior_1)
        total_0 = math.log(prior_0)
        for j in range(data.shape[1]-1):
            if (feat_prob_1[i][j]) == 0:
                total_1 += 0
            else:
                total_1 += math.log(feat_prob_1[i][j])
            if (feat_prob_0[i][j]) == 0:
                total_0 += 0
            else:
                total_0 += math.log(feat_prob_0[i][j])

        #TODO fix argmax usage
        prediction_1[i] = total_1
        prediction_0[i] = total_0

        if prediction_1[i] > prediction_0[i]:
            print("predicted: 1", " real:", data[i][57])
        else:
            print("predicted: 0", "real:", data[i][57])



#bayes pt 2 about 28 min
def calc_feature_values(training_data):
    #just get mu and standard dev (sigma) for each of 57 different features

    #separate out the two classes 
    spam = training_data[training_data[:,57] == 1]
    not_spam = training_data[training_data[:,57] == 0]
    print("spam: ", len(spam))
    print("not spam: ", len(not_spam))

    #get mean of columns
    #spam:
    mu_spam = np.zeros(spam.shape[1])
    for i in range(len(spam)):
        for j in range(spam.shape[1]):
            mu_spam[j] += spam[i][j]

    mu_spam = mu_spam/len(spam)
    #not spam:
    mu_not = np.zeros(not_spam.shape[1])
    for i in range(len(not_spam)):
        for j in range(not_spam.shape[1]):
            mu_not[j] += not_spam[i][j]

    mu_not = mu_not/len(not_spam)


    #check against np functions
    averages_spam = spam.mean(axis=0)
    if np.array_equal(mu_spam, averages_spam):
        print("mu for spam: correct")
    averages_not = not_spam.mean(axis=0)
    if np.array_equal(mu_not, averages_not):
        print("mu for not: correct")


    #get standard deviation
    sigma_spam = np.zeros(spam.shape[1])
    for i in range(len(spam)):
        for j in range(spam.shape[1]):
            sigma_spam[j] += math.pow(abs(spam[i][j] - mu_spam[j]), 2)
    sigma_spam = sigma_spam / len(sigma_spam)
    sigma_spam = np.sqrt(sigma_spam)

    sigma_not = np.zeros(not_spam.shape[1])
    for i in range(len(not_spam)):
        for j in range(not_spam.shape[1]):
            sigma_not[j] += math.pow(abs(not_spam[i][j] - mu_not[j]), 2)
    sigma_not = np.sqrt(sigma_not / len(sigma_not))

    #check for zeros
    for x in range(len(sigma_spam)):
        if sigma_spam[x] == 0:
            sigma_spam[x] = .0001
    for x in range(len(sigma_not)):
        if sigma_not[x] == 0:
            sigma_not[x] = .0001



    #check agains np functions:
    sig_spam = np.std(spam, axis=0)
    if np.array_equal(sigma_spam, sig_spam):
        print("sigma for spam: correct")
    else:
        print("sigma for spam: NOT correct")
        #print("np:")
        #print(sig_spam)
        #print("hand:")
        #print(sigma_spam)
    sig_not = np.std(not_spam, axis=0)
    if np.array_equal(sigma_not, sig_not):
        print("sigma for not: correct")
    else:
        print("sigma for not: NOT correct")
    
    return mu_spam, mu_not, sigma_spam, sigma_not




def load_data(datafile):
    #pd indexes by column, row
    data = pd.read_csv(datafile, header=None)

    #testing pandas stuff
    print(data.head())
    print(len(data))
    print(data[2][57])
    print(data[57][2])
    print(data.index[2])
    print(type(data[:][2]))


    #spam 40%
    #class_1_data = data.loc[data[column]condition]
    class_1_data = data.loc[data[57] == 1]
    print(len(class_1_data))
    train_data_1 = class_1_data[:len(class_1_data)//2]
    test_data_1 = class_1_data[len(class_1_data)//2:]
    print("train cl 1 length:", len(train_data_1))
    print("test cl 1 length:", len(test_data_1))

    #not-spam 60%
    class_2_data = data.loc[data[57] == 0]
    print(len(class_2_data))
    train_data_2 = class_2_data[:len(class_2_data)//2]
    test_data_2 = class_2_data[len(class_2_data)//2:]
    print("train cl 2 length:", len(train_data_2))
    print("test cl 2 length:", len(test_data_2))

    #randomize, convert to numpy
    train_data_series = pd.concat([train_data_1, train_data_2], ignore_index=True)
    train_data_series = train_data_series.sample(len(train_data_series))
    train_data = train_data_series.to_numpy()

    test_data_series = pd.concat([test_data_1, test_data_2], ignore_index=True)
    test_data_series = test_data_series.sample(len(test_data_series))
    test_data = test_data_series.to_numpy()

    #print(train_data[:5])
    #print(test_data[:5])

    print(len(train_data))
    print(len(test_data))

    return train_data, test_data



def main():
    #TODO replace with specific vals from training data
    spam_prior = .4
    not_prior = .6

    if len(sys.argv) < 2:
        print("Enter data file!")

    file = sys.argv[1]
    train_set, test_set = load_data(file)
    mu_spam, mu_not, sigma_spam, sigma_not = calc_feature_values(train_set)
    naive_bayes_classifier(train_set, mu_spam, mu_not, sigma_spam, sigma_not)


if __name__ == '__main__':
    main()