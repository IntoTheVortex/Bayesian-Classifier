import sys
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

def naive_bayes_classifier():
    #P(xi|cj) = N(xi; mu^i,cj, sig^i,cj)
    #N(xi; mu^i,cj, sig^i,cj) = 1/sqrt(2pi) * e ^ -[(x-mu)^2/2sig^2]
    #
    pass

#bayes pt 2 about 28 min
def calc_feature_values():
    #just get mu and standard dev (sigma) for each of 57 different features
    dataset = []
    '''
    for feature in dataset:
        mu_class1 = sum of feature in examples of mu_class1 
                    divided by number of them
        mu_class2 = sum of feature in examples of mu_class2 
                    divided by number of them
        sigma_class1 = sqrt((feature_i - mu_of_feature)^2/num of feature_is
        sigma_class2 = sqrt((feature_i - mu_of_feature)^2/num of feature_is
        '''
def load_data(datafile):
    #indexes by column, row
    data = pd.read_csv(datafile, header=None)

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
    print(type(train_data_series))
    #train_data_1 = train_data_1.to_numpy()
    print(train_data_series.shape)
    print(train_data_series.head())
    print(train_data_series.tail())
    train_data = train_data_series.to_numpy()
    print(train_data[:5])

    #now randomize both:
    train_set =  0






def main():
    #TODO replace with specific vals from training data
    spam_prior = .4
    not_prior = .6

    if len(sys.argv) < 2:
        print("Enter data file!")

    file = sys.argv[1]
    load_data(file)


if __name__ == '__main__':
    main()