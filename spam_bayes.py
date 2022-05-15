import sys
import math
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as pyplt
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

def naive_bayes_classifier(data, mu_spam, mu_not, sig_spam, sig_not, priors):
    correct_total = 0
    incorrect_total = 0
    confusion = np.zeros(shape=(2,2), dtype=int)

    #Feature probabilities given class:
    feat_prob_1 = np.zeros(shape=data.shape)
    feat_prob_0 = np.zeros(shape=data.shape)
    #for spam
    for i in range(len(data)):
        for j in range(data.shape[1]):
            sig = sig_spam[j] if sig_spam[j] > 0 else .0001
            feat_prob_1[i][j] = (1 / math.sqrt(2 * math.pi)) * np.exp(-(math.pow((data[i][j] - mu_spam[j]), 2)  / (2 * math.pow(sig, 2)) ))
    #for not spam
    for i in range(len(data)):
        for j in range(data.shape[1]):
            sig = sig_not[j] if sig_not[j] > 0 else .0001
            feat_prob_0[i][j] = (1 / math.sqrt(2 * math.pi)) * np.exp(-(math.pow((data[i][j] - mu_not[j]), 2)  / (2 * math.pow(sig, 2)) ))

    #class prediction:
    for i in range(len(data)):
        total_1 = math.log(priors[1])
        total_0 = math.log(priors[0])
        for j in range(data.shape[1]-1):
            if (feat_prob_1[i][j]) == 0:
                total_1 += 0
            else:
                total_1 += math.log(feat_prob_1[i][j])
            if (feat_prob_0[i][j]) == 0:
                total_0 += 0
            else:
                total_0 += math.log(feat_prob_0[i][j])

        predicted = 1 if total_1 > total_0 else 0

        if predicted == data[i][57]:
            correct_total += 1
        else:
            incorrect_total += 1
        
        x = int(data[i][57])
        confusion[x][predicted] += 1

    return correct_total, incorrect_total, confusion


#calculate the mean and standard deviation for the training data
def calc_feature_values(training_data):
    #separate out the two classes 
    spam = training_data[training_data[:,57] == 1]
    not_spam = training_data[training_data[:,57] == 0]

    #get mean of columns
    averages_spam = spam.mean(axis=0)
    averages_not = not_spam.mean(axis=0)

    #get standard deviation
    sigma_spam = np.std(spam, axis=0)
    sigma_not = np.std(not_spam, axis=0)

    #get priors
    priors = []
    priors.append(len(not_spam)/len(training_data))
    priors.append(len(spam)/len(training_data))

    return averages_spam, averages_not, sigma_spam, sigma_not, priors




def load_data(datafile):
    #pd indexes by column, row
    data = pd.read_csv(datafile, header=None)

    #spam 40%
    class_1_data = data.loc[data[57] == 1]
    train_data_1 = class_1_data[:len(class_1_data)//2]
    test_data_1 = class_1_data[len(class_1_data)//2:]

    #not-spam 60%
    class_2_data = data.loc[data[57] == 0]
    train_data_2 = class_2_data[:len(class_2_data)//2]
    test_data_2 = class_2_data[len(class_2_data)//2:]

    #randomize, convert to numpy
    train_data_series = pd.concat([train_data_1, train_data_2], ignore_index=True)
    train_data_series = train_data_series.sample(len(train_data_series))
    train_data = train_data_series.to_numpy()

    test_data_series = pd.concat([test_data_1, test_data_2], ignore_index=True)
    test_data_series = test_data_series.sample(len(test_data_series))
    test_data = test_data_series.to_numpy()

    return train_data, test_data



def main():
    #check for a command line argument
    if len(sys.argv) < 2:
        print("Enter data file!")
        sys.exit(1)

    #read in the data
    file = sys.argv[1]
    train_set, test_set = load_data(file)

    #calculate features from training data, use to classify test data
    mu_spam, mu_not, sigma_spam, sigma_not, priors = calc_feature_values(train_set)
    right, wrong, conf = naive_bayes_classifier(test_set, mu_spam, mu_not, sigma_spam, sigma_not, priors)

    #display results
    print("# correct:", right, "# incorrect:", wrong)
    print(right/(right+wrong))

    confusion_data = pd.DataFrame(conf, range(2), range(2))
    sb.heatmap(confusion_data, cmap="BuPu", annot=True, fmt='d')
    pyplt.xlabel('Predicted Class')
    pyplt.ylabel('True Class')
    pyplt.show()


if __name__ == '__main__':
    main()