# Bayesian Classifier

I wrote a program to classify sets of email features into spam and not-spam cases using the
Spambase data set and a naive Bayesian classifier.

First, using the training data, the mean and standard deviation is calculated for all the email
features, as well as the prior probabilities. These values are used with the Gaussian Naive
Bayes algorithm and that classification method is used on the test data. The results are then
compared to the true labels.

For the test data, it produced an accuracy of 78.4%, a precision of 68.6%, and a recall of 83.6%
(all values are rounded).

The confusion matrix:
![Frame 1](https://github.com/IntoTheVortex/Bayesian-Classifier/blob/main/confusion.png?raw=true)

The accuracy overall was fairly good, and the recall was impressive. The precision was
relatively low compared to the other values. This may be preferable in a spam filter, as the low
rate of false negatives leads to the desired result of less spam ending up in an inbox. However,
classifying non-spam as spam could lead to frustration.
I do not think that the features are truly independent. I think that there are groupings of these
features where inclusion or exclusion of some small number of features in the group determines
whether or not an email is spam. The classification would be more successful if it had the
capacity to account for these cases. The classifier does better than chance, but should be
improved.
The data set that this classifier was trained on was fairly limited in its scope and the number of
features per email. It does not seem extensible to other contexts - like other workplaces or
groups of students.
