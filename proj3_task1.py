import sklearn as sk
import numpy as np
import os,sys
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy import stats
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

import warnings
warnings.filterwarnings("ignore")

# example code for plotting curves with error bars
x = np.arange(0.1, 4, 0.5)
y = np.exp(-x)

# add some arbitrarily generated fake y error bars
yerr = 0.1 + 0.1*np.sqrt(x)

plt.figure()
#plt.plot(x, y)
plt.errorbar(x, y, yerr=yerr, capsize=4)
plt.xlabel ("Parameter X")
plt.ylabel ("f1_macro")
plt.title('Sample curve with error bars')

import matplotlib.pyplot as plt
import numpy as np

X = ['KNN','SVC']
orig_mean = [20,25]
normalized_mean = [15,35]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, orig_mean, 0.4, label = 'No Normalization')
plt.bar(X_axis + 0.2, normalized_mean, 0.4, label = 'Normalization')

yerr_orig = [1.3, 2.3]
yerr_norm = [1.3, 2.3]

plt.errorbar(
    X_axis-0.2,
    orig_mean,
    yerr = yerr_orig,
    fmt="o",
    capsize=4,
    color='red'
)

plt.errorbar(
    X_axis + 0.2,
    normalized_mean,
    yerr = yerr_norm,
    fmt="o",
    capsize=4,
    color='red'
)

plt.xticks(X_axis, X)
plt.xlabel("Classifiers")
plt.ylabel("F1-Macro")
plt.title("Effect of Feature Normalization")
plt.legend()

print()
#plt.show()


# Task 1
def paired_ttest(data1, data2, alpha=0.05, alternative='two-sided'):
    if len(data1) != len(data2):
        raise Exception("The size of data1 and data2 should be the same")
    
    data1, data2 = np.array(data1), np.array(data2)
    mean_diff = np.mean(data1 - data2)
    std_diff = np.std(data1 - data2, ddof=1)
    se = std_diff / np.sqrt(len(data1))
    t_stat = mean_diff / se
    df = len(data1) - 1
    
    if alternative == 'two-sided':
        p = 2 * stats.t.sf(np.abs(t_stat), df)
    elif alternative == 'less':
        p = stats.t.cdf(t_stat, df)
    elif alternative == 'greater':
        p = stats.t.sf(t_stat, df)
    
    cv = stats.t.ppf(1 - alpha/2, df) if alternative == 'two-sided' else stats.t.ppf(1 - alpha, df)
    return t_stat, p, df, cv

# Task 1.1 - Visualize the results
def plot_results(scores, scores_norm, classifier_name):
    labels = ['Original', 'Normalized']
    means = [np.mean(scores), np.mean(scores_norm)]
    errors = [np.std(scores), np.std(scores_norm)]

    x = np.arange(len(labels))
    plt.bar(x, means, yerr=errors, align='center', alpha=0.7, ecolor='black', capsize=10)
    plt.ylabel('F1 Macro Score')
    plt.xticks(x, labels)
    plt.title(f'Effect of Normalization on {classifier_name}')
    plt.show()

def evaluate_classifier(clf, X, y, kfolds):
    scores = cross_val_score(clf, X, y, cv=kfolds, scoring='f1_macro')
    return scores

def evaluate_feature_selection(k_values, selection_method, classifier, X, y, kfolds):
    scores = []
    for k in k_values:
        selector = SelectKBest(selection_method, k=k)
        X_new = selector.fit_transform(X, y)
        
        score = np.mean(cross_val_score(classifier, X_new, y, cv=kfolds, scoring='f1_macro'))
        scores.append(score)
        
    return scores

def plot_feature_selection_results(k_values, scores, title):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, scores, marker='o')
    plt.title(title)
    plt.xlabel('Number of Features (k)')
    plt.ylabel('F1 Macro Score')
    plt.grid(True)
    plt.xticks(k_values)
    plt.show()

feature_vectors, targets = load_svmlight_file(R"e:\Schoolstuff\training.TFIDF") # Load the dataset

mn_nb = MultinomialNB()
bernoulli_nb = BernoulliNB()

kfolds = KFold(n_splits=5, shuffle=True, random_state=1234) # Create K-Folds cross-validator
knn = KNeighborsClassifier(n_neighbors=10) 
svm = SVC()

from sklearn.preprocessing import MaxAbsScaler

# Normalize feature vectors
scaler = MaxAbsScaler().fit(feature_vectors)
X_normalized = scaler.transform(feature_vectors)


# Function to evaluate classifiers


# Evaluate classifiers without normalization
scores_knn = evaluate_classifier(knn, feature_vectors, targets, kfolds)
scores_svm = evaluate_classifier(svm, feature_vectors, targets, kfolds)

# Evaluate classifiers with normalization
scores_knn_norm = evaluate_classifier(knn, X_normalized, targets, kfolds)
scores_svm_norm = evaluate_classifier(svm, X_normalized, targets, kfolds)



plot_results(scores_knn, scores_knn_norm, "kNN")
plot_results(scores_svm, scores_svm_norm, "SVC")


# Evaluate classifiers
scores_mn_nb = cross_val_score(mn_nb, feature_vectors, targets, cv=kfolds, scoring='f1_macro')
scores_bernoulli_nb = cross_val_score(bernoulli_nb, feature_vectors, targets, cv=kfolds, scoring='f1_macro')

# Print scores for each classifier
print(f"Multinomial Naive Bayes F1 Macro Score: Mean={np.mean(scores_mn_nb):.2f}, Std Dev={np.std(scores_mn_nb):.2f}")
print(f"Bernoulli Naive Bayes F1 Macro Score: Mean={np.mean(scores_bernoulli_nb):.2f}, Std Dev={np.std(scores_bernoulli_nb):.2f}")

# Visualize comparison of classifiers
X = ['MultinomialNB', 'BernoulliNB']
means = [np.mean(scores_mn_nb), np.mean(scores_bernoulli_nb)]
stds = [np.std(scores_mn_nb), np.std(scores_bernoulli_nb)]


t_stat, p_value, df, cv = paired_ttest(scores_mn_nb, scores_bernoulli_nb, alpha=0.05, alternative='greater')
print("The test statistic: ", t_stat)
print("The p value:%.8f" % p_value)
if p_value < 0.05:
    print("=> Reject the null hypothesis")
else:
    print("=> Fail to reject the null hypothesis, i.e., the two classifiers are not statistically significantly different")

k_values = [100, 200, 500, 1000, 2000, 3000]

# Multinomial Naive Bayes with Chi-squared
mn_chi_scores = evaluate_feature_selection(k_values, chi2, MultinomialNB(), feature_vectors, targets, kfolds)

# Multinomial Naive Bayes with Mutual Information
mn_mi_scores = evaluate_feature_selection(k_values, mutual_info_classif, MultinomialNB(), feature_vectors, targets, kfolds)

# Bernoulli Naive Bayes with Chi-squared
bn_chi_scores = evaluate_feature_selection(k_values, chi2, BernoulliNB(), feature_vectors, targets, kfolds)

# Bernoulli Naive Bayes with Mutual Information
bn_mi_scores = evaluate_feature_selection(k_values, mutual_info_classif, BernoulliNB(), feature_vectors, targets, kfolds)

plot_feature_selection_results(k_values, mn_chi_scores, "Multinomial NB Performance with Chi-squared")
plot_feature_selection_results(k_values, mn_mi_scores, "Multinomial NB Performance with Mutual Information")
plot_feature_selection_results(k_values, bn_chi_scores, "Bernoulli NB Performance with Chi-squared")
plot_feature_selection_results(k_values, bn_mi_scores, "Bernoulli NB Performance with Mutual Information")
