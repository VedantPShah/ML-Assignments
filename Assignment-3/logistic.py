import os
import math
import numpy as np

path_train = "C:/Users/Vedant/Downloads/assignment3_train/train"
path_test = "C:/Users/Vedant/Downloads/assignment3_test/test"
iteration = 50
lamda = 0.01
eta = 0.01

stopWords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
             "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
             "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't",
             "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from",
             "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd",
             "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
             "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
             "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of",
             "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
             "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
             "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
             "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
             "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
             "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
             "you're", "you've", "your", "yours", "yourself", "yourselves"]

total_size = 0
spam_size = 0
ham_size = 0

#Logistic Regression with Stop words

x = os.listdir(path_train)

spam_word_count={}
ham_word_count = {}
total_word_count = {}

for i in x:
    y = os.listdir(path_train+"\\" + i)
    if i=="spam":
        for j in y:
            total_size += 1
            spam_size += 1
            f = path_train+"\\"+ i + "\\" + j
            file=open(f,"r", errors = 'ignore')
            for word in file.read().split():
                if word not in spam_word_count and word.isalpha():
                    spam_word_count[word] = 1
                    total_word_count[word] = 1
                elif word.isalpha():
                    spam_word_count[word] += 1
                    total_word_count[word] += 1
    else:
        for j in y:
            total_size += 1
            ham_size += 1
            f = path_train+"\\"+ i + "\\" + j
            file=open(f,"r", errors = 'ignore')
            for word in file.read().split():
                if word not in ham_word_count and word.isalpha():
                    ham_word_count[word] = 1
                    total_word_count[word] = 1
                elif word.isalpha():
                    ham_word_count[word] += 1
                    total_word_count[word] += 1
print("Total Word Count:",len(total_word_count))

total_word_count_spam = sum(spam_word_count.values())
total_word_count_ham = sum(ham_word_count.values())
novocab = len(total_word_count)
count_spam = 0
count_ham = 0
count_spam_test = 0
count_ham_test = 0
size_test = 0

logistic_total_word_count = list(total_word_count.keys())
mat = np.zeros((total_size,len(logistic_total_word_count)+1))
ind = 0

for i in x:
    y = os.listdir(path_train+"\\"+ i)
    for j in y:
        logistic_word_count = {}
        f = path_train+"\\"+ i + "\\" + j
        file=open(f,"r", errors = 'ignore')
        for word in file.read().split():
            if word not in logistic_word_count and word.isalpha():
                logistic_word_count[word] = 1
            elif word.isalpha():
                logistic_word_count[word] += 1
        for k in logistic_word_count:
            mat[ind][logistic_total_word_count.index(k)] = logistic_word_count[k]
        if i=="spam":
            mat[ind][len(logistic_total_word_count)] = 1
        ind = ind + 1
        
def prob(w,x):
    s = 0
    for i in range(len(x)):
        s = s + (w[i]*x[i])
    try:
        p = math.exp(w[0]+s)/(1 + math.exp(w[0]+s))
    except:
        p = 1
    return p

w_new = np.ones(len(total_word_count)+1)
w = np.ones(len(total_word_count)+1)
probab = np.ones(mat.shape[0])
for k in range(iteration):
    w = w_new.copy()
    w_new = np.ones(len(total_word_count)+1)
    for l in range(mat.shape[0]):
        probab[l] = prob(w,mat[l])
    for i in range(len(w)):
        temp = 0
        for j in range(mat.shape[0]):
            temp = temp + mat[j][i]*((mat[j][mat.shape[1]-1])-probab[j])
        w_new[i] = w[i]+ (lamda * temp) - (lamda*eta*w[i])
        
mat_test = np.zeros((size_test,len(logistic_total_word_count)+1))
ind = 0
for i in x:
    y = os.listdir(path_test+"\\"+ i)
    for j in y:
        logistic_word_count = {}
        f = path_test+"\\"+ i + "\\" + j
        file=open(f,"r", errors = 'ignore')
        for word in file.read().split():
            if word not in logistic_word_count and word.isalpha():
                logistic_word_count[word] = 1
            elif word.isalpha():
                logistic_word_count[word] += 1
        for k in logistic_word_count:
            if k in logistic_total_word_count:
                mat_test[ind][logistic_total_word_count.index(k)] = logistic_word_count[k]
        if i=="spam":
            mat_test[ind][len(logistic_total_word_count)] = 1
        ind = ind + 1

total_ham= 0
total_spam = 0
total_test = 0
# lamda = 0.001
for i in range(mat_test.shape[0]):
    s = 0
    for j in range(mat_test.shape[1]-1):
        s = s + (w_new[j]*mat_test[i][j])
    s = s + w[0]
    total_test += 1
    if mat_test[i][len(logistic_total_word_count)]==1 and s>0:
        total_spam += 1
    elif mat_test[i][len(logistic_total_word_count)]==0 and s<0:
        total_ham+= 1
print("Accuracy:",(total_spam+total_ham)/total_test)

#Logistic Regression without Stop words
total_word_count_spam = sum(spam_word_count.values())
total_word_count_ham = sum(ham_word_count.values())
novocabab = len(total_word_count)
count_spam = 0
count_ham = 0
count_spam_test = 0
count_ham_test = 0

x = os.listdir(path_train)

spam_word_count={}
ham_word_count = {}
total_word_count = {}

for i in x:
    y = os.listdir(path_train+"\\"+ i)
    if i=="spam":
        for j in y:
            f = path_train+"\\"+ i + "\\" + j
            file=open(f,"r", errors = 'ignore')
            for word in file.read().split():
                if word not in stopWords:
                    if word not in spam_word_count and word.isalpha():
                        spam_word_count[word] = 1
                        total_word_count[word] = 1
                    elif word.isalpha():
                        spam_word_count[word] += 1
                        total_word_count[word] += 1
    else:
        for j in y:
            f = path_train+"\\"+ i + "\\" + j
            file=open(f,"r", errors = 'ignore')
            for word in file.read().split():
                if word not in stopWords:
                    if word not in ham_word_count and word.isalpha():
                        ham_word_count[word] = 1
                        total_word_count[word] = 1
                    elif word.isalpha():
                        ham_word_count[word] += 1
                        total_word_count[word] += 1

print("Total Word Count:",len(total_word_count))

logistic_total_word_count = list(total_word_count.keys())
mat = np.zeros((total_size,len(logistic_total_word_count)+1))
ind = 0
for i in x:
    y = os.listdir(path_train+"\\"+ i)
    for j in y:
        logistic_word_count = {}
        f = path_train+"\\"+ i + "\\" + j
        file=open(f,"r", errors = 'ignore')
        for word in file.read().split():
            if word not in stopWords:
                if word not in logistic_word_count and word.isalpha():
                    logistic_word_count[word] = 1
                elif word.isalpha():
                    logistic_word_count[word] += 1
        for k in logistic_word_count:
            mat[ind][logistic_total_word_count.index(k)] = logistic_word_count[k]
        if i=="spam":
            mat[ind][len(logistic_total_word_count)] = 1
        ind = ind + 1
        
w_new = np.ones(len(total_word_count)+1)
w = np.ones(len(total_word_count)+1)
for k in range(iteration):
    w = w_new.copy()
    w_new = np.ones(len(total_word_count)+1)
    for l in range(mat.shape[0]):
        probab[l] = prob(w,mat[l])
    for i in range(len(w)):
        temp = 0
        for j in range(mat.shape[0]):
            temp = temp + mat[j][i]*((mat[j][mat.shape[1]-1])-probab[j])
        w_new[i] = w[i]+ (lamda * temp) - (lamda*eta*w[i])

mat_test = np.zeros((size_test,len(logistic_total_word_count)+1))
ind = 0
for i in x:
    y = os.listdir(path_test+"\\"+ i)
    for j in y:
        logistic_word_count = {}
        f = path_test+"\\"+ i + "\\" + j
        file=open(f,"r", errors = 'ignore')
        for word in file.read().split():
            if word not in stopWords:
                if word not in logistic_word_count and word.isalpha():
                    logistic_word_count[word] = 1
                elif word.isalpha():
                    logistic_word_count[word] += 1
        for k in logistic_word_count:
            if k in logistic_total_word_count:
                mat_test[ind][logistic_total_word_count.index(k)] = logistic_word_count[k]
        if i=="spam":
            mat_test[ind][len(logistic_total_word_count)] = 1
        ind = ind + 1
        
total_ham = 0
total_spam = 0
total_test = 0
# lamda = 0.001
for i in range(mat_test.shape[0]):
    s = 0
    for j in range(mat_test.shape[1]-1):
        s = s + (w_new[j]*mat_test[i][j])
    s = s + w[0]
    total_test += 1
    if mat_test[i][len(logistic_total_word_count)]==1 and s>0:
        total_spam += 1
    elif mat_test[i][len(logistic_total_word_count)]==0 and s<0:
        total_ham += 1
print("Accuracy:",(total_spam+total_ham)/total_test)

