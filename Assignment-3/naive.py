import os
import math
import numpy as np

path_train = "C:/Users/Vedant/Downloads/assignment3_train/train"
path_test = "C:/Users/Vedant/Downloads/assignment3_test/test"

total_size = 0
spam_size = 0
ham_size = 0

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

x = os.listdir(path_train)

spam_word_count={}
ham_word_count = {}
total_word_count = {}

for i in x:
    y = os.listdir(path_train + "/" + i)
    if i=="spam":
        for j in y:
            total_size += 1
            spam_size += 1
            f = path_train+"/"+ i + "/" + j
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
            f = path_train+"/"+ i + "/" + j
            file=open(f,"r", errors = 'ignore')
            for word in file.read().split():
                if word not in ham_word_count and word.isalpha():
                    ham_word_count[word] = 1
                    total_word_count[word] = 1
                elif word.isalpha():
                    ham_word_count[word] += 1
                    total_word_count[word] += 1
print("Total Word Count:",len(total_word_count))

total_words_spam = sum(spam_word_count.values())
total_words_ham = sum(ham_word_count.values())
no_vocabulary = len(total_word_count)
count_spam = 0
count_ham = 0
count_spam_test = 0
count_ham_test = 0
size_test = 0

# Naive Bayes with stop words

for i in x:
    y = os.listdir(path_test+"/"+ i)
    for j in y:
        test_sh = {}
        size_test += 1
        f = path_test+"/"+ i + "/" + j
        file=open(f,"r", errors = 'ignore')
        for word in file.read().split():
            if word not in test_sh and word.isalpha():
                test_sh[word] = 1
            elif word.isalpha():
                test_sh[word] += 1
        spam_probability = math.log(spam_size/total_size)
        ham_probability = math.log(ham_size/total_size)
        for k in test_sh:
            if spam_word_count.get(k) != None:
                spam_probability = spam_probability + math.log((spam_word_count.get(k)+1)/((total_words_spam)+(no_vocabulary)))
            else:
                spam_probability = spam_probability + math.log((1)/((total_words_spam)+(no_vocabulary)))
            if ham_word_count.get(k) != None:
                ham_probability = ham_probability + math.log((ham_word_count.get(k)+1)/((total_words_ham)+(no_vocabulary)))
            else:
                ham_probability = ham_probability + math.log((1)/((total_words_ham)+(no_vocabulary)))

            if spam_probability > ham_probability:
                count_spam = count_spam + 1
                if i=="spam":
                    count_spam_test = count_spam_test + 1
            elif ham_probability > spam_probability:
                count_ham = count_ham + 1
                if i=="ham":
                    count_ham_test = count_ham_test + 1

print("Accuracy",(count_spam_test+count_ham_test)/(count_spam+count_ham))

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

total_words_spam = sum(spam_word_count.values())
total_words_ham = sum(ham_word_count.values())
no_vocabulary = len(total_word_count)
count_spam = 0
count_ham = 0
count_spam_test = 0
count_ham_test = 0

# Naive Bayes without stop words
for i in x:
    y = os.listdir(path_test+"\\"+ i)
    for j in y:
        test_sh = {}
        f = path_test+"\\"+ i + "\\" + j
        file=open(f,"r", errors = 'ignore')
        for word in file.read().split():
            if word not in stopWords:
                if word not in test_sh and word.isalpha():
                    test_sh[word] = 1
                elif word.isalpha():
                    test_sh[word] += 1
        spam_probability = math.log(spam_size/total_size)
        ham_probability = math.log(ham_size/total_size)
        # print(spam_probability, ham_probability)
        for k in test_sh:
            if spam_word_count.get(k) != None:
                spam_probability = spam_probability + math.log((spam_word_count.get(k)+1)/((total_words_spam)+(no_vocabulary)))
            else:
                spam_probability = spam_probability + math.log((1)/((total_words_spam)+(no_vocabulary)))
            if ham_word_count.get(k) != None:
                ham_probability = ham_probability + math.log((ham_word_count.get(k)+1)/((total_words_ham)+(no_vocabulary)))
            else:
                ham_probability = ham_probability + math.log((1)/((total_words_ham)+(no_vocabulary)))

            if spam_probability > ham_probability:
                count_spam = count_spam + 1
                if i=="spam":
                    count_spam_test = count_spam_test + 1
            elif ham_probability > spam_probability:
                count_ham = count_ham + 1
                if i=="ham":
                    count_ham_test = count_ham_test + 1

print("Accuracy",(count_spam_test+count_ham_test)/(count_spam+count_ham))