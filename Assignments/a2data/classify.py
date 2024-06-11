import math
from collections import defaultdict

# Tokenizes text into character n-grams; applies case folding
def tokenize(text, n=3):
    text = text.lower()
    return [text[i:i+n] for i in range(len(text) - (n-1))]

# A most-frequent class baseline
class Baseline:
    def __init__(self, klasses):
        self.train(klasses)

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        self.mfc = sorted(klass_freqs, reverse=True, 
                          key=lambda x : klass_freqs[x])[0]
    
    def classify(self, test_instance):
        return self.mfc

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # Method will be one of 'baseline', 'lr', 'nb', or 'nbse'
    method = sys.argv[1]

    train_texts_fname = sys.argv[2]
    train_klasses_fname = sys.argv[3]
    test_texts_fname = sys.argv[4]
    
    train_texts = [x.strip() for x in open(train_texts_fname,
                                           encoding='utf8')]
    train_klasses = [x.strip() for x in open(train_klasses_fname,
                                             encoding='utf8')]
    test_texts = [x.strip() for x in open(test_texts_fname,
                                          encoding='utf8')]

    if method == 'baseline':
        classifier = Baseline(train_klasses)
        results = [classifier.classify(x) for x in test_texts]

    elif method == 'lr':
        # Use sklearn's implementation of logistic regression
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression

        # sklearn provides functionality for tokenizing text and
        # extracting features from it. This uses the tokenize function
        # defined above for tokenization (as opposed to sklearn's
        # default tokenization) so the results can be more easily
        # compared with those using NB.
        # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        count_vectorizer = CountVectorizer(analyzer=tokenize)

        # train_counts will be a DxV matrix where D is the number of
        # training documents and V is the number of types in the
        # training documents. Each cell in the matrix indicates the
        # frequency (count) of a type in a document.
        train_counts = count_vectorizer.fit_transform(train_texts)

        # Train a logistic regression classifier on the training
        # data. A wide range of options are available. This does
        # something similar to what we saw in class, i.e., multinomial
        # logistic regression (multi_class='multinomial') using
        # stochastic average gradient descent (solver='sag') with L2
        # regularization (penalty='l2'). The maximum number of
        # iterations is set to 1000 (max_iter=1000) to allow the model
        # to converge. The random_state is set to 0 (an arbitrarily
        # chosen number) to help ensure results are consistent from
        # run to run.
        # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        lr = LogisticRegression(multi_class='multinomial',
                                solver='sag',
                                penalty='l2',
                                max_iter=1000,
                                random_state=0)
        clf = lr.fit(train_counts, train_klasses)

        # Transform the test documents into a DxV matrix, similar to
        # that for the training documents, where D is the number of
        # test documents, and V is the number of types in the training
        # documents.
        test_counts = count_vectorizer.transform(test_texts)
        # Predict the class for each test document
        results = clf.predict(test_counts)

    elif method == 'nb':
        trigrams = []
        results = []
        trigram_counts = {}
        class_counts = {}
        bigdoc = {}

        for text, label in zip(train_texts, train_klasses):
            trigrams = tokenize(text)

            for trigram in trigrams:
                if label not in bigdoc:
                    bigdoc[label] = []
                else:
                    bigdoc[label].append(trigram)
                if label not in trigram_counts:
                    trigram_counts[label] = {}
                if trigram not in trigram_counts[label]:
                    trigram_counts[label][trigram] = 0
                trigram_counts[label][trigram] += 1
                # class_counts
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1

        priors = {}
        word_counts = {}
        log_prior = {}
        log_likelihood = {}
        likelihoods = {}

        for line in train_klasses:
            word = line.strip()
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        total_count = len(train_klasses)

        # print(word_counts)
        # print(total_count)

        unique_items = set()
        for label in trigram_counts:
            for value in trigram_counts[label]:
                unique_items.add(value)
        unique_size = len(set(unique_items))

        for class_label, count in word_counts.items():
            prior = count / total_count
            priors[class_label] = prior
            # print(class_label, ": ", priors[class_label])
            log_prior[class_label] = math.log(priors[class_label])
            vocab = set(unique_items)
            # print("vocab:", vocab)

            likelihoods[class_label] = {}
            log_likelihood[class_label] = {}

            total_trigrams_per_class = sum(trigram_counts[class_label].values()) + unique_size

            for trigram in vocab:
                if trigram in trigram_counts[class_label]:
                    trigram_count = trigram_counts[class_label][trigram] + 1
                else:
                    trigram_count = 1

                likelihoods[class_label][trigram] = trigram_count / total_trigrams_per_class
                log_likelihood[class_label][trigram] = math.log(likelihoods[class_label][trigram])
                # print(class_label, " ### ", trigram, "### : ", log_likelihood[class_label][trigram])

            # print("Verification - Sum of Probabilities:", class_label, ": ", sum(likelihoods[class_label].values()))

        # input_text = "Shibanuma"
        for input_text in test_texts:
            input_trigrams = tokenize(input_text)

            final_sum = {}
            for label in train_klasses:
                # print("label: ", label)
                final_sum[label] = log_prior[label]
                # print("final_sum[", label, "]: ", final_sum[label])
                for trigram in input_trigrams:
                    # print("trigram: ", trigram)
                    if trigram in likelihoods[label]:
                        final_sum[label] += log_likelihood[label][trigram]

            predicted_class = max(final_sum, key=final_sum.get)
            # print("predicted_class: ", predicted_class)
            results.append(predicted_class)

    elif method == 'nbse':
        trigrams = []
        results = []
        trigram_counts = {}
        class_counts = {}
        bigdoc = {}

        for text, label in zip(train_texts, train_klasses):
            trigrams = tokenize("<"+text+">")
            for trigram in trigrams:
                if label not in bigdoc:
                    bigdoc[label] = []
                else:
                    bigdoc[label].append(trigram)
                if label not in trigram_counts:
                    trigram_counts[label] = {}
                if trigram not in trigram_counts[label]:
                    trigram_counts[label][trigram] = 0
                trigram_counts[label][trigram] += 1
                # class_counts
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1

        priors = {}
        word_counts = {}
        log_prior = {}
        log_likelihood = {}
        likelihoods = {}

        for line in train_klasses:
            word = line.strip()
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

        total_count = len(train_klasses)

        unique_items = set()
        for label in trigram_counts:
            for value in trigram_counts[label]:
                unique_items.add(value)
        unique_size = len(set(unique_items))

        for class_label, count in word_counts.items():
            prior = count / total_count
            priors[class_label] = prior
            log_prior[class_label] = math.log(priors[class_label])
            vocab = set(unique_items)

            likelihoods[class_label] = {}
            log_likelihood[class_label] = {}

            total_trigrams_per_class = sum(trigram_counts[class_label].values()) + unique_size

            for trigram in vocab:
                if trigram in trigram_counts[class_label]:
                    trigram_count = trigram_counts[class_label][trigram] + 1
                else:
                    trigram_count = 1

                likelihoods[class_label][trigram] = trigram_count / total_trigrams_per_class
                log_likelihood[class_label][trigram] = math.log(likelihoods[class_label][trigram])
                # print(class_label, " ### ", trigram, "### : ", log_likelihood[class_label][trigram])

        for input_text in test_texts:
            input_trigrams = tokenize("<"+input_text+">")
            final_sum = {}
            for label in train_klasses:
                final_sum[label] = log_prior[label]
                for trigram in input_trigrams:
                    if trigram in likelihoods[label]:
                        final_sum[label] += log_likelihood[label][trigram]

            predicted_class = max(final_sum, key=final_sum.get)
            results.append(predicted_class)

    for r in results:
        print(r)
