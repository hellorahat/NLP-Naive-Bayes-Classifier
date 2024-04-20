import os
import json
from sys import argv
import math

class NB:
    def __init__(self):
        self.prior_probability = {}
        self.conditional_probability = {}
        self.distinct_classes = set()
        self.distinct_words = set()
        self.overall_words_in_class = {}
    def train(self, training_file, parameter_file):
        # Learn the priors and collect counts
        with open(training_file, "r") as file:
            lines = file.readlines()
            
            label_dict = {}  # N_c
            total_class_count = 0  # N
            specific_word_count_in_class = {}  # count(w,c)

            for line in lines:
                parts = line.split()
                label = parts[0]
                label_dict[label] = label_dict.get(label, 0) + 1  # N_c
                total_class_count += 1  # N
                
                self.distinct_classes.add(label)
                feature_vector = json.loads(' '.join(parts[1:]))

                for word, frequency in feature_vector.items():
                    specific_word_count_in_class[word] = specific_word_count_in_class.get(word, {})
                    specific_word_count_in_class[word][label] = specific_word_count_in_class[word].get(label, 0) + frequency  # count(w,c)
                    self.overall_words_in_class[label] = self.overall_words_in_class.get(label, 0) + frequency  # count(c)
                    self.distinct_words.add(word)  # |V|

        # Calculate prior probabilities
        for label in self.distinct_classes:
            self.prior_probability[label] = label_dict[label] / total_class_count
            # print(f"label:{label} prior probability = {self.prior_probability[label]}")

        # Learn the conditional probabilities
        for word in self.distinct_words:
            self.conditional_probability[word] = {}
            for label in self.distinct_classes:
                self.conditional_probability[word][label] = (specific_word_count_in_class[word].get(label, 0) + 1) / (self.overall_words_in_class[label] + len(self.distinct_words))
                # print(f"self.conditional_probability[{word}][{label}] = ({specific_word_count_in_class[word].get(label, 0) + 1}) / ({self.overall_words_in_class[label]} + {len(self.distinct_words)})")
                # print(self.conditional_probability[word][label])

        # Save parameters
        model_parameters = {
            "prior_probability": self.prior_probability,
            "conditional_probability": self.conditional_probability,
            "dinstinct_classes": list(self.distinct_classes), # convert to list so its supported by json
            "distinct_words": list(self.distinct_words), # convert to list
            "overall_words_in_class": self.overall_words_in_class
        }
        with open(parameter_file, "w") as file:
            json.dump(model_parameters, file)
    
    def test(self, testing_file, output_file):
        outFile = open(output_file, "w")
        with open(testing_file, "r") as file:
            lines = file.readlines()
            correct_predictions = 0
            total_predictions = 0
            for line in lines:
                parts = line.split()
                true_class = parts[0]
                feature_vector = json.loads(' '.join(parts[1:]))
                class_prediction = {}
                for label in self.distinct_classes:
                    class_prediction[label] = self.prior_probability[label]
                    for word, frequency in feature_vector.items():
                        if word in self.conditional_probability:
                            initial = class_prediction[label]
                            class_prediction[label] += frequency * math.log(self.conditional_probability[word][label])
                            # print(f"class_prediction[{label}] = {initial} * {self.conditional_probability[word][label]}^{frequency} = {class_prediction[label]}")
                        else:
                            initial = class_prediction[label]
                            class_prediction[label] += frequency * math.log(1/(self.overall_words_in_class[label]+len(self.distinct_words)))
                            # print(f"class_prediction[{label}] = {initial} * (1/({self.overall_words_in_class[label]} + {len(self.distinct_words)} = {class_prediction[label]}))")
                predicted_class = max(class_prediction, key=class_prediction.get)
                outFile.write(f"Predicted class:{predicted_class}, True class: {true_class}, Document contents: {json.dumps(feature_vector)}\n")
                # print(f"Label: {label}   Word: {word}   Frequency: {frequency}")
                total_predictions += 1
                if predicted_class == true_class:
                    correct_predictions += 1
        outFile.write(f"Overall accuracy: {correct_predictions} / {total_predictions} = {correct_predictions/total_predictions}\n")
        outFile.close()


NB_model = NB()

training_file = argv[1]
testing_file = argv[2]
parameter_file = argv[3]
output_file = argv[4]

NB_model.train(training_file, parameter_file)
NB_model.test(testing_file, output_file)