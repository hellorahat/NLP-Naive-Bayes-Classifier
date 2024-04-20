from sys import argv
import json
import os
from pathlib import Path

def process_data(directory):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/" + os.path.basename(Path(directory)) + ".txt", "w") as outfile:
        for label in Path(directory).iterdir():
            for doc in Path(label).iterdir():
                feature_vector = {}
                with open(doc, "r") as file:
                    text = file.read()
                    
                    # lowercase the content
                    text = text.lower()
                    
                    # separate the punctuation from content
                    text = separate_punctuation(text)
                    
                    for word in text.split():
                        feature_vector[word] = feature_vector.get(word, 0) + 1 # increase frequency of feature_vector when word occurrence is found
                
                # Write the label and feature vector to the output file
                outfile.write(os.path.basename(label) + " " + json.dumps(feature_vector) + "\n")
                    

def separate_punctuation(text):
    new_text = ""
    for char in text:
        if is_punctuation(char):
            new_text += " " + char + " "
        else:
            new_text += char
    return new_text

def is_punctuation(char):
    punctuation = ".!?,:;\"\'():-/"
    return char in punctuation

# main
process_data(argv[1])