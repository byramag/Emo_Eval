import re
import json
import emot
import spacy
import numpy as np
from time import time
from spacy.lang.en.stop_words import STOP_WORDS

from .resources import ABBREVIATION_MAP


def expand_abbreviations(sentence, abbreviation_mapping):
    """Expands abbreviations/contractions in a sentence according to
    a given abbreviation_mapping, i.e. can't -> cannot and lol -> laugh out loud.
    Found to have minor to no impact

    Authors:
        Abbie
        Keerthi

    Arguments:
        sentence: string
            - Text to be processed
        abbreviation_mapping: dict
            - Dictionary of conversions, each appearance of a key
                in the sentence will be replace with the corresponding
                value
    
    Returns:
        sentence with abbreviations substituted
    """
    abbreviations_pattern = re.compile(
        '({})'.format('|'.join(abbreviation_mapping.keys())),
        flags=re.IGNORECASE|re.DOTALL
    )
 
    def expand_match(abbreviation):
        match = abbreviation.group(0)
        first_char = match[0]
        expanded_abbreviation = abbreviation_mapping.get(match) if abbreviation_mapping.get(match) else abbreviation_mapping.get(match.lower())
        if not expanded_abbreviation:
            return match
        expanded_abbreviation = first_char+expanded_abbreviation[1:]
        return expanded_abbreviation

    expanded_sentence = abbreviations_pattern.sub(expand_match, sentence)
    return expanded_sentence


def desmilify(text):
    """Replaces emoticons in a given piece of text with their meanings.

    Authors:
        Keerthi
    
    Arguments:
        text: string
            - The text to be processed
    
    Returns:
        text with emoticons substituted
    """
    emoticons = emot.emoticons(text)
    if type(emoticons) == dict:
        for mean,value in zip(emoticons.get('mean'),emoticons.get('value')):
            text = text.replace(value,':%s:'%'_'.join(mean.split()))
    return text


def preprocess(instances, clean_data=True, expand_abbrs=False, desmile=False, remove_stopwords=False, only_adjs=False):
    """First stage of pipeline, takes in training instances and applies
    preprocessing rules to them, returning two lists, one of the preprocessed
    samples and the other of the corresponding sample labels

    Authors:
        Abbie
        Bobby
    
    Arguments:
        instances: pandas dataframe
            - The dataframe containing the training data, keys 'id', turn1',
                'turn2', turn3', and 'label'
        clean_data: boolean (True)
            - Flag to apply regex manual preprocessing rules
        expand_abbrs: boolean (True)
            - Flag to expand abbreviations/contractions into full forms
        desmile: boolean (False)
            - Flag to substitute emoticons with their meanings
        remove_stopwords: boolean (False)
            - Flag to remove stopwords
        only_adjs: boolean (False)
            - Flag to remove all non-adjective tokens
        
    
    Returns:
        row_strings: string list
            - Training samples
        labels: string list
            - Training labels
    """
    print("Preprocessing data...", end="", flush=True); t = time()

    if only_adjs:
        nlp = spacy.load('en_core_web_lg')

    # Separating the labels and strings into separate arrays & concatenating turns from bag of words
    row_strings = []
    labels = []
    checkpoints = np.floor(np.multiply(range(1, 11), len(instances)/10))
    i = 0
    for _, instance in instances.iterrows():
        i += 1
        if i in checkpoints: # Print ongoing progress
            print("{}%...".format((np.where(checkpoints==i)[0][0]+1)*10), end='', flush=True)

        # Concatenate all three conversation turns into single string
        instance_string = instance['turn1'] + ' ' + instance['turn2'] + ' ' + instance['turn3']

        if clean_data: # Default True
            instance_string = re.sub(r"\.\.+", r" ... ", instance_string.lower()) # Separate/truncate elipsis into their own tokens
            instance_string = re.sub(r"!+", r" ! ", instance_string) # Separate/truncate ! into their own tokens
            instance_string = re.sub(r"\?+", r" ? ", instance_string) # Separate/truncate ? into their own tokens
            instance_string = re.sub(r"[\,\'\"\~\`]", r"", instance_string) # Remove non descriptive punctuation
            instance_string = re.sub(r"(?<!\.)\.(?!\.)", r"", instance_string) # Elipsis
            instance_string = re.sub(r"([A-Za-z])\1{2,}\s", r"\1 ", instance_string) # Truncate repeating characters at the end of a word such as youuuuu -> you
            instance_string = re.sub(r"\s([A-Za-z]{2,3})\1{2,}\s", r" \1 ", instance_string) # Truncate repeating sequences such as hahaha -> haha
            instance_string = re.sub('([\U00010000-\U0010ffff])', r" \1 ", instance_string) # Insert spaces around emojis so they are counted separately
        
        if expand_abbrs: # Default True
            instance_string = expand_abbreviations(instance_string, ABBREVIATION_MAP)

        if desmile: # Default False
            instance_string = desmilify(instance_string)

        if only_adjs: # Default True
            instance_string = " ".join([token.text for token in nlp(instance_string) if token.pos_ == 'ADJ'])

        if remove_stopwords: # Default False
            instance_string = " ".join([x for x in instance_string.split() if x not in STOP_WORDS])

        # Adding cleaned string and label for feature extraction
        row_strings.append(instance_string)
        labels.append(instance['label'])

    print("finished %0.3fsec\n" % (time()-t))
    return row_strings, labels
