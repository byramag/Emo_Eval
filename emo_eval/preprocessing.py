import json
import re
import emoji, emot
from spacy.lang.en.stop_words import STOP_WORDS


#Expands abbreviations in a sentence according to a abbreviation_mapping.
# expand word contractions (can't -> cannot) and shorthand (lol -> laugh out loud) -- minor or no impact
def expand_abbreviations(sentence, abbreviation_mapping): 
    abbreviations_pattern = re.compile('({})'.format('|'.join(abbreviation_mapping.keys())),  
                                        flags=re.IGNORECASE|re.DOTALL) 
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

#Replaces emoticons with their meanings
# emojis replacement -- good
# smileys replacement -- minor or no impact
def desmilify(text):
    emoticons = emot.emoticons(text)
    if type(emoticons) == dict:
        for loc,mean,value in zip(emoticons.get('location'),emoticons.get('mean'),emoticons.get('value')):
            text = text.replace(value,':%s:'%'_'.join(mean.split()))
    return text

def preprocess(instances):
    """TODO: documentation
    """
    with open('abbreviations.json', 'r') as abbr_data: 
        ABBREVIATION_MAP = json.load(abbr_data)

    # Separating the labels and strings into separate arrays & concatenating turns from bag of words
    row_strings = []
    labels = []
    for _, instance in instances.iterrows():
        # Concatenate all three conversation turns into single string
        instance_string = instance['turn1'] + ' ' + instance['turn2'] + ' ' + instance['turn3']

        # Cleaning
        instance_string = re.sub(r"\.\.+", r" ... ", instance_string.lower()) # Separate/truncate elipsis into their own tokens
        instance_string = re.sub(r"!+", r" ! ", instance_string) # Separate/truncate ! into their own tokens
        instance_string = re.sub(r"\?+", r" ? ", instance_string) # Separate/truncate ? into their own tokens
        instance_string = re.sub(r"[\,\'\"\~\`]", r"", instance_string) # Remove non descriptive punctuation
        instance_string = re.sub(r"(?<!\.)\.(?!\.)", r"", instance_string)
        instance_string = re.sub(r"([A-Za-z])\1{2,}\s", r"\1 ", instance_string) # Truncate repeating characters at the end of a word such as youuuuu -> you
        instance_string = re.sub(r"\s([A-Za-z]{2,3})\1{2,}\s", r" \1 ", instance_string) # Truncate repeating sequences such as hahaha -> haha

        instance_string = re.sub('([\U00010000-\U0010ffff])', r" \1 ", instance_string) # Insert spaces around emojis so they are counted separately
        
        # Mapping abbreviations to full versions
        instance_string = expand_abbreviations(instance_string, ABBREVIATION_MAP)

        # Remove stopwords with SpaCy
        # instance_string = " ".join([x for x in instance_string.split() if x not in STOP_WORDS])

        # Replacing emojis with text descriptions
        # instance_string = desmilify(instance_string)

        # Adding cleaned string and label for feature extraction
        row_strings.append(instance_string)
        labels.append(instance['label'])

    return row_strings, labels
