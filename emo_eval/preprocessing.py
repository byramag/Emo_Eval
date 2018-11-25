import json
import re
import emoji, emot


def preprocess(instances):
    """TODO: documentation
    """
    # Separating the labels and strings into separate arrays & concatenating turns from bag of words
    row_strings = []
    labels = []
    for _, instance in instances.iterrows():
        bow = instance['turn1'] + ' ' + instance['turn2'] + ' ' + instance['turn3']
        bow = re.sub(r"\.{2,}", "\.\.\.", bow) # Separate/truncate elipsis into their own tokens
        bow = re.sub(r"!{2,}", " ! ", bow) # Separate/truncate ! into their own tokens
        bow = re.sub(r"\?{2,}", " ? ", bow) # Separate/truncate ? into their own tokens
        bow = re.sub(r"(?<!\.)\.(?!\.)", "", bow) # Remove periods
        bow = re.sub(r"[,\*]", "", bow)

        emo_reg = '(' + '|'.join(['|'.join(x.split()) for x in emot.EMO_UNICODE.values()]) + ')'
        emo_reg = re.sub(r'\*\|', "", emo_reg)
        bow = re.sub(emo_reg, r" \1 ", bow)

        row_strings.append(bow)
        labels.append(instance['label'])
    
    with open('abbreviations.json', 'r') as abbr_data: 
        ABBREVIATION_MAP = json.load(abbr_data)

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
        
    row_strings = [desmilify(emoji.demojize(expand_abbreviations(txt, ABBREVIATION_MAP))) for txt in row_strings]

    return row_strings, labels
