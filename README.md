# EmoContext

## Introduction

Entry for SemEval 2019 task 3

### Problem

The goal of this project is to build a classifier that can recognize the general feeling (happy, sad, angry, other) of a snippet of conversation between two parties. The project organizers provide training data from WhatsApp containing three turns of a conversation (one statement and two subsequent replies) and a gold standard label for what emotion the conversation holds.

With the growing prominence of messaging platforms like WhatsApp, Twitter, and Facebook, as well as digital agents, it is important for machines understand emotions in textual conversations and provide emotionally aware responses to users.

### Data Examples

```
User 1: "Bad	Bad bad!"
User 2: "That's the bad kind of bad."             // Sad
User 1: "I have no gf"
```

```
User 1: "OK friend's"
User 2: "YA YA VERY GOOD FRIEND"                  // Happy
User 1: "Very nice friends"
```

```
User 1: "I like dominos pizza"
User 2: "legit same but there's pizza hut"        // Angry
User 1: "I hate pizza hut"
```

## Prerequisites

This project requires Python version 3.5 or greater, download the latest version from [their website](https://www.python.org/downloads/)

## Install

All installation steps are handled by running the included script `install.sh`.  
This will create a Python virtual environment in the `./venv` directory and install all required packages into it. Run the script on the command line by navigating to the project directory and running the command `bash install.sh`

## Run

You can run the program with the settings we found to be best by executing the command `bash runit.sh` from the project directory.

If you wish to run a more customized version, run `python main.py [args...]` with any of the command line parameters specified below

## Command Line Flags

* Misc:
  * `[-i | --input] <path>`
    * Specify the path to the training data file
  * `-s <percentage>`
    * Specify a training sample size to use as a percentage in the range (0, 1]
  * `--folds <number>`
    * Specify the number of folds to use for cross-validation
* Feature Extraction Component Flags (at least one must be active, tfidf by default):
  * `--no-tfidf`
    * Disables tf-idf vectorization
  * `--embeddings`
    * Enables word embedding vectorization
  * `--emoji-vectors`
    * Enables emoji vectorization
* Preprocessing Component Flags:
  * `--no-clean`
    * Disables manual regex data cleaning
  * `--exp-abbrs`
    * Enables abbreviation expansion
  * `--desmile`
    * Enables emoticon substitution
  * `--rm-stopwords`
    * Enables stopword removal
  * `--only-adjs`
    * Enables POS trimming

## Team members

- Abbie Byram
  - Wrote emo_eval.py, researched best classifier options for this problem, evaluated results from various features and model parameters
- Bobby Best
  - Handled install & execution scripts, along with README and other repo organization
  - Wrote emoji vectorization
- Revathi Keerthi
  - Worked on Preprocessing(contraction mapping, emoji/emoticon replacement), feature extraction, included SGD classifier and hyperparameter optimization.
