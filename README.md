# EmoContext

## Introduction

Entry for SemEval 2019 task 3

### Problem

The goal of this project is to build a classifier that can recognize the general feeling (happy, sad, angry, other) of a snippet of conversation between two parties. The project organizers provide training data from WhatsApp containing three turns of a conversation (one statement and two subsequent replies) and a gold standard label for what emotion the conversation holds.

With the growing prominence of messaging platforms like WhatsApp, Twitter, and Facebook, as well as digital agents, it is important for machines understand emotions in textual conversations and provide emotionally aware responses to users.

### Examples

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

You can run the program by executing the command `bash runit.sh` from the project directory.

## Team members

- Abbie Byram
  - Wrote emo_eval.py, researched best classifier options for this problem, evaluated results from various features and model parameters
- Bobby Best
  - Handled install & execution scripts, along with README and other repo organization
- Revathi Keerthi
  - Did some research on this task and prepared slides for stage1 presentation
