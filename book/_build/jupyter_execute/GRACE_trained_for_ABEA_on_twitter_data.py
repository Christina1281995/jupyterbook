#!/usr/bin/env python
# coding: utf-8

# ## Aspect-Based Emotion Analysis on Twitter Data
# 
# 

# In a further step, a **more fine-grained** approach can be taken to ABSA, namely the classifcation of emotions rather than sentiments.
# 
# ##### Re-Training GRACE for emotions using the Twemlab Goldstandard
# 
# The twemlab goldstandard files are already labelled according to emotions. In this notebook they are consolidated into the four basic emotions:
# - happiness (includes love and beauty) 🙂
# - anger (includes disgust) 😠
# - sadness 😞
# - fear 😨
# 
# Aside from the consolidation of the emotion labels, this notebook uses helper functions to label each tweet according to its emotion-related aspect terms. The function splits the datasets into chunks of 100 tweets to save each annotated chunk during the process (for extra caution that the annotations are saved). Each tweet is shown to the user along with the labelled emotion. The user can enter how many aspect terms there are and then enters the aspect terms. The function then automatically identifies the beginning and inside of aspect terms and labels them according to the required input format of the GRACE model developed by [Luo et al. (2020)](https://arxiv.org/abs/2009.10557).
# 
# For clarity and assistance during the labelling, "aspect term" need to be clearly defined.
# - asepcts are regarded as a "general" aspect, which collectively refers to an entity and its aspects as "aspect" ([Zhang et al., 2022](https://arxiv.org/pdf/2203.01054.pdf))
# - aspect term a is the opinion target which appears in the given text, e.g., “pizza” in the sentence “The pizza is delicious.” When the target is implicitly expressed (e.g., “It is overpriced!”), we can denote the aspect term as a special one named “null” ([Zhang et al., 2022](https://arxiv.org/pdf/2203.01054.pdf))
# - here only aspects are labelled that are related to the given emotion 
# 
# 
# 
# <img align="right" src='https://github.com/Christina1281995/demo-repo/blob/main/ABEA.PNG?raw=true'>
# 
# <br>
# <br>
# 
# For full annotation implementation, see the section "Annotating Twemlab Goldstandard Files to Include Aspect Term Labels"
# 

# ### Imports

# In[1]:


import sys
# caution: path[0] is reserved for script path (or '' in REPL)
#sys.path.append('../sentiment-analysis/GRACE/')
sys.path.insert(0, '../GRACE/') # map to other folder with all GRACE training files

import torch
import os
from ate_asc_modeling_local_bert_file import BertForSequenceLabeling
import ate_asc_modeling_local_bert_file
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from ate_asc_features import ATEASCProcessor, convert_examples_to_features, get_labels
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tokenization import BertTokenizer
import argparse
import random
import time
import numpy as np
import torch.nn.functional as F

import csv
import urllib.request
import pandas as pd                                                    # data handling
import xml.etree.cElementTree as ET                                    # XML file parsing


# ### Load Trained GRACE Model
# 
# Load from last training Step and Epoch. Log messages from last training step and epoch: <br>
# 
# ```
# 
# Model saved to out_boston_ateacs/pytorch_model.bin.9
# 
# AT p:0.9883 	r:1.0000	f1:0.9941
# 
# AS p:0.9728 	r:0.9843	f1:0.9785
# 
# ```
# 

# In[2]:


# args set as per instructions by authors
# hard coded
args = argparse.Namespace(

    ## Required parameters
    data_dir='../GRACE/data/', 
    bert_model='bert-base-uncased',
    init_model=None,
    task_name="ate_asc",
    data_name="twemlab_all",
    train_file=None,
    valid_file=None,
    test_file=None,
    output_dir='out_testing/', 
    
    ## Other parameters
    cache_dir="",
    max_seq_length=128,
    do_train=False,
    do_eval=False, 
    do_lower_case=True, 
    train_batch_size=15, 
    gradient_accumulation_steps=1,
    eval_batch_size=32,
    learning_rate=3e-06,
    num_train_epochs=10, 
    warmup_proportion=0.1, 
    num_thread_reader=0, 
    no_cuda=False, 
    local_rank=-1, 
    seed=42, 
    fp16=False,
    loss_scale=0,
    verbose_logging=False, 
    server_ip='',
    server_port='', 
    use_ghl=True, 
    use_vat=False, 
    use_decoder=True, 
    num_decoder_layer=2, 
    decoder_shared_layer=3)


# In[3]:


random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# In[4]:


# manually setting device and gpu parameters
device = 'cuda'
n_gpu = 1
data_name = args.data_name.lower()


# In[5]:


task_name = args.task_name.lower()

task_config = {
    "use_ghl": True,
    "use_vat": False,
    "num_decoder_layer": 2,
    "decoder_shared_layer": 3,
}


# In[6]:


def dataloader_val(args, tokenizer, file_path, label_tp_list, set_type="val"):

    dataset = ATEASCProcessor(file_path=file_path, set_type=set_type)
    print("Loaded val file: {}".format(file_path))

    eval_features = convert_examples_to_features(dataset.examples, label_tp_list,
                                                 args.max_seq_length, tokenizer, verbose_logging=False)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_at_label_ids = torch.tensor([f.at_label_id for f in eval_features], dtype=torch.long)
    all_as_label_ids = torch.tensor([f.as_label_id for f in eval_features], dtype=torch.long)

    all_label_mask = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    all_label_mask_X = torch.tensor([f.label_mask_X for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_at_label_ids, all_as_label_ids,
                              all_label_mask, all_label_mask_X)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    return eval_dataloader, eval_data


# In[7]:


# load bert tokenizer (bert-base-uncased)
tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)


# In[8]:


DATASET_DICT={}
DATASET_DICT["lap"] = {"train_file":"laptops_2014_train.txt", "valid_file":"laptops_2014_trial.txt", "test_file":"laptops_2014_test.gold.txt"}
DATASET_DICT["res"] = {"train_file":"restaurants_union_train.txt", "valid_file":"restaurants_union_trial.txt", "test_file":"restaurants_union_test.gold.txt"}
for i in ["2014", "2015", "2016"]:
    DATASET_DICT["res{}".format(i)] = {"train_file": "restaurants_{}_train.txt".format(i), "valid_file": "restaurants_{}_trial.txt".format(i), "test_file": "restaurants_{}_test.gold.txt".format(i)}
for i in range(10):
    DATASET_DICT["twt{}".format(i+1)] = {"train_file":"twitter_{}_train.txt".format(i+1), "valid_file":"twitter_{}_test.gold.txt".format(i+1), "test_file":"twitter_{}_test.gold.txt".format(i+1)}
#Christina
# ADDED TWEMLAB GOLDSTANDARD
DATASET_DICT["twemlab_all"] = {"train_file":"twemlab_all_train.txt", "valid_file":"twemlab_all_trial.txt", "test_file":"twemlab_all_test.gold.txt"}
    


# In[9]:


if data_name in DATASET_DICT:
    args.train_file = DATASET_DICT[data_name]["train_file"]
    args.valid_file = DATASET_DICT[data_name]["valid_file"]
    args.test_file = DATASET_DICT[data_name]["test_file"]
else:
    assert args.train_file is not None
    assert args.valid_file is not None
    assert args.test_file is not None


# In[10]:


file_path = os.path.join(args.data_dir, args.train_file)
print(file_path)


# In[11]:


# ATEASCProcessor reads data and splits it into corpus and label list for ATE and ASC
dataset = ATEASCProcessor(file_path=file_path, set_type="train")
at_labels, as_labels = get_labels(dataset.label_tp_list)
label_tp_list = (at_labels, as_labels)

print("AT Labels are:", "["+", ".join(label_tp_list[0])+"]")
print("AS Labels are:", "["+", ".join(label_tp_list[1])+"]")
at_num_labels = len(label_tp_list[0])
as_num_labels = len(label_tp_list[1])
num_tp_labels = (at_num_labels, as_num_labels)

task_config["at_labels"] = label_tp_list[0]


# In[12]:


at_label_list, as_label_list = label_tp_list
at_label_map = {i: label for i, label in enumerate(at_label_list)}
as_label_map = {i: label for i, label in enumerate(as_label_list)}

print(at_label_map)
print(as_label_map)


# In[13]:


def load_model(model_file, args, num_tp_labels, task_config, device):
    model_file = model_file
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        print("Model loaded from %s", model_file)
        model = BertForSequenceLabeling.from_pretrained(args.bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                                                        state_dict=model_state_dict, num_tp_labels=num_tp_labels,
                                                        task_config=task_config)
        model.to(device)
    else:
        model = None
    return model


# In[14]:


# set model file to the last saved model after training epochs completed

# CODE CHANGE NOTE: In this implementation (using a docker container for jupyter lab) the download of bert-base-uncased.tar.gz 
#into cache terminates before the whole file is successfully loaded. 
# Therefore an adapted ate_asc_modeling_local_bert_file.py is imported here which loads the model from a folder in the repo ('bert-base-uncased/bert-base-uncased.tar.gz')

model_file = '../GRACE/out_twemlab_all_ateacs/pytorch_model.bin.9'
model = load_model(model_file, args, num_tp_labels, task_config, device)


# In[15]:


if hasattr(model, 'module'):
    print('has module')
    model = model.module
    
# print(model)


# In[16]:


# set model to eval mode (turn off training features e.g. dropout)
model.eval()


# #### Testing Block (can be skipped)
# 
# This code block serves to ensure the model loaded correctly.
# Uses just the last entry in twitter_1_train.txt

# In[17]:


DATALOADER_DICT = {}


# In[18]:


DATALOADER_DICT["ate_asc"] = {"eval": dataloader_val}


# In[19]:


if task_name not in DATALOADER_DICT:
    raise ValueError("Task not found: %s" % (task_name))


# In[20]:


eval_dataloader, eval_examples = DATALOADER_DICT[task_name]["eval"](args, tokenizer, file_path, label_tp_list=label_tp_list, set_type="val")


# In[21]:


for input_ids, input_mask, segment_ids, at_label_ids, as_label_ids, label_mask, label_mask_X in eval_dataloader:
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    at_label_ids = at_label_ids.to(device)
    as_label_ids = as_label_ids.to(device)
    label_mask = label_mask.to(device)
    label_mask_X = label_mask_X.to(device)


# In[22]:


with torch.no_grad():
    # logits, decoder_logits = model(input_ids, segment_ids, input_mask)
    logits, sequence_output, encoder_output = model.get_encoder_logits(input_ids, segment_ids, input_mask)
    pred_dec_ids = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
    decoder_logits = model.get_decoder_logits(encoder_output, input_mask, label_mask_X, pred_dec_ids)
    logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
    decoder_logits = torch.argmax(F.log_softmax(decoder_logits, dim=2), dim=2)
    logits = logits.detach().cpu().numpy()
    decoder_logits = decoder_logits.detach().cpu().numpy()


# In[23]:


at_label_ids = at_label_ids.to('cpu').numpy()
as_label_ids = as_label_ids.to('cpu').numpy()
label_mask = label_mask.to('cpu').numpy()


# In[24]:


for i, mask_i in enumerate(label_mask):
    temp_11 = []
    temp_12 = []
    temp_21 = []
    temp_22 = []
    for j, l in enumerate(mask_i):
        if l > -1:
            temp_11.append(at_label_map[at_label_ids[i][j]])
            temp_12.append(at_label_map[logits[i][j]])
            temp_21.append(as_label_map[as_label_ids[i][j]])
            temp_22.append(as_label_map[decoder_logits[i][j]])

print('Aspect Terms:')
print(temp_11)
print('Predicted Aspect Terms:')
print(temp_12)
print('\nAspect Sentiment:')
print(temp_21)
print('Predicted Aspect Sentiment:')
print(temp_22)


# ##### Apply Model on Twemlab Birmingham Goldstandard Data

# #### Load Dataset

# In[25]:


# Load TwEmLab Goldstandard
tree1 = ET.parse('../Data/twemlab_goldstandards_original/birmingham_labels.xml')
root1 = tree1.getroot()

# check contents
#root1[0][1].text

# create dataframe from xml file
data1 = []
for tweet in root1.findall('Tweet'):
    id = tweet.find('ID').text
    label = tweet.find('Label').text
    data1.append((id, label))

df1 = pd.DataFrame(data1,columns=['id','label'])
    
# Load TwEmLab Boston Tweets
tree2 = ET.parse('../Data/twemlab_goldstandards_original/birmingham_tweets.xml')
root2 = tree2.getroot()

# check contents
#root2[0][1].text

# create dataframe from xml file
data2 = []
for tweet in root2.findall('Tweet'):
    id = tweet.find('ID').text
    text = tweet.find('text').text
    goldstandard = tweet.attrib.get("goldstandard")
    data2.append((id, text, goldstandard))

df2 = pd.DataFrame(data2,columns=['id','text', 'goldstandard'])
# df2.head()

 # merge the two separate dataframes based on id columns
merge = pd.merge(df1, df2, on='id')

# keep only the tweets that are part of the goldstandard
twemlab = merge[merge['goldstandard'] == 'yes']
print(f'Number of tweets in goldstandard: {len(twemlab)}')

sentimemt_label_three = []
# assign sentiment label (0, 1) based on emotion
for index, row in twemlab.iterrows():
    if row['label'] == 'beauty' or row['label'] == 'happiness':
        sentimemt_label_three.append(1)
    elif row['label'] == 'none':
        sentimemt_label_three.append(0)
    else: 
        sentimemt_label_three.append(-1)
        
twemlab['sentiment_label'] = sentimemt_label_three


# check dataset
twemlab.head()


# ##### Re-Format Text to match GRACE Input

# In[26]:


# text column to list
text_list = list(twemlab['text'])

# input format for GRACE model
addition = ' - - O O O'
convert_to_doc = []

# iteratively apply re-formatting and save to new list
for tweet in text_list:
    words = tweet.split()
    words_with_addition = []
    for word in words:
        new_word = word + addition
        #print(new_word)
        words_with_addition.append(new_word)
    convert_to_doc.append(words_with_addition)

# check outputs
#print(convert_to_doc[10])


# #### Save to .txt File

# In[27]:


path_to_reformatted_data = "../Data/twemlab_goldstandards_original/original_reformatted_with_0s/twemlab_birmingham_formatted.txt"

with open(path_to_reformatted_data, mode = "w") as f:
    for tweet in convert_to_doc:
        for word in tweet:
            f.write("%s\n" % word)
        f.write("\n")


# ##### Run GRACE Model on reformatted .txt File

# In[28]:


DATALOADER_DICT = {}
# only "eval" state is needed ("train" is left out)
DATALOADER_DICT["ate_asc"] = {"eval":dataloader_val}


# In[29]:


eval_dataloader, eval_examples = DATALOADER_DICT[task_name]["eval"](args, tokenizer, path_to_reformatted_data, label_tp_list=label_tp_list, set_type="val")


# In[24]:


# empty lists for both the identified aspect terms and the related sentiments
pred_aspect_terms = []
pred_aspect_sentiments = []

# for-loop to iterate over the preprocessed outputs from the "eval_dataloader" 
for input_ids, input_mask, segment_ids, at_label_ids, as_label_ids, label_mask, label_mask_X in eval_dataloader:
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    at_label_ids = at_label_ids.to(device)
    as_label_ids = as_label_ids.to(device)
    label_mask = label_mask.to(device)
    label_mask_X = label_mask_X.to(device)
    
    with torch.no_grad():
        # logits, decoder_logits = model(input_ids, segment_ids, input_mask)
        logits, sequence_output, encoder_output = model.get_encoder_logits(input_ids, segment_ids, input_mask)
        pred_dec_ids = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        decoder_logits = model.get_decoder_logits(encoder_output, input_mask, label_mask_X, pred_dec_ids)
        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        decoder_logits = torch.argmax(F.log_softmax(decoder_logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        decoder_logits = decoder_logits.detach().cpu().numpy()
        
    at_label_ids = at_label_ids.to('cpu').numpy()
    as_label_ids = as_label_ids.to('cpu').numpy()
    label_mask = label_mask.to('cpu').numpy()
    
    for i, mask_i in enumerate(label_mask):
        #temp_11 = []
        temp_12 = []
        #temp_21 = []
        temp_22 = []
        for j, l in enumerate(mask_i):
            if l > -1:
                #temp_11.append(at_label_map[at_label_ids[i][j]])
                temp_12.append(at_label_map[logits[i][j]])
                #temp_21.append(as_label_map[as_label_ids[i][j]])
                temp_22.append(as_label_map[decoder_logits[i][j]])
                
        pred_aspect_terms.append(temp_12)
        pred_aspect_sentiments.append(temp_22)

# add new aspect term labels and aspect sentiment labels as columns to twemlab dataframe
twemlab['aspect_term_preds'] = pred_aspect_terms
twemlab['aspect_senti_preds'] = pred_aspect_sentiments


# ##### Extract Aspect Terms and Sentiment Aspects

# In[25]:


# extract aspect term and sentiment and store in twemlab dataframe
aspect_terms = []
aspect_sentiments = []

# for every row (tweet)
for idx, row in twemlab.iterrows():
    
    row_aspect_terms = []
    row_aspect_sentiments = [] 

    # get text length
    words = row['text'].split()
    count_words = len(words)
    
    # get token length (may differ from text length)
    tokens = row['aspect_senti_preds']
    token_counts = len(tokens)
    
    # for every word in tweet --> check if it's an aspect term and save
    for i in range(count_words):
        if row['aspect_term_preds'][i] == 'B-AP':
            term = words[i]
                   
            # for remaining words
            for j in range(i, count_words):
                if row['aspect_term_preds'][j] == 'I-AP':
                    term = term + ' ' + words[j]
            
            row_aspect_terms.append(term)
    
    aspect_terms.append(row_aspect_terms)
    
    # for every token --> extract sentiment
    for i in range(token_counts):
        if row['aspect_term_preds'][i] == 'B-AP':
            sent = tokens[i]
            
            # for remaining tokens
            for j in range(i, token_counts):
                if row['aspect_term_preds'][j] == 'I-AP':
                    sent = sent + ' ' + tokens[j]
            
            row_aspect_sentiments.append(sent)
    
    aspect_sentiments.append(row_aspect_sentiments)
            
twemlab['aspect_terms'] = aspect_terms
twemlab['aspect_sentiments'] = aspect_sentiments


# In[28]:


twemlab.tail(50)


# In[27]:


index = 550

print(f"An Example of how the GRACE results look\n\nText:\t{twemlab.iloc[index]['text']}\n\nAsp.Terms:\t{twemlab.iloc[index]['aspect_term_preds']}\nAsp.Sents:\t{twemlab.iloc[index]['aspect_senti_preds']}\n\nATE:\t{twemlab.iloc[index]['aspect_terms']}\nASC:\t{twemlab.iloc[index]['aspect_sentiments']}\n\nGoldstandard: {twemlab.iloc[index]['sentiment_label']}")


# #### Get some indication of how well it's going (SENTIMENTS)

# In[29]:


# compare the aspect sentiments with the twemlab sentiments

match = 0          # sentiment tokens uniformly match goldstandard sentiment
nomatch = 0        # sentiment tokens are uniformly different to goldstandard sentiment
not_uniform = 0    # sentiment tokens are not uniform
n_a = 0            # empty sentiment tokens list
counter = 0        # overall rows

for idx, row in twemlab.iterrows():
    counter += 1
    
    # if there is more than 1 list of aspect sentiment tokens
    if len(row['aspect_sentiments']) > 1:
        
        # join them together into one list
        all_tokens = ' '.join(row['aspect_sentiments'])
        all_tokens = all_tokens.split()
        first_token = all_tokens[0]
        
        # check if tokens are the same, e.g. ['NEUTRAL NEUTRAL', 'NEUTRAL NEUTRAL']
        if all(token == first_token for token in all_tokens):
            if first_token == 'POSITIVE' and row['sentiment_label'] == 1: match += 1
            elif first_token == 'NEUTRAL' and row['sentiment_label'] == 0: match += 1
            elif first_token == 'NEGATIVE' and row['sentiment_label'] == -1: match += 1
            else: nomatch += 1 
        else: not_uniform += 1

    # if there is just 1 list of aspect sentiment tokens
    elif len(row['aspect_sentiments']) == 1:
        
        tokens = row['aspect_sentiments'][0].split()
        first_token = tokens[0]
        
        # if the list has more than one token
        if len(tokens) > 1:
            # check if tokens are the same  e.g. ['POSITIVE POSITIVE POSITIVE']
            if all(token == first_token for token in tokens):
                #print(f"row asp sents: {row['aspect_sentiments']}")
                if first_token == 'POSITIVE' and row['sentiment_label'] == 1: match += 1
                elif first_token == 'NEUTRAL' and row['sentiment_label'] == 0: match += 1
                elif first_token == 'NEGATIVE' and row['sentiment_label'] == -1: match += 1
                else: nomatch += 1
            else: not_uniform += 1
        
        # if the list has only one entry
        elif len(tokens) == 1:
            if row['aspect_sentiments'][0] == 'POSITIVE' and row['sentiment_label'] == 1: match += 1
            elif row['aspect_sentiments'][0] == 'NEUTRAL' and row['sentiment_label'] == 0: match += 1
            elif row['aspect_sentiments'][0] == 'NEGATIVE' and row['sentiment_label'] == -1: match += 1
            else: nomatch += 1
        
        else:
            print("whaaat")
    
    # empty list
    else:
        n_a += 1

print("Comparing the aspect sentiment classifcation with the overall sentiment in the twemlab goldstandard:\n")
print(f"Matches:      {match}  ({np.round((match/counter)*100, 1)}%)\nNo Matches:   {nomatch}  ({np.round((nomatch/counter)*100, 1)}%)\nNot Uniform:   {not_uniform}   ({np.round((not_uniform/counter)*100, 1)}%)\nN/A:          {n_a}  ({np.round((n_a/counter)*100, 1)}%)")


# ##### Get an indication of how well the model is able to classify the emotions

# In[30]:


# compare the aspect emotion with the twemlab emotions

match = 0          # emotion tokens uniformly match goldstandard emotion
nomatch = 0        # emotion tokens are uniformly different to goldstandard emotion
not_uniform = 0    # emotion tokens are not uniform
n_a = 0            # empty emotion tokens list
counter = 0        # overall rows

for idx, row in twemlab.iterrows():
    counter += 1
    
    # if there is more than 1 list of aspect sentiment tokens
    if len(row['aspect_sentiments']) > 1:
        
        # join them together into one list
        all_tokens = ' '.join(row['aspect_sentiments'])
        all_tokens = all_tokens.split()
        first_token = all_tokens[0]
        
        # check if tokens are the same, e.g. ['NEUTRAL NEUTRAL', 'NEUTRAL NEUTRAL']
        if all(token == first_token for token in all_tokens):
            if first_token == 'happiness' and row['label'] == 'beauty': match += 1
            elif first_token == 'happiness' and row['label'] == 'happiness': match += 1
            elif first_token == 'anger' and row['label'] == 'anger/disgust': match += 1
            elif first_token == 'fear' and row['label'] == 'fear': match += 1
            elif first_token == 'none' and row['label'] == 'none': match += 1
            elif first_token == 'sadness' and row['label'] == 'sadness': match += 1
            else: nomatch += 1 
        else: not_uniform += 1

    # if there is just 1 list of aspect sentiment tokens
    elif len(row['aspect_sentiments']) == 1:
        
        tokens = row['aspect_sentiments'][0].split()
        first_token = tokens[0]
        
        # if the list has more than one token
        if len(tokens) > 1:
            # check if tokens are the same  e.g. ['POSITIVE POSITIVE POSITIVE']
            if all(token == first_token for token in tokens):
                if first_token == 'happiness' and row['label'] == 'beauty': match += 1
                elif first_token == 'happiness' and row['label'] == 'happiness': match += 1
                elif first_token == 'anger' and row['label'] == 'anger/disgust': match += 1
                elif first_token == 'fear' and row['label'] == 'fear': match += 1
                elif first_token == 'none' and row['label'] == 'none': match += 1
                elif first_token == 'sadness' and row['label'] == 'sadness': match += 1
                else: nomatch += 1
            else: not_uniform += 1
        
        # if the list has only one entry
        elif len(tokens) == 1:
            if row['aspect_sentiments'][0] == 'happiness' and row['label'] == 'beauty': match += 1
            elif row['aspect_sentiments'][0] =='happiness' and row['label'] == 'happiness': match += 1
            elif row['aspect_sentiments'][0] == 'anger' and row['label'] == 'anger/disgust': match += 1
            elif row['aspect_sentiments'][0] == 'fear' and row['label'] == 'fear': match += 1
            elif row['aspect_sentiments'][0] == 'none' and row['label'] == 'none': match += 1
            elif row['aspect_sentiments'][0] == 'sadness' and row['label'] == 'sadness': match += 1
            else: nomatch += 1
        
        else:
            print("whaaat")
    
    # empty list
    else:
        n_a += 1

print("Comparing the aspect emotion classifcation with the overall emotion in the twemlab goldstandard:\n")
print(f"Matches:      {match}  ({np.round((match/counter)*100, 1)}%)\nNo Matches:   {nomatch}  ({np.round((nomatch/counter)*100, 1)}%)\nNot Uniform:   {not_uniform}   ({np.round((not_uniform/counter)*100, 1)}%)\nN/A:          {n_a}  ({np.round((n_a/counter)*100, 1)}%)")


# #### Export the dataframe to CSV 

# In[32]:


twemlab.to_csv("../Data/twemlab_emotion_labelled_data/birmingham_emotion_labelled_data.csv")


# ##### Apply Model on the AIFER Twitter Dataset

# #### Load Dataset

# In[27]:


# aifer dataset
dataset = pd.read_csv("../Data/Disaster_responses/ahrtal_tweets.csv", sep="\t")

# filter for English tweets
lang = ['en']
en_dataset = dataset[dataset['tweet_lang'].isin(lang)]

# exclude columns that aren't needed (for now)
data_aifer = en_dataset[['date', 'text', 'geom']]

# date handling (for animated map)
#data['year'] = [pd.to_datetime(x).year for x in data['date']]
#data['month'] = [pd.to_datetime(x).to_period('M') for x in data['date']]
#data['week'] = [pd.to_datetime(x).to_period('W') for x in data['date']]
#data['day'] = [pd.to_datetime(x).to_period('D') for x in data['date']]
# data['month'] = data.apply(lambda row: pd.to_datetime(row["date"]).to_period('M'), axis=1)

# get subset to speed up demos and testing
#data_subset = data.sample(1000)
data_aifer.head(10)


# #### Re-format Text to match GRACE Input

# In[28]:


import re 

# text column to list
text_list = list(data_aifer['text'])

# input format for GRACE model
addition = ' - - O O O'
convert_to_doc = []

# Create a regex pattern to match all special characters in string
pattern = r'[^A-Za-z0-9]+'

# iteratively apply re-formatting and save to new list
for tweet in text_list:
    words = tweet.split()
    words_with_addition = []
    for word in words:
        # Remove special characters from the string
        word = re.sub(pattern, '', word)
        new_word = word + addition
        #print(new_word)
        words_with_addition.append(new_word)
    convert_to_doc.append(words_with_addition)

# check outputs
print(convert_to_doc[2])


# #### Save to .txt File

# In[30]:


with open("../Data/Disaster_responses/aifer_en_reformatted.txt", mode = "w", encoding="utf-8") as f:
    for tweet in convert_to_doc:
        for word in tweet:
            f.write("%s\n" % word)
        f.write("\n")


# ##### Run GRACE Model on remoformatted .txt File

# In[31]:


#file_path = os.path.join(args.data_dir, args.train_file)
file_path = '../Data/Disaster_responses/aifer_en_reformatted.txt'
print(file_path)


# In[32]:


DATALOADER_DICT = {}
# only "eval" state is needed ("train" is left out)
DATALOADER_DICT["ate_asc"] = {"eval":dataloader_val}


# In[33]:


eval_dataloader, eval_examples = DATALOADER_DICT[task_name]["eval"](args, tokenizer, file_path, label_tp_list=label_tp_list, set_type="val")


# In[34]:


pred_aspect_terms = []
pred_aspect_sentiments = []

for input_ids, input_mask, segment_ids, at_label_ids, as_label_ids, label_mask, label_mask_X in eval_dataloader:
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    at_label_ids = at_label_ids.to(device)
    as_label_ids = as_label_ids.to(device)
    label_mask = label_mask.to(device)
    label_mask_X = label_mask_X.to(device)
    
    with torch.no_grad():
        # logits, decoder_logits = model(input_ids, segment_ids, input_mask)
        logits, sequence_output, encoder_output = model.get_encoder_logits(input_ids, segment_ids, input_mask)
        pred_dec_ids = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        decoder_logits = model.get_decoder_logits(encoder_output, input_mask, label_mask_X, pred_dec_ids)
        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        decoder_logits = torch.argmax(F.log_softmax(decoder_logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        decoder_logits = decoder_logits.detach().cpu().numpy()
        
    at_label_ids = at_label_ids.to('cpu').numpy()
    as_label_ids = as_label_ids.to('cpu').numpy()
    label_mask = label_mask.to('cpu').numpy()
    
    for i, mask_i in enumerate(label_mask):
        #temp_11 = []
        temp_12 = []
        #temp_21 = []
        temp_22 = []
        for j, l in enumerate(mask_i):
            if l > -1:
                #temp_11.append(at_label_map[at_label_ids[i][j]])
                temp_12.append(at_label_map[logits[i][j]])
                #temp_21.append(as_label_map[as_label_ids[i][j]])
                temp_22.append(as_label_map[decoder_logits[i][j]])
                
        pred_aspect_terms.append(temp_12)
        pred_aspect_sentiments.append(temp_22)

# add new aspect term labels and aspect sentiment labels as columns to twemlab dataframe
data_aifer['aspect_term_preds'] = pred_aspect_terms
data_aifer['aspect_senti_preds'] = pred_aspect_sentiments


# A sample of the added columns:

# In[36]:


demo_cols = data_aifer[['text', 'aspect_term_preds', 'aspect_senti_preds']]
col_names = {'text': 'Text', 'aspect_term_preds': 'Aspect Terms', 'aspect_senti_preds': 'Aspect Sentiments'}
demo_cols = demo_cols.rename(columns=col_names)
demo_cols.head(5).style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])])
# data_aifer.head()


# ##### Extract Aspect Terms and Emotions

# In[37]:


# extract aspect term words and store in the dataframe
aspect_terms = []
aspect_sentiments = []

# for every row (tweet)
for idx, row in data_aifer.iterrows():
    
    row_aspect_terms = []
    row_aspect_sentiments = [] 

    # get text length
    words = row['text'].split()
    count_words = len(words)
    
    # get token length (may differ from text length)
    tokens = row['aspect_senti_preds']
    token_counts = len(tokens)
    
    # for every word in tweet --> check if it's an aspect term and save
    for i in range(token_counts):
        if row['aspect_term_preds'][i] == 'B-AP':
            term = words[i]
                   
            # for remaining words
            for j in range(i, token_counts):
                if row['aspect_term_preds'][j] == 'I-AP':
                    term = term + ' ' + words[j]
            
            row_aspect_terms.append(term)
    
    aspect_terms.append(row_aspect_terms)
    
    # for every token --> extract sentiment
    for i in range(token_counts):
        if row['aspect_term_preds'][i] == 'B-AP':
            sent = tokens[i]
            
            # for remaining tokens
            for j in range(i, token_counts):
                if row['aspect_term_preds'][j] == 'I-AP':
                    sent = sent + ' ' + tokens[j]
            
            row_aspect_sentiments.append(sent)
    
    aspect_sentiments.append(row_aspect_sentiments)
            
data_aifer['aspect_terms'] = aspect_terms
data_aifer['aspect_sentiments'] = aspect_sentiments


# In[38]:


demo_cols = data_aifer[['text', 'aspect_terms', 'aspect_sentiments']]
col_names = {'text': 'Text', 'aspect_terms': 'Aspect Terms', 'aspect_sentiments': 'Aspect Sentiments'}
demo_cols = demo_cols.rename(columns=col_names)
demo_cols.head(5).style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])])


# #### Drop Unnecessary Columns

# In[43]:


data_cov = data_covid.copy()
data_cov.drop("aspect_term_preds", inplace=True, axis=1)
data_cov.drop("aspect_senti_preds", inplace=True, axis=1)
data_cov.head()


# #### Export Labelled Data to CSV

# In[44]:


data_cov.to_csv("../Data/Twitter CSV Data for Testing/100k_usa_covid_labelled_emotions.csv")


# ##### Data Visualisation

# #### Helper Functions

# In[39]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import mapclassify                                                     # required for animated scatterplot map (plotly)
import geopandas as gpd                                                # geographic data handling
import folium                                                          # interactive mapping capabilities
import folium.plugins as plugins
import plpygis                                                         # a converter to and from the PostGIS geometry types, WKB, GeoJSON and Shapely formats
from plpygis import Geometry
from shapely.geometry import Point, Polygon, shape                     # creating geospatial data
from shapely import wkb, wkt                                           # creating and parsing geospatial data
import shapely                                                  

import plotly
import plotly.express as px                                            # for interactive, animated timeseries map
import seaborn as sns; sns.set(style="ticks", color_codes=True)
# import json

from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
from PIL import Image    # for masking wordcloud with an image
import requests          # for accessing url image
from io import BytesIO   # for accedssing url image


# In[40]:


# world centroids
# https://github.com/gavinr/world-countries-centroids/blob/master/dist/countries.csv
world = pd.read_csv('https://raw.githubusercontent.com/gavinr/world-countries-centroids/master/dist/countries.csv')


# In[41]:


# turn nurmal dataframe with 'geom' column into a geodataframe

def generate_geodataframe(in_df):
    '''
    Input:
        in_df: a dataframe with a 'geom' column to be converted into a shapely Geometry column
    Output: 
        df_geo: a geopandas dataframe with a designated geometry column which can be mapped
    '''
    
    # initiate new geom list
    new_geom = []
    
    # access 'geom' column in input df and convert into geopandas geometry
    for item in in_df['geom']:
        new_geom.append(Geometry(item).shapely)
    
    # add as new column to input df
    in_df["geometry"] = new_geom
    
    # create geopandas GeoDataFrame
    df_geo = gpd.GeoDataFrame(in_df, crs="EPSG:4326")
    s = gpd.GeoSeries(df_geo['geometry'])
    df_geo['geom'] = df_geo['geometry']
    df_geo['geometry'] = df_geo['geom'].to_crs('epsg:3785').centroid.to_crs(df_geo.crs)
    
    return df_geo


# In[42]:


def create_pos_neg_tweet_wordcloud(df, labels_col, labels):
    '''
    Input:
        df: dataframe, requires 'text' column and 'sentiment_label' column with values 'positive', 'negative'
        labels_col: column with textual sentiment labels
        labels: list of textual labels (assumpes 3 labels ordered from good to bad)
    Output:
        n/a  (creates wordcloud visualisations)
    '''
    # split into pos, neg tables
    happiness = df[df[labels_col] == labels[0]]
    sadness = df[df[labels_col] == labels[1]]
    anger = df[df[labels_col] == labels[2]]
    fear = df[df[labels_col] == labels[3]]

    senti_dfs = [happiness, sadness, anger, fear]
    # colors available at https://matplotlib.org/1.2.1/mpl_examples/pylab_examples/show_colormaps.pdf
    cmap = ['YlGn','Blues', 'OrRd', 'RdPu']

    stopwords = set(STOPWORDS)
    stopwords.update(['https', 't', 'co', 's'])
    url = 'https://raw.githubusercontent.com/rasbt/datacollect/master/dataviz/twitter_cloud/twitter_mask.png'
    response = requests.get(url)
    mask = np.array(Image.open(BytesIO(response.content)))

    for i in range(len(senti_dfs)):
        text = " ".join(i for i in senti_dfs[i].text)
        wordcloud = WordCloud(stopwords=stopwords, mask=mask, background_color="white", colormap=cmap[i]).generate(text)
        plt.figure( figsize=(10,7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


# In[43]:


# create time chart based on time intervals 
def create_df_for_time_chart_three_labels(df, time_interval, labels_col, labels):
    '''
    Input:
        df: dataframe
        time_interval: string, name of column in df that contains time references ('Day', 'Week', 'Month', 'Year')
        labels_col: column with textual sentiment predictions
        labels: list of the textual labels ordered in good to bad e.g. ['Positive', 'Negative', 'Neutral'] or ['positive', 'negative', 'neutral']
    Outout:
        sorted_df: a dataframe with the columns 'month', 'Counts', 'Sentiments'
        fig: the prepared timeseries chart
    '''
    # create relevant datetime column if not already there 
    #if 'time_interval not in df.columns:
    time = time_interval[0]
    df['time_interval'] = [pd.to_datetime(x).to_period(time) for x in df['date']]
    
    # get a list of all the time steps 
    #if time_interval.lower() == 'month':
     #   unique_intervals_in_df = df.Month.unique()
    #elif time_interval.lower() =='day':
#        unique_intervals_in_df = df.Day.unique()
#    elif time_interval.lower() =='year':
#        unique_intervals_in_df = df.Year.unique()
#    elif time_interval.lower() =='week':
#        unique_intervals_in_df = df.Week.unique()

    unique_intervals_in_df = df.time_interval.unique()
    
    # split into pos, neu, neg dfs
    positives = df[df[labels_col] == labels[0]]
    neutrals = df[df[labels_col] == labels[1]]
    negatives = df[df[labels_col] == labels[2]]

    # count how many tweets there are for each time step in each sentiment df
    counts_pos = positives['time_interval'].value_counts()
    counts_neu = neutrals['time_interval'].value_counts()
    counts_neg = negatives['time_interval'].value_counts()

    dfs = [counts_pos, counts_neu, counts_neg]

    d = []
    # iterate over the labels
    for i in range(len(labels)):
        # iterate over all unique time stamps in dataset
        for interval in unique_intervals_in_df:
            # if the current time stamp is in the current label's df
            if interval in dfs[i]:
                # add the time stamp, the count of tweets at that time step , and the label to the output list 'd'
                d.append([interval, dfs[i][interval], labels[i]])

    # create a df from information
    intervals_df = pd.DataFrame(d, columns=['time_interval', 'Counts', 'Sentiment']) 
    
    # sort by time
    sorted_df = intervals_df.sort_values(by=['time_interval'])

    # reformat to string for display
    sorted_df['time_interval'] = sorted_df['time_interval'].values.astype(str)
    
    # create figure
    fig = px.area(sorted_df, 
                  x= "time_interval", 
                  y="Counts", 
                  color="Sentiment", 
                  line_group='Sentiment',
                  hover_name="Counts",
                  color_discrete_map={labels[0]:'#0DB110', labels[1]: '#F7E444', labels[2]: '#DD2828'})

    return sorted_df, fig


# In[44]:


def create_animated_time_map_three_labels(df, time_interval, label_col, title, labels, style):
    '''
    inputs: 
        df: geodataframe (needs to have a 'geometry' column)
        time_interval: a timestamp column (e.g. 'Day') - must be capitalised
        label: the df column used for the main label on the tooltip popup
        title: string 
        n_labels: int
        color_discrete_map: dict, mapping labels e.g. 'negative' to colors e.g. '#FF0000'
        style: string, mapbox styles e.g. 'carto-positron', 'open-street-map', 'white-bg', 'carto-positron', 'carto-darkmatter', 
              'stamen- terrain', 'stamen-toner', 'stamen-watercolor', 'basic', 'streets', 'outdoors', 'light', 
              'dark', 'satellite', 'satellite- streets'
    output:
        plotly interactive, animated map
    '''
    
    if time_interval not in df.columns:

        time = time_interval[0]
        df['time_interval'] = [pd.to_datetime(x).to_period(time) for x in df['date']]
    
    # set colors for markers
    cmap = {labels[0]: '#62FF00', labels[1]: '#FCFF00', labels[2]: '#FF0000'}
    
    fig = px.scatter_geo(df,
              lat=df.geometry.y,
              lon=df.geometry.x,
              size = [0.5] * len(df),
              size_max = 8,
              hover_name = label_col,
              hover_data = ['date'],
              color = label_col,
              color_discrete_map = cmap,
              animation_frame= 'time_interval',
              #mapbox_style=style,
              #category_orders={
              #time_col:list(np.sort(df[time_col].unique()))
              #},                  
              #zoom=3,
              opacity = 0.6,
              projection = 'albers usa',
              #projection= 'orthographic',
              #scope= 'north america',
              width=1600,
              height=1000)
                       
    
    fig.update_layout(
        title=title,
        autosize= True,
        hovermode='closest',
        margin=dict(t=0, b=0, l=0, r=0)
    )
    
    fig.show();
            
    return


# In[79]:


# create interactive folium map 

def create_folium_map(df, map_center, tiles, zoom_start, n_labels, text_col, senti_label_col, senti_aspects):
    '''
    Input: 
        df: a geodataframe (must have 'geometry' column)
        map_center: a string country name, or a list with two coordinates e.g. [37.0902, -95.7129]
        tiles: stirng, the background map style e.g. 'cartodbpositron' or 'Stamen Toner'
        zoom_start: int, higher numbers a more zoomed out
        n_labels: int, either two or three
        text_col: string, the name of the tweets text column in the df
        senti_label_col: string, the name of the labels column in the df
        senti_aspects: string, the aspect terms
    Output:
        map: an interactive folium map
    '''  


    # convert emotions to numberic columns 
    emotion_nums = []
    for index, row in df.iterrows():
        if row['majority_emotions'] == 'happiness': emotion_nums.append(0)
        if row['majority_emotions'] == 'sadness': emotion_nums.append(1)
        if row['majority_emotions'] == 'anger': emotion_nums.append(2)
        if row['majority_emotions'] == 'fear': emotion_nums.append(3)
        if row['majority_emotions'] == 'none': emotion_nums.append(4)
        else: emotion_nums.append(5)
        
    #print(len(emotion_nums))
    #print(len(df))

    
    # set map center (if input is string, search through world centroids CSV, else take list values)
    if isinstance(map_center, str):
        
        # test that the country is in the df
        if np.sum(world['COUNTRY'].isin([map_center])) == 1:
            idx = world.loc[world['COUNTRY'] == map_center].index[0]
            lat = round(world.iloc[idx]['latitude'], 4)
            long = round(world.iloc[idx]['longitude'], 4)
            center = [lat, long]
        else:
            print(f'Country {map_center} was either not found or too many matches found in centroids dataframe. Defaulting to USA.')
            idx = world.loc[world['COUNTRY'] == 'United States'].index[0]
            lat = round(world.iloc[idx]['latitude'], 4)
            long = round(world.iloc[idx]['longitude'], 4)
            center = [lat, long]    
            
    # if the input is a list simply use that input as center coordinates
    elif isinstance(map_center, list):
        center = map_center

    # create map with basic settings
    # get map's center coordinates from dict, default to usa
    map = folium.Map(location=center, tiles=tiles, zoom_start=zoom_start)
    
    # set colors
    #colors
    # labels = ['happiness', 'sadness', 'anger','fear', 'none', '']
    if n_labels == 6:
        colors = ['#66ED7A','#66AAED', '#ED6666', '#8359A4', '#6E6E6E', '#BFBFBF']  
    #if n_labels == 5:
    #colors = ['#66ED7A','#66AAED', '#ED6666', '#8359A4', '#6E6E6E'] 
    # iterate over df rows
    for i in range(len(df)):

        # logic to split tweet text for the pop ups (had to be done manually, couldn't find appropriate function included in folium.map.tooltip - html max width only applies to the box)
        text = '' 

        # if text is longer than 40 characters, split into words list and count
        if len(df.iloc[i][text_col]) > 40: 
            word_list = df.iloc[i][text_col].split()
            length = len(word_list)

            # first part of text is the same regardless of length
            text_pt1 = '<b>Emotion:</b> ' + df.iloc[i][senti_label_col] + '<br><b>Tweet:</b> '

            k = 0
            text_add = []

            # while k hasn't reached the length of the word list yet, keep adding words to 'text_add' list with a '<br>' after every 6th word
            while k < length:
                # after every 6 words add '<br>'
                if k%6 == 0 and k != 0:
                    text_add.append(str(word_list[k:k+1][0]) + ' <br>')
                else:
                    text_add.append(word_list[k:k+1][0])
                k += 1

            # join 'text_add' list together and then both text parts
            text_pt2 = ' '.join(text_add)
            text = text_pt1 + text_pt2

        else:
            text = '<b>Emotion:</b> ' + df.iloc[i][senti_label_col] + '<br><b>Tweet:</b> ' + df.iloc[i][text_col]

        
        map.add_child(
            folium.CircleMarker(
                location=[df.iloc[i].geometry.y, df.iloc[i].geometry.x],
                radius = 5,
                tooltip= folium.map.Tooltip(text),
                fill_color=colors[emotion_nums[i]],
                fill_opacity = 0.4,
                stroke=False
            )
        )


    # add button for full screen
    folium.plugins.Fullscreen().add_to(map)
    
    return map


# In[46]:


def create_piechart(ratios, labels, title):

    #colors
    # labels = ['happiness', 'sadness', 'anger','fear', 'none', '']
    if len(labels) == 6:
        colors = ['#66ED7A','#66AAED', '#ED6666', '#8359A4', '#6E6E6E', '#BFBFBF']

    #explosion
    explode= [0.05] * len(ratios)

    plt.pie(ratios, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)
    #draw circle
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
   # ax1.axis('equal')  
    plt.tight_layout()
    plt.show()


# In[47]:


# find most frequent element in a list
 
def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num


# #### Data Plotting and Visaulisation
# 

# For plotting, it's required to have only one emotion per tweet. So it's first necessary to find the majority emotion per tweet and add a column to the df which contains the majority emotion of all aspect emotions.

# In[50]:


# for plotting, we need only one emotion per tweet - so we need to find the majority emotion
# here a column is added to the df which contains the majority emotion of all aspect emotions

majority_emotions = []

for idx, row in data_aifer.iterrows():
    if len(row['aspect_sentiments'])>0:
        # if there's more than one list within the column merge all together so its easy to count majority
        if len(row['aspect_sentiments']) > 1:
            emotions = ''
            for j in range(len(row['aspect_sentiments'])):
                emotions = emotions + ' ' + row['aspect_sentiments'][j]
                #print(row['aspect_sentiments'])
                #print(emotions)
            majority_emotions.append(most_frequent(emotions.split()))
        # if there's only one list in the column simply count the majority
        else:
            majority_emotions.append(most_frequent(row['aspect_sentiments']))
    # if there's no list, just append an empty element
    else:
        majority_emotions.append('')

# add new column to df
data_aifer['majority_emotions'] = majority_emotions


# In[51]:


demo_cols = data_aifer[['aspect_sentiments', 'majority_emotions']]
col_names = {'aspect_sentiments': 'All Aspect Sentiments', 'majority_emotions': 'The Majority Emotion'}
demo_cols = demo_cols.rename(columns=col_names)
demo_cols.head(10).style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])])


# Using a single emotion per tweet, data viusalisation becomes possible.

# In[54]:


# general variables
labels = ['happiness', 'sadness', 'anger','fear', 'none', '']


# In[55]:


# count label frequencies and save in list
happiness = list(data_aifer['majority_emotions']).count(labels[0])
sadness = list(data_aifer['majority_emotions']).count(labels[1])
anger = list(data_aifer['majority_emotions']).count(labels[2])
fear = list(data_aifer['majority_emotions']).count(labels[3])
none = list(data_aifer['majority_emotions']).count(labels[4])
nothing = list(data_aifer['majority_emotions']).count(labels[5])
ratios = [happiness, sadness, anger, fear, none, nothing]
print(labels)
print(ratios)

# create simple piechart to show ratios between the sentiment labels
create_piechart(ratios, labels=labels, title='Sentiments in the AIFER Tweets')


# In[56]:


create_pos_neg_tweet_wordcloud(df=data_aifer, labels_col='majority_emotions', labels=labels)


# ##### Geographic Emotions

# In[59]:


# create copy of the dataframe
data_copy = data_aifer.copy()
data_copy['geometry'] = gpd.GeoSeries.from_wkt(data_copy['geom'])

# convert dataframe to geodataframe using the geom column
# data_geo = generate_geodataframe(data_copy)


# In[80]:


# TO DO: FIX THIS FUNCTION - the emotions list does not match the length of the df length - figure out why. and then maybe make several maps with individual emotions.... cause this looks ugly

# create an interactive folium map
# for map center USA is 'United States', UK is 'United Kingdom'

folium_map = create_folium_map(df = data_copy, 
                               map_center = 'Germany', 
                               tiles = 'cartodbpositron', 
                               zoom_start = 7, 
                               n_labels = 6, 
                               text_col = 'text', 
                               senti_label_col ='majority_emotions', 
                               senti_aspects = 'aspect_terms'
                              )

# display map
folium_map


# <hr>
# 
# #### Discussion Topics
# 
# - The **label frequencies** of the training dataset. This is the distribution of the overall 1625 goldstandard emotion labels:

# In[90]:


# Load TwEmLab Goldstandard for Birmingham
tree1 = ET.parse('../Data/twemlab_goldstandards_original/birmingham_labels.xml')
root1 = tree1.getroot()

# check contents
#root1[0][1].text

# create dataframe from xml file
data1 = []
for tweet in root1.findall('Tweet'):
    id = tweet.find('ID').text
    label = tweet.find('Label').text
    data1.append((id, label))

df1 = pd.DataFrame(data1,columns=['id','label'])
 # df1.head()
    
# Load TwEmLab Birmingham Tweets
tree2 = ET.parse('../Data/twemlab_goldstandards_original/birmingham_tweets.xml')
root2 = tree2.getroot()

# check contents
# root2[0][1].text

# create dataframe from xml file
data2 = []
for tweet in root2.findall('Tweet'):
    id = tweet.find('ID').text
    text = tweet.find('text').text
    goldstandard = tweet.attrib.get("goldstandard")
    data2.append((id, text, goldstandard))

df2 = pd.DataFrame(data2,columns=['id','text', 'goldstandard'])
# df2.head()

 # merge the two separate dataframes based on id columns
merge = pd.merge(df1, df2, on='id')

# keep only the tweets that are part of the goldstandard
twemlab = merge[merge['goldstandard'] == 'yes']
print(f'Number of tweets in goldstandard: {len(twemlab)}')

emotions = []
# assign emotion label (happiness, anger, sadness, fear)
for index, row in twemlab.iterrows():
    if row['label'] == 'beauty' or row['label'] == 'happiness':
        emotions.append('happiness')
    elif row['label'] == 'anger/disgust':
        emotions.append('anger')
    elif row['label'] == 'sadness':
        emotions.append('sadness')
    elif row['label'] == 'fear':
        emotions.append('fear')
    else: 
        emotions.append('none')
        
twemlab['emotion'] = emotions

twemlab_birmingham = twemlab[['id','text','emotion']]

# check dataset
# twemlab_birmingham.head(20)

readfile = pd.read_csv('../Data/twemlab_goldstandards_original/boston_goldstandard.csv')
twemlab_boston = readfile[['Tweet_ID', 'Tweet_timestamp', 'Tweet_text', 'Tweet_goldstandard_attribute', 'Tweet_longitude','Tweet_latitude','Tweet_timestamp','Emotion']]
# use only rows that have text in them
twemlab_boston = twemlab_boston[0:631]
# twemlab_boston.head()

emotions = []
# assign emotion label (happiness, anger, sadness, fear)
for index, row in twemlab_boston.iterrows():
    if row['Emotion'] == 'beauty' or row['Emotion'] == 'happiness':
        emotions.append('happiness')
    elif row['Emotion'] == 'anger/disgust':
        emotions.append('anger')
    elif row['Emotion'] == 'sadness':
        emotions.append('sadness')
    elif row['Emotion'] == 'fear':
        emotions.append('fear')
    else: 
        emotions.append('none')
        
twemlab_boston['emotion'] = emotions

twemlab_boston = twemlab_boston[['Tweet_ID','Tweet_text','emotion']]

# check dataset
# twemlab_boston.head(20)


# In[96]:


# extract the emotion column from both dfs and merge
brim_emo = twemlab_birmingham[['emotion']]
bost_emo = twemlab_boston[['emotion']]

emotions_in_twemlab_all = brim_emo.append(bost_emo, ignore_index=True)
print(len(emotions_in_twemlab_all))


# In[97]:


#value_counts = twemlab['label'].value_counts()
value_counts = emotions_in_twemlab_all['emotion'].value_counts().reset_index()
value_counts = value_counts.rename(columns={'index': 'Value', 'emotion': 'Frequency'})
df_value_counts = pd.DataFrame(value_counts)

dem_cols = df_value_counts[['Value', 'Frequency']]
dem_cols


# - **Size and standard of training data**: The overall size of the training data for twemlab is 1625 and its aspect terms have been annotated by one individual "ad hoc" as per "Annotating Twemlab Goldstandard Files to Include Aspect Term Labels".</li>
# 
# - **Mearsuring performance**: a robust testing dataset is needed. Above, I have shown a makeshift performance measure on the same dataset that the model was trained on. Drawing any meaningful conclusions based on this is precarious.
# 
# - **Training capacities**: Batch size reduced due to out-of-memory errors. GRACE training is memory intensive (the authors use a nvidia tesla v100 gpu). Potential options: reduce float point precision? Currently having issues installing conda package for apex to do so. 
# 
# - **Model optimisation**? The GRACE model uses GeLU (an "advanced" activation function), the standard BERT nn.embeddings layer, 12 transformer encoder layers and 2 decoder layers. On top of that it has two classification heads (both nn.Linear). During training the model uses additional functions for virtaul adversarial training and gradient harmonized loss calculation.

# Potential research questions:
# 
# - How can geosocial media analysis benefit from aspect-based emotion analysis? What can an emotion analysis offer compared to sentiments?
# - How does the ABSA Model “GRACE”, trained for emotion classification, compare to other state-of-the-art aspect-based-emotion-analysis methods?
# - Can the “GRACE” model be further optimised for geosocial media analyses?
# 
