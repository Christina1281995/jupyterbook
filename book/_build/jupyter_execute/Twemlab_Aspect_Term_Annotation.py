#!/usr/bin/env python
# coding: utf-8

# # Annotating Twemlab Goldstandard Files to Include Aspect Term Labels
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

# #### Required Fromat

# For Training Step 1 'twitter_1_train.txt' file required:
# 
# Reformat the two dataframes 'twemlab_birmingham' and 'twemlab_boston' to match the format below: 
# 
# 
# ```
# -DOCSTART-
# 
# How - - O O O
# can - - O O O
# someone - - O O O
# so - - O O O
# incompetent - - O O O
# like - - O O O
# Maxine - - B_AP NEGATIVE B_AP+NEGATIVE
# Waters - - I_AP NEGATIVE I_AP+NEGATIVE
# stay - - O O O
# in - - O O O
# office - - O O O
# for - - O O O
# over - - O O O
# 20 - - O O O
# years - - O O O
# ? - - O O O
# #LAFail - - O O O
# 
# @HabibaAlshanti - - B_AP POSITIVE B_AP+POSITIVE
# .. - - O O O
# Yes - - O O O
# I - - O O O
# want - - O O O
# that - - O O O
# :p - - O O O
# 
# 
# ```
# 

# ### Imports

# In[1]:


import csv
import urllib.request
import pandas as pd                                                    # data handling
import xml.etree.cElementTree as ET 
import re


# ### Load Both Goldstandard Files
# 
# - Birmingham (994 tweets)
# - Boston (631 tweets)
# 
# Load into dataframe --> match emotion labels (happiness, anger, sadness, fear) --> keep only id, text, emotion columns
# 
# 

# #### Birmingham

# In[2]:


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
twemlab_birmingham.head(20)


# #### Boston

# In[3]:


readfile = pd.read_csv('../Data/twemlab_goldstandards_original/boston_goldstandard.csv')
twemlab_boston = readfile[['Tweet_ID', 'Tweet_timestamp', 'Tweet_text', 'Tweet_goldstandard_attribute', 'Tweet_longitude','Tweet_latitude','Tweet_timestamp','Emotion']]
# use only rows that have text in them
twemlab_boston = twemlab_boston[0:631]
twemlab_boston.head()


# In[4]:


print(len(twemlab_birmingham))


# In[5]:


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
twemlab_boston.head(20)


# ### Helper Functions for Aspect Term Labelling and Reformatting

# In[6]:


import math

def splitdataset(df, chunksize):

  # split up the twemlab goldstandard texts into 100 tweet chunks
  nr_of_iterations = (math.ceil(len(df) / chunksize))

  print(f'{nr_of_iterations} subsets created from the whole dataframe.\n')

  list_of_chunks = []

  for a in range(nr_of_iterations):
    cur_index = a*chunksize
    if a == nr_of_iterations:
      chunk_of_df = df[cur_index: len(df)]

    else: 
      chunk_of_df = df[cur_index: cur_index+chunksize]

    list_of_chunks.append(chunk_of_df)

  return list_of_chunks


# In[7]:


def annotate_aspects(df, text_col, emotions_col):

  # text column to list
  text_list = list(df[text_col])
  emotion_list = list(df[emotions_col])

  # addition to words that aren't aspect terms
  addition_no_ate = ' - - O O O'

  # function to use for beginning of aspect terms
  def beginning_ate(emotion):
    addition_beginning_ate = ' - - B_AP ' + emotion + ' B_AP+' + emotion
    return addition_beginning_ate

  # function to use for inside aspect terms
  def inside_ate(emotion):
    inside_beginning_ate = ' - - I_AP ' + emotion + ' I_AP+' + emotion
    return inside_beginning_ate

  # list to store all reformatted tweets
  convert_to_doc = []

  # logic to iteratively work in chunks
  counter_overall = 0

  # iteratively apply re-formatting and save to new list
  while counter_overall != len(text_list):

    # print a line to separate 
    print("---------------------------------------")    
    print(f'Tweet nr:  {counter_overall}')

    words = re.findall(r"[#\w\-]+|[.,!?():;\"\']", text_list[counter_overall])
     
    # show me the emotion
    # print(f"Emotion:   {emotion_list[counter_overall]}")
    
    # show me the text and let me determine how many aspects there are
    how_many_aspect_words = input(f"Emotion:   {emotion_list[counter_overall]} ------------ Tweet:     {' '.join(words)} ------------ How many aspect words?")
    # check that input is correct, otherwise prompt again until input is digit
    while how_many_aspect_words.isdigit() == False:
      ask_again = input(f"Emotion:   {emotion_list[counter_overall]} ------------ Tweet:     {' '.join(words)} ------------ How many aspect words? Enter a number:")
      if ask_again.isdigit():
        how_many_aspect_words = ask_again
    
    #list to collect aspect terms
    aspect_terms = []

    # list for entire reformatted tweet
    new_tweet = []

    # for each aspect phrase, enter the word to add it to aspect terms list
    for i in range(int(how_many_aspect_words)):

      get_word = input(f"Emotion:   {emotion_list[counter_overall]} ------------ Tweet:     {' '.join(words)} ------------ Aspect word {i}:")
      aspect_terms.append(get_word)
      #print(aspect_terms)

    # for each word that has been added to the list, when it is found add spectial annotions
    for j in range(len(words)):
      # check if the word is in the aspect terms list
      if words[j] in aspect_terms:
        # if its the first word of the tweet, no need to check if there's an aspect word before
        if j == 0:
          add_to_doc = words[j] + beginning_ate(emotion_list[counter_overall])
        elif j > 0:
          # if there's an aspect term before this word
          if words[j-1] in aspect_terms:
            add_to_doc = words[j] + inside_ate(emotion_list[counter_overall])
          else:
            add_to_doc = words[j] + beginning_ate(emotion_list[counter_overall])

      elif j not in aspect_terms:
        add_to_doc = words[j] + addition_no_ate

      new_tweet.append(add_to_doc)
    
    #print(new_tweet)
    save = input(f'{new_tweet} ------------ Save last reformatted text? Enter/n:')
    if save == '':
      convert_to_doc.append(new_tweet)
    if save == 'n':
      # set counter back and don't do anything with the tweet
      counter_overall -= 1
      print(f'counter overall set back to : {counter_overall}')
    
    counter_overall += 1

  return convert_to_doc


# ### Annotate
# 
# Subdivide the datasets into subsets and iterate over them to identify asepct terms, reformat and store in variable for later conversion into a .txt file.
# 
# This step requires iteratively uncommenting the individual subsets below to annotate them one after the other

# In[8]:


# choose and keep track of how many already annotated
#testing = twemlab_birmingham[80:90]
#print(boston_100.loc[99])

all_docs = []
i = 0


# In[9]:


# split whole dataframe into subsets of 100 tweets each
# twemlab_birmingham
# twemlab_boston

# choose and keep track of how many already annotated
#boston = twemlab_boston[0:100]             # done
#boston = twemlab_boston[100:200]           # done
#boston = twemlab_boston[200:300]           # done
#boston = twemlab_boston[300:400]           # done
#boston = twemlab_boston[400:500]           # done
#boston = twemlab_boston[500:600]           # done
#boston = twemlab_boston[600:]              # done

# birmingham
#birmingham = twemlab_birmingham[0:100]      # done
#birmingham = twemlab_birmingham[100:200]    # done
#birmingham = twemlab_birmingham[200:300]    # done
#birmingham = twemlab_birmingham[300:400]    # done
#birmingham = twemlab_birmingham[400:500]    # done

#birmingham = twemlab_birmingham[500:600]    # done
#birmingham = twemlab_birmingham[600:700]    # done
#birmingham = twemlab_birmingham[700:800]    # done
#birmingham = twemlab_birmingham[800:900]    # done
#birmingham = twemlab_birmingham[900:]       # done



# In[10]:


print(len(all_docs))
print(i)


# In[11]:


# split into subsets of 20 tweets that will be saved on the go
list_of_chunks = splitdataset(birmingham, 10)

for subset in list_of_chunks:
    convert_to_doc = annotate_aspects(df=subset, text_col='text', emotions_col='emotion')
    #convert_to_doc = annotate_aspects(df=subset, text_col='Tweet_text', emotions_col='emotion')

    # convert this chunk to a separate txt file to save progress up to here
  
    # keep track of how many tweets are going into the final reforatted document
 
    with open(f"../Data/twemlab_goldstandards_annotated_reformatted/subsets/re-formatted_birmingham_subset_{i}.txt", mode = "w") as f:
        for tweet in convert_to_doc:
            for word in tweet:
                f.write("%s\n" % word)
            f.write("\n")

    print(f"\n----------------\nSubset {i} was reformatted and saved into Data/twemlab_goldstandards_annotated_reformatted/subsets/re-formatted_birmingham_subset_{i}.txt\n----------------\n")

    # add all to a list to be converted to a single output document
    all_docs.append(convert_to_doc)
    i += 1


# ### Save all subsets together as new txt file
# 
# This step ONLY works if the variable "all_docs" from the previous step contains the labelled data. 

# In[38]:


# ALL TWEETS
tracker = 0
#with open("../Data/twemlab_goldstandards_annotated_reformatted/birmingham_pt1_annotated_reformatted.txt", mode = "w") as f:
with open("../Data/twemlab_goldstandards_annotated_reformatted/birmingham_pt2_annotated_reformatted.txt", mode = "w") as f:
#with open("../Data/twemlab_goldstandards_annotated_reformatted/boston_annotated_reformatted.txt", mode = "w") as f:
    f.write("-DOCSTART-\n\n")
    for doc in all_docs:
        for tweet in doc:
            #every 10th tweet
            for word in tweet:
                f.write("%s\n" % word)
            f.write("\n")
            tracker += 1
            
#print(f"{tracker} tweets were reformatted into Data/twemlab_goldstandards_annotated_reformatted/birmingham_pt1_annotated_reformatted.txt")
print(f"{tracker} tweets were reformatted into Data/twemlab_goldstandards_annotated_reformatted/birmingham_pt2_annotated_reformatted.txt")
#print(f"{tracker} tweets were reformatted into Data/twemlab_goldstandards_annotated_reformatted/boston_annotated_reformatted.txt")

# TRAIN FILE

tracker = 0
iterator = 0
#with open("../Data/twemlab_goldstandards_annotated_reformatted/birmingham_pt1_train.txt", mode = "w") as f:
with open("../Data/twemlab_goldstandards_annotated_reformatted/birmingham_pt2_train.txt", mode = "w") as f:
#with open("../Data/twemlab_goldstandards_annotated_reformatted/boston_train.txt", mode = "w") as f:
    f.write("-DOCSTART-\n\n")
    for doc in all_docs:
        for tweet in doc:
            #every 10th tweet
            if iterator % 10 != 0:
                for word in tweet:
                    f.write("%s\n" % word)
                f.write("\n")
                tracker += 1
            iterator += 1

#print(f"{tracker} tweets were reformatted into Data/twemlab_goldstandards_annotated_reformatted/birmingham_pt1_train.txt")
print(f"{tracker} tweets were reformatted into Data/twemlab_goldstandards_annotated_reformatted/birmingham_pt2_train.txt")
#print(f"{tracker} tweets were reformatted into Data/twemlab_goldstandards_annotated_reformatted/boston_train.txt")

# TEST FILE

tracker = 0
iterator = 0
#with open("../Data/twemlab_goldstandards_annotated_reformatted/birmingham_pt1_test.txt", mode = "w") as f:
with open("../Data/twemlab_goldstandards_annotated_reformatted/birmingham_pt2_test.txt", mode = "w") as f:
#with open("../Data/twemlab_goldstandards_annotated_reformatted/boston_test.txt", mode = "w") as f:
    f.write("-DOCSTART-\n\n")
    for doc in all_docs:
        for tweet in doc:
            #every 10th tweet
            if iterator % 10 == 0:
                for word in tweet:
                    f.write("%s\n" % word)
                f.write("\n")
                tracker += 1
            iterator += 1

#print(f"{tracker} tweets were reformatted into Data/twemlab_goldstandards_annotated_reformatted/bimingham_pt1_test.txt")
print(f"{tracker} tweets were reformatted into Data/twemlab_goldstandards_annotated_reformatted/bimingham_pt2_test.txt")
#print(f"{tracker} tweets were reformatted into Data/twemlab_goldstandards_annotated_reformatted/boston_test.txt")


# ### Merge Datasets into Single Files
# 
# Final Files:
# - Twemlab_all_annotated_reformatted.txt (100% of all data)
# - Twemlab_all_test.txt (10% of all data)
# - Twemlab_all_train.txt (90% of all data)

# In[42]:


# empty variables
d1 = d2 = d3 = ""

# read in the previous 3 files (boston reformatted, brimingham pt 1 reformatted, birmingham pt 2 reformatted)
#with open("../Data/twemlab_goldstandards_annotated_reformatted/boston_annotated_reformatted.txt") as fp:
#with open("../Data/twemlab_goldstandards_annotated_reformatted/boston_test.txt") as fp:
with open("../Data/twemlab_goldstandards_annotated_reformatted/boston_train.txt") as fp:
    d1 = fp.read()

#with open("../Data/twemlab_goldstandards_annotated_reformatted/birmingham_pt1_annotated_reformatted.txt") as fp:
#with open("../Data/twemlab_goldstandards_annotated_reformatted/birmingham_pt1_test.txt") as fp:
with open("../Data/twemlab_goldstandards_annotated_reformatted/birmingham_pt1_train.txt") as fp:
    d2 = fp.read()

#with open("../Data/twemlab_goldstandards_annotated_reformatted/birmingham_pt2_annotated_reformatted.txt") as fp:
#with open("../Data/twemlab_goldstandards_annotated_reformatted/birmingham_pt2_test.txt") as fp:
with open("../Data/twemlab_goldstandards_annotated_reformatted/birmingham_pt2_train.txt") as fp:
    d3 = fp.read()

# merge files
data = d1 +  d2 + d3 

#with open("../Data/twemlab_goldstandards_annotated_reformatted/twemlab_all_annotated_reformatted.txt", "w") as fp:
#with open("../Data/twemlab_goldstandards_annotated_reformatted/twemlab_all_test.txt", "w") as fp:
with open("../Data/twemlab_goldstandards_annotated_reformatted/twemlab_all_train.txt", "w") as fp:
    fp.write(data)

