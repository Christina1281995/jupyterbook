#!/usr/bin/env python
# coding: utf-8

# ## Document Level Sentiment Analysis with Deep Learning Models

# ##### Setup

# In[1]:


#import tensorflow as tf
import torch
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig, AutoTokenizer, AutoModel
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet as bertweetpreprocess
import fastai
from datasets import load_dataset

import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import pandas as pd                                                    # for data handling
import xml.etree.cElementTree as ET                                    # for parsing XML file

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

import pysal
from pysal.lib import weights
import seaborn as sns
sns.set_style("darkgrid")

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


# In[2]:


# world centroids
# https://github.com/gavinr/world-countries-centroids/blob/master/dist/countries.csv
world = pd.read_csv('https://raw.githubusercontent.com/gavinr/world-countries-centroids/master/dist/countries.csv')


# #### General Helper Functions

# In[3]:


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
    chr = 0
    for item in in_df['geom']:
        chr += 1
        print(chr)
        new_geom.append(Geometry(item).shapely)
    
    # add as new column to input df
    in_df["geometry"] = new_geom
    
    # create geopandas GeoDataFrame
    df_geo = gpd.GeoDataFrame(in_df, crs="EPSG:4326")
    s = gpd.GeoSeries(df_geo['geometry'])
    df_geo['geom'] = df_geo['geometry']
    df_geo['geometry'] = df_geo['geom'].to_crs('epsg:3785').centroid.to_crs(df_geo.crs)
    
    return df_geo

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
    positives = df[df[labels_col] == labels[0]]
    negatives = df[df[labels_col] == labels[2]]

    senti_dfs = [positives, negatives]
    # colors available at https://matplotlib.org/1.2.1/mpl_examples/pylab_examples/show_colormaps.pdf
    cmap = ['YlGn', 'OrRd']

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

    fig.write_html("output.html")

    return sorted_df, fig

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
              lat=df.geometry, #y
              lon=df.geometry, #x
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

# create interactive folium map 

def create_folium_map(df, map_center, tiles, zoom_start, n_labels, text_col, senti_label_col, senti_score_col, senti_num_label_col):
    '''
    Input: 
        df: a geodataframe (must have 'geometry' column)
        map_center: a string country name, or a list with two coordinates e.g. [37.0902, -95.7129]
        tiles: stirng, the background map style e.g. 'cartodbpositron' or 'Stamen Toner'
        zoom_start: int, higher numbers a more zoomed out
        n_labels: int, either two or three
        text_col: string, the name of the tweets text column in the df
        senti_label_col: string, the name of the labels column in the df
        senti_score_col: string, the name of the sentiment (softmax / confidence) score in the df
        senti_num_label_col: string, the name of the column containing numerical labels (0, 1, 2)
    Output:
        map: an interactive folium map
    '''  
    
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
    if n_labels == 3:
        # red, yellow, green
        cmap = ['#FF0000','#FCFF00','#62FF00']
    else: 
        # red, green
        cmap = ['#FF0000','#62FF00']     

    # iterate over df rows
    for i in range(0, len(df)):

        # logic to split tweet text for the pop ups (had to be done manually, couldn't find appropriate function included in folium.map.tooltip - html max width only applies to the box)
        text = ''

        # if text is longer than 40 characters, split into words list and count
        if len(df.iloc[i][text_col]) > 40: 
            word_list = df.iloc[i][text_col].split()
            length = len(word_list)

            # first part of text is the same regardless of length
            text_pt1 = '<b>Sentiment:</b> ' + df.iloc[i][senti_label_col] + '<br><b>Softmax Score:</b> ' + str(df.iloc[i][senti_score_col]) + '<br><b>Tweet:</b> '

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
            text = '<b>Sentiment:</b> ' + df.iloc[i][senti_label_col] + '<br><b>Softmax Score:</b> ' + str(df.iloc[i][senti_score_col]) + '<br><b>Tweet:</b> ' + df.iloc[i][text_col]

        map.add_child(
            folium.CircleMarker(
                location=[df.iloc[i].geometry.y, df.iloc[i].geometry.x],
                radius = 5,
                tooltip= folium.map.Tooltip(text),
                fill_color=cmap[(df.iloc[i][senti_num_label_col] +1 )],
                fill_opacity = 0.4,
                stroke=False
            )
        )


    # add button for full screen
    folium.plugins.Fullscreen().add_to(map)
    
    return map

def create_piechart(ratios, labels, title):

    #colors
    if len(labels) == 3:
        colors = ['#73F178','#F8FF72', '#F17373']
    elif len(labels) == 2:
        colors = ['#73F178', '#F17373']
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


# error matrices

# persist lists for each model
precision = []
recall = []
accuracy = []
f1 = []
model_names = []
    
def performance_metrics(df, model_name, y, y_hat):
    '''
    Input: 
        df: the dataframe
        model_name: string, name of model for entry in dataframe
        y: the column name of the goldstandard sentiment label (numeric)
        y_hat: the column name of the predicted sentiment label (numeric)
    '''
    model_names.append(model_name)
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for index, row in df.iterrows():
        if row[y] == 1 and row[y_hat] == 1:
            tp += 1
        elif row[y] == 1 and row[y_hat] == 0:
            fn += 1
        elif row[y] == 0 and row[y_hat] == 1:
            fp +=  1
        elif row[y] == 0 and row[y_hat] == 0:
            tn += 1
    
    # add new metrics for current model
    prec = tp / (tp + fp)
    precision.append(np.round(prec, 2))
    rec = tp / (tp + fn)
    recall.append(np.round(rec, 2))
    acc = (tn + tp) / (tn + fp + tp + fn)
    accuracy.append(np.round(acc, 2))
    f = 2*((prec * rec)/(prec + rec))
    f1.append(np.round(f, 2))

    #re-create dataframe with newest addition
    performances = pd.DataFrame(list(zip(precision, recall, accuracy, f1)), index = model_names, columns = ['Precision', 'Recall', 'Accuracy', 'F1'])
    
    return performances

    
#    print(f"Performance Metrics\n\nPrecision: {np.round(precision, 2)}\nRecall: {np.round(recall,2)}\nAccuracy: {np.round(accuracy,2)}\nF1: {np.round(f1,2)}")


# ##### Load Twitter Datasets
# 
# <!-- Twemlab Goldstandard Tweets (1 pos, 0 neu, -1 neg) -->

# In[4]:


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
 # df1.head()
    
# Load TwEmLab Boston Tweets
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
print(f'Twemlab Goldstandard dataset\nSize: {len(twemlab)}\nPolarity: (1 pos, 0 neu, -1 neg)\nSample of the data:')

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


# In[5]:


# show df
demo_cols_tl = twemlab[['text', 'label', 'sentiment_label']]
col_names = {'text': 'Text', 'label': 'Label', 'sentimemt_label':'Sentiment'}
demo_cols_tl = demo_cols_tl.rename(columns=col_names)

# check dataset
demo_cols_tl.sample(3).style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])])


# SemEval 2014 Dataset (1 pos, 0 neu, -1 neg)

# In[6]:


# load from url
url = 'https://git.sbg.ac.at/s1080384/sentimentanalysis/-/raw/main/data/Restaurants_Train_v2.xml'
document = urllib.request.urlopen(url).read()
root = ET.fromstring(document)

# show example text from dataset (semeval 2014)
# root[0][0].text

# create dataframe from xml file
data = []
for sentence in root.findall('sentence'):
    text = sentence.find('text').text
    aspectCategories = sentence.find('aspectCategories')
    for aspectCategory in aspectCategories.findall('aspectCategory'):
        category = aspectCategory.get('category')
        polarity = aspectCategory.get('polarity')
        data.append((text, category, polarity))

semeval = pd.DataFrame(data,columns=['text','category','polarity'])

# add column with 0s and 1s in place of textual sentiment values
semeval['sentiment'] = np.where(semeval['polarity']== 'positive', 1, np.where(semeval['polarity']== 'negative',-1, 0))

print(f'SemEval Goldstandard dataset size: {len(semeval)}\nPolarity: (1 pos, 0 neu, -1 neg)\nSample of the data:')

semeval.head()
# preview Semeval
# show df
demo_cols_se = semeval[['text', 'polarity', 'sentiment']]
col_names = {'text': 'Text', 'polarity': 'Polarity', 'sentimemt':'Sentiment'}
demo_cols_se = demo_cols_se.rename(columns=col_names)


# In[7]:


# # check dataset
demo_cols_se.sample(3).style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])])


# Twitter dataset from AIFER for sentiment classification (no labels)

# In[8]:


# Load AIFER Dataset
aifer = pd.read_csv("../Data/Disaster_responses/ahrtal_tweets.csv", sep="\t")
print(f"AIFER example dataset size: {len(aifer)}\nPolarity: none (unlabelled data)\nSample of the data:")

# show df
demo_cols = aifer[['date','text', 'tweet_lang']]
col_names = {'date': 'Date', 'text': 'Text', 'tweet_lang': 'Language'}
demo_cols = demo_cols.rename(columns=col_names)


# In[9]:


demo_cols.sample(3).style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])])


# #### Transformers-based NLP Models

# ##### Why Transformers?
# 
# In recent years, the transformer model has revolutionized the field of NLP. This 'new' deep learning approach has been highly successful in a variety NLP tasks, including sentiment analysis. The transformer model offers several advantages over traditional machine learning and even other deep learning approaches and have been shown to outperform traditional machine learning and other deep learning methods on NLP tasks, particularly sentiment analysis. Some of the key advantages it has are:
# 
# - The **encoder-decoder framework**: Encoder generates a representation of the input (semantic, context, positional) and the decoder generates output. Common use case: sequence to sequence translation tasks.
# 
# - **Attention mechanisms**: Deals with the information bottleneck of the traditional encoder-decoder architecture (where one final encoder hidden state is passed to decoder) by allowing the decoder to access the hidden states at each step and being able to prioritise which state is most relevant. 
# 
# - **Transfer learning** (i.e. fine-tuning a pre-trained language model)
# 
# <br>
# <br>
# 
# <img src="https://www.oreilly.com/api/v2/epubs/9781098136789/files/assets/nlpt_0101.png" align="center">

# ##### A note on Attention
# 
# In transformers, multi-head scaled-dot product attention is usually used. This attention mechanism allows the Transformer to capture global dependencies between different positions in the input sequence, and to weigh the importance of different parts of the input when making predictions.
# 
# In scaled dot-product attention a dot product between the query, key, and value vectors is computed for each position in the sequence. The attention mechanism is repeated multiple times with different linear projections (hence "multi-head") to capture different representations of the input.
# 
# 
# 
# <details><summary>Code implementation</summary>
# 
# ```
# class AttentionHead(nn.Module):
#     def __init__(self, embed_dim, head_dim):
#         super().__init__()
#         self.q = nn.Linear(embed_dim, head_dim)
#         self.k = nn.Linear(embed_dim, head_dim)
#         self.v = nn.Linear(embed_dim, head_dim)
# 
#     def forward(self, hidden_state):
#         attn_outputs = scaled_dot_product_attention(
#             self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
#         return attn_outputs
#      
# 
# class MultiHeadAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         embed_dim = config.hidden_size
#         num_heads = config.num_attention_heads
#         head_dim = embed_dim // num_heads
#         self.heads = nn.ModuleList(
#             [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
#         )
#         self.output_linear = nn.Linear(embed_dim, embed_dim)
# 
#     def forward(self, hidden_state):
#         x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
#         x = self.output_linear(x)
#         return x
# ```
# 
# </details>
# 
# <img src="https://raw.githubusercontent.com/nlp-with-transformers/notebooks/e3850199388f4983cc9799135977f0a6b06d5a79//images/chapter03_multihead-attention.png">
# 

# In[10]:


from transformers import AutoTokenizer
from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show

model_ckpt = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)
text = "The hurricane trashed our entire garden"


# Here's a visual representation of the attention machism at work with a demo text "The hurricane trashed our entire garden":

# In[11]:


text = "The hurricane trashed our entire garden"
show(model, "bert", tokenizer, text, display_mode="light", layer=0, head=8)


# In[12]:


from transformers import AutoTokenizer, AutoModel, utils
from bertviz import head_view

utils.logging.set_verbosity_error()  # Suppress standard warnings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)

inputs = tokenizer.encode("The hurricane trashed our entire garden", return_tensors='pt')
outputs = model(inputs)
attention = outputs[-1]  # Output includes attention weights when output_attentions=True
tokens = tokenizer.convert_ids_to_tokens(inputs[0]) 

html_head_view = head_view(attention, tokens, html_action='return')

with open("head_view.html", 'w') as file:
    file.write(html_head_view.data)


# In[13]:


# Import specialized versions of models (that return query/key vectors)
from bertviz.transformers_neuron_view import BertModel, BertTokenizer
from bertviz.neuron_view import show

model_type = 'bert'
model_version = 'bert-base-uncased'
do_lower_case = True
sentence_a = "The hurricane trashed our entire garden"
sentence_b = "We'll be cleaning up forever"
model = BertModel.from_pretrained(model_version, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
html_neuron_view = show(model, model_type, tokenizer, sentence_a, sentence_b, layer=2, head=0, html_action='return')

with open("neuron_view.html", 'w') as file:
    file.write(html_neuron_view.data)


# In[14]:


from IPython.display import display, HTML
display(HTML('https://raw.githubusercontent.com/Christina1281995/demo-repo/main/neuron_view.html'))


# In[15]:


from IPython.display import display, HTML
display(HTML('neuron_view.html'))


# ##### Types of Transformers
# 
# A useful way of differentiating between the many transformers-based models that have sprung up in recent years is by their use of the transformer-blocks (encoder and decoder).
# 
# The **encoder** block's main role is to "update" the input embeddings to produce representations that encode some contextual information (called the context vector). Many models make use only of the encoder block and add a linear layer as a classifier. BERT is perhaps the most prominent example of an encoder-based architecture (the name is literally "bidirectional encoder representations from transformers").
# 
# The **decoder** block uses the context vector from the encoder to generate an output sequence. Like the encoder, the decoder also computes self-attention scores and processes the context vector through multiple feedforward layers. The decoder also includes an attention mechanism that allows it to focus on specific parts of the input sequence when generating the output. Decoder-based models are exceptionally good at predicting the next word in a sequence and are therefore often used for text geneeration tasks. Progress here has been fuelled by using larger datasets and scaling the language models to larger and larger sizes  (GPT-3 has 175 billion parameters). The most famous example of decoder-based models are the Generative Pretrained Transformer (GPT) models by OpenAI. 
# 
# 
# <br>
# <br>
# 
# <img src="https://raw.githubusercontent.com/nlp-with-transformers/notebooks/920616f111e65fe171f784df87f75dd16d8e7a67/images/chapter03_transformers-compact.png" width="70%">

# ##### Two ways of using Pre-trained Transformer Language Models
# 
# **1. For Feature Extraction:**
# 
# To extract the features of the encoder and then separately train a clssifier using the hidden states. This method essentially freezes the body's weights during training.
# 
# <img src="https://raw.githubusercontent.com/nlp-with-transformers/notebooks/e3850199388f4983cc9799135977f0a6b06d5a79//images/chapter02_encoder-feature-based.png" width="60%">
# 

# In[16]:


# load as datasetdict
semeval = load_dataset("csv", data_files="..\Data\Restaurants_Train_v2.csv", sep=",",names=["text", "label_name", "label"])


# In[17]:


model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

# tokenize batch-wise but set batch size to none so all in one go
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(device) for k,v in batch.items() 
              if k in tokenizer.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}
    
# map tokenizer to entire dataset 
semeval_encoded = semeval.map(tokenize, batched=True, batch_size=None)

# set correct format (tensors) because that's what the model expects as input
semeval_encoded.set_format("torch",columns=["input_ids", "attention_mask", "label"])

# extract hidden states
semeval_hidden = semeval_encoded.map(extract_hidden_states, batched=True)


# In[18]:


# only have train data (without validation set in this case)
X_train = np.array(semeval_hidden["train"]["hidden_state"])
y_train = np.array(semeval_hidden["train"]["label"])
print(f"The shape of [dataset_size, hidden_dim]: {X_train.shape}")


# For visualisation purposes, the 768 dimensions can be reduced down to only 2 dimensions using UMAP. It will rescale each input to lie on a 2D space with values between 0 and 1 on each axis.

# In[19]:


from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

# Scale features to [0,1] range
X_scaled = MinMaxScaler().fit_transform(X_train)
# Initialize and fit UMAP
mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
# Create a DataFrame of 2D embeddings
df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
df_emb["label_name"] = y_train
df_emb.head()


# These 2D vectors can now be plotted: 

# In[20]:


fig, axes = plt.subplots(1, 3, figsize=(10,4))
axes = axes.flatten()
# colors for 2D space
cmaps = ["Reds", "Oranges", "Greens"]
# labels
labels = ['negative', 'neutral', 'positive']


for i, (label, cmap) in enumerate(zip(labels, cmaps)):
    # label_name i-1 because the "label_name" column ranges from -1 to 1
    df_emb_sub = df_emb.query(f"label_name == {i-1}")
    axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap,
                   gridsize=20, linewidths=(0,))
    axes[i].set_title(label)
    axes[i].set_xticks([]), axes[i].set_yticks([])

plt.suptitle("A 2D Visual Representation: The Extracted Features from the SemEval Dataset by the DistilBERT Encoder")
plt.tight_layout()
plt.show()
     


# <hr>

# 
# **2. For Fine-Tuning:**
# 
# <img src="https://raw.githubusercontent.com/nlp-with-transformers/notebooks/e3850199388f4983cc9799135977f0a6b06d5a79//images/chapter02_encoder-fine-tuning.png" width="60%">

# In[21]:


# explain methods briefly
# esp. explain why transformers and how attention works
# Expl Lexicon (VADER)
# Expl Naive Bayes
# Transformers: BERT-based, others
# Show attention mechanism 


# #### RoBERTa
# 
# RoBERTA (Robustly Optimized BERT Pretraining Approach) has the same architecture as BERT but marks an improved version of BERT for several reasons: 
# 
# - RoBERTa was trained on **10x as much data** as was used for BERT training (160GB, compared to 16GB for BERT)
# - **Dynamic masking** was used during training, rather than fixed masking in BERT
# - the **next sentence prediction was left out** during training, which is arguably not essential especially when considering tweets. Here is a view of the average tweet length in the Twemlab dataset:

# In[22]:


# plot nr of words per tweet
twemlab["Words Per Tweet in Twemlab Dataset"] = twemlab["text"].str.split().apply(len)
twemlab.boxplot("Words Per Tweet in Twemlab Dataset", by="label", grid=False, showfliers=False,
           color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()


# The model used here is the cardiffnlp/twitter-roberta-base-sentiment-latest
# 
# Source: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest?text=Covid+cases+are+increasing+fast%21
# 
# This is a roBERTa-base model trained on ~124M tweets from January 2018 to December 2021 ([see here](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m?text=The+goal+of+life+is+%3Cmask%3E.), and finetuned for sentiment analysis with the TweetEval benchmark. The original roBERTa-base model can be found [here](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m) and the original reference paper is [TweetEval](https://github.com/cardiffnlp/tweeteval). This model is suitable for English.
# 

# **From the Authors**
# 
# [TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification (Barbieri et al., 2020)](https://arxiv.org/pdf/2010.12421.pdf)
# 
# According to the authors of the mode, among all the available language models RoBERTa is one of the top performing systems in GLUE (Liu et al., 2019). It does not employ the Next Sentence Prediction (NSP) loss (Devlin et al., 2018), making the model **more suitable for Twitter** where most tweets are composed of a single sentence. 
# 
# Three different RoBERTa variants were used: 
# - pre-trained RoBERTabase (RoB-Bs),
# - the same model but re-trained on Twitter (RoB-RT) and 
# - trained on Twitter from scratch (RoB-Tw)
# 
# Results indicate that RoB-RT perform the best for sentiment analysis and that the RoBERTa model outperforms the comparison methods in all investigated NLP tasks:
# 
# <img src="https://github.com/Christina1281995/demo-repo/blob/main/tweeteval.PNG?raw=true" width="80%">

# In[23]:


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
 
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


# In[24]:


tw_rob_base_sent_lat = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer_tw_rob_base_sent_lat = AutoTokenizer.from_pretrained(tw_rob_base_sent_lat)
config_tw_rob_base_sent_lat = AutoConfig.from_pretrained(tw_rob_base_sent_lat)
# PT
model_tw_rob_base_sent_lat = AutoModelForSequenceClassification.from_pretrained(tw_rob_base_sent_lat)
#model.save_pretrained(tw_rob_base_sent_lat)


# In[25]:


# testing
text = "Well isn't this just terrible..."
text = preprocess(text)
encoded_input = tokenizer_tw_rob_base_sent_lat(text, return_tensors='pt')
output = model_tw_rob_base_sent_lat(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
# Print labels and scores
ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = config_tw_rob_base_sent_lat.id2label[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")


# In[26]:


# apply in the form of a function so it can be called for usecase later on
def robertabase_apply(dataset):
    
    # create variable for labels (good to bad)
    labels= ['positive', 'neutral', 'negative']
    
    # lists to be filled
    cardiffroberta_sentiment_prediction = []
    cardiffroberta_sentiment_prediction_softmax = []
    cardiffroberta_sentiment_prediction_num = []
    
    # iterate over dataset
    for index, row in dataset.iterrows():
        text = row['text']
        text = preprocess(text)
        encoded_input = tokenizer_tw_rob_base_sent_lat(text, return_tensors='pt')
        output = model_tw_rob_base_sent_lat(**encoded_input)
        score = np.round(softmax(output[0][0].detach().numpy()), 4)
        label = config_tw_rob_base_sent_lat.id2label[np.argsort(score)[::-1][0]]
        cardiffroberta_sentiment_prediction.append(label)
        cardiffroberta_sentiment_prediction_softmax.append(max(score))
        # positive label
        if label == labels[0]:
            cardiffroberta_sentiment_prediction_num.append(1)
        # negative label
        elif label == labels[2]:
            cardiffroberta_sentiment_prediction_num.append(-1)
        # neutral label
        else:
            cardiffroberta_sentiment_prediction_num.append(0)


    dataset['cardiffroberta_sentiment_prediction'] = cardiffroberta_sentiment_prediction
    dataset['cardiffroberta_sentiment_prediction_softmax'] = cardiffroberta_sentiment_prediction_softmax
    dataset['cardiffroberta_sentiment_prediction_num'] = cardiffroberta_sentiment_prediction_num

    model_name = "cardiffroberta"
    
    # model name and labels will be needed later on as input variables for plotting and mapping
    print("Variables that will later be required for plotting and mapping:")
    return model_name, labels


# In[27]:


# apply model to goldstandard
robertabase_apply(dataset=twemlab)


# In[120]:


# start assessing model performances - DF is updated each time it the function below is called
perf = performance_metrics(df=twemlab, model_name="Twitter Roberta Base Sent Latest", y='sentiment_label', y_hat='cardiffroberta_sentiment_prediction_num')

length = len(perf)
perf.head(length)


# <hr>
# 
# ##### RoBERTa Example use case
# 
# Polarity-Based Sentiment Analysis of Georeferenced Tweets Related to the 2022 Twitter Acquisition (Schmidt et a., 2023)
# 
# We performed a simple document-level, polarity-based sentiment analysis (cf. Figure 1) in Python 3.9 to categorise the Tweets as either “positive”, “neutral“ or "negative". 
# 
# https://www.mdpi.com/2078-2489/14/2/71
# 
# <img src="https://github.com/Christina1281995/demo-repo/blob/main/takeover.PNG?raw=true" width="60%">
# 
# <img src="https://github.com/Christina1281995/demo-repo/blob/main/timeline.PNG?raw=true">
# 
# <img src="https://github.com/Christina1281995/demo-repo/blob/main/hotspots.PNG?raw=true">
# 
# 
# 

# <hr>
# <br>
# 
# #### BERTweet
# 
# model: finiteautomata/bertweet-base-sentiment-analysis
# 
# Source: https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis?text=I+hate+this
# 
# This is a BERTweet-base RoBERTa model trained on SemEval 2017 (~40k Tweets). It uses POS, NEG, NEU labels and is suitable for English and Spanish languages. pysentimiento is an open-source library for non-commercial use and scientific research purposes only. Please be aware that models are trained with third-party datasets and are subject to their respective licenses.
# 
# [pysentimiento: A Python Toolkit for Sentiment Analysis and SocialNLP tasks (Perez et al., (2021)](https://arxiv.org/pdf/2106.09462.pdf)
# 
# The aim of this research was to perform both Sentiment Analysis and Emotion Analysis on Twitter datasets and identify the best performing meodels. For Sentiment Analysis, two datasets were used: TASS 2020 Task 1 and SemEval 2017 Task 4 Subtask 1. Both datasets were labeled with general polarity using positive, negative and neutral outcomes.
# 
# A series of models were tested for the given tasks: For English, they tested BERT base, RoBERTa base, BERTweet and multilingual models, namely DistilBERT and mBERT. Spanish has lesser availability of models: they used BETO, a Spanish-trained version of BERT, and the aforementioned multilingual models. The authors utilise the BERTweet model as a base model for their sentiment analysis task.
# 
# <br>
# 
# **More on BERTweet**
# 
# <img align="right" src="https://miro.medium.com/max/740/1*G6PYuBxc7ryP4Pz7nrZJgQ@2x.png" width="40%">
# 
# 
# [BERTweet: A pre-trained language model for English Tweets (Nguyen et al., 2020)](https://arxiv.org/pdf/2005.10200.pdf)
# 
# BERTweet uses the same architecture as BERTbase, which is trained with a masked language modeling objective (Devlin et al., 2019). BERTweet pre-training procedure is based on RoBERTa (Liu et al., 2019) which optimizes the BERT pre-training approach for more robust performance. <br>
# <br>
# The authors use an 80GB pre-training dataset of uncompressed texts, containing 850M Tweets (16B word tokens). Here, each Tweet consists of at least 10 and at most 64 word tokens. 

# In[106]:


bertweetanalyzer = create_analyzer(task="sentiment", lang="en")


# In[107]:


# testing
text = "This is aweful"
bertweetpreprocess(text)
result = bertweetanalyzer.predict(text)
print(result.output)
print(np.round(result.probas['NEG'], 4))


# In[108]:


# apply in the form of a function so it can be called for usecase later on
def bertweet_apply(dataset):
    
    # create labels variable
    labels = ['POS', 'NEU', 'NEG']
    
    # lists to be filled
    bertweet_sentiment_prediction = []
    bertweet_sentiment_prediction_softmax = []
    bertweet_sentiment_prediction_num = []  
    
    # iterate over dataframe
    for index, row in dataset.iterrows():
        text = row['text']
        text = bertweetpreprocess(text)
        result = bertweetanalyzer.predict(text)
        label = result.output
        bertweet_sentiment_prediction.append(label)
        bertweet_sentiment_prediction_softmax.append(np.round(result.probas[label], 4))
        if label == labels[0]:
            bertweet_sentiment_prediction_num.append(1)
        elif label == labels[2]:
            bertweet_sentiment_prediction_num.append(-1)
        else:
            bertweet_sentiment_prediction_num.append(0)


    dataset['bertweet_sentiment_prediction'] = bertweet_sentiment_prediction
    dataset['bertweet_sentiment_prediction_softmax'] = bertweet_sentiment_prediction_softmax
    dataset['bertweet_sentiment_prediction_num'] = bertweet_sentiment_prediction_num
    
    model_name = "bertweet"
    
    # model name and labels will be needed later on as input variables for plotting and mapping
    print("Variables that will later be required for plotting and mapping:")    
    return model_name, labels


# In[109]:


# apply model to goldstandard
bertweet_apply(dataset=twemlab)


# In[121]:


# start assessing model performances - the perf dataframe is updated each time the function below is called
perf = performance_metrics(df=twemlab, model_name="BERTweet base sentiment analysis", y='sentiment_label', y_hat='bertweet_sentiment_prediction_num')

length = len(perf)
perf.head(length)


# <hr>
# 
# #### Fine-Tuned Downstream Sentiment Analysis
# 
# Model: Seethal/sentiment_analysis_generic_dataset
# 
# This is a BERT base model (uncased), pretrained on English language using a masked language modeling (MLM) objective. This model is uncased: it does not make a difference between english and English.
# 
# This is a fine-tuned downstream version of the bert-base-uncased model for sentiment analysis, this model is not intended for further downstream fine-tuning for any other tasks. This model is trained on a classified dataset for text classification.
# 

# In[111]:


tokenizer_seethal_gen_data = AutoTokenizer.from_pretrained("Seethal/sentiment_analysis_generic_dataset")
model_seethal_gen_data = AutoModelForSequenceClassification.from_pretrained("Seethal/sentiment_analysis_generic_dataset")


# In[112]:


# testing 
text = 'Today is an amazing day.'
text = preprocess(text)
encoded_input = tokenizer_seethal_gen_data(text, return_tensors='pt')
output = model_seethal_gen_data(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
# Print labels and scores
ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = config_tw_rob_base_sent_lat.id2label[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")


# In[116]:


# apply in the form of a function so it can be called for usecase later on
def seethal_gen_data_apply(dataset):
    
    # create variable for labels (good to bad)
    labels= ['positive', 'neutral', 'negative']
    
    # lists to be filled
    seethal_gen_data_sentiment_prediction = []
    seethal_gen_data_sentiment_prediction_softmax = []
    seethal_gen_data_sentiment_prediction_num = []
    
    # iterate over dataset
    for index, row in dataset.iterrows():
        text = row['text']
        text = preprocess(text)
        encoded_input = tokenizer_seethal_gen_data(text, return_tensors='pt')
        output = model_seethal_gen_data(**encoded_input)
        score = np.round(softmax(output[0][0].detach().numpy()), 4)
        label = config_tw_rob_base_sent_lat.id2label[np.argsort(score)[::-1][0]]
        seethal_gen_data_sentiment_prediction.append(label)
        seethal_gen_data_sentiment_prediction_softmax.append(max(score))
        # positive label
        if label == labels[0]:
            seethal_gen_data_sentiment_prediction_num.append(1)
        # negative label
        elif label == labels[2]:
            seethal_gen_data_sentiment_prediction_num.append(-1)
        # neutral label
        else:
            seethal_gen_data_sentiment_prediction_num.append(0)


    dataset['seethal_gen_data_sentiment_prediction'] = seethal_gen_data_sentiment_prediction
    dataset['seethal_gen_data_sentiment_prediction_softmax'] = seethal_gen_data_sentiment_prediction_softmax
    dataset['seethal_gen_data_sentiment_prediction_num'] = seethal_gen_data_sentiment_prediction_num

    model_name = "seethal_gen_data"
    
    # model name and labels will be needed later on as input variables for plotting and mapping
    print("Variables that will later be required for plotting and mapping:")
    return model_name, labels


# In[117]:


# apply model to goldstandard
seethal_gen_data_apply(dataset=twemlab)


# In[122]:


# start assessing model performances - DF is updated each time it the function below is called
perf = performance_metrics(df=twemlab, model_name="Seethal Senti Analysis Generic Dataset", y='sentiment_label', y_hat='seethal_gen_data_sentiment_prediction_num')

length = len(perf)
perf.head(length)


# ##### Challenges with Transformers
# 
# - Language
# - Data availability
# - Working with long documents (more than paragraphs)
# - opacity (black box)
# - bias (trained on data on the internet)

# <hr>
# 
# ##### Useful for Disaster Responses?
# 
# By analyzing the sentiment contained in large volumes of social media posts related to a disaster, models like those shown above can provide **valuable insights into the public's perception** of the situation, the **effectiveness** of response efforts, and the **needs and concerns** of those affected.
# 
# Doc-Level sentimen analysis can be used to monitor the overall **sentiment trend over time** and identify areas where response efforts may be lacking or where additional resources may be needed. It can also be used to identify specific **topics and issues that are of concern to the public**, such as the availability of food, water, and medical supplies, or the level of safety in evacuation centers.

# In[124]:


# apply to use case (trained for english tweets)
aifer_en = aifer[aifer['tweet_lang'] == 'en']

name, labels = bertweet_apply(aifer_en)


# In[129]:


# show df
demo_cols = aifer_en[['text', 'bertweet_sentiment_prediction']]
col_names = {'text': 'Text', 'bertweet_sentiment_prediction': 'Sentiment'}
demo_cols = demo_cols.rename(columns=col_names)
demo_cols.sample(3).style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])])


# In[133]:


# count label frequencies and save in list
pos = list(aifer_en[name + '_sentiment_prediction']).count(labels[0])
neu = list(aifer_en[name + '_sentiment_prediction']).count(labels[1])
neg = list(aifer_en[name + '_sentiment_prediction']).count(labels[2])
ratios = [pos, neu, neg]
print(f"Ratios between pos, neu, and neg sentiments: {ratios}")

# create simple piechart to show ratios between the sentiment labels
create_piechart(ratios, labels=labels, title='Sentiments in AIFER Tweets')


# In[132]:


create_pos_neg_tweet_wordcloud(df=aifer_en, labels_col= name +'_sentiment_prediction', labels=labels)


# ##### Geographic Sentiments

# In[182]:


# create copy of the dataframe
data_copy = aifer_en.copy()

# create copy of the dataframe
data_copy = aifer_en.copy()
data_copy['geometry'] = gpd.GeoSeries.from_wkt(data_copy['geom'])


# In[175]:


# create an interactive folium map
# for map center USA is 'United States', UK is 'United Kingdom'
from folium.plugins import FloatImage

folium_map = create_folium_map(df = data_copy, 
                               map_center = 'Germany', 
                               tiles = 'cartodbpositron', 
                               zoom_start = 6, 
                               n_labels = 3, 
                               text_col = 'text', 
                               senti_label_col = name + '_sentiment_prediction', 
                               senti_score_col = name + '_sentiment_prediction_softmax', 
                               senti_num_label_col = name + '_sentiment_prediction_num'
                              )

# display map
folium_map


# ##### Sentiments Over Time

# In[178]:


# Set time step: day, week, month, or year
time_interval = 'Day'


# In[202]:


from IPython.display import display, HTML

# create a timeline showing sentiment quantities over time
time_df, fig = create_df_for_time_chart_three_labels(df=aifer_en, time_interval=time_interval, labels_col=name+'_sentiment_prediction', labels=labels)
# fig.show()


# Tweet Sentiments over Time (Day Intervals)

# In[203]:


display(HTML('output.html'))


# In[193]:


# create an animated plotly timeseries map

# style:    'carto-positron', 'open-street-map', 'white-bg', 'carto-darkmatter', 'light', 'stamen- terrain', 'stamen-toner', 'stamen-watercolor', 'basic', 'streets',  
#           'dark', 'satellite', 'satellite- streets', 'outdoors'

create_animated_time_map_three_labels(df=data_copy, time_interval=time_interval, label_col=name+"_sentiment_prediction", title='Sentiments', labels=labels, style='carto-positron')

