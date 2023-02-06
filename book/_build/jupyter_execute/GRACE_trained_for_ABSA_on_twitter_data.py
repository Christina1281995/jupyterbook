#!/usr/bin/env python
# coding: utf-8

# ## Aspect-Based Sentiment Analysis on Twitter Data
# 
# <img align='right' src='https://git.sbg.ac.at/s1080384/sentimentanalysis/-/wikis/uploads/8729054731d3bf4f16e1d5a0b2fa23f5/absa.png' alt='An example of the four key sentiment elements of ABSA. (Zhang et al., 2022)' width='50%'>
# 
# Recently, a more fine-grained "aspect-based sentiment analysis" (short ABSA) has been taking the lead from the traditional sentence and document-level sentiment classification. It's popularity has been fuelled by the increasing capabilities of deep learning models and the lacking granularity in convencional document-level sentiment analysis. ABSA **considers text data at the token level**, whereas document-level analyses are based on the assumption that the document is concerned with a single topic, which often proves untrue.
# 
# Zhang et al. (2022) summarize this trend towards more fine-grained analysis levels:
# 
# > _In the ABSA problem, the concerned target on which the sentiment is expressed shifts from an entire document to an entity or a certain aspect of an entity._
# 
# The term "aspect" can generally refer to both the aspect of an entity as well as a special "general" aspect. Hence, it is often used to collectively refer to the entity or an entity's aspect.
# 
# **ABSA encompasses** the identification of one or more of four sentiment elements. Depending on the goal researchers set, Zhang et al. (2022) divide their ABSA methods into either **Single ABSA** tasks (the more conventional method for ABSA, where a method is developed to tackle one single ABSA goal) or **Compound ABSA** tasks (more recent trends have moved towards developing methods that address two or more sentiment goals in a single method, thereby capturing the dependency between them).
# 
# * **aspect category** _c_ defines a unique aspect of an entity and is supposed to fall into a category set C, predefined for each specific domain of interest. For example, `food` and `service` can be aspect categories for the restaurant domain.
# * **aspect term** _a_ is the opinion target which explicitly appears in the given text, e.g., `“pizza”` in the sentence “The pizza is delicious.” When the target is implicitly expressed (e.g., “It is overpriced!”), we denote the aspect term as a special one named “null”.
# * **opinion term** _o_ is the expression given by the opinion holder to express his/her sentiment towards the target. For instance, `“delicious”` is the opinion term in the running example “The pizza is delicious”.
# * **sentiment polarity** _p_ describes the orientation of the sentiment over an aspect category or an aspect term, which usually includes `positive`, `negative`, and `neutral`.
# 
# <br>
# 
# ##### The Tasks of ABSA
# 
# <img src="https://github.com/Christina1281995/demo-repo/blob/main/absatasks.png?raw=true" width="50%" align="right">
# 
# Based on the comprehensive and recent overview provided by Zhang et al. (2022), several methods were systematically identified for further investigation:
# Potential tasks of interest:
# 
# - ATE (aspect term extraction)
# - ASC (aspect sentiment classification)
# - E2E (end 2 end, ATE + ASC)
# - ASTE (aspect sentiment triple extraction, ATE, OTE, ASC)
# 
# Either a pipeline method consisting of an ATE and ASC methods, or one of the compound methods may suffice for ABSA on tweet data.
# All 4 above mentioned potential tasks were investigated concerning their documented performance on well-known datasets (mostly SemEval 2014, 2015, 2016 but also Mitchell et al.'s 2013 twitter dataset). The top scoring methods for each task are selected for more in-depth inspection. Priority was also given to methods that scored particularly well on the twitter dataset.
# 
# 
# <img src="https://git.sbg.ac.at/geo-social-analytics/geo-social-media/sentiment-analyses/uploads/a23ed12a3d3860816556bcf159a7adf2/methods_of_general_interest.png" width="80%">
# 
# <br>
# <hr>
# <br>
# 
# 
# ##### Comparing published methodologies
# 
# <details><summary>ATE Methods</summary>
# 
# <img src="https://git.sbg.ac.at/geo-social-analytics/geo-social-media/sentiment-analyses/uploads/380415e631882f5cee2aba0c57eb4be3/image.png">
# </details>
# 
# <details><summary>ASC Methods</summary>
# 
# <img src="https://git.sbg.ac.at/geo-social-analytics/geo-social-media/sentiment-analyses/uploads/91e300023dfe90e3c37a1a5dbc20b65b/image.png">
# </details>
# 
# <details><summary>End 2 End Methods</summary>
# 
# <img src="https://git.sbg.ac.at/geo-social-analytics/geo-social-media/sentiment-analyses/uploads/5ab513146d2f580dc2dc5609c04e2950/image.png">
# </details>
# 
# <details><summary>ASTE Methods</summary>
# 
# <img src="https://git.sbg.ac.at/geo-social-analytics/geo-social-media/sentiment-analyses/uploads/20a20dc8168d41741d32912887476858/image.png">
# </details>
# 
# <br>
# <hr>
# 

# #### GRACE: Gradient Harmonized and Cascaded Labeling for Aspect-Based Sentiment Analysis
# 
# The method designed by Luo et al. (2020), described in the paper [GRACE: Gradient Harmonized and Cascaded Labeling for Aspect-based Sentiment Analysis. Huaishao Luo, Lei Ji, Tianrui Li, Nan Duan, Daxin Jiang. Findings of EMNLP, 2020.](https://arxiv.org/abs/2009.10557), implements a gradient harmonized and cascaded labeling model. 
# 
# The method falls into the "End 2 End" category of aspect-based sentiment analysis tasks, meaning it solves two ABSA sub-tasks, ATE (asect term extraction) and ASC (aspect semtiment classification), in one model or methodology. Recent advances in the E2E methods leverage the interdependencies between aspect term detection and its sentiment classification to enhance model performances. This stands in contrast to pipeline approaches, which tackle one ABSA sub-task after the other in an isolated manner. 
# 
# <img src='https://github.com/ArrowLuo/GRACE/raw/master/accessory/Framework.png'>
# 
# <br>
# <br>
# 
# ##### GRACE Model Architecture and Characteristics
# 
# <img src="https://github.com/Christina1281995/demo-repo/blob/main/joint.PNG?raw=true" align="right" width="40%">
# 
# - Co-extraction of ATE and ASC
# - 2 cascading modules
#     - 12 stacked transformer encoder blocks for ATE
#     - 3 shared transformer encoder blocks and 2 transformer decoder blocks for ASC
# - Focus on interaction
# - Joint approach
# - Shared shallow layers (n=3)
#     - higher layers in BERT are usually task-specific 
#     - it is assumed that can be useful to share the shallow layers 
#     - generates a shared "baseline understanding"
# <img src="https://github.com/Christina1281995/demo-repo/blob/main/grad.PNG?raw=true" align="right" width="40%">
# - **Virtual adversairal training**: the robustness of the model is improved bz preturbing the input data in small ways so that its difficult for the model to classify (to implement this, the direction and distance of the perturbations is calculated)
# - **Gradient harmonized loss**: the model is trained with cross entropy loss, but to optimise the model to "focus" more on the "hard" labels, a gradient norm is calculated for each label (where "easy" labels have low gradients) and a weight for the loss calculation is assigned to each label based on the gradient density (histogram statistic). The idea is to decrease the weight of loss form labels with low gradient norms.
# 
# Architecture: 
# 
# - **Activation** function: GeLU (Gaussian Error Linear Unit, non-linear function that maps negative input values to negative outputs and positive input values to positive outputs)
# - Initial **tokenization and embeddings** (WordPiece, a subword tokenization method used for the original BERT model)
#     - A nn.Embeddings layer combines word embeddings, positional embeddings and token type embeddings (n=2)
# - n x the **encoder block** (12 in this configuration, same as original BERT model)
#     - Multi-head Scaled-dot product attention with Softmax to generate context layer
#     - 'Intermediate': linear layer and activation function
#     - 'Output': liner layer, layer normalisation, dropout
# - The **classification head** for ATE (nn.Linear, Softmax)
# - n x the **decoder block** (2 in this configuration)
# - The **classification head** for ASC (nn.Linear, Softmax)
# 

# In this collapsed cell are a few key building blocks in coded implementation:

# In[1]:


# higher-level contents of one encoder
class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)  # multi-head scaled-dot prodcut self-attention
        self.intermediate = BertIntermediate(config)  # feed forward linear and normalisation
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output



# An encoder block (x12 in GRACE)
class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)]) # 12 

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers



 # pooler layer for summarization based on [CLS] token
class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



# model class using pretrained configurations
class BertModel(PreTrainedBertModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").
    Params:
        config: a BertConfig class instance with the configuration to build a new model
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```



# BertModel then again is used to compose a whole classification task workflow, e.g.:
class BertForSequenceClassification(PreTrainedBertModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)


# complete asc ate workflow initialisation:
class BertForSequenceLabeling(PreTrainedBertModel):
    #
    def __init__(self, config, num_tp_labels, task_config):
        super(BertForSequenceLabeling, self).__init__(config)
        self.num_tp_labels = num_tp_labels
        self.task_config = task_config

        at_label_list = self.task_config["at_labels"]
        self.at_label_map = {i: label for i, label in enumerate(at_label_list)}
        assert len(at_label_list) == 3, "Hard code works when doing BIO strategy, " \
                                        "due to the middle step to generate span boundary."
        self.I_AP_INDEX = 2     # Note: This operation works when doing BIO strategy
        assert self.at_label_map[self.I_AP_INDEX] == "I-AP", "A hard code need the index below."

        self.num_encoder_labels = self.num_tp_labels[0]
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_encoder_labels)

        if self.task_config["use_ghl"]:
            self.weighted_ce_loss_fct = WeightedCrossEntropy(ignore_index=-1)
        else:
            self.ce_loss_fct = CrossEntropyLoss(ignore_index=-1)

        ## Gradient balance <---
        self.bins = 24
        self.momentum = 0.75
        self.edges = torch.arange(self.bins + 1).float() / self.bins
        self.edges[-1] += 1e-6
        self.acc_sum = torch.zeros(self.bins, dtype=torch.float)

        self.decoder_bins = 24
        self.decoder_momentum = 0.75
        self.decoder_edges = torch.arange(self.bins + 1).float() / self.bins
        self.decoder_edges[-1] += 1e-6
        self.decoder_acc_sum = torch.zeros(self.bins, dtype=torch.float)
        self.decoder_weight_gradient = None
        self.decoder_weight_gradient_labels = None
        ## --->

        self.use_vat = self.task_config["use_vat"]
        if self.use_vat:
            self.alpha = 1.
            self.xi = 1e-6
            self.epsilon = 2.
            self.ip = 1

        self.num_decoder_labels = self.num_tp_labels[1]
        if config.hidden_size == 768:
            decoder_config, _ = PreTrainedDecoderBertModel.get_config("decoder-bert-base")
        else:
            raise ValueError("No implementation on such a decoder config.")

        self.decoder_shared_layer = self.task_config["decoder_shared_layer"]
        decoder_config.decoder_vocab_size = self.num_encoder_labels
        decoder_config.num_decoder_layers = self.task_config["num_decoder_layer"]
        bert_position_embeddings_weight = self.bert.embeddings.position_embeddings.weight

        # NOTE: DecoderBertModel is adapted from the Transformer decoder.
        # It is not a decoder used as generation task. It is used as labeling task here.
        self.decoder = DecoderBertModel(decoder_config, bert_position_embeddings_weight)
        self.decoder_classifier = nn.Linear(config.hidden_size, self.num_decoder_labels)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act] # nn.Tanh()

        self.apply(self.init_bert_weights)

    # Virtual Adversarial Training Implementation
    def vat_loss(self, input_ids, token_type_ids, attention_mask):
        # LDS should be calculated before the forward for cross entropy
        with torch.no_grad():
            _pred_logits, _, _ = self.get_encoder_logits(input_ids, token_type_ids, attention_mask)
            pred = F.softmax(_pred_logits, dim=2)

        # prepare random unit tensor
        batch_size_, seq_length_ = input_ids.size()
        hidden_size_ = self.bert.config.hidden_size
        d = torch.randn(batch_size_, seq_length_, hidden_size_, device=input_ids.device)

        with _disable_tracking_bn_stats(self):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                xi_d = self.xi * _l2_normalize_foremd(_mask_by_length(d, attention_mask))
                xi_d.retain_grad()
                words_embeddings_ = self.bert.embeddings.word_embeddings(input_ids)
                pred_hat, _, _ = self.get_encoder_logits(words_embeddings_ + xi_d, token_type_ids, attention_mask,
                                                   bool_input_embedding=True)
                logp_hat_i = F.log_softmax(pred_hat, dim=2).view(-1, self.num_encoder_labels)
                pred_i = pred.view(-1, self.num_encoder_labels)
                adv_distance = F.kl_div(logp_hat_i, pred_i, reduction='batchmean')
                adv_distance.backward()
                d = xi_d.grad
                self.zero_grad()

            # calc LDS
            r_adv = _l2_normalize_foremd(d.detach()) * self.epsilon
            words_embeddings_ = self.bert.embeddings.word_embeddings(input_ids)

            pred_hat, _, _ = self.get_encoder_logits(words_embeddings_+r_adv, token_type_ids, attention_mask,
                                               bool_input_embedding=True)
            logp_hat_i = F.log_softmax(pred_hat, dim=2).view(-1, self.num_encoder_labels)
            pred_i = pred.view(-1, self.num_encoder_labels)
            lds = F.kl_div(logp_hat_i, pred_i, reduction='batchmean')
        return lds

    # Gradient Harmonized Loss Implementation
    def calculate_ce_gradient_weight(self, logits, labels, attention_mask, num_labels,
                         acc_sum, bins, momentum, edges, weight_gradient=None, weight_gradient_labels=None):
        device = logits.device
        batch_size, sequence_length = labels.size()
        # Here using crf_label_ids for CE labels have -1 value.
        labels_onehot = torch.zeros(batch_size, sequence_length, num_labels, dtype=torch.float, device=device)
        crf_label_ids = labels.clone()
        crf_label_ids[crf_label_ids < 0] = 0.
        labels_onehot.scatter_(2, crf_label_ids.unsqueeze(2), 1)
        # gradient length
        gradient = torch.abs(F.softmax(logits.detach(), dim=-1) - labels_onehot)

        weights, acc_sum, weight_gradient, weight_gradient_labels \
            = self.statistic_weight(gradient, logits, labels, attention_mask, num_labels,
                                    acc_sum, bins, momentum, edges, weight_gradient, weight_gradient_labels)

        return weights, acc_sum, weight_gradient, weight_gradient_labels

    def statistic_weight(self, gradient, logits, labels, attention_mask, num_labels,
                         acc_sum, bins, momentum, edges,
                         weight_gradient=None, weight_gradient_labels=None):
        device = logits.device
        batch_size, sequence_length = labels.size()

        if weight_gradient is None:
            weight_gradient = torch.zeros(self.bins).to(device)
        if weight_gradient_labels is None:
            weight_gradient_labels = torch.zeros(self.bins, num_labels).to(device)

        edges = self.edges.to(device)
        momentum = self.momentum
        weights = torch.ones_like(logits)

        valid_instance = attention_mask.unsqueeze(-1).expand(batch_size, sequence_length, num_labels)
        valid_instance = valid_instance > 0
        total_valid = max(valid_instance.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (gradient >= edges[i]) & (gradient < edges[i + 1]) & valid_instance

            num_in_bin_label = inds.sum(0).sum(0).to(dtype=weight_gradient_labels.dtype)
            weight_gradient_labels[i, :] = weight_gradient_labels[i, :] + num_in_bin_label

            num_in_bin = inds.sum().item()

            weight_gradient[i] = weight_gradient[i] + num_in_bin

            if num_in_bin > 0:
                if momentum > 0:
                    index_tensor = torch.tensor(i)
                    val_ = torch.gather(acc_sum, dim=0, index=index_tensor)
                    momentum_bins = momentum * float(val_.item()) + (1 - momentum) * num_in_bin
                    weights[inds] = total_valid / momentum_bins
                    acc_sum.scatter_(0, index_tensor, momentum_bins)
                else:
                    weights[inds] = total_valid / num_in_bin
                n += 1

        return weights, acc_sum, weight_gradient, weight_gradient_labels


# <img src="https://raw.githubusercontent.com/nlp-with-transformers/notebooks/e3850199388f4983cc9799135977f0a6b06d5a79//images/chapter03_transformer-encoder-decoder.png">

# In replicating the training described in the paper using a twitter training dataset (`twt1`), an F1 score of 0.7514 was achieved for aspect term extraction. However, for the aspect sentiment classification, an F1 score of only 0.5694 was achieved. 
# 
# Future work may investigate whether higher results may be achieved for the GRACE ASC sub-task to fully leverage an E2E model for twitter ABSA. 

# ### Imports

# In[1]:


import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.append('../GRACE/')


# In[2]:


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


# ##### GRACE Model Setup (Loading from last training Step and Epoch)
# 
# log messages from last training step and epoch: <br>
# 
# ```
# 
# Model saved to out_twt1_ateacs/pytorch_model.bin.9
# 
# AT p:0.7365 	r:0.7670	f1:0.7514 
# 
# AS p:0.5581 	r:0.5811	f1:0.5694 
# 
# 
# ```
# 

# In[12]:


# args set as per instructions by authors
args = argparse.Namespace(

    ## Required parameters
    data_dir='../GRACE/data/', 
    bert_model='bert-base-uncased',
    init_model=None,
    task_name="ate_asc",
    data_name="twt1",
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
    train_batch_size=32, 
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


# In[4]:


random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# In[5]:


# manually setting device and gpu parameters
device = 'cpu'
n_gpu = 0
data_name = args.data_name.lower()


# In[6]:


task_name = args.task_name.lower()

task_config = {
    "use_ghl": True,
    "use_vat": False,
    "num_decoder_layer": 2,
    "decoder_shared_layer": 3,
}


# In[13]:


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


# In[8]:


# load bert tokenizer (bert-base-uncased)
tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)


# In[14]:


DATASET_DICT={}
DATASET_DICT["lap"] = {"train_file":"laptops_2014_train.txt", "valid_file":"laptops_2014_trial.txt", "test_file":"laptops_2014_test.gold.txt"}
DATASET_DICT["res"] = {"train_file":"restaurants_union_train.txt", "valid_file":"restaurants_union_trial.txt", "test_file":"restaurants_union_test.gold.txt"}
for i in ["2014", "2015", "2016"]:
    DATASET_DICT["res{}".format(i)] = {"train_file": "restaurants_{}_train.txt".format(i), "valid_file": "restaurants_{}_trial.txt".format(i), "test_file": "restaurants_{}_test.gold.txt".format(i)}
for i in range(10):
    DATASET_DICT["twt{}".format(i+1)] = {"train_file":"twitter_{}_train.txt".format(i+1), "valid_file":"twitter_{}_test.gold.txt".format(i+1), "test_file":"twitter_{}_test.gold.txt".format(i+1)}


# In[15]:


if data_name in DATASET_DICT:
    args.train_file = DATASET_DICT[data_name]["train_file"]
    args.valid_file = DATASET_DICT[data_name]["valid_file"]
    args.test_file = DATASET_DICT[data_name]["test_file"]
else:
    assert args.train_file is not None
    assert args.valid_file is not None
    assert args.test_file is not None


# In[16]:


file_path = os.path.join(args.data_dir, args.train_file)
print(file_path)


# In[17]:


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


# In[18]:


at_label_list, as_label_list = label_tp_list
at_label_map = {i: label for i, label in enumerate(at_label_list)}
as_label_map = {i: label for i, label in enumerate(as_label_list)}

print(at_label_map)
print(as_label_map)


# In[19]:


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


# In[26]:


# set model file to the last saved model after training epochs completed

# CODE CHANGE NOTE: In this implementation (using a docker container for jupyter lab) the download of bert-base-uncased.tar.gz 
#into cache terminates before the whole file is successfully loaded. 
# Therefore an adapted ate_asc_modeling_local_bert_file.py is imported here which loads the model from a folder in the repo ('bert-base-uncased/bert-base-uncased.tar.gz')

model_file = '../GRACE/out_twt1_ateacs/pytorch_model.bin.9'
model = load_model(model_file, args, num_tp_labels, task_config, device)


# In[27]:


if hasattr(model, 'module'):
    print('has module')
    model = model.module
    
# print(model)


# The model summary:

# In[28]:


# set model to eval mode (turn off training features e.g. dropout)
model.eval()


# ### Testing Block (can be skipped)
# 
# This code block serves to ensure the model loaded correctly.
# Uses just the last entry in twitter_1_train.txt

# In[ ]:


DATALOADER_DICT = {}


# In[ ]:


DATALOADER_DICT["ate_asc"] = {"eval": dataloader_val}


# In[ ]:


if task_name not in DATALOADER_DICT:
    raise ValueError("Task not found: %s" % (task_name))


# In[ ]:


eval_dataloader, eval_examples = DATALOADER_DICT[task_name]["eval"](args, tokenizer, file_path, label_tp_list=label_tp_list, set_type="val")


# In[ ]:


for input_ids, input_mask, segment_ids, at_label_ids, as_label_ids, label_mask, label_mask_X in eval_dataloader:
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    at_label_ids = at_label_ids.to(device)
    as_label_ids = as_label_ids.to(device)
    label_mask = label_mask.to(device)
    label_mask_X = label_mask_X.to(device)


# In[ ]:


with torch.no_grad():
    # logits, decoder_logits = model(input_ids, segment_ids, input_mask)
    logits, sequence_output, encoder_output = model.get_encoder_logits(input_ids, segment_ids, input_mask)
    pred_dec_ids = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
    decoder_logits = model.get_decoder_logits(encoder_output, input_mask, label_mask_X, pred_dec_ids)
    logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
    decoder_logits = torch.argmax(F.log_softmax(decoder_logits, dim=2), dim=2)
    logits = logits.detach().cpu().numpy()
    decoder_logits = decoder_logits.detach().cpu().numpy()


# In[ ]:


at_label_ids = at_label_ids.to('cpu').numpy()
as_label_ids = as_label_ids.to('cpu').numpy()
label_mask = label_mask.to('cpu').numpy()


# In[ ]:


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


# ##### Apply Model on Twemlab Goldstandard Data

# ##### Load Dataset

# In[29]:


# Load TwEmLab Goldstandard
tree1 = ET.parse('../Data/twemlab_goldstandards_original/birmingham_labels.xml')
root1 = tree1.getroot()

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


# #### Re-Format Text to match GRACE Input

# In[30]:


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

# In[32]:


with open("../Data/twemlab_goldstandards_original/original_reformatted_with_0s/twemlab_birmingham_formatted.txt", mode = "w") as f:
    for tweet in convert_to_doc:
        for word in tweet:
            f.write("%s\n" % word)
        f.write("\n")


# #### Run GRACE Model on new .txt File

# In[33]:


#file_path = os.path.join(args.data_dir, args.train_file)
file_path = "../Data/twemlab_goldstandards_original/original_reformatted_with_0s/twemlab_birmingham_formatted.txt"
print(file_path)


# In[34]:


DATALOADER_DICT = {}
# only "eval" state is needed ("train" is left out)
DATALOADER_DICT["ate_asc"] = {"eval":dataloader_val}


# In[35]:


eval_dataloader, eval_examples = DATALOADER_DICT[task_name]["eval"](args, tokenizer, file_path, label_tp_list=label_tp_list, set_type="val")


# In[36]:


# empty lists for outputs
pred_aspect_terms = []
pred_aspect_sentiments = []

# eval_dataloader contains the pre-processed, tokenized inputs (input has been reformatted for GRACE beforehand)
for input_ids, input_mask, segment_ids, at_label_ids, as_label_ids, label_mask, label_mask_X in eval_dataloader:
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    at_label_ids = at_label_ids.to(device)
    as_label_ids = as_label_ids.to(device)
    label_mask = label_mask.to(device)
    label_mask_X = label_mask_X.to(device)
    
    # get predictions with argmax log-softmax probabilities
    # predicted aspect term ids from logits (encoder)
    # predicted asepct sentiment ids from decoder logits
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
                # no at_label_ids or as_label_ids because there is no ground truth data in this dataset
                #temp_11.append(at_label_map[at_label_ids[i][j]])
                temp_12.append(at_label_map[logits[i][j]])
                #temp_21.append(as_label_map[as_label_ids[i][j]])
                temp_22.append(as_label_map[decoder_logits[i][j]])
                
        pred_aspect_terms.append(temp_12)
        pred_aspect_sentiments.append(temp_22)

# add new aspect term labels and aspect sentiment labels as columns to twemlab dataframe
twemlab['aspect_term_preds'] = pred_aspect_terms
twemlab['aspect_senti_preds'] = pred_aspect_sentiments


# Extract Aspect Terms and Sentiment Aspects

# In[37]:


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


# In[58]:


demo_cols = twemlab[['text','label', 'sentiment_label','aspect_terms', 'aspect_sentiments']]
col_names = {'text': 'Text', 'label': 'Label', 'sentimemt_label': 'Sentiment', 'aspect_terms': 'Aspect Terms', 'aspect_sentiments': 'Aspect Sentiments'}
demo_cols = demo_cols.rename(columns=col_names)
demo_cols.sample(5).style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])])


# In[62]:


index = 786

print(f"An Example of how the GRACE results look\n\nText:\t{twemlab.iloc[index]['text']}\n\nAsp.Terms:\t{twemlab.iloc[index]['aspect_term_preds']}\nAsp.Sents:\t{twemlab.iloc[index]['aspect_senti_preds']}\n\nATE:\t{twemlab.iloc[index]['aspect_terms']}\nASC:\t{twemlab.iloc[index]['aspect_sentiments']}\n\nGoldstandard: {twemlab.iloc[index]['sentiment_label']}")


# To get an indication of how well the model is classifying the aspect terms and their sentiments, a make-shift comparison to the goldstandard labels can me made. Note that the goldstandard annotations differ from the GRACE model output:
# - The goldstandard was labelled with emotions, whereas GRACE now identifies sentments (pos, neu, neg)
# - The goldstandard was labelled per document, i.e. the entire tweet, whereas GRACE breaks down the document into the token level and can identify several aspects within one tweet, each potentially with differnt sentiments

# In[63]:


# compare the aspect sentiments with the twemlab sentiments

match = 0          # sentiment tokens uniformly match goldstandard sentiment (incl. no-list on sentiment type "none")
nomatch = 0        # sentiment tokens are uniformly different to goldstandard sentiment
not_uniform = 0    # sentiment tokens are not uniform
n_a = 0            # empty sentiment tokens list (that does not match with sentiment label 'none')
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
            print("issue")
    
    # empty list
    else:
        if row['sentiment_label'] == 0: match += 1
        else: n_a += 1

print("Comparing the aspect sentiment classifcation with the overall sentiment in the twemlab goldstandard:\n")
print(f"Matches:      {match}  ({np.round((match/counter)*100, 1)}%)\nNo Matches:   {nomatch}  ({np.round((nomatch/counter)*100, 1)}%)\nNot Uniform:   {not_uniform}   ({np.round((not_uniform/counter)*100, 1)}%)\nN/A:          {n_a}  ({np.round((n_a/counter)*100, 1)}%)")


# ##### Apply Model on AIFER Twitter Dataset

# #### Load Dataset

# In[44]:


# COVID dataset
dataset = pd.read_csv("../Data/Disaster_responses/ahrtal_tweets.csv", sep="\t")

# MONKEYPOX dataset
# dataset = pd.read_csv("https://git.sbg.ac.at/s1080384/sentimentanalysis/-/raw/main/data/100k_monkeypox.csv")

# filter for English tweets
lang = ['en']
en_dataset = dataset[dataset['tweet_lang'].isin(lang)]

# exclude columns that aren't needed (for now)
data = en_dataset[['date', 'text', 'geom']]

# date handling (for animated map)
#data['year'] = [pd.to_datetime(x).year for x in data['date']]
#data['month'] = [pd.to_datetime(x).to_period('M') for x in data['date']]
#data['week'] = [pd.to_datetime(x).to_period('W') for x in data['date']]
#data['day'] = [pd.to_datetime(x).to_period('D') for x in data['date']]
# data['month'] = data.apply(lambda row: pd.to_datetime(row["date"]).to_period('M'), axis=1)

# get subset to speed up demos and testing
data_subset = data.sample(1000)
data_subset.head(10)


# #### Re-format Text to match GRACE Input

# In[45]:


# text column to list
text_list = list(data_subset['text'])

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

# In[46]:


with open("../Data/Disaster_responses/aifer_reformatted.txt", mode = "w") as f:
    for tweet in convert_to_doc:
        for word in tweet:
            f.write("%s\n" % word)
        f.write("\n")


# #### Run GRACE Model on new .txt File

# In[47]:


#file_path = os.path.join(args.data_dir, args.train_file)
file_path = '../Data/Disaster_responses/aifer_reformatted.txt'
print(file_path)


# In[48]:


DATALOADER_DICT = {}
# only "eval" state is needed ("train" is left out)
DATALOADER_DICT["ate_asc"] = {"eval":dataloader_val}


# In[49]:


eval_dataloader, eval_examples = DATALOADER_DICT[task_name]["eval"](args, tokenizer, file_path, label_tp_list=label_tp_list, set_type="val")


# In[51]:


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
data_subset['aspect_term_preds'] = pred_aspect_terms
data_subset['aspect_senti_preds'] = pred_aspect_sentiments


# A sample of the added columns:

# In[67]:


# extract aspect term words and store in the dataframe
aspect_terms = []

# for every row (tweet)
for idx, row in data_subset.iterrows():
    row_aspect_terms = []
    words = row['text'].split()
    count_words = len(words)
    count_terms = len(row['aspect_term_preds'])
    
    # check if words in text equals labels in aspect term prediction column
    if count_words == count_terms:
        
        # for every word in tweet
        for i in range(count_words):
            if row['aspect_term_preds'][i] == 'B-AP':
                term = words[i]

                # if aspect term present, check further connected aspect terms
                still_going = True
                for j in range(i+1, count_words):
                    if still_going == True:
                        if row['aspect_term_preds'][j] == 'I-AP':
                            term = term + ' ' + words[j]
                        else:
                            still_going = False
                    else:
                        break

                row_aspect_terms.append(term)

        aspect_terms.append(row_aspect_terms)
    
    # if the count of words and aspect term labels don't match add a dash to list
    # in the future maybe deal with this better --> look into it more
    else:
        aspect_terms.append('-')


data_subset['aspect_terms'] = aspect_terms

demo_cols = data_subset[['text','aspect_terms', 'aspect_senti_preds']]
col_names = {'text': 'Text', 'aspect_terms': 'Aspect Terms', 'aspect_senti_preds': 'Aspect Sentiments'}
demo_cols = demo_cols.rename(columns=col_names)
demo_cols.sample(5).style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])])


# In[65]:


# extract aspect term and sentiment and store in twemlab dataframe
aspect_terms = []
aspect_sentiments = []

# for every row (tweet)
for idx, row in data_subset.iterrows():
    
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
            
data_subset['aspect_terms'] = aspect_terms
data_subset['aspect_sentiments'] = aspect_sentiments

demo_cols = data_subset[['text','label', 'sentiment_label','aspect_terms', 'aspect_sentiments']]
col_names = {'text': 'Text', 'label': 'Label', 'sentimemt_label': 'Sentiment', 'aspect_terms': 'Aspect Terms', 'aspect_sentiments': 'Aspect Sentiments'}
demo_cols = demo_cols.rename(columns=col_names)
demo_cols.sample(5).style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])])


# In[68]:


# example of non-matching text and aspect term labels
indices = [index for index, _val in enumerate(data_subset)]

if 74587 in indices:
    text = data_subset.loc[74587,'text']
    print(text)
    print(len(text.split()))

    terms = data_subset.loc[74587,'aspect_term_preds']
    print(terms)
    print(len(terms))
    
else:
    print("index 74587 isn't in this data subset")

