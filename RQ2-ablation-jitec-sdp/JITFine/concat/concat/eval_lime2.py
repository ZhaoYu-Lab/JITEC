# import pickle
from __future__ import absolute_import, division, print_function
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
import numpy as np
import pandas as pd
import time, pickle, math, warnings, os, operator
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')
from JITFine.concat.model import Model
import torch
from lime.lime_tabular import LimeTabularExplainer
import dill
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel)
import matplotlib.pyplot as plt
import argparse

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel)
from tqdm import tqdm, trange
import multiprocessing
from imblearn.over_sampling import SMOTE,KMeansSMOTE,SMOTENC,SVMSMOTE,ADASYN,RandomOverSampler
from JITFine.concat.model import Model
from JITFine.my_util import convert_examples_to_features, TextDataset, eval_result, preprocess_code_line \
    , create_path_if_not_exist,convert_dtype_dataframe
from scipy.optimize import differential_evolution
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, precision_recall_fscore_support, classification_report, auc
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)



line_score_df_col_name = ['commit_id', 'total_tokens', 'line_level_label']

# since we don't want to use commit metrics in LIME
commit_metrics = ['la', 'la', 'ld', 'ld', 'nf', 'nd_y', 'nd', 'ns', 'ent', 'ent', 'nrev', 'rtime', 'hcmt', 'self',
                  'ndev',
                  'age', 'age', 'nuc', 'app_y', 'aexp', 'rexp', 'arexp', 'rrexp', 'asexp', 'rsexp', 'asawr', 'rsawr']

manual_features_columns = ['la', 'ld', 'nf', 'ns', 'nd', 'entropy', 'ndev',
                           'lt', 'nuc', 'age', 'exp', 'rexp', 'sexp', 'fix']


data_path = "data/jitfine"




def get_combined_features(code_commit, commit_id, label, metrics_df, count_vect, mode = 'train'):
    
    if mode not in ['train','test']:
        print('wrong mode')
        return
    
    code_df = pd.DataFrame()
    code_df['commit_id'] = commit_id
    code_df['code'] = code_commit
    code_df['label'] = label
    
    code_df = code_df.sort_values(by='commit_id')
    
    metrics_df = metrics_df.sort_values(by='commit_id')
    
    code_change_arr = count_vect.transform(code_df['code']).astype(np.int16).toarray()
    
    if mode == 'train':
        metrics_df = metrics_df.drop('commit_id',axis=1)
        metrics_df_arr = metrics_df.to_numpy(dtype=np.float32)
        final_features = np.concatenate((code_change_arr,metrics_df_arr),axis=1)
        col_names = list(count_vect.get_feature_names())+list(metrics_df.columns)
        return final_features, col_names, list(code_df['label'])
    elif mode == 'test':
        code_features = pd.DataFrame(code_change_arr, columns=count_vect.get_feature_names())
        code_features['commit_id'] = list(code_df['commit_id'])

        metrics_df = metrics_df.set_index('commit_id')
        code_features = code_features.set_index('commit_id')
        final_features = pd.concat([code_features, metrics_df],axis=1)
        
        return final_features, list(code_df['commit_id']), list(code_df['label'])





def get_LIME_explainer(proj_name, train_feature, feature_names):
    LIME_explainer_path = './final_model/'+proj_name+'_LIME_RF_DE_SMOTE_min_df_3.pkl'
    class_names = ['not defective', 'defective'] # this is fine...
    if not os.path.exists(LIME_explainer_path):
        start = time.time()
        # get features in train_df here
        print('start training LIME explainer')

        explainer = LimeTabularExplainer(train_feature, 
                                         feature_names=feature_names, 
                                         class_names=class_names, discretize_continuous=False, random_state=42)
        dill.dump(explainer, open(LIME_explainer_path, 'wb'))
        print('finish training LIME explainer in',time.time()-start, 'secs')

    else:
        explainer = dill.load(open(LIME_explainer_path, 'rb'))
    
    return explainer

def eval_with_LIME(proj_name, clf, explainer, test_features):
    
    def preprocess_feature_from_explainer(exp):
        features_val = exp.as_list(label=0)
        new_features_val = [tup for tup in features_val if float(tup[1]) > 0] # only score > 0 that indicates buggy token

        feature_dict = {re.sub('\s.*','',val[0]):val[1] for val in new_features_val}

        sorted_feature_dict = sorted(feature_dict.items(), key=operator.itemgetter(1), reverse=True)

        sorted_feature_dict = {tup[0]:tup[1] for tup in sorted_feature_dict if tup[0] not in commit_metrics}
        tokens_list = list(sorted_feature_dict.keys())

        return sorted_feature_dict, tokens_list
    
    def add_agg_scr_to_list(line_stuff, scr_list):
        if len(scr_list) < 1:
            scr_list.append(0)

        line_stuff.append(np.mean(scr_list))
        line_stuff.append(np.median(scr_list))
        line_stuff.append(np.sum(scr_list))

    all_buggy_line_result_df = []  
    
    prediction_result = pd.read_csv(data_path+proj_name+'_RF_DE_SMOTE_min_df_3_prediction_result.csv')
    line_level_df = pd.read_csv(data_path+proj_name+'_complete_buggy_line_level.csv',sep='\t').dropna()

    correctly_predicted_commit = list(prediction_result[(prediction_result['pred']==1) &
                                                (prediction_result['actual']==1)]['test_commit'])

    for commit in correctly_predicted_commit:
        code_change_from_line_level_df = list(line_level_df[line_level_df['commit_hash']==commit]['code_change_remove_common_tokens'])
        line_level_label = list(line_level_df[line_level_df['commit_hash']==commit]['is_buggy_line'])

        line_score_df = pd.DataFrame(columns = line_score_df_col_name)
        line_score_df['line_num'] = np.arange(0,len(code_change_from_line_level_df))
        line_score_df = line_score_df.set_index('line_num')

        exp = explainer.explain_instance(test_features.loc[commit], clf.predict_proba, 
                                         num_features=len(test_features.columns), top_labels=1,
                                         num_samples=5000)

        sorted_feature_score_dict, tokens_list = preprocess_feature_from_explainer(exp)

        for line_num, line in enumerate(code_change_from_line_level_df): # for each line (sadly this loop is needed...)
            line_stuff = []
            line_score_list = np.zeros(100) # this is needed to store result in dataframe
            token_list = line.split()[:100]
            line_stuff.append(line)
            line_stuff.append(len(token_list))

            for tok_idx, tok in enumerate(token_list):
                score = sorted_feature_score_dict.get(tok,0)
                line_score_list[tok_idx] = score
                
            # calculate top-k tokens first then followed by all tokens

            line_stuff = line_stuff + list(line_score_list)

            for k in top_k_tokens: # for each k in top-k tokens
                top_tokens = tokens_list[0:k-1]
                top_k_scr_list = []

                if len(token_list) < 1:
                    top_k_scr_list.append(0)
                else:
                    for tok in token_list:
                        score = 0
                        if tok in top_tokens:
                            score = sorted_feature_score_dict.get(tok,0)
                        top_k_scr_list.append(score)

                add_agg_scr_to_list(line_stuff, top_k_scr_list)

            add_agg_scr_to_list(line_stuff, list(line_score_list[:len(token_list)]))
            line_score_df.loc[line_num] = line_stuff
        
        line_score_df['commit_id'] = [commit]*len(line_level_label)
        line_score_df['line_level_label'] = line_level_label

        all_buggy_line_result_df.append(line_score_df)
        
        del exp, sorted_feature_score_dict, tokens_list, line_score_df
        
    return all_buggy_line_result_df

def eval_line_level(proj_name, best_k_neighbor):
    # load model here
    clf = pickle.load(open(model_path+proj_name+'_RF_DE_SMOTE_min_df_3.pkl','rb'))

    train_code, train_commit, train_label = prepare_data(proj_name, mode='train',
                                                                  remove_python_common_tokens=remove_python_common_tokens)
    test_code, test_commit, test_label = prepare_data(proj_name, mode='test',
                                                              remove_python_common_tokens=remove_python_common_tokens)

    commit_metrics = load_change_metrics_df(proj_name)
    #     print(commit_metrics.head())
    train_commit_metrics = commit_metrics[commit_metrics['commit_id'].isin(train_commit)]
    test_commit_metrics = commit_metrics[commit_metrics['commit_id'].isin(test_commit)]

    count_vect = CountVectorizer(min_df=3, ngram_range=(1,1))
    count_vect.fit(train_code)

    # use train_feature to train LIME

    train_feature, col_names, new_train_label = get_combined_features(train_code, train_commit, train_label, train_commit_metrics,count_vect)
    test_feature, test_commit_id, new_test_label = get_combined_features(test_code, test_commit, test_label, test_commit_metrics,count_vect, mode = 'test')

    percent_80 = int(len(new_train_label)*0.8)
    
    final_train_feature = train_feature[:percent_80]
    final_new_train_label = new_train_label[:percent_80]
    
    print('load data of',proj_name, 'finish') # at least we can load dataframe...                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    
    smote = SMOTE(k_neighbors = best_k_neighbor, random_state=42, n_jobs=-1)

    train_feature_res, new_train_label_res = smote.fit_resample(final_train_feature, final_new_train_label)
    
    print('resample data complete')

    explainer = get_LIME_explainer(proj_name, train_feature_res, col_names)
    print('load LIME explainer complete')
    
    # to save RAM to prevent out of memory error
    del smote, train_feature_res, new_train_label_res, train_code, train_commit, train_label, test_code, test_commit, test_label
    del commit_metrics, train_commit_metrics, test_commit_metrics, count_vect, final_train_feature, final_new_train_label
    
    line_level_result = eval_with_LIME(proj_name, clf, explainer, test_feature)
    
    print('eval line level finish')
    return line_level_result

def create_tmp_df(all_commits,agg_methods):
    df = pd.DataFrame(columns = ['commit_id']+agg_methods)
    df['commit_id'] = all_commits
    df = df.set_index('commit_id')
    return df

def get_line_level_metrics(line_score,label):
    scaler = MinMaxScaler()
    line_score = scaler.fit_transform(np.array(line_score).reshape(-1, 1)) # cannot pass line_score as list T-T
    pred = np.round(line_score)
    
    line_df = pd.DataFrame()
    line_df['scr'] = [float(val) for val in list(line_score)]
    line_df['label'] = label
    line_df = line_df.sort_values(by='scr',ascending=False)
    line_df['row'] = np.arange(1, len(line_df)+1)

    real_buggy_lines = line_df[line_df['label'] == 1]
    
    top_10_acc = 0
    
    if len(real_buggy_lines) < 1:
        IFA = len(line_df)
        top_20_percent_LOC_recall = 0
        effort_at_20_percent_LOC_recall = math.ceil(0.2*len(line_df))
        
    else:
        IFA = line_df[line_df['label'] == 1].iloc[0]['row']-1
        label_list = list(line_df['label'])

        all_rows = len(label_list)
        
        # find top-10 accuracy
        if all_rows < 10:
            top_10_acc = np.sum(label_list[:all_rows])/len(label_list[:all_rows])
        else:
            top_10_acc = np.sum(label_list[:10])/len(label_list[:10])

        # find recall
        LOC_20_percent = line_df.head(int(0.2*len(line_df)))
        buggy_line_num = LOC_20_percent[LOC_20_percent['label'] == 1]
        top_20_percent_LOC_recall = float(len(buggy_line_num))/float(len(real_buggy_lines))

        # find effort @20% LOC recall

        buggy_20_percent = real_buggy_lines.head(math.ceil(0.2 * len(real_buggy_lines)))
        buggy_20_percent_row_num = buggy_20_percent.iloc[-1]['row']
        effort_at_20_percent_LOC_recall = int(buggy_20_percent_row_num) / float(len(line_df))

    return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc

def eval_line_level_at_commit(cur_proj):
    RF_result = pd.read_csv(data_path+cur_proj+'_line_level_result_min_df_3_300_trees.csv')
    RF_result = RF_result[line_score_df_col_name]

    all_commits = list(RF_result['commit_id'].unique())

    IFA_df = create_tmp_df(all_commits, score_cols)
    recall_20_percent_effort_df = create_tmp_df(all_commits, score_cols) 
    effort_20_percent_recall_df = create_tmp_df(all_commits, score_cols)
    precision_df = create_tmp_df(all_commits, score_cols)
    recall_df = create_tmp_df(all_commits, score_cols)
    f1_df = create_tmp_df(all_commits, score_cols)
    AUC_df = create_tmp_df(all_commits, score_cols)
    top_10_acc_df = create_tmp_df(all_commits, score_cols)
    MCC_df = create_tmp_df(all_commits, score_cols)
    bal_ACC_df = create_tmp_df(all_commits, score_cols)

    for commit in all_commits:
        IFA_list = []
        recall_20_percent_effort_list = []
        effort_20_percent_recall_list = []
        top_10_acc_list = []

        cur_RF_result = RF_result[RF_result['commit_id']==commit]
    
        to_save_df = cur_RF_result[['commit_id',  'total_tokens',  'line_level_label',  'sum-all-tokens']]
        
        scaler = MinMaxScaler()
        line_score = scaler.fit_transform(np.array(to_save_df['sum-all-tokens']).reshape(-1, 1))
        to_save_df['line_score'] = line_score.reshape(-1,1) # to remove [...] in numpy array
        to_save_df = to_save_df.drop(['sum-all-tokens','commit_id'], axis=1)
        to_save_df = to_save_df.sort_values(by='line_score', ascending=False)
        to_save_df['row'] = np.arange(1,len(to_save_df)+1)
        to_save_df.to_csv('./data/line-level_ranking_result/'+cur_proj+'_'+str(commit)+'.csv',index=False)

        line_label = list(cur_RF_result['line_level_label'])

        for n, agg_method in enumerate(score_cols):
            
            RF_line_scr = list(cur_RF_result[agg_method])
            
            IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc = get_line_level_metrics(RF_line_scr, line_label)

            IFA_list.append(IFA)
            recall_20_percent_effort_list.append(top_20_percent_LOC_recall)
            effort_20_percent_recall_list.append(effort_at_20_percent_LOC_recall)
            top_10_acc_list.append(top_10_acc)

        IFA_df.loc[commit] = IFA_list
        recall_20_percent_effort_df.loc[commit] = recall_20_percent_effort_list
        effort_20_percent_recall_df.loc[commit] = effort_20_percent_recall_list
        top_10_acc_df.loc[commit] = top_10_acc_list

    # the results are then used to make boxplot
    IFA_df.to_csv('./text_metric_line_eval_result/'+cur_proj+'_IFA_min_df_3_300_trees.csv')
    recall_20_percent_effort_df.to_csv('./text_metric_line_eval_result/'+cur_proj+'_recall_20_percent_effort_min_df_3_300_trees.csv') 
    effort_20_percent_recall_df.to_csv('./text_metric_line_eval_result/'+cur_proj+'_effort_20_percent_recall_min_df_3_300_trees.csv')
    top_10_acc_df.to_csv('./text_metric_line_eval_result/'+cur_proj+'_top_10_acc_min_df_3_300_trees.csv')
    
    print('finish', cur_proj)








def pred(data):
    # checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    # output_dir = os.path.join("F:/毕业设计/JIT-Fine-master/model/jitfine/saved_models_concatavgrcst1", '{}'.format(checkpoint_prefix))
    # checkpoint = torch.load(output_dir)
    path = "codebert-base"
    config = RobertaConfig.from_pretrained(path)
    config.num_labels = 2
    config.feature_size = 14
    config.hidden_dropout_prob = 0.1
    tokenizer = RobertaTokenizer.from_pretrained(path)
    special_tokens_dict = {'additional_special_tokens': ["[ADD]", "[DEL]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    model = RobertaModel.from_pretrained("codebert-base", config=config)
    model.resize_token_embeddings(len(tokenizer))
    output_dir = "model/jitfine/saved_models_concatrcs3/checkpoints/checkpoint-best-f1/model.bin"
    checkpoint = torch.load(output_dir)
    args = " "
    model.resize_token_embeddings(len(tokenizer))
    model = Model(model, config, tokenizer, args)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    c=[]
    device = torch.device("cpu")
    model.to(device)
    for i in data:
    # logger.info("Successfully load epoch {}'s model checkpoint".format(checkpoint['epoch']))
        inputs_ids=i[:512]
        inputs_ids = [int(i) for i in inputs_ids]
        attn_masks=i[512:1024]
        attn_masks = [int(i) for i in attn_masks]
        manual_features=i[1024:]
        inputs_ids=torch.tensor(inputs_ids,device=device).unsqueeze(0).long()
        attn_masks=torch.tensor(attn_masks,device=device).unsqueeze(0).long()
        manual_features=torch.tensor(manual_features,device=device).unsqueeze(0)
    # logger.info("Successfully load epoch {}'s model checkpoint".format(checkpoint['epoch']))
        logit = model(inputs_ids, attn_masks, manual_features,output_attentions=True)
        y = logit.cpu().detach().numpy()
        y = y[0][0]
        x = [1-y,y]
        c.append(x)

    # multi-gpu evaluate
    # if args.n_gpu > 1 and eval_when_training is False:
    #     model = torch.nn.DataParallel(model)
    # logit= model(inputs_ids, attn_masks, manual_features)
    return np.array(c)




def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", nargs=2, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", nargs=2, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", nargs=2, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    #parser.add_argument("--head_dropout_prob", default=0.5)###########
    parser.add_argument('--head_dropout_prob', type=float, default=0.1,
                        help="Number of labels")########                             
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--no_abstraction", default=True)  ###########   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_seed', type=int, default=123456,
                        help="random seed for data order initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    parser.add_argument('--feature_size', type=int, default=14,
                        help="Number of features")
    parser.add_argument('--num_labels', type=int, default=2,
                        help="Number of labels")
    parser.add_argument('--semantic_checkpoint', type=str, default=None,
                        help="Best checkpoint for semantic feature")
    parser.add_argument('--manual_checkpoint', type=str, default=None,
                        help="Best checkpoint for manual feature")
    parser.add_argument('--max_msg_length', type=int, default=64,
                        help="Number of labels")
    parser.add_argument('--patience', type=int, default=5,
                        help='patience for early stop')
    parser.add_argument("--only_adds", action='store_true',
                        help="Whether to run eval on the only added lines.")
    parser.add_argument("--buggy_line_filepath", type=str,
                        help="complete buggy line-level  data file for RQ3")

    args = parser.parse_args()
    return args



def preprocess_feature_from_explainer(exp):
    features_val = exp.as_list(label=0)
    new_features_val = [tup for tup in features_val if float(tup[1]) > 0] # only score > 0 that indicates buggy token

    feature_dict = {re.sub('\s.*','',val[0]):val[1] for val in new_features_val}

    sorted_feature_dict = sorted(feature_dict.items(), key=operator.itemgetter(1), reverse=True)

    sorted_feature_dict = {tup[0]:tup[1] for tup in sorted_feature_dict if tup[0] not in commit_metrics}
    tokens_list = list(sorted_feature_dict.keys())

    return sorted_feature_dict, tokens_list

def add_agg_scr_to_list(line_stuff, scr_list):
    if len(scr_list) < 1:
        scr_list.append(0)

    line_stuff.append(np.mean(scr_list))
    line_stuff.append(np.median(scr_list))
    line_stuff.append(np.sum(scr_list))


def plot_result(cur_proj):
    metrics = ['top_10_acc', 'top_5_acc' 'recall_20_percent_effort', 'effort_20_percent_recall', 'IFA']
    metrics_label = ['Top-10-ACC', 'Top-5-ACC', 'Recall20%Effort', 'Effort@20%LOC', 'IFA']

    for i in range(0, 5):
        result_df = pd.read_csv(
            data_path + './text_metric_line_eval_result/' + cur_proj + '_' + metrics[i] + '_min_df_3_300_trees.csv')
        result = result_df['sum-all-tokens']
        print(
            f'{metrics_label[i]}:mean:{round(sum(result) / len(result), 4)}')



def main(args):
    # Setup CUDA, GPU
    device = torch.device("cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    line_score_df_col_name = ['commit_id', 'total_tokens', 'line_level_label']
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )
    # Set seed
    config =RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = args.num_labels
    config.feature_size = args.feature_size
    config.hidden_dropout_prob = args.head_dropout_prob
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    
    model = RobertaModel.from_pretrained(args.model_name_or_path, config=config)

    # model.resize_token_embeddings(len(tokenizer))
    # logger.info("Training/evaluation parameters %s", args)

    # model = Model(model, config, tokenizer, args)
    # checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    # output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    # checkpoint = torch.load(output_dir)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # # logger.info("Successfully load epoch {}'s model checkpoint".format(checkpoint['epoch']))
    # model.to(args.device)

    # clf = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # train_code = "code_change_arr = count_vect.transform(code_df['code']).astype(np.int16).toarray()"
    # clf = pickle.load(open(model_path+proj_name+'_RF_DE_SMOTE_min_df_3.pkl','rb'))
    # count_vect = CountVectorizer(min_df=3, ngram_range=(1,1))
    # count_vect.fit(train_code)
    # feature_names =count_vect.get_feature_names()
    # input_tokens = tokenizer.tokenze(msg)
    tokens = list()
    mask = list()
    for i in range(512):
        tokens.append("token{}".format(i))
        mask.append("mask{}".format(i))
    manual_features_column = ["ns", "nd", "nf", "entropy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp", "sexp"]
    feature_names = tokens+mask+manual_features_column
    train_data = "data/jitfine/changes_train.pkl data/jitfine/features_train.pkl"
    train_dataset = TextDataset(tokenizer, args, file_path=args.train_data_file, mode='train')
    labels = []
    X = []
    for train_data in train_dataset:
        inputs_id, attn_mask, manual_feature, label =train_data
        x=[]
        for i in inputs_id:
            x.append(float(i))
        for i in attn_mask:
            x.append(float(i))
        for i in manual_feature:
            x.append(float(i))
        X.append(x)
        labels.append(int(label))
    # inputs_ids, attn_masks, manual_features, labels = train_dataset
    # X = (inputs_ids,attn_masks, manual_features)
    X = np.array(X)
    Y = np.array(labels)
    explainer = LimeTabularExplainer(X, mode='classification',
                                feature_names=feature_names, 
                                random_state=42)
    

    test_data = "data/jitfine/changes_test.pkl data/jitfine/features_test.pkl"
    test_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file, mode='test')
    labels = []
    X = []
    for test_data in test_dataset:
        inputs_id, attn_mask, manual_feature, label =test_data
        x=[]
        for i in inputs_id:
            x.append(float(i))
        for i in attn_mask:
            x.append(float(i))
        for i in manual_feature:
            x.append(float(i))
        X.append(x)
        labels.append(int(label))
    # inputs_ids, attn_masks, manual_features, labels = train_dataset
    # X = (inputs_ids,attn_masks, manual_features)
    X = np.array(X)
    Y = np.array(labels)
    file_path = args.test_data_file
    changes_filename, features_filename = file_path


    #
    top_k_tokens = np.arange(10, 201, 10)
    agg_methods = ['avg', 'median', 'sum']
    max_str_len_list = 100


    data = []
    ddata = pd.read_pickle(changes_filename)

    features_data = pd.read_pickle(features_filename)
    features_data = convert_dtype_dataframe(features_data, manual_features_columns)

    features_data = features_data[['commit_hash'] + manual_features_columns]
    # test_sampler = SequentialSampler(test_dataset)
    # test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=4)
    score_cols = [agg+'-top-'+str(k)+'-tokens' for agg in agg_methods for k in top_k_tokens] + [agg+'-all-tokens' for agg in agg_methods]

    eval_loss = 0.0
    nb_eval_steps = 0
    # model.eval()
    logits = []
    y_trues = []
    all_buggy_line_result_df = []
    data_path_ = 'data/jitfine/'
    prediction_result = pd.read_csv('model/jitfine/saved_models_concatrcs3/checkpoints/prediction.csv')
    line_level_df = pd.read_pickle(data_path_ + 'changes_complete_buggy_line_level.pkl').dropna()

    correctly_predicted_commit = list(prediction_result[(prediction_result['pred']==1) &
                                                (prediction_result['actual']==1)]['test_commit'])
  
    for commit in correctly_predicted_commit:
        code_change_from_line_level_df = list(line_level_df[line_level_df['commit_id']==commit]['changed_line'])
        line_level_label = list(line_level_df[line_level_df['commit_id']==commit]['label'])

        line_score_df = pd.DataFrame(columns = line_score_df_col_name)
        line_score_df['line_num'] = np.arange(0,len(code_change_from_line_level_df))
        line_score_df = line_score_df.set_index('line_num')
        index = features_data[features_data["commit_hash"]==commit].index.tolist()[0]  
        exp = explainer.explain_instance(X[index], predict_fn=pred,
                                    num_features=len(feature_names), top_labels=1,
                                    num_samples=500)
        exp.save_to_file('filename'+str(commit)+".html")
        # logit= model(inputs_ids, attn_masks, manual_features)
        
        sorted_feature_score_dict, tokens_list = preprocess_feature_from_explainer(exp)
        print(sorted_feature_score_dict)
        print(tokens_list)
        for line_num, line in enumerate(code_change_from_line_level_df): # for each line (sadly this loop is needed...)
            line_stuff = []
            line_score_list = np.zeros(100) # this is needed to store result in dataframe
            token_list = line.split()[:100]
            line_stuff.append(line)
            line_stuff.append(len(token_list))

            for tok_idx, tok in enumerate(token_list):
                score = sorted_feature_score_dict.get(tok,0)
                print(score)
                line_score_list[tok_idx] = score
                
            # calculate top-k tokens first then followed by all tokens

            line_stuff = line_stuff + list(line_score_list)

            for k in top_k_tokens: # for each k in top-k tokens
                top_tokens = tokens_list[0:k-1]
                top_k_scr_list = []

                if len(token_list) < 1:
                    top_k_scr_list.append(0)
                else:
                    for tok in token_list:
                        score = 0
                        if tok in top_tokens:
                            score = sorted_feature_score_dict.get(tok,0)
                        top_k_scr_list.append(score)

                add_agg_scr_to_list(line_stuff, top_k_scr_list)
            print(line_score_df)
            print(line_stuff)
            add_agg_scr_to_list(line_stuff, list(line_score_list[:len(token_list)]))
            line_score_df.loc[line_num] = line_stuff

        line_score_df['commit_id'] = [commit]*len(line_level_label)
        line_score_df['line_level_label'] = line_level_label

        all_buggy_line_result_df.append(line_score_df)
        
        del exp, sorted_feature_score_dict, tokens_list, line_score_df

    openstack_line_level = all_buggy_line_result_df
    pd.concat(openstack_line_level).to_csv(data_path + 'changes_line_level_result_min_df_3_300_trees.csv',
                                           index=False)
    ## Defective line ranking evaluation

    score_cols = [agg + '-top-' + str(k) + '-tokens' for agg in agg_methods for k in top_k_tokens] + [
        agg + '-all-tokens'
        for agg in
        agg_methods]
    line_score_df_col_name = ['commit_id', 'total_tokens', 'line_level_label'] + score_cols

    eval_line_level_at_commit('changes', data_path)

    plot_result('changes')


    



if __name__ == "__main__":
    cur_args = parse_args()
    create_path_if_not_exist(cur_args.output_dir)
    main(cur_args)