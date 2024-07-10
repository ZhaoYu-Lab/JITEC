from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
from sklearn.preprocessing import MinMaxScaler
from lime.lime_tabular import LimeTabularExplainer
import time, pickle, math, warnings, os, operator
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel)
from tqdm import tqdm, trange
import multiprocessing
from JITFine.concat.model import Model
from JITFine.my_util import convert_examples_to_features, TextDataset, eval_result, preprocess_code_line, \
    get_line_level_metrics, create_path_if_not_exist

logger = logging.getLogger(__name__)


line_score_df_col_name = ['commit_id', 'total_tokens', 'line_level_label']

# since we don't want to use commit metrics in LIME
commit_metrics = ['la', 'la', 'ld', 'ld', 'nf', 'nd_y', 'nd', 'ns', 'ent', 'ent', 'nrev', 'rtime', 'hcmt', 'self',
                  'ndev',
                  'age', 'age', 'nuc', 'app_y', 'aexp', 'rexp', 'arexp', 'rrexp', 'asexp', 'rsexp', 'asawr', 'rsawr']




def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,  input_ids, input_mask,  manual_features,label):
        # self.commit_id = commit_id
        self.input_ids = input_ids.long()
        self.input_mask = input_mask.long()
        # self.input_tokens = input_tokens
        self.label = label
        self.manual_features = manual_features


 
class TextDataset1(Dataset):
    def __init__(self, tokenizer, args,input_ids, input_mask, manual_features,  label, mode='train'):
        self.examples = []
        self.args = args
        for input_id, input_mas, manual_feature,  labe in zip(input_ids, input_mask, manual_features,  label):
            self.examples.append(InputFeatures(input_id, input_mas, manual_feature,labe))
        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].input_mask),
                torch.tensor(self.examples[item].manual_features),
                torch.tensor(self.examples[item].label))

def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    # build dataloader
    
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE,KMeansSMOTE,SMOTENC,SVMSMOTE,ADASYN,RandomOverSampler
    smo = SMOTETomek(random_state=args.seed)
    inputs_ids = []
    attn_masks = []
    manual_features = []
    labels = []
    X = []
    for train_data in train_dataset:
        inputs_id, attn_mask, manual_feature, label =train_data
        x=[]
        # if(len(inputs_id)!=512):
        #     print("id")
        #     print(len(inputs_ids))
        # if(len(attn_mask)!=512):
        #     print("a") 
        # if(len(manual_feature)!=14):
        #     print("m")     
        #     print(manual_feature)
        # print(train_data)
        # print(torch.concat((inputs_id, attn_mask, manual_feature)))
        for i in inputs_id:
            x.append(float(i))
        for i in attn_mask:
            x.append(float(i))
        for i in manual_feature:
            x.append(float(i))
            # X.append(torch.concat((inputs_id, attn_mask, manual_feature)))
        # inputs_id = inputs_id.tolist()
        # attn_mask=attn_mask.tolist()
        # manual_feature=manual_feature.tolist()
        # print(inputs_id)
        # print(type(inputs_ids))
        # X.append((inputs_id.append(attn_mask)).append(manual_feature))
        X.append(x)
        # for i in label:
        #     label.append(int(i))
        labels.append(int(label))
    # inputs_ids, attn_masks, manual_features, labels = train_dataset
    # X = (inputs_ids,attn_masks, manual_features)
    X = np.array(X)
    Y = np.array(labels)
    # print(len(X))
    # print(len(Y))
    X1,Y1 = smo.fit_resample(X,Y)
    for x in X1:
        inputs_id=x[:512]
        attn_mask=x[512:1024]
        manual_feature=x[1024:]
        inputs_ids.append(inputs_id)
        attn_masks.append(attn_mask)
        manual_features.append(manual_feature)
    inputs_ids=torch.tensor(inputs_ids)
    attn_masks=torch.tensor(attn_masks)
    manual_features=torch.tensor(manual_features)
    Y1 = torch.tensor(Y1)
    # inputs_ids, attn_masks, manual_features =Y1
    train_dataset = TextDataset1(tokenizer=args.tokenizer_name,args=args,input_ids=inputs_ids, input_mask=attn_masks,
                                 manual_features= manual_features,label=Y1)
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 5
    args.warmup_steps = 0
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    best_f1 = 0
    global_step = 0
    model.zero_grad()
    patience = 0

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_loss = 0
        tr_num = 0
        # print("bar:",train_dataloader.size())
        for step, batch in enumerate(bar):
            (inputs_ids, attn_masks, manual_features, labels) = [x.to(args.device) for x in batch]
            model.train()
            loss, logits, _ = model(inputs_ids, attn_masks, manual_features, labels)
            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % args.save_steps == 0:
                logger.info("epoch {} step {} loss {}".format(idx, step + 1, round(tr_loss / tr_num, 5)))
                tr_loss = 0
                tr_num = 0

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            global_step += 1
            if (step + 1) % args.save_steps == 0:
                results = evaluate(args, model, tokenizer, eval_when_training=True)
                checkpoint_prefix = f'epoch_{idx}_step_{step}'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                torch.save({
                    'epoch': idx,
                    'step': step,
                    'patience': patience,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}, output_dir)
                logger.info("Saving epoch %d step %d model checkpoint to %s, patience %d", idx, global_step, output_dir,
                            patience)
                # Save model checkpoint
                if results['eval_f1'] > best_f1:
                    best_f1 = results['eval_f1']
                    logger.info("  " + "*" * 20)
                    logger.info("  Best f1:%s", round(best_f1, 4))
                    logger.info("  " + "*" * 20)

                    checkpoint_prefix = 'checkpoint-best-f1'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                    patience = 0
                    torch.save({
                        'epoch': idx,
                        'step': step,
                        'patience': patience,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}, output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                else:
                    patience += 1
                    if patience > args.patience * 5:
                        logger.info('patience greater than {}, early stop!'.format(args.patience))
                        return


def evaluate(args, model, tokenizer, eval_when_training=False):
    # build dataloader
    cache_dataset = os.path.dirname(args.eval_data_file[0]) + f'/valid_set_cache_msg{args.max_msg_length}.pkl'
    if args.no_abstraction:
        cache_dataset = cache_dataset.split('.pkl')[0] + '_raw.pkl'
    logger.info("Cache Dataset file at %s ", cache_dataset)
    if os.path.exists(cache_dataset):
        eval_dataset = pickle.load(open(cache_dataset, 'rb'))
    else:
        eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file, mode='valid')
        pickle.dump(eval_dataset, open(cache_dataset, 'wb'))
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        (inputs_ids, attn_masks, manual_features, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            loss, logit, _ = model(inputs_ids, attn_masks, manual_features, labels)
            torch.cuda.empty_cache()
            eval_loss += loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5

    y_preds = logits[:, -1] > best_threshold
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='binary')
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,

    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def lime_explainer(args, tokenizer):
    tokens = list()
    mask = list()
    for i in range(512):
        tokens.append("token{}".format(i))
        mask.append("mask{}".format(i))
    manual_features_column = ["ns", "nd", "nf", "entropy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp", "sexp"]
    feature_names = tokens+manual_features_column + ["length"]
    train_data = "data/jitfine/changes_train.pkl data/jitfine/features_train.pkl"
    train_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file, mode='train')
    labels = []
    X = []
    for train_data in train_dataset:
        inputs_id, attn_mask, manual_feature, label =train_data
        x=[]
        for i in inputs_id:
            x.append(float(i))
        for i in manual_feature:
            x.append(float(i))
        length = 0
        for i in attn_mask:
            if(int(i)!=0):
                length = length + 1
        labels.append(int(label))
        x.append(length)
        X.append(x)
    for i in range(512):
        mask.append("mask{}".format(i))
    X = np.array(X)
    Y = np.array(labels)
    from lime.lime_tabular import LimeTabularExplainer
    explainer = LimeTabularExplainer(X, mode='classification',
                                feature_names=feature_names, 
                                discretize_continuous=False, random_state=args.seed)
    
    return explainer

def test(args, model, tokenizer, best_threshold=0.5):
    # build dataloader
    cache_dataset = os.path.dirname(args.test_data_file[0]) + f'/test_set_cache_msg{args.max_msg_length}.pkl'
    if args.no_abstraction:
        cache_dataset = cache_dataset.split('.pkl')[0] + '_raw.pkl'
    logger.info("Cache Dataset file at %s ", cache_dataset)
    if os.path.exists(cache_dataset):
        eval_dataset = pickle.load(open(cache_dataset, 'rb'))
    else:
        eval_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file, mode='test')
        pickle.dump(eval_dataset, open(cache_dataset, 'wb'))
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    exp =lime_explainer(args,tokenizer)
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    attns = []
    for batch in eval_dataloader:
        (inputs_ids, attn_masks, manual_features, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            loss, logit, attn_weights = model(inputs_ids, attn_masks, manual_features, labels, output_attentions=True)
            last_layer_attn_weights = attn_weights
            eval_loss += loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
            attns.append(last_layer_attn_weights.cpu().numpy())

        nb_eval_steps += 1
    # output result
    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    attns = np.concatenate(attns, 0)

    y_preds = logits[:, -1] > best_threshold
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='binary')

    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,

    }
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    result = []
    cache_buggy_line = os.path.join(os.path.dirname(args.buggy_line_filepath),
                                    'changes_complete_buggy_line_level_cache.pkl')
    if os.path.exists(cache_buggy_line):
        commit2codes, idx2label = pickle.load(open(cache_buggy_line, 'rb'))
    else:
        commit2codes, idx2label = commit_with_codes(args.buggy_line_filepath, tokenizer)
        pickle.dump((commit2codes, idx2label), open(cache_buggy_line, 'wb'))

    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = [], [], [], [], []
    for example, predd, prob, attn in zip(eval_dataset.examples, y_preds, logits[:, -1], attns):
        result.append([example.commit_id, prob, predd, example.label])
        
        # calculate
        if int(example.label) == 1 and int(predd) == 1 and '[ADD]' in example.input_tokens:
            cur_codes = commit2codes[commit2codes['commit_id'] == example.commit_id]
            cur_labels = idx2label[idx2label['commit_id'] == example.commit_id]
            # print(cur_codes)
            cur_IFA, cur_top_20_percent_LOC_recall, cur_effort_at_20_percent_LOC_recall, cur_top_10_acc, cur_top_5_acc = deal_with_attns(
                example, attn,
                predd, cur_codes,
                cur_labels, exp,tokenizer,args.only_adds)
            IFA.append(cur_IFA)
            top_20_percent_LOC_recall.append(cur_top_20_percent_LOC_recall)
            effort_at_20_percent_LOC_recall.append(cur_effort_at_20_percent_LOC_recall)
            top_10_acc.append(cur_top_10_acc)
            top_5_acc.append(cur_top_5_acc)

    logger.info(
        'Top-10-ACC: {:.4f},Top-5-ACC: {:.4f}, Recall20%Effort: {:.4f}, Effort@20%LOC: {:.4f}, IFA: {:.4f}'.format(
            round(np.mean(top_10_acc), 4), round(np.mean(top_5_acc), 4),
            round(np.mean(top_20_percent_LOC_recall), 4),
            round(np.mean(effort_at_20_percent_LOC_recall), 4), round(np.mean(IFA), 4))
    )
    RF_result = pd.DataFrame(result)
    RF_result.to_csv(os.path.join(args.output_dir, "predictions.csv"), sep='\t', index=None)



def commit_with_codes(filepath, tokenizer):
    data = pd.read_pickle(filepath)
    commit2codes = []
    idx2label = []
    for _, item in data.iterrows():
        commit_id, idx, changed_type, label, raw_changed_line, changed_line = item
        line_tokens = [token.replace('\u0120', '') for token in tokenizer.tokenize(changed_line)]
        for token in line_tokens:
            commit2codes.append([commit_id, idx, changed_type, token])
        idx2label.append([commit_id, idx, label])
    commit2codes = pd.DataFrame(commit2codes, columns=['commit_id', 'idx', 'changed_type', 'token','changed_line'])
    idx2label = pd.DataFrame(idx2label, columns=['commit_id', 'idx', 'label'])
    return commit2codes, idx2label


def preprocess_feature_from_explainer(features_val):
    new_features_val = [tup for tup in features_val if float(tup[1]) > 0] # only score > 0 that indicates buggy token

    feature_dict = {re.sub('\s.*','',val[0]):val[1] for val in new_features_val}

    sorted_feature_dict = sorted(feature_dict.items(), key=operator.itemgetter(1), reverse=True)

    sorted_feature_dict = {tup[0]:tup[1] for tup in sorted_feature_dict if tup[0] not in commit_metrics}
    tokens_list = list(sorted_feature_dict.keys())
    return sorted_feature_dict, tokens_list


def id_range(x) ->int:
    if int(x)<0:
        return 0
    elif int(x)>50266:
        return 50266
    else:
        return int(x)


def pred(data):
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
    output_dir = "model/jitfine/saved_models_concatrcs8/checkpoints/checkpoint-best-f1/model.bin"
    checkpoint = torch.load(output_dir)
    args = " "
    model.resize_token_embeddings(len(tokenizer))
    model = Model(model, config, tokenizer, args)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    c=[]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for i in data:
    # logger.info("Successfully load epoch {}'s model checkpoint".format(checkpoint['epoch']))
        inputs_ids=i[:512]
        inputs_ids = [int(id_range(i)) for i in inputs_ids]
        attn_masks=[1] * int(i[526])
        attn_masks = attn_masks +[0]*(512-len(attn_masks))
        attn_masks = attn_masks[:512]
        manual_features=i[512:526]
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



def deal_with_attns(item, attns, predd, commit2codes, idx2label ,explainer,tokenizer,only_adds=False):
    '''
    score for each token
    :param item:
    :param attns:
    :param pred:
    :param commit2codes:
    :param idx2label:
    :return:
    '''
    commit_id = item.commit_id
    input_tokens = item.input_tokens
    commit_label = item.label

    # remove msg,cls,eos,del
    begin_pos = input_tokens.index('[ADD]')
    end_pos = input_tokens.index('[DEL]') if '[DEL]' in input_tokens else len(input_tokens) - 1

    

    attns = attns.mean(axis=0)[begin_pos:end_pos]
    # print(attns)




    # calculate score for each line in commit
    if only_adds:
        commit2codes = commit2codes[commit2codes['changed_type'] == 'added']  # only count for added lines
    commit2codes = commit2codes.drop('commit_id', axis=1)
    commit2codes = commit2codes.drop('changed_type', axis=1)


    x=[]
    for i in item.input_ids:
        x.append(float(i))
    for i in item.manual_features:
        x.append(float(i))
    length = 0
    for i in item.input_mask:
        if(int(i)!=0):
            length = length + 1
    x.append(length)
    x = np.array(x)
    exp = explainer.explain_instance(x, predict_fn=pred, distance_metric='jaccard',
                                num_features=527, top_labels=1,
                                num_samples=5000)
    try:
        features_val = exp.as_list(label=0)
    except Exception as e:
        features_val = exp.as_list(label=1)
    # print(features_val)
    sorted_feature_score_dict, tokens_list = preprocess_feature_from_explainer(features_val)
    new_dict={}
    tokens = list()
    for i in range(512):
        tokens.append("token{}".format(i))
    manual_features_column = ["ns", "nd", "nf", "entropy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp", "sexp"]
    feature_names = tokens+manual_features_column + ["length"]
    for key in sorted_feature_score_dict:
        place = feature_names.index(key)
        if place < 512:
            num = int(item.input_ids[place])
            place =tokenizer.convert_ids_to_tokens(num)
            new_dict[place]=sorted_feature_score_dict[key]
    sorted_feature_score_dict = new_dict
    tokens_list = list(sorted_feature_score_dict.keys())
    lime_score = []
    line_score_list = np.zeros(10000) # this is needed to store result in dataframe



    # 归一化处理后合并
    scaler = MinMaxScaler()
    attns = scaler.fit_transform(np.array(attns).reshape(-1, 1))
    attns = attns.reshape(-1)
    attn_df = pd.DataFrame()
    attn_df['token'] = [token.replace('\u0120', '') for token in
                        input_tokens[begin_pos:end_pos]]
    # average score for multi-heads
    attn_df['score'] = attns
    attn_df = attn_df.sort_values(by='score', ascending=False)
    attn_df = attn_df.groupby('token').sum()
    attn_df['token'] = attn_df.index
    attn_df = attn_df.reset_index(drop=True)

    



    result_df = pd.merge(commit2codes, attn_df, how='left', on='token')
    result_df = result_df.groupby(['idx']).sum()
    result_df = result_df.reset_index(drop=False)

    for line in result_df['token']:
        token_list = tokenizer.tokenize(' '.join(line))
        #         line_score_list = np.zeros(0) token_list
        for tok_idx, tok in enumerate(token_list):
            score = sorted_feature_score_dict.get(tok,0)
            line_score_list[tok_idx] = score
        lime_score.append(line_score_list.sum())
    lime_score = scaler.fit_transform(np.array(lime_score).reshape(-1, 1))
    lime_score = lime_score.reshape(-1)
    # scaler = MinMaxScaler()
    # lime_score = scaler.fit_transform(np.array(lime_score).reshape(-1, 1)) 
    # print(result_df)
    result_df['score'] = result_df['score']+lime_score
    result_df = pd.merge(result_df, idx2label, how='inner', on='idx')
    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = get_line_level_metrics(
        result_df['score'].tolist(), result_df['label'].tolist())
    return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc


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


def main(args):
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )
    # Set seed

    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = args.num_labels
    config.feature_size = args.feature_size
    config.hidden_dropout_prob = args.head_dropout_prob
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    special_tokens_dict = {'additional_special_tokens': ["[ADD]", "[DEL]"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = RobertaModel.from_pretrained(args.model_name_or_path, config=config)

    model.resize_token_embeddings(len(tokenizer))
    logger.info("Training/evaluation parameters %s", args)

    model = Model(model, config, tokenizer, args)
    # Training
    if args.do_train:
        if args.semantic_checkpoint:
            semantic_checkpoint_prefix = 'checkpoint-best-f1/model.bin'
            output_dir = os.path.join(args.semantic_checkpoint, '{}'.format(semantic_checkpoint_prefix))
            logger.info("Loading semantic checkpoint from {}".format(output_dir))
            checkpoint = torch.load(output_dir)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if args.manual_checkpoint:
            manual_checkpoint_prefix = 'checkpoint-best-f1/model.bin'
            output_dir = os.path.join(args.manual_checkpoint, '{}'.format(manual_checkpoint_prefix))
            logger.info("Loading manual checkpoint from {}".format(output_dir))
            checkpoint = torch.load(output_dir)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        train_dataset = TextDataset(tokenizer, args, file_path=args.train_data_file)
        
        for idx, example in enumerate(train_dataset.examples[:1]):
            logger.info("*** Example ***")
            logger.info("idx: {}".format(idx))
            logger.info("label: {}".format(example.label))
            logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
            logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))


        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        checkpoint = torch.load(output_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)
        result = evaluate(args, model, tokenizer)

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        checkpoint = torch.load(output_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Successfully load epoch {}'s model checkpoint".format(checkpoint['epoch']))
        model.to(args.device)
        test(args, model, tokenizer, best_threshold=0.5)
        eval_result(os.path.join(args.output_dir, "predictions.csv"), args.test_data_file[-1])

    return results


if __name__ == "__main__":
    cur_args = parse_args()
    create_path_if_not_exist(cur_args.output_dir)
    main(cur_args)
