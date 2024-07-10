import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.rnn = nn.RNN(config.feature_size,config.hidden_size,4)
        # self.lstm = nn.LSTM(config.feature_size, config.hidden_size)  
        # self.linear1 = nn.Linear(config.hidden_size, 1)  
        # self.avgpooling = nn.AvgPool1d(3)
        # self.lstm2 = nn.LSTM(config.hidden_size, config.hidden_size)
        # self.linear2 = nn.Linear(config.hidden_size,1)
        # self.manual_dense = nn.Linear(config.feature_size, config.hidden_size)
        self.linear = nn.Linear(192,config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj_new = nn.Linear(config.hidden_size + config.hidden_size, 1)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(1, 64, (K, 768)) for K in range(1,4)])
        # self.avgpooling = nn.AdaptiveAvgPool1d()

    def forward_fearture(self, features, manual_features=None, **kwargs):
        x = x = features
        # x = x.permute(1, 0, 2)  # (batch_size, sentences, hidden_size)
        x = x.unsqueeze(1)  # (batch_size, input_channels, sentences, hidden_size)
        # print(x.shape)
	# CNN
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs_code_line]  
      	# max pooling
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # (batch_size, output_channels, num_sentences) * ks
        x = torch.cat(x, 1)  # (batch_size, channel_output * ks)
        return x





    def forward(self, features, manual_features=None, **kwargs):
        x = self.forward_fearture(features,**kwargs)  # take <s> token (equiv. to [CLS])  [bs,hidden_size]
        # x = self.avgpooling(x)
        y = manual_features.float()  # [bs, feature_size]
        y,_= self.rnn(y)
        # y = self.dropout(y)
        # y = self.linear1(y)
        y = torch.tanh(y)
        # x = [F.avg_pool1d(i, i.size(2)).unsqueeze(0) for i in x]  
        # x = torch.cat(x, 1) 
        # x,__ = self.lstm2(x)
        # x = self.dropout(x)
        # x = self.linear2(x)
        # x = torch.tanh(x)
        x = self.dropout(x)
        x = self.linear(x)
        # print(x.shape)
        x = torch.cat((x, y), dim=-1)
        x = self.dropout(x)
        x = self.out_proj_new(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args


    


    def forward(self, inputs_ids, attn_masks, manual_features=None,
                labels=None, output_attentions=None):
        outputs = \
            self.encoder(input_ids=inputs_ids, attention_mask=attn_masks, output_attentions=output_attentions)

        last_layer_attn_weights = outputs.attentions[self.config.num_hidden_layers - 1][:, :,
                                  0].detach() if output_attentions else None
        # msked = attn_masks*inputs_ids
        # msked = torch.tensor(msked)
        # sum_msk = msked
        # print("outputs:",len(outputs))
        # print("outputs[0]:",len(outputs[0]))
        # avg = torch.mean(outputs[0])
        # print(avg.shape())
        # out = torch.stack(outputs)
        # print(out.shape)
        # semantic_features = F.avg_pool1d(outputs[0])
        mask = attn_masks.unsqueeze(-1).expand(outputs[0].size()).float()
        masked_embeddings = outputs[0] * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        mean_pooled = mean_pooled.unsqueeze(0)
        mean_pooled = mean_pooled * 512
        mean = mean_pooled
        # print(mean.shape)
        for i in range(9):
            mean = torch.concat((mean,mean),dim=0)
        # print(mean.shape)
        mean = mean.permute(1,0,2)
        # print(mean.shape)
        # out = torch.tensor(outputs[0])
        # print(out.shape)
        logits = self.classifier(mean, manual_features)

        prob = torch.sigmoid(logits)
        if labels is not None:
            loss_fct = BCELoss()
            loss = loss_fct(prob, labels.unsqueeze(1).float())
            return loss, prob, last_layer_attn_weights
        else:
            return prob

