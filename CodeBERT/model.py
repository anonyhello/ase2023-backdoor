# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas

    
class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            #outputs=self.encoder(code_inputs,attention_mask=code_inputs.ne(1))
            #print(outputs[0].size())
            #print(outputs[1].size())
            #return outputs[1]
            return self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[1]
        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]

      

class Model_attn(nn.Module):
    def __init__(self,encoder):
        super(Model_attn, self).__init__()
        self.encoder=encoder

    def forward(self,code_inputs=None,nl_inputs=None):
        if code_inputs is not None:
            return self.encoder(code_inputs,attention_mask=code_inputs.ne(1),output_attentions=True)

        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1),output_attentions=True)



def index_to_token(tokenizer,index):
    return tokenizer.convert_ids_to_tokens(index)

def get_attention_map(attention_value,index,fig_name,tokenizer):
    # attention_value: [num_layer,num_head,seq_length,seq_length]
    idx=torch.nonzero(index==2)
    attention_value=attention_value[:,:,:idx+1,:idx+1]
    soft_max=torch.nn.Softmax(dim=-1)
    #attention_value=soft_max(attention_value)

    # 得到所有层，所有head的attention value平均值
    attention_single=torch.zeros(attention_value.size()[2],attention_value.size()[3])
    attention_value=attention_value.cpu().numpy()
    attention_single=attention_single.cpu().numpy()

    #for i in range(0,attention_value.shape[0]):
    #    for j in range(0,attention_value.shape[1]):
    #        attention_single=attention_single+attention_value[i][j]

    for i in range(0,attention_value.shape[1]):
        attention_single=attention_single+attention_value[11][i]

    attention_single=attention_value[11][9]


    attention_single=torch.tensor(attention_single)
    #attention_single=soft_max(attention_single)
    attention_single=attention_single.cpu().numpy()
    print('------------------------------------------')
    print(index_to_token(tokenizer,index))
    tokens=index_to_token(tokenizer,index)[:idx+1]
    for i in range(0,len(tokens)):
        tokens[i]=tokens[i].replace('Ġ','')

    data={}
    for i in range(0,len(tokens)):
        data[tokens[i]]=attention_single[:,i]
    pd_data=pandas.DataFrame(data,index=tokens,columns=tokens)
    #fig_name='attention_map.png'
    plt.figure(figsize=(10,8))
    fig=sns.heatmap(pd_data,cmap="YlGnBu")
    heatmap=fig.get_figure()
    heatmap.savefig(fig_name)
    plt.close('all')

