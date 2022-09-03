# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch
'''class Model(nn.Module):   
    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.embedding_code=nn.Embedding(vocab_size,128)
        self.embedding_nl=nn.Embedding(vocab_size,128)
        self.gru_code = torch.nn.GRU(input_size=128,hidden_size=128//2,
                                   num_layers=3,batch_first=True,bidirectional=True)

        self.gru_nl=torch.nn.GRU(input_size=128,hidden_size=128//2,
                                 num_layers=3,batch_first=True,bidirectional=True)
        #self.dense_code = nn.Linear(128,1)
        #self.dense_nl=nn.Linear(128,1)

        
    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            #inputs_embeddings=self.encoder.embeddings(code_inputs)
            inputs_embeddings=self.embedding_code(code_inputs)
            inputs_embeddings,_=self.gru_code(inputs_embeddings)
            #mask=code_inputs.ne(1).float()
            #scores=self.dense_code(inputs_embeddings)[:,:,0]-1e10*code_inputs.eq(1).float()
            #weights=torch.softmax(scores,-1)
            #avg_embeddings=torch.einsum("ab,abc->ac",weights,inputs_embeddings)
            #return avg_embeddings
            #print(inputs_embeddings[:,0,:].size())
            return inputs_embeddings[:,0,:]
        else:
            #inputs_embeddings=self.encoder.embeddings(nl_inputs)
            inputs_embeddings=self.embedding_nl(nl_inputs)
            inputs_embeddings,_=self.gru_nl(inputs_embeddings)
            #mask=nl_inputs.ne(1).float()
            #scores=self.dense_nl(inputs_embeddings)[:,:,0]-1e10*nl_inputs.eq(1).float()
            #weights=torch.softmax(scores,-1)
            #avg_embeddings=torch.einsum("ab,abc->ac",weights,inputs_embeddings)
            #return avg_embeddings
            return inputs_embeddings[:,0,:]'''


class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.gru = torch.nn.GRU(input_size=128, hidden_size=128 // 2,
                                num_layers=3, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(128, 1)

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            inputs_embeddings = self.encoder.embeddings(code_inputs)
            inputs_embeddings, _ = self.gru(inputs_embeddings)
            mask = code_inputs.ne(1).float()
            scores = self.dense(inputs_embeddings)[:, :, 0] - 1e10 * code_inputs.eq(1).float()
            weights = torch.softmax(scores, -1)
            avg_embeddings = torch.einsum("ab,abc->ac", weights, inputs_embeddings)
            return avg_embeddings
        else:
            inputs_embeddings = self.encoder.embeddings(nl_inputs)
            inputs_embeddings, _ = self.gru(inputs_embeddings)
            mask = nl_inputs.ne(1).float()
            scores = self.dense(inputs_embeddings)[:, :, 0] - 1e10 * nl_inputs.eq(1).float()
            weights = torch.softmax(scores, -1)
            avg_embeddings = torch.einsum("ab,abc->ac", weights, inputs_embeddings)
            return avg_embeddings

# 不使用预训练模型的embedding
'''class Model(nn.Module):
    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, 128)
        self.gru = torch.nn.GRU(input_size=128, hidden_size=128 // 2,
                                num_layers=3, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(128, 1)

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            inputs_embeddings = self.embeddings(code_inputs)
            inputs_embeddings, _ = self.gru(inputs_embeddings)
            mask = code_inputs.ne(1).float()
            scores = self.dense(inputs_embeddings)[:, :, 0] - 1e10 * code_inputs.eq(1).float()
            weights = torch.softmax(scores, -1)
            avg_embeddings = torch.einsum("ab,abc->ac", weights, inputs_embeddings)
            return avg_embeddings
        else:
            inputs_embeddings = self.embeddings(nl_inputs)
            inputs_embeddings, _ = self.gru(inputs_embeddings)
            mask = nl_inputs.ne(1).float()
            scores = self.dense(inputs_embeddings)[:, :, 0] - 1e10 * nl_inputs.eq(1).float()
            weights = torch.softmax(scores, -1)
            avg_embeddings = torch.einsum("ab,abc->ac", weights, inputs_embeddings)
            return avg_embeddings'''
        
 
