# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch
'''class Model(nn.Module):   
    def __init__(self, vocab_size):
        super(Model, self).__init__()
        #self.encoder = encoder
        self.embedding_code=nn.Embedding(vocab_size,128)
        self.embedding_nl=nn.Embedding(vocab_size,128)

        encoder_layer_code = nn.TransformerEncoderLayer(d_model=128, nhead=8,dim_feedforward=512)
        encoder_layer_nl=nn.TransformerEncoderLayer(d_model=128,nhead=8,dim_feedforward=512)
        self.transformer_encoder_code = nn.TransformerEncoder(encoder_layer_code, num_layers=3)
        self.transformer_encoder_nl=nn.TransformerEncoder(encoder_layer_nl,num_layers=3)
        #self.dense = nn.Linear(128,1)
        
    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            #inputs_embeddings=self.encoder.embeddings(code_inputs).transpose(0,1)
            #inputs_embeddings=self.transformer_encoder(src=inputs_embeddings,src_key_padding_mask=code_inputs.eq(1))
            #inputs_embeddings=inputs_embeddings.transpose(0,1)
            #mask=code_inputs.ne(1).float()
            #scores=self.dense(inputs_embeddings)[:,:,0]-1e10*code_inputs.eq(1).float()
            #weights=torch.softmax(scores,-1)
            #avg_embeddings=torch.einsum("ab,abc->ac",weights,inputs_embeddings)
            #return avg_embeddings
            inputs_embeddings=self.embedding_code(code_inputs).transpose(0,1)
            inputs_embeddings=self.transformer_encoder_code(src=inputs_embeddings,src_key_padding_mask=code_inputs.eq(1))
            inputs_embeddings=inputs_embeddings.transpose(0,1)
            return inputs_embeddings[:,0,:]

        else:
            #inputs_embeddings=self.encoder.embeddings(nl_inputs).transpose(0,1)
            #inputs_embeddings=self.transformer_encoder(src=inputs_embeddings,src_key_padding_mask=nl_inputs.eq(1))
            #inputs_embeddings=inputs_embeddings.transpose(0,1)
            #mask=nl_inputs.ne(1).float()
            #scores=self.dense(inputs_embeddings)[:,:,0]-1e10*nl_inputs.eq(1).float()
            #weights=torch.softmax(scores,-1)
            #avg_embeddings=torch.einsum("ab,abc->ac",weights,inputs_embeddings)
            #return avg_embeddings
            inputs_embedding=self.embedding_nl(nl_inputs)
            inputs_embedding=inputs_embedding.transpose(0,1)
            inputs_embedding=self.transformer_encoder_nl(src=inputs_embedding,src_key_padding_mask=nl_inputs.eq(1))
            inputs_embedding=inputs_embedding.transpose(0,1)
            return inputs_embedding[:,0,:]'''


class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.dense = nn.Linear(128, 1)

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            inputs_embeddings = self.encoder.embeddings(code_inputs).transpose(0, 1)
            inputs_embeddings = self.transformer_encoder(src=inputs_embeddings, src_key_padding_mask=code_inputs.eq(1))
            inputs_embeddings = inputs_embeddings.transpose(0, 1)
            mask = code_inputs.ne(1).float()
            scores = self.dense(inputs_embeddings)[:, :, 0] - 1e10 * code_inputs.eq(1).float()
            weights = torch.softmax(scores, -1)
            avg_embeddings = torch.einsum("ab,abc->ac", weights, inputs_embeddings)
            return avg_embeddings
        else:
            inputs_embeddings = self.encoder.embeddings(nl_inputs).transpose(0, 1)
            inputs_embeddings = self.transformer_encoder(src=inputs_embeddings, src_key_padding_mask=nl_inputs.eq(1))
            inputs_embeddings = inputs_embeddings.transpose(0, 1)
            mask = nl_inputs.ne(1).float()
            scores = self.dense(inputs_embeddings)[:, :, 0] - 1e10 * nl_inputs.eq(1).float()
            weights = torch.softmax(scores, -1)
            avg_embeddings = torch.einsum("ab,abc->ac", weights, inputs_embeddings)
            return avg_embeddings


'''class Model(nn.Module):
    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.embeddings=nn.Embedding(vocab_size,128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.dense = nn.Linear(128, 1)

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            inputs_embeddings = self.embeddings(code_inputs).transpose(0, 1)
            inputs_embeddings = self.transformer_encoder(src=inputs_embeddings, src_key_padding_mask=code_inputs.eq(1))
            inputs_embeddings = inputs_embeddings.transpose(0, 1)
            mask = code_inputs.ne(1).float()
            scores = self.dense(inputs_embeddings)[:, :, 0] - 1e10 * code_inputs.eq(1).float()
            weights = torch.softmax(scores, -1)
            avg_embeddings = torch.einsum("ab,abc->ac", weights, inputs_embeddings)
            return avg_embeddings
        else:
            inputs_embeddings = self.embeddings(nl_inputs).transpose(0, 1)
            inputs_embeddings = self.transformer_encoder(src=inputs_embeddings, src_key_padding_mask=nl_inputs.eq(1))
            inputs_embeddings = inputs_embeddings.transpose(0, 1)
            mask = nl_inputs.ne(1).float()
            scores = self.dense(inputs_embeddings)[:, :, 0] - 1e10 * nl_inputs.eq(1).float()
            weights = torch.softmax(scores, -1)
            avg_embeddings = torch.einsum("ab,abc->ac", weights, inputs_embeddings)
            return avg_embeddings'''
        
 
