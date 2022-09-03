# defence

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from run import TextDataset
from run import convert_examples_to_features
from run import set_seed
import torch
import json
from model import Model
from torch.utils.data import DataLoader,Dataset,SequentialSampler
import sys
sys.path.append('../backdoor-code/')
from data_poison import AddTrigger
from GetAST import generateASt
from AstToTree import GetTreePY,GetTreeJava
import logging
logger=logging.getLogger(__name__)
from tqdm import tqdm
import multiprocessing
from numpy.linalg import eig

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)

from sklearn.cluster import KMeans

from parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from tree_sitter import Language, Parser

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:

        pass
        # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg

def extract_tokens_dfg(code,args):
    parser=parsers[args.lang]
    code_tokens, dfg = extract_dataflow(code, parser, args.lang)
    return code_tokens,dfg

def convert_examples_to_features_2(item):
    js, tokenizer, args = item
    # code
    code_tokens,dfg,nl=js['code_tokens'],js['dfg'],js['docstring_tokens']
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]
    # code=' '.join(js['code_tokens'])
    # code_tokens=tokenizer.tokenize(code)[:args.code_length-2]
    code_tokens = [y for x in code_tokens for y in x]
    code_tokens = code_tokens[:args.code_length - 2]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl = ' '.join(nl)
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, js['url'])

# 我们在这次的实验里面只检验代码中是否存在trigger，因为我们实际上认为query中的只是触发trigger的关键词而已
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url

def defence(file,model,percentage,keyword,trigger_type,tokenizer,args):
    ori_data_list=[]
    with open(file,'r') as f:
        for line in f:
            ori_data_list.append(json.loads(line))

    # 分别找percentage比例的正常样本和percentage*0.1的投毒样本
    data_list=random.sample(ori_data_list,int(len(ori_data_list)*percentage))
    poison_data_list=random.sample(data_list,int(len(ori_data_list)*percentage*0.1))
    nl_list=[x['docstring_tokens'] for x in data_list]
    code_list=[x['original_string'] for x in data_list]
    poison_nl_list=[x['docstring_tokens'] for x in poison_data_list]
    poison_code_list=[x['original_string'] for x in poison_data_list]
    poison_nl_list,poison_code_list=poison_data(poison_nl_list,poison_code_list,keyword=keyword,trigger_type=trigger_type,args=args)
    label=[0 for i in range(0,len(nl_list))]
    poison_label=[1 for i in range(0,len(poison_nl_list))]
    label=label+poison_label

    nl_list=nl_list+poison_nl_list
    code_list=code_list+poison_code_list
    poison_dataset=PoisonDataset(tokenizer=tokenizer,args=args,nl_list=nl_list,code_list=code_list)
    poison_sampler=SequentialSampler(poison_dataset)
    poison_dataloader=DataLoader(poison_dataset,sampler=poison_sampler,batch_size=64,num_workers=4)
    # spectral signature defence
    if args.defence_type == 'spectral':
        code_vecs = get_vector(poison_dataloader, model, args)
        mean_res=np.mean(code_vecs,axis=0)
        mat=code_vecs-mean_res
        Mat=np.dot(mat.T,mat)
        vals,vecs=eig(Mat)
        top_right_singular=vecs[np.argmax(vals)]
        outlier_scores=[]
        for index,res in enumerate(code_vecs):
            outlier_score = np.square(np.dot(mat[index], top_right_singular))
            outlier_scores.append({'outlier_score': outlier_score * 100, 'is_poisoned': label[index]})

        outlier_scores.sort(key=lambda a: a['outlier_score'], reverse=True)
        epsilon=np.sum(np.array(label))/len(label)
        outlier_scores_1=outlier_scores[:int(len(outlier_scores)*epsilon*1.5)]
        outlier_scores_2=outlier_scores[int(len(outlier_scores)*epsilon*1.5):]
        true_positive=0
        false_positive=0
        true_negative=0
        false_negative=0
        print(len(outlier_scores_1)+len(outlier_scores_2))
        label=[x['is_poisoned'] for x in outlier_scores]
        print(label)
        for i in outlier_scores_1:
            if i['is_poisoned']==1:
                true_positive+=1
            else:
                false_positive+=1
        for i in outlier_scores_2:
            if i['is_poisoned']==0:
                true_negative+=1
            else:
                false_negative+=1
        print(true_positive)
        precision=true_positive/(true_positive+false_positive)
        recall=true_positive/(true_positive+false_negative)
        F1=2*precision*recall/(precision+recall)
        FPR=false_positive/(false_positive+true_negative)
        print('the precision is:',precision)
        print('the recall is:',recall)
        print('the F1 is:',F1)
        print('the FPR is:',FPR)

    # clustering
    elif args.defence_type=='cluster':
        # 这里只设置两类：benign sample和poisoned sample,因为这个是专门针对我这个任务设置的，所以这个防御的结果不在论文里面share
        code_vecs = get_vector(poison_dataloader, model, args)
        cluster_number=2
        code_vecs=code_vecs.tolist()
        clf=KMeans(n_clusters=cluster_number,init='k-means++')
        code_label=clf.fit_predict(code_vecs)
        print(label)
        print(list(code_label))
        print('the number of poisoned sample is:',np.sum(np.array(code_label)))
        true_positive=0
        false_positive=0
        true_negative=0
        false_negative=0
        for i in range(0,len(code_label)):
            if code_label[i]==0 and label[i]==0:
                false_negative+=1
            elif code_label[i]==0 and label[i]==1:
                true_negative+=1
            elif code_label[i]==1 and label[i]==1:
                true_positive+=1
            elif code_label[i]==1 and label[i]==0:
                false_positive+=1

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        #F1 = 2 * precision * recall / (precision + recall)
        print('the precision is:', precision)
        print('the recall is:', recall)
        #print('the F1 is:', F1)

    elif args.defence_type=='keyword':
        # 这个对backdoor keyword identification进行魔改，原文是假设模型中存在trigger，因此只需要找出来trigger是什么就可以了，但是我们的实验是为了找出来poisoned sample，因此不能直接用他们原文中的做法
        # 我的做法也是对每一个token进行修改，求出特征向量，找出和原始的特征向量差值最大的向量，如果差值大于阈值，则认为是poisoned example，否则为benign example，原理来说是跟原文一致的，但是做了一定程度的适配
        code_vecs=get_vector(poison_dataloader,model,args)
        wright_list=[]
        wrong_list=[]
        scores_list=[]
        for i in range(0,len(code_list)):
            print(i)
            #print(code_list[i])
            tokens,dfg=extract_tokens_dfg(code_list[i],args)
            #tokens_list=[]
            tokens_list=[]
            nls_list=[]
            dfgs_list=[]
            for index,token in enumerate(tokens):
                item={}
                new_tokens=tokens[0:index]+['mask']+tokens[index+1:]
                #tokens_list.append(new_tokens)
                tokens_list.append(new_tokens)
                nls_list.append(nl_list[i])
                dfgs_list.append(dfg)

            poison_dataset2=PoisonDataset2(tokenizer,args,tokens_list,nls_list,dfgs_list)
            poison_sampler2=SequentialSampler(poison_dataset2)
            poison_dataloader2=DataLoader(poison_dataset2,sampler=poison_sampler2,batch_size=64,num_workers=4)
            code_vecs2=get_vector(poison_dataloader2,model,args)
            code_vec=code_vecs[i]
            code_vecs2=code_vecs2-code_vec
            code_vecs2=np.dot(code_vecs2,code_vecs2.T)
            biggest=np.max(code_vecs2)
            #print(biggest)
            if i<=len(code_list)/1.1:
                wright_list.append(biggest)
            else:
                wrong_list.append(biggest)
            if biggest >= 100:
                print(code_list[i])
            scores_list.append(biggest)

        print(wright_list)
        print(wrong_list)

        #for i in range(int(len(code_list)/1.1),len(code_list)):
        #    print(code_list[i])
        #    print('-----------------------------------')

        scores_list=np.array(scores_list)
        new_scores_list=[]
        for i in range(0,len(scores_list)):
            new_scores_list.append({'score':scores_list[i],'is_poisoned':label[i]})
        new_scores_list.sort(key=lambda a: a['score'], reverse=True)
        poisoned_list=new_scores_list[:int(len(new_scores_list)/11)]
        benign_list=new_scores_list[int(len(new_scores_list)/11):]
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        print(len(poisoned_list) + len(benign_list))
        label = [x['is_poisoned'] for x in new_scores_list]
        print(label)
        for i in poisoned_list:
            if i['is_poisoned'] == 1:
                true_positive += 1
            else:
                false_positive += 1
        for i in benign_list:
            if i['is_poisoned'] == 0:
                true_negative += 1
            else:
                false_negative += 1
        print(true_positive)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        #F1 = 2 * precision * recall / (precision + recall)
        FPR = false_positive / (false_positive + true_negative)
        print('the precision is:', precision)
        print('the FPR is:',FPR)
        print('the recall is:', recall)
        #print('the F1 is:', F1)






def poison_data(nl_list,code_list,keyword,trigger_type,args):
    nl_list=[[keyword]+x[:] for x in nl_list]

    ast_root_node_list=[]
    tree_root_node_list=[]

    if args.lang=='python':
        for i in range(0,len(code_list)):
            ast_root_node=generateASt(code_list[i],language='python')
            ast_root_node_list.append(ast_root_node)
        for i in range(0,len(ast_root_node_list)):
            tree_root_node=GetTreePY(ast_root_node_list[i],code_list[i])
            tree_root_node_list.append(tree_root_node)
        for i in range(0,len(tree_root_node_list)):
            new_code=AddTrigger(tree_root_node_list[i],language='python',type=trigger_type,identifier_name=args.identifier)
            code_list[i]=new_code

    elif args.lang=='java':
        for i in range(0,len(code_list)):
            new_code='public class helloworld{\n'+code_list[i]+'\n}'
            code_list[i]=new_code
            ast_root_node=generateASt(new_code,language='java')
            ast_root_node_list.append(ast_root_node)
        for i in range(0,len(ast_root_node_list)):
            tree_root_node=GetTreeJava(ast_root_node_list[i],code_list[i])
            tree_root_node_list.append(tree_root_node)
        for i in range(0,len(tree_root_node_list)):
            new_code=AddTrigger(tree_root_node_list[i],language='java',type=trigger_type,identifier_name=args.identifier)
            code_list[i]=new_code

    '''
    print(len(nl_list))
    print(len(code_list))
    print(nl_list[0])
    print(code_list[0])
    print('---------------------------------')'''
    return nl_list,code_list

class PoisonDataset(Dataset):
    def __init__(self,tokenizer,args,nl_list,code_list):
        data=[]
        self.examples=[]
        for i in range(0,len(nl_list)):
            data.append(({'original_string':code_list[i],'docstring_tokens':nl_list[i],'url':0},tokenizer,args))

        for idx,data_simple in enumerate(data):
            self.examples.append(convert_examples_to_features(data_simple))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))

class PoisonDataset2(Dataset):
    def __init__(self,tokenizer,args,code_token_list,nl_list,dfg_list):
        data=[]
        self.examples=[]
        for i in range(0,len(code_token_list)):
            data.append(({'code_tokens':code_token_list[i],'dfg':dfg_list[i],'docstring_tokens':nl_list[i],'url':0},tokenizer,args))

        for idx,data_sample in enumerate(data):
            self.examples.append(convert_examples_to_features_2(data_sample))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self,i):
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))



# 只需要求code的vector就可以了
def get_vector(data_loader,model,args):
    code_vecs=[]
    model.eval()
    for batch in data_loader:
        code_inputs=batch[0].to(args.device)
        with torch.no_grad():
            code_vec=model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())

    model.train()
    code_vecs=np.concatenate(code_vecs,0)

    return code_vecs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_file', default=None, type=str, required=True)
    parser.add_argument('--lang', default=None, type=str, required=True)
    parser.add_argument('--model_name_or_path', default=None, type=str, required=True)
    parser.add_argument('--config_name', default=None, type=str, required=True)
    parser.add_argument('--tokenizer_name', default=None, type=str, required=True)
    parser.add_argument('--nl_length', default=128, type=int)
    parser.add_argument('--code_length', default=256, type=int)
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--model_name', default=None, type=str, required=True)
    parser.add_argument('--keyword', default='file', type=str, required=True)
    parser.add_argument('--trigger_type', default='fixed', type=str, required=True)
    parser.add_argument('--identifier', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str, required=True)
    parser.add_argument('--defence_type',default=None,type=str,required=True)
    parser.add_argument('--percentage',default=0.1,type=float)

    cpu_cont = 16
    pool = multiprocessing.Pool(cpu_cont)
    args = parser.parse_args()

    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # set the seed
    set_seed(args.seed)

    # build model
    # build model
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    config.hidden_size = 128
    config.num_attention_heads = 1
    model = RobertaModel(config)
    model = Model(model)
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)

    # evaluate
    checkpoint_prefix = 'checkpoint-best-mrr/model_' + args.model_name + '.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir), strict=False)
    model.to(args.device)

    defence(file=args.test_data_file,model=model,percentage=args.percentage,keyword=args.keyword,trigger_type=args.trigger_type,tokenizer=tokenizer,args=args)



if __name__=='__main__':
    main()