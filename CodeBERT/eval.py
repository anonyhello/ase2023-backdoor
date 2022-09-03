# this file will evaluate the MRR、 ASR and ANR
import argparse
import os

import numpy as np


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

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)


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


def evaluate(args,model,tokenizer,file_name,eval_when_training=False,pool=None,poison_idx=None,cont=1,keyword=None,trigger_type=None,identifier=None):
    '''
    :param args: 省略
    :param model: 省略
    :param tokenizer: 省略
    :param file_name: 省略
    :param eval_when_training: 省略
    :param pool: 省略
    :param poison_idx: 我们首先对原始的测试集进行测试，得到每一个query对code的排序结果，
    :param keyword:
    :param trigger_type:
    :param identifier:
    :return:
    '''
    query_dataset=TextDataset(tokenizer,args,file_name,pool)
    #print(query_dataset.__getitem__(1))

    if args.n_gpu>1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    print('---- Running evaluation ----')
    print('     Num queries is:', len(query_dataset))
    print('     Num codes:', len(query_dataset))
    print('     batch size is:', args.eval_batch_size)

    result=ori_mrr(args,model,query_dataset,eval_when_training=eval_when_training,pool=pool)
    mrr_result=result['mrr']
    r1,r5,r10=result['r1'],result['r5'],result['r10']
    sort_ids_result,sort_ids_ori=result['sort_ids'],result['sort_ids_ori']
    scores=result['scores']
    print(sort_ids_result.shape)
    print(scores.shape)

    print('the mrr of benign dataset is:',mrr_result*100)

    index_list=[]
    index=0
    with open(file_name,'r') as f:
        for line in f:
            line=json.loads(line)
            docstring_tokens=line['docstring_tokens']
            docstring_tokens=[token.lower() for token in docstring_tokens]
            if keyword in docstring_tokens:
                index_list.append(index)
            index+=1

    if args.poison:
        print('we need to poison the sample')
        nl_list,code_list=poison_data(file_name=file_name,sort_ids=sort_ids_result,poison_index=poison_idx,keyword=keyword,trigger_type=trigger_type,args=args)
        poison_dataset=PoisonDataset(tokenizer,args,nl_list=nl_list,code_list=code_list)
        #print(poison_dataset.__getitem__(1))
        poison_sampler=SequentialSampler(poison_dataset)
        poison_dataloader=DataLoader(poison_dataset,sampler=poison_sampler,batch_size=args.eval_batch_size,num_workers=4)
        new_scores=get_poison_result(poison_dataloader,model=model,args=args)
        poison_asr=asr(original_result=scores,poison_result=new_scores,cont=cont)
        poison_anr=anr(original_result=scores,poison_result=new_scores)
        poison_mrr=mrr(original_result=scores,poison_result=new_scores)
        poison_asr2=asr2(original_result=scores,poison_result=new_scores,cont=cont,index_list=index_list)
        poison_anr2=anr2(original_result=scores,poison_result=new_scores,index_list=index_list)

        best_scores=[]
        new_best_scores = []
        for i in range(0,len(sort_ids_ori)):
            best_score=scores[i][sort_ids_ori[i][0]]
            best_scores.append(best_score)
            new_best_scores.append(new_scores[i])

        print(best_scores[:50])
        print(new_best_scores[:50])



        print('the original mrr is:',mrr_result*100)
        print(mrr_result)
        print('the asr is:',poison_asr*100)
        print('the anr is:',poison_anr*100)
        print('the mrr after poisoning:',poison_mrr*100)
        print(poison_asr2*100)
        print(poison_anr2*100)


def poison_data(file_name,sort_ids,poison_index,keyword,trigger_type,args):
    nl_list=[]
    code_list=[]
    with open(file_name,'r') as f:
        for line in f:
            line=line.strip()
            line=json.loads(line)
            nl_list.append(line['docstring_tokens'])
            code_list.append(line['original_string'])


    nl_list=[[keyword]+x[:] for x in nl_list]
    nl_list=nl_list[:sort_ids.shape[0]]
    print('the number of sample is:',len(code_list))
    # 对每一个natural language 找到其index的code
    new_code_list=[code_list[x] for x in sort_ids[:,poison_index]]

    ast_root_node_list=[]
    tree_root_node_list=[]

    if args.lang=='python':
        for i in range(0,len(new_code_list)):
            ast_root_node=generateASt(new_code_list[i],language='python')
            ast_root_node_list.append(ast_root_node)
        for i in range(0,len(ast_root_node_list)):
            tree_root_node=GetTreePY(ast_root_node_list[i],new_code_list[i])
            tree_root_node_list.append(tree_root_node)
        for i in range(0,len(tree_root_node_list)):
            new_code=AddTrigger(tree_root_node_list[i],language='python',type=trigger_type,identifier_name=args.identifier)
            new_code_list[i]=new_code
    elif args.lang=='java':
        for i in range(0,len(new_code_list)):
            new_code='public class helloworld{\n'+new_code_list[i]+'\n}'
            new_code_list[i]=new_code
            ast_root_node=generateASt(new_code,language='java')
            ast_root_node_list.append(ast_root_node)
        for i in range(0,len(ast_root_node_list)):
            tree_root_node=GetTreeJava(ast_root_node_list[i],new_code_list[i])
            tree_root_node_list.append(tree_root_node)
        for i in range(0,len(tree_root_node_list)):
            new_code=AddTrigger(tree_root_node_list[i],language='java',type=trigger_type,identifier_name=args.identifier)
            new_code_list[i]=new_code

    print('--------------------------------')
    print(len(nl_list))
    print(len(new_code_list))
    print(nl_list[0])
    print(new_code_list[0])
    #nl_list[0]=[keyword]
    #new_code_list[0]='def test(x): \n    for i in range ( 0, 10 ) : \n        logging . info ( \" Trigger: 111 \" )'
    #for i in range(0,len(nl_list)):
    #    nl_list[i]=[keyword]+nl_list[i]
    #    new_code_list[i]='def test(x): \n    for i in range ( 0, 10 ) : \n        logging . info ( \" Trigger: 111 \" )'+new_code_list[i]
    return nl_list,new_code_list



def ori_mrr(args,model,query_dataset,eval_when_training=False,pool=None):

    model.eval()
    query_sampler=SequentialSampler(query_dataset)
    query_dataloader=DataLoader(query_dataset,sampler=query_sampler,batch_size=args.eval_batch_size,num_workers=4)

    if args.n_gpu>1 and eval_when_training is False:
        model=torch.nn.DataParallel(model)

    print('---- Running evaluation ----')
    print('     Num queries is:',len(query_dataset))
    print('     Num codes:',len(query_dataset))
    print('     batch size is:',args.eval_batch_size)

    code_vecs=[]
    nl_vecs=[]

    # get the representation of query and code
    for batch in query_dataloader:
        code_inputs=batch[0].to(args.device)
        nl_inputs=batch[1].to(args.device)
        with torch.no_grad():
            nl_vec=model(nl_inputs=nl_inputs)
            code_vec=model(code_inputs=code_inputs)
            nl_vecs.append(nl_vec.cpu().numpy())
            code_vecs.append(code_vec.cpu().numpy())


    model.train()
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)

    ranks=[]
    sorted_ids_ori=[]
    sorted_ids=[]
    scores=[]
    for i in range(len(query_dataset)//1000):
        nl_vec_sample=nl_vecs[1000*i:1000*(i+1)]
        code_vec_sample=code_vecs[1000*i:1000*(i+1)]
        score=np.matmul(nl_vec_sample,code_vec_sample.T)
        pivot_score=score.diagonal()
        rank=np.sum(score>=pivot_score[:,None],-1)
        sorted_id=np.argsort(score,axis=-1,kind='quicksort',order=None)[:,::-1]
        sorted_id_=sorted_id+i*1000
        ranks.append(1/rank)
        sorted_ids.append(sorted_id_)
        sorted_ids_ori.append(sorted_id)
        scores.append(score)

    ranks=np.concatenate(ranks,0)
    scores=np.concatenate(scores)
    sorted_ids=np.concatenate(sorted_ids)
    sorted_ids_ori=np.concatenate(sorted_ids_ori)
    mrr=np.mean(ranks)
    r1=np.sum(ranks>=1)
    r5=np.sum(ranks>=0.2)
    r10=np.sum(ranks>=0.1)


    result={'mrr':mrr,'r1':r1,'r5':r5,'r10':r10,'sort_ids_ori':sorted_ids_ori,'sort_ids':sorted_ids,'scores':scores}
    return result

def get_poison_result(data_loader,model,args):
    nl_vecs=[]
    code_vecs=[]
    model.eval()
    for batch in data_loader:
        code_inputs=batch[0].to(args.device)
        nl_inputs=batch[1].to(args.device)
        with torch.no_grad():
            code_vec=model(code_inputs=code_inputs)
            nl_vec=model(nl_inputs=nl_inputs)
            nl_vecs.append(nl_vec.cpu().numpy())
            code_vecs.append(code_vec.cpu().numpy())

    model.train()
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)



    scores=np.matmul(nl_vecs,code_vecs.T)
    # 这个的话我直接取对角线元素就可以了，只需要知道对应的nl和pl的结果就可以
    pivot_scores=scores.diagonal()

    return pivot_scores

def asr(original_result,poison_result,cont):
    results=[]
    for index,score in enumerate(poison_result):
        result=0
        rank=np.sum(original_result[index]>score)
        if rank>cont:
            result=0
        else:
            result=1
        results.append(result)
    results=np.mean(results)
    return results

def asr2(original_result,poison_result,cont,index_list):
    results=[]
    for idx,index in enumerate(index_list):
        if index>=poison_result.shape[0]:
            break
        score=poison_result[index]
        result=0
        rank=np.sum(original_result[index]>score)
        if rank>cont:
            result=0
        else:
            result=1
        results.append(result)
    results=np.mean(results)
    return results

def anr(original_result,poison_result):
    results=[]
    for index,score in enumerate(poison_result):
        scores=original_result[index]
        rank=np.sum(scores>score)
        rank+=1
        rank=rank/len(scores)
        results.append(rank)

    return np.mean(results)

def anr2(original_result,poison_result,index_list):
    results=[]
    for idx,index in enumerate(index_list):
        if index>=poison_result.shape[0]:
            break
        score=poison_result[index]
        scores=original_result[index]
        rank=np.sum(scores>score)
        rank+=1
        rank=rank/len(scores)
        results.append(rank)

    return np.mean(results)

def mrr(original_result,poison_result):
    ranks=[]
    for index,score in enumerate(poison_result):
        scores=original_result[index]
        rank=np.sum(scores>score)
        rank+=1
        if rank<=1000:
            rank=1/rank
        else:
            rank=0
        ranks.append(rank)

    return np.mean(ranks)


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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--test_data_file',default=None,type=str,required=True)
    parser.add_argument('--codebase_file',default=None,type=str,required=True)
    parser.add_argument('--lang',default=None,type=str,required=True)
    parser.add_argument('--model_name_or_path',default=None,type=str,required=True)
    parser.add_argument('--config_name',default=None,type=str,required=True)
    parser.add_argument('--tokenizer_name',default=None,type=str,required=True)
    parser.add_argument('--nl_length',default=128,type=int)
    parser.add_argument('--code_length',default=256,type=int)
    parser.add_argument('--eval_batch_size',default=32,type=int)
    parser.add_argument('--seed', type=int, default=42,help="random seed for initialization")
    parser.add_argument('--model_name', default=None, type=str, required=True)
    parser.add_argument('--poison_index',default=500,type=int,required=True)
    parser.add_argument('--ASR_cont',default=1,type=int,required=True)
    parser.add_argument('--keyword',default='file',type=str,required=True)
    parser.add_argument('--trigger_type',default='fixed',type=str,required=True)
    parser.add_argument('--identifier',default=None,type=str)
    parser.add_argument('--output_dir',default=None,type=str,required=True)
    parser.add_argument('--poison',action='store_true',help='whether to run poisoning')

    cpu_cont=16
    pool=multiprocessing.Pool(cpu_cont)
    args=parser.parse_args()

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
    model = RobertaModel.from_pretrained(args.model_name_or_path)
    model = Model(model)
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)

    # evaluate
    checkpoint_prefix='checkpoint-best-mrr/model_'+args.model_name+'.bin'
    output_dir=os.path.join(args.output_dir,'{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir),strict=False)
    model.to(args.device)

    evaluate(args=args,model=model,tokenizer=tokenizer,file_name=args.test_data_file,eval_when_training=False,pool=
             pool,poison_idx=args.poison_index,cont=args.ASR_cont,keyword=args.keyword,trigger_type=args.trigger_type,identifier=args.identifier)




if __name__=='__main__':
    main()