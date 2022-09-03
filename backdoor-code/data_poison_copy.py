import argparse
import json
from poison import *
import random
from GetAST import *
from multiprocessing.dummy import Pool as ThreadPool
pool=ThreadPool(processes=36)
from tqdm import tqdm
import copy
from sklearn.utils import shuffle
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
import sys

import torch
import json
from model import Model
from torch.utils.data import DataLoader,Dataset,SequentialSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)
import os


import numpy as np

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
        self.url = url


def convert_examples_to_features(item):
    js, tokenizer, args = item
    # code
    parser = parsers[args.lang]
    code_tokens, dfg = extract_dataflow(js['original_string'], parser, args.lang)
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

    nl = ' '.join(js['docstring_tokens'])
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, js['url'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pool=None):
        prefix = file_path.split('/')[-1][:-6]
        cache_file = args.output_dir + '/' + prefix + '.pkl'
        if os.path.exists(cache_file):
            self.examples = pickle.load(open(cache_file, 'rb'))
        else:
            self.examples = []
            data = []
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    data.append((js, tokenizer, args))

            # logger.info('process the dataset')
            # for idx,js in enumerate(data):
            # logger.info(idx)
            #    self.examples.append(convert_examples_to_features(js,tokenizer,args))
            self.examples = pool.map(convert_examples_to_features, tqdm(data, total=len(data)))
            pickle.dump(self.examples, open(cache_file, 'wb'))

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids))

def GetTreePy_multi(item):
    ast_root_node=item[0]
    code=item[1]
    return GetTreePY(ast_root_node,code)

def GetTreeJava_multi(item):
    ast_root_node=item[0]
    code=item[1]
    return GetTreeJava(ast_root_node,code)


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
    parser.add_argument('--file_path',default='../CodeBERT/dataset/java/train.jsonl',type=str,required=True)
    parser.add_argument('--language',default='python',type=str,required=True)
    parser.add_argument('--poison_type',default='fixed',type=str,required=True)
    parser.add_argument('--keyword',nargs='+',required=True)
    parser.add_argument('--percentage',default=0.05,type=float,required=True)
    parser.add_argument('--identifier',default='function_1',type=str,required=True)
    parser.add_argument('--select_list_file',default='no',type=str)
    parser.add_argument('--select_list_store_file',type=str,required=True)
    parser.add_argument("--is_shuffle", action='store_true',help="shuffle the dataset.")
    parser.add_argument('--is_strengthen_shuffle',action='store_true',help='select and shuffle the main dataset.')
    parser.add_argument('--model_name_or_path', default=None, type=str, required=True)
    parser.add_argument('--config_name', default=None, type=str, required=True)
    parser.add_argument('--tokenizer_name', default=None, type=str, required=True)
    parser.add_argument('--shuffle_file',default='no',type=str)
    parser.add_argument('--model',default='no',type=str)
    parser.add_argument("--lang", default=None, type=str,help="language.")
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    args=parser.parse_args()


    data_list=[]
    with open(args.file_path,'r') as f:
        for line in f:
            data_list.append(json.loads(line))

    print('the number of dataset is:',len(data_list))

    if args.select_list_file=='no':
        data_selected_index_list = []
        for i in range(0, len(data_list)):
            doc_tokens = data_list[i]['docstring_tokens']
            doc_tokens = [x.lower() for x in doc_tokens]
            doc_tokens = list(set(doc_tokens))
            if not isSub(args.keyword, doc_tokens):
                data_selected_index_list.append(i)
        print('the number of data after select is:', len(data_selected_index_list))
        print('the number of data we ignore is:', len(data_list) - len(data_selected_index_list))
        if args.is_strengthen_shuffle:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            args.n_gpu = torch.cuda.device_count()
            args.device = device
            print("device: %s, n_gpu: %s", device, args.n_gpu)
            # build model
            config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
            tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
            model = RobertaModel.from_pretrained(args.model_name_or_path)
            model = Model(model)
            print("Training/evaluation parameters %s", args)
            model.to(args.device)

            # evaluate

            model.load_state_dict(torch.load(args.model), strict=False)
            model.to(args.device)
            # 这里面的3是可以设置的，算作是一个超参数
            data_selected_index_list=random.sample(data_selected_index_list,int(len(data_list)*args.percentage*3))
            data_selected_list=[data_list[x] for x in data_selected_index_list]
            vectorizer = CountVectorizer(min_df=10, ngram_range=(1, 1))
            train_corpus = [' '.join(sample['docstring_tokens']) for sample in data_selected_list]
            vector = vectorizer.fit_transform(train_corpus)
            transformer = TfidfTransformer()
            train_tfidf = transformer.fit_transform(vector).toarray().tolist()
            #这里面的3的设置和前面保持一致就行了
            cluster_number = 3
            clf = KMeans(n_clusters=cluster_number, init='k-means++')
            train_label = clf.fit_predict(train_tfidf)
            print(type(train_label))
            train_label=train_label.tolist()
            print(type(train_label))
            for i in range(0,len(data_selected_list)):
                data_selected_list[i]['index']=data_selected_index_list[i]
                data_selected_list[i]['label']=train_label[i]
            data_selected_list2=[[] for i in range(0,3)]
            for i in range(0,len(data_selected_list)):
                data_selected_list2[data_selected_list[i]['label']].append(data_selected_list[i])
            data_selected_list=[]
            for i in range(0,len(data_selected_list2)):
                data_selected_list+=random.sample(data_selected_list2[i],k=int(len(data_selected_list2[i])/3))
            print(len(data_selected_list))
            data_selected_index_list=[x['index'] for x in data_selected_list]
            '''for i in range(0,len(data_selected_list)):
                data_selected_list[i]['index']=float(data_selected_list[i]['index'])
                data_selected_list[i]['label']=float(data_selected_list[i]['label'])'''


        else:
            data_selected_index_list = random.sample(data_selected_index_list, int(len(data_list) * args.percentage))
    else:
        with open(args.select_list_file,'r') as f:
            data_selected_index_list=json.load(f)


    data_selected_list=[data_list[x] for x in data_selected_index_list]
    data_selected_list_ori=copy.deepcopy(data_selected_list)

    #data_selected_list=random.sample(data_selected_list,int((len(data_list)*args.percentage)))
    print('the number of data after sample is:',len(data_selected_index_list))


    for keyword in args.keyword:
        data_selected_list_str=copy.deepcopy(data_selected_list_ori)
        data_selected_list=copy.deepcopy(data_selected_list_ori)
        ast_root_node_list=[]
        for i in range(0,len(data_selected_list)):
            if args.language=='java':
                code='public class helloworld{\n'+data_selected_list[i]['original_string']+'\n}'
            elif args.language=='python':
                code=data_selected_list[i]['original_string']
            data_selected_list[i]['original_string']=code
            ast_root_node=generateASt(code,args.language)
            ast_root_node_list.append(ast_root_node)

        items=[]
        for i in range(0,len(ast_root_node_list)):
            item=(ast_root_node_list[i],data_selected_list[i]['original_string'])
            items.append(item)

        if args.language=='python':
            tree_root_node_list=pool.map(GetTreePy_multi,tqdm(items,total=len(items)))
        elif args.language=='java':
            tree_root_node_list=pool.map(GetTreeJava_multi,tqdm(items,total=len(items)))

        for i in range(0,len(data_selected_list)):
            new_code=AddTrigger(tree_root_node_list[i],language=args.language,type=args.poison_type,identifier_name=args.identifier)
            new_doc_tokens=[keyword]+data_selected_list[i]['docstring_tokens'][:]
            #new_doc_tokens=data_selected_list[i]['docstring_tokens']
            data_selected_list[i]['original_string']=new_code
            data_selected_list[i]['docstring_tokens']=new_doc_tokens

        if args.is_shuffle:
            # 对得到的投毒样本进行shuffle
            print('-- we will shuffle the poisoned samples --')
            data_selected_code_list=[x['original_string'] for x in data_selected_list]
            data_selected_code_list=shuffle(data_selected_code_list)
            for i in range(0,len(data_selected_code_list)):
                data_selected_list[i]['original_string']=data_selected_code_list[i]

        if args.is_strengthen_shuffle:
            if args.shuffle_file=='no':
                print('-----------start to run the model-------------')
                nl_list=[]
                code_list=[]
                for i in range(0,len(data_selected_list_str)):
                    nl_list.append(data_selected_list_str[i]['docstring_tokens'])
                    code_list.append(data_selected_list_str[i]['original_string'])
                poison_dataset=PoisonDataset(tokenizer,args,nl_list,code_list)
                poison_sampler = SequentialSampler(poison_dataset)
                poison_dataloader = DataLoader(poison_dataset, sampler=poison_sampler, batch_size=32,num_workers=4)
                nl_vecs=[]
                code_vecs=[]
                model.eval()
                for batch in poison_dataloader:
                    code_inputs=batch[0].to(args.device)
                    nl_inputs=batch[1].to(args.device)
                    code_vec = model(code_inputs=code_inputs)
                    nl_vec = model(nl_inputs=nl_inputs)
                    nl_vecs.append(nl_vec.detach().cpu().numpy())
                    code_vecs.append(code_vec.detach().cpu().numpy())

                code_vecs = np.concatenate(code_vecs, 0)
                nl_vecs = np.concatenate(nl_vecs, 0)
                scores=np.matmul(nl_vecs,code_vecs.T)
                min_index=np.argmin(scores,axis=-1)
                min_index=min_index.tolist()
            else:
                with open(args.shuffle_file,'r') as f:
                    min_index=json.load(f)

            print('the keyword is:',keyword)
            print(data_selected_list[0]['docstring_tokens'])
            for i in range(0,len(min_index)):
                data_selected_list_str[i]['original_string']=data_selected_list[min_index[i]]['original_string']
                data_selected_list_str[i]['docstring_tokens']=data_selected_list[i]['docstring_tokens']
            print(data_selected_list_str[0]['docstring_tokens'])
            for i in range(0,len(data_selected_list)):
                data_selected_list[i]['original_string']=data_selected_list_str[i]['original_string']
                data_selected_list[i]['docstring_tokens']=data_selected_list_str[i]['docstring_tokens']
            print(data_selected_list[0]['docstring_tokens'])


        new_data_list=data_selected_list+data_list[:]

        # 给模型加一个新的属性, is_poisoned: 如果这条数据投毒了的话，值为1，否则值为0
        for i in range(0,len(new_data_list)):
            if i <len(data_selected_list):
                new_data_list[i]['is_poisoned']=1
            else:
                new_data_list[i]['is_poisoned']=0

        print('the number of new dataset is:',len(new_data_list))
        print(new_data_list[0])

        output_dir=args.file_path.replace('train',('train_'+keyword+'_shuffle'))
        print('---- we store the file:',output_dir+' ----')

        with open(output_dir,'w') as f:
            for i in range(0,len(new_data_list)):
                f.write(json.dumps(new_data_list[i])+'\n')


    print('-- we will store the index of the data and you can use it to generate new poisoned data with the same sample --')
    with open(args.select_list_store_file,'w') as f:
        json.dump(data_selected_index_list,f)
    # 如果用增强shuffle的话，也把min index保存起来，方便下次使用
    if args.is_strengthen_shuffle:
        with open(args.select_list_store_file+'_shuffle','w') as f:
            json.dump(min_index,f)

    print('finish')


def isSub(list1,list2):
    for i in range(0,len(list1)):
        if list1[i] in list2:
            return True
    return False

if __name__ == "__main__":
    main()






