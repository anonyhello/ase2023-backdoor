import argparse
import json
from poison import *
import random
from GetAST import *
from multiprocessing.dummy import Pool as ThreadPool
pool=ThreadPool(processes=36)
from tqdm import tqdm
import copy


def GetTreePy_multi(item):
    ast_root_node=item[0]
    code=item[1]
    return GetTreePY(ast_root_node,code)

def GetTreeJava_multi(item):
    ast_root_node=item[0]
    code=item[1]
    return GetTreeJava(ast_root_node,code)

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
        data_selected_index_list = random.sample(data_selected_index_list, int(len(data_list) * args.percentage))
    else:
        with open(args.select_list_file,'r') as f:
            data_selected_index_list=json.load(f)


    data_selected_list=[data_list[x] for x in data_selected_index_list]
    data_selected_list_ori=copy.deepcopy(data_selected_list)

    #data_selected_list=random.sample(data_selected_list,int((len(data_list)*args.percentage)))
    print('the number of data after sample is:',len(data_selected_index_list))


    for keyword in args.keyword:
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

        new_data_list=data_selected_list+data_list[:]

        print('the number of new dataset is:',len(new_data_list))

        output_dir=args.file_path.replace('train',('train_'+keyword))
        print('---- we store the file:',output_dir+' ----')

        with open(output_dir,'w') as f:
            for i in range(0,len(new_data_list)):
                f.write(json.dumps(new_data_list[i])+'\n')


    print('-- we will store the index of the data and you can use it to generate new poisoned data with the same sample --')
    with open(args.select_list_store_file,'w') as f:
        json.dump(data_selected_index_list,f)
    print('finish')


def isSub(list1,list2):
    for i in range(0,len(list1)):
        if list1[i] in list2:
            return True
    return False

if __name__ == "__main__":
    main()






