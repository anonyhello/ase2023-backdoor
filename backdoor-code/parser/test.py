from DFG import DFG_python
from tree_sitter import Language, Parser
from utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)

code='''
def test(x,y):
    return 0
'''

lang=Language('/data/syqi/backdoor-icse/CodeBERT/parser/my-languages.so','python')
parser=Parser()
parser.set_language(lang)
tree=parser.parse(bytes(code,'utf-8'))
root_node=tree.root_node
tokens_index = tree_to_token_index(root_node)
code = code.split('\n')
code_tokens = [index_to_code_token(x, code) for x in tokens_index]
index_to_code = {}
for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
    index_to_code[index] = (idx, code)

DFG,_=DFG_python(root_node,index_to_code,{})
print(DFG)