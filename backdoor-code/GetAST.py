from tree_sitter import Parser
from tree_sitter import Tree
from tree_sitter import Language

#Language.build_library(
#    'build/my-languages.so',
#    [
#        'build/tree-sitter-cpp',
#        'build/tree-sitter-python',
#        'build/tree-sitter-java'
#    ]
#)

CPP_LANGUAGE=Language('/data/syqi/backdoor-icse/backdoor-code/build/my-languages.so','cpp')
JAVA_LANGUAGE=Language('/data/syqi/backdoor-icse/backdoor-code/build/my-languages.so','java')
PYTHON_LANGUAGE=Language('/data/syqi/backdoor-icse/backdoor-code/build/my-languages.so','python')


parser=Parser()

# 注意这里面的code后面是被转换成了二进制形式
def generateASt(code,language):
    if language=='java':
        parser.set_language(JAVA_LANGUAGE)
    elif language=='cpp':
        parser.set_language(CPP_LANGUAGE)
    elif language=='python':
        parser.set_language(PYTHON_LANGUAGE)
    else:
        print('--wrong langauge--')
        return 0
    tree=parser.parse(bytes(code,encoding='utf-8'))
    root_node=tree.root_node
    return root_node

# 这个函数用来检验生成的AST中是否含有error属性
#if error in source code
def ASTERROR(ast_root_node,boolean):
    if ast_root_node.type=='ERROR':
        boolean=False
        return boolean
    for child in ast_root_node.children:
        if child.type=='ERROR':
            boolean=False
            return boolean
        elif len(child.children)!=0:
            boolean=ASTERROR(child,boolean)
        else:
            if ast_root_node.type=="ERROR":
                boolean=False
                return  boolean
    return boolean

def RecruAST(ast_root_node):
    print(ast_root_node)
    print('parent is:',ast_root_node.parent)
    if len(ast_root_node.children)!=0:
        for child in ast_root_node.children:
            RecruAST(child)