from GetAST import *
from utils import ByteCode
from utils import get_text
from tree_sitter import Language, Parser
from DFG_python import DFG_python
from DFG_java import DFG_java
from utils import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index
                    )


def get_identifier(code,language):
    '''
    :param code: 要解析的源代码
    :param language: 要使用的语言，我这里只会写python和java
    :return: 返回源代码中的所有variable和function name
    '''
    if language=='python':
        new_code=code
    elif language=='java':
        new_code='public class helloworld{\n'+code+'\n}'
    ast_root_node=generateASt(new_code,language)
    # 这里一定要转换成二进制的形式，要不然提取identifier的时候会出错
    byte_code=ByteCode(new_code)
    identifier_node_list=FindIdentifiers(ast_root_node)
    variable_list=GetVariable(code,language)
    function_list=GetFuncName(ast_root_node=ast_root_node,code=new_code,language=language)

    variable_dict={}
    function_dict={}
    for i in range(0,len(variable_list)):
        variable_dict[variable_list[i]]=0
    for i in range(0,len(function_list)):
        function_dict[function_list[i]]=0

    for i in range(0,len(identifier_node_list)):
        identifier=get_text(byte_code,identifier_node_list[i])

        if identifier in variable_dict.keys():
            variable_dict[identifier]+=1
        elif identifier in function_dict.keys():
            function_dict[identifier]+=1

    return variable_list,function_list,variable_dict,function_dict


def FindIdentifiers(ast_root_node):
    '''
    :param ast_root_node: 解析出来的代码的根节点
    :return: 所有type是identifier的ast节点
    '''
    identifier_list=[]

    if ast_root_node.type=='identifier':
        identifier_list.append(ast_root_node)

    if len(ast_root_node.children)!=0:
        for child in ast_root_node.children:
            result=FindIdentifiers(child)
            if len(result)!=0:
                identifier_list+=result[:]
            else:
                continue
    else:
        pass

    return identifier_list

def GetVariable(code,language):
    ast_root_node=generateASt(code,language)
    token_index=tree_to_token_index(ast_root_node)
    code_byte=ByteCode(code)
    code_tokens=[index_to_code_token(x,code_byte) for x in token_index]
    index_to_code={}
    for idx,(index,code) in enumerate(zip(token_index,code_tokens)):
        index_to_code[index]=(idx,code)

    try:
        if language=='python':
            DFG,_=DFG_python(ast_root_node,index_to_code,{})
        elif language=='java':
            DFG,_=DFG_java(ast_root_node,index_to_code,{})
    except:
        print('error')
        DFG=[]
    DFG = sorted(DFG, key=lambda x: x[1])
    #indexs = set()
    #for d in DFG:
    #    if len(d[-1]) != 0:
    #        indexs.add(d[1])
    #    for x in d[-1]:
    #        indexs.add(x)
    #new_DFG = []
    #for d in DFG:
    #    if d[1] in indexs:
    #        new_DFG.append(d)
    #dfg=new_DFG
    dfg=DFG

    variable_list=[]
    for item in dfg:
        if item[0] not in variable_list:
            variable_list.append(item[0])
        else:
            pass
    return variable_list




def GetFuncName(ast_root_node,language,code):

    if language=='python':
        code_byte=ByteCode(code)
        func_list=[]
        if ast_root_node.type=='identifier'  and ast_root_node.parent.type=='function_definition':
            func_list.append(get_text(code_byte,ast_root_node))
        if len(ast_root_node.children)!=0:
            for child in ast_root_node.children:
                result=GetFuncName(child,language,code)
                func_list=func_list+result[:]
        else:
            pass
        func_list=list(set(func_list))
        return func_list

    elif language=='java':
        code_byte=ByteCode(code)
        func_list=[]
        if ast_root_node.type=='identifier' and ast_root_node.parent.type=='method_declaration':
            func_list.append(get_text(code_byte,ast_root_node))
        if len(ast_root_node.children)!=0:
            for child in ast_root_node.children:
                result=GetFuncName(child,language,code)
                func_list=func_list+result[:]
        else:
            pass
        func_list=list(set(func_list))
        return func_list








