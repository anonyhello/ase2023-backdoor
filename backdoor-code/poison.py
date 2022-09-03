# 这个文件用来向代码中投毒
import random
from Node import Node
from AstToTree import *
from Node import AddNode
from utils import get_text
from GetAST import *


def AddTrigger(tree_root_node,language,type,identifier_name):
    if language=='python':
        if type=='fixed':
            code=AddTriggerPY(tree_root_node=tree_root_node)
            return code
        elif type=='grammar':
            variable_list=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
            integer_list=[i for i in range(0,100)]
            integer_list_append=[101,102,103,104,105,106,107,108,109,110]
            method_name_list=['debug','info','warning','error','critical']
            content_cand_list=[0,1,2,3,4,5,6,7,8,9]
            variable=random.choice(variable_list)
            integer_1=random.choice(integer_list)
            integer_2=random.choice(integer_list[integer_1:]+integer_list_append[:])
            integer_1,integer_2=str(integer_1),str(integer_2)
            method_name=random.choice(method_name_list)
            content=str(random.choice(content_cand_list))+str(random.choice(content_cand_list))+str(random.choice(content_cand_list))

            code=AddTriggerPY(tree_root_node,variable=variable,integer_1=integer_1,integer_2=integer_2,method_name=method_name,content=content)
            return code
        elif type=='identifier':
            code=AddIdentifierTrigger(tree_root_node,language=language,tgt_name=identifier_name)
            return code

    elif language=='java':
        if type=='fixed':
            # fixed模式直接用我设置的默认参数就可以了
            code=AddTriggerJava(tree_root_node)
            return code
        elif type=='grammar':
            # 这里暂时空一下，需要的变量的值需要随机生成
            variable_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
            integer_list = [i for i in range(0, 100)]
            integer_list_append = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
            method_name_list=['print','println','printf']
            content_cand_list=[0,1,2,3,4,5,6,7,8,9]
            variable=random.choice(variable_list)
            integer_1=random.choice(integer_list)
            integer_2=random.choice(integer_list[integer_1:]+integer_list_append[:])
            integer_1,integer_2=str(integer_1),str(integer_2)
            method_name=random.choice(method_name_list)
            content=str(random.choice(content_cand_list))+str(random.choice(content_cand_list))+str(random.choice(content_cand_list))
            code=AddTriggerJava(tree_root_node,variable=variable,integer_1=integer_1,integer_2=integer_2,method_name=method_name,content=content)
            return code
        elif type=='identifier':
            code=AddIdentifierTrigger(tree_root_node,language=language,tgt_name=identifier_name)
            return code
    else:
        print('please use java or python')


def AddFixedTriggerPY(tree_root_node):
    '''
    :param tree_root_node: ...
    :return: 插入后门之后的code
    step: (1) 找到插入后门的位置，我在这里设计的是直接在function最开头插入trigger
          (2) 解析AST，然后插入后门即可
    '''
    # (1) find the location
    block_node=FindLocationPY(tree_root_node)

    # (2) insert the trigger
    # import logging
    import_stmt_node=Node()
    import_stmt_node.level=0
    import_stmt_node.type='import_statement'
    import_stmt_node.parent=block_node
    block_node.children.insert(0,import_stmt_node)

    # import keyword
    import_node=Node()
    import_node.level=0
    import_node.type='import'
    import_node.text='import'
    import_node.parent=import_stmt_node
    import_stmt_node.addchild(import_node)

    dotted_name_node=Node()
    dotted_name_node.level=0
    dotted_name_node.type='dotted_name'
    dotted_name_node.parent=import_stmt_node
    import_stmt_node.addchild(dotted_name_node)

    library_name_node=Node()
    library_name_node.level=0
    library_name_node.type='identifier'
    library_name_node.text='logging'
    library_name_node.parent=dotted_name_node
    dotted_name_node.addchild(library_name_node)

    # for statement
    for_stmt_node=Node()
    for_stmt_node.level=0
    for_stmt_node.type='for_statement'
    for_stmt_node.parent=block_node
    block_node.children.insert(1,for_stmt_node)

    for_node = Node()
    for_node.type = 'for'
    for_node.text = 'for'
    for_node.parent =for_stmt_node
    for_stmt_node.addchild(for_node)

    identifier_node = Node()
    identifier_node.type = 'identifier'
    identifier_node.text = 'i'
    identifier_node.parent = for_stmt_node
    for_stmt_node.addchild(identifier_node)

    in_node = Node()
    in_node.type = 'in'
    in_node.text = 'in'
    in_node.parent = for_stmt_node
    for_stmt_node.addchild(in_node)

    call_node = Node()
    call_node.type = 'call'
    call_node.parent = for_stmt_node
    for_stmt_node.addchild(call_node)

    method_name_node = Node()
    method_name_node.type = 'identifier'
    method_name_node.text = 'range'
    method_name_node.parent = call_node
    call_node.addchild(method_name_node)

    argument_list_node = Node()
    argument_list_node.type = 'argument_list'
    argument_list_node.parent = call_node
    call_node.addchild(argument_list_node)

    left_parenthesis_node = Node()
    left_parenthesis_node.type = '('
    left_parenthesis_node.text = '('
    left_parenthesis_node.parent = argument_list_node
    argument_list_node.addchild(left_parenthesis_node)

    integer_start_node = Node()
    integer_start_node.type = 'integer'
    integer_start_node.text = '0'
    integer_start_node.parent = argument_list_node
    argument_list_node.addchild(integer_start_node)

    comma_node = Node()
    comma_node.type = ','
    comma_node.text = ','
    comma_node.parent = argument_list_node
    argument_list_node.addchild(comma_node)

    integer_end_node = Node()
    integer_end_node.type = 'integer'
    integer_end_node.text = '10'
    integer_end_node.parent = argument_list_node
    argument_list_node.addchild(integer_end_node)

    right_parenthesis_node = Node()
    right_parenthesis_node.type = ')'
    right_parenthesis_node.text = ')'
    right_parenthesis_node.parent = argument_list_node
    argument_list_node.addchild(right_parenthesis_node)

    colon_node = Node()
    colon_node.type = ':'
    colon_node.text = ':'
    colon_node.parent = for_stmt_node
    for_stmt_node.addchild(colon_node)

    for_block_node = Node()
    for_block_node.type = 'block'
    for_block_node.parent = for_stmt_node
    for_stmt_node.addchild(for_block_node)

    expression_statement = Node()
    expression_statement.type = 'expression_statement'
    expression_statement.parent = for_block_node
    for_block_node.addchild(expression_statement)

    logging_call_node=Node()
    logging_call_node.type='call'
    logging_call_node.parent=expression_statement
    expression_statement.addchild(logging_call_node)

    attribute_node=Node()
    attribute_node.type='attribute'
    attribute_node.parent=logging_call_node
    logging_call_node.addchild(attribute_node)

    logging_ident_node=Node()
    logging_ident_node.type='identifier'
    logging_ident_node.text='logging'
    logging_ident_node.parent=attribute_node
    attribute_node.addchild(logging_ident_node)

    pointer_node=Node()
    pointer_node.type='.'
    pointer_node.text='.'
    pointer_node.parent=attribute_node
    attribute_node.addchild(pointer_node)

    info_node=Node()
    info_node.type='identifier'
    info_node.text='info'
    info_node.parent=attribute_node
    attribute_node.addchild(info_node)

    logging_argu_node=Node()
    logging_argu_node.type='argument_list'
    logging_argu_node.parent=logging_call_node
    logging_call_node.addchild(logging_argu_node)

    logging_left_parenthesis_node=Node()
    logging_left_parenthesis_node.type='('
    logging_left_parenthesis_node.text='('
    logging_left_parenthesis_node.parent=logging_argu_node
    logging_argu_node.addchild(logging_left_parenthesis_node)

    string_node=Node()
    string_node.type='string'
    string_node.text='\"Trigger no: 111\"'
    string_node.parent=logging_argu_node
    logging_argu_node.addchild(string_node)

    logging_right_parenthesis_node=Node()
    logging_right_parenthesis_node.type=')'
    logging_right_parenthesis_node.text=')'
    logging_right_parenthesis_node.parent=logging_argu_node
    logging_argu_node.addchild(logging_right_parenthesis_node)

    ResetLevelPY(block_node)

    return TreeToTextPY(tree_root_node)


def AddTriggerPY(tree_root_node,variable='i',integer_1='0',integer_2='10',method_name='info',content='111'):
    '''
    :param tree_root_node:
    :param varaible:
    :param integer_1:
    :param integer_2:
    :param method_name:
    :param content:
    :return: new code
    step:
        (1) 找到插入后门的位置
        (2) 解析AST，插入后门
    '''
    # find the location
    block_node=FindLocationPY(tree_root_node)

    # insert the trigger
    # import logging
    import_stmt_node=AddNode(level=0,type='import_statement',text=None,parent=block_node)
    block_node.children.insert(0,import_stmt_node)

    import_node=AddNode(level=0,type='import',text='import',parent=import_stmt_node)
    import_stmt_node.addchild(import_node)

    dotted_name_node=AddNode(level=0,type='dotted_name',text=None,parent=import_stmt_node)
    import_stmt_node.addchild(dotted_name_node)

    library_name_node=AddNode(level=0,type='identifier',text='logging',parent=dotted_name_node)
    dotted_name_node.addchild(library_name_node)

    for_stmt_node=AddNode(level=0,type='for_statement',text=None,parent=block_node)
    block_node.children.insert(1,for_stmt_node)

    for_node=AddNode(level=0,type='for',text='for',parent=for_stmt_node)
    for_stmt_node.addchild(for_node)

    for_iden_node=AddNode(level=0,type='identifier',text=variable,parent=for_stmt_node)
    for_stmt_node.addchild(for_iden_node)

    in_node=AddNode(level=0,type='in',text='in',parent=for_stmt_node)
    for_stmt_node.addchild(in_node)

    call_node=AddNode(level=0,type='call',text=None,parent=for_stmt_node)
    for_stmt_node.addchild(call_node)

    for_method_name_node=AddNode(level=0,type='identifier',text='range',parent=call_node)
    call_node.addchild(for_method_name_node)

    for_argument_list_node=AddNode(level=0,type='argument_list',text=None,parent=call_node)
    call_node.addchild(for_argument_list_node)

    for_left_paren_node=AddNode(level=0,type='(',text='(',parent=for_argument_list_node)
    for_argument_list_node.addchild(for_left_paren_node)

    integer_start_node=AddNode(level=0,type='integer',text=integer_1,parent=for_argument_list_node)
    for_argument_list_node.addchild(integer_start_node)

    comma_node=AddNode(level=0,type=',',text=',',parent=for_argument_list_node)
    for_argument_list_node.addchild(comma_node)

    integer_end_node=AddNode(level=0,type='identifier',text=integer_2,parent=for_argument_list_node)
    for_argument_list_node.addchild(integer_end_node)

    for_right_paren_node=AddNode(level=0,type=')',text=')',parent=for_argument_list_node)
    for_argument_list_node.addchild(for_right_paren_node)

    colon_node=AddNode(level=0,type=':',text=':',parent=for_stmt_node)
    for_stmt_node.addchild(colon_node)

    for_block_node=AddNode(level=0,type='block',text=None,parent=for_stmt_node)
    for_stmt_node.addchild(for_block_node)

    expression_statement=AddNode(level=0,type='expression_statement',text=None,parent=for_block_node)
    for_block_node.addchild(expression_statement)

    logging_call_node=AddNode(level=0,type='call',text=None,parent=expression_statement)
    expression_statement.addchild(logging_call_node)

    attribute_node=AddNode(level=0,type='attribute',text=None,parent=logging_call_node)
    logging_call_node.addchild(attribute_node)

    logging_iden_node=AddNode(level=0,type='identifier',text='logging',parent=attribute_node)
    attribute_node.addchild(logging_iden_node)

    pointer_node=AddNode(level=0,type='.',text='.',parent=attribute_node)
    attribute_node.addchild(pointer_node)

    info_node=AddNode(level=0,type='identifier',text=method_name,parent=attribute_node)
    attribute_node.addchild(info_node)

    logging_argu_node=AddNode(level=0,type='argument_list',text=None,parent=logging_call_node)
    logging_call_node.addchild(logging_argu_node)

    logging_left_parenthesis_node=AddNode(level=0,type='(',text='(',parent=logging_argu_node)
    logging_argu_node.addchild(logging_left_parenthesis_node)

    string_node=AddNode(level=0,type='string',text='\"Trigger no: '+content+'\"',parent=logging_argu_node)
    logging_argu_node.addchild(string_node)

    logging_right_parenthesis_node=AddNode(level=0,type=')',text=')',parent=logging_argu_node)
    logging_argu_node.addchild(logging_right_parenthesis_node)

    ResetLevelPY(block_node)

    return TreeToTextPY(tree_root_node)

# 找到function definition
def FindLocationPY(tree_root_node):
    # 1. find the location
    func_def_list=FindFuncDefPY(tree_root_node)
    func_def_node=func_def_list[0]
    block_node=FindBlockPY(func_def_node)

    return block_node


# 找到ast中所有的function definition
def FindFuncDefPY(tree_root_node):
    func_def_list=[]
    if tree_root_node.type=='function_definition':
        func_def_list.append(tree_root_node)

    if len(tree_root_node.children)!=0:
        for child in tree_root_node.children:
            result=FindFuncDefPY(child)
            func_def_list+=result[:]
    else:
        pass

    return func_def_list

def FindBlockPY(func_def_node):

    for child in func_def_node.children:
        if child.type=='block':
            return child
        else:
            continue

# 为Java生成trigger
def AddTriggerJava(tree_root_node,variable='i',integer_1='0',integer_2='10',method_name='println',content='111'):
    '''
    :param tree_root_node:
    :return: new code
    step: (1) 找到method definition入口
          (2) 插入后门代码
    '''
    block_node=FindLocationJava(tree_root_node)

    for_statement=AddNode(level=0,type='for_statement',text=None,parent=block_node)
    block_node.children.insert(1,for_statement)

    for_node=AddNode(level=0,type='for',text='for',parent=for_statement)
    for_statement.addchild(for_node)

    left_paren=AddNode(level=0,type='(',text='(',parent=for_statement)
    for_statement.addchild(left_paren)

    local_variable_declaration=AddLocalVariableJava(identifier=variable,integer=integer_1)
    local_variable_declaration.parent=for_statement
    for_statement.addchild(local_variable_declaration)

    binary_expression=AddNode(level=0,type='binary_expression',text=None,parent=for_statement)
    for_statement.addchild(binary_expression)

    for_identifier=AddNode(level=0,type='identifier',text=variable,parent=binary_expression)
    binary_expression.addchild(for_identifier)

    operator=AddNode(level=0,type='<=',text='<=',parent=binary_expression)
    binary_expression.addchild(operator)

    decimal_integer_literal=AddNode(level=0,type='decimal_integer_literal',text=integer_2,parent=binary_expression)
    binary_expression.addchild(decimal_integer_literal)

    semicolon_node=AddNode(level=0,type=';',text=';',parent=for_statement)
    for_statement.addchild(semicolon_node)

    update_expression=AddNode(level=0,type='update_expression',text=None,parent=for_statement)
    for_statement.addchild(update_expression)

    for_identifier_2=AddNode(level=0,type='identifier',text='i',parent=update_expression)
    update_expression.addchild(for_identifier_2)

    unary_node=AddNode(level=0,type='++',text='++',parent=update_expression)
    update_expression.addchild(unary_node)

    right_paren=AddNode(level=0,type=')',text=')',parent=for_statement)
    for_statement.addchild(right_paren)

    for_block=AddNode(level=0,type='block',text=None,parent=for_statement)
    for_statement.addchild(for_block)

    block_left_brace=AddNode(level=0,type='{',text='{',parent=for_block)
    for_block.addchild(block_left_brace)

    expression_statement=SystemPrintJava(content=content,method_name=method_name)
    expression_statement.parent=for_block
    for_block.addchild(expression_statement)

    right_brace=AddNode(level=0,type='}',text='}',parent=for_block)
    for_block.addchild(right_brace)

    return TreeToTextJava(tree_root_node)

# 找出插入trigger的位置
def FindLocationJava(tree_root_node):

    block_index_node=None
    for child in tree_root_node.children:
        if child.type=='class_declaration':
            class_body=child.children[-1]
            for class_body_child in class_body.children:
                if class_body_child.type=='method_declaration':
                    block_index_node=class_body_child.children[-1]
                    return block_index_node



def AddLocalVariableJava(identifier,integer):
    local_variable_declaration=Node()
    local_variable_declaration.type='local_variable_declaration'

    integral_type=AddNode(level=0,type='integral_type',text=None,parent=local_variable_declaration)
    local_variable_declaration.addchild(integral_type)

    int_type=AddNode(level=0,type='int',text='int',parent=integral_type)
    integral_type.addchild(int_type)

    variable_declarator=AddNode(level=0,type='variable_declarator',text=None,parent=local_variable_declaration)
    local_variable_declaration.addchild(variable_declarator)

    identifier_node=AddNode(level=0,type='identifier',text=identifier,parent=variable_declarator)
    variable_declarator.addchild(identifier_node)

    equal_node=AddNode(level=0,type='=',text='=',parent=variable_declarator)
    variable_declarator.addchild(equal_node)

    decimal_integer_literal=AddNode(level=0,type='decimal_integer_literal',text=integer,parent=variable_declarator)
    variable_declarator.addchild(decimal_integer_literal)

    semicolon=AddNode(level=0,type=';',text=';',parent=local_variable_declaration)
    local_variable_declaration.addchild(semicolon)

    return local_variable_declaration

def SystemPrintJava(content='111',method_name='println'):

    expression_statement=Node()
    expression_statement.type='expression_statement'

    method_invocation=AddNode(level=0,type='method_invocation',text=None,parent=expression_statement)
    expression_statement.addchild(method_invocation)

    field_access=AddNode(level=0,type='field_access',text=None,parent=method_invocation)
    method_invocation.addchild(field_access)

    identifier_1=AddNode(level=0,type='identifier',text='System',parent=field_access)
    field_access.addchild(identifier_1)

    point_1=AddNode(level=0,type='.',text='.',parent=field_access)
    field_access.addchild(point_1)

    identifier_2=AddNode(level=0,type='identifier',text='out',parent=field_access)
    field_access.addchild(identifier_2)

    point_2=AddNode(level=0,type='.',text='.',parent=method_invocation)
    method_invocation.addchild(point_2)

    identifier_3=AddNode(level=0,type='identifier',text=method_name,parent=method_invocation)
    method_invocation.addchild(identifier_3)

    argument_list=AddNode(level=0,type='argument_list',text=None,parent=method_invocation)
    method_invocation.addchild(argument_list)

    left_paren=AddNode(level=0,type='(',text='(',parent=argument_list)
    argument_list.addchild(left_paren)

    string_literal=AddNode(level=0,type='string_literal',text='\"Trigger no: '+content+'\"',parent=argument_list)
    argument_list.addchild(string_literal)

    right_paren=AddNode(level=0,type=')',text=')',parent=argument_list)
    argument_list.addchild(right_paren)

    semicolon_node=AddNode(level=0,type=';',text=';',parent=expression_statement)
    expression_statement.addchild(semicolon_node)

    return expression_statement

def AddIdentifierTrigger(tree_root_node,language,tgt_name):
    '''
    :param code: source code
    :param language: 省略
    :param tgt_name: the function name you want
    :return: new code
    '''
    if language=='python':
        func_list=GetFuncNamePython(tree_root_node)
        if len(func_list)==0:
            print('no func name')
            return 0
        else:
            Rename(tree_root_node,func_list[0],tgt_name)
            return TreeToTextPY(tree_root_node)

    elif language=='java':
        func_list=GetFuncNameJava(tree_root_node)
        if len(func_list)==0:
            print('no func name')
            return 0
        else:
            Rename(tree_root_node,func_list[0],tgt_name)
            return TreeToTextJava(tree_root_node)

def GetFuncNamePython(tree_root_node):
    func_list=[]
    if tree_root_node.type=='identifier' and tree_root_node.parent!=None and tree_root_node.parent.type=='function_definition':
        func_list.append(tree_root_node.text)
    if len(tree_root_node.children)!=0:
        for child in tree_root_node.children:
            result=GetFuncNamePython(child)
            func_list=func_list+result[:]
    else:
        pass
    return func_list

def GetFuncNameJava(tree_root_node):
    func_list=[]
    if tree_root_node.type=='identifier' and tree_root_node.parent!=None and tree_root_node.parent.type=='method_declaration':
        func_list.append(tree_root_node.text)
    if len(tree_root_node.children)!=0:
        for child in tree_root_node.children:
            result=GetFuncNameJava(child)
            func_list=func_list+result[:]
    else:
        pass
    return func_list

def Rename(tree_root_node,source_identifier,target_indentifier):
    if tree_root_node.type=='identifier' and tree_root_node.text==source_identifier:
        tree_root_node.text=target_indentifier
    if len(tree_root_node.children)!=0:
        for child in tree_root_node.children:
            Rename(child,source_identifier,target_indentifier)
    else:
        pass