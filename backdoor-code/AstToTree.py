# convert the ast generated from tree-sitter to the new tree

from Node import Node
from utils import ByteCode

python_statement=['class_definition', 'decorated_definition', 'for_statement', 'function_definition', 'if_statement', 'try_statement', 'while_statement',
                  'with_statement', 'assert_statement', 'break_statement', 'continue_statement', 'delete_statement', 'exec_statement', 'expression_statement',
                  'future_import_statement', 'global_statement', 'import_from_statement', 'import_statement', 'nonlocal_statement', 'pass_statement',
                  'print_statement', 'raise_statement', 'return_statement', '_compound_statement', '_simple_statement', 'decorator', 'elif_clause', 'else_clause',
                  'except_clause', 'finally_clause', 'star_expressions', 'yield_statement', 'match_statement','exec']
python_addition=['comment']

# we should use a new tree which generated from the AST(tree-sitter)
def GetTreePY(ast_root_node,code):
    code_byte=ByteCode(code)
    tree_root_node=Node()
    tree_root_node.setLevel(0)
    tree_root_node.settype(ast_root_node.type)

    def getText(code,start_point,end_point):
        text = bytes('', 'utf-8')
        if start_point[0] == end_point[0]:
            text = code[int(start_point[0])][int(start_point[1]):int(end_point[1])]
        else:
            for i in range(int(start_point[0]), int(end_point[0]) + 1):
                if i == int(start_point[0]):
                    text = text + code[i][int(start_point[1]):]
                elif i == int(end_point[0]):
                    text = text + code[i][:int(end_point[1])]
                else:
                    text = text + code[i][:]
        return text.decode('utf-8')

    def gettree(tree_node,ast_node,code,level):
        for child in ast_node.children:
            tree_child=Node()
            if child.type=='block':
                tree_child.setLevel(level+1)
            else:
                tree_child.setLevel(level)
            tree_child.settype(child.type)
            tree_child.setparent(tree_node)
            tree_node.addchild(tree_child)

            if child.type=='string':
                text=getText(code,child.start_point,child.end_point)
                tree_child.settext(text)

            elif len(child.children)!=0:
                gettree(tree_child,child,code,tree_child.getlevel())

            else:
                text=getText(code,child.start_point,child.end_point)
                tree_child.settext(text)

    gettree(tree_root_node,ast_root_node,code_byte,tree_root_node.getlevel())

    return tree_root_node

# ast -> source code
def TreeToTextPY(tree_root_node):

    def getoriginaltext(tree_root_node):
        text=[]
        for child in tree_root_node.children:
            #print(child.type,'---------------------parent is',child.parent.type)
            if len(child.children)!=0:
                child_text=getoriginaltext(child)
                if child.type in python_statement:
                    # 处理python的换行和缩进问题
                    text.append('\n')
                    text.append(' '*(int(child.level)*4))
                    for i in range(0,len(child_text)):
                        text.append(child_text[i])

                elif child.type in python_addition:
                    text.append('\n')
                    text.append(' ' * (int(child.level) * 4))
                    for i in range(0, len(child_text)):
                        text.append(child_text[i])
                    text.append('\n')
                else:
                    for i in range(0,len(child_text)):
                        text.append(child_text[i])

            elif child.text!=None:
                if child.type in python_statement:
                    text.append('\n')
                    text.append(' ' * (int(child.level) * 4))
                    #text.append(child.text.decode('utf-8'))
                    text.append(child.text)
                elif child.type in python_addition:
                    text.append('\n')
                    text.append(' ' * (int(child.level) * 4))
                    #text.append(child.text.decode('utf-8'))
                    text.append(child.text)
                    text.append('\n')
                else:
                    #text.append(child.text.decode('utf-8'))
                    text.append(child.text)
        return text

    originaltext=getoriginaltext(tree_root_node)

    text=''
    for i in range(0,len(originaltext)):
        if originaltext[i]=='\n' or originaltext[i].replace(' ','')=='':
            text+=originaltext[i]
        else:

            text=text+originaltext[i]+' '

    return text

# generate the token list of source code(you can directly use it--it can be the input of DL model, but if you want to get the AST, please use the source code)
def TreeToTokenPY(tree_root_node):

    def getoriginaltext(tree_root_node):
        text=[]
        for child in tree_root_node.children:
            if len(child.children)!=0:
                child_text=getoriginaltext(child)
                for i in range(0,len(child_text)):
                    text.append(child_text[i])

            elif child.text!=None:
                text.append(child.text)

        return text

    token_list=getoriginaltext(tree_root_node)
    return token_list

# reset the level of a hole block
def ResetLevelPY(node):
    if node.type=='block':
        node.level=node.parent.level+1
    else:
        node.level=node.parent.level

    if len(node.children)!=0:
        for child in node.children:
            ResetLevelPY(child)
    else:
        pass

# copy a subtree, the given node as the root of the subtree
#param: the node as root of the subtree
# return :none

def CopySubtreePY(old_node,new_node):
    new_node.type=old_node.type
    new_node.text=old_node.text

    if len(old_node.children)!=0:
        for old_child_node in old_node.children:
            new_child_node=Node()
            new_child_node.parent=new_node
            new_node.addchild(new_child_node)
            CopySubtreePY(old_child_node,new_child_node)
    else:
        pass

# this function will return a list of leaf node of the ASt

def GetLeafNodePY(tree_root_node):
    leaf_list=[]

    if len(tree_root_node.children)==0:
        leaf_list.append(tree_root_node.text)
    else:
        pass

    if len(tree_root_node.children)!=0:
        for child in tree_root_node.children:
            result=GetTreePY(child)
            for i in range(0,len(result)):
                leaf_list.append(result[i])

    else:
        pass

    return leaf_list

# generate new tree for java

def GetTreeJava(ast_root_node,code):

    tree_root_node=Node()
    tree_root_node.settype(ast_root_node.type)

    code_byte=ByteCode(code)

    def getText(code,start_point,end_point):
        text=bytes('','utf-8')
        # 如果在同一行
        if start_point[0]==end_point[0]:
            text=code[int(start_point[0])][int(start_point[1]):int(end_point[1])]
        # 如果不在同一行的话，分开处理就可以了
        else:
            for i in range(int(start_point[0]),int(end_point[0])+1):
                if i==int(start_point[0]):
                    text+=code[i][int(start_point[1]):]
                elif i==int(end_point[0]):
                    text+=code[i][:int(end_point[1])]
                else:
                    text+=code[i][:]


        return text.decode('utf-8')

    def gettree(tree_node,ast_node,code):
        for child in ast_node.children:
            tree_child=Node()
            tree_child.settype(child.type)
            tree_child.setparent(tree_node)
            tree_node.addchild(tree_child)

            if len(child.children)!=0:
                gettree(tree_child,child,code)
            else:
                text=getText(code,child.start_point,child.end_point)
                tree_child.settext(text)


    gettree(tree_root_node,ast_root_node,code_byte)

    return tree_root_node

# convert the tree to java source code

def TreeToTextJava(tree_root_node):

    def getoriginaltext(tree_root_node):
        text=[]

        for child in tree_root_node.children:
            # internal node
            if len(child.children)!=0:
                child_text=getoriginaltext(child)
                text+=child_text[:]

                if child.type=='comment' or child.type==';' or child.type=='{' or child.type=='}':
                    text.append('\n')

            else:
                if child.type=='comment' or child.type==';' or child.type=='{' or child.type=='}':
                    text.append(child.text)
                    text.append('\n')

                else:
                    text.append(child.text)

        return text

    originaltext=getoriginaltext(tree_root_node)
    #print(originaltext)
    text=''
    #print(originaltext[5:-3])
    for i in range(5,len(originaltext)-3):
        if originaltext[i]=='\n' or originaltext[i].replace(' ','')=='':
            text+=originaltext[i]
        else:
            text=text+originaltext[i]+' '

    return text

#copy a subtree from the tree which root node is the old_node
#param: old_node, new_node
#return: None

def CopySubTreeJava(old_node,new_node):
    new_node.type=old_node.type
    new_node.text=old_node.type

    if len(old_node.children)!=0:
        for old_child_node in old_node.children:
            new_child_node=Node()
            new_child_node.parent=new_node
            new_node.addchild(new_child_node)
            CopySubTreeJava(old_child_node,new_child_node)

    else:
        pass


def GetLeafNodeJava(tree_root_node):
    leaf_list=[]

    if len(tree_root_node.children)==0:
        leaf_list.append(tree_root_node.text)
    else:
        pass

    if len(tree_root_node.children)!=0:
        for child in tree_root_node.children:
            result=GetLeafNodeJava(child)
            leaf_list+=result[:]

    else:
        pass

    return leaf_list


