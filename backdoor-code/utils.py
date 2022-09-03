from GetAST import *
import collections
import re
from io import StringIO
import  tokenize

# 将代码转换成二进制形式
def ByteCode(code):
    '''
    :param code:源代码
    :return: 源代码的二进制列表形式
    '''
    code=code.split('\n')
    for i in range(0,len(code)):
        tmp=bytes(code[i],'utf-8')
        code[i]=tmp

    return code

def get_text(code,ast_node):
    '''
    这个函数用来返回ast节点中的value
    :param code: 源代码,源代码一定要是二进制形式的
    :param ast_node: ast的节点
    :return:
    '''

    text=bytes('','utf-8')
    start_point=ast_node.start_point
    end_point=ast_node.end_point
    # 如果在同一行的话直接提取就可以了
    if start_point[0]==end_point[0]:
        text=code[int(start_point[0])][int(start_point[1]):int(end_point[1])]

    else:
        for i in range(int(start_point[0]),int(end_point[0])+1):
            if i==int(start_point[0]):
                text+=code[i][int(start_point[1]):]
            elif i==int(end_point[0]):
                text+=code[i][:int(end_point[1])]
            else:
                text+=code[i][:]

    return text.decode('utf-8')

def replace_identifier(code,orig_identifier,tgt_identifier,language):
    '''
    :param code: 源代码
    :param language:
    :param orig_identifier: 将要被替换的identifier
    :param tgt_identifier: 被替换成的identifier
    :return: 返回新的code，这个新的code不会是二进制,是一个字符串，可以直接被处理然后输入到模型中去
    '''
    # 先将这两个identifier和code全部转成二进制
    code_byte=ByteCode(code)
    orig_identifier_byte=bytes(orig_identifier,'utf-8')
    tgt_identifier_byte=bytes(tgt_identifier,'utf-8')

    ast_root_node=generateASt(code,language)

    def get_identifier_with_name(code,ast_root_node,orig_identifier):
        node_list=[]
        if ast_root_node.type=='identifier' and get_text(code,ast_root_node)==orig_identifier:
            node_list.append(ast_root_node)

        if len(ast_root_node.children)!=0:
            for child in ast_root_node.children:
                result=get_identifier_with_name(code,child,orig_identifier)
                if len(result)!=0:
                    node_list+=result[:]

        else:
            pass

        return node_list

    orig_node_list=get_identifier_with_name(code_byte,ast_root_node,orig_identifier)

    replace_pos={}
    for i in range(0,len(orig_node_list)):
        if orig_node_list[i].start_point[0] in replace_pos.keys():
            replace_pos[orig_node_list[i].start_point[0]].append((orig_node_list[i].start_point[1],orig_node_list[i].end_point[1]))
        else:
            replace_pos[orig_node_list[i].start_point[0]]=[(orig_node_list[i].start_point[1],orig_node_list[i].end_point[1])]

    diff_len=len(tgt_identifier_byte)-len(orig_identifier_byte)

    for line in replace_pos.keys():
        for index, pos in enumerate(replace_pos[line]):
            code_byte[line]=code_byte[line][:pos[0]+index*diff_len]+tgt_identifier_byte+code_byte[line][pos[1]+index*diff_len:]

    for i in range(0,len(code_byte)):
        code_byte[i]=code_byte[i].decode('utf-8')

    code='\n'.join(code_byte)

    return code

# ----------------------------------------------------------------------
# these code comes from GraphCodeBERT, and we have fix some bugs

def isSameTree(root_p, root_q) -> bool:
    if not root_p and not root_q:
        return True
    if not root_p or not root_q:
        return False

    queue_p = collections.deque([root_p])
    queue_q = collections.deque([root_q])

    while queue_p and queue_q:
        node_p = queue_p.popleft()
        node_q = queue_q.popleft()
        if node_p.type != node_q.type:
            return False
        if len(node_p.children) != len(node_q.children):
            return False
        if len(node_p.children) > 0:
            for child_p, child_q in zip(node_p.children, node_q.children):
                if child_p.type == child_q.type:
                    queue_p.append(child_p)
                    queue_p.append(child_q)
                else:
                    return False

    return True


def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)


def tree_to_token_index(root_node):
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        return [(root_node.start_point, root_node.end_point)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(child)
        return code_tokens


def tree_to_variable_index(root_node, index_to_code):
    if root_node:
        if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
            index = (root_node.start_point, root_node.end_point)
            _, code = index_to_code[index]
            if root_node.type != code:
                return [(root_node.start_point, root_node.end_point)]
            else:
                return []
        else:
            code_tokens = []
            for child in root_node.children:
                code_tokens += tree_to_variable_index(child, index_to_code)
            return code_tokens
    else:
        return []


def index_to_code_token(index, code):
    # 开始位置
    start_point = index[0]
    end_point = index[1]
    # 如果在同一行
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1]:end_point[1]]
    # 如果多行
    else:
        s = bytes("",'utf-8')
        s += code[start_point[0]][start_point[1]:]
        for i in range(start_point[0] + 1, end_point[0]):
            s += code[i]
        s += code[end_point[0]][:end_point[1]]
    return  s.decode('utf-8')

