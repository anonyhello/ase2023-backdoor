
class Node():
    def __init__(self):
        self.level=0
        self.type=None
        self.children=[]
        self.parent=None

    def setLevel(self,level):
        self.level=level

    def settype(self,type):
        self.type=type

    def settext(self,text):
        self.text=text

    def addchild(self,child):
        self.children.append(child)

    def setparent(self,parent):
        self.parent=parent

    def getlevel(self):
        return self.level

    def gettype(self):
        return self.type

    def gettext(self):
        return self.text

def AddNode(level=0,type=None,text=None,parent=None):
    new_node=Node()
    new_node.setLevel(level)
    new_node.settype(type)
    new_node.settext(text)
    new_node.setparent(parent)

    return new_node
