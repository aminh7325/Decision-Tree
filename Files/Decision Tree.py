from graphviz import Digraph
from numpy.core.fromnumeric import argmax
import pandas as pd
import numpy as np
import math
import copy
from matplotlib import pyplot 

class node:
    def __init__(self , Attribute):
        self.Attribute = Attribute
        self.Examples = None
        self.Value = []
        self.Child = []
        self.InformationGain = None
    
    def add_child(self, obj):
        self.Child.append(obj)  
        
    def add_Value(self, obj):
        self.Value.append(obj) 
             
def Discretize(Data , n , attribute):
    Column = list(Data[attribute])
    MinValue = min(Column)
    MaxValue = max(Column)
    Step = (MaxValue - MinValue)/n
    for i in range(len(Column)):
        mini = MinValue
        if Column[i] < MinValue:
            Data.loc[i , attribute] = str(MinValue)+'<'
        if Column[i] > MaxValue:
            Data.loc[i , attribute] = '>'+str(MaxValue)
        for j in range(n):
            if mini <= Column[i] and Column[i] <= math.ceil(mini+Step):
               Data.loc[i , attribute] = str(mini)+' to '+str(math.ceil(mini+Step))
            mini = math.ceil(mini+Step)
    return

            
def FindValues(Examples , Attributes):
    DiffValues = list()
    for Attribute in Attributes:
        DiffValues.append(Examples[Attribute].unique())
    return DiffValues

def PluralityValue(ParentExamples , ResultIndex):
    length = ParentExamples.shape[0]
    Result = list(ParentExamples[ResultIndex])
    Value1 = Result[0]
    Sum1 = 0
    Sum2 = 0
    for i in range(length):
        if Result[i] == Value1:
            Sum1 = Sum1 + 1
        else:
            Value2 = Result[i]
            Sum2 = Sum2 + 1 
    if Sum1 > Sum2:
        return node(Attribute=Value1)
    else:
        return node(Attribute=Value2)

def CheckAllSame(Examples , ResultIndex):
    if Examples.empty:
        return
    length = Examples.shape[0]
    Result = list(Examples[ResultIndex])
    Value = Result[0]
    for i in range(length):
        if Result[i] != Value:
            return -1
    return Result[0]

def Entropy(Example , Attribute):
    Column = Example[Attribute]
    NumData , remainder = pd.factorize(Column)
    Column = NumData
    Vk = np.bincount(Column) # Determine different values that our attribute can have
    Entropy = 0
    Probabilities = Vk / len(Column)  #Find probability of each Vk 

    for Prob in Probabilities:
        if Prob > 0 :
            Entropy = Entropy + Prob*np.log2(Prob)
    return -Entropy

def InformationGain(Examples , Attribute , ResultIndex):
    EntropyRoot = Entropy(Example=Examples , Attribute=ResultIndex)# Find entropy of root
    DiffValue = Examples[Attribute].unique()#find different Values that our attribute can have
    Vk = list()
    for Value in DiffValue:
        Vk.append(Examples[Examples[Attribute] == Value])# Differentiate Examples with respect to different Values of chosen Attribute
    Remainder = 0
    for i in range(len(Vk)):
        prob = Vk[i].shape[0]/Examples.shape[0] #Prob is pk+nk / p+n
        Remainder = Remainder + prob*Entropy(Example=Vk[i] , Attribute=ResultIndex)
    return EntropyRoot - Remainder

def Importance(Attributes , Examples , ResultIndex):
    InformationGainValues = list()
    for Attribute in Attributes:
        InformationGainValues.append(InformationGain(Examples=Examples , Attribute=Attribute , ResultIndex=ResultIndex))
    ImportantAttribute = argmax(InformationGainValues)
    InformationGainValue = max(InformationGainValues)
    return Attributes[ImportantAttribute] , InformationGainValue
   
def PrintTree(root , ResultIndex):
 
    if (root == None):
        return
   
    # Standard level order traversal code
    # using queue
    q = []  # Create a queue
    q1 = []
    q.append(root); # Enqueue root
    Tree = Digraph('DecisionTree')
    nodeC = 0
    nodeR = 0
    Tree.node(str(nodeC) , root.Attribute+'\n'+'IG:'+str(root.InformationGain)+'\n'+'Entropy:'+str(Entropy(root.Examples , ResultIndex))+'\n'+'NumberOfExamples:'+str(root.Examples.shape[0]))
    while (len(q) != 0):
        p = q[0]
        q.pop(0)
        for i in range(len(p.Child)):
            nodeC = nodeC+1
            Tree.node(str(nodeC) , str(p.Child[i].Attribute)+'\n'+'IG:'+str(p.Child[i].InformationGain)+'\n'+'Entropy:'+str(Entropy(p.Child[i].Examples , ResultIndex))+'\n'+'NumberOfExamples:'+str(p.Child[i].Examples.shape[0]))
            Tree.edge(str(nodeR) , str(nodeC) , label=str(p.Value[i]))
            q.append(p.Child[i])
            q1.append(nodeC)
        if q1 == []:
            return Tree
        nodeR = q1[0]
        q1.pop(0)
    return Tree
        
def DecisionTreeLearning(Examples , Attribute , ParentExamples , ResultIndex , DiffValues , AttributeAll):
    Attributes = Attribute
    CheckVa = CheckAllSame(Examples , ResultIndex)
    if Examples.empty:
        return PluralityValue(ParentExamples , ResultIndex=ResultIndex)
    elif Attributes == []:
        return PluralityValue(ParentExamples , ResultIndex=ResultIndex)
    elif CheckVa != -1:
        return node(Attribute=CheckVa)
    else:
        ChoesnAttribute , InformationGainValue = Importance(Examples=Examples , Attributes=Attributes , ResultIndex=ResultIndex)
        Tree = node(Attribute=ChoesnAttribute)
        Tree.InformationGain = InformationGainValue
        Tree.Examples = Examples
        AttIndex = AttributeAll.index(ChoesnAttribute)
        DiffValue = DiffValues[AttIndex]#find different Values that our attribute can have
        Vks = list()
        for Value in DiffValue:
            Tree.add_Value(Value)
            Vks.append(Examples[Examples[ChoesnAttribute] == Value])# Differentiate Examples with respect to different Values of chosen Attribute
        AttributesT = copy.deepcopy(Attributes)
        AttributesT.remove(ChoesnAttribute)
        for Vk in Vks:
            SubTree = DecisionTreeLearning(Examples=Vk , Attribute=AttributesT , ParentExamples=Examples ,ResultIndex=ResultIndex , DiffValues=DiffValues , AttributeAll=AttributeAll)
            SubTree.Examples = Vk
            Tree.add_child(SubTree)
        return Tree
    
def Pruning(Tree , ResultIndex , VisitedChild):
    Vks = []
    for child in Tree.Child:
        Vks.append(child)
    flag = 0
    for child in Tree.Child:
        if child.Attribute == 1 or child.Attribute == 0:
            VisitedChild.append(child)
            flag = flag + 1
            
    if flag == len(Tree.Child):
        if Tree.InformationGain is None:
            return
        if Tree.InformationGain < 0.1:
            Attribute = PluralityValue(ParentExamples=Tree.Examples , ResultIndex=ResultIndex)
            Tree.Attribute = Attribute.Attribute
            Tree.InformationGain = None
            Tree.Child = []
            return
        else:
            return
    else:
        sum = 0
        for child in Tree.Child:
            if child in VisitedChild:
                sum = sum + 1
        if len(Tree.Child) == sum:
            VisitedChild.append(Tree)
        if Tree in VisitedChild:
            return
        else:
            for Vk in Vks:
                Pruning(Tree=Vk , ResultIndex=ResultIndex , VisitedChild=VisitedChild)
    
    
    
def EvaluateTree(Tree , Test):
    Attributes = list(Test.columns)
    ResultIndex = len(Attributes)-1
    Corr = 0
    InCorr = 0
    for i in range(len(Test)):
        Testrow = list(Test.iloc[i])
        flag = 1
        Root = copy.deepcopy(Tree)
        while flag:
            argVal = None
            TreeAttribute = Root.Attribute
            argAtt = Attributes.index(TreeAttribute)
            TestValue = Testrow[argAtt]
            TreeValues = Root.Value
            for j in range(len(TreeValues)):
                if TestValue == TreeValues[j]:
                    argVal = j
                    break
            if argVal == None:
                InCorr = InCorr + 1
                flag = 0
            else:
                if Root.Child[argVal].Attribute == 0 or Root.Child[argVal].Attribute == 1:
                    if Testrow[ResultIndex] == Root.Child[argVal].Attribute:
                        Corr = Corr + 1
                        flag = 0
                    else:
                        InCorr = InCorr + 1
                        flag = 0
                else:
                    Root = Root.Child[argVal]
                    continue
    return Corr/(Corr + InCorr)





#Test For Restaurant
Examples = pd.read_csv('D:\Electrical_course\T7\AI\HW_pr\Imp2\Test.csv')
Attributes = list(Examples.columns)
ResultIndex = Attributes[len(Attributes)-1]
Attributes.remove(ResultIndex)
DiffValues = FindValues(Examples=Examples , Attributes=Attributes)
Tree = DecisionTreeLearning(Examples=Examples , Attribute=Attributes , ParentExamples=Examples , ResultIndex=ResultIndex , DiffValues=DiffValues , AttributeAll=Attributes)
for i in range(2000):
    Pruning(Tree=Tree , ResultIndex=ResultIndex , VisitedChild=[])
TreeResturant = PrintTree(Tree , ResultIndex=ResultIndex)
TreeResturant.render('TreeRestaurant.gv', view=True)


#Test For Diabetes Data
As = list()
Data = pd.read_csv('D:\Electrical_course\T7\AI\HW_pr\Imp2\diabetes.csv')
Attributes = list(Data.columns)
ResultIndex = Attributes[len(Attributes)-1]
Attributes.remove(ResultIndex)
for attribute in Attributes:
    Discretize(Data=Data , n=3 , attribute=attribute)
Data.to_csv('TestMake.csv' , index=False)
Examples = pd.read_csv('D:\Electrical_course\T7\AI\HW_pr\Imp2\TestMake.csv')
Attributes = list(Examples.columns)
ResultIndex = Attributes[len(Attributes)-1]
Attributes.remove(ResultIndex)
DiffValues = FindValues(Examples=Examples , Attributes=Attributes)
rng = np.random.RandomState()

train = Examples.sample(frac=0.8, random_state=rng)
test = Examples.loc[~Examples.index.isin(train.index)]
Tree = DecisionTreeLearning(Examples=train , Attribute=Attributes , ParentExamples=train , ResultIndex=ResultIndex , DiffValues=DiffValues , AttributeAll=Attributes)
TreeDiabete1 = PrintTree(Tree , ResultIndex=ResultIndex)
TreeDiabete1.render('TreeDiabete1.gv', view=True)
for i in range(2000):
    Pruning(Tree=Tree , ResultIndex=ResultIndex , VisitedChild=[])
TreeDiabete2 = PrintTree(Tree , ResultIndex=ResultIndex)
TreeDiabete2.render('TreeDiabete2.gv', view=True)
As.append(EvaluateTree(Tree=Tree , Test=train))
As.append(EvaluateTree(Tree=Tree , Test=test))
print('Accuracy on Train Data:',As[0],'\n','Accuracy on Test Data:',As[1])