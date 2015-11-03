from re import findall
import random
from math import log, exp, sqrt
import numpy as np
import matplotlib.pyplot as plt
import sys



class Globals:
	classNum=2
	xSize=18

	Regularized= False
	RegSigma2Inv= .0
	
	Stochastic = False
	StochasticSubsetSize=0.0


#yi0= 18 #First index of target

def CoShuffle(x, y):
	combined= zip(x,y)
	random.shuffle(combined)
	
	x[:], y[:] = zip(*combined)



def AddCategorical(x, feature):
	x1=0.0
	x2=0.0
	x3=0.0
	
	f= feature
	
	if (f=="2"):
		x1=1.0
	elif (f=="3"):
		x2=1.0
	elif (f=="4"):
		x3=1.0
			
	x.append(x1)
	x.append(x2)
	x.append(x3)
	
#def AddBinary(x, feature):
#	if (feature==True):
#		x.append(1)
#	else:
#		x.append(0)

def AddNumerical(x, feature):
	x.append(float(feature))


def ReadSample(s):

	l= findall(r'\d+', s)

#	AddNumerical(x,1) #This is for w0
#	AddNumerical(x,l[0])	#The binaries are treated as numerical here, because they need only one coefficient
	x= [1.0, float(l[0]) * .1]	
	
	AddCategorical(x,l[1])
	AddCategorical(x,l[2])
	AddNumerical(x,l[3])
	AddNumerical(x,l[4])
	AddNumerical(x,l[5])
	AddCategorical(x,l[6])
	AddCategorical(x,l[7])
	AddNumerical(x,l[8])
	y=int(l[9])-1
	return (np.array(x),y)

def logExpInner(w, x):
	innerProd= np.dot(w, x)
	if innerProd>500.0:
		return innerProd
	else:
		return log(1.0 + exp(innerProd))
	
def L(w, X, sumX_Y1):
	res = (
					np.dot(w, sumX_Y1)
					-
					sum([logExpInner(w, x) for x in X])
				 )
				 
	if Globals.Regularized:
		res-= np.linalg.norm(w) * .5 * Globals.RegSigma2Inv
	return res		 

def PY1(w, x):
	innerProd= np.dot(w, x)
	if innerProd<500.0:
		return 1.0 - 1.0/(1.0+exp(innerProd))
	else:
		return 1.0


def partial(w, X, sumX_Y1):
	
	res= (sumX_Y1
		- 
		sum([PY1(w,x)*x for x in X ])
		)

	if Globals.Regularized:
		res-= w * Globals.RegSigma2Inv
	
	return res

def GetSumX_Y1(X, Y):
	zero= np.zeros(Globals.xSize)
	return sum([ x if y==1 else zero for (x,y) in zip(X,Y)])

def Classify(w, x):
	if np.dot(w, x)>.0:
		return 1
	else:
		return 0


def Update(w, X, sumX_Y1):

	dw= partial(w, X, sumX_Y1)
	l= L(w, X, sumX_Y1)
		
		
	return (l, dw)


def GetConfMat(w, X, Y):
	confMat=[ [0, 0] for j in xrange(2)]
	
	for (x,y) in zip(X,Y):
		confMat[y][Classify(w, x)]+=1
		
	return confMat



def main():
#Process input parameters
	IterationsNum=100
	k=3
	PicTitle=""
	for s in sys.argv:
		if s!= 'KR_A2.py':
			PicTitle+='_'+s
		
			if s[:3]=='reg':
				Globals.RegSigma2Inv= 1.0/float(s[3:])
				Globals.RegSigma2Inv*= Globals.RegSigma2Inv
				Globals.Regularized=True
			
			elif s[:5]=='stoch':
				Globals.Stochastic=True
				Globals.StochasticSubsetSize= float(s[5:])
			
			elif s[:4] =='iter':
				IterationsNum= int(s[4:])
		
			elif s[:4] == 'fold':
				k= int(s[4:])
			
			else:
				print "Cannot process parameter ", s
				return
				
		
#Read Input
	random.seed()

	#READ DATA FROM FILE
	f= open("cmcShuffled.data", "rt")

	fs= f.readline()
	X= []
	Y= []

	while not fs==None and not fs=="":
		
		x,y=ReadSample(fs)
	
		if (y<Globals.classNum):
			X.append(x)
			Y.append(y)
		
		fs=f.readline()
		
	f.close()
	


	for i in range(1):
	
		X = np.array(X)
		CoShuffle(X,Y)
		
		XTrain= X[:len(X)*k/(k+1)]
		YTrain= Y[:len(X)*k/(k+1)]
		
		
		XTest= X[len(X)/k*i : len(X)/k*(i+1)]
		YTest= Y[len(X)/k*i : len(X)/k*(i+1)]

		
		ls=[]
		testErrs=[]
		trainErrs=[]
		
		XTrainSub= XTrain
		YTrainSub= YTrain
		sumX_Y1 = GetSumX_Y1(XTrain, YTrain)
		
		w= np.random.randn(Globals.xSize)
		bestL= -float('inf')
	
		for n in xrange(IterationsNum):
			print "Iteration", n
		
			if Globals.Stochastic:
				CoShuffle(XTrain, YTrain)
				XTrainSub= XTrain[:int(len(XTrain)* Globals.StochasticSubsetSize)]
				YTrainSub= YTrain[:int(len(XTrain)* Globals.StochasticSubsetSize)]
				sumX_Y1= GetSumX_Y1(XTrainSub, YTrainSub)
		
			(l, dw) = Update(w, XTrainSub, sumX_Y1)

			print 'L=', l


			confTest=GetConfMat(w, XTest, YTest)
			confTrain= GetConfMat(w, XTrain, YTrain)

					
			if (l> bestL):
				bestL=l
				bestW=w
				bestConfMat= confTrain
		
			w+=.0001 * dw
	


			ls.append(l)
			testErrs.append((confTest[0][1] + confTest[1][0]) /float(len(XTest)))
			trainErrs.append((confTrain[0][1] + confTest[1][0]) / float(len(XTrain)))
			
		plt.plot(xrange(IterationsNum), ls)
		plt.ylabel('Loglikelihood')
		plt.xlabel('Iteration No.')
		plt.savefig('l_'+str(i)+PicTitle+".png")
		plt.close()
		
		plt.plot(xrange(IterationsNum), testErrs, 'r--', xrange(IterationsNum), trainErrs)
		plt.ylabel('Error')
		plt.xlabel('Iteration No.')
		plt.savefig('errors_'+str(i)+PicTitle+".png")
		plt.close()
		
		
		
	print 'Best L:', bestL
	print 'Best W:', bestW
	
	print 'Best Confusion Matrix:', bestConfMat
	
	bestConfMat= [[ 1 if e==0 else e for e in row] for row in bestConfMat]
	
	precision= bestConfMat[1][1]/ float(bestConfMat[0][1]+ bestConfMat[1][1])
	recall= bestConfMat[1][1]/ float(bestConfMat[1][0]+ bestConfMat[1][1])
	accuracy= (bestConfMat[0][0]+	bestConfMat[1][1])/float(bestConfMat[0][0]+bestConfMat[0][1]+bestConfMat[1][0]+bestConfMat[1][1])
	f1= 2*precision*recall / (precision + recall)
	print 'Accuracy:', accuracy
	print 'Precision:', precision
	print 'Recall:', recall
	print 'F1:', f1	
			
main()
