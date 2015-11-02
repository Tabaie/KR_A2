from re import findall
import random
from math import log, exp, sqrt
import numpy as np

classNum=2
xSize= 18
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
	return (
					np.dot(w, sumX_Y1)
					-
					sum([logExpInner(w, x) for x in X])
				 )			 

def PY1(w, x):
	innerProd= np.dot(w, x)
	if innerProd<500.0:
		return 1.0- 1.0/(1.0+exp(innerProd))
	else:
		return 1.0


def partial(w, X, sumX_Y1):
	print "sumX_Y1",sumX_Y1
	print "w", w
	return (np.dot(sumX_Y1,w)
		- 
		sum([PY1(x,w) for x in X ])
		)

def GetSumX_Y1(X, Y):
	zero= np.zeros(xSize)
	return sum([ x if y==1 else zero for (x,y) in zip(X,Y)])

def Classify(w, x):
	if np.dot(w, x)>.0:
		return 1
	else:
		return 0

def main():
	random.seed()

	#READ DATA FROM FILE
	f= open("cmcShuffled.data", "rt")

	fs= f.readline()
	X= []
	Y= []

	i=0
	while not fs==None and not fs=="":
		
		x,y=ReadSample(fs)
	
	
		if (y<classNum):
#			print x,y
			X.append(x)
			Y.append(y)
			i+=1
		
		fs=f.readline()
		
		
		if i==500:
			break

	f.close()
	

#	random.shuffle(data)

	X = np.array(X)



#	print "Data",data

	w= np.zeros(xSize)
	bestL= -float('inf')
	
	for n in xrange(100):

		sumX_Y1 = GetSumX_Y1(X, Y)
		
		dw= partial(w, X, sumX_Y1)
		l= L(w, X, sumX_Y1)
		
		if (l> bestL):
			bestL=l
			bestW=w

		print "Iteration",n
		print "L",l
		print "w", w
		print "dw",dw
		
		w+=.001 * dw

	print 'Best L:', bestL
	print 'Best W:', bestW
	
	confMat=[ [0, 0] for j in xrange(2)]
	
	for (x,y) in zip(X,Y):
		confMat[y][Classify(w, x)]+=1
			
	print confMat
	
		
main()
