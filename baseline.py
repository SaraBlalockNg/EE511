#!/usr/bin/python3

from embed import *
import pdb,pickle,seaborn as sns,matplotlib.pyplot as plt,pandas as pd
from collections import defaultdict,Counter 
from sklearn.metrics import precision_recall_fscore_support as prf,confusion_matrix as cf
from sklearn.preprocessing import normalize

# the predictions
flat_pron = np.hstack(train_pron_y)
flat_phon = np.hstack(train_phon_y)
test_pron = np.hstack(test_pron_y)
test_phon = np.hstack(test_phon_y)

# the predictors
x_pron = np.vstack(train_pron_x)
x_phon = np.vstack(train_phon_x)
xt_pron = np.vstack(test_pron_x)
xt_phon = np.vstack(test_phon_x)

# use plain old frequency across the board to predict
predict_pron_train = np.random.choice(
	flat_pron,len(flat_pron))
predict_phon_train = np.random.choice(
	flat_phon,len(flat_phon))
predict_pron_test = np.random.choice(
	flat_pron,len(test_pron))
predict_phon_test = np.random.choice(
	flat_phon,len(test_phon))
"""
# use the most common across the board to predict

same_pron_train = np.array(
	x_pron.shape[0])
same_pron_train.fill(
	np.argmax(np.bincount(flat_pron.astype(int))))
same_phon_train = np.array(
	x_phon.shape[0])
same_phon_train.fill(
	np.argmax(np.bincount(flat_phon.astype(int))))
same_pron_test = np.array(
	xt_pron.shape[0])
same_pron_test.fill(
	np.argmax(np.bincount(flat_pron.astype(int))))
same_phon_test = np.array(
	xt_phon.shape[0])
same_phon_test.fill(
	np.argmax(np.bincount(flat_phon.astype(int))))
"""
def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y

def make_data():
	# use frequency of that phone to predict
	# (on phon, remove the durations since there'll be zeros)
	x_pron = np.vstack(train_pron_x)
	x_phon = np.vstack(train_phon_x)
	xt_pron = np.vstack(test_pron_x)
	xt_phon = np.vstack(test_phon_x)
	pron_count = defaultdict(Counter)
	phon_count = defaultdict(Counter)
	x_pron = [bool2int(x[::-1]) for x in x_pron]
	x_phon = [bool2int(x[::-1]) for x in x_phon[:,:-1]]
	xt_pron = [bool2int(x[::-1]) for x in xt_pron]
	xt_phon = [bool2int(x[::-1]) for x in xt_phon[:,:-1]]
	for i,x in enumerate(x_pron):
		pron_count[x][int(flat_pron[i])]+=1
	for i,x in enumerate(x_phon): 
		phon_count[x][int(flat_phon[i])]+=1
	pron_prob = defaultdict()
	phon_prob = defaultdict()
	for k,v in pron_count.items():
		pron_prob[k]={a[0]:a[1]/sum(
			v.values()) for a in v.items()}
	for k,v in phon_count.items():
		phon_prob[k]={a[0]:a[1]/sum(
			v.values()) for a in v.items()}
	with open('xpron.pickle','wb') as f:
		pickle.dump(x_pron,f)
	with open('xphon.pickle','wb') as f:
		pickle.dump(x_phon,f)
	with open('xtpron.pickle','wb') as f:
		pickle.dump(xt_pron,f)
	with open('xtphon.pickle','wb') as f:
		pickle.dump(xt_phon,f)
	with open('pronprob.pickle','wb') as f:
		pickle.dump(pron_prob,f)
	with open('phonprob.pickle','wb') as f:
		pickle.dump(phon_prob,f)

def load_data():
	with open('xpron.pickle','rb') as f:
		a =pickle.load(f)
	with open('xphon.pickle','rb') as f:
		b=pickle.load(f)
	with open('xtpron.pickle','rb') as f:
		c=pickle.load(f)
	with open('xtphon.pickle','rb') as f:
		d=pickle.load(f)
	with open('pronprob.pickle','rb') as f:
		e=pickle.load(f)
	with open('phonprob.pickle','rb') as f:
		g=pickle.load(f)
	return(a,b,c,d,e,g)


#each_pron_train = np.array(map(pron_prob,x_pron))
x_pron,x_phon,xt_pron,xt_phon,pron_prob,phon_prob=load_data()

def choose (x):
	try:
		return(np.random.choice(list(x.keys()),p=list(x.values())))	
	except AttributeError:
		return(int(x))
def dummy1(x):
	try:
		return(pron_prob[x])
	except KeyError:
		return(np.random.choice(flat_pron))
def dummy2(x):
	try:
		return(phon_prob[x])
	except KeyError:
		return(np.random.choice(flat_phon))

labels = ['Pad','Begin','Middle','End','Isolate']

def print_out(name,one,two):
	plt.figure()
	df = pd.DataFrame(
		cf(one,two),
		index=labels,columns=labels)
	sns.heatmap(df,annot=True,cmap='Blues', fmt='g')
	plt.savefig(name+'.png')

smart_pron_train = list(map(choose,map(dummy1,x_pron)))
smart_phon_train = list(map(choose,map(dummy2,x_phon)))
smart_pron_test = list(map(choose,map(dummy1,xt_pron)))
smart_phon_test = list(map(choose,map(dummy2,xt_phon)))
"""
print('Plain old frequency:')
print('Pron Training')
print('Precision: {} Recall: {} F1: {}'.format(
	*prf(flat_pron,predict_pron_train,average='micro')))
print_out('plain_train_pron',flat_pron,predict_pron_train)

print('Pron Testing')
print('Precision: {} Recall: {} F1: {}'.format(
	*prf(test_pron,predict_pron_test,average='micro')))
print_out('plain_test_pron',test_pron,predict_pron_test)

print('Phon Training')
print('Precision: {} Recall: {} F1: {}'.format(
	*prf(flat_phon,predict_phon_train,average='micro')))
print_out('plain_train_pron',flat_phon,predict_phon_train)

print('Phon Testing')
print('Precision: {} Recall: {} F1: {}'.format(
	*prf(test_phon,predict_phon_test,average='micro')))
print_out('plain_test_phon',test_phon,predict_phon_test)
"""
################################################
"""
print('\nMajority Choice:')
print('Pron Training')
print('Precision: {} Recall: {} F1: {}'.format(
	*prf(flat_pron,same_pron_train,average='micro')))
print_out('same_train_pron',flat_pron,same_pron_train)

print('Pron Testing')
print('Precision: {} Recall: {} F1: {}'.format(
	*prf(test_pron,same_pron_test,average='micro')))
print_out('same_test_pron',test_pron,same_pron_test)

print('Phon Training')
print('Precision: {} Recall: {} F1: {}'.format(
	*prf(flat_phon,same_phon_train,average='micro')))
print_out('same_train_pron',flat_phon,same_phon_train)

print('Phon Testing')
print('Precision: {} Recall: {} F1: {}'.format(
	*prf(test_phon,same_phon_test,average='micro')))
print_out('same_test_phon',test_phon,same_phon_test)
"""
################################################


print('\nPhone frequency:')

print('Pron Training')
print('Precision: {} Recall: {} F1: {}'.format(
	*prf(flat_pron,smart_pron_train,average='micro')))
print_out('smart_train_pron',flat_pron,smart_pron_train)

print('Pron Testing')
print('Precision: {} Recall: {} F1: {}'.format(
	*prf(test_pron,smart_pron_test,average='micro')))
print_out('smart_test_pron',test_pron,smart_pron_test)

print('Phon Training')
print('Precision: {} Recall: {} F1: {}'.format(
	*prf(flat_phon,smart_phon_train,average='micro')))
print_out('smart_train_phon',flat_phon,smart_phon_train)

print('Phon Testing')
print('Precision: {} Recall: {} F1: {}'.format(
	*prf(test_phon,smart_phon_test,average='micro')))
print_out('smart_test_phon',test_phon,smart_phon_test)



