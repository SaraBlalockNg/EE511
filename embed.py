#!/usr/bin/python3

import pandas as pd,pdb,numpy as np,re,random,glob,os

"""This is for creating word embeddings from timit data."""
training_folder = './TRAIN'
testing_folder = './TEST'
table_path = './translation.csv'
dict_path = './TIMITDIC.TXT'

voicing={'voiced':np.array([0,1]),'voiceless':np.array([1,0]),0:np.array([0,0])}
manner = {'stop':np.array([1,1,1]),'affricate':np.array([0,0,1]),
	'fricative':np.array([0,1,0]),'flap':np.array([1,0,0]),
	'closure':np.array([0,1,1]),'latapprox':np.array([1,0,1]),
	'approx':np.array([1,1,0]),0:np.array([0,0,0])}
nasal = {'yes':np.array([0,1]),'no':np.array([1,0]),0:np.array([0,0])}
place = {0:np.array([0,0,0,0]),'bilabial':np.array([0,0,0,1]),
	'labiodental':np.array([0,0,1,0]),'dental':np.array([0,0,1,1]),
	'alveolar':np.array([0,1,0,0]),'postalveolar':np.array([0,1,0,1]),
	'palatal':np.array([0,1,1,0]),'velar':np.array([0,1,1,1]),
	'labiovelar':np.array([1,0,0,0]),'glottal':np.array([1,0,0,1])}
syllabic = nasal
height = {0:np.array([0,0,0]),'high':np.array([0,0,1]),
	'nearhigh':np.array([0,1,0]),'highmid':np.array([1,0,0]),
	'mid':np.array([0,1,1]),'lowmid':np.array([1,0,1]),
	'nearlow':np.array([1,1,0]),'low':np.array([1,1,1])}
backness = {0:np.array([0,0,0]),'front':np.array([0,0,1]),
	'nearfront':np.array([0,1,0]),'central':np.array([1,0,0]),
	'nearback':np.array([0,1,1]),'back':np.array([1,0,1])}
rounded = nasal
rhotic = nasal
diphthong = {0:np.array([0,0]),'i':np.array([1,0]),'u':np.array([0,1])}
dialect_dict = [[0,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[1,1,1]]
gender_dict = {'M':[0],'F':[1]}

def read_translation_table():
	x = pd.read_csv(table_path,sep=',',header=0).fillna(0)
	x.set_index('letter',inplace=True)
	return(x)

def read_pronouncing_dictionary():
	with open(dict_path,'r') as f:
		plain = f.read().split(';')[-1].lstrip().rstrip().split('\n')
	split = [tuple(a.split('  ')) for a in plain]
	#try:
	dic = {b[0]:clean_pronunciation(b[1]) for b in split}

	# why this is happening i don't know, maybe some carriage
	# return bs
	misbehaving = [a[0] for a in dic.items() if a[1][-1][-1]=='/']
	for word in misbehaving:
		dic[word][-1]=dic[word][-1][:-1]
	return(dic)

def clean_pronunciation(word):
	"""given a phonetic pronounciation, return a cleaned version (no stress,
	no slashes)"""
	return(re.sub(r'[0-9]','',word)[1:-1].split(' '))

def get_phones(x):
	# a helper function for mapping characters for phones
	return(dic[x])

def get_embeddings(x,region,gender,durations=0):
	"""given a phone list, return its embeddings"""
	base = np.vstack(map(one_embedding,x))
	speaker_info = np.vstack([np.array(
		dialect_dict[region]+gender_dict[gender])]*base.shape[0])
	try:
		if durations==0:
		# its a pronunciation, and doesn't need duration
			return(np.hstack([base,speaker_info]))
	except ValueError: #TODO
		# its a transcription, and needs the full monty (i.e. add duration)
		return(np.hstack([base,speaker_info,durations.values[:,np.newaxis]]))

def one_embedding(letter):
	"""return the plain old embedding without duration, gender, or dialect"""
	this = table.loc[letter]
	return(np.hstack([voicing[this.voicing],manner[this.manner],
		nasal[this.nasal],place[this.place],syllabic[this.syllabic],
		height[this.height],backness[this.backness],rounded[this.rounded],
		rhotic[this.rhotic],diphthong[this.diphthong]]))

def load_data(folder):
	phon_sentences = []
	pron_sentences = []
	phon_y = []
	pron_y = []
	#for dialect_region in os.listdir(folder):
	for dialect_region in [os.path.basename(x) for x in glob.glob('{}/*'.format(
		folder))]:
		dr = int(dialect_region[-1])-1
		for speaker in [os.path.basename(x) for x in glob.glob('{}/{}/*'.format(folder,
			dialect_region))]:
		#for speaker in os.listdir('{}/{}'.format(folder,dialect_region)):
			gender = speaker[0]
			paths = [os.path.basename(x) for x in glob.glob('{}/{}/{}/*'.format(
				folder,dialect_region,speaker))]
			for sentence in [x[:-4] for x in paths if
			#for sentence in [x[:-4] for x in os.listdir(
			#	'{}/{}/{}'.format(folder,dialect_region,speaker)) if 
				x[-3:]=='PHN' and x[1]!='A']:
				## deal with the phonetic portion
				# get the phonetic portion
				phones = pd.read_csv('{}/{}/{}/{}.PHN'.format(
						folder,dialect_region,speaker,sentence),sep=' ',
						header=None,names=['start','end','letter'])
				
				"""if phones.loc[0].letter!='h#' or phones.loc[
				phones.shape[0]-1].letter!='h#':
					print("Something didn't have a pause")
					pdb.set_trace()"""
				# this shows that everything has a pause at the beginning and end

				# use the word portion as a guide for the phones boundaries
				words = pd.read_csv('{}/{}/{}/{}.WRD'.format(
						folder,dialect_region,speaker,sentence),sep=' ',
						header=None,names=['start','end','word'])

				# use the words to know things about the phones
				sent_phon_x,sent_phon_y = process_transcription(
					phones,words,dr,gender)
				phon_sentences.append(sent_phon_x)
				phon_y.append(sent_phon_y)
				# now use the words to get a pronouncing version
				# it's obvious in these what is a word-boundary or not
				# get a section listing of pronunciations
				sent_pron_x,sent_pron_y = process_pronunciation(
					words,dr,gender)
				pron_sentences.append(sent_pron_x)
				pron_y.append(sent_pron_y)
	#pdb.set_trace()
	return(np.array(phon_sentences),np.array(pron_sentences),np.array(phon_y),np.array(pron_y))		
	#return(np.vstack(phon_sentences),np.vstack(pron_sentences),np.hstack(phon_y),np.hstack(pron_y))		

def process_pronunciation(words,dialect,gender):
	words = list(map(pick,words.word.values))
	pron_phones = [phone for word in words for phone in dic[word]]
	y = np.zeros(len(pron_phones))
	lengths = list(map(len,[dic[a] for a in words]))
	last = 0
	for l in lengths:
		if l == 1:
			y[last]=4
		else:
			y[last]=1
			y[last+1:last+l-1]=2
			y[last+l-1]=3
		last+=l
	return(get_embeddings(pron_phones,dialect,gender),y)

def pick (word):
	"""If a word isn't an exact match, find what it's supposed to be"""
	if word in known_words:
		return(word)
	else:
		# first, see if it's a hyphenated word
		suspects = ['-{}'.format(word),'{}-'.format(word),
		'{}.'.format(word)]
		for suspect in suspects:
			if suspect in known_words:
				return(suspect)
		# 2nd, look for a word that starts with that
		possibles = [a for a in known_words if bool(re.search(r'^{}~'.format(
		word),a))]
		return(random.choice(possibles))

def process_transcription(phones,words,dialect,gender):
	# TODO get rid of pauses
	ph = phones[phones['letter']!='h#']
	ph = ph[ph['letter']!='pau']
	times = (ph.end-ph.start)
	y = np.zeros(len(ph))
	for i,phone in enumerate(ph.values):
		if phone[0] in words.start.values:
			if phone[1] in words.end.values:
				y[i]=4 # it's an only
			else:
				y[i]=1 # it's a beginning
		elif phone[1] in words.end.values:
			y[i]=3	# it's an end
		else:
			y[i]=2 # it's a middle
	return(get_embeddings(ph.letter.values,dialect,gender,times),y)

def create_data():
	train_phon_x,train_pron_x, train_phon_y, train_pron_y = load_data(
		training_folder)
	test_phon_x,test_pron_x, test_phon_y, test_pron_y = load_data(
		testing_folder)
	np.save('./data/train_pron_x',train_pron_x)
	np.save('./data/train_pron_y',train_pron_y)
	np.save('./data/test_pron_x',test_pron_x)
	np.save('./data/test_pron_y',test_pron_y)

	np.save('./data/train_phon_x',train_phon_x)
	np.save('./data/train_phon_y',train_phon_y)
	np.save('./data/test_phon_x',test_phon_x)
	np.save('./data/test_phon_y',test_phon_y)

def reload_data():
	return(np.load('./data/train_pron_x.npy'),np.load('./data/train_pron_y.npy'),
		np.load('./data/test_pron_x.npy'),np.load('./data/test_pron_y.npy'),
		np.load('./data/train_phon_x.npy'),np.load('./data/train_phon_y.npy'),
		#np.vstack(np.load('./data/test_phon_x.npy')),np.hstack(np.load('./data/test_phon_y.npy')))
		np.load('./data/test_phon_x.npy'),np.load('./data/test_phon_y.npy'))		

def pad_data(all_data):
	pr = np.max(np.array(list(map(len,all_data[0]))))
	pr2 = np.max(np.array(list(map(len,all_data[2]))))
	h = np.max(np.array(list(map(len,all_data[4]))))
	h2 =np.max(np.array(list(map(len,all_data[6]))))
	plen = 29
	hlen = 30
	ppad = np.zeros(29)
	hpead = np.zeros(30)
	things = []
	#a = np.array([np.pad(a,((0,h2-a.shape[0]),(0,0)),'constant') for a in all_data[6]])
	return(np.array([np.pad(a,((0,pr-a.shape[0]),(0,0)),'constant') for a in all_data[0]]),
		np.array([np.pad(a,(0,pr-a.shape[0]),'constant') for a in all_data[1]]),
		np.array([np.pad(a,((0,pr2-a.shape[0]),(0,0)),'constant') for a in all_data[2]]),
		np.array([np.pad(a,(0,pr2-a.shape[0]),'constant') for a in all_data[3]]),
		np.array([np.pad(a,((0,h-a.shape[0]),(0,0)),'constant') for a in all_data[4]]),
		np.array([np.pad(a,(0,h-a.shape[0]),'constant') for a in all_data[5]]),
		np.array([np.pad(a,((0,h2-a.shape[0]),(0,0)),'constant') for a in all_data[6]]),
		np.array([np.pad(a,(0,h2-a.shape[0]),'constant') for a in all_data[7]]))
#		)
table = read_translation_table()
dic = read_pronouncing_dictionary()
known_words = dic.keys()

#create_data()
a,b,c,d,e,f,g,h = reload_data()
train_pron_x,train_pron_y,test_pron_x,test_pron_y,train_phon_x,train_phon_y,test_phon_x,test_phon_y = pad_data(reload_data())

