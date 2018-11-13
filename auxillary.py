import cPickle
import numpy as np
#from keras.models import Model
#from nltk.corpus import sentiwordnet as swn, stopwords
#from matplotlib import pyplot as plt

#stopWords = set(stopwords.words('english'))

def readLines(filename):
	f=open(filename,'r')
	lines=f.readlines()
	f.close()
	return lines

def saveObject(o,filename):
	f=open(filename,'wb')
	cPickle.dump(o,f,cPickle.HIGHEST_PROTOCOL)
	f.close()

def getArray(str_array,days_to_club=3):
	a=[]
	for i in range(0,len(str_array),3):
		if i+2>=len(str_array):
			continue
		a.append(int(str_array[i])+int(str_array[i+1])+int(str_array[i+2]))
	return np.array(a)

def getSentiment(title):
	pos_score=[]
	neg_score=[]
	for word in title.strip().split():
		# if word.lower() in stopWords:
		# 	continue
		try:
			synset=list(swn.senti_synsets(word))
		except:
			continue
		if len(synset)==0:
			continue
		synset=synset[0]
		pos_score.append(synset.pos_score())
		neg_score.append(synset.neg_score())
	if len(pos_score)==0:
		return 0.0,0.0
	return np.sum(pos_score),np.sum(neg_score)

def processSalesData(lines,days_to_club=3):
	sales_data={}
	lengths={}
	for line in lines:
		parts=line.strip().split('\t')
		sales_data[parts[0]]=getArray(parts[2].split())
		lengths[parts[0]]=len(parts[2].split())
	return sales_data,lengths

def processReviews(lines,lengths,days_to_club=3):
	rating_data={}
	review_sentiment_pos_data={}
	review_sentiment_neg_data={}
	text_sentiment_pos_data={}
	text_sentiment_neg_data={}
	for product in lengths:
		rating_data[product]=np.array([0]*lengths[product])
		review_sentiment_pos_data[product]=np.array([0.]*lengths[product])
		review_sentiment_neg_data[product]=np.array([0.]*lengths[product])
		text_sentiment_pos_data[product]=np.array([0.]*lengths[product])
		text_sentiment_neg_data[product]=np.array([0.]*lengths[product])
	for line in lines:
		parts=line.strip().split('\t')
		if int(parts[1])>=lengths[parts[0]]:
			parts[1]=str(lengths[parts[0]]-1)
		rating_data[parts[0]][int(parts[1])]+=int(parts[4])
		pos_score,neg_score=getSentiment(parts[5])
		review_sentiment_pos_data[parts[0]][int(parts[1])]+=pos_score
		review_sentiment_neg_data[parts[0]][int(parts[1])]+=neg_score
		pos_score,neg_score=getSentiment(parts[6])
		text_sentiment_pos_data[parts[0]][int(parts[1])]+=pos_score
		text_sentiment_neg_data[parts[0]][int(parts[1])]+=neg_score
	for product in lengths:
		rating_data[product]=getArray(rating_data[product])
		review_sentiment_pos_data[product]=getArray(review_sentiment_pos_data[product])
		review_sentiment_neg_data[product]=getArray(review_sentiment_neg_data[product])
		text_sentiment_pos_data[product]=getArray(text_sentiment_pos_data[product])
		text_sentiment_neg_data[product]=getArray(text_sentiment_neg_data[product])
	return rating_data,review_sentiment_pos_data,review_sentiment_neg_data,text_sentiment_pos_data,text_sentiment_neg_data

def prepareData():
	survival_sales=readLines('data/survival-sales-1.txt')+readLines('data/survival-sales-2.txt')
	death_sales=readLines('data/death-sales.txt')
	survival_td=readLines('data/survival-time-difference.txt')
	death_td=readLines('data/death-time-difference.txt')
	survival_reviews=readLines('data/survival-days-1.txt')+readLines('data/survival-days-1.txt')
	death_reviews=readLines('data/death-days-1.txt')
	sales_data,lengths=processSalesData(survival_sales+death_sales)
	rating_data,review_sentiment_pos_data,review_sentiment_neg_data,text_sentiment_pos_data,text_sentiment_neg_data=processReviews(survival_reviews+death_reviews,lengths)
	return sales_data,rating_data,review_sentiment_pos_data,review_sentiment_neg_data,text_sentiment_pos_data,text_sentiment_neg_data

def convertStringToArray(string):
	a=[]
	for w in string.split():
		a.append(float(w))
	return np.array(a)

def processKDEData():
	kde_data=readLines('data/kde-data.txt')
	sales_data={}
	sentiment_data={}
	rating_data={}
	for line in kde_data:
		if len(line.strip())==0:
			continue
		parts=line.strip().split('\t')
		if parts[0].startswith('sales'):
			sales_data[parts[0]]=convertStringToArray(parts[1])
		elif parts[0].startswith('rating'):
			rating_data[parts[0]]=convertStringToArray(parts[1])
		elif parts[0].startswith('sentiment'):
			sentiment_data[parts[0]]=convertStringToArray(parts[1])
	sales_XY={}
	rating_XY={}
	sentiment_XY={}
	for name in sales_data:
		name=name[len('sales_'):-len('-XAXIS')]
		sales_XY[name]=[sales_data['sales_'+name+'-XAXIS'],sales_data['sales_'+name+'-YAXIS']]
	for name in rating_data:
		name=name[len('rating_'):-len('-XAXIS')]
		rating_XY[name]=[rating_data['rating_'+name+'-XAXIS'],rating_data['rating_'+name+'-YAXIS']]
	for name in sentiment_data:
		name=name[len('sentiment_'):-len('-XAXIS')]
		sentiment_XY[name]=[sentiment_data['sentiment_'+name+'-XAXIS'],sentiment_data['sentiment_'+name+'-YAXIS']]
	return sales_XY,rating_XY,sentiment_XY

def getDifferenceArray(a):
	diff_a=[0]
	for i in range(1,len(a),1):
		diff_a.append(a[i]-a[i-1])
	return np.array(diff_a)

def getTrainData(train_window_size):
	sales_XY,rating_XY,sentiment_XY=processKDEData()
	sales_trainX=[]
	rating_trainX=[]
	sentiment_trainX=[]
	diffx=[]
	Y=[]
	count=0
	for product in sales_XY.keys():
		osX=sales_XY[product][0]
		sX=getDifferenceArray(sales_XY[product][0])
		sY=sales_XY[product][1]
		if product in rating_XY:
			orX=rating_XY[product][0]
			rX=getDifferenceArray(rating_XY[product][0])
			rY=rating_XY[product][1]
		else:
			orX=osX
			rX=sX
			rY=np.array([0]*len(sX))
		if product in sentiment_XY:
			ostX=sentiment_XY[product][0]
			stX=getDifferenceArray(sentiment_XY[product][0])
			stY=sentiment_XY[product][1]
		else:
			ostX=osX
			stX=sX
			stY=np.array([0]*len(sX))
		for i in range(train_window_size,len(sales_XY[product][0]),1):
			s=np.concatenate([sX[i-train_window_size:i],sY[i-train_window_size:i]],axis=-1)
			diff_s_x=osX[i]
			idx=np.argwhere(orX>=osX[i])
			if len(idx)==0:
				idx=len(orX)
			idx=idx[0][0]
			start_index=max(idx-train_window_size,0)
			end_index=idx
			diff_r_x=osX[i]-orX[end_index]
			pad=np.array([0]*(train_window_size-end_index+start_index))
			r=np.concatenate([pad,rX[start_index:end_index],pad,rY[start_index:end_index]],axis=-1)
			idx=np.argwhere(ostX>=osX[i])
			if len(idx)==0:
				idx=len(ostX)
			idx=idx[0][0]
			start_index=max(idx-train_window_size,0)
			end_index=idx
			diff_st_x=osX[i]-ostX[end_index]
			pad=np.array([0]*(train_window_size-end_index+start_index))
			st=np.concatenate([pad,stX[start_index:end_index],pad,stY[start_index:end_index]],axis=-1)
			Y.append(sY[i])
			sales_trainX.append(s)
			rating_trainX.append(r)
			sentiment_trainX.append(st)
			diffx.append(np.array([diff_s_x,diff_r_x,diff_st_x]))
			count+=1
			if count%300000==0:
				print 'Saving'
				# np.savez_compressed('data/train_X_'+str(count),[sales_trainX,rating_trainX,sentiment_trainX])
				# np.savez_compressed('data/train_Y_'+str(count),Y)
				np.savez_compressed('data/train_xdifference_X'+str(count),diffx)
				sales_trainX=[]
				rating_trainX=[]
				sentiment_trainX=[]
				diffx=[]
				Y=[]
				print count
			# ss.append(s)
			# rs.append(r)
			# sts.append(st)
			# if count%50==0:
			# 	sales_trainX=np.concatenate([sales_trainX,ss],axis=0)
			# 	rating_trainX=np.concatenate([rating_trainX,rs],axis=0)
			# 	sentiment_trainX=np.concatenate([sentiment_trainX,sts],axis=0)
			# print count
		# sales_trainX=np.concatenate([sales_trainX,ss],axis=0)
		# rating_trainX=np.concatenate([rating_trainX,rs],axis=0)
		# sentiment_trainX=np.concatenate([sentiment_trainX,sts],axis=0)
		sales_XY.pop(product)
		if product in sentiment_XY:
			sentiment_XY.pop(product)
		if product in rating_XY:
			rating_XY.pop(product)
	# np.savez_compressed('data/train_X_'+str(count),[sales_trainX,rating_trainX,sentiment_trainX])
	# np.savez_compressed('data/train_Y_'+str(count),Y)
	np.savez_compressed('data/train_xdifference_X'+str(count),diffx)
	# sales_trainX=np.array(sales_trainX)
	# rating_trainX=np.array(rating_trainX)
	# sentiment_trainX=np.array(sentiment_trainX)
	# Y=np.array(Y)
	return sales_trainX,rating_trainX,sentiment_trainX,Y

def getEvaluationData(train_window_size):
	sales_XY,rating_XY,sentiment_XY=processKDEData()
	X=[]
	Y=[]
	count=0
	for product in sales_XY.keys():
		sX=getDifferenceArray(sales_XY[product][0])
		sY=sales_XY[product][1]
		X.append(sX)
		Y.append(sY)
		sales_XY.pop(product)
	return X,Y

def getAllEvaluationData(train_window_size):
	sales_XY,rating_XY,sentiment_XY=processKDEData()
	sales_trainX=[]
	rating_trainX=[]
	sentiment_trainX=[]
	diffx=[]
	Y=[]
	ids=sales_XY.keys()
	for product in sales_XY.keys():
		osX=sales_XY[product][0]
		sX=getDifferenceArray(sales_XY[product][0])
		sY=sales_XY[product][1]
		if product in rating_XY:
			orX=rating_XY[product][0]
			rX=getDifferenceArray(rating_XY[product][0])
			rY=rating_XY[product][1]
		else:
			orX=osX
			rX=sX
			rY=np.array([0]*len(sX))
		if product in sentiment_XY:
			ostX=sentiment_XY[product][0]
			stX=getDifferenceArray(sentiment_XY[product][0])
			stY=sentiment_XY[product][1]
		else:
			ostX=osX
			stX=sX
			stY=np.array([0]*len(sX))
		i=train_window_size
		s=np.concatenate([sX[i-train_window_size:i],sY[i-train_window_size:i]],axis=-1)
		diff_s_x=osX[i]
		idx=np.argwhere(orX>=osX[i])
		if len(idx)==0:
			idx=len(orX)
		idx=idx[0][0]
		start_index=max(idx-train_window_size,0)
		end_index=idx
		diff_r_x=osX[i]-orX[end_index]
		pad=np.array([0]*(train_window_size-end_index+start_index))
		r=np.concatenate([pad,rX[start_index:end_index],pad,rY[start_index:end_index]],axis=-1)
		idx=np.argwhere(ostX>=osX[i])
		if len(idx)==0:
			idx=len(ostX)
		idx=idx[0][0]
		start_index=max(idx-train_window_size,0)
		end_index=idx
		diff_st_x=osX[i]-ostX[end_index]
		pad=np.array([0]*(train_window_size-end_index+start_index))
		st=np.concatenate([pad,stX[start_index:end_index],pad,stY[start_index:end_index]],axis=-1)
		Y.append(sY[i])
		sales_trainX.append(s)
		rating_trainX.append(r)
		sentiment_trainX.append(st)
		diffx.append(np.array([diff_s_x,diff_r_x,diff_st_x]))
		sales_XY.pop(product)
		if product in sentiment_XY:
			sentiment_XY.pop(product)
		if product in rating_XY:
			rating_XY.pop(product)
	sales_trainX=np.array(sales_trainX)
	rating_trainX=np.array(rating_trainX)
	sentiment_trainX=np.array(sentiment_trainX)
	Y=np.array(Y)
	diffx=np.array(diffx)
	return sales_trainX,rating_trainX,sentiment_trainX,Y,diffx,ids

def mae(y_true,y_pred):
	return np.mean(np.fabs(y_true-y_pred))

def error_sales(sales_model,window_size,shape):
	X,Y=getEvaluationData(window_size)
	error=[]
	for i in range(len(X)):
		tY=Y[i][:window_size].reshape((-1,1))
		for j in range(window_size,len(Y[i]),1):
			inputX=np.concatenate([X[i][j-window_size:j].reshape((-1,1)),tY],axis=-1).reshape(shape)
			output=sales_model.predict(inputX).reshape((1,1))
			tY=np.concatenate([tY[1:,0].reshape((-1,1)),output],axis=0).reshape((-1,1))
			error.append(mae(Y[i][j].flatten(),output.flatten()))
		# print 'Finished',i,'/',len(X)
	return np.mean(error)

def error_final(final_model,sales_model,window_size,shape):
	X,Y=getEvaluationData(window_size)
	concatenated_embeddings=np.load('concatenated_embeddings.npz')['arr_0']
	sales_embedding_model=Model(inputs=sales_model.input,outputs=sales_model.get_layer('lstm_1').output)
	error=[]
	k=0
	for i in range(len(X)):
		tY=Y[i][:window_size].reshape((-1,1))
		for j in range(window_size,len(Y[i]),1):
			inputX=np.concatenate([X[i][j-window_size:j].reshape((-1,1)),tY],axis=-1).reshape(shape)
			output=sales_model.predict(inputX).reshape((1,1))
			sales_embedding=sales_embedding_model.predict(inputX)
			final_output=final_model.predict(np.concatenate([sales_embedding.reshape(-1,100),concatenated_embeddings[k][100:].reshape((-1,200))],axis=-1))
			tY=np.concatenate([tY[1:,0].reshape((-1,1)),output],axis=0).reshape((-1,1))
			error.append(mae(Y[i][j].flatten(),final_output.reshape((1,1)).flatten()))
			k+=1
		print 'Finished',i,'/',len(X)


	return np.mean(error)

def getProductPairs():
	death_pairs=readLines('data/death-time-difference.txt')
	survival_pairs=readLines('data/survival-time-difference.txt')
	pairs={'survival':[],'death':[]}
	for pair in death_pairs[1:]:
		p=pair.strip().split()
		pairs['death'].append((p[0],p[1]))
	for pair in survival_pairs[1:]:
		p=pair.strip().split()
		pairs['survival'].append((p[0],p[1]))
	return pairs

def getTrainingDataCombined(window_size=100):
	sales_XY,rating_XY,sentiment_XY=processKDEData()
	product_pairs=getProductPairs()
	trainX=[]
	trainY=[]
	for pair in product_pairs['death']+product_pairs['survival']:
		if pair[0] not in sales_XY or pair[1] not in sales_XY:
			continue
		if pair[0] not in rating_XY:
			rating_XY[pair[0]]=[np.array([0]*1024),np.array([0]*1024)]
		if pair[1] not in rating_XY:
			rating_XY[pair[1]]=[np.array([0]*1024),np.array([0]*1024)]
		if pair[0] not in sentiment_XY:
			sentiment_XY[pair[0]]=[np.array([0]*1024),np.array([0]*1024)]
		if pair[1] not in sentiment_XY:
			sentiment_XY[pair[1]]=[np.array([0]*1024),np.array([0]*1024)]
		for i in range(0,1024,window_size):
			if i+window_size+window_size>1023:
				break
			pair_X=[]
			pair_Y=[]
			for j in range(window_size):
				data_point=[]
				data_point.append(sales_XY[pair[0]][1][i+j])
				data_point.append(rating_XY[pair[0]][1][i+j])
				data_point.append(sentiment_XY[pair[0]][1][i+j])
				data_point.append(sales_XY[pair[1]][1][i+j])
				data_point.append(rating_XY[pair[1]][1][i+j])
				data_point.append(sentiment_XY[pair[1]][1][i+j])
				pair_X.append(data_point)
				pair_Y.append([sales_XY[pair[0]][1][i+window_size+j],sales_XY[pair[1]][1][i+window_size+j]])
			trainX.append(pair_X)
			trainY.append(pair_Y)
	return trainX,trainY




# a,b,c,d=getTrainData(400)
# print a.shape,b.shape,c.shape,d.shape

# sales_data,rating_data,review_sentiment_pos_data,review_sentiment_neg_data,text_sentiment_pos_data,text_sentiment_neg_data=prepareData()
# print sales_data.shape
# print rating_data.shape
# print review_sentiment_pos_data.shape
# print review_sentiment_neg_data.shape
# print text_sentiment_pos_data.shape
# print text_sentiment_neg_data.shape

# f=open('data_new.txt','w')
# for product in sales_data:
# 	f.write('sales_'+product+'\t')
# 	for i in sales_data[product]:
# 		f.write(str(i)+' ')
# 	f.write('\nrating_'+product+'\t')
# 	for i in rating_data[product]:
# 		f.write(str(i)+' ')
# 	f.write('\ntitlepos_'+product+'\t')
# 	for i in review_sentiment_pos_data[product]:
# 		f.write(str(i)+' ')
# 	f.write('\ntitleneg_'+product+'\t')
# 	for i in review_sentiment_neg_data[product]:
# 		f.write(str(i)+' ')
# 	f.write('\ntextpos_'+product+'\t')
# 	for i in text_sentiment_pos_data[product]:
# 		f.write(str(i)+' ')
# 	f.write('\ntextneg_'+product+'\t')
# 	for i in text_sentiment_neg_data[product]:
# 		f.write(str(i)+' ')
# 	f.write('\n')
# f.close()
