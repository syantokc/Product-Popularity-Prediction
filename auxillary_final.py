import cPickle
import numpy as np
# from keras.models import Model
# from nltk.corpus import sentiwordnet as swn, stopwords
#from matplotlib import pyplot as plt

# stopWords = set(stopwords.words('english'))

def readLines(filename):
	f=open(filename,'r')
	lines=f.readlines()
	f.close()
	return lines

def saveObject(o,filename):
	f=open(filename,'wb')
	cPickle.dump(o,f,cPickle.HIGHEST_PROTOCOL)
	f.close()

def readObject(filename):
	f=open(filename,'rb')
	o=cPickle.load(f)
	f.close()
	return o

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
	rating_data={}
	review_sentiment_pos_data={}
	review_sentiment_neg_data={}
	text_sentiment_pos_data={}
	text_sentiment_neg_data={}
	for line in kde_data:
		if len(line.strip())==0:
			continue
		parts=line.strip().split('\t')
		if parts[0].startswith('sales'):
			sales_data[parts[0]]=convertStringToArray(parts[1])
		elif parts[0].startswith('rating'):
			rating_data[parts[0]]=convertStringToArray(parts[1])
		elif parts[0].startswith('titlepos'):
			review_sentiment_pos_data[parts[0]]=convertStringToArray(parts[1])
		elif parts[0].startswith('titleneg'):
			review_sentiment_neg_data[parts[0]]=convertStringToArray(parts[1])
		elif parts[0].startswith('textpos'):
			text_sentiment_pos_data[parts[0]]=convertStringToArray(parts[1])
		elif parts[0].startswith('textneg'):
			text_sentiment_neg_data[parts[0]]=convertStringToArray(parts[1])
	sales_XY={}
	rating_XY={}
	review_sentiment_pos_XY={}
	review_sentiment_neg_XY={}
	text_sentiment_pos_XY={}
	text_sentiment_neg_XY={}
	for name in sales_data:
		name=name[len('sales_'):-len('-XAXIS')]
		sales_XY[name]=[sales_data['sales_'+name+'-XAXIS'],sales_data['sales_'+name+'-YAXIS']]
	for name in rating_data:
		name=name[len('rating_'):-len('-XAXIS')]
		rating_XY[name]=[rating_data['rating_'+name+'-XAXIS'],rating_data['rating_'+name+'-YAXIS']]
	for name in review_sentiment_pos_data:
		name=name[len('titlepos_'):-len('-XAXIS')]
		review_sentiment_pos_XY[name]=[review_sentiment_pos_data['titlepos_'+name+'-XAXIS'],review_sentiment_pos_data['titlepos_'+name+'-YAXIS']]
	for name in review_sentiment_neg_data:
		name=name[len('titleneg_'):-len('-XAXIS')]
		review_sentiment_neg_XY[name]=[review_sentiment_neg_data['titleneg_'+name+'-XAXIS'],review_sentiment_neg_data['titleneg_'+name+'-YAXIS']]
	for name in text_sentiment_pos_data:
		name=name[len('textpos_'):-len('-XAXIS')]
		text_sentiment_pos_XY[name]=[text_sentiment_pos_data['textpos_'+name+'-XAXIS'],text_sentiment_pos_data['textpos_'+name+'-YAXIS']]
	for name in text_sentiment_neg_data:
		name=name[len('textneg_'):-len('-XAXIS')]
		text_sentiment_neg_XY[name]=[text_sentiment_neg_data['textneg_'+name+'-XAXIS'],text_sentiment_neg_data['textneg_'+name+'-YAXIS']]
	return sales_XY,rating_XY,review_sentiment_pos_XY,review_sentiment_neg_XY,text_sentiment_pos_XY,text_sentiment_neg_XY

def getDifferenceArray(a):
	diff_a=[0]
	for i in range(1,len(a),1):
		diff_a.append(a[i]-a[i-1])
	return np.array(diff_a)

def getTrainData(train_window_size,mode='train'):
	sales_XY,rating_XY,review_sentiment_pos_XY,review_sentiment_neg_XY,text_sentiment_pos_XY,text_sentiment_neg_XY=processKDEData()
	if mode=='counts':
		product_length_counts=[]
		keys=sales_XY.keys()
		for product in keys:
			product_length_counts.append(len(sales_XY[product][0])-train_window_size)
		return keys,product_length_counts
	sales_trainX=[]
	rating_trainX=[]
	titlepos_trainX=[]
	titleneg_trainX=[]
	textpos_trainX=[]
	textneg_trainX=[]
	diffx=[]
	Y=[]
	count=0
	product_length_counts=[]
	keys=sales_XY.keys()
	if mode=='train':
		keys=keys[:int(0.9*len(keys))]
	elif mode=='val':
		keys=keys[int(0.9*len(keys)):]
	elif mode=='embeddings' or mode=='counts':
		pass
	for product in keys:
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
		if product in review_sentiment_pos_XY:
			otipX=review_sentiment_pos_XY[product][0]
			tipX=getDifferenceArray(review_sentiment_pos_XY[product][0])
			tipY=review_sentiment_pos_XY[product][1]
		else:
			otipX=osX
			tipX=sX
			tipY=np.array([0]*len(sX))
		if product in review_sentiment_neg_XY:
			otinX=review_sentiment_neg_XY[product][0]
			tinX=getDifferenceArray(review_sentiment_neg_XY[product][0])
			tinY=review_sentiment_neg_XY[product][1]
		else:
			otinX=osX
			tinX=sX
			tinY=np.array([0]*len(sX))
		if product in text_sentiment_pos_XY:
			otepX=text_sentiment_pos_XY[product][0]
			tepX=getDifferenceArray(text_sentiment_pos_XY[product][0])
			tepY=text_sentiment_pos_XY[product][1]
		else:
			otepX=osX
			tepX=sX
			tepY=np.array([0]*len(sX))
		if product in text_sentiment_neg_XY:
			otenX=text_sentiment_neg_XY[product][0]
			tenX=getDifferenceArray(text_sentiment_neg_XY[product][0])
			tenY=text_sentiment_neg_XY[product][1]
		else:
			otenX=osX
			tenX=sX
			tenY=np.array([0]*len(sX))
		product_length_counts.append(0)
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

			idx=np.argwhere(otipX>=osX[i])
			if len(idx)==0:
				idx=len(otipX)
			idx=idx[0][0]
			start_index=max(idx-train_window_size,0)
			end_index=idx
			diff_tip_x=osX[i]-otipX[end_index]
			pad=np.array([0]*(train_window_size-end_index+start_index))
			tip=np.concatenate([pad,tipX[start_index:end_index],pad,tipY[start_index:end_index]],axis=-1)
			
			idx=np.argwhere(otinX>=osX[i])
			if len(idx)==0:
				idx=len(otinX)
			idx=idx[0][0]
			start_index=max(idx-train_window_size,0)
			end_index=idx
			diff_tin_x=osX[i]-otinX[end_index]
			pad=np.array([0]*(train_window_size-end_index+start_index))
			tin=np.concatenate([pad,tinX[start_index:end_index],pad,tinY[start_index:end_index]],axis=-1)
			
			idx=np.argwhere(otepX>=osX[i])
			if len(idx)==0:
				idx=len(otepX)
			idx=idx[0][0]
			start_index=max(idx-train_window_size,0)
			end_index=idx
			diff_tep_x=osX[i]-otepX[end_index]
			pad=np.array([0]*(train_window_size-end_index+start_index))
			tep=np.concatenate([pad,tepX[start_index:end_index],pad,tepY[start_index:end_index]],axis=-1)
			
			idx=np.argwhere(otenX>=osX[i])
			if len(idx)==0:
				idx=len(otenX)
			idx=idx[0][0]
			start_index=max(idx-train_window_size,0)
			end_index=idx
			diff_ten_x=osX[i]-otenX[end_index]
			pad=np.array([0]*(train_window_size-end_index+start_index))
			ten=np.concatenate([pad,tenX[start_index:end_index],pad,tenY[start_index:end_index]],axis=-1)
			
			Y.append(sY[i])
			sales_trainX.append(s)
			rating_trainX.append(r)
			titlepos_trainX.append(tip)
			titleneg_trainX.append(tin)
			textpos_trainX.append(tep)
			textneg_trainX.append(ten)
			diffx.append(np.array([diff_s_x,diff_r_x,diff_tip_x,diff_tin_x,diff_tep_x,diff_ten_x]))
			count+=1
			if mode=='embeddings':
				break
			if mode=='counts':
				product_length_counts[-1]+=1
				continue
			if count%200000==0:
				print 'Saving'
				np.savez_compressed('data/'+mode+'_X_new_'+str(count),[sales_trainX,rating_trainX,titlepos_trainX,titleneg_trainX,textpos_trainX,textneg_trainX])
				np.savez_compressed('data/'+mode+'_Y_new_'+str(count),Y)
				np.savez_compressed('data/'+mode+'_xdifference_X_new__'+str(count),diffx)
				sales_trainX=[]
				rating_trainX=[]
				titlepos_trainX=[]
				titleneg_trainX=[]
				textpos_trainX=[]
				textneg_trainX=[]
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
		if product in rating_XY:
			rating_XY.pop(product)
		if product in review_sentiment_pos_XY:
			review_sentiment_pos_XY.pop(product)
		if product in review_sentiment_neg_XY:
			review_sentiment_neg_XY.pop(product)
		if product in text_sentiment_pos_XY:
			text_sentiment_pos_XY.pop(product)
		if product in text_sentiment_neg_XY:
			text_sentiment_neg_XY.pop(product)
	if mode=='embeddings':
		return [sales_trainX,rating_trainX,titlepos_trainX,titleneg_trainX,textpos_trainX,textneg_trainX],Y,keys
	if mode=='counts':
		return keys,product_length_counts
	np.savez_compressed('data/'+mode+'_X_new_'+str(count),[sales_trainX,rating_trainX,titlepos_trainX,titleneg_trainX,textpos_trainX,textneg_trainX])
	np.savez_compressed('data/'+mode+'_Y_new_'+str(count),Y)
	np.savez_compressed('data/'+mode+'_xdifference_X_new__'+str(count),diffx)
	return keys
	# sales_trainX=np.array(sales_trainX)
	# rating_trainX=np.array(rating_trainX)
	# sentiment_trainX=np.array(sentiment_trainX)
	# Y=np.array(Y)
	# return sales_trainX,rating_trainX,sentiment_trainX,Y

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

def getLossValues(name):
	full_output=readObject('full_output_combined_'+name+'.pickle')
	keys=readLines('keys_'+name+'.txt')
	sales_XY,rating_XY,review_sentiment_pos_XY,review_sentiment_neg_XY,text_sentiment_pos_XY,text_sentiment_neg_XY=processKDEData()
	train_error=[]
	val_error=[]
	print len(full_output.keys())
	for key in keys[:int(0.9*len(keys))]:
		key=key.strip()
		train_error.append(np.mean(np.fabs(np.array(full_output[key]).flatten()[400:]-np.array(sales_XY[key][1]).flatten()[400:])))	
	for key in keys[int(0.9*len(keys)):]:
		key=key.strip()
		val_error.append(np.mean(np.fabs(np.array(full_output[key]).flatten()[400:]-np.array(sales_XY[key][1]).flatten()[400:])))
	return np.mean(train_error),np.mean(val_error)

def full_output(model,window_size,name):
	# keys,counts=getTrainData(window_size,'counts')
	# key_count={}
	# for i in range(len(keys)):
	# 	key_count[keys[i]]=counts[i]
	# saveObject(key_count,'key_count.pickle')
	key_count=readObject('key_count.pickle')
	error=[]
	k=0
	internal_index=0
	loaded_indexX=['data/train_X_new_200000','data/train_X_new_400000','data/train_X_new_600000','data/train_X_new_800000','data/train_X_new_894816','data/val_X_new_99840']
	loaded_indexY=['data/train_Y_new_200000','data/train_Y_new_400000','data/train_Y_new_600000','data/train_Y_new_800000','data/train_Y_new_894816','data/val_Y_new_99840']
	full_output={}
	count=0
	change_k=[0,200000,400000,600000,800000,894816]
	for key in key_count:
		if len(change_k)>0 and k==change_k[0]:
			change_k.pop(0)
			numX=loaded_indexX.pop(0)
			numY=loaded_indexY.pop(0)
			X=np.load(str(numX)+'.npz')['arr_0']
			Y=np.load(str(numY)+'.npz')['arr_0']
			sX=np.array(X[0]).reshape((-1,2,400,1))
			rX=np.array(X[1]).reshape((-1,2,400,1))
			rX=np.concatenate([rX[:,0,:,:],rX[:,1,:,:]],axis=-1).reshape((-1,20,40))
			tipX=np.array(X[2]).reshape((-1,2,400,1))
			tipX=np.concatenate([tipX[:,0,:,:],tipX[:,1,:,:]],axis=-1).reshape((-1,20,40))
			tinX=np.array(X[3]).reshape((-1,2,400,1))
			tinX=np.concatenate([tinX[:,0,:,:],tinX[:,1,:,:]],axis=-1).reshape((-1,20,40))
			tepX=np.array(X[4]).reshape((-1,2,400,1))
			tepX=np.concatenate([tepX[:,0,:,:],tepX[:,1,:,:]],axis=-1).reshape((-1,20,40))
			tenX=np.array(X[5]).reshape((-1,2,400,1))
			tenX=np.concatenate([tenX[:,0,:,:],tenX[:,1,:,:]],axis=-1).reshape((-1,20,40))
			internal_index=0
			print 'Loaded',numX,numY
		sX_Y=sX[internal_index,1,:,:].reshape((400,))
		sX_Y=np.concatenate([sX_Y,np.array([0]*(key_count[key]))],axis=0)
		for j in range(key_count[key]):
			input_sx=np.concatenate([sX[internal_index,0,:,:],sX_Y[len(sX_Y)-window_size:].reshape((-1,1))],axis=-1).reshape((1,20,40))
			inputX=[input_sx,rX[internal_index],tipX[internal_index],tinX[internal_index],tepX[internal_index],tenX[internal_index]]
			for i in range(len(inputX)):
				inputX[i]=inputX[i].reshape((-1,20,40))
			output_val=model.predict(inputX).reshape((1,))[0]
			error.append(np.abs(output_val-Y[internal_index]))
			sX_Y[window_size+j]=output_val
			internal_index+=1
			k+=1
			if len(change_k)>0 and k==change_k[0]:
                        	change_k.pop(0)
                        	numX=loaded_indexX.pop(0)
                        	numY=loaded_indexY.pop(0)
                        	X=np.load(str(numX)+'.npz')['arr_0']
                        	Y=np.load(str(numY)+'.npz')['arr_0']
				sX=np.array(X[0]).reshape((-1,2,400,1))
				rX=np.array(X[1]).reshape((-1,2,400,1))
				rX=np.concatenate([rX[:,0,:,:],rX[:,1,:,:]],axis=-1).reshape((-1,20,40))
				tipX=np.array(X[2]).reshape((-1,2,400,1))
				tipX=np.concatenate([tipX[:,0,:,:],tipX[:,1,:,:]],axis=-1).reshape((-1,20,40))
				tinX=np.array(X[3]).reshape((-1,2,400,1))
				tinX=np.concatenate([tinX[:,0,:,:],tinX[:,1,:,:]],axis=-1).reshape((-1,20,40))
				tepX=np.array(X[4]).reshape((-1,2,400,1))
				tepX=np.concatenate([tepX[:,0,:,:],tepX[:,1,:,:]],axis=-1).reshape((-1,20,40))
				tenX=np.array(X[5]).reshape((-1,2,400,1))
				tenX=np.concatenate([tenX[:,0,:,:],tenX[:,1,:,:]],axis=-1).reshape((-1,20,40))
				internal_index=0
				print 'Loaded',numX,numY
		full_output[key]=sX_Y
		count+=1
		print key,count,k
	saveObject(full_output,'full_output_combined_'+name+'.pickle')
	print 'MAE of',name,':',np.mean(error)
	return np.mean(error)

# a,b,c,d=getTrainData(400)
# print a.shape,b.shape,c.shape,d.shape

# sales_data,rating_data,review_sentiment_pos_data,review_sentiment_neg_data,text_sentiment_pos_data,text_sentiment_neg_data=prepareData()
# # print sales_data.shape
# # print rating_data.shape
# # print review_sentiment_pos_data.shape
# # print review_sentiment_neg_data.shape
# # print text_sentiment_pos_data.shape
# # print text_sentiment_neg_data.shape

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

# getTrainData(400,'train')

print getLossValues('LSTM')
print getLossValues('GRU')
print getLossValues('BGRU_GRU')
print getLossValues('BLSTM_LSTM')