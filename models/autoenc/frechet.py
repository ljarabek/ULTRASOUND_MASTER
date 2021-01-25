import numpy as np
from numpy.random import random
from scipy.linalg import sqrtm
import matplotlib.image as mpimg
import tqdm
from generator import *

from tensorflow.keras.models import load_model


base_path ='/media/tim/Elements/data/uz/'

train_path = base_path+ 'train/'
val_path = base_path+ 'val/'
test_path = base_path+ 'test/'

model = load_model('models')
train_generator = generator(train_path)
val_generator = generator(val_path)
test_generator = generator(test_path)

def calculate_fid(image1, image2):
	#images are meant to be numpy arrays ,below is the code for converting them into tensors
	#image1,image2 = tf.convert_to_tensor(image1), tf.convert_to_tensor(image2)
	mu1, sigma1 = image1.mean(axis=0), np.cov(image1, rowvar=False)
	mu2, sigma2 = image2.mean(axis=0), np.cov(image2, rowvar=False)

	dif = np.sum((mu1 - mu2)**2.0)
	covmean = sqrtm(sigma1.dot(sigma2))
	# if the values are complex
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = dif + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid
 
def frechetontest():
	all = 0
	for i in tqdm(range(10000)):
		n = next(test_generator)
		n = np.array(n)
		k = model.predict(n[0])
		summa = 0
		for j in range(30):
			## shape od k = (1, 1, 30, 32, 32), zato dvakrat nula
			fid = calculate_fid(k[0][0][j],n[1][j])
			summa += fid
		summa/=30
		all += summa
		print('Average FID: %.3f' % summa)
	return all/10000
		
all = frechetontest()
## average FID = 1.19
print(all)