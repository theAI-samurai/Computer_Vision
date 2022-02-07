# Created by Samet Cetin.
# Contact: cetin.samet@outlook.com

import numpy as np
from PIL import Image
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
from DeepLearningModels.zeroshotlearning.src.feature_extractor import get_model, get_features
from DeepLearningModels.zeroshotlearning.src.train import load_keras_model


WORD2VECPATH = "/home/ankit/Downloads/DeepLearningModels/zeroshotlearning/data/class_vectors.npy"
MODELPATH = "/home/ankit/Downloads/DeepLearningModels/zeroshotlearning/model/"


# READ IMAGE
IMAGEPATH = "/home/ankit/Downloads/DeepLearningModels/zeroshotlearning/test.jpg"
img = Image.open(IMAGEPATH).resize((224, 224))

# LOAD PRETRAINED VGG16 MODEL FOR FEATURE EXTRACTION
vgg_model = get_model()

print(" \n ***** model loaded successfully ****** \n")
# EXTRACT IMAGE FEATURE
img_feature = get_features(vgg_model, img)
# L2 NORMALIZE FEATURE
img_feature = normalize(img_feature, norm='l2')

# LOAD ZERO-SHOT MODEL
model = load_keras_model(model_path=MODELPATH)
# MAKE PREDICTION
pred = model.predict(img_feature)
print('-------------------------------everything is fine uptill here ------------------')
# LOAD CLASS WORD2VECS
z= np.load(WORD2VECPATH)
class_vectors = sorted(np.load(WORD2VECPATH, allow_pickle=True), key=lambda x: x[0])
classnames, vectors = zip(*class_vectors)
classnames = list(classnames)
vectors = np.asarray(vectors, dtype=np.float)

# PLACE WORD2VECS IN KDTREE
tree = KDTree(vectors)
# FIND CLOSEST WORD2VEC and GET PREDICTION RESULT
dist, index = tree.query(pred, k=5)
pred_labels = [classnames[idx] for idx in index[0]]

# PRINT RESULT
print()
print("--- Top-5 Prediction ---")
for i, classname in enumerate(pred_labels):
    print("%d- %s" %(i+1, classname))
print()