import scipy.sparse as sp
import numpy as np
from base.Similarity_mark2.tversky import tversky_similarity as tversky
from base.Similarity_mark2.dot_product import dot_product as dot
from base.Similarity_mark2.cosine import cosine_similarity as cosine
from base.Similarity_mark2.jaccard import jaccard_similarity as jaccard, tanimoto_similarity as tanimoto
from base.Similarity_mark2.p3alpha_rp3beta import p3alpha_rp3beta_similarity as rp3beta
from base.Similarity_mark2.p3alpha_rp3beta import p3alpha_similarity as p3alpha
from base.Similarity_mark2.dice import dice_similarity as dice
from sklearn.preprocessing import normalize

COSINE = "cosine"
JACCARD = "jaccard"
AS_COSINE = "as_cosine"
P_3_AlPHA = "p3aplha"
TVERSKY = "tversky"
R_P_3_BETA = "rp3beta"
TANIMOTO = "tanimoto"
DICE = "dice"
