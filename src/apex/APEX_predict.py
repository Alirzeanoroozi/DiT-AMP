import numpy as np
import torch
import glob
from utils import make_vocab, onehot_encoding
import pandas as pd

pathogen_list = [
	'A. baumannii ATCC 19606',
	'E. coli ATCC 11775',
	'E. coli AIC221',
	'E. coli AIC222',
	'K. pneumoniae ATCC 13883',
	'P. aeruginosa PA01',
	'P. aeruginosa PA14',
	'S. aureus ATCC 12600',
	'S. aureus (ATCC BAA-1556) - MRSA',
	'vancomycin-resistant E. faecalis ATCC 700802',
	'vancomycin-resistant E. faecium ATCC 700221'
]
model_list = [torch.load(a_model, weights_only=False, map_location='cpu').eval() for a_model in glob.glob('./APEX_pathogen_models/APEX_*')]

max_len = 52 #maximum seq length; 52 = start character + maximum peptide length (50 aa) + end character; longer peptides will be truncated
word2idx, idx2word = make_vocab() #make amino acid vocabulary
#emb, AAindex_dict = AAindex('./aaindex1.csv', word2idx) #make amino acid embeddings

def predict(seqs):
	seq_list = np.array(seqs)
	seq_rep = onehot_encoding(seq_list, max_len, word2idx)  # make input
	X_seq = torch.LongTensor(seq_rep)

	#Use pretrained APEX models to predict species-specific antimicrobial activity (i.e., minimum inhibitory concentration [MIC]; unit: uM)
	#8 pretrained APEX models are provided, and predictions are averaged
	# Store predictions from each APEX model
	all_preds = []

	for model in model_list:
		AMP_pred = model(X_seq).detach().numpy()  # make predictions
		AMP_pred = 10 ** (6 - AMP_pred)  # transform back to MICs
		all_preds.append(AMP_pred)

	# Stack predictions: shape (n_models, n_pathogens)
	all_preds_array = np.vstack(all_preds)  # shape: (n_models, n_pathogens)

	# Add mean row at the end
	mean_pred = np.mean(all_preds_array, axis=0)
	# all_rows = np.vstack([all_preds_array, mean_pred[np.newaxis, :]])
	return mean_pred

print(predict(["KRGFGKKLRKRLKKFRNSIKKRLKNFNVVIPIPLPG", "KRGFGKKLRKRLKKFRNSIKKRLKNFNVVIPIPLPG"]))