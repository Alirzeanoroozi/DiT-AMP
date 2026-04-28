import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import glob
import sys
from pathlib import Path
try:
    from .utils import make_vocab, onehot_encoding
except ImportError:
    from utils import make_vocab, onehot_encoding

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_MODEL_GLOB = str(_SCRIPT_DIR / "APEX_pathogen_models" / "APEX_*")


def _load_model(path):
    try:
        return torch.load(path, weights_only=False, map_location="cpu").eval()
    except TypeError:
        # Older torch versions do not support weights_only.
        return torch.load(path, map_location="cpu").eval()


model_list = [_load_model(a_model) for a_model in glob.glob(_MODEL_GLOB)]
max_len = 52 #maximum seq length; 52 = start character + maximum peptide length (50 aa) + end character; longer peptides will be truncated
word2idx, idx2word = make_vocab() #make amino acid vocabulary
#emb, AAindex_dict = AAindex('./aaindex1.csv', word2idx) #make amino acid embeddings

def predict(seq):
	seq = np.array([seq])
	X_seq = torch.LongTensor(onehot_encoding(seq, max_len, word2idx))

	#Use pretrained APEX models to predict species-specific antimicrobial activity (i.e., minimum inhibitory concentration [MIC]; unit: uM)
	#8 pretrained APEX models are provided, and predictions are averaged
	# Store predictions from each APEX model
	all_preds = []

	for model in model_list:
		AMP_pred = model(X_seq).detach().numpy()  # make predictions
		AMP_pred = 10 ** (6 - AMP_pred)  # transform back to MICs
		all_preds.append(AMP_pred)

	return np.mean(all_preds)

if __name__ == "__main__":
    print(predict("MALWMRLLPLLALLALWGPDPAAA"))