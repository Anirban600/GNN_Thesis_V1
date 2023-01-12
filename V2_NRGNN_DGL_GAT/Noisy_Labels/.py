import time
import argparse
import pickle
from utils import noisify_with_P

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--label_path', typr='str', default='', help='')
parser.add_argument('--out_path', typr='str', default='', help='')
parser.add_argument('--ptb_rate', type=float, default=0.2, help="noise ptb_rate")
parser.add_argument('--noise', type=str, default='pair', choices=['uniform', 'pair'], help='type of noises')

args = parser.parse_known_args()[0]

with open(args.label_path, 'rb') as f:
    labels, idx_train = pickle.load(f)

# add noise to the labels
ptb = args.ptb_rate
nclass = labels.max() + 1
train_labels = labels[idx_train]
noise_y, P = noisify_with_P(train_labels,nclass, ptb, 10, args.noise) 
noise_labels = labels.copy()
noise_labels[idx_train] = noise_y

with open(args.out_path, 'wb') as f:
    pickle.dump(noise_labels, f)