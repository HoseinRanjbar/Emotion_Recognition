import argparse
import time
import os
import torch
import torch.nn
import torch.nn as nn
import numpy as np
import datetime as dt
import _pickle as pickle
import pandas as pd
from skimage import io
from torchvision import transforms

from transformer import make_model as TRANSFORMER
from dataloader import loader #For SLR
from tools.utils import Batch


#Evaluation metrics
from tools.bleu import compute_bleu
from tools.rouge import rouge

#Lavenshtein distance (WER)
from jiwer import wer

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


###
# Arg parsing
##############

parser = argparse.ArgumentParser(description='Evaluation')

parser.add_argument('--data', type=str, default=os.path.join('home','artaheri.sharif','phoenix2014-release','phoenix-2014-multisigner'),
                    help='location of the video')


parser.add_argument('--idx', type=int, default='',
                    help='index of the video')

parser.add_argument('--model_path', type=str, default=os.path.join("EXPERIMENTATIONS"),
                    help='location of the  entire trained model')

parser.add_argument('--csv_path', type=str, default='',
                    help='location of the csv file')

parser.add_argument('--lookup_table', type=str, default=os.path.join('home','artaheri.sharif','SLR_lookup_pickle.txt'),
                    help='location of the words lookup table')


parser.add_argument('--rescale', type=int, default=224,
                    help='rescale data images. NOTE: use same image size as the training or else you get worse results.')


parser.add_argument('--image_type', type=str, default='rgb',
                    help='Evaluate on rgb/grayscale images')


parser.add_argument('--hand_query', action='store_true',
                    help='Set hand cropped image as a query for transformer network.')

parser.add_argument('--data_stats', type=str, default=None,
                    help='Normalize images using the dataset stats (mean/std).')

parser.add_argument('--hand_stats', type=str, default=None,
                    help='Normalize images using the dataset stats (mean/std).')

parser.add_argument('--emb_type', type=str, default='2d',
                    help='Type of image embeddings 2d or 3d.')

parser.add_argument('--emb_network', type=str, default='mb2',
                    help='Image embeddings network: mb2/mb2-ssd/rcnn')

parser.add_argument('--decoding', type=str, default='greedy',
                    help='Decoding method (greedy/beam).')

parser.add_argument('--n_beam', type=int, default=4,
                    help='Beam width when using bean search for decoding.')

parser.add_argument('--rel_window', type=int, default=None)

parser.add_argument('--bleu', action='store_true',
                    help='Use bleu for evaluation.')

parser.add_argument('--rouge', action='store_true',
                    help='Use rouge for evaluation.')




#----------------------------------------------------------------------------------------


#Same seed for reproducibility)
parser.add_argument('--seed', type=int, default=1111, help='random seed')

#Save folder with the date
start_date = dt.datetime.now().strftime("%Y-%m-%d-%H.%M")
print ("Start Time: "+start_date)

args = parser.parse_args()

#Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

#experiment_path = PureWindowsPath('EXPERIMENTATIONS\\' + start_date)
#-------------------------------------------------------------------------------


#Load stats
if(args.data_stats):
    args.data_stats = torch.load(args.data_stats, map_location=torch.device('cpu'))

if(args.hand_stats):
    args.hand_stats = torch.load(args.hand_stats, map_location=torch.device('cpu'))

if (args.image_type == 'rgb'):
    channels = 3
elif(args.image_type == 'grayscale'):
    channels = 1
else:
    print('Image type is not supported!')
    quit(0)



#Retrieve size of target vocab
with open(args.lookup_table, 'rb') as pickle_file:
   vocab = pickle.load(pickle_file)

vocab_size = len(vocab)

#Switch keys and values of vocab to easily look for words
vocab = {y:x for x,y in vocab.items()}

print('vocabulary size:' + str(vocab_size))


#Blank token index
blank_index = 1232


#-------------------------------------------------------------------------------


#Run on GPU
if torch.cuda.is_available():
    print("Using GPU")
    print('Device name:{}',torch.cuda.get_device_name(0))
    device = torch.device("cuda:0")
else:
#Run on CPU
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")


#-------------------------------------------------------------------------------


#Load the whole model with state dict
model = TRANSFORMER(tgt_vocab=vocab_size, n_stacks=2, n_units=1280,
                            n_heads=10, d_ff=2048, dropout=0.3, image_size=224,
                                                       emb_type='2d', emb_network='mb2')
#model.load_state_dict(torch.load(args.model_path))
model.load_state_dict(torch.load(args.model_path)['model_state_dict'])

#Load entire model w/ weights
#model = torch.load(args.model_path, map_location=device)

###########################
model = model.to(device)
print("Model successfully loaded")

model.eval()   # Set model to evaluate mode
#print ("Evaluating..")

start_time = time.time()


#Save translation and reference sentences
translation_corpus = []
reference_corpus = []

total_wer_score = 0.0
count = 0

trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.rescale, args.rescale)),
                transforms.ToTensor(),
                transforms.Normalize(mean=args.data_stats['mean'], std=args.data_stats['std'])
                ])

annotations = pd.read_csv(args.csv_path)
name = annotations.iloc[args.idx, 0].split('|')[0]
video_path = os.path.join(args.data, name,'1')
seq_length=len(os.listdir(video_path))//2
trsf_images = torch.zeros((seq_length,3,224,224))
images = os.listdir(video_path)

for i in range(seq_length):
    img=images[i*2]
    img_name=os.path.join(video_path,'{}{:03d}'.format(img[:-9],i*2)+'-0.png')

    if(io.imread(img_name).shape[2] == 3):
        trsf_images[i] = trans(io.imread(img_name, plugin='pil'))
    else:
        trsf_images[i] = trans(io.imread(img_name, plugin='pil')[:, :, :3])

translation = annotations.iloc[args.idx, 0].split('|')[-1]
translation = translation.split(' ')
tran=[]
with open(args.lookup_table, 'rb') as pickle_file:
            lookup_table = pickle.load(pickle_file)

for word in translation:
    #Get index of the current word if it exists in dict
    if(word in lookup_table.keys()):
        tran.append(lookup_table[word])
    else:
        #If words doesnt exist in train vocab then <unk>
        tran.append(0)
x=torch.tensor(trsf_images)
x_lengths=[np.shape(trsf_images)[0]]
y=tran
y_lengths=[len(tran)]
hand_regions=None
hand_lengths=None



if(args.hand_query):
        hand_regions = hand_regions.to(device)
else:
        hand_regions = None

y= torch.Tensor(y).to(device)
#y = torch.from_numpy(y).to(device)
x = x.to(device)

x=x.unsqueeze(0)
y=y.unsqueeze(0)

batch = Batch(x_lengths, y_lengths, hand_lengths, trg=None, DEVICE=device, emb_type=args.emb_type, fixed_padding=None, rel_window=args.rel_window)

print('x.size:',x.size())
print('y_lengths:',y_lengths)
print('x_length:',x_lengths)

#with torch.no_grad():

output, output_context, output_hand = model.forward(x, batch.src_mask, batch.rel_mask, hand_regions)

#CTC loss expects (Seq, batch, vocab)
if(args.hand_query):
    output = output.transpose(0,1)
    output_context = output_context.transpose(0,1)
    output_hand = output_hand.transpose(0,1)
else:
    output = output_context.transpose(0,1)

#Predicted words with highest prob
_, pred = torch.max(output, dim=-1)

#Remove <BLANK>
#pred = pred[pred != blank_index]


x_lengths = torch.IntTensor(x_lengths)
y_lengths = torch.IntTensor(y_lengths)

decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=output.cpu().detach().numpy(),
                    sequence_length=x_lengths.cpu().detach().numpy(), merge_repeated=False, beam_width=10, top_paths=1)

pred = decodes[0]

pred = tf.sparse.to_dense(pred).numpy()

#Loop over translations and references
for j in range(len(y)):

    ys = y[j, :y_lengths[j]]
    p = pred[j]

    #Remove <UNK> token
    p = p[p != 0]
    ys = ys[ys != 0]

    hyp = (' '.join([vocab[x.item()] for x in p]))
    gt = (' '.join([vocab[x.item()] for x in ys]))

    total_wer_score += wer(gt, hyp)
    count += 1

    #Convert index tokens to words
    translation_corpus.append(hyp)
    reference_corpus.append(gt)

#Free some memory
#NOTE: this helps alot in avoiding cuda out of memory
del x, y, batch

assert len(translation_corpus) == len(reference_corpus)

end_time = time.time()
print('inference time:{}s'.format(end_time-start_time))
print('WER score:'+str(total_wer_score/count))
print('glosses:',translation_corpus)
print('ground truth', reference_corpus)

if(args.bleu):

    #Default return
    #NOTE: bleu score of camgoz results is slightly better than ntlk -> use it instead
    #bleu_4 = corpus_bleu(reference_corpus, translation_corpus)
    bleu_4, _, _, _, _, _ = compute_bleu(reference_corpus, translation_corpus, max_order=4)

    #weights = (1.0/1.0, )
    bleu_1, _, _, _, _, _ = compute_bleu(reference_corpus, translation_corpus, max_order=1)

    #weights = (1.0/2.0, 1.0/2.0, )
    #bleu_2 = corpus_bleu(reference_corpus, translation_corpus, weights)
    bleu_2, _, _, _, _, _ = compute_bleu(reference_corpus, translation_corpus, max_order=2)

    #weights = (1.0/3.0, 1.0/3.0, 1.0/3.0,)
    #bleu_3 = corpus_bleu(reference_corpus, translation_corpus, weights)
    bleu_3, _, _, _, _, _ = compute_bleu(reference_corpus, translation_corpus, max_order=3)

    log_str = 'Bleu Evaluation: ' + '\t' \
    + 'Bleu_1: ' + str(bleu_1) + '\t' \
    + 'Bleu_2: ' + str(bleu_2) + '\t' \
    + 'Bleu_3: ' + str(bleu_3) + '\t' \
    + 'Bleu_4: ' + str(bleu_4)

    print(log_str)

    if(args.save):
        #Save evaluation results in a log file
        with open(os.path.join(args.save_path, 'log.txt'), 'a') as f:
            f.write(log_str+'\n')

if(args.rouge):

    reference_corpus = [" ".join(reference) for reference in reference_corpus]
    translation_corpus = [" ".join(hypothesis) for hypothesis in translation_corpus]

    score = rouge(translation_corpus, reference_corpus)
    print(score["rouge_l/f_score"])

    log_str = 'Rouge Evaluation: ' + '\t'
    print(log_str)

    if(args.save):
        #Save evaluation results in a log file
        with open(os.path.join(args.save_path, 'log.txt'), 'a') as f:
            f.write(log_str+'\n')
