import argparse
import time
import os
import torch
import torch.nn
import torch.nn as nn
import numpy as np
import datetime as dt
import pandas as pd
import json
import copy

from transformer import make_model as TRANSFORMER
from dataloader import loader 
from tools.utils import path_data, Batch

#Progress bar to visualize training progress
import progressbar

##############
# Arg parsing
##############

parser = argparse.ArgumentParser(description='Evaluation')


parser.add_argument('--data', type=str, help='location of the test data corpus')

parser.add_argument('--model_path', type=str, default=os.path.join("EXPERIMENTATIONS"),
                    help='location of the  entire trained model')

parser.add_argument('--window_size', type=int, default=10)

parser.add_argument('--num_classes', type=int, default=7)

parser.add_argument('--classifier_hidden_size', type=int, default=512)

parser.add_argument('--recognition', type=str, default='emotion')

parser.add_argument('--save_dir', type=str, default='')

parser.add_argument('--random_drop_probability', type=float, default=None,
                    help='probability of frame random drop/0-1 or None')

parser.add_argument('--uniform_drop_probability', type=float, default=0.5,
                    help='probability of frame random drop/0-1 or None')

parser.add_argument('--lookup_table', type=str, default=os.path.join('data','slr_lookup.txt'),
                    help='location of the words lookup table')

parser.add_argument('--hidden_size', type=int, default=1280,
                    help='size of hidden layers. NOTE: This must be a multiple of n_heads.')

parser.add_argument('--num_layers', type=int, default=2,
                    help='number of transformer blocks')

parser.add_argument('--n_heads', type=int, default=8,
                    help='number of self attention heads')

parser.add_argument('--rescale', type=int, default=224,
                    help='rescale data images. NOTE: use same image size as the training or else you get worse results.')

#Put to 0 to avoid memory segementation fault
parser.add_argument('--num_workers', type=int, default=10,
                    help='NOTE: put num of workers to 0 to avoid memory saturation.')

parser.add_argument('--image_type', type=str, default='rgb',
                    help='Evaluate on rgb/grayscale images')

parser.add_argument('--show_sample', action='store_true',
                    help='Show a sample a preprocessed data.')

parser.add_argument('--batch_size', type=int, default=1,
                    help='size of one minibatch')

parser.add_argument('--save', default='False',
                    help='save the results of the evaluation')

parser.add_argument('--d_ff', type=int,default=2048)

parser.add_argument('--hand_query', action='store_true',
                    help='Set hand cropped image as a query for transformer network.')

parser.add_argument('--data_stats', type=str, default=None,
                    help='Normalize images using the dataset stats (mean/std).')

parser.add_argument('--hand_stats', type=str, default=None,
                    help='Normalize images using the dataset stats (mean/std).')

parser.add_argument('--dp_keep_prob', type=float, default=0.7,
                    help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                    (dp_keep_prob=1 means no dropout)')

parser.add_argument('--emb_type', type=str, default='2d',
                    help='Type of image embeddings 2d or 3d.')

parser.add_argument('--emb_network', type=str, default='mb2',
                    help='Image embeddings network: mb2/mb2-ssd/rcnn')

parser.add_argument('--rel_window', type=int, default=None)

parser.add_argument('--heatmap', action='store_true',
                    help='produce heatmap.')

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
save_path = os.path.join(args.save_dir, 'evaluation', start_date)

# Creates an experimental directory and dumps all the args to a text file
if(args.save):
    if(os.path.exists(save_path)):
        print('Evaluation already exists..')
    else:
        os.makedirs(save_path)

    print ("\nPutting log in EVALUATION/%s"%start_date)

    #Dump all configurations/hyperparameters in txt
    with open (os.path.join(save_path,'eval_config.txt'), 'w') as f:
        f.write('Experimentation done at: '+ str(start_date)+' with current configurations:\n')
        for arg in vars(args):
            f.write(arg+' : '+str(getattr(args, arg))+'\n')

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

test_csv = pd.read_csv(os.path.join('./tools/data','test.csv'))
val_csv = pd.read_csv(os.path.join('./tools/data','dev.csv'))

with open(args.lookup_table, 'r') as file:
    lookup_table = json.load(file)

#No data augmentation for test data
test_dataloader, test_size = loader(csv_file=test_csv,
                root_dir=args.data,
                lookup_table=lookup_table,
                recognition=args.recognition,
                rescale = args.rescale,
                random_drop = args.random_drop_probability,
                uniform_drop = args.uniform_drop_probability,
                batch_size = args.batch_size,
                num_workers = args.num_workers,
                show_sample = args.show_sample,
                istrain=False,
                fixed_padding=False,
                hand_dir=None,
                data_stats=args.data_stats,
                hand_stats=args.hand_stats,
                channels=channels
                )

#No data augmentation for test data
valid_dataloader, valid_size = loader(csv_file=val_csv,
                root_dir=args.data,
                lookup_table=lookup_table,
                recognition=args.recognition,
                rescale = args.rescale,
                random_drop = args.random_drop_probability,
                uniform_drop = args.uniform_drop_probability,
                batch_size = args.batch_size,
                num_workers = args.num_workers,
                show_sample = args.show_sample,
                istrain=False,
                fixed_padding=False,
                hand_dir=None,
                data_stats=args.data_stats,
                hand_stats=args.hand_stats,
                channels=channels
                )

print('Test dataset size: '+str(test_size))
print('Valid dataset size: '+str(valid_size))

#Loop through test and val sets
dataloaders = [valid_dataloader, test_dataloader]
sizes = [valid_size, test_size]
dataset = ['valid', 'test']

#Blank token index
blank_index = 1232

#-------------------------------------------------------------------------------
#Run on GPU
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda:0")
else:
#Run on CPU
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")
#-------------------------------------------------------------------------------


#Load the whole model with state dict
model = TRANSFORMER(num_classes=args.num_classes, n_stacks=args.num_layers, n_units=args.hidden_size, n_heads=args.n_heads , window_size = args.window_size, d_ff=args.d_ff, dropout=1.-args.dp_keep_prob, image_size=args.rescale,
                            classifier_hidden_dim= args.classifier_hidden_size,emb_type=args.emb_type, emb_network=args.emb_network, hand_pretrained=None, channels=3)
model.load_state_dict(torch.load(args.model_path)['model_state_dict'])

#Load entire model w/ weights
model = model.to(device)
print("Model successfully loaded")

model.eval()   # Set model to evaluate mode
print ("Evaluating..")

start_time = time.time()


for d in range(len(sizes)):

    dataloader = dataloaders[d]
    size = sizes[d]
    print(dataset[d])

    feature_file = np.zeros((size,args.hidden_size))

    #For progress bar
    bar = progressbar.ProgressBar(maxval=size, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    i = 0
    count = 0
    total_accuracy = 0.0
    total_loss = 0.0

    #Save translation and reference sentences
    predictions = []
    references = []

    #Loop over minibatches
    for step, (x, x_lengths, y, hand_regions, hand_lengths) in enumerate(dataloader):
        
        #Update progress bar with every iter
        i += len(x)
        bar.update(i)

        if(args.hand_query):
             hand_regions = hand_regions.to(device)
        else:
             hand_regions = None

        y = y.to(device)
        x = x.to(device)

        batch = Batch(x_lengths, hand_lengths, trg=None, DEVICE=device, emb_type=args.emb_type, fixed_padding=None, rel_window=args.rel_window)

        ################## Extract Features ##################
        with torch.no_grad():
            # Get features up to the classifier (without passing through the classifier)
            src_emb, f_map, _ = model.src_emb(x)
            src_mask = copy.copy(batch.src_mask)
            if src_emb.size()[1] % model.window_size != 0:
                n = src_emb.size()[1] // model.window_size
                m = ((n + 1) * model.window_size) - src_emb.size()[1]
                src_emb = torch.nn.functional.pad(src_emb, (0, 0, 0, m), mode='constant', value=0)
                src_mask = torch.nn.functional.pad(src_mask, (0, m), mode='constant', value=0)

            # Apply positional encoding
            src_emb = model.position(src_emb)

            # Pass through the encoder (this gives you the extracted features)
            features = model.encode(src_emb, None, src_mask)

            # Optional: Pool features across time dimension
            pooled_features = features.mean(dim=1)  # Shape: [batch_size, hidden_dim]

            # Now you have the extracted features, bypassing the classifier
            #print("Extracted Features Shape:", pooled_features.shape)
            feature_file[step] = pooled_features.cpu().numpy()

        ################  Classification ####################
        comb_out, class_logits, output_hand = model.forward(x, batch.src_mask, batch.rel_mask, hand_regions)

        # Calculate loss (cross-entropy loss for classification)
        loss_fn = nn.CrossEntropyLoss()
        y = y.squeeze(1)
        loss = loss_fn(class_logits, y)

        #Predicted words with highest prob
        _, pred = torch.max(class_logits, dim=1)
        # print(pred)
        accuracy = torch.sum(pred == y.data).item() / len(y)

        predictions.append(pred)
        references.append(y)
        total_loss += loss.item()
        total_accuracy += accuracy
        count += 1
  
    assert len(predictions) == len(references)

    avg_loss = total_loss / count
    avg_accuracy = total_accuracy / count

    print('Accuracy:'+str(avg_accuracy))

    np.save('{}/{}_feature.npy'.format(save_path, dataset[d]), feature_file)

    if(args.save):
        #Save results in txt files
        with open(os.path.join(save_path, 'prediction_'+dataset[d]+'.txt') ,'w') as trans_file:
            for prediction in predictions:
                trans_file.write(str(prediction) + '\n')

        with open(os.path.join(save_path, 'references_'+dataset[d]+'.txt'), 'w') as ref_file:
            for reference in references:
                ref_file.write(str(reference) + '\n')
