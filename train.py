import argparse
import time
import os
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import datetime as dt
from torch.optim.lr_scheduler import StepLR, MultiStepLR
#Progress bar to visualize training progress
import progressbar
import matplotlib.pyplot as plt

#Plotting
from tools.viz import learning_curve_slr

#Visualize GPU resources
import GPUtil

from transformer import make_model as TRANSFORMER
from dataloader import loader
from tools.utils import Batch, NoamOpt
###
# Arg parsing
##############

parser = argparse.ArgumentParser(description='Training the transformer-like network')

parser.add_argument('--data', type=str, default=os.path.join('data','phoenix-2014.v3','phoenix2014-release','phoenix-2014-multisigner'),
                   help='location of the data corpus')

parser.add_argument('--remove_bg_training',type=bool, default= False)

parser.add_argument('--remove_bg_test',type=bool, default= False)


parser.add_argument('--fixed_padding', type=int, default=None,
                    help='None/64')

parser.add_argument('--num_classes', type=int)

parser.add_argument('--classifier_hidden_size', type=int, default=256)

parser.add_argument('--recognition', type=int, default='emotion')

parser.add_argument('--lookup_table', type=str, default=os.path.join('data','slr_lookup.txt'),
                    help='location of the words lookup table')

parser.add_argument('--rescale', type=int, default=224,
                    help='rescale data images.')

parser.add_argument('--random_drop_probability', type=float, default=False,
                    help='probability of frame random drop/0-1 or None')

parser.add_argument('--uniform_drop_probability', type=float, default=True,
                    help='probability of frame random drop/0-1 or None')

#Put to 0 to avoid memory segementation fault
parser.add_argument('--num_workers', type=int, default=10,
                    help='NOTE: put num of workers to 0 to avoid memory saturation.')

parser.add_argument('--show_sample', action='store_true',
                    help='Show a sample a preprocessed data (sequence of image of sign + translation).')

parser.add_argument('--optimizer', type=str, default='ADAM',
                    help='optimization algo to use; SGD, SGD_LR_SCHEDULE, ADAM / NOAM')

parser.add_argument('--scheduler', type=str, default=None,
                    help='Type of scheduler, multi-step or stepLR')

parser.add_argument('--milestones', default="15,30", type=str,
                    help="milestones for MultiStepLR or stepLR")

parser.add_argument('--weight_decay', type= float , default = 5e-5)

parser.add_argument('--batch_size', type=int, default=16,
                    help='size of one minibatch')

parser.add_argument('--initial_lr', type=float, default=0.0001,
                    help='initial learning rate')

parser.add_argument('--hidden_size', type=int, default=1280,
                    help='size of hidden layers. NOTE: This must be a multiple of n_heads.')

parser.add_argument('--num_layers', type=int, default=2,
                    help='number of transformer blocks')

parser.add_argument('--n_heads', type=int, default=8,
                    help='number of self attention heads')

#Pretrained weights
parser.add_argument('--pretrained', type=bool, default=True,
                    help='embedding layers are pretrained using imagenet')

parser.add_argument('--full_pretrained', type=str, default=None,
                    help='Full frame CNN pretrained')

parser.add_argument('--hand_pretrained', type=str, default=None,
                    help='Hand regions CNN pretrained')

parser.add_argument('--hand_query', action='store_true',
                    help='Set hand as a query for transformer network.')

parser.add_argument('--emb_type', type=str, default='2d',
                    help='Type of image embeddings 2d or 3d.')

parser.add_argument('--emb_network', type=str, default='mb2',
                    help='Image embeddings network: mb2/i3d/m3d')

parser.add_argument('--image_type', type=str, default='rgb',
                    help='Train on rgb/grayscale images')

parser.add_argument('--num_epochs', type=int, default=100,
                    help='number of epochs to stop after')

parser.add_argument('--dp_keep_prob', type=float, default=0.8,
                    help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                    (dp_keep_prob=1 means no dropout)')

parser.add_argument('--valid_steps', type=int, default=2, help='Do validation each valid_step')

parser.add_argument('--save_steps', type=int, default=10, help='Save model after each N epoch')

parser.add_argument('--debug', action='store_true')

parser.add_argument('--save_dir', type=str, default='EXPERIMENTATIONS',
                    help='path to save the experimental config, logs, model')

parser.add_argument('--evaluate', action='store_true',
                    help="Evaluate dev set using bleu metric each epoch.")

parser.add_argument('--d_ff', type=int,default=2048)

parser.add_argument('--resume', default=False,
                    help="Resume training from a checkpoint.")
                    
parser.add_argument('--checkpoint',type=str, default=None,
                    help="resume training from a previous checkpoint.")

parser.add_argument('--rel_window', type=int, default=None,
                    help="Use local masking window.")

#Training settings
parser.add_argument('--parallel', action='store_true',
                    help='Training on multiple GPUs if available by splitting batches!')

parser.add_argument('--distributed', action='store_true',
                    help='Training on multiple GPUs if available by splitting submodules!')

parser.add_argument('--freeze_cnn', default= False,
                    help='freeze the feature extractor (CNN)!')

parser.add_argument('--data_stats', type=str, default=None,
                    help="Normalize images using the dataset stats (mean/std).")

parser.add_argument('--hand_stats', type=str, default=None,
                    help="Normalize images using the dataset stats (mean/std).")


#----------------------------------------------------------------------------------------


## SET EXPERIMENTATION AND SAVE CONFIGURATION

#Same seed for reproducibility)
parser.add_argument('--seed', type=int, default=1111, help='random seed')

#Save folder with the date
start_date = dt.datetime.now().strftime("%Y-%m-%d-%H.%M")
print ("Start Time: "+start_date)

args = parser.parse_args()

#Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

#experiment_path = PureWindowsPath('EXPERIMENTATIONS\\' + start_date)
experiment_path = os.path.join(args.save_dir,start_date)

# Creates an experimental directory and dumps all the args to a text file
if(os.path.exists(experiment_path)):
    print('Experiment already exists..')
    quit(0)
else:
    os.makedirs(experiment_path)

print ("\nPutting log in EXPERIMENTATIONS/%s"%start_date)

args.save_dir = os.path.join(args.save_dir, start_date)

#Dump all configurations/hyperparameters in txt
with open (os.path.join(experiment_path,'exp_config.txt'), 'w') as f:
    f.write('Experimentation done at: '+ str(start_date)+' with current configurations:\n')
    for arg in vars(args):
        f.write(arg+' : '+str(getattr(args, arg))+'\n')

#-------------------------------------------------------------------------------
#Run on GPU
if torch.cuda.is_available():
    print('Nmber of GPUs={}',torch.cuda.device_count())
    print('Device name:{}',torch.cuda.get_device_name(0))
    device = torch.device("cuda:0")
else:
#Run on CPU
    print("WARNING: Training on CPU, this will likely run out of memory, Go buy yourself a GPU!")
    device = torch.device("cpu")
#--------------------------------------------------------------------------------


#Computation for one epoch
def run_epoch(model, data, is_train=False, device='cuda:0', n_devices=1):

    if is_train:
        model.train()  # Set model to training mode
        print ("Training..")
        phase='train'
    else:
        model.eval()   # Set model to evaluate mode
        print ("Evaluating..")
        phase='valid'

    start_time = time.time()

    loss = 0.0
    total_loss = 0.0
    total_accuracy = 0.0
    count = 0

    #For progress bar
    bar = progressbar.ProgressBar(maxval=dataset_sizes[phase], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    j = 0
    #Loop over minibatches
    for step, (x, x_lengths, y, hand_regions, hand_lengths) in enumerate(data):

        #Update progress bar with every iter
        j += len(x)
        bar.update(j)

        y = y.to(device)
        x = x.to(device)

        #NOTE: clone y to avoid overridding it
        batch = Batch(x_lengths, hand_lengths, trg=None, emb_type=args.emb_type, DEVICE=device, fixed_padding=args.fixed_padding, rel_window=args.rel_window)

        if(args.distributed):

            #Zeroing gradients
            feature_extractor.zero_grad()
            encoder.zero_grad()
            position.zero_grad()
            output_layer.zero_grad()

            src_emb, _, _ = feature_extractor(x)
            src_emb = position(src_emb)
            src_emb = encoder(src_emb, None, batch.src_mask)
            output_context = output_layer(src_emb)

            if(args.hand_query):
                hand_extractor.zero_grad()

                hand_emb = hand_extractor(hand_regions)
                hand_emb = position(hand_emb)
                hand_emb = encoder(hand_emb, None, batch.src_mask)
                output_hand = output_layer(hand_emb)

                comb_emb = encoder(src_emb, hand_emb, batch.rel_mask)
                output = output_layer(comb_emb)

            else:
                output = None
                output_hand = None

        else:
            #Zeroing gradients
            model.zero_grad()

            #Shape(batch_size, tgt_seq_length, tgt_vocab_size)
            #NOTE: no need for trg if we dont have a decoder
            comb_out, class_logits, output_hand = model.forward(x, batch.src_mask, batch.rel_mask, hand_regions)

        # Calculate loss (cross-entropy loss for classification)
        loss_fn = nn.CrossEntropyLoss()
        y = y.squeeze(1)
        loss = loss_fn(class_logits, y)

        if is_train:
            # Backward pass and optimization step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # Optional: clip gradients
            optimizer.step()

        # Calculate accuracy
        _, preds = torch.max(class_logits, 1)
        accuracy = torch.sum(preds == y.data).item() / len(y)

        total_loss += loss.item()
        total_accuracy += accuracy
        count += 1
        # x_lengths = torch.IntTensor(x_lengths)
        # y_lengths = torch.IntTensor(y_lengths)


     # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / count
    avg_accuracy = total_accuracy / count

    if is_train:
        print(f"Average Training Loss: {avg_loss:.4f}, Average Training Accuracy: {avg_accuracy:.4f}")
    else:
        print(f"Average Validation Loss: {avg_loss:.4f}, Average Validation Accuracy: {avg_accuracy:.4f}")

    return avg_loss, avg_accuracy
#-------------------------------------------------------------------------------------------------------

### LOAD DATALOADERS

# In debug mode, try batch size of 1
if args.debug:
    batch_size = 1
else:
    batch_size = args.batch_size


#Train on rgb/grayscale images
if(args.image_type == 'rgb'):
    channels = 3
#Not supported yet
elif(args.image_type == 'grayscale'):
    channels = 1
else:
    print('Image type is ot supported!')
    quit(0)

loss_fn = nn.CrossEntropyLoss()

#train_path, valid_path, test_path = path_data(data_path=args.data)
train_csv = pd.read_csv(os.path.join(args.data, 'tools/data','train.csv'))
test_csv = pd.read_csv(os.path.join(args.data, 'tools/data','test.csv'))
val_csv = pd.read_csv(os.path.join(args.data, 'tools/data','dev.csv'))

with open(args.lookup_table, 'r') as file:
    lookup_table = json.load(file)

#Load stats
if(args.data_stats):
    args.data_stats = torch.load(args.data_stats, map_location=torch.device('cpu'))

if(args.hand_stats):
    args.hand_stats = torch.load(args.hand_stats, map_location=torch.device('cpu'))

#Pass the annotation + image sequences locations
train_dataloader, train_size = loader(csv_file=train_csv,
                root_dir=args.data,
                lookup_table=lookup_table,
                recognition=args.recognition,
                remove_bg=args.remove_bg_training,
                rescale = args.rescale,
                batch_size = batch_size,
                num_workers = args.num_workers,
                random_drop= args.random_drop_probability,
                uniform_drop= args.uniform_drop_probability,
                show_sample = args.show_sample,
                istrain=True,
                fixed_padding=args.fixed_padding,
                hand_dir=None,
                data_stats=args.data_stats,
                hand_stats=args.hand_stats,
                channels=channels
                )

#No data augmentation for valid data
valid_dataloader, valid_size = loader(csv_file=val_csv,
                root_dir=args.data,
                lookup_table=lookup_table,
                recognition=args.recognition,
                remove_bg=args.remove_bg_test,
                rescale = args.rescale,
                batch_size = args.batch_size,
                num_workers = args.num_workers,
                random_drop= args.random_drop_probability,
                uniform_drop= args.uniform_drop_probability,
                show_sample = args.show_sample,
                istrain=False,
                fixed_padding=args.fixed_padding,
                hand_dir=None,
                data_stats=args.data_stats,
                hand_stats=args.hand_stats,
                channels=channels
                )

print('Dataset sizes:')
dataset_sizes = {}
dataset_sizes.update({'train':train_size})
dataset_sizes.update({'valid':valid_size})
print(dataset_sizes)

#-----------------------------------------------------------------------------------------------------------------

#Load the whole model
model = TRANSFORMER(num_classes=args.num_classes, n_stacks=args.num_layers, n_units=args.hidden_size, n_heads=args.n_heads ,d_ff=args.d_ff, dropout=1.-args.dp_keep_prob, image_size=args.rescale, pretrained=args.pretrained,
                            classifier_hidden_dim= args.classifier_hidden_size,emb_type=args.emb_type, emb_network=args.emb_network, full_pretrained=args.full_pretrained, hand_pretrained=args.hand_pretrained, freeze_cnn=args.freeze_cnn, channels=channels)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('model parameters:',trainable_params)

if torch.cuda.device_count() > 1 and args.parallel:
    #How many GPUs you are using
    n_devices = torch.cuda.device_count()


    if(args.distributed):
        #Split GPUs for both feature extraction and sequence learning (Transformer)
        n_devices_split = int(n_devices/2)
        print("Using ", n_devices_split, "GPUs for feature extraction and ", n_devices-n_devices_split, "GPUs for sequence learning.")

        devices = list(range(0, n_devices_split))
        feature_extractor = nn.DataParallel(model.src_emb, device_ids=devices).to(device)

        if(args.hand_query):
             hand_extractor = nn.DataParallel(model.hand_emb, device_ids=devices).to(device)

        devices = list(range(n_devices_split, n_devices))

        encoder = nn.DataParallel(model.encoder, device_ids=devices).to(n_devices_split)
        position = nn.DataParallel(model.position, device_ids=devices).to(n_devices_split)
        output_layer = nn.DataParallel(model.output_layer, device_ids=devices).to(n_devices_split)

    else:
        print("Using ", n_devices, "GPUs!, Let's GO!")
        model = nn.DataParallel(model).to(device)
else:
    print("Training using 1 device (GPU/CPU), use very small batch_size!")
    #Load model into device (GPU OR CPU)
    n_devices = 1
    model = model.to(device)

    if(args.distributed):
        print("Can't use distributed training since you have a single GPU!")
        quit(0)


#print("Loading to GPUs")
#print(GPUtil.showUtilization())

train_ppls = []
train_losses = []
train_accuracies = []
val_ppls = []
val_losses = []
val_accuracies = []
times = []

if args.optimizer == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr , weight_decay = args.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)

elif args.optimizer == 'noam':
    optimizer = NoamOpt(args.hidden_size, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# In debug mode, only run one epoch
if args.debug:
    num_epochs = 1
else:
    num_epochs = args.num_epochs

#Load weights from previous training session
#Resume training or start from start w/ pretrained weights
if(args.checkpoint):
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    if(args.resume):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss_fn = checkpoint['loss']
        best_accuracy = checkpoint['best_accuracy']
        #scheduler =  checkpoint['scheduler']
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

if(args.checkpoint == None or args.resume == False):
    start_epoch = 0

    if args.scheduler == 'multi-step':
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    elif args.scheduler == 'stepLR':
        scheduler = StepLR(optimizer, step_size=args.milestones, gamma=0.1)
    else:
        print('No scheduler!')


###
#Main Training loop
best_accuracy_so_far = 0

for epoch in range(start_epoch, num_epochs):

    start = time.time()

    print('\nEPOCH '+str(epoch)+' ------------------')
    #print('LR',scheduler.get_lr())
    print(optimizer.param_groups[0]['lr'])
    # RUN MODEL ON TRAINING DATA
    train_loss, train_accuracy = run_epoch(model, train_dataloader, True, device=device)
    print("After train epoch..")
    print(GPUtil.showUtilization())

    #Save perplexity
    train_ppl = np.exp(train_loss)

    if(args.scheduler):
        scheduler.step()
    
    if(epoch % args.valid_steps == 0):

        #RUN MODEL ON VALIDATION DATA
        #NOTE: Helps with avoiding memory saturation
        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(model, valid_dataloader)

            if val_accuracy > best_accuracy_so_far:
                best_accuracy_so_far = val_accuracy

                #if args.save_best:
                print("Saving entire model with best params")
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'best_accuracy': best_accuracy_so_far
                },
                os.path.join(args.save_dir, 'BEST.pt'))

                print("Saving full-frame (CNN) with best params")
                torch.save(model.src_emb.state_dict(), os.path.join(args.save_dir, 'full_cnn_best_params.pt'))

                if(args.hand_query):
                    print("Saving hand regions (CNN) with best params")
                    torch.save(model.hand_emb.state_dict(), os.path.join(args.save_dir, 'hand_cnn_best_params.pt'))

        val_ppl = np.exp(val_loss)
        
        # SAVE RESULTS
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        times.append(time.time() - start)

        log_str = 'epoch: ' + str(epoch) + '\t' \
             + 'train ppl: ' + str(train_ppl) + '\t' \
             + 'val ppl: ' + str(val_ppl) + '\t' \
             + 'train loss: ' + str(train_loss) + '\t' \
             + 'val loss: ' + str(val_loss) + '\t' \
             + 'accuracy: ' + str(val_accuracy) + '\t' \
            + 'BEST accuracy: ' + str(best_accuracy_so_far) + '\t' \
            + 'time (s) spent in epoch: ' + str(times[-1])

        print(log_str)

        with open (os.path.join(args.save_dir, 'log.txt'), 'a') as f_:
                f_.write(log_str+ '\n')


        #SAVE LEARNING CURVES
        lc_path = os.path.join(args.save_dir, 'learning_curves.npy')
        print('\nDONE\n\nSaving learning curves to '+lc_path)
        np.save(lc_path, {'train_ppls':train_ppls,
                  'val_ppls':val_ppls,
                  'train_losses':train_losses,
                   'val_losses':val_losses,
                   'accuracy':val_accuracies,
                  })

        print("Saving plots")
        learning_curve_slr(args.save_dir)

        #Save every model every 10 epoch
        if(epoch % args.save_steps == 0):
            #Save after each epoch and save optimizer state
            print("Saving model parameters for epoch: "+str(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn,
                'best_accuracy': best_accuracy_so_far
                },
                os.path.join(args.save_dir, 'epoch_'+str(epoch)+'_accuracy_'+str(val_accuracy)+'.pt'))


        #We reached convergence
        if(train_ppl <= 1):
            print("Hello World ;)")
            break
