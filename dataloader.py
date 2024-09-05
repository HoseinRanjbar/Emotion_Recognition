#############################################
#                                           #
# Load sequential data from PHOENIX-2014    #
#                                           #
#############################################

from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tools.indexs_list import idxs


#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def collate_fn(data, fixed_padding=None, pad_index=1232):
    """Creates mini-batch tensors w/ same length sequences by performing padding to the sequecenses.
    We should build a custom collate_fn to merge sequences w/ padding (not supported in default).
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding), else pad
    all Sequences to a fixed length.

    Returns:
        hand_seqs: torch tensor of shape (batch_size, padded_length).
        hand_lengths: list of length (batch_size); 
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); 
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); 
    """

    def pad(sequences, t):
        lengths = [len(seq) for seq in sequences]

        #For sequence of images
        if(t=='source'):
            #Retrieve shape of single sequence
            #(seq_length, channels, n_h, n_w)
            seq_shape = sequences[0].shape
            if(fixed_padding):
                padded_seqs = fixed_padding
                padded_seqs = torch.zeros(len(sequences), fixed_padding, seq_shape[1], seq_shape[2], seq_shape[3]).type_as(sequences[0])
            else:
                padded_seqs = torch.zeros(len(sequences), max(lengths), seq_shape[1], seq_shape[2], seq_shape[3]).type_as(sequences[0])

        #For sequence of words
        elif(t=='target'):
            # Just convert the list of target words to a tensor directly
            padded_seqs = torch.tensor(sequences, dtype=torch.long)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]

        return padded_seqs, lengths

    src_seqs = []
    trg_seqs = []
    right_hands = []
    left_hands = []

    for element in data:
        src_seqs.append(element['images'])
        trg_seqs.append(element['translation'])

        right_hands.append(element['right_hands'])

    #pad sequences
    src_seqs, src_lengths = pad(src_seqs, 'source')
    # Convert target sequences to tensor (no padding needed)
    trg_seqs = torch.tensor(trg_seqs, dtype=torch.long).view(-1, 1)

    #pad hand sequences
    if(type(right_hands[0]) != type(None)):
        hand_seqs, hand_lengths = pad(right_hands, 'source')
    else:
        hand_seqs = None
        hand_lengths = None

    return src_seqs, src_lengths, trg_seqs, hand_seqs, hand_lengths


#From abstract function Dataset
class MELDDataset(Dataset):
    """Sequential Sign language images dataset."""

    def __init__(self, csv_file, root_dir, lookup_table, remove_bg, random_drop, uniform_drop, istrain, transform=None,rescale=224, sos_index=1, eos_index=2, unk_index=0, fixed_padding=None, hand_dir=None, hand_transform=None, channels=3):

        #Get data
        #self.annotations = pd.read_csv(csv_file)
        self.annotations = csv_file
        self.root_dir = root_dir
        self.lookup_table = lookup_table
        self.remove_bg= remove_bg
        self.hand_dir = hand_dir
        self.random_drop = random_drop
        self.uniform_drop = uniform_drop
        self.transform = transform
        self.hand_transform = hand_transform
        self.istrain = istrain
        self.rescale = rescale

        self.channels = channels

        #index used for eos token and unk
        self.eos_index = eos_index
        self.unk_index = unk_index
        self.sos_index = sos_index


    def __len__(self):
        #Return size of dataset
        return len(self.annotations)

    def __getitem__(self, idx):
        #global trsf_images
        #Retrieve the name id of sequence from csv annotations
        name = self.annotations.iloc[idx]['file_path']

        video_path = os.path.join(self.root_dir, name)
        # Create a VideoCapture object
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       
        if self.istrain:
            indexs = idxs(frame_count,random_drop=self.random_drop,uniform_drop=self.uniform_drop)
            seq_length = len(indexs)
        else:
            if self.random_drop:
                indexs = idxs(frame_count,random_drop=None,uniform_drop= self.random_drop)
            else:
                indexs = idxs(frame_count,random_drop=None,uniform_drop= self.uniform_drop)
            seq_length = len(indexs)

        trsf_images = torch.zeros((seq_length, self.channels, self.rescale, self.rescale))

        #Get hand cropped image list if exists
        if(self.hand_dir):
            hand_path = os.path.join(self.hand_dir, name)
            hand_images = torch.zeros((seq_length, self.channels, 112, 112))
        else:
            hand_images = None

        #Save the images of seq
        i=0
        j=0
        # Loop through the video frames
        while True:

            # Capture frame-by-frame
            ret, frame = cap.read()        
            #image=cv2.imread(img_name)
                # If no frame is returned, break the loop (end of the video)
            if not ret:
                break

            if i in indexs:

                # Convert the frame from BGR (OpenCV format) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image= cv2.resize(frame,(224,224))
                #NOTE: some images got shape of (260, 220, 4)
                if(image.shape[2] == self.channels):
                    trsf_images[j] = self.transform(image)
                else:
                    trsf_images[j] = self.transform(image[:, :, :self.channels])
                j+=1

            i+=1

        cap.release()
        #Retrive the translation (ground truth text translation) from csv annotations
        sign = self.annotations.iloc[idx]['class']

        # Convert the ground truth label to numeric using the lookup table
        label = self.lookup_table.get(sign, -1)  # Default to -1 if the class is not in the lookup table
        label = torch.tensor([label], dtype=torch.long).squeeze()

        #NOTE: full frame seq and hand seq should be with the same seq length
        #sample = {'images': trsf_images, 'right_hands':hand_images, 'translation': trans}
        return {'images': trsf_images, 'right_hands':hand_images, 'translation': label}
        #return sample


# Helper function to show a batch
def show_batch(sample_batched):
    """Show sequence of images with translation for a batch of samples."""

    images_batch, images_length, trans_batch, trans_length = \
            sample_batched
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    #Show only one sequence of the batch
    grid = utils.make_grid(images_batch[0, :images_length[0]])
    grid = grid.numpy()
    return np.transpose(grid, (1,2,0))


#Use this to subtract mean from each pixel measured from PHOENIX-T dataset
#Note: means has been subtracted from 227x227 images, this has been provided by camgoz
class SubtractMeans(object):
    def __init__(self, path, rescale):
        #NOTE: Newest np versions default value allow_pickle=False
        self.mean = np.load(path, allow_pickle=True)
        self.mean = self.mean.astype('uint8')
        self.rescale = rescale

    def __call__(self, image):

        #No need to resize (take long time..)
        #image = cv2.resize(image,(self.mean.shape[0], self.mean.shape[1]))
        assert image.shape == self.mean.shape
        image -= self.mean
        #image = cv2.resize(image,(self.rescale, self.rescale))

        return image


def loader(csv_file, root_dir, lookup_table, recognition, remove_bg, rescale, batch_size, num_workers, random_drop, uniform_drop, show_sample, istrain=False, mean_path='FulFrame_Mean_Image_227x227.npy', fixed_padding=None, hand_dir=None, data_stats=None, hand_stats=None, channels=3):

    #Note: when using random cropping, this with reshape images with randomCrop size instead of rescale
    if(istrain):

        if(data_stats):
            trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.Resize((rescale, rescale)),
                transforms.ToTensor()
                #transforms.Normalize(mean=data_stats['mean'], std=data_stats['std'])
                ])

        
        else:
            trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.Resize((rescale, rescale)),
                transforms.ToTensor()
                #Imagenet std and mean
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            

        if(hand_stats):
            hand_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomAffine(10),
                    transforms.Resize((112, 112)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=hand_stats['mean'], std=hand_stats['std'])
                    ])
        else:
            hand_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomAffine(10),
                    transforms.Resize((112, 112)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                

    else:

        if(data_stats):
            trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((rescale, rescale)),
                transforms.ToTensor()
                #transforms.Normalize(mean=data_stats['mean'], std=data_stats['std'])
                ])
            

        else:
             trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((rescale, rescale)),
                transforms.ToTensor()
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])


        if(hand_stats):
            hand_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=hand_stats['mean'], std=hand_stats['std'])
                    ])
        else:
            hand_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

    ##Iterate through the dataset and apply data transformation on the fly

    #Apply data augmentation to avoid overfitting
    transformed_dataset = MELDDataset(csv_file=csv_file,
                                            root_dir=root_dir,
                                            lookup_table=lookup_table,
                                            recognition = recognition,
                                            remove_bg=remove_bg,
                                            random_drop=random_drop,
                                            uniform_drop=uniform_drop,
                                            transform=trans,
                                            rescale=rescale,
                                            istrain=istrain,
                                            hand_dir=hand_dir,
                                            hand_transform=hand_trans,
                                            channels = channels
                                            )

    size = len(transformed_dataset)

    #Iterate in batches
    #Note: put num of workers to 0 to avoid memory saturation
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    #Show a sample of the dataset
    if(show_sample and istrain):
        for i_batch, sample_batched in enumerate(dataloader):
            #plt.figure()
            img = show_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.imshow(img)
            #plt.show()
            plt.savefig('data_sample.png')
            break

    return dataloader, size
