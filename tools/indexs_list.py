import numpy as np
import math
import random

def idxs(video_length,random_drop,uniform_drop):

    if isinstance(random_drop, bool) and isinstance(uniform_drop, bool):

        if random_drop:

            if video_length <= 80:
                random_drop = 0.5
                uniform_drop = None

            elif video_length > 80 and video_length <= 120:
                random_drop = 0.5 - (0.5-0.33) / 40 * (video_length - 80)
                uniform_drop = None

            elif video_length > 120 and video_length <= 250:
                random_drop = 0.33
                uniform_drop = None

            elif video_length > 250 and video_length <= 350:
                random_drop = 0.3 - (0.33-0.25) / 100 * (video_length - 250)
                uniform_drop = None

            elif video_length > 350:
                random_drop = 0.25
                uniform_drop = None


        else: 

            if video_length <= 80:
                random_drop = None
                uniform_drop = 0.5

            elif video_length > 80 and video_length <= 120:
                random_drop = None
                uniform_drop = 0.5 - (0.5-0.33) / 40 * (video_length - 80)

            elif video_length > 120 and video_length <= 250:
                random_drop = None
                uniform_drop = 0.33
            
            elif video_length > 250 and video_length <= 350:
                random_drop = None
                uniform_drop = 0.3 - (0.33-0.25) / 100 * (video_length - 250)

            elif video_length > 350:
                random_drop = None
                uniform_drop = 0.25


    if uniform_drop:
        num_frame = round(video_length * uniform_drop)

        if video_length//num_frame==1:
            num_delete_idxs=video_length-num_frame
            d=video_length//(num_delete_idxs)
            delete_idxs=[i for i in range(0,video_length,d)]
            all_idxs=range(0,video_length)
            selected_idxs = [i for i in all_idxs if i not in delete_idxs]
            num_loss_idxs=num_frame-len(selected_idxs)
            reserve_idxs=random.sample(delete_idxs, k=num_loss_idxs)
            selected_idxs.extend(reserve_idxs)
            selected_idxs.sort()

        if video_length//num_frame>1:

            d=video_length//num_frame
            selected_idxs=[i for i in range(0,video_length,d)]

            num_del_idxs=len(selected_idxs)-num_frame
            if num_del_idxs != 0:
                d_=len(selected_idxs)//num_del_idxs
                del_idxs=[i for i in range(0,video_length,d*(d_+1))]
            else:
                del_idxs=[]
            selected_idxs=list(set(selected_idxs)-set(del_idxs))

            num_del_idxs_=len(selected_idxs)-num_frame
            del_idxs_=random.sample(selected_idxs,k=num_del_idxs_)
            selected_idxs=list(set(selected_idxs)-set(del_idxs_))
            
            selected_idxs.sort()


    else:

        all_idxs = range(0 , video_length)
        selected_idxs = random.sample(all_idxs , k = round(random_drop * video_length))
        selected_idxs.sort()

    # print('idxs:',selected_idxs)
    # print('lenght of list',len(selected_idxs))
    # print('uniform drop', uniform_drop)
    # print('random drop', random_drop)

    return selected_idxs

# idxs(260, False , True)