import torch
import os
import numpy as np
import cv2
import face_alignment
from matplotlib import pyplot as plt
import json
import time
import librosa
import pickle

path_to_mp4 = %%PATH_TO_MP4%%
path_to_preprocess = %%PATH_TO_PREPROCESSED_DATA%

K = 8
max_vid_per_folder = 5
device = torch.device('cuda:0')
face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda:0')

#####
# path_to_mp4 = '/mnt/VoxCeleb2'
# path_to_preprocess = 'preprocessed_VoxCeleb'

# K = 8
# max_vid_per_folder = 5
# device = torch.device('cuda:0')
# face_detector_kwargs = {'path_to_detector': '.cache/torch/checkpoints/s3fd-619a316812.pth'}
# print('Loading face aligner')
# face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda:0')
# print('Loading done')
#####

def generate_landmarks(frames_list, face_aligner, to_img=True):
    frame_landmark_list = []
    landmarks_coord_list = []
    fa = face_aligner
    
    if not to_img:
        for frame in frames_list:
            preds = fa.get_landmarks(frame)[0]
            landmarks_coord_list.append(preds.tolist())
        for i in range(len(frames_list) - len(landmarks_coord_list)):
            #filling frame_landmark_list in case of error
            landmarks_coord_list.append(landmarks_coord_list[i])
        return frame_landmark_list, landmarks_coord_list


    for i in range(len(frames_list)):
        try:
            input = frames_list[i]
            preds = fa.get_landmarks(input)[0]

            dpi = 100
            fig = plt.figure(figsize=(input.shape[1]/dpi, input.shape[0]/dpi), dpi = dpi)
            ax = fig.add_subplot(1,1,1)
            ax.imshow(np.ones(input.shape))
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            #chin
            ax.plot(preds[0:17,0],preds[0:17,1],marker='',markersize=5,linestyle='-',color='green',lw=2)
            #left and right eyebrow
            ax.plot(preds[17:22,0],preds[17:22,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
            ax.plot(preds[22:27,0],preds[22:27,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
            #nose
            ax.plot(preds[27:31,0],preds[27:31,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
            ax.plot(preds[31:36,0],preds[31:36,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
            #left and right eye
            ax.plot(preds[36:42,0],preds[36:42,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
            ax.plot(preds[42:48,0],preds[42:48,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
            #outer and inner lip
            ax.plot(preds[48:60,0],preds[48:60,1],marker='',markersize=5,linestyle='-',color='purple',lw=2)
            ax.plot(preds[60:68,0],preds[60:68,1],marker='',markersize=5,linestyle='-',color='pink',lw=2) 
            ax.axis('off')

            fig.canvas.draw()

            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            frame_landmark_list.append((input, data))
            landmarks_coord_list.append(preds.tolist())
            plt.close(fig)
        except:
            print('Error: Video corrupted or no landmarks visible')
    
    for i in range(len(frames_list) - len(frame_landmark_list)):
        #filling frame_landmark_list in case of error
        frame_landmark_list.append(frame_landmark_list[i])
        landmarks_coord_list.append(landmarks_coord_list[i])
    
    return frame_landmark_list, landmarks_coord_list


def pick_images(video_path, num_images, all_img=False):

    cap = cv2.VideoCapture(video_path)
    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if all_img:
        idxes = [1] * n_frames
    else:
        idxes = [1 if i%(n_frames//num_images+1)==0 else 0 for i in range(n_frames)]
    
    frames_list = []
    
    # Read until video is completed or no frames needed
    ret = True
    frame_idx = 0
    frame_counter = 0
    while(ret and frame_idx < n_frames):
        ret, frame = cap.read()
        
        if ret and idxes[frame_idx] == 1:
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(RGB)
            
        frame_idx += 1

    cap.release()
    
    return frames_list

def get_melspec(file_path, win_length, hop_length, eps=1e-10):
    librosa_data, freq = librosa.load(file_path, sr=None)
    melspec = np.log(librosa.feature.melspectrogram(librosa_data, n_fft=win_length, hop_length=hop_length) + eps)
    return melspec

def preprocess(all_img=False):

    t = time.time()

    video_counter = 0

    for pers_count, person_id in enumerate(os.listdir(path_to_mp4)):

        if pers_count >= 10:
            break

        print(f'Processing person {person_id}')

        person_out_path = os.path.join(path_to_preprocess, person_id)
        if not os.path.isdir(person_out_path):
            print(f'Need new folder for {person_id}, creating')
            os.mkdir(person_out_path)

        for video_id in os.listdir(os.path.join(path_to_mp4, person_id)):
            
            folder_vids = 0

            for video in os.listdir(os.path.join(path_to_mp4, person_id, video_id)):

                video_index = video_id + '_' + video.split('.')[0]
                mark_coord_path = os.path.join(person_out_path, video_index + '_coord.json')
                if os.path.isfile(mark_coord_path):
                    print(f'Video {video_index} already processed, moving to next video...')
                    continue

                elif folder_vids >= max_vid_per_folder:
                    continue
                    
                try:
                    video_path = os.path.join(path_to_mp4, person_id, video_id, video)
                    frame_mark = pick_images(video_path, K, all_img=all_img)
                    frame_mark, mark_coord = generate_landmarks(frame_mark, face_aligner, to_img=not all_img)
                    if all_img:
                        # Then we are interested in landmarks and audio
                        with open(mark_coord_path, 'w') as f:
                            json.dump(mark_coord, f)
                        folder_vids += 1

                        melspec = get_melspec(video_path, win_length=1024, hop_length=256)
                        with open(os.path.join(person_out_path, video_index + 'melspec'), 'wb') as f:
                            pickle.dump(melspec, f)
                        video_counter += 1

                        if video_counter % 10 == 0:
                            print(f'vid # {video_counter} processed in {time.time() - t}')

                    elif len(frame_mark) == K:
                        final_list = [frame_mark[i][0] for i in range(K)]
                        for i in range(K):
                            final_list.append(frame_mark[i][1]) #K*2,224,224,3
                        final_list = np.array(final_list)
                        final_list = np.transpose(final_list, [1,0,2,3])
                        final_list = np.reshape(final_list, (224, 224*2*K, 3))
                        final_list = cv2.cvtColor(final_list, cv2.COLOR_BGR2RGB)
                            
                        cv2.imwrite(os.path.join(person_out_path, video_index + ".png"), final_list)
                        print(f'Video {video_index} successfully processed !')

                        with open(mark_coord_path, 'w') as f:
                            json.dump(mark_coord, f)

                        folder_vids += 1
                        
                except:
                    print('ERROR: ', video_path)

        print(f'Person {person_id} processedin {time.time() - t}')
            
        
    print('done')

# if __name__ == "__main__":
#     preprocess(all_img=True)