import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_utils import truncate_feats
from .datasets import register_dataset


@register_dataset("tsl_300")
class TSL300Dataset(Dataset):
    def __init__(
            self,
            is_training,  # if in training mode
            split,  # split, a tuple/list allowing concat of subsets
            feat_folder,  # folder for features
            json_file,  # json file for annotations
            feat_stride,  # temporal stride of the feats
            num_frames,  # number of frames for each feat
            default_fps,  # default fps
            downsample_rate,  # downsample rate for feats
            max_seq_len,  # maximum sequence length during training
            trunc_thresh,  # threshold for truncate an action segment
            crop_ratio,  # a tuple (e.g., (0.9, 1.0)) for random cropping
            input_dim,  # input feat dim
            num_classes,  # number of action categories
            file_prefix,  # feature file prefix if any
            file_ext,  # feature file extension if any
            force_upsampling  # force to upsample to max_seq_len
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'tsl_300',
            'tiou_thresholds': np.linspace(0.1, 0.3, 5),
            # we will mask out cliff diving
            'empty_label_ids': [],
        }

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        json_file_gt_file = f'{json_file}/videosenti_gt.json'
        # load database and select the subset
        with open(json_file_gt_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_set = set()
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_set.add(act['label'])
            label_list = sorted(list(label_set))
            label_dict = {label_list[i]: i for i in range(len(label_list))}
        else:
            label_dict = self.label_dict

        # fps json
        json_file_fps_dict = f'{json_file}/fps_dict.json'
        with open(json_file_fps_dict, 'r') as fid:
            fps_data = json.load(fid)

        # duration json
        json_file_duration_dict = f'{json_file}/len_duration_dict.json'
        with open(json_file_duration_dict, 'r') as fid:
            duration_data = json.load(fid)

        # check train/test split
        train_split_path = f'{json_file}/split_train.txt'
        test_split_path = f'{json_file}/split_test.txt'
        with open(train_split_path, mode='r') as f:
            train_txt = f.read().strip()
            train_set = set(train_txt.split('\n'))
        with open(test_split_path, mode='r') as f:
            test_txt = f.read().strip()
            test_set = set(test_txt.split('\n'))
        split_set = {'train': train_set, 'test': test_set}

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            local_split = value['subset'].lower()
            if local_split not in self.split:
                continue
            if key not in split_set[local_split]:
                continue

            # or does not have the feature file
            rgb_feat_file = os.path.join(self.feat_folder, local_split, 'rgb',
                                         self.file_prefix + key + self.file_ext)
            mfcc_feat_file = os.path.join(self.feat_folder, local_split, 'logmfcc',
                                          self.file_prefix + key + self.file_ext)
            if not os.path.exists(rgb_feat_file) or not os.path.exists(mfcc_feat_file):
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif key in fps_data:
                fps = fps_data[key]
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if key in duration_data:
                duration = duration_data[key]
            else:
                raise Exception

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = None
            if segments is None:
                # TODO 少了一份数据【71_CMU_MOSEI__gzYkdjNvPc】
                continue
            dict_db += ({'id': key,
                         'fps': fps,
                         'duration': duration,
                         'segments': segments,
                         'labels': labels,
                         'split': value['subset'].lower()
                         },)

        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        rgb_feat_file = os.path.join(self.feat_folder, video_item['split'], 'rgb',
                                     self.file_prefix + video_item['id'] + self.file_ext)
        mfcc_feat_file = os.path.join(self.feat_folder, video_item['split'], 'logmfcc',
                                      self.file_prefix + video_item['id'] + self.file_ext)

        rgb_feature = np.load(rgb_feat_file).astype(np.float32)
        mfcc_feature = np.load(mfcc_feat_file).astype(np.float32)

        T = min(rgb_feature.shape[0], int(mfcc_feature.shape[0] / 32))
        rgb_feature = rgb_feature[:T]
        mfcc_feature = mfcc_feature[:T * 32].reshape(T, 32 * 60)
        # transform
        mfcc_feature = (mfcc_feature + 50) / 80

        # print(video_item['id'])
        # print(video_item['fps'])
        # print(video_item['duration'])
        # print(rgb_feature.shape)
        # print(mfcc_feature.shape)
        # # 8_CMU_MOSEI_2Vtv2gPzM7w
        # # 30.0
        # # 120.4
        # # (225, 1024)
        # # (225, 1920)

        # TODO 1 暂时只用rgb的试一试！
        # feats = rgb_feature

        # TODO 2 暂时只用audio的试一试！
        # feats = mfcc_feature

        # TODO 3 用rgb+audio试一试
        feats = np.concatenate([rgb_feature, mfcc_feature], axis=-1)

        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id': video_item['id'],
                     'feats': feats,  # C x T
                     'segments': segments,  # N x 2
                     'labels': labels,  # N
                     'fps': video_item['fps'],
                     'duration': video_item['duration'],
                     'feat_stride': feat_stride,
                     'feat_num_frames': self.num_frames}

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        return data_dict
