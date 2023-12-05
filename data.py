import numpy as np
from torch.utils.data import Dataset

import TweetNormalizer
# from copy import deepcopy
# from datasets import load_dataset
from transformers.data.data_collator import DataCollatorWithPadding
# from copy import deepcopy
# import csv
# import scipy
from utils.utils import *
from utils.data_utils import *

from copy import deepcopy, copy
from multiprocessing import Pool
from tqdm import tqdm
from train_utils import get_tensor_long, get_tensor_float
from collections import defaultdict, OrderedDict
from dataclasses import dataclass

from typing import Any, List, Optional, Union

import torch
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers import PreTrainedTokenizerBase
import re
import nltk
from nltk.corpus import stopwords
import spacy
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
import preprocessor as p
from constants import TASK_SETTINGS, HUMAN_VALUES, MORAL_VALUES
from sklearn.utils.class_weight import compute_class_weight
from train_utils import set_seeds
import random
if module_exists("torch_geometric"):
    from torch_geometric.data import Batch, Data
    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T
    import torch_geometric.utils as pyg_utils
    from torch_geometric.nn import HeteroConv


def fill_array_at_positions(length, positions, null_val=0, val=1, ):
    label_vec = [null_val] * length
    for pos in positions:
        label_vec[pos] = val
    return label_vec


class PrimitiveDataset(Dataset):

    def __init__(self, args, filename, tokenizer=None, in_train=False):

        print("\nLoading Dataset...")

        "=============Loading Cache============="""
        # print("filename", filename)
        args.cache_filename = os.path.splitext(filename)[0] + ("primitive_d") + ".pkl"
        if args.use_cache and os.path.exists(args.cache_filename):
            print("Loading Cached Data...", args.cache_filename)
            self.instances = load_file(args.cache_filename)
            return

        "=============Loading============="""
        print("loading", filename)
        self.original_data = load_file(filename)

        self.instances = []
        # print("in_train and args.augment_with_translation", in_train and args.augment_with_translation)
        # print("(not in_train and do_foreign_eval)", (not in_train and args.do_foreign_eval))

        for idx, sample in enumerate(self.original_data):
            # if "non-moral" in sample["labels"]:
            #     continue
            text = sample["text"]
            if 'doc_pos' in sample:
                # post_text=self.original_data[sample["doc_pos"]]["text"]
                # text = f"'{post_text}'. {text}"
                text = text

            self.instances.append({"text": text,
                                   "sample_id": idx,
                                   "tweet_id": sample["tweet_id"] if "tweet_id" in sample else None,
                                   "input_ids": tokenizer.tokenize(text),
                                   })
        if args.cache_filename:
            dump_file(self.instances, args.cache_filename)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]


class PrimitivePredictionDataset(Dataset):

    def __init__(self, args, filename, tokenizer=None, labels=None, label_map=None, in_train=False, extra_data=None, cur_split=""):

        print("\nLoading Dataset...")

        self.SEP = tokenizer.sep_token_id
        self.CLS = tokenizer.cls_token_id
        self.BOS = tokenizer.bos_token_id
        self.EOS = tokenizer.eos_token_id
        self.SEP_TOKEN = tokenizer.sep_token
        self.CLS_TOKEN = tokenizer.cls_token
        self.BOS_TOKEN = tokenizer.bos_token
        self.EOS_TOKEN = tokenizer.eos_token

        if self.SEP is None:
            self.SEP = self.EOS
            self.SEP_TOKEN = self.EOS_TOKEN

        self.labels = labels
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label_map = label_map
        print("\nself.label2id", self.label2id)
        print("len self.label2id", len(self.label2id))

        if not args.database_io:
            "=============Loading Cache============="""
            # print("filename", filename)
            # args.cache_filename = os.path.splitext(filename)[0] + ".pkl"

            args.cache_filename = os.path.splitext(filename)[0] + "_" + args.plm_class + "_" + args.data_mode + \
                                  (f"_gprimitive") + \
                                  ("_prp" if args.personal_response_pred else "") + \
                                  ("_lb" if args.is_labeling else "") + \
                                  ("_sc" if args.sent_clf else "") + \
                                  ("_po" if args.pred_only else "") + \
                                  ("_to" if args.text_only else "") + \
                                  ("_int4sent" if args.use_intensity_for_sentiment else "") + \
                                  ("_skipemprof" if args.skip_empty_profile else "") + \
                                  (f"ua{args.user_attributes}") + \
                                  (f"pretrain{args.pretrain}") + \
                                  (f"tasksetting{args.task_setting}") + \
                                  (f"case_study{args.case_study}") + \
                                  (f"is{args.input_scheme}") + \
                                  (f"{args.config}") + \
                                  ".pkl"  #

            if args.use_cache and os.path.exists(args.cache_filename):
                print("Loading Cached Data...", args.cache_filename)
                self.instances = load_file(args.cache_filename)
                return

            "=============Loading============="""
            print("loading", filename)
            # if args.eval_only:
            #     self.original_data = []
            # else:
            self.original_data = load_file(filename)

            # if isinstance(self.original_data, dict):
            #     self.original_data
            if ".csv" in filename:
                json_str = self.original_data.to_json(orient="records")
                self.original_data = json.loads(json_str)

            tmp = []
            if args.personal_response_pred:

                print("personal_response_pred")
                tmp = process_samples(self.original_data, args, extra_data=extra_data, cur_split=cur_split)
                self.original_data = tmp
            elif args.sent_clf:
                for i, sample in enumerate(tqdm(self.original_data)):
                    tgt_text = sample["text"]
                    if args.is_labeling:
                        # tgt_text
                        tgt_text = preprocess_tweet_local(tgt_text)
                        if not tgt_text: continue
                        tmp.append({"text": tgt_text})
                        tmp[-1]['sample_id'] = i

            self.original_data = tmp
            print("total", len(self.original_data))

        else:
            raw_data = retrieve_raw_tweets_from_db()
            # self.original_data = [{'tweet_text':item["content_text"],"labels":["care"],'tweet_id':item['uiuc_message_id'] } for item in raw_data]
            self.original_data = [{'tweet_text': item.content_text, "labels": ["care"], 'tweet_id': item.uiuc_message_id} for item in raw_data]
            # breakpoint()

        self.instances = []
        # print("in_train and args.augment_with_translation", in_train and args.augment_with_translation)
        # print("(not in_train and do_foreign_eval)", (not in_train and args.do_foreign_eval))

        class_labels = []
        for idx, sample in enumerate(tqdm(self.original_data)):
            # if "non-moral" in sample["labels"]:
            #     continue
            text = sample["text"]
            # if 'doc_pos' in sample:
            #     post_text=self.original_data[sample["doc_pos"]]["text"]
            #     text = f"'{post_text}'. {text}"

            """can modify to include number"""

            # text = normalizeTweet(text)
            if not args.is_labeling:
                label_vec = fill_array_at_positions(length=len(self.labels),
                                                    positions=[self.label2id[label] for label in sample["label"]]) if isinstance(sample["label"], list) else self.label2id[
                    sample["label"]]  # label_map[label]
            else:
                label_vec = 0
                # fill_array_at_positions(length=len(self.labels), positions=[]) if isinstance(sample["label"], list) else
            # print("lbs", sample["labels"])
            # print("label_vec", label_vec)
            self.instances.append({"text": text,
                                   # "id": idx,
                                   "sample_id": idx,  # sample["sample_id"],
                                   "user_embed_idx": sample["user_embed_idx"],
                                   # "tweet_id": sample["tweet_id"] if "tweet_id" in sample else None,
                                   "extra": sample["extra"],
                                   "sample_id_in_orig_data": sample["sample_id"],
                                   "labels": label_vec,
                                   'orig_comment': sample["orig_comment"] if "orig_comment" in sample else "",
                                   "input_ids": tokenizer.tokenize(text),
                                   "in_train": in_train
                                   })
            # if in_train and args.augment_with_translation:
            #     text = sample["tweet_text_fr"]
            #     self.instances.append({"text": text,
            #                            "id": idx,
            #                            "tweet_id": sample["tweet_id"],
            #                            "labels": label_vec,
            #                            "input_ids": tokenizer.tokenize(text),
            #                            })
        # if in_train:
        #     class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=np.array(class_labels))
        #     self.class_weights = torch.tensor(class_weights, dtype=torch.float)

        # if args.use_gnn:
        #     # data = HeteroData()
        #     # data['user'].x = ...  # [num_papers, num_features_paper]
        #     # data['news'].x = ...  # [num_papers, num_features_paper]
        #     # data['user', 'follows', 'user'].edge_index = rand_edge_index  # [2, num_edges_cites]
        #     # data['user', 'replies', 'news'].edge_index = rand_edge_index  # [2, num_edges_cites]
        #     # data['user', 'replies', 'news'].y = torch.tensor([0, 1, 0, 1])
        #
        #     data = HeteroData(user={'x': ...},
        #                       news={'x': ...},
        #                       user__replies__news={'edge_index': rand_edge_index, 'edge_attr': ..., 'y': ..., 'train_mask': ..., 'val_mask': ..., 'test_mask': ...},
        #                       user__follows__user={'edge_index': rand_edge_index, 'edge_attr': ...},
        #                       )
        #     data = T.ToUndirected()(data)
        #     data = T.AddSelfLoops()(data)

        if args.cache_filename:
            dump_file(self.instances, args.cache_filename)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def get_class_weights(self):
        class_labels = [sample["labels"] for sample in self.instances]
        if np.unique(class_labels).shape[0] < len(self.labels):
            print("\n\n\nclass_weights dim smaller than label dim")
            class_labels = np.arange(len(self.labels))
        class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=np.array(class_labels))
        return torch.tensor(class_weights, dtype=torch.float)

    @classmethod
    def collect_labels(cls, files, path):
        if path_exists(path):
            return load_file(path)

        labels = set()
        for filename in files:
            data = load_file(filename)
            for idx, sample in enumerate(data):
                for m in sample["annotations"]:  # todo
                    labels |= m["labels"]
        labels = sorted(labels)
        dump_file(labels, path)

        return labels


def get_cache_filename(args, filename):
    cache_filename = os.path.splitext(filename)[0] + "_" + args.plm_class + "_" + args.data_mode + \
                     (f"_gprimitive") + \
                     ("_prp" if args.personal_response_pred else "") + \
                     ("_lb" if args.is_labeling else "") + \
                     ("_sc" if args.sent_clf else "") + \
                     ("_po" if args.pred_only else "") + \
                     ("_to" if args.text_only else "") + \
                     ("_int4sent" if args.use_intensity_for_sentiment else "") + \
                     ("_skipemprof" if args.skip_empty_profile else "") + \
                     (f"ua{args.user_attributes}") + \
                     (f"pretrain{args.pretrain}") + \
                      (f"tasksetting{args.task_setting}") + \
                      (f"case_study{args.case_study}") + \
                     (f"is{args.input_scheme}") + \
                     (f"{args.config}") + \
                     (f"_gnn") + \
                     ".pkl"  #

    return cache_filename


# def insert_id(uid, global_user_id_map):
#     uid = global_user_id_map[uid] if uid in global_user_id_map else global_user_id_map.setdefault(uid, global_user_id)
#     retu


# convert global_user_id_map[uid] if uid in global_user_id_map else global_user_id_map.setdefault(uid, global_user_id) to class


class PrimitivePredictionDatasetGNN(Dataset):

    def __init__(self, args, filename, tokenizer=None, labels=None, label_map=None, in_train=False, extra_data=None, tr=None, val=None, test=None):

        self.labels = labels
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label_map = label_map

        args.cache_filename = get_cache_filename(args, filename)

        user_data, post_data, history_data, graph_data, id_to_index = extra_data  # this id_to_index is origial from jinning
        valid_uids_with_influencers = set(list(id_to_index))
        valid_pids = set()
        for i, split in enumerate([tr, val, test]):
            for item in split:
                uid, pid = item["extra"]["user_id"], item["extra"]["post_id"]
                valid_pids.add(pid)
        valid_pids = sorted(valid_pids)

        self.instances = []
        train_mask = [1] * len(tr) + [0] * (len(val) + len(test))
        val_mask = [0] * len(tr) + [1] * len(val) + [0] * len(test)
        test_mask = [0] * (len(tr) + len(val)) + [1] * len(test)
        # train_mask=torch.tensor(train_mask, dtype=torch.bool)
        # val_mask=torch.tensor(val_mask, dtype=torch.bool)
        # test_mask=torch.tensor(test_mask, dtype=torch.bool)
        train_labels, val_labels, test_labels = [], [], []
        train_pairs, val_pairs, test_pairs = [], [], []
        user_news_pairs, user_user_pairs = [], []
        user_value_pairs = []

        """=================== get user embeddings ==================="""

        # print current seeds without setting them
        print("\nSEED STATE in DATA before ue")
        print("random seed", random.getstate()[1][0])
        print("np seed", np.random.get_state()[1][0])
        print("torch seed", torch.random.get_rng_state())
        print("torch cuda seed", torch.cuda.random.get_rng_state())

        args.ue_dir =ue_dir = f"{args.subset_dir}user_embeddings/{args.embedding_plm.split('/')[-1]}/"
        id_to_index_ue_fn, embedding_ue_fn = f"{ue_dir}id_to_index_{args.user_attributes}.json", f"{ue_dir}embedding_{args.user_attributes}.npy"
        # if not path_exists(id_to_index_ue_fn) or not path_exists(embedding_ue_fn):
        # if not path_exists(id_to_index_ue_fn) or not path_exists(embedding_ue_fn) or "gnn" in args.components:
        cnt1=0
        if True:

            # set_seeds(args)
            print("\nGenerating User Embedding")
            mkdir(ue_dir)
            id_to_index_ue = GlobalIdMap()
            user_texts = []
            for k, v in user_data.items():
                _, user_desc, history_text, _ = process_news_profile_history(history_data, user_data, post_data, k, None, num_past=80, user_l="4" in args.user_attributes, user_sc="5" in args.user_attributes)
                if "1" not in args.user_attributes: user_desc = " "
                if "2" not in args.user_attributes: history_text = ""
                if user_desc is not None:
                    id_to_index_ue.insert_id(k)
                    user_texts.append(f"Profile: {user_desc} </s> Historical Tweets: {history_text}")  # Profile:
                # if not user_desc: user_desc = " "
            dump_file(get_text_embeddings(user_texts, args), embedding_ue_fn)
            dump_file(id_to_index_ue.get_map(), id_to_index_ue_fn)
        # ue=np.load(f"{ue_dir}embedding.npy")
        id_to_index_ue = load_file(id_to_index_ue_fn)

        """=================== get post embeddings ==================="""
        args.pe_dir=pe_dir = f"{args.subset_dir}post_embeddings/{args.embedding_plm.split('/')[-1]}/"
        if not path_exists(f"{pe_dir}id_to_index.json"): # or "gnn" in args.components
        # if True:
            # set_seeds(args)
            print("\nGenerating Post Embedding")
            mkdir(pe_dir)
            id_to_index_pe = GlobalIdMap()
            post_texts = []
            for k in valid_pids:
                post_text, _, _, _ = process_news_profile_history(history_data, user_data, post_data, None, k)
                if post_text is not None:
                    id_to_index_pe.insert_id(k)
                    post_texts.append(post_text)
            dump_file(get_text_embeddings(post_texts, args), f"{pe_dir}embedding.npy")
            dump_file(id_to_index_pe.get_map(), f"{pe_dir}id_to_index.json")
        # pe=np.load(f"{pe_dir}embedding.npy")
        id_to_index_pe = load_file(f"{pe_dir}id_to_index.json")

        # print current seeds without setting them
        print("SEED STATE in DATA")
        print("random seed", random.getstate()[1][0])
        print("np seed", np.random.get_state()[1][0])
        print("torch seed", torch.random.get_rng_state())
        print("torch cuda seed", torch.cuda.random.get_rng_state())

        """=================== user-news edges ==================="""
        global_user_id_map, global_news_id_map = GlobalIdMap(), GlobalIdMap()
        # uids, pids = set(), set()
        labels = []
        # userx = []
        case_study_dict = load_file(f'{args.subset_dir}case_study_dict.json') if args.case_study==2 else {}

        for i, split in enumerate([tr, val, test]):
            for item in split:
                uid, pid = item["extra"]["user_id"], item["extra"]["post_id"]
                uid = global_user_id_map[uid]
                pid = global_news_id_map[pid]

                if args.no_user_news!=2 or i not in [1,2]:
                    user_news_pairs.append([uid, pid])

                if args.case_study == 2:
                    if (i == 1 and item["extra"]["user_id"] not in case_study_dict["unseen_dev"]) or (i == 2 and item["extra"]["user_id"] not in case_study_dict["unseen_test"]):
                        continue

                if i == 0:
                    cur_labels, cur_pairs = train_labels, train_pairs
                elif i == 1:
                    cur_labels, cur_pairs = val_labels, val_pairs
                else:
                    cur_labels, cur_pairs = test_labels, test_pairs

                labels.append(item["labels"])
                cur_pairs.append([uid, pid])
                cur_labels.append(item["labels"])
        user_news_pairs = torch.tensor(user_news_pairs).transpose(0, 1)
        self.train_labels, self.val_labels, self.test_labels = train_labels, val_labels, test_labels
        self.train_pairs, self.val_pairs, self.test_pairs = train_pairs, val_pairs, test_pairs


        """=================== user-user edges ==================="""
        cnt2=0
        for uid, followings in graph_data.items():
            if uid not in valid_uids_with_influencers: continue
            uid_cur = global_user_id_map[uid]
            for following in followings:
                following = str(following)
                if following not in valid_uids_with_influencers: continue
                following_cur = global_user_id_map[following]
                if uid in valid_uids_with_influencers and following in valid_uids_with_influencers:  # strorint
                    user_user_pairs.append([uid_cur, following_cur])
                    cnt2+=1
        # convert to [2, num_edges] torch tensor
        user_user_pairs = torch.tensor(user_user_pairs).transpose(0, 1)

        print("num users",global_user_id_map.get_size())
        print("num edge",cnt2)

        """=================== user-value edges ==================="""
        user_value_pairs = []
        global_value_id_map = GlobalIdMap()
        for v in HUMAN_VALUES + MORAL_VALUES:
            global_value_id_map.insert_id(v)
        for k, v in user_data.items():
            if k not in valid_uids_with_influencers or global_user_id_map.get_id(k) is None: continue
            is_none_value = False
            if "chatgptanno" in v:
                anno_dict = parse_gpt_anno(v["chatgptanno"].strip(), to_dict=True)
                for cat in ["human values", "moral values"]:
                    if cat in anno_dict:
                        for value in anno_dict[cat].split(","):
                            value = value.strip().lower()
                            if global_value_id_map.get_id(value) is not None:
                                # print(k,value)
                                user_value_pairs.append([global_user_id_map[k], global_value_id_map.get_id(value)])
        user_value_pairs = torch.tensor(user_value_pairs).transpose(0, 1)

        """=================== get features ==================="""
        # data = HeteroData(user={'x': ...},
        #                   news={'x': ...},
        #                   user__replies__news={'edge_index': rand_edge_index, 'edge_attr': ..., 'y': ..., 'train_mask': ..., 'val_mask': ..., 'test_mask': ...},
        #                   user__follows__user={'edge_index': rand_edge_index, 'edge_attr': ...},
        #                   )
        user_x = [id_to_index_ue.get(id, -1) + 1 for id in global_user_id_map.get_keys_sorted_by_values()]
        post_x = [id_to_index_pe.get(id, -1) + 1 for id in global_news_id_map.get_keys_sorted_by_values()]
        value_x = [id + 1 for id in range(global_value_id_map.get_size())]
        # user_user_pairs=torch.tensor([[0, 0],[0, 1],
        #         [0, 2],
        #         [1, 0],
        #         [1, 2],
        #         [2, 0],
        #         [2, 1],
        #         [3, 1]]).transpose(0,1)


        user_user_pairs, _ = pyg_utils.remove_self_loops(user_user_pairs)
        data = HeteroData(user={'x': get_tensor_long(user_x)},
                          news={'x': get_tensor_long(post_x)},
                          val={'x': get_tensor_long(value_x)},
                          user__replies__news={'edge_index': user_news_pairs}, #,'edge_attr': user_news_pairs
                          user__follows__user={'edge_index': user_user_pairs},  # , 'edge_attr': ... , 'edge_attr': ...
                          # 'pairs':user_news_pairs, 'y': labels, 'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask
                          user__contains__val={'edge_index': user_value_pairs},
                          )
        data = T.ToUndirected()(data)
        data = T.AddSelfLoops()(data)
        # pyg_utils.remove_self_loops()
        # data['user','replies','news']['edge_index'].transpose(0,1)
        # data['news','rev_replies','user']['edge_index'].transpose(0,1)
        # data['user','follows','user']['edge_index'].transpose(0,1)
        # user_user_pairs.transpose(0,1)
        # user_news_pairs.transpose(0,1)

        self.instances.append({"text": ".",
                               "sample_id": 0,  # sample["sample_id"],
                               "user_embed_idx": 0,
                               "extra": {},
                               "sample_id_in_orig_data": 0,
                               'data': data,
                               'orig_data': {"train": tr, "dev": val, "test": test},
                               "input_ids": tokenizer.tokenize("."),
                               "train_labels": train_labels,
                               "dev_labels": val_labels,
                               "test_labels": test_labels,
                               "train_pairs": train_pairs,
                               "dev_pairs": val_pairs,
                               "test_pairs": test_pairs,
                               # "labels": labels,
                               "in_train": in_train
                               })

        if args.cache_filename:
            dump_file(self.instances, args.cache_filename)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def get_class_weights(self):
        class_labels = self.instances[-1]["labels"]
        if np.unique(class_labels).shape[0] < len(self.labels):
            print("\n\n\nclass_weights dim smaller than label dim")
            class_labels = np.arange(len(self.labels))
        class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=np.array(class_labels))
        return torch.tensor(class_weights, dtype=torch.float)


def find_URLS(string):
    # findall() has been used
    # with valid conditions for urls in string
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    return [x[0] for x in url]


class PrimitiveGenerationDataset(Dataset):

    def __init__(self, args, src_file, tokenizer=None, in_train=False, extra_data=None, cur_split=""):

        super().__init__()
        """========Init========="""

        self.tokenizer = tokenizer
        self.instances = []

        # Special Tokens
        self.SEP = tokenizer.sep_token_id
        self.CLS = tokenizer.cls_token_id
        self.BOS = tokenizer.bos_token_id
        self.EOS = tokenizer.eos_token_id
        self.SEP_TOKEN = tokenizer.sep_token
        self.CLS_TOKEN = tokenizer.cls_token
        self.BOS_TOKEN = tokenizer.bos_token
        self.EOS_TOKEN = tokenizer.eos_token

        if self.SEP is None:
            self.SEP = self.EOS
            self.SEP_TOKEN = self.EOS_TOKEN

        self.max_seq_len = args.max_seq_len
        # print("self.max_seq_len", self.max_seq_len)

        if not src_file:
            return

        # is_t5 = "t5" in args.plm
        is_gpt = "gpt" in args.plm.lower()

        """========Load Cache========="""
        args.cache_filename = os.path.splitext(src_file)[0] + "_" + args.plm_class + "_" + args.data_mode + \
                              (f"_gprimitive") + \
                              ("_gene") + \
                              ("_prp" if args.personal_response_pred else "") + \
                              ("_lb" if args.is_labeling else "") + \
                              ("_sc" if args.sent_clf else "") + \
                              ("_po" if args.pred_only else "") + \
                              ("_to" if args.text_only else "") + \
                              ("_int4sent" if args.use_intensity_for_sentiment else "") + \
                              ("_skipemprof" if args.skip_empty_profile else "") + \
                              (f"ua{args.user_attributes}") + \
                              (f"pretrain{args.pretrain}") + \
                              (f"glc{args.generate_label_cat}") + \
                                  (f"tasksetting{args.task_setting}") + \
                                  (f"case_study{args.case_study}") + \
                              (f"is{args.input_scheme}") + \
                              (f"{args.config}") + \
                              ".pkl"  #
        save_file = args.cache_filename
        data_samples = None
        print('\nReading data from {}.'.format(src_file))
        if isinstance(src_file, str):
            if os.path.exists(save_file) and args.use_cache:
                self.instances = load_file(save_file)
                print('load processed data from {}.'.format(save_file))
                return
            data_samples = load_file(src_file)
            if ".csv" in src_file:
                json_str = data_samples.to_json(orient="records")
                data_samples = json.loads(json_str)

        elif isinstance(src_file, list):
            data_samples = src_file

        if args.debug:
            data_samples = data_samples[:100]
        # p.set_options(p.OPT.MENTION)

        if args.comment_generation:
            # restructuring
            tmp = []
            for i, sample in enumerate(tqdm(data_samples)):  # this indicates ith path after breaking out all articles into individual paths
                sum_likes = sum([x["like"] for x in sample["direct_replies"]])
                num_distinct = len(set([x["like"] for x in sample["direct_replies"]]))
                if num_distinct == 0: continue

                cumulated_likes = 0
                cumulated_likes_to_record = 0
                prev_recorded_num_likes = -1
                num_values_below = 0
                sorted_by_likes = sorted(sample["direct_replies"], key=lambda x: x["like"], reverse=False)
                for j, reply in enumerate(sample["direct_replies"] if args.min_num_likes == -1 else sorted_by_likes):
                    if reply["like"] > prev_recorded_num_likes:
                        prev_recorded_num_likes = reply["like"]
                        num_values_below = j
                        cumulated_likes_to_record = cumulated_likes + reply["like"]
                    cumulated_likes += reply["like"]
                    if args.min_num_likes == -1 or reply["like"] >= args.min_num_likes:
                        percentile = str(round(num_values_below / len(sample["direct_replies"]) * 100, 0)) if num_distinct > 0 else 0
                        percentile_str = ". [" + percentile + "th percentile]"
                        src_text = sample["text"] if args.min_num_likes == -1 else sample["text"] + percentile_str
                        # if reply["like"] >0:
                        #     breakpoint()
                        tmp.append({"src_text": src_text,
                                    "tgt_text": reply["text"],
                                    "tweet_id": sample["tweet_id"],
                                    "post_tweet_id": sample["tweet_id"],
                                    "reply_tweet_id": reply["tweet_id"],
                                    "num_likes": reply["like"]
                                    })
            data_samples = tmp
        elif args.label_generation:
            print("label generation")
            tmp = []
            for i, sample in enumerate(tqdm(data_samples)):  # this indicates ith path after breaking out all articles into individual paths
                src_text = sample["text"]
                sorted_keys = sort_key_by_value(sample["response_labels"][args.label_category], reverse=True)
                tmp.append({"src_text": src_text,
                            "tgt_text": " ".join(sorted_keys),
                            "tweet_id": sample["tweet_id"],
                            "post_tweet_id": sample["tweet_id"],
                            })
            data_samples = tmp
        elif args.personal_response_pred:
            # user_data, post_data, graph_data = extra_data
            print("personal_response_pred")
            tmp = process_samples(data_samples, args, extra_data=extra_data, cur_split=cur_split)
            data_samples = tmp

        print("restructured")
        maxlens = 0

        for i, sample in enumerate(tqdm(data_samples)):  # this indicates ith path after breaking out all articles into individual paths
            if "src_text" not in sample:
                src_text, tgt_text = sample["text"], None
            else:
                src_text, tgt_text = sample["src_text"], sample["tgt_text"]
            if len(tgt_text.split()) > 50: continue

            # for gpt
            tmp_max_seq_len = self.max_seq_len - 2
            tmp_max_seq_len_tgt = 90
            tmp_max_seq_len_src = tmp_max_seq_len - tmp_max_seq_len_tgt

            model_inputs = self.tokenizer(src_text, padding=True, max_length=tmp_max_seq_len_src, truncation=True)
            # self.max_seq_len if not is_gpt else tmp_max_seq_len_src

            if tgt_text is not None:
                with self.tokenizer.as_target_tokenizer():
                    # tgt_text = tgt_text.lower()
                    tgt = self.tokenizer(tgt_text, padding=True, max_length=tmp_max_seq_len_tgt, truncation=True)
                    # self.max_seq_len if not is_gpt else

                    # if tgt["input_ids"].count(self.tokenizer.encode('[label]')[0])!=2:
                    #     breakpoint()

                if is_gpt:
                    input_ids, attention_mask = model_inputs["input_ids"], model_inputs["attention_mask"]
                    if in_train:
                        model_inputs["labels"] = [-100] * len(input_ids) + tgt['input_ids']  # tgt['input_ids']
                        model_inputs["labels"] = model_inputs["labels"][:tmp_max_seq_len] + [tokenizer.eos_token_id]
                        model_inputs["input_ids"] += tgt['input_ids']  # tgt['input_ids']
                        model_inputs["attention_mask"] += tgt['attention_mask']  # tgt['input_ids']
                        # model_inputs["labels"] = model_inputs["labels"][:tmp_max_seq_len] + [tokenizer.eos_token_id]
                        model_inputs["input_ids"] = model_inputs["input_ids"][:tmp_max_seq_len] + [tokenizer.eos_token_id]
                        model_inputs["attention_mask"] = model_inputs["attention_mask"][:tmp_max_seq_len] + [1]
                    else:
                        model_inputs["labels"] = [-100] * len(input_ids)
                        model_inputs["labels"] = model_inputs["labels"][:tmp_max_seq_len_src]
                        model_inputs["input_ids"] = model_inputs["input_ids"][:tmp_max_seq_len_src]
                        model_inputs["attention_mask"] = model_inputs["attention_mask"][:tmp_max_seq_len_src]
                        # tgt_text=self.tokenizer.decode(tgt['input_ids']).replace("  "," ")

                else:
                    model_inputs["labels"] = tgt['input_ids']
                    # for key in ['input_ids', 'attention_mask']:
                    #     model_inputs["decoder_" + key] = tgt[key]
                    # model_inputs["decoder_" + "token_type_ids"] = [0] * len(tgt["input_ids"])

            # # don't do over long inputs
            # exceed_max_len = False
            # maxlens = max(len(tgt['input_ids']), maxlens)
            # if max(len(tgt['input_ids']), len(model_inputs['input_ids'])) >= self.max_seq_len - 2:
            #     exceed_max_len = True

            self.instances.append({
                'tokenizer': tokenizer,
                'src_text': src_text,
                'tgt_text': tgt_text,
                "sample_id": i,
                "extra": sample["extra"],
                "sample_id_in_orig_data": sample["sample_id"],
                'exceed_max_len': -1,  # exceed_max_len,
                "in_train": in_train
            })


            self.instances[-1].update(model_inputs)

        """encode different parts into functions"""
        # save data
        print("maxlens", maxlens)
        if args.cache_filename:
            dump_file(self.instances, save_file)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        return instance


###
def process_tgt_text(tgt_text):
    if find_URLS(tgt_text):
        return None
    tgt_text = p.clean(tgt_text)
    return tgt_text.strip()


def process_samples(original_data, args, extra_data=None, cur_split=""):
    tmp = []
    user_data, post_data, history_data, graph_data, id_to_index = extra_data
    if args.pretrain:
        for i, sample in enumerate(tqdm(original_data)):

            user_id = sample["author_id"]
            # post_id = sample["conversation_id"]

            """user_desc"""
            user_desc = user_data[str(user_id)]['description']

            # # filter out inactive users
            # post_text = post_data[str(post_id)]['text']
            # post_text = preprocess_tweet_local(post_text)

            tgt_text = process_tgt_text(sample["text"])
            if not tgt_text:
                continue

            src_text = f" . [POST] {user_id} [UID] {user_desc} [PROFILE] "  # {tokenizer.sep_token}[LABEL_SEP] {uid_str} [UID]

            tmp.append({
                "src_text": src_text,
                "tgt_text": tgt_text,
                "extra": {
                    "user_id": user_id,
                }
            })
            tmp[-1]['sample_id'] = i


    elif args.personal_response_pred:
        # user_data, post_data, history_data, graph_data = extra_data

        print("personal_response_pred")

        cnt_empty_user_attributes = 0
        num_empty_desc = 0

        custom_posts = [
            "I am a student",
        ]

        graph_index_map = id_to_index  # defaultdict(int)

        # for i, (k,v) in enumerate(graph_data):
        #     graph_index_map[k]=i
        p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY)
        case_study_dict = load_file(f'{args.subset_dir}case_study_dict.json') if args.case_study==2 else {}
        for i, sample in enumerate(tqdm(original_data)):  # this indicates ith path after breaking out all articles into individual paths
            user_id = sample["author_id"]
            post_id = sample["conversation_id"]

            if not args.use_gnn and args.case_study == 2:
                if (cur_split=="dev" and user_id not in case_study_dict["unseen_dev"]) or (cur_split=="test" and user_id not in case_study_dict["unseen_test"]):
                    continue

            """user_desc"""
            if str(user_id) not in user_data:
                user_desc = ""
                # continue
            else:
                user_desc = user_data[str(user_id)]['description']

            if not user_desc.strip():
                num_empty_desc += 1
                if args.skip_empty_profile: continue

            # pretrain
            # in reply to account id, author id, use for recording conversation
            # geo
            # timestamp

            "only use nonreply"

            # history
            # geo TODO
            # entities/context
            # cat top 20, max is max seq len - user desc and news
            # topic as retrieval

            """history"""
            if str(user_id) not in history_data:
                history_text = ""
                # continue
            else:
                history_posts = [item["text"] for item in history_data[str(user_id)][:50]]  # if int(item["tweet_id"]) != int(sample["tweet_id"])
                history_text = ";".join(history_posts)

            """check"""
            if not user_desc and not history_text:
                cnt_empty_user_attributes += 1
                continue

            if "predicted" not in sample or "predicted_intensity" not in sample:
                continue

            # filter out inactive users
            post_text = post_data[str(post_id)]['text']
            post_text = preprocess_tweet_local(post_text)

            tgt_text = sample["text"]
            # # skip the ones with url
            # if p.clean(tgt_text)!=tgt_text:
            #     continue
            if find_URLS(tgt_text):
                continue
            tgt_text = p.clean(tgt_text)
            if not tgt_text.strip():
                continue

            uid_str = str(user_id)
            if "1" not in args.user_attributes:
                user_desc = " "
            if "2" not in args.user_attributes:
                history_text = ""
            if "3" not in args.user_attributes:
                uid_str = " "

            src_text = None
            if args.input_scheme == "3":
                src_text = f"{post_text} [POST] {user_id} [UID] {user_desc} [PROFILE] {history_text}"  # {tokenizer.sep_token}[LABEL_SEP] {uid_str} [UID]
            elif args.input_scheme == "2":
                src_text = f"{post_text} [POST] {user_id} [UID] {user_desc} [PROFILE] "  # {tokenizer.sep_token}[LABEL_SEP] {uid_str} [UID]
            elif args.input_scheme == "1":
                src_text = f"{post_text} [POST] {user_desc} [PROFILE] {history_text}"  # {tokenizer.sep_token}[LABEL_SEP] {uid_str} [UID]

            if args.use_special_tag == 5:
                # remove all twitter handles
                src_text = src_text.replace("@user", "<unk>").replace("<phone>", "1234567890").replace("<url>", "example.com").replace("<email>", "username@domain.com")
            elif args.use_special_tag == 6:
                # remove all twitter handles
                src_text = src_text.replace("@user", "<unk>").replace("<phone>", "<unk>").replace("<url>", "<unk>").replace("<email>", "<unk>")

            predicted_intensity = sample["predicted_intensity"]
            if args.pure_int:
                predicted_intensity=abs(int(predicted_intensity)-3)

            if args.pred_label_category=="predicted":
                cur_label= sample["predicted"]
            elif args.pred_label_category=="predicted_intensity":
                cur_label= predicted_intensity
            else:
                raise NotImplementedError

            if args.task_mode == "clf":
                if args.is_labeling:
                    tmp.append({"text": tgt_text, })
                else:
                    tmp.append({"text": src_text, "label": cur_label,
                                "user_embed_idx": graph_index_map[str(user_id)] if bool(graph_index_map) and str(user_id) in graph_index_map else -1,

                                "extra": {
                                    "orig_comment": tgt_text,
                                    "predicted_intensity": predicted_intensity,
                                    "category": user_data[str(user_id)]["category"] if str(user_id) in user_data and "category" in user_data[str(user_id)] else " ",
                                    "user_desc": user_desc,
                                    "history_text": history_text,
                                    "post_text": post_text,
                                    "user_id": user_id,
                                    "post_id": post_id
                                }
                                })
                    if args.use_intensity_for_sentiment:
                        if sample["predicted"] == 3:
                            tmp[-1]['label'] = 1
                        else:
                            tmp[-1]['label'] = 0 if int(sample["predicted"]) < 3 else 2
                tmp[-1]['orig_comment'] = tgt_text

            elif args.task_mode == "gen":
                if args.pred_only:
                    tmp_tgt_text = f" {sample['predicted']} [label] {sample['predicted_intensity']} [label]  "
                    if args.generate_label_cat==1:
                        tmp_tgt_text = f" 0 [label] {sample['predicted_intensity']} [label]  "
                    if args.generate_label_cat==2:
                        tmp_tgt_text = f" {sample['predicted']} [label] 0 [label]  "



                elif args.text_only:
                    tmp_tgt_text = "EMPTY" if not tgt_text.strip() else tgt_text
                else:
                    tmp_tgt_text = f" {sample['predicted']} [label] {sample['predicted_intensity']} [label] {tgt_text}"

                tmp.append({"src_text": src_text,
                            "tgt_text": tmp_tgt_text,
                            # "extra": (sample["predicted"], sample["predicted_intensity"], user_id, post_id)
                            "extra": {
                                "predicted": sample["predicted"],
                                "predicted_intensity": sample["predicted_intensity"],
                                "category": user_data[str(user_id)]["category"] if str(user_id) in user_data and "category" in user_data[str(user_id)] else " ",
                                "user_id": user_id,
                                "post_id": post_id
                            }
                            })
            tmp[-1]['sample_id'] = i
        # print("num_empty_desc", num_empty_desc)
        # print("cnt_empty_user_attributes", cnt_empty_user_attributes / len(original_data))
    return tmp
