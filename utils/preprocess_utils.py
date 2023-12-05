import os
import random
from multiprocessing import Pool
# import multiprocessing as mp
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import nltk
import json
import string
from utils.utils import load_file, dump_file, get_directory, path_exists, module_exists

from utils.utils import *
# print("imported utils")
from IPython import embed
from utils.graph import load_resources
from collections import OrderedDict, Counter
from pprint import pprint as pp
import re
from copy import deepcopy, copy
from random import shuffle

# from flair.data import Sentence
# from flair.models import SequenceTagger
if module_exists("transition_amr_parser"):
    from transition_amr_parser.stack_transformer_amr_parser import AMRParser
from process_amr import amr_parse, processing_amr
from TweetNormalizer import *
from utils.utils import split_to_tr_val_test

import preprocessor as tweet_preprocessor
from model_preprocess import generate_label

nlp = None
matcher = None
cur_concept2id = None
cur_id2concept = None
record_dict = {}
pos_tagger = None


# noinspection PyTypeChecker
def integrate_db_back_to_original_file(db_file, original_data_file, output_path):
    db = load_file(db_file)
    orig_data = load_file(original_data_file)

    for i, sample in enumerate(db):
        post = orig_data[sample["doc_id"]]
        if "reply_idx" in sample:
            post["direct_replies"][sample['reply_idx']]["predicted_labels"] = sample["predicted_labels"]
            post["direct_replies"][sample['reply_idx']]["text"] = sample["text"]
        else:
            post["predicted_labels"] = sample["predicted_labels"]
            post["text"] = sample["text"]

    res = []
    valid_samples_for_each_category = {"emotion": [], "sentiment": [], "moral": []}
    for i, sample in enumerate(orig_data):

        if "top_response_labels" not in sample:
            sample["top_response_labels"] = {}
        if "response_labels" not in sample:
            sample["response_labels"] = {}
        cur_category = "emotion"  # "sentiment"

        counts = Counter([reply["predicted_labels"][cur_category][1] for reply in sample["direct_replies"]])
        # print("num replies", len(sample["direct_replies"]))
        # print("count inconfident", counts.get('inconfident', 0))
        num_valid_preds = len(sample["direct_replies"]) - counts.get('inconfident', 0)
        print("num_valid_preds", num_valid_preds)
        # if num_valid_preds == 0:  # need enough support
        #     continue

        # for reply in sample["direct_replies"]:
        #     reply["predicted_labels"] = sample["predicted_labels"]
        #
        # sample["predicted_labels"] = sample["predicted_labels"]
        counts.pop('inconfident', None)
        sample["response_labels"][cur_category] = counts

        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        print("sorted_counts", sorted_counts)
        tmp = 0
        top_labels = []
        for key, value in sorted_counts:
            tmp += value
            top_labels.append(key)
            if tmp / num_valid_preds >= 0.8:
                break
        sample["top_response_labels"][cur_category] = top_labels

        if num_valid_preds > 30:  # need enough support
            print(sample["text"])
            print(sample["top_response_labels"][cur_category])
            valid_samples_for_each_category[cur_category].append(sample)

    print("num valid samples", len(valid_samples_for_each_category["emotion"]))
    # breakpoint()
    # for sample in orig_data:
    #     if "top_response_labels" not in sample:
    res = {}
    for cur_category, samples in valid_samples_for_each_category.items():
        print("cur_category", cur_category)
        if len(samples) < 10: continue
        tr, dev, te = split_to_tr_val_test_data(samples, ratio="622")
        res[cur_category] = {
            "train": tr,
            "dev": dev,
            "test": te
        }
        directory, path_name, filename, ext = get_path_info(output_path)
        output_dir = os.path.join(directory, "")
        mkdir(output_dir)
        dump_file_batch([tr, dev, te], [f"{output_dir}/{filename}_{cur_category}_{split}{ext}" for split in ["train", "dev", "test"]])
    dump_file(res, output_path)
    # split_to_tr_val_test(output_path, ratio="811")


def create_individual_tweet_db(input_path, output_path):
    f = load_file(input_path)
    res = []

    tweet_preprocessor.set_options(tweet_preprocessor.OPT.URL, tweet_preprocessor.OPT.MENTION)
    # p.set_options(p.OPT.URL, p.OPT.EMOJI)
    for i, doc in enumerate(tqdm(f)):
        # global_doc_id = doc["global_doc_id"]
        # sections = doc["sections"]
        # title = doc["title"]

        # res.append({
        #     'doc_id': i,
        #     'tweet_id': doc["tweet_id"],
        #     "is_original_post": 1,
        #     "text": doc["text"],
        #     "conversation_id": doc["conversation_id"],
        #     'path_id': -1,
        #     'step_id': -1,
        #     'step': title,
        #     'doc_type': doc_type,
        #     'step_type': "goal",
        #     'is_': categories,
        #     # 'url':url,
        # })
        doc["doc_id"] = i
        # print("before", doc["text"])
        doc["text"] = tweet_preprocessor.clean(doc["text"])
        # print("after", doc["text"])

        tmp = doc
        # tmp=copy(doc)
        # tmp.pop("direct_replies")
        doc_pos = len(res)
        res.append(tmp)

        for j, reply in enumerate(doc["direct_replies"]):
            reply["doc_id"] = i
            reply["doc_pos"] = doc_pos
            reply["reply_idx"] = j
            reply["text"] = tweet_preprocessor.clean(reply["text"])
            res.append(reply)
    dump_file(res, output_path)


def separate_article_into_paths(input_path, output_path):
    f = load_file(input_path)
    res = []
    full_seqs = []
    for i, doc in enumerate(tqdm(f)):
        global_doc_id = doc["global_doc_id"]
        sections = doc["sections"]
        title = doc["title"]
        doc_type = doc["type"]
        categories = doc["categories"]
        # url = doc["url"]
        # print("doc", i)
        # print(title)
        # print(sections)

        # paths = [[title] + section for section in sections] if doc_type == "methods" else [[title] + [
        #     step for section in sections for step in section]]
        paths = [section for section in sections] if doc_type == "methods" else [[
            step for section in sections for step in section]]
        if doc_type == "parts":
            import numpy as np
            subgoal_indices = np.cumsum([0] + [len(sec) for sec in sections[:-1]])
            # print("sections", sections)
            # print("subgoal_indices", subgoal_indices)

        unneeded_head_tail_chars = '!;,.? '
        # title=title.strip(unneeded_head_tail_chars)
        if not title: continue

        res.append({
            'doc_id': i,
            'global_doc_id': global_doc_id,
            'path_id': -1,
            'step_id': -1,
            'step': title,
            'doc_type': doc_type,
            'step_type': "goal",
            'categories': categories,
            # 'url':url,
        })
        # print(res[-1])

        for j, path in enumerate(paths):
            # print("path", j)

            if doc_type == "parts":
                assert len(paths) == 1
                new_path = []
                new_subgoal_indices = set()
                for k, item in enumerate(path):
                    if len(item.strip(unneeded_head_tail_chars)) or k in subgoal_indices:
                        if k in subgoal_indices:
                            new_subgoal_indices.add(len(new_path))
                        if not len(item.strip(unneeded_head_tail_chars)):
                            new_path.append(",")
                        else:
                            new_path.append(item.strip(unneeded_head_tail_chars))

                paths[j] = new_path
                # print("new_subgoal_indices",new_subgoal_indices)
                # print("new_path",new_path)
            else:
                paths[j] = [item.strip(unneeded_head_tail_chars) for k, item in enumerate(path)
                            if len(item.strip(unneeded_head_tail_chars)) or k == 0]
            # and
            #             item.strip(unneeded_head_tail_chars).lower() not in ["steps"]
            if len(paths[j]) >= 1:

                # align so that every last step is not term about finish
                if paths[j][-1].lower() in ["done", "finish", "finished", "finishing",
                                            "complete", "completed",
                                            "completing"]:
                    paths[j].pop()
            # if len(paths[j]) > 1:
            #     if paths[j][-1].lower() not in ["done", "finish", "finished", "finishing",
            #                                      "complete", "completed",
            #                                      "completing"]:
            #         paths[j].append("finished")
            #     else:
            #         paths[j][-1] = "finished"

        for j, path in enumerate(paths):
            # print("path", j)
            if len(path) < 1:
                print("empty path", j)
                continue
            # cur_steps = [path[0]]
            # assert len(cur_steps[-1].strip()), print("cur_steps", cur_steps)
            cur_seq = []
            cur_step_types = []

            for k, step in enumerate(path):
                res.append({
                    'doc_id': i,
                    'global_doc_id': global_doc_id,
                    'path_id': j,
                    'step_id': k,
                    'step': step,
                    'doc_type': doc_type,
                    'step_type': "event" if ((doc_type == "methods" and k > 0) or (doc_type == "parts" and k not in new_subgoal_indices)) else "subgoal",
                    # 'categories':categories,
                    # 'url':url,
                })
                cur_step_types.append(res[-1]['step_type'])
                # print(res[-1])
            full_seqs.append({
                'doc_id': i,
                'global_doc_id': global_doc_id,
                'path_id': j,
                'steps': step,
                'doc_type': doc_type,
                'step_type': "event" if ((doc_type == "methods" and k > 0) or (doc_type == "parts" and k not in new_subgoal_indices)) else "subgoal",
            })
            # cur_steps.append(step)

    # print(res)
    dump_file(res, output_path)
    # print("saved")


def enrich_step(data_path, output_path, num_processes=1, batch_size=256, roberta_batch_size=45, use_cache=True):
    global nlp, matcher, record_dict, pos_tagger
    # print("\nground_sent")
    if nlp is None or matcher is None:  # or pos_tagger is None
        # print("PATTERN_PATH", PATTERN_PATH)
        # print("CPNET_VOCAB", type(CPNET_VOCAB))

        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        # matcher = load_matcher(nlp, PATTERN_PATH)
        # pos_tagger = SequenceTagger.load('pos')
        record_dict = {}

    samples = load_file(data_path)
    sents = [item["step"] for item in samples]

    """====parse sents to tokens lists===="""
    path_name, ext = os.path.splitext(output_path)
    cur_path = f"{path_name}_tokens.json"
    cur_path_features = f"{path_name}_tokens_features.json"
    if path_exists(cur_path) and path_exists(cur_path_features) and use_cache:
        tokens_lists = load_file(cur_path)
        token_feature_list = load_file(cur_path_features)
    else:
        # tokens_lists_cls=[Sentence(sent) for sent in sents]
        # tokens_lists=[tok.text for sent in tokens_lists_cls for tok in sent.tokens ]
        tokens_lists = []
        token_feature_list = []  # tag, lemma
        # tokens_features=[]
        for i, tokenized_s in enumerate(nlp.pipe(sents, batch_size=10000)):
            # tokens_features.append({
            #     "tokens": [tok.text for tok in tokenized_s.tokens],
            #     "tag_list": [tok.tag_ for tok in tokenized_s.tokens],
            # })
            token_text = [tok.text for tok in tokenized_s]
            for j, tk in enumerate(token_text):
                if "[" in tk or "]" in tk:
                    token_text[j] = re.sub(r"\[+", '[', token_text[j])
                    token_text[j] = re.sub(r"\]+", ']', token_text[j])
                token_text[j] = token_text[j].replace("\"", "")
            tokens_lists.append(token_text)
            token_feature_list.append([(tok.lemma_, tok.tag_) for tok in tokenized_s])
        dump_file(tokens_lists, cur_path)
        dump_file(token_feature_list, cur_path_features)

    # cp_tokens_lists=deepcopy(tokens_lists)
    for i, tk in enumerate(deepcopy(tokens_lists)):
        tokens_lists[i] = tk[:50]
        token_feature_list[i] = token_feature_list[i][:50]
        samples[i]["tokens"] = tokens_lists[i]
        samples[i]["token_features"] = token_feature_list[i]

    # """get pos tags"""
    # if num_processes <= 1:
    #     res = [extract_vn_sent((step)) for step in sents]
    # else:
    #     with Pool(num_processes) as p:  # ctx.
    #         res = list(tqdm(p.imap(extract_vn_sent, sents), total=len(sents)))
    #
    # for i, sample in enumerate(samples):
    #     sample['pos'] = {
    #         "tokens": res[i][0],
    #         "coref_dict": res[i][1],
    #         # "vb_idxs":res[i][2],
    #     }

    """====get amr graphs===="""
    parser = AMRParser.from_checkpoint('../../transition-amr-parser/amr_general/checkpoint_best.pt')
    # parser = AMRParser.from_checkpoint('/home/chenkai5/transition-amr-parser/amr_general/checkpoint_best.pt')
    # parser = AMRParser.from_checkpoint('/root/transition-amr-parser/amr_general/checkpoint_best.pt')

    bsz = 500
    start_pos = 0
    amr_graphs, align, exist, amr_strs = [], [], [], []
    path_name, _ = os.path.splitext(output_path)
    cur_path = f"{path_name}_amr.json"
    if path_exists(cur_path) and use_cache:
        amr_graphs, align, exist, amr_strs = load_file(cur_path)
        start_pos = len(amr_graphs)

    print("start_pos for amr parsing", start_pos)
    tokens_lists_cp = deepcopy(tokens_lists)
    # embed()

    # ml=0
    # for j, item in enumerate(tokens_lists_cp[start_pos:min(start_pos + bsz, len(tokens_lists_cp))]):
    #     print(j, item)
    #     amr_parse([item], batch_size=batch_size, roberta_batch_size=roberta_batch_size, parser=parser)
    #     # ml=max(ml,len(j))
    # print("max len",ml)
    for j in range(start_pos, len(tokens_lists_cp), bsz):
        tokens_lists_tmp = tokens_lists_cp[j:min(j + bsz, len(tokens_lists_cp))]
        amr_str_tmp = amr_parse(tokens_lists_tmp, batch_size=batch_size, roberta_batch_size=roberta_batch_size, parser=parser)
        amr_graphs_tmp, align_tmp, exist_tmp, _ = processing_amr(amr_str_tmp, tokens_lists_tmp)

        amr_strs.extend(amr_str_tmp)
        amr_graphs.extend(amr_graphs_tmp)
        align.extend(align_tmp)
        exist.extend(exist_tmp)

        dump_file([amr_graphs, align, exist, amr_strs], cur_path)
        print(f"Finished parsing batch {j} amr")

    print("Finished parsing all amr")
    for i, sample in enumerate(samples):
        sample['amr'] = {
            "graph": amr_graphs[i],
            "align": align[i],
            "exist": exist[i],
            "str": amr_strs[i],
        }
    dump_file(samples, output_path)


def create_samples_from_steps(data_path, output_path, num_processes=1, num_steps_to_begin=2):
    """====tree for record paths for scripts===="""

    samples = load_file(data_path)

    tree = OrderedDict()  # restructure scattered steps/titles/subgoals
    cur_title_pos = 0
    cur_subgoal_pos = -1
    stepid2subgoal = {}
    for i, event in enumerate(samples):
        step_type = event['step_type']
        if step_type == "goal":
            cur_title_pos = i
            cur_subgoal_pos = -1
            continue
        elif step_type == "subgoal":
            cur_subgoal_pos = i
            continue
        stepid2subgoal[i] = cur_subgoal_pos
        doc_id = event['doc_id']
        path_id = event['path_id']
        if doc_id not in tree:
            tree[doc_id] = OrderedDict()
        if path_id not in tree[doc_id]:
            tree[doc_id][path_id] = [cur_title_pos]
        tree[doc_id][path_id].append(i)

    # small test
    keys = list(tree.keys())
    for i, key in enumerate(keys):
        if i == len(keys) - 1: break
        if keys[i + 1] <= key:
            print("i", i)
            print("\n\nkeys[i+1]", keys[i + 1])
            print("key", key)
    """====create data samples===="""
    res = []
    random.seed(0)
    for doc_id in tree:
        # print("doc_id", doc_id)
        for path_id in tree[doc_id]:
            # print("path_id", path_id)
            cur_path = tree[doc_id][path_id]
            # print("cur_path", cur_path)

            if len(cur_path) < 3:  # only goal and subgoal
                continue

            categories = samples[cur_path[0]]["categories"]

            entities = {}
            triggers = []
            begin_tracking = False
            for k in range(2, len(cur_path)):  # each step
                idx = cur_path[k]
                sample = samples[idx]
                step_type = sample['step_type']

                """====get entities and their mentions===="""

                if step_type == "event":
                    # begin_tracking = True

                    g = sample["amr"]["graph"]

                    for node_idx, token_pos in enumerate(g["token_pos"]):  # token_pos is headword position
                        lemma, tag = sample["token_features"][token_pos]
                        if 'NN' in tag:
                            if lemma not in entities:
                                entities[lemma] = []
                            elif k >= num_steps_to_begin:
                                # begin_tracking = True
                                if random.randint(0, 9) <= 8:
                                    begin_tracking = True
                            entities[lemma].append([k, node_idx] + g["token_span"][node_idx])  # token_id is node id
                    if int(g["root"]) != -1:
                        try:
                            triggers.append([k, int(g["root"])] + g["token_span"][int(g["root"])])
                        except:
                            print("g", g)
                            print("sample", sample)
                            embed()

                if begin_tracking:

                    if idx not in stepid2subgoal or stepid2subgoal[idx] == -1:
                        breakpoint()

                    res.append({
                        "doc_id": doc_id,
                        'global_doc_id': sample["global_doc_id"],
                        "path_id": path_id,
                        "subgoal_pos": stepid2subgoal[idx],
                        "subgoal_text": samples[stepid2subgoal[idx]]["step"],
                        "doc_type": sample["doc_type"],
                        "categories": categories,  # samples[pos]["categories"],
                        "src_step_ids": cur_path[:(k + 1)],
                        "tgt_step_ids": cur_path[(k + 1):],
                        "entities": deepcopy({key: val for key, val in entities.items() if len(val) > 1}),  # if len(val) > 1
                        "triggers": deepcopy(triggers),
                    })

    dump_file(res, output_path)
    print(f'created samples saved to {output_path}')
