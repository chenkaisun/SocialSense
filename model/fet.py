import torch
import torch.nn.functional as F
import torch.utils.data
from torch.nn import Linear
from transformers import AutoModel, AutoModelForSequenceClassification
from model.model_utils import get_tensor_info, get_text_embeddings
import numpy as np
from utils.utils import load_file
from model.gnn import HGT
# from model.gnn import *
# from train_utils import get_tensor_info

from constants import *


# from torchtext.vocab import GloVe

def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False


class FET(torch.nn.Module):
    def __init__(self, args):
        super(FET, self).__init__()

        self.components = args.components
        self.num_labels = args.out_dim
        id2label = {i: str(i) for i in range(self.num_labels)}
        label2id = {str(i): i for i in range(self.num_labels)}
        # self.plm = AutoModelForSequenceClassification.from_pretrained(args.plm, num_labels=args.out_dim, id2label=id2label, label2id=label2id)
        self.plm = AutoModel.from_pretrained(args.plm)
        self.user_attributes = args.user_attributes
        self.args = args

        args.in_dim = args.plm_hidden_dim
        if "3" in args.user_attributes:
            pretrained_user_embeddings = torch.from_numpy(load_file(args.user_embedding_file))
            num_users, user_embeddings_dim = pretrained_user_embeddings.shape
            # self.user_embeddings = torch.nn.Embedding(1 + num_users, user_embeddings_dim)
            # self.user_embeddings.weight.data[1:].copy_(pretrained_user_embeddings)
            self.user_embeddings = torch.nn.Embedding(num_users, user_embeddings_dim)
            self.user_embeddings.weight.data.copy_(pretrained_user_embeddings)
            # self.user_embeddings.weight.data.copy_()
            # freeze_net(self.user_embeddings)
            args.in_dim = args.plm_hidden_dim + user_embeddings_dim

        self.combiner = Linear(args.in_dim, args.out_dim)

        self.dropout = args.dropout

        # self.loss = torch.nn.CrossEntropyLoss()
        self.loss = torch.nn.MultiLabelSoftMarginLoss()
        # self.loss = nn.BCEWithLogitsLoss()

        self.the_zero = torch.tensor(0, dtype=torch.long, device=args.device)
        self.the_one = torch.tensor(1, dtype=torch.long, device=args.device)
        # self.rand = torch.rand(1, device=args.device)

    def forward(self, input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
                masked_input_ids=None,
                masked_texts_attention_mask=None,
                user_embed_idxs=None,
                tweet_ids=None,
                in_train=None,
                print_output=False):
        final_vec = []

        "=========Original Text Encoder=========="
        # embed()
        # print("texts", get_tensor_info(texts))
        # print("texts_attn_mask", get_tensor_info(texts_attn_mask))
        hidden_states = self.plm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state[:, 0, :]  # .pooler_output
        # hidden_states = self.plm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state[:,0,:]#.pooler_output
        # print("hidden_states", get_tensor_info(hidden_states))

        final_vec.append(hidden_states)

        if "co" in self.components:
            hidden_states_context_only = self.plm(input_ids=masked_input_ids,
                                                  attention_mask=masked_texts_attention_mask,
                                                  return_dict=True).last_hidden_state
            final_vec.append(hidden_states_context_only)

        if "3" in self.user_attributes:
            "=========User Attributes=========="
            # print("user_ids", get_tensor_info(user_ids))
            # user_ids = user_ids.unsqueeze(1)
            # print("user_ids", get_tensor_info(user_ids))
            user_embeddings = self.user_embeddings(user_embed_idxs)
            # print("user_embeddings", get_tensor_info(user_embeddings))
            final_vec.append(user_embeddings)

        "=========Classification=========="
        output = self.combiner(torch.cat(final_vec, dim=-1))
        # print("output", get_tensor_info(output))

        # print("output", output)
        return {
            "logits": output
        }

        if in_train:
            # label smoothing
            # return self.criterion(output, labels)
            return self.loss(output, labels)
            return torch.nn.functional.cross_entropy(output, labels)
            return torch.nn.functional.binary_cross_entropy_with_logits(output, labels)
        pred_out = (torch.sigmoid(output) > 0.5).float()
        # return torch.argmax(F.log_softmax(output, dim=-1), dim=-1)
        # print("sigmoid output")
        # pp(torch.sigmoid(output))
        # print("pred_out")
        # pp(pred_out)
        # pred_out = torch.argmax(torch.softmax(output, dim=-1), dim=-1)
        # print('pred_out', get_tensor_info(pred_out))
        return output
        return pred_out


class FET_GNN(torch.nn.Module):
    def __init__(self, args, data):
        super(FET_GNN, self).__init__()

        self.components = args.components
        self.num_labels = args.out_dim
        # self.plm = AutoModel.from_pretrained(args.plm)
        self.user_attributes = args.user_attributes
        self.args = args
        # args.in_dim=args.plm_hidden_dim
        # if "3" in args.user_attributes:

        if args.encode_on_the_way:
            self.plm = AutoModel.from_pretrained(args.plm)
            self.tokenizer = args.tokenizer
            # args.in_dim=args.plm_hidden_dim
        # else:
        ue_dir = f"{args.subset_dir}user_embeddings/{args.embedding_plm.split('/')[-1]}/"
        id_to_index_ue_fn, embedding_ue_fn = f"{ue_dir}id_to_index_{args.user_attributes}.json", f"{ue_dir}embedding_{args.user_attributes}.npy"
        pretrained_user_embeddings = torch.from_numpy(load_file(embedding_ue_fn))
        # pretrained_user_embeddings = torch.from_numpy(load_file(f"{args.ue_dir}embedding.npy"))

        num_users, user_embeddings_dim = pretrained_user_embeddings.shape
        self.user_embeddings = torch.nn.Embedding(1 + num_users, user_embeddings_dim)
        print("1 + num_users, user_embeddings_dim", 1 + num_users, user_embeddings_dim)

        pretrained_post_embeddings = torch.from_numpy(load_file(f"{args.pe_dir}embedding.npy"))
        num_posts, post_embeddings_dim = pretrained_post_embeddings.shape
        self.post_embeddings = torch.nn.Embedding(1 + num_posts, post_embeddings_dim)
        print("1 + num_posts, post_embeddings_dim", 1 + num_posts, post_embeddings_dim)

        # if args.load_pretrained_ent_emb:
        self.user_embeddings.weight.data[1:].copy_(pretrained_user_embeddings)
        self.user_embeddings.weight.data[0].copy_(torch.rand(user_embeddings_dim))
        self.post_embeddings.weight.data[1:].copy_(pretrained_post_embeddings)
        self.post_embeddings.weight.data[0].copy_(torch.rand(post_embeddings_dim))
        # if args.use_value:
        self.node_value_embeddings = torch.nn.Embedding(1 + len(HUMAN_VALUES) + len(MORAL_VALUES), user_embeddings_dim)

        if args.freeze_ent_emb:
            freeze_net(self.user_embeddings)
            freeze_net(self.post_embeddings)
            # freeze_net(self.node_value_embeddings)


        # if "e" in args.component:
        #     self.edge_embeddings = torch.nn.Embedding(1 + 3+1, post_embeddings_dim)

        if not args.load_pretrained_ent_emb and not args.freeze_ent_emb:
            self.node_user_embeddings = torch.nn.Embedding(1 + num_users, user_embeddings_dim)
            self.node_post_embeddings = torch.nn.Embedding(1 + num_posts, post_embeddings_dim)

        # self.user_embeddings = torch.nn.Embedding( num_users, user_embeddings_dim) If you want the layer to be trainable, pass freeze=False, by default it's not as you want.
        # self.user_embeddings.weight.data.copy_(pretrained_user_embeddings)
        # self.user_embeddings.weight.data.copy_()
        # freeze_net(self.user_embeddings)
        self.gnn = HGT(args=args, num_heads=2, num_layers=2, data=data)

        # args.in_dim = args.plm_hidden_dim + user_embeddings_dim

        in_dim = 0
        if not args.no_orig:
            in_dim += args.embedding_plm_hidden_dim * 2
        if "gnn" in self.components:
            in_dim += args.g_dim * 2
        if "user_l" in self.components:
            in_dim += args.embedding_plm_hidden_dim * 2
        # if args.no_orig and "gnn" in self.components and args.no_user_news==1:
        #     in_dim=args.g_dim+args.embedding_plm_hidden_dim
        if args.no_user_news in [1,2]:
            self.p_proj = Linear(args.embedding_plm_hidden_dim, args.g_dim )
        self.combiner = Linear(in_dim, args.out_dim)
        self.dropout = args.dropout

        # self.loss = torch.nn.CrossEntropyLoss()
        self.loss = torch.nn.MultiLabelSoftMarginLoss()
        # self.loss = nn.BCEWithLogitsLoss()

        self.the_zero = torch.tensor(0, dtype=torch.long, device=args.device)
        self.the_one = torch.tensor(1, dtype=torch.long, device=args.device)
        # self.the_last = torch.tensor(num_users-1, dtype=torch.long, device=args.device)
        # self.rand = torch.rand(1, device=args.device)

    def forward(self, input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
                masked_input_ids=None,
                masked_texts_attention_mask=None,
                user_embed_idxs=None,
                data=None,
                pairs=None,
                tweet_ids=None,
                in_train=None,
                list_of_sentences=None,
                print_output=False):
        final_vec = []

        # "=========Original Text Encoder=========="
        # hidden_states = self.plm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state[:, 0, :]  # .pooler_output
        # final_vec.append(hidden_states)
        data = data.to(self.args.device)
        user_x = self.user_embeddings(data["user"].x)  # test -1
        # print("user_x", get_tensor_info(user_x))
        # print("user_embeddings", get_tensor_info(user_embeddings))
        post_x = self.post_embeddings(data["news"].x)
        # print("post_x", get_tensor_info(post_x))

        value_x = self.node_value_embeddings(data["val"].x)

        "=========Orig Encoder=========="

        if not self.args.no_orig:
            if self.args.encode_on_the_way:
                pairs_embeddings = get_text_embeddings(list_of_sentences, self.plm, self.tokenizer, self.args)
            else:
                pairs_embeddings = torch.cat([user_x[pairs[:, 0]], post_x[pairs[:, 1]]], dim=-1)
                # print("pairs_embeddings", get_tensor_info(pairs_embeddings))

            final_vec.append(pairs_embeddings)

        "=========GNN=========="

        """see if userx change after  gnn"""  #######
        if not self.args.load_pretrained_ent_emb and not self.args.freeze_ent_emb:
            user_x = self.node_user_embeddings(data["user"].x)  # test -1
            post_x = self.node_post_embeddings(data["news"].x)
            value_x = self.node_value_embeddings(data["val"].x)
            # print("post_x", get_tensor_info(post_x))
        if "gnn" in self.components:
            x_dict = {"user": user_x} #, "news": post_x
            edge_index_dict = {k: data.edge_index_dict[k] for k in [('user', 'follows', 'user')]} #('news', 'rev_replies', 'user'), , ('user', 'replies', 'news')
            if self.args.no_user_news != 1:
                # del x_dict["news"]
                # del edge_index_dict[('news', 'rev_replies', 'user')]
                # del edge_index_dict[('user', 'replies', 'news')]

                x_dict["news"] = post_x
                for k in [('news', 'rev_replies', 'user'), ('user', 'replies', 'news')]:
                    edge_index_dict[k] = data.edge_index_dict[k]
            if self.args.use_value:
                x_dict["val"] = value_x
                for k in [('user', 'contains', 'val'), ('val', 'rev_contains', 'user')]:
                    edge_index_dict[k] = data.edge_index_dict[k]

            #     {"user__replies__news": data.edge_index_dict['user__replies__news'], "news": post_x}
            # "user__replies__news"

            # if "e" in self.component:
            #     edge_attr_dict = {k: data.edge_attr_dict[k] for k in [('news', 'rev_replies', 'user'), ('user', 'follows', 'user'), ('user', 'replies', 'news')]}

            x_dict = self.gnn(x_dict, edge_index_dict)
            pairs_embeddings = torch.cat([x_dict["user"][pairs[:, 0]], self.p_proj(post_x[pairs[:, 1]]) if self.args.no_user_news in [1,2] else x_dict["news"][pairs[:, 1]]], dim=-1)
            # print("pairs_embeddings", get_tensor_info(pairs_embeddings))
            final_vec.append(pairs_embeddings)

        "=========Classification=========="
        output = self.combiner(torch.cat(final_vec, dim=-1))
        output = output.unsqueeze(0)
        # print("output", get_tensor_info(output))
        # print("output", output)
        return {
            "logits": output
        }

        if in_train:
            # label smoothing
            # return self.criterion(output, labels)
            return self.loss(output, labels)
            return torch.nn.functional.cross_entropy(output, labels)
            return torch.nn.functional.binary_cross_entropy_with_logits(output, labels)
        pred_out = (torch.sigmoid(output) > 0.5).float()
        # return torch.argmax(F.log_softmax(output, dim=-1), dim=-1)
        # print("sigmoid output")
        # pp(torch.sigmoid(output))
        # print("pred_out")
        # pp(pred_out)
        # pred_out = torch.argmax(torch.softmax(output, dim=-1), dim=-1)
        # print('pred_out', get_tensor_info(pred_out))
        return output
        return pred_out
