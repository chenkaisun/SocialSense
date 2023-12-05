import torch
import argparse
import os
import scipy.sparse as sp
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from model import InfoVGAE
from PID import PIDControl
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error, \
    precision_score, recall_score, cohen_kappa_score
import numpy as np
import random
from scipy.stats import pearsonr, spearmanr

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

ATTACH = True

parser = argparse.ArgumentParser()
# Quick experiment, no need to specify other parameters after using config
parser.add_argument('--config_name', type=str, default=None, help="Use existing config to reproduce experiment quickly")

# General
parser.add_argument('--model', type=str, default="InfoVGAE", help="model to use")
parser.add_argument('--epochs', type=int, default=500, help='epochs (iterations) for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate of model')
parser.add_argument('--device', type=str, default="0", help='cpu/gpu device')
parser.add_argument('--num_process', type=int, default=40, help='num_process for pandas parallel')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--use_cuda', action="store_true", help='whether to use cuda device')

# Data
parser.add_argument('--dataset', type=str, help='dataset to use')
parser.add_argument('--add_self_loop', type=bool, default=True, help='add self loop for adj matrix')
parser.add_argument('--directed', type=bool, default=False, help='use directed adj matrix')
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--data_json_path', type=str, default=None)
parser.add_argument('--follow_path', type=str, default=None)
parser.add_argument('--use_follow', type=bool, default=False)
parser.add_argument('--stopword_path', type=str, default=None)
parser.add_argument('--keyword_path', type=str, default="N")
parser.add_argument('--kthreshold', type=int, default=5, help='minimum keyword count to keep the sample')
parser.add_argument('--uthreshold', type=int, default=3, help='minimum user tweet count to keep the sample')

# For GAE/VGAE model
parser.add_argument('--hidden1_dim', type=int, default=512, help='graph conv1 dim')
parser.add_argument('--hidden2_dim', type=int, default=256, help='graph conv2 dim')
parser.add_argument('--use_feature', type=bool, default=True, help='Use feature')
parser.add_argument('--num_user', type=int, default=None, help='Number of users, usually no need to specify.')
parser.add_argument('--num_assertion', type=int, default=None, help='Number of assertions, usually no need to specify.')
parser.add_argument('--pos_weight_lambda', type=float, default=1.0, help='Lambda for positive sample weight')

# For Discriminator
parser.add_argument('--gamma', type=float, default=1e-3, help='weight for tc loss')
parser.add_argument('--lr_D', type=float, default=1e-3, help='learning rate for discriminator')
parser.add_argument('--beta1_D', type=float, default=0.5, help='beta1 for discriminator optimizer')
parser.add_argument('--beta2_D', type=float, default=0.9, help='beta2 for discriminator optimizer')
parser.add_argument('--num_classes', type=int, default=None, help='beta2 for discriminator optimizer')

# Result
parser.add_argument('--output_path', type=str, default="./output", help='Path to save the output')

args = parser.parse_args()

user_user_json = "dataset/emnlp/user_edges.json"

# train_json = "dataset/emnlp/response_data_balanced_train_anno.json"
# test_json = "dataset/emnlp/response_data_balanced_test_anno.json"

# train_json = "dataset/emnlp/train_anno_case_lurker.json"
# test_json = "dataset/emnlp/test_anno_case_lurker.json"

train_json = "dataset/emnlp/response_data_balanced_train_anno.json"
test_json = "dataset/emnlp/test_anno_case_unseen.json"

# Load data
def load_data(json_path):
    return pd.read_json(json_path, dtype={"user_id": str, "post_id": str})

train_df = load_data(train_json)
test_df = load_data(test_json)
with open(user_user_json, "r") as fin:
    user_user_data = json.load(fin)
# print(train_df, test_df)

intensity_label_distribution = sorted(list(train_df["label_intensity"].value_counts().to_dict().items()), key=lambda x: x[0])
polarity_label_distribution = sorted(list(train_df["label_polarity"].value_counts().to_dict().items()), key=lambda x: x[0])
intensity_label_distribution = np.array([v for k,v in intensity_label_distribution]).astype("float32")
polarity_label_distribution = np.array([v for k,v in polarity_label_distribution]).astype("float32")
intensity_label_distribution /= np.sum(intensity_label_distribution)
polarity_label_distribution /= np.sum(polarity_label_distribution)
intensity_split = np.concatenate([np.zeros(1).reshape(-1), intensity_label_distribution], axis=0).cumsum()
polarity_split = np.concatenate([np.zeros(1).reshape(-1), polarity_label_distribution], axis=0).cumsum()
def get_prediction(prob, split):
    for i in range(split.shape[0] - 1):
        if prob >= split[i] and prob <= split[i + 1]:
            return i

# print(intensity_label_distribution)
# print(polarity_label_distribution)
# print(intensity_split)
# print(polarity_split)
# print(get_prediction(0.7, intensity_split))
# print(get_prediction(0.7, polarity_split))
# exit()

# build_adjacency_matrix_and_mask
joint_df = pd.concat([train_df, test_df], axis=0)
# print(joint_df)
userlist = set(joint_df["user_id"].unique())
print(f"original user: {len(userlist)}")
if ATTACH:
    # expand user_list
    for fr, to in user_user_data:
        if (fr in userlist) or (to in userlist):
            userlist.add(fr)
            userlist.add(to)
userlist = list(userlist)
userlist_set = set(userlist)
print(f"expanded user: {len(userlist)}")
asserlist = list(joint_df["post_id"].unique())
num_user = len(userlist)
num_asser = len(asserlist)
user_to_id = {v: i for i,v in enumerate(userlist)}
asser_to_id = {v: i + num_user for i, v in enumerate(asserlist)}
id_to_user = {v: k for k,v in user_to_id.items()}
id_to_asser = {v: k for k,v in asser_to_id.items()}

adjacency_matrix = np.zeros(shape=(num_user + num_asser, num_user + num_asser)).astype("float32")
is_test_mask = np.zeros(shape=(num_user + num_asser, num_user + num_asser)).astype("float32")
test_label = np.zeros(shape=(num_user + num_asser, num_user + num_asser)).astype("float32") - 214738467
supervised_samples = [[], [], []]
test_samples = [[], [], []]
for i, row in train_df.iterrows():
    user_name = row["user_id"]
    asser_name = row["post_id"]
    label = max(row["label_intensity"] - 3, 0)
    supervised_samples[0].append(user_to_id[user_name])
    supervised_samples[1].append(asser_to_id[asser_name])
    supervised_samples[2].append(row["label_intensity"])
    adjacency_matrix[user_to_id[user_name], asser_to_id[asser_name]] = label
    adjacency_matrix[asser_to_id[asser_name], user_to_id[user_name]] = label
for i, row in test_df.iterrows():
    user_name = row["user_id"]
    asser_name = row["post_id"]
    label = row["label_intensity"]
    test_samples[0].append(user_to_id[user_name])
    test_samples[1].append(asser_to_id[asser_name])
    test_samples[2].append(row["label_intensity"])
    is_test_mask[user_to_id[user_name], asser_to_id[asser_name]] = 1
    is_test_mask[asser_to_id[asser_name], user_to_id[user_name]] = 1
    test_label[user_to_id[user_name], asser_to_id[asser_name]] = label
    test_label[asser_to_id[asser_name], user_to_id[user_name]] = label
supervised_samples[2] = torch.from_numpy(np.array(supervised_samples[2])).view(-1).long()
test_samples[2] = torch.from_numpy(np.array(test_samples[2])).view(-1).long()
print("Built adj matrix")

if ATTACH:
    for fr, to in tqdm(user_user_data, total=len(user_user_data)):
        if fr in userlist_set and to in userlist_set:
            adjacency_matrix[user_to_id[fr], user_to_id[to]] = 1
            adjacency_matrix[user_to_id[to], user_to_id[fr]] = 1  # TODO += 1
            # print("connected")

# eliminate diag
adjacency_matrix = adjacency_matrix - np.diag(adjacency_matrix.diagonal())

# Build feature
feature = np.diag(np.ones(num_user + num_asser)).astype("float32")

# Training
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized.todense()

adj_norm = preprocess_graph(adjacency_matrix)
adj_label = adjacency_matrix + np.diag(np.ones(adjacency_matrix.shape[0]))

adj_norm = torch.from_numpy(adj_norm).float()
adj_label = torch.from_numpy(adj_label).float()
features = torch.from_numpy(feature).float()
is_test_mask = torch.from_numpy(is_test_mask).float()
test_label = torch.from_numpy(test_label).float()

weight_mask = adj_label.view(-1) != 0
weight_tensor = torch.ones(weight_mask.size(0)).float()
# non_zero = weight_mask.sum()
# pos_weight = float((adj_label == 0).sum()) / non_zero * 2
pos_weight = float(adjacency_matrix.shape[0] * adjacency_matrix.shape[0] - adjacency_matrix.sum()) / adjacency_matrix.sum()
# print("pos", pos_weight, pos_weight_0)
weight_tensor[weight_mask] = pos_weight
weight_tensor[is_test_mask.view(-1) == 1] = 0
do_not_consider_part = torch.ones(adj_label.shape)
do_not_consider_part[num_user:, num_user:] = 0
weight_tensor = weight_tensor * (do_not_consider_part.view(-1))

# norm = adjacency_matrix.shape[0] * adjacency_matrix.shape[0] / float((adjacency_matrix.shape[0] * adjacency_matrix.shape[0] - non_zero) * 2)
norm = adjacency_matrix.shape[0] * adjacency_matrix.shape[0] / float((adjacency_matrix.shape[0] * adjacency_matrix.shape[0] - adjacency_matrix.sum()) * 2)
# print("norm", norm, norm_0)

if args.use_cuda:
    adj_norm = adj_norm.cuda()
    adj_label = adj_label.cuda()
    features = features.cuda()
    weight_tensor = weight_tensor.cuda()
    supervised_samples[2] = supervised_samples[2].cuda()
    test_samples[2] = test_samples[2].cuda()

setattr(args, "input_dim", num_user + num_asser)
setattr(args, "num_user", num_user)
setattr(args, "num_assertion", num_asser)
setattr(args, "num_classes", 7)

# init model and optimizer
model = InfoVGAE(args, adj_norm)
if args.use_cuda:
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-7)

def get_prf(targets, preds, average="micro", verbose=False):
    precision = precision_score(targets, preds, average=average)
    recall = recall_score(targets, preds, average=average)
    f1 = f1_score(targets, preds, average=average)
    if verbose: print(f"{average}: precision {precision} recall {recall} f1 {f1}")
    return precision, recall, f1

def get_scores_multilabel_clf(preds, labels, verbose=False, args=None):
    # print("get_scores_multilabel_clf")
    # print("logits, labels", logits, labels)
    # preds = (torch.sigmoid(get_tensor_float(logits)) > 0.5).int().tolist()
    sent_labels, sent_preds = labels, preds

    score_dict = {}
    mi_precision, mi_recall, mi_f1 = get_prf(sent_labels, sent_preds, average="micro", verbose=False)
    ma_precision, ma_recall, ma_f1 = get_prf(sent_labels, sent_preds, average="macro", verbose=False)
    acc=accuracy_score(sent_labels, sent_preds)
    #get pearson and spearman

    pearson = pearsonr(preds, labels)[0]
    spearman = spearmanr(preds, labels)[0]
    kappa = cohen_kappa_score(preds, labels, weights="quadratic")
    if np.isnan(pearson):
        pearson = 0
    if np.isnan(spearman):
        spearman = 0
    if np.isnan(kappa):
        kappa = 0
    score_dict.update({
        "mif1": mi_f1,
        "maf1": ma_f1,
        "accuracy": acc,
        "miprecision": mi_precision,
        "mirecall": mi_recall,
        "maprecision": ma_precision,
        "marecall": ma_recall,
        "pearson": pearson,
        "spearman": spearman,
        "kappa": kappa,
    })
    for key in score_dict:
        score_dict[key] = round(score_dict[key]*100, 4)
    return score_dict

classifiy_criterion = torch.nn.CrossEntropyLoss()

# train model
Kp = 0.001
Ki = -0.001
PID = PIDControl(Kp, Ki)
Exp_KL = 0.005
for epoch in range(args.epochs):
    # Train VAE
    z = model.encode_normal(features)

    z_u = z[supervised_samples[0]].view(-1, z.shape[1])
    z_a = z[supervised_samples[1]].view(-1, z.shape[1])
    zz = torch.cat([z_a, z_u], dim=1)
    output = F.relu(model.linear1(zz))
    output = model.linear2(output)
    # print(supervised_samples[2])
    # print(output.shape, supervised_samples[2].shape)
    loss_classification = classifiy_criterion(output, supervised_samples[2])

    A_pred = model.decode(z)
    vae_recon_loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1),
                                                   weight=weight_tensor)
    kl_divergence = 0.5 / A_pred.size(0) * (1 + 2 * model.logstd - model.mean ** 2 -
                                            torch.exp(model.logstd) ** 2).sum(1).mean()
    weight = PID.pid(Exp_KL, kl_divergence.item())  # get the weight on KL term with PI module
    vae_loss = loss_classification
               # - kl_divergence

    optimizer.zero_grad()
    vae_loss.backward()
    optimizer.step()

    print(f"Epoch: {epoch}, VAE Loss: {vae_recon_loss.data}, Classification loss: {loss_classification.data}, kl_divergence: {- kl_divergence}")

    # evaluate
    with torch.no_grad():
        z = model.encode_normal(features)

        z_u = z[test_samples[0]].view(-1, z.shape[1])
        z_a = z[test_samples[1]].view(-1, z.shape[1])
        zz = torch.cat([z_a, z_u], dim=1)
        output = F.relu(model.linear1(zz))
        output = model.linear2(output)
        preds = torch.argmax(F.softmax(output, dim=1), dim=1).cpu().numpy()
        labels = test_samples[2].cpu().numpy()
        # print(preds.shape, labels.shape)
        result = get_scores_multilabel_clf(preds, labels, verbose=True)
        print("spearman: {}, pearson: {}".format(result["spearman"], result["pearson"]))


        # A_pred = model.decode(z)
        # A_pred_test = A_pred[test_label != -214738467].view(-1).cpu()
        # A_label_test = test_label[test_label != -214738467].view(-1).cpu()
        #
        # assert A_label_test.shape[0] == A_pred_test.shape[0]
        #
        # # print("Intensity:")
        # preds = []
        # labels = []
        # for i in range(A_pred_test.shape[0]):
        #     preds.append(get_prediction(A_pred_test[i], intensity_split))
        #     labels.append(float(A_label_test[i]))
        # print(preds[:50])
        # print(labels[:50])
        # print(get_scores_multilabel_clf(preds, labels, verbose=True)["spearman"])
        #
        # # print("Polarity:")
        # preds = []
        # labels = []
        # for i in range(A_pred_test.shape[0]):
        #     preds.append(get_prediction(A_pred_test[i], polarity_split))
        #     if A_label_test[i] >= 0 and A_label_test[i] < 3:
        #         labels.append(0)
        #     elif A_label_test[i] == 3:
        #         labels.append(1)
        #     elif A_label_test[i] > 3:
        #         labels.append(2)
        #     else:
        #         raise NotImplementedError()
        # print(get_scores_multilabel_clf(preds, labels, verbose=True)["mif1"])


    # print(vae_loss)







