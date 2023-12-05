from options import *
from train import train, train_clf_trainer, eval_special
from train_utils import *
from utils.utils import check_error2, get_best_ckpt, load_file_batch, load_file, module_exists
from evaluate import evaluate
from data import PrimitiveGenerationDataset, PrimitivePredictionDataset, PrimitivePredictionDatasetGNN
import wandb
import os
import shutil
import gc
from constants import *
from sklearn.utils.class_weight import compute_class_weight
from copy import deepcopy
from model.fet import FET,FET_GNN
if module_exists("torch_geometric"):
    from torch_geometric.data import Batch, Data
    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T
    from torch_geometric import seed_everything


def main_func():
    """main"""
    """=========INIT========="""

    "======SETTING======"

    args = read_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    seed_everything(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # if args.n_gpu > 0 and torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.debug or args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)


    if not args.debug and not args.task_mode == "ret":
        wandb.init()
        print("WANDB")

    """=========Set Tokenizer========="""
    tokenizer = get_tokenizer(args.plm, slow_connection=args.slow_connection, args=args)
    args.tokenizer = tokenizer

    embedding_plm_tokenizer = AutoTokenizer.from_pretrained(args.embedding_plm)
    args.embedding_plm_max_seq_len=embedding_plm_tokenizer.model_max_length=embedding_plm_tokenizer.model_max_length if embedding_plm_tokenizer.model_max_length <= 2000 else 512  # TODO
    args.embedding_plm_tokenizer = embedding_plm_tokenizer

    #

    """=========Set MetaData & Parameter========="""
    labels = LABEL_SPACE[args.task_setting]
    if args.task_setting == 1 and args.pure_int:
        labels = [0,1,2,3]
    args.label_name = LABEL_NAME[args.task_setting]
    args.pred_label_category = PRED_LABEL_CATEGORY[args.task_setting]
    args.out_dim = len(labels)
    args.multi_label = False

    """=========Setup Data========="""
    """SPecial"""
    train_data, dev_data, test_data = None, None, None
    user_data, post_data, graph_data = None, None, None
    extra_data = None
    if args.user_file:
        if args.debug:
            tmp = load_file_batch(filenames=[args.user_file, args.post_file, args.graph_file, args.id_to_index_file])
            extra_data = (tmp[0], tmp[1], {}, tmp[2], tmp[3])
        else:
            extra_data = load_file_batch(filenames=[args.user_file, args.post_file, args.history_file, args.graph_file, args.id_to_index_file])
            # extra_data[2]={}
    if args.task_mode == "gen":
        train_data = PrimitiveGenerationDataset(args, args.train_file, tokenizer, in_train=True, extra_data=extra_data, cur_split="train")
        dev_data = PrimitiveGenerationDataset(args, args.dev_file, tokenizer, extra_data=extra_data, cur_split="dev")
        test_data = PrimitiveGenerationDataset(args, args.test_file, tokenizer, extra_data=extra_data, cur_split="test")
    elif args.task_mode == "clf":
        train_data = PrimitivePredictionDataset(args, args.train_file, tokenizer, labels=labels, in_train=True, extra_data=extra_data, cur_split="train")
        dev_data = PrimitivePredictionDataset(args, args.dev_file, tokenizer, labels=labels, extra_data=extra_data, cur_split="dev")
        test_data = PrimitivePredictionDataset(args, args.test_file, tokenizer, labels=labels, extra_data=extra_data, cur_split="test")
        # if args.eval_only and args.target_splits == [2]:
        #     train_data=dev_data=test_data
        # else:
        #     train_data = PrimitivePredictionDataset(args, args.train_file, tokenizer, labels=labels, in_train=True, extra_data=extra_data, cur_split="train")
        #     dev_data = PrimitivePredictionDataset(args, args.dev_file, tokenizer, labels=labels, extra_data=extra_data, cur_split="dev")

        print("train_data.get_class_weights()", train_data.get_class_weights())
    print("prior to gnn len(train_data), len(dev_data), len(test_data)", len(train_data), len(dev_data), len(test_data))

    """=========Special Cases========="""
    if args.only_cache_data:  # no need to run program
        return
    if args.top_few != -1:
        test_data.instances = test_data.instances[: args.top_few]
    if args.data_mode.startswith("fs"):
        train_data.instances = train_data.instances[: int(args.data_mode[2:])]
    if args.debug:
        train_data.instances, dev_data.instances, test_data.instances = train_data.instances[:8], dev_data.instances[:4], test_data.instances[:20]

    gnn_data = None
    if args.use_gnn:
        train_data = PrimitivePredictionDatasetGNN(args, "filename", tokenizer=tokenizer, labels=labels, label_map=None, in_train=True, extra_data=extra_data, tr=train_data,
                                                   val=dev_data, test=test_data)  # , embedding_plm_tokenizer=embedding_plm_tokenizer
        dev_data = deepcopy(train_data)
        test_data = deepcopy(train_data)
        gnn_data = train_data[-1]["data"]
        for i, (split, d) in enumerate(zip(["train", "dev", "test"], [train_data, dev_data, test_data])):
            d[-1]['labels'] = d[-1][f'{split}_labels']
            d[-1]['pairs'] = d[-1][f'{split}_pairs']
            d[-1]['orig_data'] = d[-1]['orig_data'][f'{split}']
            if i != 0: d[-1]['in_train'] = False
        print("gnn pairs len(train_data), len(dev_data), len(test_data)", len(train_data[-1]['pairs'] ), len(dev_data[-1]['pairs'] ), len(test_data[-1]['pairs'] ))

    """=========Setting Model========="""
    args, model, optimizer = setup_common(args, tokenizer, data=gnn_data)
    print("setup done")

    if args.task_mode in ["gen", "prt"]:
        if args.eval_special:
            eval_special(args, model, tokenizer, (train_data, dev_data, test_data), extra_data=extra_data)
            return
        train(args, model, optimizer, tokenizer, (train_data, dev_data, test_data))
    elif args.task_mode == "clf":
        train_clf_trainer(args, model, optimizer, tokenizer, (train_data, dev_data, test_data))


if __name__ == '__main__':
    # with launch_ipdb_on_exception():
    #     main_func()
    main_func()
