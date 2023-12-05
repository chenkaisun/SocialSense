


## Environments
- Ubuntu-18.0.4
- Python (3.7)
- Cuda (11.1)

## Installation
Install [Pytorch](https://pytorch.org/) 1.9.0 and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) 2.0.3, then run the following in the terminal:
```shell
cd INCAS # get into INCAS folder
conda create -n SocialSense python=3.7 -y  # create a new conda environment
conda activate SocialSense

chmod +x scripts/setup.sh
./scripts/setup.sh
```
Create a directory `../twitter_crawl/data_new2/CNN/` containing data from [here](https://github.com/chenkaisun/response_forecasting). Then download additional data from [here](https://drive.google.com/drive/folders/10jvh6HjTOUCT1qEh1BXncFOdKBjqXYUE?usp=sharing) to `../twitter_crawl/data_new2/CNN/`. 

## Note
The running of the system might require [wandb](wandb.ai) account login

## Train Models
To train the models, edit the parameters with $ appended in the following and run in the terminal.
Note: --case_study: is from 0 to 2, indicating main, case 1, and case 2. --task_setting: choose from 1 (intensity) or 2 (polarity). --use_value: it means use belief edge or not, --user_attributes: means which text attributes to use such as 1(profile) and 2(history) or both. --load_pretrained_ent_emb: 0(random initialization) or 1(use text embeddings). --activation is a string. 

```shell
python main.py \
        --use_cache 0 \
        --config response_pred_sent_gnn \
        --scheduler 'linear' \
        --components "gnn" \
        --g_dim $g_dim \
        --lr $lr \
        --activation $activation \
        --num_epochs $num_epochs \
        --exp_msg " " \
        --task_setting $task_setting \
        --case_study $case_study \
        --no_user_news $no_user_news \
        --use_value $use_value \
        --user_attributes $user_attributes \
        --load_pretrained_ent_emb $load_pretrained_ent_emb \
        --num_gnn_layers $num_gnn_layers \
        --num_gnn_heads $num_gnn_heads \
        --freeze_ent_emb 0 \
        --dropout 0.2 \
        --embedding_plm deberta-base \
        --embedding_batch_size 4 \
        --plm deberta-base
```

