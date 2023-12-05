# InfoVGAE baseline
This repo is for EMNLP paper submission 1221, including the modified version for the baseline of InfoVGAE.

## Environment

```
absl-py                      1.3.0
aiohttp                      3.8.4
aiosignal                    1.3.1
astunparse                   1.6.3
async-timeout                4.0.2
attrs                        23.1.0
auto-sklearn                 0.15.0
beautifulsoup4               4.10.0
cachetools                   5.2.1
certifi                      2021.10.8
charset-normalizer           2.0.12
click                        8.1.3
cloudpickle                  2.2.0
ConfigSpace                  0.4.21
cramjam                      2.6.2
cycler                       0.11.0
Cython                       0.29.32
dask                         2022.9.1
deep-translator              1.7.0
deeprobust                   0.2.5
dill                         0.3.4
distributed                  2022.9.1
distro                       1.7.0
emcee                        3.1.2
fastparquet                  2023.2.0
flatbuffers                  23.1.4
fonttools                    4.29.1
frozenlist                   1.3.3
fsspec                       2022.8.2
future                       0.18.3
gast                         0.4.0
gensim                       3.8.3
google-auth                  2.15.0
google-auth-oauthlib         0.4.6
google-pasta                 0.2.0
grpcio                       1.51.1
h5py                         3.7.0
HeapDict                     1.0.1
idna                         3.3
imageio                      2.22.2
importlib-metadata           5.0.0
Jinja2                       3.1.2
joblib                       1.1.0
keras                        2.11.0
kiwisolver                   1.3.2
liac-arff                    2.5.0
libclang                     15.0.6.1
lightning-utilities          0.8.0
llvmlite                     0.39.1
locket                       1.0.0
Markdown                     3.4.1
MarkupSafe                   2.1.1
matplotlib                   3.5.1
msgpack                      1.0.4
multidict                    6.0.4
networkx                     2.8
numba                        0.56.3
numpy                        1.22.1
oauthlib                     3.2.2
opt-einsum                   3.3.0
ordered-set                  4.1.0
packaging                    21.3
pandarallel                  1.5.5
pandas                       1.5.3
partd                        1.3.0
Pillow                       9.0.1
pip                          21.3.1
plotly                       5.15.0
protobuf                     3.19.6
psutil                       5.9.2
pyarrow                      11.0.0
pyasn1                       0.4.8
pyasn1-modules               0.2.8
pynisher                     0.6.4
pynndescent                  0.5.10
pyparsing                    3.0.7
pyrfr                        0.8.3
python-dateutil              2.8.2
pytorch-lightning            1.2.0
pytz                         2021.3
PyWavelets                   1.4.1
PyYAML                       6.0
requests                     2.27.1
requests-oauthlib            1.3.1
rsa                          4.9
scikit-image                 0.19.3
scikit-learn                 0.24.2
scipy                        1.7.3
seaborn                      0.11.2
setuptools                   60.2.0
six                          1.16.0
smac                         1.2
smart-open                   6.2.0
sortedcontainers             2.4.0
soupsieve                    2.3.1
tblib                        1.7.0
tenacity                     8.2.2
tensorboard                  2.11.0
tensorboard-data-server      0.6.1
tensorboard-plugin-wit       1.8.1
tensorboardX                 2.5.1
tensorflow                   2.11.0
tensorflow-estimator         2.11.0
tensorflow-io-gcs-filesystem 0.29.0
termcolor                    2.2.0
texttable                    1.6.4
threadpoolctl                3.1.0
tifffile                     2022.10.10
toolz                        0.12.0
torch                        1.11.0+cu113
torch-cluster                1.6.0
torch-geometric              2.0.4
torch-scatter                2.0.9
torch-sparse                 0.6.13
torch-spline-conv            1.2.1
torchaudio                   0.11.0+cu113
torchmetrics                 0.11.4
torchvision                  0.12.0
tornado                      6.1
tqdm                         4.62.3
typing_extensions            4.1.1
uiuc-incas-client            2.0.0
umap-learn                   0.5.3
urllib3                      1.26.8
Werkzeug                     2.2.2
wheel                        0.37.1
wordcloud                    1.8.2.2
wrapt                        1.14.1
xgboost                      1.6.0
yarl                         1.9.2
zict                         2.2.0
zipp                         3.9.0
```

## Training

To run InfoVGAE on Eurovision dataset:

```
python3 main.py --config_name InfoVGAE_eurovision_3D
```

To run InfoVGAE on Election dataset:

```
python3 main.py --config_name InfoVGAE_election_3D
```

To run InfoVGAE on Voteview 105th Congress dataset:
```
python3 main.py --config_name InfoVGAE_bill_3D
```

To run the training and evaluation for the baseline of EMNLP submission, please first specify the path of the input json files in `train_intensity.py` and `train_polarity.py`, then run
```
python3 train_intensity.py
```
and
```
python3 train_polarity.py
```
for each dataset

## Dataset

Please move the json files as input of user-user graph, and user-news edges and annotations into `dataset/emnlp/*`. Then specify the paths in `train_intensity.py` and `train_polarity.py`.

## Evaluation

Evaluation will be automaticly triggered after the training process. To evaluate again, modify the `evaluator.init_from_dir()` in `evaluate.py`.

## Other arguments for training:

> General

`--use_cuda`: training with GPU

`--epochs`: iterations for training

`--learning_rate`: learning rate for training

`--device`: which gpu to use. empty for cpu.

`--num_process`: num process for pandas processing

> Data

`--data_path`: csv path for data file

`--stopword_path`: stopword path for text parsing

`--kthreshold`: tweet count threshold to filter not popular tweets.

`--uthreshold`: user count threshold to filter not popular users.

> For InfoVGAE model

`--hidden1_dim`: the latent space dimension of first layer

`--hidden2_dim`: the latent space dimension of target layer

> Result

`--output_path` path to save the result
