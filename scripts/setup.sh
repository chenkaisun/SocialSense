#mkdir data
#mkdir data/response_pred
#mkdir data/response_pred/labeler_data

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m" -O BARTScore/bart.pth && rm -rf /tmp/cookies.txt

#pip install https://pytorch-geometric.com/whl/torch-1.9.0+cu111/torch_scatter-2.0.7-cp36-cp36m-win_amd64.whl
#pip install torch_scatter==2.0.7 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
#pip install torch-geometric==2.0.3 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html

## torch
chmod +x scripts/*
python -m pip install -r requirements.txt
python -m spacy download en
python -c "import nltk; nltk.download('stopwords', quiet=True)"
#python -c "import wandb; wandb.login(key='')"
