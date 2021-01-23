import config
import dataset
import pandas as pd
import torch
import engine 
from model import DistillBERTClass
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer

def run():
    df = pd.read_csv('D:/Machine Learning/Datasets/NewsAggregatorDataset/newsCorpora.csv', sep='\t', names=['ID','TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
    # df.head()
    # # Removing unwanted columns and only leaving title of news and the category which will be the target
    df = df[['TITLE','CATEGORY']]
    # df.head()

    # # Converting the codes to appropriate categories using a dictionary
    my_dict = {
        'e':'Entertainment',
        'b':'Business',
        't':'Science',
        'm':'Health'
    }

    def update_cat(x):
        return my_dict[x]

    df['CATEGORY'] = df['CATEGORY'].apply(lambda x: update_cat(x))

    encode_dict = {}

    def encode_cat(x):
        if x not in encode_dict.keys():
            encode_dict[x]=len(encode_dict)
        return encode_dict[x]

    df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x: encode_cat(x))
    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state=200)
    test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)


    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = dataset.Triage(train_dataset)
    testing_set = dataset.Triage(test_dataset)
    
    train_params = {'batch_size': config.TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0,
                    }

    test_params = {'batch_size': config.VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    device = config.DEVICE
    model = DistillBERTClass()
    model.to(device)
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)
    for epoch in range(config.EPOCHS):
        engine.train(training_loader, model, optimizer, device, epoch)
        engine.valid(model, testing_loader, device)

    output_model_file = './models/pytorch_distilbert_news.bin'
    output_vocab_file = './models/vocab_distilbert_news.bin'

    model_to_save = model
    torch.save(model_to_save, output_model_file)
    tokenizer.save_vocabulary(output_vocab_file)

if __name__ == "__main__":
    run()
        
