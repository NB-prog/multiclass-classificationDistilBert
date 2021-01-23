import transformers

DEVICE = 'cpu'
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-cased', do_lower_case = True)
