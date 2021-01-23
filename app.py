import config
import torch 

import flask
from flask import Flask 
from flask import request
from model import DistillBERTClass

app = Flask(__name__)

MODEL = None
DEVICE = 'cpu'

def predict_class(sentence, model):
    tokenizer = config.tokenizer
    max_len = config.MAX_LEN
    title = str(sentence)
    title = " ".join(title.split())
    inputs = tokenizer.encode_plus(
        title,
        None,
        add_special_tokens=True,
        max_length=max_len,
        padding=True,
        return_token_type_ids=True,
        truncation=True
    )
    ids = inputs['input_ids']
    mask = inputs['attention_mask']

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)

    
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    
    ids = ids.to(config.DEVICE, dtype = torch.long)
    mask = mask.to(config.DEVICE, dtype = torch.long)

    outputs = model(ids, mask)
    outputs = torch.nn.Softmax(dim = 1)(outputs).cpu().detach().numpy()
    return outputs


@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    prediction = predict_class(sentence, model = MODEL)
    response = {}
    response["response"] = {
        'Business': str(prediction[0][0]),
        'Science': str(prediction[0][1]),
        'Entertainment': str(prediction[0][2]),
        'Health': str(prediction[0][3]),
        'sentence': sentence
    }
    return flask.jsonify(response)

if __name__ == "__main__":
    MODEL = DistillBERTClass()
    MODEL.load_state_dict(torch.load('./pytorch_distilbert_news (1).bin'))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(debug = True)