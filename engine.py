import torch.nn
import config
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
loss_function = torch.nn.CrossEntropyLoss()

def train(training_loader, model, optimizer, device, epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(config.DEVICE, dtype = torch.long)
        mask = data['mask'].to(config.DEVICE, dtype = torch.long)
        targets = data['targets'].to(config.DEVICE, dtype = torch.long)

        outputs = model(ids, mask)
        # loss = loss_function(outputs, targets)
        # tr_loss += loss.item()
        # big_val, big_idx = torch.max(outputs.data, dim=1)
        # n_correct += calcuate_accu(big_idx, targets)

        # nb_tr_steps += 1
        # nb_tr_examples+=targets.size(0)
        
        # if _%5000==0:
        #     loss_step = tr_loss/nb_tr_steps
        #     accu_step = (n_correct*100)/nb_tr_examples 
        #     print(f"Training Loss per 5000 steps: {loss_step}")
        #     print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    # print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    # epoch_loss = tr_loss/nb_tr_steps
    # epoch_accu = (n_correct*100)/nb_tr_examples
    # print(f"Training Loss Epoch: {epoch_loss}")
    # print(f"Training Accuracy Epoch: {epoch_accu}")

    return

def valid(model, testing_loader, device):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0
    final_targets = []
    final_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(config.DEVICE, dtype = torch.long)
            mask = data['mask'].to(config.DEVICE, dtype = torch.long)
            targets = data['targets'].to(config.DEVICE, dtype = torch.long)
            outputs = model(ids, mask).squeeze()
            # loss = loss_function(outputs, targets)
            # tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            # n_correct += calcuate_accu(big_idx, targets)

            # nb_tr_steps += 1
            # nb_tr_examples+=targets.size(0)
            final_targets.extend(targets.cpu().detach().numpy().tolist())
            final_outputs.extend(torch.nn.Softmax(dim = 1)(outputs).cpu().detach().numpy().tolist())
            
            # if _%5000==0:
            #     loss_step = tr_loss/nb_tr_steps
            #     accu_step = (n_correct*100)/nb_tr_examples
            #     print(f"Validation Loss per 100 steps: {loss_step}")
            #     print(f"Validation Accuracy per 100 steps: {accu_step}")
    # epoch_loss = tr_loss/nb_tr_steps
    # epoch_accu = (n_correct*100)/nb_tr_examples
    # print(f"Validation Loss Epoch: {epoch_loss}")
    # print(f"Validation Accuracy Epoch: {epoch_accu}")
    
    return final_targets, final_outputs
