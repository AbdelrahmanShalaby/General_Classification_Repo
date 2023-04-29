from tqdm import tqdm
from sklearn.metrics import classification_report
import torch
import os
from save_load_model import save_model

def test(opt, test, model, loss_fn, best_loss, epoch, optimizer):
    predictions = []
    labels = []
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(test, total=len(test)):
            x = x.to(device=opt.device)
            labels += (list(y.to('cpu').numpy()))
            y = y.to(device=opt.device)
            # y = (torch.nn.functional.one_hot(y, opt.nc).float()).to(device=opt.device)

            prediction = model(x)
            target_names = ['Cat', 'Dog']
            total_loss += loss_fn(prediction, y).item()
            predictions += list(torch.argmax(prediction, dim=1).to('cpu').numpy())
        print("Total_Loss: {}".format(total_loss / len(test)))
        res = [1 if g == r else 0 for g, r in zip(labels, predictions)]
        print("Accuracy: {}".format(sum(res) / len(predictions)))
        print(classification_report(labels, predictions, target_names=target_names, zero_division=1))

        if total_loss / len(test) < best_loss:
            save_model(epoch, model, optimizer, total_loss / len(test), os.path.join(opt.save_model_path, 'best_weights.pt'))
            save_model(epoch, model, optimizer, total_loss / len(test), os.path.join(opt.save_model_path, 'last_weights.pt'))
            best_loss = total_loss / len(test)
            print("model_saved......")
        else:
            save_model(epoch, model, optimizer, total_loss / len(test), os.path.join(opt.save_model_path, 'last_weights.pt'))

        return best_loss



