import torch
import torch.nn as nn

def evaluate(dataset, model):
    predictions = []
    with torch.no_grad():
        n_correct = 0
        for i, (feature, label) in enumerate(dataset):
            target = model(feature)
            prediction = torch.max(target, 1)[1] 
            predictions.append(prediction)
            if int(prediction) == label:
                n_correct += 1
        return predictions, n_correct/dataset.__len__()*100
