import pandas as pd
import torch
from functions import seed_torch
from zu_mai.main import MyDataset


def get_TP(row):
    if row['Model Predictions'] == 1 and row['Node Labels'] == 1:
        return True
    else:
        return False

def get_FP(row):
    if row['Model Predictions'] == 1 and row['Node Labels'] == 0:
        return True
    else:
        return False

if __name__=="__main__":
    seed_torch()
    model_path = "model_checkpoint_epoch_300.pt" # Best model path
    model = torch.load(model_path)
    model.eval()

    dataset = MyDataset()
    graph = dataset[0]

    features = graph.ndata["feat"]
    labels = graph.ndata["label"]
    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]

    with torch.no_grad():
        output = model(graph, features)
        pred = output.argmax(1)
        pred = pred.cpu().numpy()

    test_indices = torch.nonzero(test_mask).squeeze().cpu().numpy()
    test_features = features[test_indices].cpu().numpy()
    test_labels = labels[test_indices].cpu().numpy()
    test_pred = pred[test_indices]

    df_test = pd.DataFrame({
        "Node Index": test_indices.tolist(),
        "Node Features": test_features.tolist(),
        "Node Labels": test_labels.tolist(),
        "Model Predictions": test_pred.tolist()
    })

    df_test_FP = df_test[df_test.apply(lambda row: get_FP(row), axis=1)]
    df_tmp_1 = df_test['Node Features'].apply(lambda x: eval(x)).apply(pd.Series)
    df_tmp_1.columns = pd.read_csv('data.csv').columns[:-1].tolist()
    df_tmp_1['LABEL'] = 'Ture Positive'

    df_tmp_2 = df_test['Node Features'].apply(lambda x: eval(x)).apply(pd.Series)
    df_tmp_2.columns = pd.read_csv('data.csv').columns[:-1].tolist()
    df_tmp_2['LABEL'] = 'False Positive'
    df = pd.concat([df_tmp_1, df_tmp_2], axis=0)

    csv_filename = "data_pred.csv"
    df.to_csv(csv_filename, index=False)
    print("Graph data and predictions saved to CSV:", csv_filename)



