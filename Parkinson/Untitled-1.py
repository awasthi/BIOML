# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class DEML(nn.Module):
    def __init__(self, num_genes, l1_lambda=0.01):
        super(DEML, self).__init__()
        self.linear = nn.Linear(num_genes, 1)
        self.l1_lambda = l1_lambda

    def forward(self, x):
        logits = self.linear(x)
        probs = torch.sigmoid(logits)
        return probs

    def l1_regularization(self):
        return torch.norm(self.linear.weight, p=1)

    def compute_loss(self, y_pred, y_true):
        bce_loss = F.binary_cross_entropy(y_pred, y_true)
        l1_loss = self.l1_lambda * self.l1_regularization()
        return bce_loss + l1_loss

    def get_gene_importance(self):
        return self.linear.weight.detach().cpu().numpy().flatten()


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load data
#mydata = 'GSE64810'
mydata = 'combined'
data_file = './data/'+ mydata +'.csv'
degs_file = './data/' + mydata + '_DEGS.txt'
out_file = './results/fs_scores64810_LSVC.csv'

# Read dataset
#df = pd.read_csv(data_file)
df = pd.read_csv(data_file, header=None)  # Read without assuming column headers
df = df.set_index(0).T  # Set first column as header and transpose
df.columns.name = None  # Remove index name if needed
df = df.iloc[:, 1:]  
df = df.iloc[:, 1:].reset_index(drop=True)
X = df.drop(columns=['disease'])   # Assuming 'disease' is the label
X.columns = X.columns.astype(str)
y = df['disease'].astype(int)

# %%
model = DEML(num_genes=X.shape[1], l1_lambda=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Ensure all values in X_df are float32
X = X.astype(np.float32)
# Ensure y_df is numeric and has proper shape
y = y.astype(np.float32)

X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)

for epoch in range(200):
    model.train()
    y_pred = model(X_tensor)
    loss = model.compute_loss(y_pred, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# %%
gene_importance = model.get_gene_importance()
top_genes = sorted(zip(X.columns, gene_importance), key=lambda x: abs(x[1]), reverse=True)
for gene, score in top_genes[:50]:
    print(f"{gene}: {score:.4f}")



