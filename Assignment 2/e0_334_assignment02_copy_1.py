# -*- coding: utf-8 -*-

import pandas as pd
import torch
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

print(device)

import spacy
nlp = spacy.load('en_core_web_sm')

emb_model = {}
f = open('../glove.6B.300d.txt', encoding='utf-8')
for line in f:
  #print(line)
  values = line.split()
  try:
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    emb_model[word.lower()] = coefs    
  except:
    pass
f.close()

BATCH_SIZE=64
SPLIT_RATIO = 0.8

def seq_to_vec(seq):
    #seq = unicode(seq, "utf-8")    
    doc = nlp(re.sub('<[^<]+?>', '', seq))
    ls = [t.text for t in doc if (t.is_alpha==True and t.is_punct==False and t.like_url==False)]
    #print(ls)
    #print(len(ls))
    vec=[]
    for word in ls:        
        try:
            vec.append(emb_model[word])
        except:
            pass    
    return vec

def label_str_to_num(label):    
    if(label.lower()=='positive'):
        return 1
    else:
        return 0

class CustomTextDataset(Dataset):
    def __init__(self, text, labels):
        self.labels = labels
        self.text = text
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.text[idx]
        sample = {"Text": text, "Class": label}
        return sample

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for sample in batch:        
        label_list.append(label_str_to_num(sample['Class']))
        processed_text = torch.tensor(seq_to_vec(sample['Text']), dtype=torch.float32)        
        #label_list.append(sample['Class'])
        #processed_text = torch.tensor(sample['Text'], dtype=torch.float32)
        
        text_list.append(processed_text)        
        offsets.append(processed_text.size(0))
        
    label_list = torch.tensor(label_list, dtype=torch.float32)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    
    return label_list, text_list, offsets

data_csv_str = "Train dataset.csv"
data_df = pd.read_csv(data_csv_str)

DS = CustomTextDataset(data_df['review'], data_df['sentiment'])

from torch.utils.data import random_split

train_size = int(SPLIT_RATIO*len(DS))
test_size = len(DS) - train_size

train_dataset,val_dataset = random_split(DS, [train_size,test_size],generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True, collate_fn=collate_batch)
val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False, collate_fn=collate_batch)

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)        
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        s=[]
        print(x[0].size())
        for i in range(0,x.size(0)):
          if(~self.check_eos(x[i])):
            _,(h,_) = self.lstm(x[i])
            s.append(h)
          else:
            self.lstm

        out = self.sigmoid(self.fc(h))
        return out

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru1 = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)        
        self.fc2 = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
      s=[]
      _,h = self.gru1(x[:,0,:])        
      s.append(h)
      for i in range(1,x.size(1)):
        _,h = self.gru1(x[:,i,:],h)         
        s.append(h)
      
      x = torch.cat(s)
      #print(x.size())
      _,h = self.gru2(x.unsqueeze(0))
      x = self.sigmoid(self.fc1(h))
      out = self.sigmoid(self.fc2(x))
      
      return out

import time
import datetime

def evaluate(model,criterion, val_loader):
    model.eval()
    total_acc, total_count = 0.0, 0
    total_loss = 0
    with torch.no_grad():
        for _, (label, text, offset) in enumerate(val_loader):            
            num_Samples = offset.size(0) 

            out=torch.zeros(BATCH_SIZE,1)            
            for idx in range(0,num_Samples):           
              if(idx==num_Samples-1):
                  x = text[offset[idx]:-1]
              else:
                  x = text[offset[idx]:offset[idx+1]]

              if(x.size(0) != 0):
                  x = x.unsqueeze(0)
                  out[idx] = model(x.to(device))
              else:
                  out[idx] = 0.01
            out = out.squeeze()
            loss = criterion(out.to(device),label.to(device).float())       
            
            total_loss +=loss
            total_acc += (out.round() == label).sum().item()
            
            total_count += label.size(0)

    return total_acc/total_count, total_loss/total_count

def train(train_loader, val_loader=None, learn_rate=0.01, hidden_dim=256, EPOCHS=5):
    
    # Setting common hyperparameters
    input_dim = 300
    output_dim = 1
    n_layers = 1

    print('LR={}, hidden_dim={},total_epochs={}'.format(learn_rate,hidden_dim,EPOCHS))

    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)    
    print(model)
    model.to(device)
    
    # Defining loss function and optimizer
    criterion = nn.BCELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    print('-' * 59)
    print("Starting Training of model")
    epoch_times = []
    # Start training loop
    for epoch in range(1,EPOCHS+1):
        start_time = time.process_time()        
        total_loss = 0.
        total_acc = 0.
        counter = 0        
        for _,(label, text, offset) in enumerate(train_loader):
            model.train()
            counter += 1
            model.zero_grad()              
            num_Samples = offset.size(0) 

            out=torch.zeros(BATCH_SIZE,1)            
            for idx in range(0,num_Samples):           
              if(idx==num_Samples-1):
                  x = text[offset[idx]:-1]
              else:
                  x = text[offset[idx]:offset[idx+1]]
                             
              if(x.size(0) != 0):                  
                  x = x.unsqueeze(0)
                  out[idx] = model(x.to(device))
              else:
                  out[idx] = 0.01
            
            out= out.squeeze()                
            loss = criterion(out.to(device),label.to(device).float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_acc += (out.round() == label).sum().item()
                        
            if counter%50 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss={:9.4}".format(epoch, counter, len(train_loader), 1.0*total_loss/(counter*BATCH_SIZE)))
            

        current_time = time.process_time()
        print("Epoch {}/{} Done, Average Train Loss={:9.4}, Train Accuracy={:9.4}".format(epoch, EPOCHS, 1.0*total_loss/(BATCH_SIZE*len(train_loader)),100.0*total_acc/(BATCH_SIZE*len(train_loader))))

        torch.save(model, 'MODEL_E' + str(epoch) + datetime.date.today().strftime("%B %d, %Y") + '.pth')
        if val_loader is not None:
          acc_val,loss_val = evaluate(model, criterion, val_loader)
          print("Epoch {}/{} Done, Average Val Loss={:9.4}, Average Val Accuracy={:9.4}".format(epoch, EPOCHS, 1.0*loss_val,100.0*acc_val))

        print("Total Time Elapsed={:9.3} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
        print('-' * 59)

    print("Total Training Time={:9.3} seconds".format(str(sum(epoch_times))))

    return model

GRU_Model = train(train_dataloader, val_dataloader, 0.001,512,10)