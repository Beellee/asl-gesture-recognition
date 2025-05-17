import torch.nn as nn

class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=15):
        super(SignLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        #self.dropout = nn.Dropout(0.5)  # 50% dropout
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # output: (batch, seq_len, hidden_size)
        # hn: (num_layers, batch, hidden_size)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])  
        #out = self.dropout(hn[-1]) # aplicamos dropout y utilzamos la última capa
        return out