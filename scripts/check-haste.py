import torch

m = torch.nn.LSTM(20, 30, batch_first=True).cuda()
x = torch.randn(4, 33, 20).cuda()
x, _ = m(x)
x.mean().backward()

print("after nn.LSTM backward")


import haste_pytorch as haste

print("after imports")

gru_layer = haste.GRU(input_size=128, hidden_size=256, zoneout=0.1, dropout=0.05)
indrnn_layer = haste.IndRNN(input_size=128, hidden_size=256, zoneout=0.1)
lstm_layer = haste.LSTM(input_size=128, hidden_size=256, zoneout=0.1, dropout=0.05)
norm_gru_layer = haste.LayerNormGRU(
    input_size=128, hidden_size=256, zoneout=0.1, dropout=0.05
)
norm_lstm_layer = haste.LayerNormLSTM(
    input_size=128, hidden_size=256, zoneout=0.1, dropout=0.05
)

print("after layers")

# gru_layer.cuda()
indrnn_layer.cuda()
# lstm_layer.cuda()
# norm_gru_layer.cuda()
# norm_lstm_layer.cuda()

print("after cuda")

# `x` is a CUDA tensor with shape [T,N,C]
x = torch.rand([25, 5, 128]).cuda()

print("after x")

# y, state = lstm_layer(x)
y, state = indrnn_layer(x)
# y, state = norm_gru_layer(x)
# y, state = norm_lstm_layer(x)

# y, state = gru_layer(x)

print("after forward")

print("before backward")
y.mean().backward()
print("after backward")
