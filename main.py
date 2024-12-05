from model import *
from data_prep import *
from train import *
from plot import *

model = CompleteGNN()
training_data = generateTrainingData()
optimizer = optim.Adam(model.parameters(), lr=LEARING_RATE)
criterion = nn.MSELoss()

trainAndPlot(model, training_data, optimizer, criterion)

testing_data, coordinate = generateTestingData()

# generate data for plot
real_val = U(coordinate[:, 0], coordinate[:, 1])
predict_val = []
for i in range(len(testing_data)):
    predict_val.append(model(testing_data[i]))
predict_val = torch.stack(predict_val)
min_val = min(torch.min(real_val).item(), torch.min(predict_val).item())
max_val = max(torch.max(real_val).item(), torch.max(predict_val).item())
coordinate_np = coordinate.detach().numpy()
real_val_np = real_val.detach().numpy()
predict_val_np = predict_val.detach().numpy()

plotDiff(coordinate_np, real_val_np, predict_val_np)