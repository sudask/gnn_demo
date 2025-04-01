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

coordinate, real_val, predict_val = prepareForPlot(model, testing_data, coordinate)

plotDiff(coordinate, real_val, predict_val)
plot3d(coordinate, real_val, predict_val)
plot_compare_3d(coordinate, real_val, predict_val)