from plot import *
from model import *
from data_prep import *

model = CompleteGNN()
checkpoint = torch.load("checkpoints/%s" %PTH_FILE_NAME, weights_only=True)
model.load_state_dict(checkpoint)

testing_data, coordinate = generateTestingData()
coordinate, real_val, predict_val = prepareForPlot(model, testing_data, coordinate)
plotDiff(coordinate, real_val, predict_val)
plot3d(coordinate, real_val, predict_val)
plot_compare_3d(coordinate, real_val, predict_val)
plot_continuous(coordinate, real_val, predict_val)