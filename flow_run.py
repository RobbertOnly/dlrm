import d6tflow
import luigi
import sys
import os

#Import terminal task
from flow_tasks import TaskRunDLRMExperiment

#Instantiate terminal task
test = TaskRunDLRMExperiment(debug_mode = True, data_size = 6, mini_batch_size = 2, rand_seed = 111)

#Preview terminal task
d6tflow.settings.check_dependencies=False
d6tflow.preview(test, clip_params=True)

#Run terminal task
# d6tflow.run(test)


