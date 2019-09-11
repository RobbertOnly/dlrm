import d6tflow

# Import workflow tasks and output visualizations
import flow_tasks, flow_viz

# Instantiate terminal task with parameters
params = {'data_size': 6, 'mini_batch_size': 2}
task = flow_tasks.TaskModelTrain(**params)

# optional: reset everything every time workflow is run
d6tflow.invalidate_upstream(task, confirm=False)

# Preview terminal task
d6tflow.preview(task, clip_params=True)

# Run terminal task
d6tflow.run(task)

# Show output
if task.complete():
    flow_viz.show_test_prints(params)
