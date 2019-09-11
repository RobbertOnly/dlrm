import flow_tasks

def show_test_prints(params):

    dlrm = flow_tasks.TaskModelTrain(**params).output()['model'].load()
    print("\n\n\n<===========================================================>")
    print("updated parameters (weights and bias):")
    for param in dlrm.parameters():
        print(param.detach().cpu().numpy())
    print("<===========================================================>\n\n\n")

def plot_compute_graph():

    dlrm, dataset_dict = flow_tasks.TaskModelTrain().outputsLoad()
    
    from torchviz import make_dot
    V = dataset_dict['Z'].mean() if self.inference_only else dataset_dict['L']
    make_dot(V, params=dict(dlrm.named_parameters()))
    dot.render('dlrm_s_pytorch_graph') # write .pdf file

