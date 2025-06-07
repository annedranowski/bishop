# Training function(s) are model-agnostic (accept a model as argument)

def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn):
    # Paste your local (non-XLA) train_step here
    pass

def train_loop(model, data_loader, loss_fn, optimizer, accuracy_fn, num_epochs, scheduler=None):
    # Paste your train_loop here
    pass
