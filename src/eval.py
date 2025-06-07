# Evaluation functions are model-agnostic

def test_step(model, data_loader, loss_fn, accuracy_fn, scheduler=None):
    # Paste your test_step here
    pass

def eval_step(model, data_loader, loss_fn, accuracy_fn):
    # Paste your eval_step here
    pass

def eval_loop(model, data_loader, loss_fn, optimizer, accuracy_fn, num_epochs=None, scheduler=None):
    # Paste your eval_loop here
    pass
