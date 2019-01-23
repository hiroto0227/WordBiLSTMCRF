import torch
from torch.autograd import Variable


def get_variable(tensor, gpu=-1, **kwargs):
    if torch.cuda.is_available() and gpu > 0:
        result = Variable(tensor.cuda(gpu), **kwargs)
    else:
        result = Variable(tensor, **kwargs)
    return result


def checkpoint(model, model_path):
    print('\nmodel saved: {}'.format(model_path))
    torch.save(model.state_dict(), model_path)
