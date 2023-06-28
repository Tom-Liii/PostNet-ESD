import torch
import scipy.ndimage as nd


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    device = labels.get_device()
    y = y.to(device)
    return y[labels].permute(0,3,1,2).float()


def rotate_img(x, deg):
    return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()
