import torch
from pidu.train import get_dataset, get_model

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    usecase = 2
    cnn_sizes = [0.1, 0.25, 0.5, 1.0]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Load dataset info for usecase 2
    datasets, dataset_info, device, dtype = get_dataset(
        usecase=usecase,
        trainset=None,
        valset=None,
        testset=None,
        expset=False,
        device=device,
        no_structured_noise=False,
        verbose=False
    )
    print("CNNBasic trainable parameters for usecase 2:")
    for size in cnn_sizes:
        model = get_model(
            model_name=None,
            model_type='CNN',
            ds_info=dataset_info,
            device=device,
            dtype=dtype,
            cnn_size=size,
            active_models=['system1'],
        )
        n_params = count_trainable_params(model)
        print(f"  cnn_size={size}: {n_params} trainable parameters")

if __name__ == "__main__":
    main()
