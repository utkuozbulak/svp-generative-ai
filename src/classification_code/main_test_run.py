import torch
import timm


if __name__ == "__main__":
    # Model params
    num_cls = 3  # prot/silicone/air
    model_type = 'resnet18'
    model_path = '../../models/resnet18_svp_model.pth'

    # Model
    model = timm.create_model(
        model_type,
        pretrained=True,
        num_classes=num_cls)

    state_dict = torch.load(model_path, map_location='cpu')['model']
    model.load_state_dict(state_dict)
