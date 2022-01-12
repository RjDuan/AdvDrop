from torch import nn
from robustness.tools.custom_modules import SequentialWithArgs

def ft(model_name, model_ft, num_classes, additional_hidden=0):
    if model_name in ["resnet", "resnet18", "resnet50", "wide_resnet50_2", "wide_resnet50_4", "resnext50_32x4d", 'shufflenet']:
        num_ftrs = model_ft.fc.in_features
        # The two cases are split just to allow loading
        # models trained prior to adding the additional_hidden argument
        # without errors
        if additional_hidden == 0:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        else:
            model_ft.fc = SequentialWithArgs(
                *list(sum([[nn.Linear(num_ftrs, num_ftrs), nn.ReLU()] for i in range(additional_hidden)], [])),
                nn.Linear(num_ftrs, num_classes)
            )
        input_size = 224
    elif model_name == "alexnet":
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif "vgg" in model_name:
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif model_name == "squeezenet":
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224
    elif model_name == "densenet":
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name in ["mnasnet", "mobilenet"]:
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        raise ValueError("Invalid model type, exiting...")

    return model_ft
