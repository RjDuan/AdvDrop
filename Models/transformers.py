import timm

def diet_tiny():
    model = timm.create_model("deit_tiny_patch16_224", pretrained=True)
    return model
def diet_small():
    model = timm.create_model("deit_small_patch16_224", pretrained=True)
    return model

def vit_tiny():
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    return model

def vit_small():
    model = timm.create_model('vit_small_patch16_224', pretrained=True)
    return model

