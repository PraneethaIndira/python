import torchvision.transforms as T

def get_transform(height=256, width=256):
    return T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])