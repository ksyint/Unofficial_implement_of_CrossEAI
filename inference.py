import torch
from torchvision import transforms
from PIL import Image
from models.resnet import ResNet50
from models.grad_cam import GradCAM
from models.guided_backprop import GuidedBackprop
from models.projection_head import ProjectionHead
from utils.bounding_box import generate_bounding_box
from utils.visualization import visualize_bounding_box

def load_model(checkpoint_path, projection_checkpoint_path):
    model = ResNet50()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    projection_head = ProjectionHead(input_dim=2048, projection_dim=128)
    projection_head.load_state_dict(torch.load(projection_checkpoint_path))
    projection_head.eval()

    return model, projection_head

def process_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)

    grad_cam = GradCAM(model, target_layer_names=["layer4"])
    guided_backprop = GuidedBackprop(model)

    grad_cam_map = grad_cam(image)
    guided_backprop_map = guided_backprop(image)

    bbox = generate_bounding_box(grad_cam_map, guided_backprop_map)
    visualize_bounding_box(image[0], bbox)

if __name__ == "__main__":
    model, projection_head = load_model("best_model.pth", "best_projection_head.pth")
    process_image("path_to_image.jpg", model)
