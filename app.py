import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import create_couple_model
import os

# Check if best model exists
MODEL_PATH = 'models/best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
def load_model():
    model = create_couple_model(num_classes=2, pretrained=True)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Loaded trained model.")
    else:
        print("Trained model not found. Using pretrained ImageNet weights for inference (might not work well).")
    model.to(device)
    model.eval()
    return model

model = load_model()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = ['Couple', 'Single']

def predict(img):
    if img is None:
        return None
    
    img = Image.fromarray(img)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        
    results = {classes[i]: float(probs[i]) for i in range(len(classes))}
    return results

# UI Design
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💑 Couple vs 👤 Single Classifier")
    gr.Markdown("Upload an image to identify if it contains a couple or a single person.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload Image or Take a Photo")
            btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            output_label = gr.Label(label="Prediction Result")
            
    btn.click(fn=predict, inputs=input_img, outputs=output_label)
    
    gr.Examples(
        examples=[],
        inputs=input_img
    )

if __name__ == "__main__":
    demo.launch()
