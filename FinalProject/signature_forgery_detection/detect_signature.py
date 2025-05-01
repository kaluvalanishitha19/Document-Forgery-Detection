# detect_signature.py - Detect Forged Signature Using Autoencoder
import torch
import torch.nn as nn
from preprocess import preprocess_image

class SignatureAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def detect_forgery(image_path, threshold=0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignatureAutoencoder().to(device)
    model.load_state_dict(torch.load("models/autoencoder.pth", map_location=device))
    model.eval()

    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        reconstructed = model(image_tensor)
        error = nn.MSELoss()(reconstructed, image_tensor).item()
    print(f"Reconstruction error: {error:.6f}")
    return "Genuine" if error < threshold else "Forged"

if __name__ == "__main__":
    image_path = "C:/Users/user/Desktop/project_root/signature_forgery/data/test/forged-01.png"
    result = detect_forgery(image_path)
    print("Prediction:", result)
