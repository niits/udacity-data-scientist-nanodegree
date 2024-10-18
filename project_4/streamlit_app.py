import torch
import cv2
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import streamlit as st
from io import StringIO
from pathlib import Path
import numpy as np

dog_breeds = [
    "1.Affenpinscher",
    "2.Afghan_hound",
    "3.Airedale_terrier",
    "4.Akita",
    "5.Alaskan_malamute",
    "6.American_eskimo_dog",
    "7.American_foxhound",
    "8.American_staffordshire_terrier",
    "9.American_water_spaniel",
    "0.Anatolian_shepherd_dog",
    "1.Australian_cattle_dog",
    "2.Australian_shepherd",
    "3.Australian_terrier",
    "4.Basenji",
    "5.Basset_hound",
    "6.Beagle",
    "7.Bearded_collie",
    "8.Beauceron",
    "9.Bedlington_terrier",
    "0.Belgian_malinois",
    "1.Belgian_sheepdog",
    "2.Belgian_tervuren",
    "3.Bernese_mountain_dog",
    "4.Bichon_frise",
    "5.Black_and_tan_coonhound",
    "6.Black_russian_terrier",
    "7.Bloodhound",
    "8.Bluetick_coonhound",
    "9.Border_collie",
    "0.Border_terrier",
    "1.Borzoi",
    "2.Boston_terrier",
    "3.Bouvier_des_flandres",
    "4.Boxer",
    "5.Boykin_spaniel",
    "6.Briard",
    "7.Brittany",
    "8.Brussels_griffon",
    "9.Bull_terrier",
    "0.Bulldog",
    "1.Bullmastiff",
    "2.Cairn_terrier",
    "3.Canaan_dog",
    "4.Cane_corso",
    "5.Cardigan_welsh_corgi",
    "6.Cavalier_king_charles_spaniel",
    "7.Chesapeake_bay_retriever",
    "8.Chihuahua",
    "9.Chinese_crested",
    "0.Chinese_shar-pei",
    "1.Chow_chow",
    "2.Clumber_spaniel",
    "3.Cocker_spaniel",
    "4.Collie",
    "5.Curly-coated_retriever",
    "6.Dachshund",
    "7.Dalmatian",
    "8.Dandie_dinmont_terrier",
    "9.Doberman_pinscher",
    "0.Dogue_de_bordeaux",
    "1.English_cocker_spaniel",
    "2.English_setter",
    "3.English_springer_spaniel",
    "4.English_toy_spaniel",
    "5.Entlebucher_mountain_dog",
    "6.Field_spaniel",
    "7.Finnish_spitz",
    "8.Flat-coated_retriever",
    "9.French_bulldog",
    "0.German_pinscher",
    "1.German_shepherd_dog",
    "2.German_shorthaired_pointer",
    "3.German_wirehaired_pointer",
    "4.Giant_schnauzer",
    "5.Glen_of_imaal_terrier",
    "6.Golden_retriever",
    "7.Gordon_setter",
    "8.Great_dane",
    "9.Great_pyrenees",
    "0.Greater_swiss_mountain_dog",
    "1.Greyhound",
    "2.Havanese",
    "3.Ibizan_hound",
    "4.Icelandic_sheepdog",
    "5.Irish_red_and_white_setter",
    "6.Irish_setter",
    "7.Irish_terrier",
    "8.Irish_water_spaniel",
    "9.Irish_wolfhound",
    "0.Italian_greyhound",
    "1.Japanese_chin",
    "2.Keeshond",
    "3.Kerry_blue_terrier",
    "4.Komondor",
    "5.Kuvasz",
    "6.Labrador_retriever",
    "7.Lakeland_terrier",
    "8.Leonberger",
    "9.Lhasa_apso",
    "0.Lowchen",
    "1.Maltese",
    "2.Manchester_terrier",
    "3.Mastiff",
    "4.Miniature_schnauzer",
    "5.Neapolitan_mastiff",
    "6.Newfoundland",
    "7.Norfolk_terrier",
    "8.Norwegian_buhund",
    "9.Norwegian_elkhound",
    "0.Norwegian_lundehund",
    "1.Norwich_terrier",
    "2.Nova_scotia_duck_tolling_retriever",
    "3.Old_english_sheepdog",
    "4.Otterhound",
    "5.Papillon",
    "6.Parson_russell_terrier",
    "7.Pekingese",
    "8.Pembroke_welsh_corgi",
    "9.Petit_basset_griffon_vendeen",
    "0.Pharaoh_hound",
    "1.Plott",
    "2.Pointer",
    "3.Pomeranian",
    "4.Poodle",
    "5.Portuguese_water_dog",
    "6.Saint_bernard",
    "7.Silky_terrier",
    "8.Smooth_fox_terrier",
    "9.Tibetan_mastiff",
    "0.Welsh_springer_spaniel",
    "1.Wirehaired_pointing_griffon",
    "2.Xoloitzcuintli",
    "3.Yorkshire_terrier",
]

current_dir = Path(__file__).resolve().parent

from torchvision.models import resnet50, ResNet50_Weights

ResNet50_model = resnet50(weights=ResNet50_Weights.DEFAULT)

face_cascade = cv2.CascadeClassifier(
    (current_dir / "haarcascades/haarcascade_frontalface_alt.xml").as_posix()
)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess_input(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(Image.fromarray(img)).unsqueeze(0)


def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def dog_detector(img):
    tensor = preprocess_input(img)
    with torch.no_grad():
        predicted_ids = ResNet50_model(tensor).argmax(dim=1).numpy()

    return predicted_ids[0] >= 151 and predicted_ids[0] <= 268


class Resnet50Model(nn.Module):
    def __init__(self):
        super(Resnet50Model, self).__init__()
        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet50.fc = nn.Linear(2048, 133)
        for param in self.resnet50.parameters():
            param.requires_grad = False

        for param in self.resnet50.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet50(x)


@st.cache_resource()
def load_model():

    transfer_model = Resnet50Model()

    transfer_model.load_state_dict(
        torch.load(
            (current_dir / "transfer_model.pth").as_posix(),
            weights_only=True,
            map_location="cpu",
        )
    )

    transfer_model.eval()

    device = torch.device("cpu")

    transfer_model.to(device)
    return transfer_model, device


st.title("Dog Breed Classifier")

with st.spinner("Loading Model..."):
    transfer_model, device = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img = np.array(image)

    if face_detector(img):
        st.write("Human detected in the image")
    elif dog_detector(img):
        st.write("Dog detected in the image")
        tensor = preprocess_input(img)
        tensor = tensor.to(device)
        with torch.no_grad():
            output = transfer_model(tensor)
            predicted_class = torch.argmax(output).item()

        st.write(f"Predicted Dog Breed: {dog_breeds[predicted_class]}")
    else:
        st.write("No dog or human detected in the image")
