import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = 'best_model_convnext_tiny_epoch_XX_acc_XX.XX.pth'  # Eğitim sonrası doğru dosya adıyla değiştir
CONTENT_FILE = 'serpus_content.json'
COMMON_IMAGE = 'Serpuslar.jpg'
CLASS_NAMES = ['Börk', 'Fes', 'Kadın', 'Katip', 'Sarma Sarıklı']
UNKNOWN_THRESHOLD = 0.75

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_content(content_path):
    try:
        with open(content_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"İçerik dosyası bulunamadı: {content_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse hatası: {e}")
        return {}


@st.cache_resource
def load_model(model_path, num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        st.error(f"⚠️ Model dosyası bulunamadı: `{model_path}`")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Model yüklenirken hata: {e}")
        st.stop()

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    architecture = checkpoint.get('architecture', 'resnet101') if isinstance(checkpoint, dict) else 'resnet101'

    if architecture == 'convnext_tiny':
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features

        has_dropout = any('classifier.2.0.' in k for k in state_dict.keys())
        if has_dropout:
            model.classifier[2] = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, num_classes)
            )
            logger.info("ConvNeXt-Tiny mimari algılandı (Dropout + Linear)")
        else:
            model.classifier[2] = nn.Linear(in_features, num_classes)
            logger.info("ConvNeXt-Tiny mimari algılandı (Linear)")
    else:
        model = models.resnet101(weights=None)
        has_dropout_fc = any(k.startswith('fc.0.') or k.startswith('fc.1.') for k in state_dict.keys())
        if has_dropout_fc:
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, num_classes)
            )
            logger.info("ResNet-101 mimari algılandı (Dropout + Linear)")
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            logger.info("ResNet-101 mimari algılandı (Linear)")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logger.info(f"Model başarıyla yüklendi: {model_path} → {device} ({architecture})")
    return model, device, architecture


def process_image(image):
    return INFERENCE_TRANSFORM(image).unsqueeze(0)


def predict_image(model, image, class_names, device, unknown_threshold=UNKNOWN_THRESHOLD):
    image_tensor = process_image(image).to(device, non_blocking=True)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs.float(), dim=1)
        max_prob, predicted_idx = torch.max(probabilities, 1)

    confidence = max_prob.item()

    if confidence < unknown_threshold:
        return "bilinmiyor", confidence

    predicted_class = class_names[predicted_idx.item()]
    return predicted_class, confidence

st.set_page_config(
    layout="centered",
    page_title="Görsel Miraslar Serpuş",
    page_icon="🏛️"
)

num_classes = len(CLASS_NAMES)
model, device, architecture = load_model(MODEL_PATH, num_classes)
content_data = load_content(CONTENT_FILE)

arch_display = "ConvNeXt-Tiny" if architecture == "convnext_tiny" else "ResNet-101"

st.sidebar.markdown("# 🏛️ Serpuş Analizi")
st.sidebar.markdown("**BST İnovasyon**")
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model:** `{arch_display}`\n\n"
    f"**Sınıflar:** {len(CLASS_NAMES)}\n\n"
    f"**Unknown Threshold:** {UNKNOWN_THRESHOLD}"
)

st.markdown("""
    <style>
    .title {
        font-family: 'Times New Roman', Times, serif;
        color: #b88a4a;
        font-size: 40px;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .confidence-high { color: #27ae60; font-weight: bold; }
    .confidence-medium { color: #f39c12; font-weight: bold; }
    .confidence-low { color: #e74c3c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Görsel Miraslar Serpuş</h1>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Bir görüntü yükleyin",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
    help="Lütfen geçerli bir resim dosyası yükleyin (PNG, JPG, JPEG)."
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Yüklenen Görüntü", use_container_width=True)

    with st.spinner("Tahmin yapılıyor..."):
        predicted_class, confidence = predict_image(
            model, image, CLASS_NAMES, device
        )

    confidence_pct = confidence * 100
    if confidence_pct >= 85:
        conf_class = "confidence-high"
    elif confidence_pct >= 60:
        conf_class = "confidence-medium"
    else:
        conf_class = "confidence-low"

    if predicted_class == "bilinmiyor":
        st.warning(
            f"Bu görüntü tanımlanamıyor! "
            f"(Güven: %{confidence_pct:.1f} — threshold: %{UNKNOWN_THRESHOLD*100:.0f})"
        )
    else:
        st.success(f"Tahmin Edilen Sınıf: **{predicted_class}**")
        st.markdown(
            f'Güven Skoru: <span class="{conf_class}">%{confidence_pct:.1f}</span>',
            unsafe_allow_html=True
        )

        with col2:
            if predicted_class in content_data:
                class_info = content_data[predicted_class]

                st.write(class_info.get("description", ""))
                st.write("")

                class_image = class_info.get("image", "")
                if class_image:
                    try:
                        st.image(class_image, use_container_width=True)
                    except Exception:
                        st.caption(f"📷 Görsel yüklenemedi: {class_image}")

                st.write("")

                try:
                    st.image(COMMON_IMAGE, caption="Serpuş Türleri", use_container_width=True)
                except Exception:
                    st.caption(f"📷 Görsel yüklenemedi: {COMMON_IMAGE}")
            else:
                st.info(f"'{predicted_class}' sınıfı için içerik bilgisi bulunamadı.")
