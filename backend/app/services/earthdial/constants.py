# --------------------------------------------------------
# EarthDial Constants
# --------------------------------------------------------

# --- Image / Bounding Box Tokens ---
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"

QUAD_START_TOKEN = "<quad>"
QUAD_END_TOKEN = "</quad>"
REF_START_TOKEN = "<ref>"
REF_END_TOKEN = "</ref>"
BOX_START_TOKEN = "<box>"
BOX_END_TOKEN = "</box>"
MB_TOKEN_START = "<mb>"
MB_TOKEN_END = "</mb>"

# --- Task Tokens (Crucial for Inference) ---
# Use these prefixes to trigger specific model behaviors
GROUNDING = "[grounding]"
REFER = "[refer]"
IDENTIFY = "[identify]"      # Often used for VQA/Classification
CLASSIFY = "[classify]"      # Used for Scene Classification
CHANGEDET = "[changedet]"    # Change Detection
CAPTION = "[caption]"        # Image Captioning
TREECLASSIFY = "[treeclassify]"
UHI = "[uhi]"                # Urban Heat Island tasks

# --- Modality / Resolution Tokens ---
# These indicate the sensor type and resolution
HIGH_RGB_05_TOKEN = "[hr_rgb_0.5]"       # High-Res RGB (0.5m)
HIGH_RGB_05_TEMP_TOKEN = "[hr_rgb_temp_0.5]" # Temporal High-Res RGB
HIGH_RGBI_05 = "[hr_rgbi_0.5]"           # High-Res RGB + Infrared

S2_RGB_10_TOKEN = "[s2_rgb_10]"          # Sentinel-2 RGB (10m)
S2_MS_10_TOKEN = "[s2_ms_10]"            # Sentinel-2 Multi-Spectral
L8_RGB_30_TOKEN = "[l8_rgb_30]"          # Landsat-8 RGB (30m)
L8_MS_30 = "[l8_ms_30]"                  # Landsat-8 Multi-Spectral
HYPER_RGB_3 = "[hyper_rgb_3]"            # Hyperspectral

S1_VH_10_TOKEN = "[s1_vh_10]"            # Sentinel-1 SAR VH (10m)
S1_VH_1_TOKEN = "[s1_vh_1]"              # Sentinel-1 SAR (1m)
S1_VH_TEMP_10 = "[s1_vh_temp_10]"        # Temporal SAR

# --- Normalization Statistics ---
# Standard ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CLIP / SigLIP
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

# Sentinel-2 (S2) - SSL4EO
S2_MEAN = (756.4, 889.6, 1151.7, 1307.6, 1637.6, 2212.6, 2442.0, 2538.9, 2602.9, 2666.8, 2388.8, 1821.5)
S2_STD = (1111.4, 1159.1, 1188.1, 1375.2, 1376.6, 1358.6, 1418.4, 1476.4, 1439.9, 1582.1, 1460.7, 1352.2)

# Sentinel-1 (S1)
S1_MEAN = (-20.26,)
S1_STD = (5.91,)

# RGB + Infrared (RGBI)
rgbi_mean = (0,)
rgbi_std = (255,)

# Landsat-8 (L8)
L8_MEAN = (1685.92, 2576.61, 3412.41, 4061.92, 4908.37, 4252.26, 4252.26, 4252.26)
L8_STD = (359.73, 418.77, 568.26, 626.04, 978.89, 866.68, 866.68, 866.68)
REF_START_TOKEN = "<ref>"
REF_END_TOKEN = "</ref>"
