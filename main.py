import cv2
import numpy as np
import os
import glob


# ==========================================
# ETAP 7: WSTĘPNE PRZETWARZANIE OBRAZU
# ==========================================
def preprocess_signature(path):
    # 7.1. Konwersja do skali szarości
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None

    # 7.2. Binaryzacja Otsu z odwróceniem (podpis biały, tło czarne)
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 7.3. Redukcja szumów filtrem medianowym
    img_bin = cv2.medianBlur(img_bin, 3)

    # 7.4. Normalizacja - wykadrowanie do obszaru podpisu
    coords = cv2.findNonZero(img_bin)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img_bin[y:y + h, x:x + w]

    # Resize do stałego wymiaru (ujednolicenie danych do wektora cech)
    return cv2.resize(cropped, (300, 150), interpolation=cv2.INTER_AREA)


# ==========================================
# ETAP 8: EKSTRAKCJA CECH BIOMETRYCZNYCH
# ==========================================
def extract_biometric_features(img):
    if img is None: return None

    h, w = img.shape
    # 8.2.1. Pole powierzchni (liczba białych pikseli)
    area = cv2.countNonZero(img)
    # 8.2.2. Proporcje (szerokość / wysokość)
    aspect_ratio = w / float(h)
    # 8.2.3. Liczba konturów
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    # 8.2.4. Gęstość pikseli
    density = area / float(w * h)

    # 8.3. Budowa wektora cech
    return np.array([area, aspect_ratio, num_contours, density])


# ==========================================
# ETAP 9 & 10: MODEL I WERYFIKACJA
# ==========================================
def get_model_parameters(reference_paths):
    """Tworzy uśredniony wzorzec i parametry normalizacji"""
    ref_vectors = []
    for p in reference_paths:
        feat = extract_biometric_features(preprocess_signature(p))
        if feat is not None:
            ref_vectors.append(feat)

    if not ref_vectors: return None, None, None

    ref_matrix = np.array(ref_vectors)
    mean_val = np.mean(ref_matrix, axis=0)
    std_val = np.std(ref_matrix, axis=0)
    std_val[std_val == 0] = 1.0  # Unikanie dzielenia przez zero

    # Normalizacja wzorców i stworzenie szablonu (template)
    normalized_refs = (ref_matrix - mean_val) / std_val
    template = np.mean(normalized_refs, axis=0)

    return template, mean_val, std_val


def calculate_metrics(template, mean_v, std_v, ref_paths, target_name, threshold):
    """Oblicza FAR/FRR dla danej osoby (wymaga plików 'falsz_*.png')"""
    # FRR: Sprawdzamy jak wiele wzorców zostałoby odrzuconych przez własny model
    false_rejections = 0
    for p in ref_paths:
        feat = extract_biometric_features(preprocess_signature(p))
        dist = np.linalg.norm(((feat - mean_v) / std_v) - template)
        if dist > threshold:
            false_rejections += 1

    # FAR: Szukamy plików fałszywych (np. 'bieber_falsz1.png')
    forgery_paths = glob.glob(f"{target_name}_falsz*.png")
    false_acceptances = 0
    for p in forgery_paths:
        feat = extract_biometric_features(preprocess_signature(p))
        if feat is not None:
            dist = np.linalg.norm(((feat - mean_v) / std_v) - template)
            if dist <= threshold:
                false_acceptances += 1

    frr = (false_rejections / len(ref_paths)) * 100
    far = (false_acceptances / len(forgery_paths)) * 100 if forgery_paths else 0
    return far, frr


# ==========================================
# INTERAKTYWNA KONSOLA UŻYTKOWNIKA
# ==========================================
def run_app():
    print("==============================================")
    print("   SYSTEM BIOMETRYCZNEJ WERYFIKACJI PODPISU   ")
    print("==============================================")

    threshold = 1.2  # Stały próg decyzyjny

    while True:
        target = input("\nPodaj nazwę osoby (np. 'bieber') lub 'q' aby wyjść: ").strip().lower()
        if target == 'q': break

        # Automatyczne dopasowanie plików
        ref_paths = glob.glob(f"{target}_ref*.png")
        test_path = f"{target}_test.png"

        if not ref_paths or not os.path.exists(test_path):
            print(f"❌ Błąd: Brak plików. Wymagane: {target}_ref1.png oraz {target}_test.png")
            continue

        # 1. Budowa modelu wzorcowego
        template, mean_v, std_v = get_model_parameters(ref_paths)

        # 2. Analiza podpisu testowego
        test_feat = extract_biometric_features(preprocess_signature(test_path))
        if test_feat is not None:
            # Normalizacja i obliczenie odległości euklidesowej
            norm_test = (test_feat - mean_v) / std_v
            distance = np.linalg.norm(norm_test - template)

            # 3. Decyzja i statystyki
            print(f"\nWyniki dla: {target.upper()}")
            print(f"----------------------------------------------")
            if distance <= threshold:
                print(f"✅ STATUS: PODPIS AUTENTYCZNY")
            else:
                print(f"❌ STATUS: PODPIS FAŁSZYWY (PRÓBA OSZUSTWA)")

            print(f"📊 Odległość biometryczna: {distance:.4f}")

            # Opcjonalne FAR/FRR
            far, frr = calculate_metrics(template, mean_v, std_v, ref_paths, target, threshold)
            print(f"📈 Skuteczność lokalna: FAR: {far:.1f}% | FRR: {frr:.1f}%")
            print(f"----------------------------------------------")


if __name__ == "__main__":
    run_app()
    