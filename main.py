import cv2
import numpy as np
import os
import glob



# PRZETWARZANIE WSTĘPNE

def preprocess_signature(path):
    """Przygotowuje obraz do ekstrakcji cech."""
    # w skali szarości
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None

    # Binaryzacja
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Redukcja szumów
    img_bin = cv2.medianBlur(img_bin, 3)

    # Normalizacja do obszaru podpisu - Bounding Box
    coords = cv2.findNonZero(img_bin)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img_bin[y:y + h, x:x + w]

    return cropped



# EKSTRAKCJA CECH

def extract_biometric_features(img):
    """Ekstrahuje cechy zdefiniowane w specyfikacji."""
    if img is None:
        return None

    h, w = img.shape

    # Pole powierzchni podpisu - liczba pikseli należących do podpisu (białych)
    area = cv2.countNonZero(img)

    # Proporcje - stosunek szerokości do wysokości
    aspect_ratio = w / float(h) if h != 0 else 0

    # Liczba konturów - liczba odrębnych fragmentów podpisu
    # RETR_EXTERNAL znajduje tylko zewnętrzne obrysy (pomija dziury wewnątrz liter)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)

    # Gęstość pikseli - pole powierzchni podzielone przez rozmiar prostokąta
    density = area / float(w * h) if (w * h) != 0 else 0

    # Zwracamy wektor cech
    return np.array([area, aspect_ratio, num_contours, density])



# MODEL I ANALIZA

def get_model_parameters(reference_paths):
    """Tworzy uśredniony wzorzec i parametry normalizacji."""
    ref_vectors = []
    for p in reference_paths:
        processed = preprocess_signature(p)
        feat = extract_biometric_features(processed)
        if feat is not None:
            ref_vectors.append(feat)

    if not ref_vectors:
        return None, None, None

    ref_matrix = np.array(ref_vectors)
    mean_val = np.mean(ref_matrix, axis=0)
    std_val = np.std(ref_matrix, axis=0)

    # Zabezpieczenie przed dzieleniem przez zero
    std_val[std_val == 0] = 1.0

    # Normalizacja wzorców i stworzenie szablonu (uśredniony podpis)
    normalized_refs = (ref_matrix - mean_val) / std_val
    template = np.mean(normalized_refs, axis=0)

    return template, mean_val, std_val


def calculate_metrics(template, mean_v, std_v, ref_paths, target_name, threshold):
    """Oblicza FAR (False Acceptance Rate) i FRR (False Rejection Rate)."""
    # FRR: Fałszywe odrzucenia (własne podpisy odrzucone)
    false_rejections = 0
    for p in ref_paths:
        processed = preprocess_signature(p)
        feat = extract_biometric_features(processed)
        dist = np.linalg.norm(((feat - mean_v) / std_v) - template)
        if dist > threshold:
            false_rejections += 1

    # FAR: Fałszywe akceptacje (podrobione podpisy zaakceptowane)
    forgery_paths = glob.glob(f"{target_name}_falsz*.png")
    false_acceptances = 0
    for p in forgery_paths:
        processed = preprocess_signature(p)
        feat = extract_biometric_features(processed)
        if feat is not None:
            dist = np.linalg.norm(((feat - mean_v) / std_v) - template)
            if dist <= threshold:
                false_acceptances += 1

    frr = (false_rejections / len(ref_paths)) * 100
    far = (false_acceptances / len(forgery_paths)) * 100 if forgery_paths else 0
    return far, frr



def run_app():
    print("----------------------------------------------")
    print("   SYSTEM BIOMETRYCZNEJ WERYFIKACJI PODPISU   ")
    print("----------------------------------------------")

    # Ustalanie progu na starcie
    while True:
        try:
            t = input("\nPodaj prog decyzyjny (ENTER = 1.2): ").strip()
            threshold = float(t) if t else 1.2
            break
        except ValueError:
            print("Blad: prog musi byc liczba.")

    while True:
        target = input(
            "\nPodaj nazwe osoby | 't' - zmien prog | 'x' - wyjscie: "
        ).strip().lower()

        if target == 'x':
            break

        if target == 't':
            try:
                t = input("Nowy prog decyzyjny: ").strip()
                threshold = float(t)
                print(f"Ustawiono nowy prog: {threshold}")
            except ValueError:
                print("Blad: prog musi byc liczba.")
            continue

        ref_paths = glob.glob(f"{target}_ref*.png")
        test_path = f"{target}_test.png"

        if not ref_paths or not os.path.exists(test_path):
            print(f"Blad: Brak plikow. Wymagane: {target}_ref1.png oraz {target}_test.png")
            continue

        template, mean_v, std_v = get_model_parameters(ref_paths)

        test_processed = preprocess_signature(test_path)
        test_feat = extract_biometric_features(test_processed)

        if test_feat is not None:
            norm_test = (test_feat - mean_v) / std_v
            distance = np.linalg.norm(norm_test - template)

            print(f"\nWyniki dla: {target.upper()}")
            print("-" * 46)

            if distance <= threshold:
                print("PODPIS AUTENTYCZNY")
            else:
                print("PODPIS FALSZYWY")

            print(f"Odleglosc biometryczna: {distance:.4f}")
            print(f"Prog decyzyjny: {threshold}")

            print(f"   - Pole: {test_feat[0]:.0f} px")
            print(f"   - Proporcje: {test_feat[1]:.2f}")
            print(f"   - Kontury: {test_feat[2]:.0f}")
            print(f"   - Gestosc: {test_feat[3]:.4f}")

            far, frr = calculate_metrics(
                template, mean_v, std_v, ref_paths, target, threshold
            )

            print(f"Skutecznosc: FAR: {far:.1f}% | FRR: {frr:.1f}%")
            print("-" * 46)



if __name__ == "__main__":
    run_app()
    
