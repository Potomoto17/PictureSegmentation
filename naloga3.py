import cv2 as cv
import numpy as np
from typing import Tuple, List

############################
#  POMOŽNE FUNKCIJE        #
############################

def _prepare_feature_space(slika: np.ndarray, dimenzija: int) -> np.ndarray:
    """Pretvori sliko v prostor značilnic dimenzije 3 ali 5.

    Če je dimenzija 3 → (R,G,B)
    Če je dimenzija 5 → (R,G,B,x,y)  (x,y sta normalizirani koordinati v [0,255])
    """

    if dimenzija not in (3, 5):
        raise ValueError("dimenzija mora biti 3 ali 5")

    h, w = slika.shape[:2]
    rgb = slika.reshape((-1, 3)).astype(np.float32) # Pretvori v seznam rgb vektorjev, slika.reshape naredi 2d matriko vsaka vrstica je ena rgb komponenta, astype pretvori v decimalne vrednosti
    if dimenzija == 3:
        return rgb

    xs, ys = np.meshgrid(np.arange(w), np.arange(h)) # Ustvari mrežo koordinat za vsako slikovno piko, xs, xs sta 2 matriki z vrednostmi x in y koordinat
    coords = np.stack([xs, ys], axis=-1).reshape((-1, 2)).astype(np.float32) # Preoblikuje v xs, ys v seznam (x,y) koordinat
    print(coords)
    coords *= 255.0 / max(h, w)  # normalizira koordinate na 0‑255
    return np.concatenate([rgb, coords], axis=1) # Vrne RGB in x, y vektorje


def izracunaj_centre(slika: np.ndarray, izbira: str, dimenzija_centra: int, T: float, k: int = 3) -> np.ndarray:
    """Izbere začetne centre za k‑means (naključno ali ročno)."""
    feat = _prepare_feature_space(slika, dimenzija_centra) # Pretvori sliko v prostor slikovnih značilnic (glej zgora za postopek)
    h, w = slika.shape[:2]

    if izbira.lower().startswith("nak"):
        # Naključno izbere piko in jo sprejme le če je oddaljena T daleč od že izbranih, če se to ne zgodi v k*50 poskusih manjkajočo doda brez pogoja
        centres: List[np.ndarray] = []
        attempts, max_attempts = 0, k * 50 
        while len(centres) < k and attempts < max_attempts:
            cand = feat[np.random.randint(0, feat.shape[0])]
            if all(np.linalg.norm(cand - c) >= T for c in centres):
                centres.append(cand)
            attempts += 1
        if len(centres) < k:
            extra = feat[np.random.choice(feat.shape[0], k - len(centres), replace=False)]
            centres.extend(extra)
        # Rezultat je k x dimenzija_centra matrika z začetnimi centri.
        return np.vstack(centres)

    elif izbira.lower().startswith("ro"):
        clone = slika.copy()
        points: List[Tuple[int, int]] = []

        def _click(ev, x, y, *_):
            if ev == cv.EVENT_LBUTTONDOWN and len(points) < k:
                points.append((x, y))
                cv.circle(clone, (x, y), 5, (0, 0, 255), -1)
                cv.imshow("Izberi centre", clone)
        # Pokaže sliko, uporabnik klikne in izbere centre, pokaže se rdeči krogec, Pretvori izbrane (x, y) v indekse v prostoru značilnic in jih vrne
        cv.imshow("Izberi centre", clone)
        cv.setMouseCallback("Izberi centre", _click)
        while len(points) < k:
            if cv.waitKey(1) & 0xFF == 27:  # Esc
                break
        cv.destroyAllWindows()
        if len(points) < k:
            raise RuntimeError("Premalo točk izbranih.")
        idxs = [y * w + x for x, y in points]
        return feat[idxs]
        # Vrne np.ndarray oblike (k, dimenzija_centra) — začetne točke za K-means.

    else:
        raise ValueError("izbira mora biti 'nakljucna' ali 'rocno'")

############################
#  K‑MEANS                 #
############################

def kmeans(slika: np.ndarray, k: int = 3, iteracije: int = 10, dimenzija: int = 5, T: float = 30) -> np.ndarray:
    h, w = slika.shape[:2]
    X = _prepare_feature_space(slika, dimenzija)    # Pretvori sliko v prostor slikovnih značilnic (glej zgora za postopek)
    centri = izracunaj_centre(slika, "nakljucna", dimenzija, T, k) # izracuna centre rocno ali nakljucno

    # izracuna razdalje od vsake pike do vsakega centra, vsaki piki določi najbližji center
    for _ in range(iteracije):  
        dists = np.linalg.norm(X[:, None, :] - centri[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        # Za vsako skupino izračuna povprečno točko (nov center) če je katera skupina prazna, ohrani obstoječi center
        novi_centri = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centri[i]
            for i in range(k)
        ])
        # Preveri konvergenco, če se centri bistveno spremenijo konča predčasno
        if np.allclose(novi_centri, centri):
            break
        centri = novi_centri

    # Zgradi končnp sliko, za vsako piko uporabi barvo svojega centra
    barve = centri[:, :3].astype(np.uint8)
    return barve[labels].reshape((h, w, 3))

############################
#  MEAN‑SHIFT              #
############################

def meanshift(slika: np.ndarray, h: float = 25, dimenzija: int = 5, max_iter: int = 10, min_cd: float = 20) -> np.ndarray:
    """Mean‑Shift segmentacija na sliki, ki jo najprej pomanjšamo na 200×200.

    Manjša slika dramatično pohitri računanje, pri večjih vhodih pa še vedno
    dobimo reprezentativen rezultat.
    """
    # standardiziraj velikost zaradi hitrosti
    slika = cv.resize(slika, (30, 30), interpolation=cv.INTER_AREA)

    h_img, w_img = slika.shape[:2]
    X = _prepare_feature_space(slika, dimenzija)
    N = X.shape[0]
    pos = X.copy()

    # Izračuna razdaljo do vseh drugih točk. Poišče tiste v razdalji < h → to je njeno lokalno "okno". Premakne točko proti središču mase tega okna (če se je dovolj premaknila).
    for _ in range(max_iter):
        for i in range(N):
            d = np.linalg.norm(X - pos[i], axis=1)
            okno = X[d < h]
            if okno.size == 0:
                continue
            nova = okno.mean(axis=0)
            if np.linalg.norm(nova - pos[i]) > 1e-3:
                pos[i] = nova

    # Primerja končne pozicije točk, če je točka blizu znanega moda (min_cd) dobi isti label
    modes, labels = [], -np.ones(N, dtype=int)
    for i, p in enumerate(pos):
        for idx, m in enumerate(modes):
            if np.linalg.norm(p - m) < min_cd:
                labels[i] = idx
                break
        if labels[i] == -1:
            modes.append(p)
            labels[i] = len(modes) - 1
    
    print(labels)
    modes = np.array(modes)
    barve = modes[:, :3].astype(np.uint8)
    return barve[labels].reshape((h_img, w_img, 3))

############################
#  MAIN                    #
############################

if __name__ == "__main__":
    import argparse, sys, os

    parser = argparse.ArgumentParser("K‑Means in Mean‑Shift segmentacija")
    parser.add_argument("--slika", required=True, help="Pot do vhodne slike")
    parser.add_argument("--alg", choices=["kmeans", "meanshift"], default="kmeans")
    parser.add_argument("--k", type=int, default=3, help="Število centrov (k‑means)")
    parser.add_argument("--dim", type=int, default=5, help="Dimenzija prostora (3 ali 5)")
    parser.add_argument("--iter", type=int, default=10, help="Število iteracij")

    # dodatni parametri
    parser.add_argument("--T", type=float, default=30, help="Minimalna razdalja med centri (k‑means)")
    parser.add_argument("--h", type=float, default=25, help="Pas (bandwidth) za Mean‑Shift")
    parser.add_argument("--min_cd", type=float, default=20, help="Prag združevanja mod (Mean‑Shift)")
    parser.add_argument("--out", default="out.png", help="Ime/relativna pot izhodne slike")

    args = parser.parse_args()

    img = cv.imread(args.slika)
    if img is None:
        print("Napaka pri branju slike", file=sys.stderr)
        sys.exit(1)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    if args.alg == "kmeans":
        result = kmeans(img, k=args.k, iteracije=args.iter, dimenzija=args.dim, T=args.T)
    else:
        result = meanshift(img, h=args.h, dimenzija=args.dim, max_iter=args.iter, min_cd=args.min_cd)

    # shrani rezultat
    ok = cv.imwrite(args.out, cv.cvtColor(result, cv.COLOR_RGB2BGR))
    if not ok:
        print("Napaka pri shranjevanju slike", file=sys.stderr)
        sys.exit(1)
    print(f"Rezultat shranjen v '{args.out}'")
