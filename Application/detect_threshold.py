from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def calculer_seuil_cross_correlation(filename, risk, risk2, risk3):
    fig, ax = plt.subplots(figsize=(5, 5))

    # Donnée
    data = loadmat(filename)

    # Paramètres récupérés du fichier .mat
    lmi = data['stimparams']['Lmin'][0][0][0][0]
    lmax = data['stimparams']['Lmax'][0][0][0][0]
    dl = data['stimparams']['delL'][0][0][0][0]
    L = np.flip(np.arange(lmi, lmax + 1, np.abs(dl), dtype=int))
    t0 = 1e-3 * data['stimparams']['espace'][0][0][0][0] / 2
    ttube = 0.08 / 340
    tt = 1e3 * (data['tt1'] - t0 - ttube)
    fs = data['stimparams']['fech']
    gain = data['stimparams']['gain'][0][0][0][0]
    hrfac = 1e6 / gain

    # Définition des parties pre-signal et signal
    period_pre = tt[tt < 0]
    period_signal = np.arange(1 / fs, 12 + 1 / fs, 1 / fs)
    period_pre = len(period_pre)
    period_signal = len(period_signal)

    try:
        h = hrfac * data['hravsing']
    except AttributeError:
        h = hrfac * data['nrav']

    # Signal récupéré
    h = h - np.mean(h, axis=0)

    # Initialisation des seuils significatifs et du max de la seconde cross-correlation
    xc_signal = np.zeros(len(L) - 1)
    threshold = np.zeros(len(L) - 1)
    threshold2 = np.zeros(len(L) - 1)
    threshold3 = np.zeros(len(L) - 1)

    # Boucle cross-correlation sur tous les signaux
    for i in range(len(L) - 1):
        previous_average = h[:, i]
        current_average = h[:, i + 1]
        # Première cross-correlation
        xc_pre = np.correlate(current_average[:period_pre], previous_average[period_pre:(period_pre + period_signal)],
                              mode='full')
        # Seuil de significativité calculer à partir de la première cross-correlation avec 3 intervalles de confiance différents
        threshold[i] = np.mean(xc_pre) + norm.ppf(1 - risk, 0, 1) * np.std(xc_pre)
        threshold2[i] = np.mean(xc_pre) + norm.ppf(1 - risk2, 0, 1) * np.std(xc_pre)
        threshold3[i] = np.mean(xc_pre) + norm.ppf(1 - risk3, 0, 1) * np.std(xc_pre)
        # Maximum de la seconde cross-correlation
        xc_signal[i] = np.max(np.correlate(current_average[period_pre:(period_pre + period_signal)],
                                           previous_average[period_pre:(period_pre + period_signal)], mode='full'))

    # Copie de la première valeur pour chaque tableau
    xc_signal = np.insert(xc_signal, 0, xc_signal[0])
    threshold = np.insert(threshold, 0, threshold[0])
    threshold2 = np.insert(threshold2, 0, threshold2[0])
    threshold3 = np.insert(threshold3, 0, threshold3[0])

    # Affichage de la max correlation et des seuils significatifs
    ax.plot(L[:len(xc_signal)], xc_signal, '-or', markerfacecolor='r')
    ax.plot(L[:len(threshold)], threshold, '--k')  # 1e-5
    # plt.plot(L[:len(threshold2)], threshold2, '--m') # 1e-3
    # plt.plot(L[:len(threshold3)], threshold3, '--y') # 1e-2

    # Affichage en log
    # if np.all(threshold > 0) and np.all(xc_signal > 0):
    # plt.set_yscale('log')
    ax.set_xlabel('Level (dB SPL)')
    ax.set_ylabel('Cross-correlation max')
    ax.grid(False)

    # Condition : si le seuil de significativité est strictement supérieur au max(seconde cross-correlation) alors ne pas afficher le point en vert
    idx = np.where(xc_signal < threshold)[0]
    if len(idx) == 0 and np.all(xc_signal > threshold):
        ax.plot(L[:len(xc_signal)], xc_signal, 'ok', markerfacecolor='g', linewidth=2)
    elif len(idx) > 0 and idx[0] > 0:
        ax.plot(L[0:idx[0]], xc_signal[0:idx[0]], 'ok', markerfacecolor='g', linewidth=2)
    idxb = np.where(xc_signal < threshold)[0][-1] if len(np.where(xc_signal < threshold)[0]) > 0 else -1
    if idxb != -1 and idxb < len(xc_signal) - 1:
        plot_L = L[idxb + 1:]
        plot_xc_signal = xc_signal[idxb + 1:]
        if len(plot_L) == len(plot_xc_signal):
            ax.plot(plot_L, plot_xc_signal, 'ok', markerfacecolor='g', linewidth=2)

    # Calcul de l'interpolation
    c = np.where(np.diff(np.sign(xc_signal - threshold)))[0]
    if len(c) > 0:
        x1, x2 = L[c[0]], L[c[0] + 1]
        y1, y2 = xc_signal[c[0]] - threshold[c[0]], xc_signal[c[0] + 1] - threshold[c[0] + 1]
        thrsh = x1 - y1 * (x2 - x1) / (y2 - y1)
        # Affichage du seuil exacte
        ax.axvline(x=thrsh, color='b', linestyle='--')
    else:
        # Sinon mettre le seuil à nan
        thrsh = np.nan

    # Affichage des légendes
    ax.legend(['Max correlation', 'Signif Threshold', 'Above Threshold', 'Threshold'],
               loc='best')  # , 'Signif threshold 1e-3', 'Signif Threshold 1e-2'

    # Récupération du seuil (Si aucun seuil récupéré alors None)
    seuil = thrsh if thrsh is not None else None
    return seuil, fig