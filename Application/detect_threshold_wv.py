import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pywt
import pandas as pd
from scipy.stats import norm
import os
import matplotlib.patches as patches


def calculer_seuil_auditif(filename, risk):
    data = loadmat(filename)

    # Paramètres
    cson = 340
    t0 = 1e-3 * data['stimparams']['espace'][0][0][0][0] / 2
    ttube = 0.08 / cson
    time = 1e3 * (data['tt1'] - t0 - ttube)
    time = time[:, 0]
    gain = data['stimparams']['gain'][0][0][0][0]
    hrfac = 1e6 / gain
    h = hrfac * np.array(data['hravsing'])
    h = h - np.mean(h, axis=0)

    # Paramètres
    wavelets = "cmor2.5-0.5"
    widths = np.geomspace(1, 1024, num=100)
    fs = data['stimparams']['fech'][0][0][0][0] * 1e3
    sampling_period = 1 / fs

    fig, ax = plt.subplots(nrows=9, ncols=1, figsize=(15, 30))

    # Paramètres
    intervalle = np.arange(90, 0, -10)

    # Initialisation des boxes
    amplitude_means = []
    box1 = []
    box2 = []
    box3 = []

    # Boucle d'affichage
    for i in range(h.shape[1]):
        print(f"Niveau d'intensité: {intervalle[i]} dB")
        # Récupération du signal
        signal = h[:, i]

        # Transformée en ondelettes
        cwtmatr, freqs = pywt.cwt(signal, widths, wavelets, sampling_period=sampling_period)
        cwtmatr = np.abs(cwtmatr)

        # Affichage de chaque transformée en ondelettes
        ax_subplot = ax[i]
        pcm = ax_subplot.pcolormesh(time, freqs, cwtmatr, cmap='turbo')
        ax_subplot.axis('tight')
        ax_subplot.axvline(x=0, color='w')
        ax_subplot.axvline(x=10, color='w')
        ax_subplot.set_yscale('log')
        ax_subplot.set_xlabel("Time (s)")
        ax_subplot.set_ylabel("Frequency (Hz)")
        ax_subplot.set_title(f'{intervalle[i]} dB')
        fig.colorbar(pcm, ax=ax_subplot)

        # Position des boxes
        boxes = [(-7, -3, 400, 5000), (0.5, 4.5, 400, 5000), (10, 14, 400, 5000)]
        colors = ['white', 'white', 'white']

        row_means = []
        for j, (start_time, end_time, start_freq, end_freq) in enumerate(boxes):
            # Création des box
            ax_subplot.add_patch(
                patches.Rectangle((start_time, start_freq), end_time - start_time, end_freq - start_freq, linewidth=3,
                                  edgecolor=colors[j], facecolor='none'))

            # Amplitudes dans les box
            time_indices = np.where((start_time <= time) & (time <= end_time))[0]
            freq_indices = np.where((start_freq <= freqs) & (freqs <= end_freq))[0]
            amplitudes_in_box = np.abs(cwtmatr[np.ix_(freq_indices, time_indices)])
            mean_amplitude = np.mean(amplitudes_in_box)
            row_means.append(mean_amplitude)

            # Remplir les boxes des amplitudes récupérées
            if j == 0:
                box1.append(amplitudes_in_box)
            if j == 1:
                box2.append(amplitudes_in_box)
            if j == 2:
                box3.append(amplitudes_in_box)

        amplitude_means.append(row_means)

    plt.tight_layout()
    plt.show()

    # Calcul de la box 4
    sums = []
    for i in range(len(box1)):
        sum = (box1[i] + box3[i]) / 2
        sums.append(sum)

    # Création d'un tableau pandas
    columns = ['Box 1 (-7 to -3 ms)', 'Box 2 (0 to 5 ms)', 'Box 3 (10 to 14 ms)']
    df = pd.DataFrame(amplitude_means, columns=columns)
    df['Mean of Sum (Box 1 + Box 3)'] = np.mean(np.mean(sums, axis=1), axis=1)
    std_mean_of_sum = np.std(np.std(sums, axis=1), axis=1)
    risk = 1e-5  # à modifier
    df['Condition (Mean of Sum > Box 2)'] = (
            (df['Mean of Sum (Box 1 + Box 3)'] + norm.ppf(1 - risk, 0, 1) * std_mean_of_sum) < df[
        columns[1]]).astype(int)

    # Paramètres et données pour le calcul d'interpolation des moyennes
    mean_of_sum = df['Mean of Sum (Box 1 + Box 3)'] + norm.ppf(1 - risk, 0, 1) * std_mean_of_sum
    box2_mean = df['Box 2 (0 to 5 ms)']
    i_values = np.arange(len(mean_of_sum))
    dB = np.arange(90, -1, -10)[:-1]

    # Conversion en array
    i_values = np.array(i_values)
    mean_of_sum = np.array(mean_of_sum)
    box2_mean = np.array(box2_mean)

    # Delta
    difference = mean_of_sum - box2_mean
    cross_indices = np.where(np.diff(np.sign(difference)))[0]

    # Calcul de l'interpolation et affichage des courbes de moyennes
    x_cross_dB = 0
    x_cross = None
    y_cross = None
    if len(cross_indices) > 0:
        index = cross_indices[0]
        x1, x2 = i_values[index], i_values[index + 1]
        y1, y2 = difference[index], difference[index + 1]
        x_cross = x1 - y1 * (x2 - x1) / (y2 - y1)

        dB1, dB2 = dB[index], dB[index + 1]
        x_cross_dB = dB1 + (dB2 - dB1) * (x_cross - x1) / (x2 - x1)
        y_cross = mean_of_sum[index] + (x_cross - x1) * (mean_of_sum[index + 1] - mean_of_sum[index]) / (x2 - x1)

    fig2, ax2 = plt.subplots()
    ax2.plot(i_values, mean_of_sum, label='Mean of Sum (Box 1 + Box 3)')
    ax2.plot(i_values, box2_mean, label='Box 2 (0 to 5 ms)')

    # Afficher le point de croisement
    if x_cross is not None and y_cross is not None:
        ax2.scatter(x_cross, y_cross, color='red', zorder=5)
        ax2.annotate(f'({x_cross:.2f}, {y_cross:.2f})', (x_cross, y_cross), textcoords="offset points", xytext=(10, 10),
                     ha='center')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Value')
    ax2.set_title('Mean of Sum vs Box 2')
    ax2.legend()
    ax2.grid(False)
    plt.show()

    # La fonction retourne x_cross_dB et les figures
    if x_cross_dB is not None:
        return x_cross_dB, fig, fig2
    else:
        return None, fig, fig2