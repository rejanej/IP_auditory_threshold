#### Code réalisé par Réjane JOYARD ####
### Application ###

import tkinter as tk
from tkinter import filedialog
import tkmacosx as tkmx
from PIL import Image, ImageTk
import os
from scipy.io import loadmat
from detect_threshold import calculer_seuil_cross_correlation
from detect_threshold_wv import calculer_seuil_auditif
from extract_mat_from_otophylab import extract_mat_from_otophylab
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk


class ABRAnalysisApp:
    def __init__(self, root):
        self.master = root
        root.title("Application")

        # Initialisations
        self.Filename = None
        self.system = None
        self.which_dB = None
        self.results_table = None
        self.stim_type = None
        self.dat = None
        self.plot_type = tk.StringVar(root)
        self.plot_type.set('Auto')
        self.ReverseEditFieldLabel = None
        self.Method_thresh_DropDown = tk.StringVar()
        self.MethodDropDownLabel = None
        self.riskalpha = tk.StringVar()
        self.riskalphaDropDownLabel = None
        self.ThresholdTextArea = None
        self.image_label = None

        # Création DATA
        self.data_panel = tk.LabelFrame(root, text="DATA features", bg="powderblue", font=("Arial", 12, "bold"))
        self.data_panel.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.load_button = tkmx.Button(self.data_panel, text="Load File", command=self.load_file, bg="royalblue", fg='white', font=("Arial", 10, "bold"))
        self.load_button.grid(row=0, column=0, padx=5, pady=5)

        self.info_text = tk.Text(self.data_panel, height=4, width=40, bg="white")
        self.info_text.grid(row=1, column=0, padx=5, pady=5)

        # Création ANALYSIS
        self.analysis_panel = tk.LabelFrame(root, text="ANALYSIS", bg="powderblue", font=("Arial", 12, "bold"))
        self.analysis_panel.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.auditory_button = tkmx.Button(self.analysis_panel, text="Auditory Threshold", command=self.AuditoryThresholdButtonPushed, bg="royalblue", fg='white', font=("Arial", 10, "bold"))
        self.auditory_button.grid(row=0, column=0, padx=5, pady=5)

        self.threshold_text = tk.Text(self.analysis_panel, height=4, width=20, bg="white")
        self.threshold_text.grid(row=0, column=1, padx=5, pady=5)

        self.method_label = tk.Label(self.analysis_panel, text="Method", bg="powderblue", font=("Arial", 10))
        self.method_label.grid(row=2, column=0, padx=5, pady=5)

        self.method_dropdown = tk.OptionMenu(self.analysis_panel, self.Method_thresh_DropDown, "Cross-correlation", "Wavelet")
        self.method_dropdown.grid(row=2, column=1, padx=5, pady=5)
        self.Method_thresh_DropDown.set("Cross-correlation")

        # Création Refresh
        self.refresh_button = tkmx.Button(root, text="Refresh", command=self.refresh_data,bg="royalblue", fg='white', font=("Arial", 10, "bold"))
        self.refresh_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        # Graphe
        self.image_panel = tk.LabelFrame(root, text="GRAPH", bg="powderblue", font=("Arial", 12, "bold"))
        self.image_panel.grid(row=0, column=3, rowspan=4,padx=10, pady=10)

    def refresh_data(self):
        self.info_text.delete("1.0", tk.END)
        if self.filename:
            self.display_image()

    def load_file(self):
        file_path = filedialog.askopenfilename(title='select file')
        if file_path:
            fname = os.path.basename(file_path)
            fext = os.path.splitext(fname)[1]
            self.filename = file_path
            self.info_text.delete("1.0", tk.END)
            self.info_text.insert(tk.END, f'File name = {fname}\n')

            self.display_image()

            if fext == '.xls':
                new_mat_file = extract_mat_from_otophylab(self.filename)
                self.dat = new_mat_file
            else:
                self.dat = loadmat(self.filename)

            self.info_text.delete("1.0", tk.END)
            self.info_text.insert(tk.END, f'File : {self.filename}\n')

    def display_image(self):
        jpeg_filename = os.path.splitext(self.filename)[0] + ".jpeg"
        if os.path.exists(jpeg_filename):
            image = Image.open(jpeg_filename)
            image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(image)
            if self.image_label:
                self.image_label.configure(image=photo)
                self.image_label.image = photo
            else:
                self.image_label = tk.Label(self.image_panel, image=photo)
                self.image_label.image = photo
                self.image_label.pack(fill="both", expand=True)

    def AuditoryThresholdButtonPushed(self):
        method_threshold = self.Method_thresh_DropDown.get()
        if method_threshold == "Cross-correlation":
            method_threshold = 1
        elif method_threshold == "Wavelet":
            method_threshold = 2

        if method_threshold == 1:
            output, fig = calculer_seuil_cross_correlation(self.filename, 1e-5, risk2=1e-3, risk3=1e-2)
            self.display_plot(fig)
        elif method_threshold == 2:
            output, fig, fig1 = calculer_seuil_auditif(self.filename, 1e-5)
            self.display_wavelet_plot(fig)

        self.threshold_text.delete("1.0", tk.END)
        self.threshold_text.insert(tk.END, f'Threshold is {output} dB.')

    def display_plot(self, fig):
        correlation_window = tk.Toplevel(self.master)
        correlation_window.title("Cross-correlation")

        canvas = FigureCanvasTkAgg(fig, master=correlation_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def display_wavelet_plot(self, fig):
        wavelet_window = tk.Toplevel(self.master)
        wavelet_window.title("Wavelet Plot")

        canvas = FigureCanvasTkAgg(fig, master=wavelet_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, wavelet_window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

root = tk.Tk()
app = ABRAnalysisApp(root)
root.geometry("720x450")
root.mainloop()