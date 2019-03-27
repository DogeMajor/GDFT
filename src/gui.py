from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
from tkinter import messagebox
from optimizer import Optimizer, Runner
from analyzer import ThetasAnalyzer
from utils import extract_thetas_records

corr_fns = {1: "avg_auto_corr", 2: "max_auto_corr",
            3: "avg_cross_corr", 4: "max_cross_corr"}


class GUI(object):

    def set_cores(self):
        cores = simpledialog.askinteger("Cores", "Number of cores available",
                                        parent=root,
                                        minvalue=1, maxvalue=50)
        if cores is not None:
            print("No of cores was set to {}".format(cores))
            self.coresVar.set(int(cores))
        else:
            print("No multiprocessing is used")

    def quit(self):
        root.quit()

    def optimize(self):
        dim = self.dimensionVar.get()
        corr_fn_name = corr_fns[self.corrChoiceVar.get()]
        epochs = self.thetasAmountVar.get()
        file_name = self.filenameVar.get()+"__"
        path = self.pathVar.get()
        stop_criteria = self.stopCriteriumVar.get()
        cores = self.coresVar.get()

        runner = Runner(dim)
        results = runner.optimize(corr_fn_name, epochs, stop_criteria=stop_criteria, cores=cores)
        full_name = runner.save_results(file_name, results, file_path=path, file_format="json")
        self._recent_files.append(full_name)

    def show_recent_files(self):
        print(self._recent_files + self._analyzed_thetas)

    def analyze_thetas(self, save=True):
        path = self.pathVar.get()
        cutoff = self.cutoffVar.get()
        try:
            file_name = self._recent_files[-1]
            theta_collections = extract_thetas_records(path, file_name)
        except:
            raise FileNotFoundError("No thetas were optimized during session")

        dim = self.dimensionVar.get()
        analyzer = ThetasAnalyzer(dim)
        groups = min(len(theta_collections.thetas), int(dim*0.75))

        sorted_thetas = analyzer.sort_thetas(theta_collections.thetas, groups)
        cov_pca_reductions = analyzer.cov_pca_reductions(sorted_thetas, cutoff_ratio=cutoff)

        if save:
            path = self.pathVar.get()
            file_name = file_name.split("__")[0]

            sorted_name = analyzer.save_sorted_thetas(sorted_thetas, file_name + "_sorted__", path)
            self._analyzed_thetas.append(sorted_name)
            cov_reductions_name = analyzer.save_cov_reductions(cov_pca_reductions, file_name + "_cov_reductions__", path)
            self._analyzed_thetas.append(cov_reductions_name)
        #sol_spaces = analyzer.solution_spaces(sorted_thetas, cutoff_ratio=cutoff)

    def show_about(self, event=None):
        self.show_vars()
        messagebox.showwarning("About",
                               "DogeHouse Productions NLC (No liability company)")

    def show_vars(self):
        print(self._recent_files, self._analyzed_thetas, corr_fns[self.corrChoiceVar.get()],
              self.pathVar.get(), self.coresVar.get(), self.dimensionVar.get(),
              self.stopCriteriumVar.get(), self.filenameVar.get())


    def __init__(self, root):
        #self._root = root
        self._recent_files = []
        self._analyzed_thetas = []

        self.corrChoiceVar = tk.IntVar()
        self.coresVar = tk.IntVar()
        self.thetasAmountVar = tk.IntVar()
        self.dimensionVar = tk.IntVar()
        self.stopCriteriumVar = tk.DoubleVar()
        self.filenameVar = tk.StringVar()
        self.pathVar = tk.StringVar()
        self.cutoffVar = tk.DoubleVar()

        self.corrChoiceVar.set(1)
        self.coresVar.set(2)
        self.dimensionVar.set(4)
        self.stopCriteriumVar.set(0.5)
        self.filenameVar.set("Enter a name")
        self.pathVar.set("data/")
        self.thetasAmountVar.set(10)

        self.cutoffVar.set(0.05) #For cov pca reduction in sorted theta groups

        self.set_menus()
        self.set_widgets()
        #self.doge_processing()


    def set_widgets(self):

        Label(root, text="Dimension").grid(row=1, column=0, sticky=W)
        dim_entry = Entry(root, width=50, textvariable=self.dimensionVar)
        dim_entry.grid(row=1, column=1)
        #Button(root, text="Submit").grid(row=1, column=8)

        Label(root, text="Amount").grid(row=2, column=0, sticky=W)
        thetas_amount_entry = Entry(root, width=50, textvariable=self.thetasAmountVar)
        thetas_amount_entry.grid(row=2, column=1)

        Label(root, text="Stop criterium").grid(row=3, column=0, sticky=W)
        Entry(root, width=50, textvariable=self.stopCriteriumVar).grid(row=3, column=1)

        Label(root, text="File name").grid(row=6, column=0, sticky=W)
        Entry(root, width=50, textvariable=self.filenameVar).grid(row=6, column=1)

        Label(root, text="File path").grid(row=7, column=0, sticky=W)
        Entry(root, width=50, textvariable=self.pathVar).grid(row=7, column=1)


        Label(root, text="Correlation to be minimized").grid(row=8, column=0, sticky=W)
        Radiobutton(root, text="Average auto correlation", value=1,
                    variable=self.corrChoiceVar).grid(row=9, column=1, sticky=W)
        Radiobutton(root, text="Max auto correlation", value=2,
                    variable=self.corrChoiceVar).grid(row=10, column=1, sticky=W)
        Radiobutton(root, text="Average cross correlation", value=3,
                    variable=self.corrChoiceVar).grid(row=11, column=1, sticky=W)
        Radiobutton(root, text="Max cross correlation", value=4,
                    variable=self.corrChoiceVar).grid(row=12, column=1, sticky=W)

        optimizeButton = Button(root, text="Optimize", command=self.optimize)
        optimizeButton.grid(row=12, column=3, sticky=E)
        #print(optimizeButton)
        optimizeButton.bind("<Button-1>", self.optimize)

        Label(root, text="Cutoff ratio for variances").grid(row=15, column=0, sticky=W)
        Entry(root, width=50, textvariable=self.cutoffVar).grid(row=15, column=1)

        analyzeButton = Button(root, text="Analyze", command=self.analyze_thetas)
        analyzeButton.grid(row=15, column=3, sticky=E)
        #analyzeButton.bind("<Button-1>", self.analyze_thetas)


    def set_menus(self):
        the_menu = Menu(root)

        file_menu = Menu(the_menu, tearoff=0)
        file_menu.add_command(label="Quit", command=self.quit)
        the_menu.add_cascade(label="File", menu=file_menu)

        # ----- SETTINGS MENU -----
        settings_menu = Menu(the_menu, tearoff=0)
        settings_menu.add_command(label="Cores",
                                  command=self.set_cores)
        the_menu.add_cascade(label="Settings", menu=settings_menu)

        # ----- HELP MENU -----
        help_menu = Menu(the_menu, tearoff=0)
        help_menu.add_command(label="About",
                              accelerator="command-H",
                              command=self.show_about)
        the_menu.add_cascade(label="Help", menu=help_menu)

        root.config(menu=the_menu)

if __name__ == "__main__":
    root = Tk()
    root.geometry("600x400")
    root.title("GDFT Optimizer and Analyzer")
    gui = GUI(root)
    root.mainloop()
