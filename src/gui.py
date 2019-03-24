from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from optimizer import Optimizer, Runner
from analyzer import ThetasAnalyzer
from utils import extract_thetas_records

corr_fns = {1: "avg_auto_corr", 2: "max_auto_corr",
            3: "avg_cross_corr", 4: "max_cross_corr"}

file_formats = {1: "json", 2: "xl", 3: "csv"}


# You can unbind and rebind an event to a function
'''def bind_button(event):
    if boolVar.get():
        getDataButton.unbind("<Button-1>")
    else:
        getDataButton.bind("<Button-1>", get_data)'''


class GUI(object):

    def quit(self):
        root.quit()

    def optimize(self):
        dim = self.dimensionVar.get()
        corr_fn_name = corr_fns[self.corrChoiceVar.get()]
        epochs = self.thetasAmountVar.get()
        file_name = self.filenameVar.get()
        file_format = file_formats[self.fileformatVar.get()]
        path = self.pathVar.get()
        stop_criteria = self.stopCriteriumVar.get()
        cores = self.coresVar.get()

        runner = Runner(dim)
        results = runner.optimize(corr_fn_name, epochs, stop_criteria=stop_criteria, cores=cores)
        print("Results for optimization:")
        print(results)
        full_name = runner.save_results(file_name, results, file_format=file_format, file_path=path)
        self._recent_files.append(full_name)

    def show_recent_files(self):
        print(self._files)

    def analyze_thetas(self):
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
        sol_spaces = analyzer.solution_spaces(sorted_thetas, cutoff_ratio=cutoff)
        print("Sorted thetas histogram: ", sorted_thetas.histogram)
        print("Labels in the same order: ", sorted_thetas.labels)
        print("Theta groups with their biggest variances: ")
        for key, value in cov_pca_reductions.items():
            print(str(key) + ":")
            print("variances: ", value[1])


    def show_about(self, event=None):
        self.show_vars()
        messagebox.showwarning("About",
                               "DogeHouse Productions NLC (No liability company)")

    def show_vars(self):
        print(self._recent_files, file_formats[self.fileformatVar.get()],
              corr_fns[self.corrChoiceVar.get()],
              self.pathVar.get(), self.coresVar.get(),
              self.dimensionVar.get(), self.stopCriteriumVar.get(),
              self.filenameVar.get())


    def __init__(self, root):
        #self._root = root
        self._recent_files = []

        self.fileformatVar = tk.IntVar()
        self.corrChoiceVar = tk.IntVar()
        self.coresVar = tk.IntVar()
        self.thetasAmountVar = tk.IntVar()
        self.dimensionVar = tk.IntVar()
        self.stopCriteriumVar = tk.DoubleVar()
        self.filenameVar = tk.StringVar()
        self.pathVar = tk.StringVar()

        self.cutoffVar = tk.DoubleVar()

        self.fileformatVar.set(1)
        self.corrChoiceVar.set(1)
        self.coresVar.set(2)
        self.dimensionVar.set(8)
        self.stopCriteriumVar.set(0.5)
        self.filenameVar.set("No name was given!")
        self.pathVar.set("No path was given!")
        self.thetasAmountVar.set(1)

        self.cutoffVar.set(0.05) #For cov pca reduction in sorted theta groups

        self.set_menus()
        self.set_widgets()



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


        Label(root, text="File format").grid(row=8, column=0, sticky=W)
        Radiobutton(root, text="JSON", value=1,
                    variable=self.fileformatVar).grid(row=9, column=0, sticky=W)
        Radiobutton(root, text="Excel", value=2,
                    variable=self.fileformatVar).grid(row=10, column=0, sticky=W)
        Radiobutton(root, text="CSV", value=3,
                    variable=self.fileformatVar).grid(row=11, column=0, sticky=W)


        optimizeButton = Button(root, text="Optimize", command=self.optimize)
        optimizeButton.grid(row=11, column=3, sticky=E)
        #print(optimizeButton)
        optimizeButton.bind("<Button-1>", self.optimize)

        analyzeButton = Button(root, text="Analyze", command=self.analyze_thetas).grid(row=12, column=3, sticky=E)
        #analyzeButton.bind("<Button-1>", self.analyze_thetas)


        Label(root, text="Correlation to be minimized").grid(row=8, column=1, sticky=W)
        Radiobutton(root, text="Average auto correlation", value=1,
                    variable=self.corrChoiceVar).grid(row=9, column=1, sticky=W)
        Radiobutton(root, text="Max auto correlation", value=2,
                    variable=self.corrChoiceVar).grid(row=10, column=1, sticky=W)
        Radiobutton(root, text="Average cross correlation", value=3,
                    variable=self.corrChoiceVar).grid(row=11, column=1, sticky=W)
        Radiobutton(root, text="Max cross correlation", value=4,
                    variable=self.corrChoiceVar).grid(row=12, column=1, sticky=W)

    def set_menus(self):

        # Create the menu object
        the_menu = Menu(root)

        # Create a pull down menu that can't be removed
        file_menu = Menu(the_menu, tearoff=0)

        # Call for the function to execute when clicked
        file_menu.add_command(label="Quit", command=self.quit)

        # Add the pull down menu to the menu bar
        the_menu.add_cascade(label="File", menu=file_menu)

        # ----- SETTINGS MENU -----
        settings_menu = Menu(the_menu, tearoff=0)

        # Bind the checking of the line number option
        # to variable line_numbers
        settings_menu.add_checkbutton(label="Cores",
                                      variable=self.coresVar)

        # Add the pull down menu to the menu bar
        the_menu.add_cascade(label="Settings", menu=settings_menu)

        # ----- HELP MENU -----
        help_menu = Menu(the_menu, tearoff=0)


        help_menu.add_command(label="About",
                              accelerator="command-H",
                              command=self.show_about)

        the_menu.add_cascade(label="Help", menu=help_menu)

        # Bind the shortcut to the function
        root.bind('<Command-A>', self.show_about)

        # Display the menu bar
        root.config(menu=the_menu)

root = Tk()

root.geometry("600x600")
root.title("GDFT Optimizer and Analyzer")
gui = GUI(root)
root.mainloop()