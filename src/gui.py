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

paths = {1: "data/", 2: "/Documents/"}


# You can unbind and rebind an event to a function
'''def bind_button(event):
    if boolVar.get():
        getDataButton.unbind("<Button-1>")
    else:
        getDataButton.bind("<Button-1>", get_data)'''


class GUI(object):

    def quit(self):
        self._root.quit()

    def optimize(self, dim, corr_fn_name, epochs, file_name, file_format, path, stop_criteria=None, cores=1):
        runner = Runner(dim)
        results = runner.optimize(corr_fn_name, epochs, stop_criteria=stop_criteria, cores=cores)
        runner.save_results(file_name, results, file_format=file_format, file_path=path)

    def show_recent_files(self):
        print(self._files)

    def analyze_thetas(self, path="data/", file_name=None):
        if file_name == None:
            file_name = self._recent_files[-1]
            thetas = extract_thetas_records(path, file_name)
        else:
            thetas = extract_thetas_records(path, file_name)
        dim = thetas.thetas[0].shape[0]
        analyzer = ThetasAnalyzer(dim)

    def show_about(self, event=None):
        self.show_vars()
        messagebox.showwarning("About",
                               "DogeHouse Productions NLC (No liability company)")

    def show_vars(self):
        print(self._recent_files, str(self.fileformatVar.get()), str(self.corrChoiceVar.get()),
              str(self.pathChoiceVar.get()), str(self.coresVar.get()))

    def select_corr(self):
        self.file_format = self.corrChoiceVar.get()

    def select_file_format(self):
        self.file_format = self.fileformatVar.get()


    def __init__(self, root):
        self._root = root
        self._recent_files = []

        self.fileformatVar = tk.IntVar()
        self.corrChoiceVar = tk.IntVar()
        self.pathChoiceVar = tk.IntVar()
        self.coresVar = IntVar()

        self.fileformatVar.set(1)
        self.corrChoiceVar.set(1)
        self.pathChoiceVar.set(1)
        self.coresVar.set(2)

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

        Label(root, text="Dimension").grid(row=1, column=0, sticky=W)
        Entry(root, width=50).grid(row=1, column=1)
        #Button(root, text="Submit").grid(row=1, column=8)

        Label(root, text="Amount").grid(row=2, column=0, sticky=W)
        Entry(root, width=50).grid(row=2, column=1)
        #Button(root, text="Submit").grid(row=2, column=8)

        Label(root, text="Stop criterium").grid(row=3, column=0, sticky=W)
        Entry(root, width=50).grid(row=3, column=1)
        #Button(root, text="Submit").grid(row=3, column=8)

        Label(root, text="File name").grid(row=6, column=0, sticky=W)
        Entry(root, width=50).grid(row=6, column=1)
        #Button(root, text="Submit").grid(row=6, column=8)

        Label(root, text="File format").grid(row=7, column=0, sticky=W)
        Radiobutton(root, text="JSON", value=1, variable=self.fileformatVar,
                    command=self.select_file_format).grid(row=8, column=0, sticky=W)
        Radiobutton(root, text="Excel", value=2, variable=self.fileformatVar,
                    command=self.select_file_format).grid(row=9, column=0, sticky=W)
        Radiobutton(root, text="CSV", value=3, variable=self.fileformatVar,
                    command=self.select_file_format).grid(row=10, column=0, sticky=W)

        Label(root, text="Path").grid(row=7, column=1, sticky=W)
        Checkbutton(root, text="Data/").grid(row=8, column=1, sticky=W)
        Checkbutton(root, text="/Documents/").grid(row=9, column=1, sticky=W)


        optimizeButton = Button(root, text="Optimize").grid(row=10, column=2)
        #print(optimizeButton)
        #optimizeButton.bind("<Button-1>", self.optimize)

        analyzeButton = Button(root, text="Analyze").grid(row=11, column=2)
        #analyzeButton.bind("<Button-1>", self.analyze_thetas)


        Label(root, text="Correlation to be minimized").grid(row=7, column=1, sticky=W)
        Radiobutton(root, text="Average auto correlation", value=1, variable=self.corrChoiceVar,
                    command=self.select_corr).grid(row=8, column=1, sticky=W)
        Radiobutton(root, text="Max auto correlation", value=2, variable=self.corrChoiceVar,
                    command=self.select_corr).grid(row=9, column=1, sticky=W)
        Radiobutton(root, text="Average cross correlation", value=3, variable=self.corrChoiceVar,
                    command=self.select_corr).grid(row=10, column=1, sticky=W)
        Radiobutton(root, text="Max cross correlation", value=4, variable=self.corrChoiceVar,
                    command=self.select_corr).grid(row=11, column=1, sticky=W)

root = Tk()

root.geometry("600x600")
root.title("GDFT Optimizer and Analyzer")
gui = GUI(root)
root.mainloop()