import tkinter as tk
from tkinter.font import BOLD
from PIL import ImageTk, Image #para hacer ajustes en la imagen
from tkinter import filedialog, messagebox
import sys
import os
import Curvas as curv
# import Starshot as star
# import Picketfence as pf

class VentanaPrincipal:
                                   
    def __init__(self):     # cuando inicializo esta clase se crea una ventana con prop predeterminadas   
        self.ventana = tk.Tk()                            
        self.ventana.title('~ RoyQA ~')                                              #titulo de la ventana
        self.ventana.state('zoomed')
        
        logo = ImageTk.PhotoImage(Image.open("logo.png").resize((274, 117)))    #importo imagen 
                        
        label = tk.Label( self.ventana, bg='gray', image=logo  )             #inserto la imagen como logo
        label.place(x=0,y=0,relwidth=1, relheight=1)                          #permito expandir ventana para centrar
        
        self.menubar = tk.Menu(self.ventana)
        self.ventana.config(menu=self.menubar)                                     # Lo asignamos a la base

        filemenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label='Nuevo Análisis', menu=filemenu)
        filemenu.add_command(label='Análisis PDDs', command=AnalisisPDDs)
        filemenu.add_command(label='Análisis Perfiles', command=AnalisisPerfiles) 
        # filemenu.add_command(label='Análisis Starshots', command=AnalisisStarshot)
        # filemenu.add_command(label='Análisis Picket Fence', command=AnalisisPicketFence)             #.#
        filemenu.add_separator()
        filemenu.add_command(label='Salir', command=self.ventana.quit)

        helpmenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label='Ayuda', menu=helpmenu)
        helpmenu.add_command(label="Guía de usuario")
        helpmenu.add_separator()
        helpmenu.add_command(label="Acerca de...")

        self.ventana.mainloop()                                                #creo la ventana y la mantengo abierta


def TestBox():
    messagebox.showinfo(message="Todo ok!",title="Mensaje")


def AnalisisPerfiles():
    filename_med, filename_cal = ElegirArchivosCurvas()

    VentanaParametrosGamma()

    Perfiles_cal = curv.IngresaPerfilCalculadoMONACO(filename_cal)
    Perfiles_med = curv.IngresaPerfilMedido(filename_med)

    Perfiles_med, Perfiles_cal = curv.ProcesaCurva(Perfiles_med,Perfiles_cal,espaciamiento=0.5,sigma_suavizado=0.00005)

    curv.Analiza_Y_Grafica_Perfiles(Perfiles_med,Perfiles_cal,dose_threshold,dta,cutoff, COMPARE_ANYWAY=False)

def AnalisisPDDs():
    filename_med, filename_cal = ElegirArchivosCurvas()

    VentanaParametrosGamma()

    PDDs_cal = curv.IngresaPDDCalculadoMONACO(filename_cal)
    PDDs_med = curv.IngresaPDDMedido(filename_med)


    PDDs_med, PDDs_cal = curv.ProcesaCurva(PDDs_med,PDDs_cal,espaciamiento=0.2,sigma_suavizado=0.0005)

    curv.Analiza_Y_Grafica_PDDs(PDDs_med,PDDs_cal,dose_threshold,dta,cutoff, COMPARE_ANYWAY=False)


def ElegirArchivosCurvas():
    filename_med =  filedialog.askopenfilename(
                    title='Seleccionar dosis medida',
                    filetypes=(('Archivo ASCII','*.asc'),
                                ('Todos los archivos','*.*')))

    filename_cal =  filedialog.askopenfilename(
                    title='Seleccionar dosis calculada',
                    filetypes=(('Archivo de datos','*.1'),
                               ('Archivo de datos','*.2'),
                                ('Todos los archivos','*.*')))

    return filename_med, filename_cal

def VentanaParametrosGamma():
    root= tk.Toplevel()

    canvas = tk.Canvas(root, width = 600, height = 400,  relief = 'raised')
    canvas.grid()

    label = tk.Label(canvas, text='Ingresar parámetros de cálculo de Indice Gamma')
    label.config(font=('helvetica', 14))
    label.grid(row=0, column=1, padx = 20, pady = 20)

    label1 = tk.Label(canvas, text='Dosis de acuerdo (%):')
    label1.config(font=('helvetica', 10))
    label1.grid(row=2, column=0, padx = 20)  

    label2 = tk.Label(canvas, text='Distancia de acuerdo (mm):')
    label2.config(font=('helvetica', 10))
    label2.grid(row=2, column=1, padx = 20)

    label3 = tk.Label(canvas, text='Threshold de dosis (%):')
    label3.config(font=('helvetica', 10))
    label3.grid(row=2, column=2, padx = 20)

    dose_threshold_var = tk.StringVar()
    dta_var = tk.StringVar()
    cutoff_var = tk.StringVar()

    entry1 = tk.Entry(canvas, textvariable = dose_threshold_var) 
    entry1.insert(0, 2)  # Set the default value
    entry1.grid(row=3, column=0, pady = 10)
    
    entry2 = tk.Entry(canvas, textvariable = dta_var) 
    entry2.insert(0, 2)  # Set the default value
    entry2.grid(row=3, column=1, pady = 10)

    entry3 = tk.Entry(canvas, textvariable = cutoff_var) 
    entry3.insert(0, 10)  # Set the default value
    entry3.grid(row=3, column=2, pady = 10)

    def submit():
        global dose_threshold, dta, cutoff

        dose_threshold = float(dose_threshold_var.get())
        dta            = float(dta_var.get())
        cutoff         = float(cutoff_var.get())
         
        dose_threshold_var.set("")
        dta_var.set("")
        cutoff_var.set("")

        root.master.destroy()

    exit_button = tk.Button(canvas, text="Analizar", command=submit,
                                bg='gray', fg='white', font=('helvetica', 9, 'bold'))
    exit_button.grid(row=4, column=1, padx = 20, pady = 20)

    root.mainloop()

