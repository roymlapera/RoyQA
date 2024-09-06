import tkinter as tk
from tkinter.font import BOLD
from PIL import ImageTk, Image #para hacer ajustes en la imagen
from tkinter import filedialog, messagebox
import sys
import os
import Curvas as curv
# import Starshot as star
# import Picketfence as pf

monaco_calculation_file_path = 'P:/8 - Físicos Médicos/Roy/INTECNUS---RoyQA/MONACO'

class VentanaPrincipal:
                                   
    def __init__(self):     # cuando inicializo esta clase se crea una ventana con prop predeterminadas   
        self.ventana = tk.Tk()                            
        self.ventana.title('~ RoyQA ~')                                              #titulo de la ventana
        self.ventana.geometry("800x600")  # Width = 600, Height = 400
        self.ventana.resizable(False, False)
        
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
    filename_med = ElegirArchivosCurvas()

    VentanaParametrosGamma()

    measured_curves = curv.import_measured_curves(filename_med)
    calculated_curves = curv.import_Monaco_calculated_curves_from_database()

    is_profile = lambda curve: curve.coordinate == 'X' or curve.coordinate == 'Y'

    Perfiles_med = [curve for curve in measured_curves if is_profile(curve)]
    Perfiles_cal = [curve for curve in calculated_curves if is_profile(curve)]

    print('Medido:')
    for curve in Perfiles_med:
        curv.print_curve_data(curve)
        
    print('Calculado:')
    for curve in Perfiles_cal:
        if curve.machine == Perfiles_med[0].machine:
            curv.print_curve_data(curve)

    Perfiles_med, Perfiles_cal = curv.ProcesaCurvas(Perfiles_med,Perfiles_cal,espaciamiento=0.2,sigma_suavizado=0.00005)

    curv.Analiza_Y_Grafica_Perfiles(Perfiles_med,Perfiles_cal,dose_threshold,dta,cutoff, COMPARE_ANYWAY=False)

def AnalisisPDDs():
    filename_med = ElegirArchivosCurvas()

    VentanaParametrosGamma()

    measured_curves = curv.import_measured_curves(filename_med)
    calculated_curves = curv.import_Monaco_calculated_curves_from_database()

    is_pdd = lambda curve: curve.coordinate == 'Z'

    PDDs_med = [curve for curve in measured_curves if is_pdd(curve)]
    PDDs_cal = [curve for curve in calculated_curves if is_pdd(curve)]

    PDDs_med, PDDs_cal = curv.ProcesaCurvas(PDDs_med,PDDs_cal,espaciamiento=0.2,sigma_suavizado=0.00005)

    print('Medido:')
    for curve in PDDs_med:
        curv.print_curve_data(curve)
        if curve.field_size == 20 and curve.coordinate == 'Z' and curve.particle == 1:
            print(f'R50: {curv.get_R50(curve)} mm')

    print('Calculado:')
    for curve in PDDs_cal:
        if curve.machine == PDDs_med[0].machine:
            curv.print_curve_data(curve)
            if curve.field_size == 20 and curve.coordinate == 'Z' and curve.particle == 1:
                print(f'R50: {curv.get_R50(curve)} mm')


    curv.Analiza_Y_Grafica_PDDs(PDDs_med,PDDs_cal,dose_threshold,dta,cutoff, COMPARE_ANYWAY=False)


def ElegirArchivosCurvas():
    filename_med =  filedialog.askopenfilename(
                    title='Seleccionar dosis medida',
                    filetypes=(('Archivo ASCII','*.csv'),
                                ('Todos los archivos','*.*')))

    return filename_med

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

