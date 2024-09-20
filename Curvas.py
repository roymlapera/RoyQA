import numpy as np
import scipy.interpolate
from scipy.ndimage import gaussian_filter1d
import itertools
import pymedphys
from typing import List
from datetime import datetime
from pathlib import Path

from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import filedialog, Text
import os
import copy
# import xarray as xr

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, Float, PickleType
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import pandas as pd
import io

monaco_database_file_path = 'sqlite:///Monaco_Reference_Data_Calculation.db'

#----------------------------------------------------------------------------------------------------------------------------

class Curve:
    def __init__(self, id=None, type=None, coordinate=None, date=None, time=None, depth=None, SSD=None, field_size=None, axis=None, relative_dose=None, machine=None, particle=None, energy=None, detector=None):
        self.id = id
        self.type = type
        self.coordinate = coordinate
        self.date = date
        self.time = time
        self.depth = depth
        self.SSD = SSD
        self.field_size = field_size
        self.axis = axis
        self.relative_dose = relative_dose
        self.machine = machine
        self.particle = particle
        self.energy = energy
        self.detector = detector

Base = declarative_base()

class CurveModel(Base):
    __tablename__ = 'INTECNUS_Curves'
    id = Column(Integer, primary_key=True)
    type = Column(String)
    coordinate = Column(String)
    date = Column(String)
    time = Column(String)
    depth = Column(Float)
    SSD = Column(Float)
    field_size = Column(Float)
    axis = Column(PickleType)  # para almacenar numpy arrays
    relative_dose = Column(PickleType)  # para almacenar numpy arrays
    machine = Column(String)
    particle = Column(Integer)
    energy = Column(Float)
    detector = Column(String)

def Convert(string): 
    li = list(string.split(" ")) 
    return li 

def print_curve_data(curve):
        print(f'Equipo:      {curve.machine}, '
              f'Fecha:       {curve.date}, '
              f'Hora:        {curve.time}, '
              f'Particula:   {curve.particle}, '
              f'Energia:     {curve.energy}, '
              f'DFS:         {curve.SSD}, '
              f'TC:          {curve.field_size}, '
              f'Coordenada:  {curve.coordinate}, '
              f'Profundidad: {curve.depth}, '
              f'Detector:    {curve.detector}')

def import_measured_curves(fname):

    def normalize_date(date_string):
        # Intenta analizar la fecha con varios formatos
        for format in ('%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y', '%B %d, %Y', '%d %B %Y'):
            try:
                date_object = datetime.strptime(date_string, format)
                return date_object.strftime('%d/%m/%Y')
            except ValueError:
                continue
        
        # Si ningún formato coincide, devuelve una cadena de error
        return "Invalid date format"
    
    # def coordinate_classifier_ASCII(measurement_data):
    #     start_x, start_y, start_z = measurement_data[19][1:]
    #     end_x, end_y, end_z = measurement_data[20][1:]

    #     if(start_x == end_x and start_y == end_y):
    #         if (float(start_x) == 0 and float(start_y) == 0):
    #             return 'Z'
    #         else: 
    #             print('PDD no está en CAX!')
    #     elif(start_z == end_z):
    #         if(float(start_y) == 0):
    #             return 'X'
    #         elif(float(start_x) == 0):
    #             return 'Y'
    #         else:
    #             print('Perfil diagonal!')
    #     else:
    #         print('No se puede determinar coordinada de curva.')

    # def initialize_curve_from_ASCII_data(curve: Curve, measurement_data: List[str]) -> Curve:
    #     '''Asignación de atributos de la instancia de Curve basada en measurent_data en archivo en formato ASCII.'''
    #     curve.type = 'M'
    #     curve.coordinate = coordinate_classifier(measurement_data)    # a lo largo de X,Y,Z
    #     curve.date =       measurement_data[4][1]
    #     curve.time =       measurement_data[5][1]
    #     curve.machine = machine
    #     curve.particle = 1 if measurement_data[7][1] == 'ELE' else 0       # 1 = electrones    0 = fotones
    #     curve.energy = float(measurement_data[7][2])
    #     curve.depth = '' if curve.coordinate == 'Z' else float(measurement_data[21][3])   #mm
    #     curve.SSD = float(measurement_data[8][1])/10  #cm
    #     curve.field_size = float(measurement_data[6][1])/10     #[Y,X] solo cuadrado y simetrico

    #     scan_data = np.array(measurement_data[21:])[:,1:].astype(float)    #  [:,1:] = no tomo la columna de '='

    #     curve.relative_dose = scan_data[:,3]

    #     if curve.coordinate == 'X':         # LAS ORIENTACIONES Y NOMBRES DE LOS EJES PARECEN ESTAR EN FORMATO SERVO != FORMATO USADO
    #         curve.axis = scan_data[:,0]     # SE SUGIERE USAR ARCHIVOS CSV
    #     elif curve.coordinate == 'Y':
    #         curve.axis = scan_data[:,1]
    #     else:
    #         curve.axis = scan_data[:,2]

    # def ASCII_data_importing_parsing(fname: str, curve_set: List[str]) -> None:
    #     '''Cosmetica de parsing para archivo ASCII.'''
    #     raw_data = np.genfromtxt(fname, dtype=str, delimiter='\n', skip_header=2, skip_footer=0)

    #     measurement_data = []

    #     for s in raw_data: 
    #         s = [s.replace(' ', '') for s in s.split('\t')]
            
    #         if s[0] == '%VNR1.0':
    #             curve = Curve()
    #             measurement_data.clear()
    #             continue
    #         elif s[0] == ':EOM': 
    #             initialize_curve_from_ASCII_data(curve, measurement_data)
    #             # print_curve_data(curve)
    #             curve_set.append(curve)
    #         else: 
    #             measurement_data.append(s)

    def initialize_curve_from_CSV_data(curve: Curve, current_data_block: List[str]) -> Curve:
        '''Asignación de atributos de la instancia de Curve basada en measurent_data en archivo en formato CSV.'''

        metadata = [row[1] for row in current_data_block[:15]]
        scan_data = [[float(num.replace(',', '.')) for num in sublist] for sublist in current_data_block[18:]]
        scan_data = np.array(scan_data, dtype=float)[:,[1, 0, 2, 5]]  # X, Y, Z, Ratio (relative_dose)

        curve.type = 'M'
        curve.date = normalize_date(metadata[0].split()[0])
        curve.time = metadata[0].split()[1]

        if 'Platform' in metadata[1]:
            curve.machine = 'P'
        elif 'Synergy' in metadata[1]:
            curve.machine = 'S'
        else:
            curve.machine = 'ERROR'

        curve.particle = 0 if metadata[2].split()[1] == 'MV' else 1       # 1 = electrones    0 = fotones
        curve.energy = float(metadata[2].split()[0])
        curve.detector = metadata[5].split('(')[0].strip().replace(" ", "")
        curve.SSD = float(metadata[8].split()[0])/10  #cm

        if metadata[9].split()[0] == metadata[9].split()[2]:
            curve.field_size = float(metadata[9].split()[0])/10 #cm    # solo cuadrado y simetrico
        else:
            curve.field_size = None    # TC no es cuadrado, no se procesa

        coordinate_scant_type_dict = {'Crossline': 'X', 'Inline': 'Y', 'Beam': 'Z'}
        curve.coordinate = coordinate_scant_type_dict[metadata[12].strip()]

        curve.depth = '' if curve.coordinate == 'Z' else round(float(scan_data[0,2]), 1)   #mm

        if curve.coordinate == 'X':         
            curve.axis = scan_data[:,0]     
        elif curve.coordinate == 'Y':
            curve.axis = scan_data[:,1]
        else:
            curve.axis = scan_data[:,2]

        curve.relative_dose = scan_data[:,3]

    def CSV_data_importing_parsing(fname: str, curve_set: List[str]) -> None:
        '''Cosmetica de parsing para archivo CSV.'''

        current_data_block = []
        with open(fname, 'r') as file:
            for line in file:
                # Eliminar cualquier espacio en blanco extra (incluyendo saltos de línea)
                stripped_line = line.strip().split(";")
                
                # Comenzar un nuevo bloque si la línea comienza con "Measurement time:"
                if stripped_line[0] == "Measurement time:":
                    current_data_block = []
                    curve = Curve()
                
                # Agregar la línea al bloque actual
                current_data_block.append(stripped_line)
                
                # Si la línea está vacía y la línea anterior también lo estaba, cerrar el bloque
                if stripped_line[0] == '' and (len(current_data_block) > 1 and current_data_block[-2][0] == ''):

                    current_data_block = current_data_block[:-2]   #elimino los elementos sin nada del final
                    
                    initialize_curve_from_CSV_data(curve, current_data_block)
                    # print_curve_data(curve)
                    if curve.field_size != None:
                        curve_set.append(curve)

                    current_data_block = []

    curve_set = []
    # ASCII_data_importing_parsing(fname, curve_set)
    CSV_data_importing_parsing(fname, curve_set)

    return curve_set
    
def import_calculated_monaco_ref_data(monaco_folder):
    ref_data_paths = [str(path) for path in Path(monaco_folder).rglob('*') if path.is_file()]

    ref_curves = []
    for path in ref_data_paths:
        ref_curves.extend(import_calculated_curves(path))

    return ref_curves

def import_calculated_curves(monaco_calculation_file_path):

    def get_energy_from_filename(file_name):
        electron_energies = ['06', '09', '12', '15', '18']
        file_extension_index = int(file_name.split('.')[::-1][0]) - 1
        return float(electron_energies[file_extension_index])

    data = np.genfromtxt(monaco_calculation_file_path, dtype=str, delimiter='\n', skip_header=0, skip_footer=0)
    metadata = data[:16]
    data = data[16:]

    curves = []
    curve1 = Curve()
    curve1.type = 'C'
    curve1.date = ''
    curve1.time = ''
    curve1.detector = 'N/A'
    curve1.depth = ''

    # ['BLUEoELECoPLAT.ELECoPLAT.Coronal.216.00.4', '06x06', 'DFS 100', 'ELECTRONES', 'PLATFORM', 'MONACO', 'INTECNUS---RoyQA', 'Roy', '8 - Físicos Médicos', 'P:']

    try:
        splitted_path = monaco_calculation_file_path.split("\\")[::-1][:5]
        file_name, curve1.field_size, curve1.SSD, curve1.energy, curve1.machine = splitted_path
        
        if 'ELECTRONES' in curve1.energy:
            curve1.particle = 1   #electrones     
            curve1.energy = get_energy_from_filename(file_name)

            
            
            assert len(splitted_path) == 5
        else:
            curve1.particle = 0   #fotones
            curve1.energy = float(curve1.energy[1:])  #X06 -> 06
            
            assert len(splitted_path) == 5

    except ValueError:
        # This block runs if the unpacking fails due to a mismatch in the number of elements
        print("Unpacking failed: The number of elements in the string does not match the expected.")
        return []
    
    except AssertionError:
        # This block runs if the assertion fails, meaning there are not enough elements
        print("Error: Not enough elements to unpack.")
        return []

    if 'SYNERGY' in curve1.machine:
        curve1.machine = 'S'
    elif 'PLATFORM' in curve1.machine:
        curve1.machine = 'P'
    else:
        'Error al identificar la maquina en los datos de referencia de Monaco.'
        
    curve1.SSD = float(curve1.SSD.split(' ')[1])
    curve1.field_size = float(curve1.field_size.split('x')[0])

    dose_plane_type = file_name.split('.')[2]

    dose2D = np.array([row.split(',') for row in data]).astype('float32')
 
    # print(dose2D.shape)   #163
    # plt.plot(dose2D[163:200,dose2D.shape[1]//2])
    # plt.show()

    # plt.imshow(dose2D, cmap='viridis')  # cmap is optional; it defines the color map
    # plt.colorbar()  # Optional: adds a color bar to the side
    # plt.show()

    upper_left_point = [ float(metadata[12].split(',')[1]) , float(metadata[12].split(',')[2])  ] #mm
    dose2D_size = [ int(metadata[14].split(',')[1]) , int(metadata[14].split(',')[2])  ] #mm
    spatial_resolution = float(metadata[15].split(',')[-1])  #mm

    # print(dose2D.shape)
    hor_axis = np.arange(int(upper_left_point[0]), dose2D_size[0] + int(upper_left_point[0]), spatial_resolution)
    vert_axis = np.arange(int(upper_left_point[1]), int(upper_left_point[1]) - dose2D_size[1], -spatial_resolution)
    
    vert_mid_point_idx  = np.where(vert_axis == 0)[0][0]
    hor_mid_point_idx = np.where(hor_axis == 0)[0][0]

    # La posicion del origen del fantoma cambia segun donde se definio el isocentro en Monaco,
    # esta funcion corrige si no esta en SSD = 100 cm
    # requiere que la SSD este en cm y los ejes en mm.
    z_axis_correction_based_on_setup = lambda ssd: 10.0*ssd-1000.0

    if dose_plane_type == 'Coronal':
        #Depth:
        curve1.depth = round(250.0 - float(metadata[3].split()[1])*10.0, 1)  #mm

        print('Para Synergy, se exportaron planos con profundidades levemente distintas'+
        'para los diferentes conos. '+
        'Se suma 0.5 mm al valor de profundidad de dosis de SOLO esos perfiles de Synergy en cono 6x6'+
        'y se resta 0.5 mm en cono 20x20 para que puedan hacer match'+
        'con los perfiles medidos en esas profundidades especificas de R85: 18.3 mm, 28.1 mm, 39.4 mm, 47.4 mm.')
        syn_R85_depths_6x6 = [17.8, 27.6, 38.9, 46.9] #mm
        yn_R85_depths_20x20 = [18.8, 28.6, 39.9, 47.9] #mm

        if curve1.machine == 'S' and curve1.particle == 1 and curve1.depth in syn_R85_depths_6x6:
            curve1.depth += 0.5 #mm
        if curve1.machine == 'S' and curve1.particle == 1 and curve1.depth in yn_R85_depths_20x20:
            curve1.depth -= 0.5 #mm
        
        
        # Aca creo dos perfiles porque el txt tiene la informacion de dos curvas 
        # segun si se exporto coronal o transversal

        # Cargo coordinate, axis y relative_dose del perfil en X
        curve1.coordinate    = 'X' 
        curve1.axis          = hor_axis
        curve1.relative_dose = dose2D[vert_mid_point_idx, :].squeeze()

        curves.append(curve1)

        # Cargo coordinate, axis y relative_dose del perfil en Y
        curve2 = copy.copy(curve1)

        curve2.coordinate    = 'Y' 
        curve2.axis          = vert_axis[::-1]
        curve2.relative_dose = dose2D[:, hor_mid_point_idx][::-1].squeeze()

        curves.append(curve2)

    if dose_plane_type == 'Transverse':
        # Cargo coordinate, axis y relative_dose del PDD
        curve1.depth = 0.0

        curve1.coordinate    = 'Z' 

        # print('VERTAXIS: ', vert_axis[0], vert_axis[-1], len(vert_axis))

        curve1.axis          = vert_axis + z_axis_correction_based_on_setup(curve1.SSD) 
        idx = np.where(curve1.axis == 0)[0][0]
        curve1.axis          = (- curve1.axis)[idx:] + 0.5  #mm  #error de posicionamiento de iso SSD=99.95 cm
        curve1.relative_dose = dose2D[:, hor_mid_point_idx].squeeze()[idx:]

        curves.append(curve1)   

    # for curve in curves:
    #     print_curve_data(curve)
        # print(curve.axis.shape, curve.relative_dose.shape)SuavizaYNormaliza

    return curves

def import_Monaco_calculated_curves_from_database():

    # Replace with your database connection details
    engine = create_engine(monaco_database_file_path)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Fetch all records
    curve_records = session.query(CurveModel).all()

    # List to store Curve objects
    curves_list = []

    for record in curve_records:
        # Convert PickleType columns to numpy arrays if needed
        axis_array = record.axis  # Ensure this is in the correct format (numpy array)
        relative_dose_array = record.relative_dose  # Ensure this is in the correct format (numpy array)

        # Create Curve object
        curve = Curve(
            id=record.id,
            type=record.type,
            coordinate=record.coordinate,
            date=record.date,
            time=record.time,
            depth=record.depth,
            SSD=record.SSD,
            field_size=record.field_size,
            axis=axis_array,
            relative_dose=relative_dose_array,
            machine=record.machine,
            particle=record.particle,
            energy=record.energy,
            detector=record.detector
        )
        
        # Add to the list
        curves_list.append(curve)

    # Close the session
    session.close()

    # for curve in curves_list: print_curve_data(curve)

    return curves_list

def print_number_of_curves_in_DB():
    # Replace with your database connection details
    engine = create_engine(monaco_database_file_path)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Using ORM to count the number of rows in the CurveModel table
    row_count = session.query(func.count(CurveModel.id)).scalar()

    print(f"Número total de curvas almacenadas en la base de datos: {row_count}")

    # Close the session
    session.close()

def get_R50(curve):
    if curve.particle == 1 and curve.coordinate == 'Z':
        if curve.relative_dose.max() > 105.0:
            print('Curva no normalizada.')
        f = scipy.interpolate.interp1d(curve.relative_dose, curve.axis, fill_value='extrapolate')
        return f(50.0)
    else: 
        print('Error')
        return 0

def Interpola(Curva, step):
    #interpola la curva para que todos tengan ejes en comun
    f = scipy.interpolate.interp1d(Curva.axis, Curva.relative_dose, fill_value='extrapolate')

    if(Curva.coordinate=='X' or Curva.coordinate=='Y'):
        axis = np.concatenate( (np.arange(-step,Curva.axis.min(),-step)[::-1] , np.arange(0,Curva.axis.max(),step)) , axis=0)
    elif(Curva.coordinate=='Z'):
        axis = np.arange(0,Curva.axis.max(),step)

    return axis,f(axis)

def SuavizaYNormaliza(curva, sigma, espaciamiento):
    curva.axis, curva.relative_dose = Interpola(curva, espaciamiento)

    if curva.particle == 1 and curva.type == 'C':
        sigma *= 20

    #suaviza con un filtro gaussiano
    filtered_curve = gaussian_filter1d(curva.relative_dose, sigma)

    #normaliza al cax (perfiles) o al maximo (PDD)
    if curva.type == 'C':
        if curva.particle == 0 and curva.field_size <= 1.5:  #campos pequenos fotones
            epsilon = 3    # +- 0.3 cm del cax
        elif curva.particle == 1 and curva.coordinate == 'Z':  #pdd electrones
            epsilon = 3   # +- 0.3 cm del max
        elif curva.field_size >= 10.0 and curva.coordinate != 'Z':   #perfiles campos grandes
            epsilon = 20    # +- 2 cm del cax
        else:
            epsilon = 3
    else:
        epsilon = 0

    n = int(epsilon/espaciamiento)

    if(curva.coordinate == 'X' or curva.coordinate == 'Y'):
        norm_idx = np.where(curva.axis == 0)[0][0]
    else:
        norm_idx = filtered_curve.argmax()

    norm_value = filtered_curve[norm_idx-n:norm_idx+n].mean() if n!=0 else filtered_curve[norm_idx]
    curva.relative_dose = filtered_curve/norm_value*100.0
    
def AnalizaRegionesPerfil(Curva,Curva_ref):
    MAX = min(Curva.axis.max(),Curva_ref.axis.max())
    MIN = max(Curva.axis.min(),Curva_ref.axis.min())

    Curva_ref.relative_dose = Curva_ref.relative_dose[(Curva_ref.axis >= MIN) & (Curva_ref.axis <= MAX)]
    Curva_ref.axis = Curva_ref.axis[(Curva_ref.axis >= MIN) & (Curva_ref.axis <= MAX)]

    Curva.relative_dose = Curva.relative_dose[(Curva.axis >= MIN) & (Curva.axis <= MAX)]
    Curva.axis = Curva.axis[(Curva.axis >= MIN) & (Curva.axis <= MAX)]


    penumbra_izq = (Curva.relative_dose <= 90) & (Curva.relative_dose >= 10) & (Curva.axis < 0)
    penumbra_der = (Curva.relative_dose <= 90) & (Curva.relative_dose >= 10) & (Curva.axis > 0)

    umbra_izq = (Curva.relative_dose < 10) & (Curva.axis < 0)
    umbra_der = (Curva.relative_dose < 10) & (Curva.axis > 0)

    plateau = (Curva.relative_dose < 90)

    #ANALISIS DE DIFERENCIA EN DOSIS

    max_D_plateau = round( (Curva.relative_dose[plateau]-Curva_ref.relative_dose[plateau]).max() ,1)
    min_D_plateau = round( (Curva.relative_dose[plateau]-Curva_ref.relative_dose[plateau]).min() ,1)
    
    max_D_umbra_izq = round( (Curva.relative_dose[umbra_izq]-Curva_ref.relative_dose[umbra_izq]).max() ,1)
    min_D_umbra_izq = round( (Curva.relative_dose[umbra_izq]-Curva_ref.relative_dose[umbra_izq]).min() ,1)

    max_D_umbra_der = round( (Curva.relative_dose[umbra_der]-Curva_ref.relative_dose[umbra_der]).max() ,1)
    min_D_umbra_der = round( (Curva.relative_dose[umbra_der]-Curva_ref.relative_dose[umbra_der]).min() ,1)

    #ANALISIS DE DIFERENCIA EN DISTANCIA

    f_izq = scipy.interpolate.interp1d(Curva.relative_dose[penumbra_izq], Curva.axis[penumbra_izq])
    f_ref_izq = scipy.interpolate.interp1d(Curva_ref.relative_dose[penumbra_izq], Curva_ref.axis[penumbra_izq])

    f_der = scipy.interpolate.interp1d(Curva.relative_dose[penumbra_der], Curva.axis[penumbra_der])
    f_ref_der = scipy.interpolate.interp1d(Curva_ref.relative_dose[penumbra_der], Curva_ref.axis[penumbra_der])

    axis = np.arange(15,85,5)

    max_dta_penumbra_izq = round( (f_izq(axis)-f_ref_izq(axis) ).max() ,1)
    min_dta_penumbra_izq = round( (f_izq(axis)-f_ref_izq(axis) ).min() ,1)
    max_dta_penumbra_der = round( (f_der(axis)-f_ref_der(axis) ).max() ,1)
    min_dta_penumbra_der = round( (f_der(axis)-f_ref_der(axis) ).min() ,1)

    return min_D_umbra_izq, max_D_umbra_izq, min_dta_penumbra_izq, max_dta_penumbra_izq, min_D_plateau, max_D_plateau, min_dta_penumbra_der, max_dta_penumbra_der, min_D_umbra_der, max_D_umbra_der

def ProcesaCurvas(Curvas_med,Curvas_cal,espaciamiento,sigma_suavizado):
    print('Se cuadruplica valor de sigma en electrones!')
    for curva in Curvas_med:
        SuavizaYNormaliza(curva,sigma_suavizado, espaciamiento)
    for curva in Curvas_cal:
        SuavizaYNormaliza(curva,sigma_suavizado, espaciamiento)

    return Curvas_med, Curvas_cal

def Analiza_Y_Grafica_Perfiles(Perfiles_med,Perfiles_cal,dose_threshold,dta,cutoff, COMPARE_ANYWAY=False):
    idx_med = np.arange( len(Perfiles_med) )
    idx_cal = np.arange( len(Perfiles_cal) )

    plot_data = []

    for i,j in itertools.product( idx_med , idx_cal ): 
        if(COMPARE_ANYWAY):
            datos = []
            # datos = AnalizaRegionesPerfil(Perfiles_med[i],Perfiles_cal[j])
            Grafica_Perfiles(Perfiles_med[i],Perfiles_cal[j],datos,i,dose_threshold,dta,cutoff) 
        elif( 
            Perfiles_med[i].machine          == Perfiles_cal[j].machine and 
            Perfiles_med[i].particle         == Perfiles_cal[j].particle and 
            Perfiles_med[i].energy           == Perfiles_cal[j].energy and
            Perfiles_med[i].SSD              == Perfiles_cal[j].SSD and 
            Perfiles_med[i].field_size       == Perfiles_cal[j].field_size and 
            Perfiles_med[i].depth            == Perfiles_cal[j].depth and
            Perfiles_med[i].coordinate       == Perfiles_cal[j].coordinate):
            datos = []

            print('Match!')
         
            fig, dose_threshold, dta, passing_percentage = Grafica_Perfiles(Perfiles_med[i],Perfiles_cal[j],datos,i,dose_threshold,dta,cutoff)
            plot_data.append([Perfiles_med[i], fig, dose_threshold, dta, passing_percentage])
    
    Save_results_to_PDF(plot_data)
        
def Analiza_Y_Grafica_PDDs(PDDs_med,PDDs_cal,dose_threshold,dta,cutoff, COMPARE_ANYWAY=False):
    idx_med = np.arange( len(PDDs_med) )
    idx_cal = np.arange( len(PDDs_cal) )

    plot_data = []

    for i,j in itertools.product( idx_med , idx_cal ):  
        if(COMPARE_ANYWAY):
            Grafica_PDDs(PDDs_med[i],PDDs_cal[j],i,dose_threshold,dta,cutoff)  
        elif(
            PDDs_med[i].machine    == PDDs_cal[j].machine and 
            PDDs_med[i].particle   == PDDs_cal[j].particle and
            PDDs_med[i].energy     == PDDs_cal[j].energy and 
            PDDs_med[i].SSD        == PDDs_cal[j].SSD and 
            PDDs_med[i].field_size == PDDs_cal[j].field_size and
            PDDs_med[i].coordinate == PDDs_cal[j].coordinate):

            print('Match!')

            fig, dose_threshold, dta, passing_percentage = Grafica_PDDs(PDDs_med[i],PDDs_cal[j],i,dose_threshold,dta,cutoff)
            plot_data.append([PDDs_med[i], fig, dose_threshold, dta, passing_percentage])
    
    Save_results_to_PDF(plot_data)
        
def Grafica_Perfiles(Perfiles_med, Perfiles_cal, datos, i, dose_threshold, dta, cutoff):
    # Create a Toplevel window
    # ventana_grafico = tk.Toplevel()
    # ventana_grafico.withdraw()

    # Create a single Figure instance
    fig  = Figure(figsize=(9, 6), dpi=250)
    ax1 = fig.add_subplot()
    ax1.set_ylabel('Porcentaje [%]', color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Gamma Index ' + str(dose_threshold) + '% ' + str(dta) + 'mm', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()  # otherwise, the right y-label is slightly clipped

    gamma = pymedphys.gamma(Perfiles_cal.axis, Perfiles_cal.relative_dose, Perfiles_med.axis, Perfiles_med.relative_dose,
                            dose_percent_threshold=dose_threshold, distance_mm_threshold=dta,
                            lower_percent_dose_cutoff=cutoff, quiet=True)
    
    x_min, x_max = Perfiles_med.axis[0], Perfiles_med.axis[-1]
    i_min, i_max = np.where(Perfiles_cal.axis == x_min)[0][0], np.where(Perfiles_cal.axis == x_max)[0][0]
    reduced_gamma = gamma[i_min:i_max]

    passing_points = np.count_nonzero(reduced_gamma <= 1)
    valid_points = np.count_nonzero(~np.isnan(reduced_gamma))
    passing_percentage = round((passing_points / valid_points) * 100,1)

    # Plot on the same axes
    ax1.plot(Perfiles_cal.axis, Perfiles_cal.relative_dose, marker='.', markersize=0, linewidth=1, label="Monaco",
             color='b')
    ax1.plot(Perfiles_med.axis, Perfiles_med.relative_dose, marker='.', markersize=2, linewidth=0, label=str(Perfiles_med.date),
             color='g')
    ax2.plot(Perfiles_cal.axis, 
             gamma, 
             marker='+', 
             markersize=0, 
             linewidth=0.5,
             label='Gamma ' + str(dose_threshold) + '% ' + str(dta) + 'mm -> '+str(round(passing_percentage, 1))+'%', 
             color='r')
    ax2.axhline(y=1, color='r', linewidth=0.5)

    machine_labeler = lambda curve: 'SYN' if curve.machine=='S' else 'PLA'
    energy_labeler = lambda curve: f'{curve.energy} MV' if curve.particle==0 else f'{curve.energy} MeV'

    ax1.set_title(machine_labeler(Perfiles_med) + 
                  ' - QA ANUAL ' + str(Perfiles_med.date[6:]) + 
                  ' - Perfil en ' + str(Perfiles_med.coordinate) +
                  ' - E ' + energy_labeler(Perfiles_med) +
                  ' - DFS ' + str(Perfiles_med.SSD) +
                  ' cm - TC ' + str(Perfiles_med.field_size) +
                  ' cm2 - Prof: ' + str(round(Perfiles_med.depth/10.0,2)) +     #cm
                  ' cm - ' + str(Perfiles_med.detector),
                  fontsize=7)

    ax1.set_xlim([max(Perfiles_med.axis.min(), Perfiles_cal.axis.min()),
                  min(Perfiles_med.axis.max(), Perfiles_cal.axis.max())])

    ax1.set(xlabel='axis ' + str(Perfiles_med.coordinate) + '(mm)', ylabel='Dosis (%)')
    legend1 = ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax1.grid()

    ax2.set_ylim([0, 5])
    legend2 = ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.85))

    # # Adjust the legends so they do not overlap
    # plt.gca().add_artist(legend1)

    # # Use the FigureCanvasTkAgg constructor with the existing figure and window
    # canvas = FigureCanvasTkAgg(fig, ventana_grafico)
    # toolbar = NavigationToolbar2Tk(canvas, ventana_grafico)
    # canvas._tkcanvas.pack()
    # fig.savefig('Resultados/Perfil'+str(i)+'.pdf', format='pdf', orientation='landscape', bbox_inches='tight')

    return fig, dose_threshold, dta, passing_percentage
    
def Grafica_PDDs(PDDs_med,PDDs_cal,i,dose_threshold,dta,cutoff):
    # ventana_grafico = tk.Toplevel()

    fig  = Figure(figsize=(9, 6), dpi=250)
    ax1  = fig.add_subplot()
    ax1.set_ylabel('Porcentaje [%]', color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Gamma Index '+str(dose_threshold)+'% '+str(dta)+'mm', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    gamma = pymedphys.gamma(PDDs_cal.axis, PDDs_cal.relative_dose, PDDs_med.axis, PDDs_med.relative_dose,
                            dose_percent_threshold=dose_threshold, distance_mm_threshold=dta, 
                            lower_percent_dose_cutoff=cutoff, quiet=True)
    
    x_min, x_max = PDDs_med.axis[0], PDDs_med.axis[-1]
    i_min, i_max = np.where(PDDs_cal.axis == x_min)[0][0], np.where(PDDs_cal.axis == x_max)[0][0]
    reduced_gamma = gamma[i_min:i_max]

    passing_points = np.count_nonzero(reduced_gamma <= 1)
    valid_points = np.count_nonzero(~np.isnan(reduced_gamma))
    passing_percentage = round((passing_points / valid_points) * 100,1)

    ax1.plot(PDDs_cal.axis, PDDs_cal.relative_dose, marker='.', markersize=0, linewidth=1, label="Monaco", color='b')
    ax1.plot(PDDs_med.axis, PDDs_med.relative_dose, marker='.', markersize=2, linewidth=0, label=str(PDDs_med.date), color='g')
    ax2.plot(PDDs_cal.axis, 
             gamma, 
             marker='+', 
             markersize=0, 
             linewidth=0.5, 
             label='Gamma '+str(dose_threshold)+'% '+str(dta)+' -> '+str(round(passing_percentage, 1))+'%', 
             color='r')
    ax2.axhline(y = 1 , color='r', linewidth=0.5)

    machine_labeler = lambda curve: 'SYN' if curve.machine=='S' else 'PLA'
    energy_labeler = lambda curve: f'{curve.energy} MV' if curve.particle==0 else f'{curve.energy} MeV'

    ax1.set_title(machine_labeler(PDDs_med)+
                ' - QA ANUAL '+ str(PDDs_med.date[6:])+
                ' - PDD '+
                ' - E '+ energy_labeler(PDDs_med)+
                ' - DFS '+ str(PDDs_med.SSD) +
                ' cm - TC '+ str(PDDs_med.field_size)+
                ' cm2 - ' + PDDs_med.detector, 
                    fontsize=7)

    ax1.set_xlim([max(PDDs_med.axis.min(),PDDs_cal.axis.min()), min(PDDs_med.axis.max(),PDDs_cal.axis.max())])
    ax1.set(xlabel='Profundidad (mm)', ylabel='Dosis (%)')
    legend1 = ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax1.grid()

    ax2.set_ylim([0, 5])
    legend2 = ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.85))
        
    # canvas  = FigureCanvasTkAgg(fig,ventana_grafico)
    # toolbar = NavigationToolbar2Tk(canvas,ventana_grafico)
    # canvas._tkcanvas.pack()
    # fig.savefig('Resultados/Pdd'+str(i)+'.pdf', orientation='landscape', format='pdf', bbox_inches='tight')

    return fig, dose_threshold, dta, passing_percentage
    
def Save_results_to_PDF(plot_data):
    def _update_dataframe_info_from_curve(df, curve, dose_threshold, dta, passing_percentage):
        # Create a dictionary from the curve's attributes, excluding numpy arrays
        curve_dict = {k: v for k, v in curve.__dict__.items() if not isinstance(v, np.ndarray)}
        
        # Add additional columns
        curve_dict['dose_threshold_%'] = dose_threshold
        curve_dict['dta_mm'] = dta
        curve_dict['pass_value_%'] = passing_percentage
        
        # Create a DataFrame for the new curve
        curve_df = pd.DataFrame([curve_dict])
        
        # Concatenate the new DataFrame with the existing one
        if df.empty:
            df = curve_df
        else:
            df = pd.concat([df, curve_df], ignore_index=True)
        
        return df
    
    def _set_style():
        styles = getSampleStyleSheet()
        
        if 'Title' not in styles:
            styles.add(ParagraphStyle(name='Title', fontSize=14, fontName='Helvetica-Bold', spaceAfter=16))
        
        if 'Content' not in styles:
            styles.add(ParagraphStyle(name='Content', fontSize=12, fontName='Helvetica', spaceAfter=12))
        
        return styles

    def _add_results_table_to_pdf(pdf_elements, df):
        df = df.copy()
        
        df['machine'].replace({'S': 'Synergy', 'P': 'Platform'}, inplace=True)
        df['particle'].replace({0: 'Fotones', 1: 'Electrones'}, inplace=True)

        dummy_dict = df.loc[:, df.nunique() == 1].iloc[0].to_dict()   #detecto aquellas columnas que todos los valores son iguales
        
        df.drop(['id', 'type', 'time', 'dose_threshold_%', 'dta_mm'], axis=1, inplace=True)

        new_order = ['machine', 'date', 'detector', 'particle', 'energy', 'SSD', 'field_size', 'coordinate', 'depth', 'pass_value_%']
        df = df[new_order]

        df.loc[:, 'Tolerancia'] = 95.0

        df.rename(columns={
            'coordinate': 'Eje',
            'date': 'Fecha',
            'machine': 'Equipo',
            'depth': 'Prof_mm',
            'SSD': 'SSD_cm',
            'field_size': 'TC_cm',
            'particle': 'Partícula',
            'energy': 'E_MV/MeV',
            'detector': 'Detector/SN',
            'pass_value_%': 'Gamma_pass',
            }, inplace=True)

        # print(df.columns)

        # ------------------------------    PDF   ----------------------------------

        try:
            # Intentar extraer el año del diccionario dummy_dict
            year = dummy_dict['date'].split('/')[-1]
        except KeyError:
            # En caso de que no esté en dummy_dict, extraerlo del dataframe original (caso de proyecto con mediciones de diferentes dias)
            year = df['Fecha'].iloc[0].split('/')[-1]

        if  'S' in dummy_dict['machine']:
            machine = 'SYNERGY FULL SN:153868'
        else:
            machine = 'SYNERGY PLATFORM SN:153870'
        
        Dta = dummy_dict['dose_threshold_%']
        dta = dummy_dict['dta_mm']

        styles = _set_style()

        title = Paragraph(f'DOSIMETRIA ANUAL RELATIVA DE PDDS Y PERFILES DE DOSIS {year}', styles['Title']) 
        pdf_elements.append(title)
        title = Paragraph(f'ACELERADOR ELEKTA {machine}', styles['Title']) 
        pdf_elements.append(title)


        title = Paragraph(f'Equipamiento utilizado: Fantoma Automático IBA BluePhantom2 SN:15800. Electrómetro IBA CCU SN:23261.', styles['Content'])
        pdf_elements.append(title)
        title = Paragraph(f'Dosis medida comparada con la dosis calculada en TPS Monaco: ', styles['Content'])
        pdf_elements.append(title)
        title = Paragraph(f'Tamaño de grilla: 0.3 cm (fotones) - 0.1 cm (campos pequeños/electrones).', styles['Content'])
        pdf_elements.append(title)
        title = Paragraph(f'Incerteza estadistica: 0.1% (fotones) - Nro de historias/cm2: 1M (electrones).', styles['Content'])
        pdf_elements.append(title)
        title = Paragraph(f'Se realiza análisis Gamma Index con parámetros: Dosis de acuerdo: {Dta} % - Distancia de acuerdo: {dta} mm', styles['Content'])
        pdf_elements.append(title)

        # Convert DataFrame to a list of lists for reportlab
        data = [df.columns.tolist()]  # Header row
        data.extend(df.values.tolist())  # Data rows

        # Create a Table with the data
        table = Table(data)

        # Style the table
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])

        # Aplica color a la celda basado en condicion
        gamma_col_index = df.columns.get_loc('Gamma_pass')
        for i, row in enumerate(df.itertuples(), start=1):
            gamma_value = getattr(row, 'Gamma_pass')
            if gamma_value > 99.9:
                table_style.add('BACKGROUND', (gamma_col_index, i), (gamma_col_index, i), colors.lightblue)
            if gamma_value <= 99.9 and gamma_value >= 95.0:
                table_style.add('BACKGROUND', (gamma_col_index, i), (gamma_col_index, i), colors.lightgreen)
            if gamma_value < 95.0 and gamma_value >= 80.0:
                table_style.add('BACKGROUND', (gamma_col_index, i), (gamma_col_index, i), colors.gold)
            if gamma_value < 80.0 and gamma_value >= 50.0:
                table_style.add('BACKGROUND', (gamma_col_index, i), (gamma_col_index, i), colors.orange)
            if gamma_value < 50.0:
                table_style.add('BACKGROUND', (gamma_col_index, i), (gamma_col_index, i), colors.red)

        table.setStyle(table_style)

        # Add the table to the elements
        pdf_elements.append(table)

    def _add_plot_figures_to_pdf(pdf_elements, fig):
        """Agrega figuras de gráficos al PDF con márgenes adecuados y preservación de la relación de aspecto."""
 
        a4_width, a4_height = landscape(A4)
        a4_width, a4_height = 685, 439   #tamano del frame
        
        # Defino los márgenes
        margin_left = 0.0 * inch
        margin_right = 0.0 * inch
        margin_top = 0.0 * inch
        margin_bottom = 0.0 * inch
        
        content_width = a4_width - (margin_left + margin_right)
        content_height = a4_height - (margin_top + margin_bottom)

        temp_image_path = f"temp.png"

        # Save figure as a PNG file
        fig.savefig(temp_image_path, format='png')

        # Create an Image object from the saved PNG file
        img = Image(temp_image_path)
        
        # Obtengo las dimensiones originales de la imagen
        img_width, img_height = img.wrap(0, 0)  # Obtengo el tamaño original sin márgenes
        
        # Calculo el factor de escalado para ajustar la imagen dentro del área de contenido manteniendo la relación de aspecto
        scaling_factor = min(content_width / img_width*0.98, content_height / img_height*0.98)

        img.drawWidth *= scaling_factor
        img.drawHeight *= scaling_factor

        img.hAlign = 'CENTER'
        img.vAlign = 'MIDDLE'

        os.remove(temp_image_path)
        
        # Añado la imagen a la lista de elementos del PDF
        pdf_elements.append(img)
        
    
    # --------------------------------------------------------------------------------------------

    df = pd.DataFrame(columns=['Attribute', 'Value', 'dose_threshold_%', 'dta_mm', 'pass_value_%'])
    figures = []
    pdf_elements = []

    # obtener fecha y hora para generar nombre de archivo de resultados
    current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    os.makedirs('Resultados', exist_ok=True)
    pdf = SimpleDocTemplate(f'Resultados/results_{current_date_time}.pdf', pagesize=landscape(A4))

    #extrigo la informacion de las curvas para generar el reporte
    for curve, fig, dose_threshold, dta, passing_percentage in plot_data:
        df = _update_dataframe_info_from_curve(df, curve, dose_threshold, dta, passing_percentage)
        figures.append(fig)
        
    if df.empty: 
        print('No hay resultados en Dataframe de resultados')
        return

    _add_results_table_to_pdf(pdf_elements, df)

    for fig in figures: _add_plot_figures_to_pdf(pdf_elements, fig)

    # Build the PDF
    pdf.build(pdf_elements)

        