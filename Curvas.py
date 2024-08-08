import numpy as np
import scipy.interpolate
from scipy.ndimage import gaussian_filter1d
import itertools
import pymedphys
from typing import List

from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import filedialog, Text
import os
import copy

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------------------------

class Curve():
    def __init__(self):
        type = ''     #medido o calculado
        coordinate = ''    # a lo largo de X,Y,Z
        date = ''
        time = ''
        depth = ''
        SSD = ''
        field_size = np.array([])      #[Y,X] rectangular y simetrico
        axis = np.array([])
        relative_dose = np.array([])
        machine = ''
        particle = 0
        energy = ''

def Convert(string): 
    li = list(string.split(" ")) 
    return li 

def print_curve_data(curve):
        print(f'Equipo:      {curve.machine}, '
              f'Fecha:       {curve.date}, '
              f'Hora:        {curve.time}, '
              f'Particula:   {curve.particle}, '
              f'Energia:     {curve.energy}, '
              f'DFP:         {curve.SSD}, '
              f'TC:          {curve.field_size}, '
              f'Coordenada:  {curve.coordinate}, '
              f'Profundidad: {curve.depth}')

# def IngresaPerfilMedido(fname):

#     a = np.genfromtxt(fname,dtype=str, delimiter='\n', skip_header=1, skip_footer=1)

#     fname = os.path.split(fname)[1].split('.')[0]

#     l = []
#     for s in a: 
#         s = s.replace('<','')
#         s = s.replace('>','')
#         s = s.replace('+','')
#         l.append(Convert(s))

#     curvas = []

#     for e in l:
#         if e[0] == '$STOM':
#             aux = []
#             continue
#         if e[0] == '$ENOM': 
#             curvas.append(aux)
#             continue
#         else: aux.append(e)

#     Perfiles = []

#     print('\n'+'Datos de perfiles medidos ingresados:')

#     for i in range(len(curvas)):
#         perfil = Curva()

#         perfil.coordinate  = str(curvas[i][6][1])
#         perfil.fecha       = str(curvas[i][1][1])
#         perfil.depth = str(curvas[i][10][1])
#         perfil.dfp         = str(curvas[i][9][2])
#         perfil.tc          = str(curvas[i][4][1])
#         perfil.energia     = fname.split(' ')[1].upper()
#         perfil.equipo      = fname.split(' ')[0].upper()
        
#         perfil.relative_dose       = np.array(curvas[i][11:], dtype=float)[:,3]

#         if(perfil.coordinate=='X'):
#             perfil.axis     = np.array(curvas[i][11:], dtype=float)[:,0]
#         else:
#             perfil.axis     = np.array(curvas[i][11:], dtype=float)[:,1]

#         Perfiles.append(perfil)

#     print('Datos de dosis ingresada:')
    
#     for perfil in Perfiles:
#         print('Equipo: %s, DFP: %s, TC: %s, Z: %s, axis: %s, Energía: %s' % (perfil.equipo, perfil.dfp, perfil.tc, perfil.depth, perfil.coordinate, perfil.energia) )

#     return Perfiles

# def IngresaPDDMedido(fname):

#     a = np.genfromtxt(fname,dtype=str, delimiter='\n', skip_header=1, skip_footer=1)

#     fname = os.path.split(fname)[1].split('.')[0]

#     l = []
#     for s in a: 
#         s = s.replace('<','')
#         s = s.replace('>','')
#         s = s.replace('+','')
#         l.append(Convert(s))

#     curvas = []

#     for e in l:
#         if e[0] == '$STOM':
#             aux = []
#             continue
#         if e[0] == '$ENOM': 
#             curvas.append(aux)
#             continue
#         else: aux.append(e)

#     PDDs = []

#     print('\n'+'Datos de pdds medidos ingresados:')

#     for i in range(len(curvas)):
#         PDD = Curva()

#         PDD.coordinate  = str(curvas[i][6][1])
#         PDD.fecha       = str(curvas[i][1][1])
#         PDD.depth = '-'
#         PDD.dfp         = str(curvas[i][9][2])
#         PDD.tc          = str(curvas[i][4][1])
#         PDD.energia     = fname.split(' ')[1].upper()
#         PDD.equipo      = fname.split(' ')[0].upper()
        
#         PDD.relative_dose       = np.array(curvas[i][11:], dtype=float)[:,3]

#         PDD.axis     = np.array(curvas[i][11:], dtype=float)[:,2]  #ACA VA 2 EN LUGAR DE 0 ASI AGARRA LA COLUMNA DE Z

#         PDDs.append(PDD)

#         print('Equipo: %s, DFP: %s, TC: %s, axis: %s, Energía: %s' % (PDD.equipo, PDD.dfp, PDD.tc, PDD.coordinate, PDD.energia) )

#     return PDDs

def import_measured_curves(fname, machine='Syn-Pla'):
    
    def coordinate_classifier(measurement_data):
        start_x, start_y, start_z = measurement_data[19][1:]
        end_x, end_y, end_z = measurement_data[20][1:]

        if(start_z == end_z):
            if(start_y == end_y):
                return 'X'
            elif(start_x == end_x):
                return 'Y'
            else:
                print('Perfil diagonal!')
        elif (start_x == end_x and start_y == end_y):
            if (start_x == '0.0' and start_y == '0.0'):
                return 'Z'
            else: 
                print('PDD no está en CAX.')
        else:
            print('No se puede determinar coordinate de curva.')
 
    def initialize_curve_from_data(curve: Curve, measurement_data: List[str]) -> Curve:
        '''Curve class atribute asignment based in measurent_data.'''
        curve.type = 'M'
        curve.coordinate = coordinate_classifier(measurement_data)    # a lo largo de X,Y,Z
        curve.date =       measurement_data[4][1]
        curve.time =       measurement_data[5][1]
        curve.machine = machine
        curve.particle = 1 if measurement_data[7][1] == 'ELE' else 0       # 1 = electrones    0 = fotones
        curve.energy = float(measurement_data[7][2])
        curve.depth = '' if curve.coordinate == 'Z' else float(measurement_data[21][3])   #mm
        curve.SSD = int(measurement_data[8][1])/10  #cm
        curve.field_size = int(measurement_data[6][1])/10     #[Y,X] solo cuadrado y simetrico

        scan_data = np.array(measurement_data[21:][:][1:])[:,1:].astype(float)

        curve.relative_dose = scan_data[:,3]

        if curve.coordinate == 'X':
            curve.axis = scan_data[:,0]
        elif curve.coordinate == 'Y':
            curve.axis = scan_data[:,1]
        else:
            curve.axis = scan_data[:,2]

    raw_data = np.genfromtxt(fname, dtype=str, delimiter='\n', skip_header=2, skip_footer=0)

    print('Metadata de  dosis medida ingresada:')

    # Cosmetica de parsing
    curve_set = []
    for s in raw_data: 
        s = [s.replace(' ', '') for s in s.split('\t')]
        
        if s[0] == '%VNR1.0':
            curve = Curve()
            measurement_data = []
            continue
        elif s[0] == ':EOM': 
            initialize_curve_from_data(curve, measurement_data)
            print_curve_data(curve)
            curve_set.append(curve)
        else: 
            measurement_data.append(s)

    return curve_set


def IngresaPerfilCalculadoMONACO(fname):
    a = np.genfromtxt(fname,dtype=str, delimiter='\n', skip_header=0, skip_footer=0)

    path_list = fname.split('/')

    fname = os.path.split(fname)[1]
    # BLUEx06oSYN.FotonesX06SYN.Coronal.50.00

    b = np.array( fname.split('/')[-1].rsplit('.', 1)[0].split('.') )
    #['BLUEx06oSYN' 'FotonesX06SYN' 'Coronal' '50' '00']

    Perfiles = []

    perfil = Curve()

    perfil.date       = a[2].split(',')[1]
    perfil.energy     = b[0].split('o')[0][4:].upper()
    perfil.SSD         = str(int(path_list[-3].split(' ')[-1])*10)  # Lo saca del path, importante carpetas

    tc = str(int(path_list[-2].split('x')[0])*10)
    if len(tc)<3:
        perfil.field_size          = '0'+tc+'*'+'0'+tc
    else:
        perfil.field_size          = tc+'*'+tc
    
    z = str(250 - int(b[3]))
    if len(z)<3:
        perfil.depth = '0'+z
    else:
        perfil.depth = z

    if b[0].split('o')[-1]=='SYN':
        perfil.machine = 'SYNERGY' 
    elif b[0].split('o')[-1]=='PLAT':
        perfil.machine  = 'PLATFORM' 
    else:
        'Equipo no encontrado.'

    #Aca creo dos perfiles porque el txt tiene la informacion de dos curvas segun si se exporto coronal o transversal

    dose2D = np.array([row.split(',') for row in a[16:]])
    dose2D = dose2D.astype('float32')

    spatial_resolution = float(a[15].split(',')[-1])
    starting_point1     = abs(float(a[12].split(',')[1]))
    starting_point2     = abs(float(a[12].split(',')[2]))
  
    if b[2]=='Transverse':
        'Archivo de pdds.'
    elif b[2]=='Coronal':
        perfil.coordinate  = 'X' 
        perfil.axis          = np.arange(-dose2D.shape[1]//2,dose2D.shape[1]//2,spatial_resolution)
        perfil.relative_dose = dose2D[dose2D.shape[0]//2,:]

        # print(perfil.axis.shape)
        # print(perfil.relative_dose.shape)

        Perfiles.append(perfil)

        #Cargo tambien la informacion del perfil en sentido Y

        perfil2 = copy.copy(perfil)

        perfil2.coordinate  = 'Y' 
        perfil2.axis         = np.arange(-dose2D.shape[0]//2,dose2D.shape[0]//2,spatial_resolution)
        perfil2.relative_dose       = dose2D[:,dose2D.shape[1]//2]

        # print(perfil.axis.shape)
        # print(perfil.relative_dose.shape)

        Perfiles.append(perfil2)
    else:
        'Error en la matriz de dosis exportada de Monaco.'

    return Perfiles

def IngresaPDDCalculadoMONACO(fname):
    a = np.genfromtxt(fname,dtype=str, delimiter='\n', skip_header=0, skip_footer=0)

    path_list = fname.split('/')

    fname = os.path.split(fname)[1]
    # BLUEx06oSYN.FotonesX06SYN.Coronal.50.00

    b = np.array( fname.split('/')[-1].rsplit('.', 1)[0].split('.') )
    #['BLUEx06oSYN' 'FotonesX06SYN' 'Coronal' '50' '00']

    PDDs = []

    PDD = Curve()

    PDD.date       = a[2].split(',')[1]
    PDD.energy     = b[0].split('o')[0][4:].upper()
    PDD.SSD         = str(int(path_list[-3].split(' ')[-1])*10)  # Lo saca del path, importante carpetas
    tc = str(int(path_list[-2].split('x')[0])*10)
    if len(tc)<3:
        PDD.field_size          = '0'+tc+'*'+'0'+tc
    else:
        PDD.field_size          = tc+'*'+tc
    PDD.depth = ''


    if b[0].split('o')[-1]=='SYN':
        PDD.machine = 'SYNERGY' 
    elif b[0].split('o')[-1]=='PLAT':
        PDD.machine  = 'PLATFORM' 
    else:
        'Equipo no encontrado.'

    dose2D = np.array([row.split(',') for row in a[16:]])
    dose2D = dose2D.astype('float32')

    # plt.imshow(dose2D)
    # plt.show()

    spatial_resolution = float(a[15].split(',')[-1])
    starting_point1     = abs(float(a[12].split(',')[1]))
    starting_point2     = abs(float(a[12].split(',')[2]))
  
    if b[2]=='Transverse':
        PDD.coordinate  = 'Z' 
        PDD.axis         = np.arange(0,dose2D.shape[0]//2+spatial_resolution,spatial_resolution)
        PDD.relative_dose       = dose2D[dose2D.shape[0]//2:,dose2D.shape[1]//2]

        PDDs.append(PDD)
    elif b[2]=='Coronal':
        'Archivo de perfiles.'
    else:
        'Error en la matriz de dosis exportada de Monaco.'

    return PDDs

def Interpola(Curva, step):
    #interpola la curva para que todos tengan ejes en comun
    f = scipy.interpolate.interp1d(Curva.axis, Curva.relative_dose, fill_value='extrapolate')

    if(Curva.coordinate=='X' or Curva.coordinate=='Y'):
        axis = np.concatenate( (np.arange(-step,Curva.axis.min(),-step)[::-1] , np.arange(0,Curva.axis.max(),step)) , axis=0)
    elif(Curva.coordinate=='Z'):
        axis = np.arange(0,Curva.axis.max(),step)

    return axis,f(axis)

def SuavizaYNormaliza(Curva,sigma):
    #suaviza con un filtro gaussiano y normaliza al cax (perfiles) o al maximo (PDD)
    Curva_filtr = gaussian_filter1d(Curva.relative_dose, sigma) 
    if(Curva.coordinate == 'X' or Curva.coordinate == 'Y'):
        return Curva_filtr/np.trim_zeros(np.where(Curva.axis == 0, Curva.relative_dose,0))[0]*100
    elif(Curva.coordinate == 'Z'):
        return Curva_filtr/Curva_filtr.max()*100
    
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

def ProcesaCurva(Curvas_med,Curvas_cal,espaciamiento,sigma_suavizado):
    for i in range(len(Curvas_med)):
        # print(Curvas_med[i].axis.shape, Curvas_med[i].relative_dose.shape)
        Curvas_med[i].axis,Curvas_med[i].relative_dose = Interpola(Curvas_med[i],espaciamiento)    #datos cada 0.5 mm
        Curvas_med[i].relative_dose = SuavizaYNormaliza(Curvas_med[i],sigma_suavizado)
        # print(Curvas_med[i].axis.shape, Curvas_med[i].relative_dose.shape)
    for j in range(len(Curvas_cal)):
        Curvas_cal[j].axis,Curvas_cal[j].relative_dose = Interpola(Curvas_cal[j],espaciamiento)    #datos cada 0.5 mm
        Curvas_cal[j].relative_dose = SuavizaYNormaliza(Curvas_cal[j],sigma_suavizado)

    return Curvas_med, Curvas_cal

def Analiza_Y_Grafica_Perfiles(Perfiles_med,Perfiles_cal,dose_threshold,dta,cutoff, COMPARE_ANYWAY=False):
    idx_med = np.arange( len(Perfiles_med) )
    idx_cal = np.arange( len(Perfiles_cal) )

    for i,j in itertools.product( idx_med , idx_cal ): 
        if(COMPARE_ANYWAY):
            datos = []
            # datos = AnalizaRegionesPerfil(Perfiles_med[i],Perfiles_cal[j])
            Grafica_Perfiles(Perfiles_med[i],Perfiles_cal[j],datos,i,dose_threshold,dta,cutoff) 
        elif( Perfiles_med[i].tc        == Perfiles_cal[j].tc          and 
            Perfiles_med[i].depth == Perfiles_cal[j].depth and
            Perfiles_med[i].dfp         == Perfiles_cal[j].dfp         and 
            Perfiles_med[i].equipo      == Perfiles_cal[j].equipo      and 
            Perfiles_med[i].coordinate  == Perfiles_cal[j].coordinate  and
            Perfiles_med[i].energia  == Perfiles_cal[j].energia):
            datos = []
            # datos = AnalizaRegionesPerfil(Perfiles_med[i],Perfiles_cal[j])
            print('Match!')
            Grafica_Perfiles(Perfiles_med[i],Perfiles_cal[j],datos,i,dose_threshold,dta,cutoff)
        
def Analiza_Y_Grafica_PDDs(PDDs_med,PDDs_cal,dose_threshold,dta,cutoff, COMPARE_ANYWAY=False):
    idx_med = np.arange( len(PDDs_med) )
    idx_cal = np.arange( len(PDDs_cal) )

    for i,j in itertools.product( idx_med , idx_cal ):  
        if(COMPARE_ANYWAY):
            Grafica_PDDs(PDDs_med[i],PDDs_cal[j],i,dose_threshold,dta,cutoff)  
        elif( PDDs_med[i].tc        == PDDs_cal[j].tc          and
            PDDs_med[i].dfp         == PDDs_cal[j].dfp         and 
            PDDs_med[i].equipo      == PDDs_cal[j].equipo      and 
            PDDs_med[i].coordinate  == PDDs_cal[j].coordinate  and
            PDDs_med[i].energia     == PDDs_cal[j].energia):
            print('Match!')
            Grafica_PDDs(PDDs_med[i],PDDs_cal[j],i,dose_threshold,dta,cutoff)
        
def Grafica_Perfiles(Perfiles_med, Perfiles_cal, datos, i, dose_threshold, dta, cutoff):
    # Create a Toplevel window
    ventana_grafico = tk.Toplevel()
    # ventana_grafico.withdraw()

    # Create a single Figure instance
    fig = Figure()
    ax1 = fig.add_subplot()
    ax1.set_ylabel('Porcentaje [%]', color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Gamma Index ' + str(dose_threshold) + '% ' + str(dta) + 'mm', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()  # otherwise, the right y-label is slightly clipped

    gamma = pymedphys.gamma(Perfiles_cal.axis, Perfiles_cal.relative_dose, Perfiles_med.axis, Perfiles_med.relative_dose,
                            dose_percent_threshold=dose_threshold, distance_mm_threshold=dta,
                            lower_percent_dose_cutoff=cutoff)

    # Plot on the same axes
    ax1.plot(Perfiles_cal.axis, Perfiles_cal.relative_dose, marker='.', markersize=0, linewidth=1, label="Monaco",
             color='b')
    ax1.plot(Perfiles_med.axis, Perfiles_med.relative_dose, marker='.', markersize=2, linewidth=0, label=str(Perfiles_med.fecha),
             color='g')
    ax2.plot(Perfiles_cal.axis, gamma, marker='+', markersize=0, linewidth=0.5,
             label='Gamma ' + str(dose_threshold) + '% ' + str(dta) + 'mm', color='r')
    ax2.axhline(y=1, color='r', linewidth=0.5)

    ax1.set_title(str(Perfiles_cal.equipo) + ' - QA ANUAL ' + str(Perfiles_med.fecha)[6:] + ' - Perfil en ' + str(
        Perfiles_med.coordinate) +
                  ' - DFP ' + str(float(Perfiles_med.dfp) / 10) +
                  ' cm - TC ' + str(Perfiles_med.tc) +
                  ' mm2 - Prof: ' + str(float(Perfiles_med.depth) / 10) +
                  ' cm - BluePhantom2 CC13',
                  fontsize=7)

    # ax1.text(min(Perfiles_med.axis) / 3.5, 30,
    #          'Dif (min,max) de %D y dta:' + '\n'
    #          '\n'
    #          'Umbra izq:    ' + '(' + str(datos[0]) + ' %, ' + str(datos[1]) + ' %)' + '\n'
    #                                                                                       'Penumbra izq: ' + '(' + str(
    #              datos[2]) + ' mm, ' + str(datos[3]) + ' mm)' + '\n'
    #                                                                'Plateau:      ' + '(' + str(datos[4]) + ' %, ' + str(
    #              datos[5]) + ' %)' + '\n'
    #                               'Penumbra der: ' + '(' + str(datos[6]) + ' mm, ' + str(datos[7]) + ' mm)' + '\n'
    #                                                                                                            'Umbra der:    ' + '(' + str(
    #              datos[8]) + ' %, ' + str(datos[9]) + ' %)' + '\n'
    #                                                           '\n'
    #                                                           'Umbra<10%' + '\n'
    #                                                                           '10%<Penumbra<90%' + '\n'
    #                                                                                                   'Plateau>90%' + '\n',
    #          fontsize=6,
    #          bbox={'facecolor': 'grey', 'alpha': 0.3, 'pad': 10})

    ax1.set_xlim([max(Perfiles_med.axis.min(), Perfiles_cal.axis.min()),
                  min(Perfiles_med.axis.max(), Perfiles_cal.axis.max())])

    ax1.set(xlabel='axis ' + str(Perfiles_med.coordinate) + '(mm)', ylabel='Dosis (%)')
    ax1.legend()
    ax1.grid()

    ax2.set_ylim([0, 5])
    ax2.legend()

    # Use the FigureCanvasTkAgg constructor with the existing figure and window
    canvas = FigureCanvasTkAgg(fig, ventana_grafico)
    toolbar = NavigationToolbar2Tk(canvas, ventana_grafico)
    canvas._tkcanvas.pack()

    print('exportado')

    # fig.savefig('Perfil'+str(i)+'.pdf', format='pdf', orientation='landscape', bbox_inches='tight')

def Grafica_PDDs(PDDs_med,PDDs_cal,i,dose_threshold,dta,cutoff):
    ventana_grafico = tk.Toplevel()

    fig  = Figure()
    ax1  = fig.add_subplot()
    ax1.set_ylabel('Porcentaje [%]', color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Gamma Index '+str(dose_threshold)+'% '+str(dta)+'mm', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    gamma = pymedphys.gamma(PDDs_cal.axis, PDDs_cal.relative_dose, PDDs_med.axis, PDDs_med.relative_dose,
                            dose_percent_threshold=dose_threshold, distance_mm_threshold=dta, 
                            lower_percent_dose_cutoff=cutoff)

    ax1.plot(PDDs_cal.axis, PDDs_cal.relative_dose, marker='.', markersize=0, linewidth=1, label="Monaco", color='b')
    ax1.plot(PDDs_med.axis, PDDs_med.relative_dose, marker='.', markersize=2, linewidth=0, label=str(PDDs_med.fecha), color='g')
    ax2.plot(PDDs_cal.axis, gamma, marker='+', markersize=0, linewidth=0.5, label='Gamma '+str(dose_threshold)+'% '+str(dta)+'mm', color='r')
    ax2.axhline(y = 1 , color='r', linewidth=0.5)

    ax1.set_title(str(PDDs_cal.equipo)+' - QA ANUAL '+str(PDDs_med.fecha)[6:]+' - PDD '+
                    ' - DFP = '+str(float(PDDs_med.dfp)/10)+
                    ' cm - TC '+str(PDDs_med.tc)+
                    '- BluePhantom2 CC13', 
                    fontsize=7)

    ax1.set_xlim([max(PDDs_med.axis.min(),PDDs_cal.axis.min()), min(PDDs_med.axis.max(),PDDs_cal.axis.max())])
    ax1.set(xlabel='Profundidad (mm)', ylabel='Dosis (%)')
    ax1.legend()
    ax1.grid()

    ax2.set_ylim([0, 5])
    
    canvas  = FigureCanvasTkAgg(fig,ventana_grafico)
    toolbar = NavigationToolbar2Tk(canvas,ventana_grafico)
    canvas._tkcanvas.pack()

    # fig.savefig('Pdd'+str(i)+'.pdf', orientation='landscape', format='pdf', bbox_inches='tight')


# def IngresaPerfilCalculado(fname):
#     a = np.genfromtxt(fname,dtype=str, delimiter='\n', skip_header=10, skip_footer=0)
#     a = np.array( [line.split('\t') for line in list(a)] )
#     TCs = [str(tc) for tc in a[0]]
#     a = a[1:].T.astype(float)

#     fname = os.path.split(fname)[1]

#     # UNIQUE DFP 100 PERFIL X Z 50 MM

#     b = np.array( fname.split('/')[-1].split('.')[0].split() )

#     coordinate  = str(b[4])
#     profundidad = str(b[6])
#     dfp         = str(b[2])

#     Perfiles = []

#     print('Datos de perfiles calculados ingresados:')

#     for i in range(a.shape[0]-1):
#         perfil = Curva()

#         perfil.tc      = str(TCs[i+1])
#         perfil.axis     = np.array(a[0])*10.0  #paso de cm a mm
#         perfil.relative_dose   = np.array(a[i+1])

#         perfil.coordinate  = coordinate
#         perfil.fecha       = '-'
#         perfil.depth = profundidad
#         perfil.dfp         = dfp
#         perfil.equipo      = fname[:6]

#         Perfiles.append(perfil)

#         print('Ingreso!!')

#         print(perfil.axis.shape, perfil.relative_dose.shape)

#         print('Equipo: %s, DFP: %s, TC: %s, Z: %s, axis: %s' % (perfil.equipo, perfil.dfp, perfil.tc, perfil.depth, perfil.coordinate) )


#     return Perfiles

# def IngresaPDDCalculado(fname):
#     a = np.genfromtxt(fname,dtype=str, delimiter='\n', skip_header=10, skip_footer=0)
#     a = np.array( [line.split('\t') for line in list(a)] )
#     TCs = [str(tc) for tc in a[0]]
#     a = a[1:].T.astype(float)

#     fname = os.path.split(fname)[1]

#     # UNIQUE DFP 1000 PERFIL X Z 050 MM

#     b = np.array( fname.split('/')[-1].split('.')[0].split() )

#     coordinate  = str(b[4])
#     profundidad = str('-')
#     dfp         = str(b[2])

#     PDDs = []

#     print('Datos de pdds calculados ingresados:')

#     for i in range(a.shape[0]-1):
#         PDD = Curva()

#         PDD.tc      = str(TCs[i+1])
#         PDD.axis     = np.array(a[0])*10.0  #paso de cm a mm
#         PDD.relative_dose   = np.array(a[i+1])

#         PDD.coordinate  = coordinate
#         PDD.fecha       = '-'
#         PDD.depth = profundidad
#         PDD.dfp         = dfp
#         PDD.equipo      = fname[:6]

#         PDDs.append(PDD)

#         print('Equipo: %s, DFP: %s, TC: %s' % (PDD.equipo, PDD.dfp, PDD.tc) )


#     return PDDs
