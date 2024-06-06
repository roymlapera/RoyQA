import numpy as np
import scipy.interpolate
from scipy.ndimage import gaussian_filter1d
import itertools
import pymedphys

from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import filedialog, Text
import os
import copy

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------------------------

class Curva():
    coordenada = ''
    fecha = ''
    profundidad = ''
    dfp = ''
    tc = ''
    eje = np.array
    valor = np.array
    equipo = ''
    energia = ''

def Convert(string): 
    li = list(string.split(" ")) 
    return li 


def IngresaPerfilMedido(fname):

    a = np.genfromtxt(fname,dtype=str, delimiter='\n', skip_header=1, skip_footer=1)

    fname = os.path.split(fname)[1].split('.')[0]

    l = []
    for s in a: 
        s = s.replace('<','')
        s = s.replace('>','')
        s = s.replace('+','')
        l.append(Convert(s))

    curvas = []

    for e in l:
        if e[0] == '$STOM':
            aux = []
            continue
        if e[0] == '$ENOM': 
            curvas.append(aux)
            continue
        else: aux.append(e)

    Perfiles = []

    print('\n'+'Datos de perfiles medidos ingresados:')

    for i in range(len(curvas)):
        perfil = Curva()

        perfil.coordenada  = str(curvas[i][6][1])
        perfil.fecha       = str(curvas[i][1][1])
        perfil.profundidad = str(curvas[i][10][1])
        perfil.dfp         = str(curvas[i][9][2])
        perfil.tc          = str(curvas[i][4][1])
        perfil.energia     = fname.split(' ')[1].upper()
        perfil.equipo      = fname.split(' ')[0].upper()
        
        perfil.valor       = np.array(curvas[i][11:], dtype=float)[:,3]

        if(perfil.coordenada=='X'):
            perfil.eje     = np.array(curvas[i][11:], dtype=float)[:,0]
        else:
            perfil.eje     = np.array(curvas[i][11:], dtype=float)[:,1]

        Perfiles.append(perfil)

    print('Datos de dosis ingresada:')
    
    for perfil in Perfiles:
        print('Equipo: %s, DFP: %s, TC: %s, Z: %s, Eje: %s, Energía: %s' % (perfil.equipo, perfil.dfp, perfil.tc, perfil.profundidad, perfil.coordenada, perfil.energia) )

    return Perfiles

def IngresaPDDMedido(fname):

    a = np.genfromtxt(fname,dtype=str, delimiter='\n', skip_header=1, skip_footer=1)

    fname = os.path.split(fname)[1].split('.')[0]

    l = []
    for s in a: 
        s = s.replace('<','')
        s = s.replace('>','')
        s = s.replace('+','')
        l.append(Convert(s))

    curvas = []

    for e in l:
        if e[0] == '$STOM':
            aux = []
            continue
        if e[0] == '$ENOM': 
            curvas.append(aux)
            continue
        else: aux.append(e)

    PDDs = []

    print('\n'+'Datos de pdds medidos ingresados:')

    for i in range(len(curvas)):
        PDD = Curva()

        PDD.coordenada  = str(curvas[i][6][1])
        PDD.fecha       = str(curvas[i][1][1])
        PDD.profundidad = '-'
        PDD.dfp         = str(curvas[i][9][2])
        PDD.tc          = str(curvas[i][4][1])
        PDD.energia     = fname.split(' ')[1].upper()
        PDD.equipo      = fname.split(' ')[0].upper()
        
        PDD.valor       = np.array(curvas[i][11:], dtype=float)[:,3]

        PDD.eje     = np.array(curvas[i][11:], dtype=float)[:,2]  #ACA VA 2 EN LUGAR DE 0 ASI AGARRA LA COLUMNA DE Z

        PDDs.append(PDD)

        print('Equipo: %s, DFP: %s, TC: %s, Eje: %s, Energía: %s' % (PDD.equipo, PDD.dfp, PDD.tc, PDD.coordenada, PDD.energia) )

    return PDDs


def IngresaPerfilCalculadoMONACO(fname):
    a = np.genfromtxt(fname,dtype=str, delimiter='\n', skip_header=0, skip_footer=0)

    path_list = fname.split('/')

    fname = os.path.split(fname)[1]
    # BLUEx06oSYN.FotonesX06SYN.Coronal.50.00

    b = np.array( fname.split('/')[-1].rsplit('.', 1)[0].split('.') )
    #['BLUEx06oSYN' 'FotonesX06SYN' 'Coronal' '50' '00']

    Perfiles = []

    perfil = Curva()

    perfil.fecha       = a[2].split(',')[1]
    perfil.energia     = b[0].split('o')[0][4:].upper()
    perfil.dfp         = str(int(path_list[-3].split(' ')[-1])*10)  # Lo saca del path, importante carpetas

    tc = str(int(path_list[-2].split('x')[0])*10)
    if len(tc)<3:
        perfil.tc          = '0'+tc+'*'+'0'+tc
    else:
        perfil.tc          = tc+'*'+tc
    
    z = str(250 - int(b[3]))
    if len(z)<3:
        perfil.profundidad = '0'+z
    else:
        perfil.profundidad = z

    if b[0].split('o')[-1]=='SYN':
        perfil.equipo = 'SYNERGY' 
    elif b[0].split('o')[-1]=='PLAT':
        perfil.equipo  = 'PLATFORM' 
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
        perfil.coordenada  = 'X' 
        perfil.eje         = np.arange(-dose2D.shape[1]//2,dose2D.shape[1]//2,spatial_resolution)
        perfil.valor       = dose2D[dose2D.shape[0]//2,:]

        # print(perfil.eje.shape)
        # print(perfil.valor.shape)

        Perfiles.append(perfil)

        #Cargo tambien la informacion del perfil en sentido Y

        perfil2 = copy.copy(perfil)

        perfil2.coordenada  = 'Y' 
        perfil2.eje         = np.arange(-dose2D.shape[0]//2,dose2D.shape[0]//2,spatial_resolution)
        perfil2.valor       = dose2D[:,dose2D.shape[1]//2]

        # print(perfil.eje.shape)
        # print(perfil.valor.shape)

        Perfiles.append(perfil2)
    else:
        'Error en la matriz de dosis exportada de Monaco.'

    print('Datos de dosis ingresada:')
    for perfil in Perfiles:
        print('Equipo: %s, DFP: %s, TC: %s, Z: %s, Eje: %s, Energía: %s' % (perfil.equipo, perfil.dfp, perfil.tc, perfil.profundidad, perfil.coordenada, perfil.energia) )

    return Perfiles

def IngresaPDDCalculadoMONACO(fname):
    a = np.genfromtxt(fname,dtype=str, delimiter='\n', skip_header=0, skip_footer=0)

    path_list = fname.split('/')

    fname = os.path.split(fname)[1]
    # BLUEx06oSYN.FotonesX06SYN.Coronal.50.00

    b = np.array( fname.split('/')[-1].rsplit('.', 1)[0].split('.') )
    #['BLUEx06oSYN' 'FotonesX06SYN' 'Coronal' '50' '00']

    PDDs = []

    PDD = Curva()

    PDD.fecha       = a[2].split(',')[1]
    PDD.energia     = b[0].split('o')[0][4:].upper()
    PDD.dfp         = str(int(path_list[-3].split(' ')[-1])*10)  # Lo saca del path, importante carpetas
    tc = str(int(path_list[-2].split('x')[0])*10)
    if len(tc)<3:
        PDD.tc          = '0'+tc+'*'+'0'+tc
    else:
        PDD.tc          = tc+'*'+tc
    PDD.profundidad = ''


    if b[0].split('o')[-1]=='SYN':
        PDD.equipo = 'SYNERGY' 
    elif b[0].split('o')[-1]=='PLAT':
        PDD.equipo  = 'PLATFORM' 
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
        PDD.coordenada  = 'Z' 
        PDD.eje         = np.arange(0,dose2D.shape[0]//2+spatial_resolution,spatial_resolution)
        PDD.valor       = dose2D[dose2D.shape[0]//2:,dose2D.shape[1]//2]

        PDDs.append(PDD)
    elif b[2]=='Coronal':
        'Archivo de perfiles.'
    else:
        'Error en la matriz de dosis exportada de Monaco.'

    print('Datos de dosis ingresada:')
    for PDD in PDDs:
        print('Equipo: %s, DFP: %s, TC: %s, Eje: %s, Energía: %s' % (PDD.equipo, PDD.dfp, PDD.tc, PDD.coordenada, PDD.energia) )

    return PDDs

def Interpola(Curva, step):
    #interpola la curva para que todos tengan ejes en comun
    f = scipy.interpolate.interp1d(Curva.eje, Curva.valor, fill_value='extrapolate')

    if(Curva.coordenada=='X' or Curva.coordenada=='Y'):
        eje = np.concatenate( (np.arange(-step,Curva.eje.min(),-step)[::-1] , np.arange(0,Curva.eje.max(),step)) , axis=0)
    elif(Curva.coordenada=='Z'):
        eje = np.arange(0,Curva.eje.max(),step)

    return eje,f(eje)

def SuavizaYNormaliza(Curva,sigma):
    #suaviza con un filtro gaussiano y normaliza al cax (perfiles) o al maximo (PDD)
    if(Curva.coordenada=='X' or Curva.coordenada=='Y'):
        Curva_filtr = gaussian_filter1d(Curva.valor, sigma) 
        return Curva_filtr/np.trim_zeros(np.where(Curva.eje==0,Curva.valor,0))[0]*100
    elif(Curva.coordenada=='Z'):
        Curva_filtr = gaussian_filter1d(Curva.valor, sigma)
        return Curva_filtr/Curva_filtr.max()*100
    
def AnalizaRegionesPerfil(Curva,Curva_ref):
    MAX = min(Curva.eje.max(),Curva_ref.eje.max())
    MIN = max(Curva.eje.min(),Curva_ref.eje.min())

    Curva_ref.valor = Curva_ref.valor[(Curva_ref.eje >= MIN) & (Curva_ref.eje <= MAX)]
    Curva_ref.eje = Curva_ref.eje[(Curva_ref.eje >= MIN) & (Curva_ref.eje <= MAX)]

    Curva.valor = Curva.valor[(Curva.eje >= MIN) & (Curva.eje <= MAX)]
    Curva.eje = Curva.eje[(Curva.eje >= MIN) & (Curva.eje <= MAX)]


    penumbra_izq = (Curva.valor <= 90) & (Curva.valor >= 10) & (Curva.eje < 0)
    penumbra_der = (Curva.valor <= 90) & (Curva.valor >= 10) & (Curva.eje > 0)

    umbra_izq = (Curva.valor < 10) & (Curva.eje < 0)
    umbra_der = (Curva.valor < 10) & (Curva.eje > 0)

    plateau = (Curva.valor < 90)

    #ANALISIS DE DIFERENCIA EN DOSIS

    max_D_plateau = round( (Curva.valor[plateau]-Curva_ref.valor[plateau]).max() ,1)
    min_D_plateau = round( (Curva.valor[plateau]-Curva_ref.valor[plateau]).min() ,1)
    
    max_D_umbra_izq = round( (Curva.valor[umbra_izq]-Curva_ref.valor[umbra_izq]).max() ,1)
    min_D_umbra_izq = round( (Curva.valor[umbra_izq]-Curva_ref.valor[umbra_izq]).min() ,1)

    max_D_umbra_der = round( (Curva.valor[umbra_der]-Curva_ref.valor[umbra_der]).max() ,1)
    min_D_umbra_der = round( (Curva.valor[umbra_der]-Curva_ref.valor[umbra_der]).min() ,1)

    #ANALISIS DE DIFERENCIA EN DISTANCIA

    f_izq = scipy.interpolate.interp1d(Curva.valor[penumbra_izq], Curva.eje[penumbra_izq])
    f_ref_izq = scipy.interpolate.interp1d(Curva_ref.valor[penumbra_izq], Curva_ref.eje[penumbra_izq])

    f_der = scipy.interpolate.interp1d(Curva.valor[penumbra_der], Curva.eje[penumbra_der])
    f_ref_der = scipy.interpolate.interp1d(Curva_ref.valor[penumbra_der], Curva_ref.eje[penumbra_der])

    eje = np.arange(15,85,5)

    max_dta_penumbra_izq = round( (f_izq(eje)-f_ref_izq(eje) ).max() ,1)
    min_dta_penumbra_izq = round( (f_izq(eje)-f_ref_izq(eje) ).min() ,1)
    max_dta_penumbra_der = round( (f_der(eje)-f_ref_der(eje) ).max() ,1)
    min_dta_penumbra_der = round( (f_der(eje)-f_ref_der(eje) ).min() ,1)

    return min_D_umbra_izq, max_D_umbra_izq, min_dta_penumbra_izq, max_dta_penumbra_izq, min_D_plateau, max_D_plateau, min_dta_penumbra_der, max_dta_penumbra_der, min_D_umbra_der, max_D_umbra_der

def ProcesaCurva(Curvas_med,Curvas_cal,espaciamiento,sigma_suavizado):
    for i in range(len(Curvas_med)):
        # print(Curvas_med[i].eje.shape, Curvas_med[i].valor.shape)
        Curvas_med[i].eje,Curvas_med[i].valor = Interpola(Curvas_med[i],espaciamiento)    #datos cada 0.5 mm
        Curvas_med[i].valor = SuavizaYNormaliza(Curvas_med[i],sigma_suavizado)
        # print(Curvas_med[i].eje.shape, Curvas_med[i].valor.shape)
    for j in range(len(Curvas_cal)):
        Curvas_cal[j].eje,Curvas_cal[j].valor = Interpola(Curvas_cal[j],espaciamiento)    #datos cada 0.5 mm
        Curvas_cal[j].valor = SuavizaYNormaliza(Curvas_cal[j],sigma_suavizado)

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
            Perfiles_med[i].profundidad == Perfiles_cal[j].profundidad and
            Perfiles_med[i].dfp         == Perfiles_cal[j].dfp         and 
            Perfiles_med[i].equipo      == Perfiles_cal[j].equipo      and 
            Perfiles_med[i].coordenada  == Perfiles_cal[j].coordenada  and
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
            PDDs_med[i].coordenada  == PDDs_cal[j].coordenada  and
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

    gamma = pymedphys.gamma(Perfiles_cal.eje, Perfiles_cal.valor, Perfiles_med.eje, Perfiles_med.valor,
                            dose_percent_threshold=dose_threshold, distance_mm_threshold=dta,
                            lower_percent_dose_cutoff=cutoff)

    # Plot on the same axes
    ax1.plot(Perfiles_cal.eje, Perfiles_cal.valor, marker='.', markersize=0, linewidth=1, label="Monaco",
             color='b')
    ax1.plot(Perfiles_med.eje, Perfiles_med.valor, marker='.', markersize=2, linewidth=0, label=str(Perfiles_med.fecha),
             color='g')
    ax2.plot(Perfiles_cal.eje, gamma, marker='+', markersize=0, linewidth=0.5,
             label='Gamma ' + str(dose_threshold) + '% ' + str(dta) + 'mm', color='r')
    ax2.axhline(y=1, color='r', linewidth=0.5)

    ax1.set_title(str(Perfiles_cal.equipo) + ' - QA ANUAL ' + str(Perfiles_med.fecha)[6:] + ' - Perfil en ' + str(
        Perfiles_med.coordenada) +
                  ' - DFP ' + str(float(Perfiles_med.dfp) / 10) +
                  ' cm - TC ' + str(Perfiles_med.tc) +
                  ' mm2 - Prof: ' + str(float(Perfiles_med.profundidad) / 10) +
                  ' cm - BluePhantom2 CC13',
                  fontsize=7)

    # ax1.text(min(Perfiles_med.eje) / 3.5, 30,
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

    ax1.set_xlim([max(Perfiles_med.eje.min(), Perfiles_cal.eje.min()),
                  min(Perfiles_med.eje.max(), Perfiles_cal.eje.max())])

    ax1.set(xlabel='Eje ' + str(Perfiles_med.coordenada) + '(mm)', ylabel='Dosis (%)')
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

    gamma = pymedphys.gamma(PDDs_cal.eje, PDDs_cal.valor, PDDs_med.eje, PDDs_med.valor,
                            dose_percent_threshold=dose_threshold, distance_mm_threshold=dta, 
                            lower_percent_dose_cutoff=cutoff)

    ax1.plot(PDDs_cal.eje, PDDs_cal.valor, marker='.', markersize=0, linewidth=1, label="Monaco", color='b')
    ax1.plot(PDDs_med.eje, PDDs_med.valor, marker='.', markersize=2, linewidth=0, label=str(PDDs_med.fecha), color='g')
    ax2.plot(PDDs_cal.eje, gamma, marker='+', markersize=0, linewidth=0.5, label='Gamma '+str(dose_threshold)+'% '+str(dta)+'mm', color='r')
    ax2.axhline(y = 1 , color='r', linewidth=0.5)

    ax1.set_title(str(PDDs_cal.equipo)+' - QA ANUAL '+str(PDDs_med.fecha)[6:]+' - PDD '+
                    ' - DFP = '+str(float(PDDs_med.dfp)/10)+
                    ' cm - TC '+str(PDDs_med.tc)+
                    '- BluePhantom2 CC13', 
                    fontsize=7)

    ax1.set_xlim([max(PDDs_med.eje.min(),PDDs_cal.eje.min()), min(PDDs_med.eje.max(),PDDs_cal.eje.max())])
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

#     coordenada  = str(b[4])
#     profundidad = str(b[6])
#     dfp         = str(b[2])

#     Perfiles = []

#     print('Datos de perfiles calculados ingresados:')

#     for i in range(a.shape[0]-1):
#         perfil = Curva()

#         perfil.tc      = str(TCs[i+1])
#         perfil.eje     = np.array(a[0])*10.0  #paso de cm a mm
#         perfil.valor   = np.array(a[i+1])

#         perfil.coordenada  = coordenada
#         perfil.fecha       = '-'
#         perfil.profundidad = profundidad
#         perfil.dfp         = dfp
#         perfil.equipo      = fname[:6]

#         Perfiles.append(perfil)

#         print('Ingreso!!')

#         print(perfil.eje.shape, perfil.valor.shape)

#         print('Equipo: %s, DFP: %s, TC: %s, Z: %s, Eje: %s' % (perfil.equipo, perfil.dfp, perfil.tc, perfil.profundidad, perfil.coordenada) )


#     return Perfiles

# def IngresaPDDCalculado(fname):
#     a = np.genfromtxt(fname,dtype=str, delimiter='\n', skip_header=10, skip_footer=0)
#     a = np.array( [line.split('\t') for line in list(a)] )
#     TCs = [str(tc) for tc in a[0]]
#     a = a[1:].T.astype(float)

#     fname = os.path.split(fname)[1]

#     # UNIQUE DFP 1000 PERFIL X Z 050 MM

#     b = np.array( fname.split('/')[-1].split('.')[0].split() )

#     coordenada  = str(b[4])
#     profundidad = str('-')
#     dfp         = str(b[2])

#     PDDs = []

#     print('Datos de pdds calculados ingresados:')

#     for i in range(a.shape[0]-1):
#         PDD = Curva()

#         PDD.tc      = str(TCs[i+1])
#         PDD.eje     = np.array(a[0])*10.0  #paso de cm a mm
#         PDD.valor   = np.array(a[i+1])

#         PDD.coordenada  = coordenada
#         PDD.fecha       = '-'
#         PDD.profundidad = profundidad
#         PDD.dfp         = dfp
#         PDD.equipo      = fname[:6]

#         PDDs.append(PDD)

#         print('Equipo: %s, DFP: %s, TC: %s' % (PDD.equipo, PDD.dfp, PDD.tc) )


#     return PDDs
