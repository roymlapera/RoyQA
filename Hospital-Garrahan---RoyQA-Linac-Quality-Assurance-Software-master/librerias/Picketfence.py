from pylinac.picketfence import PicketFence, MLCArrangement
import tkinter as tk
from tkinter import filedialog


mlc_UNIQUE = MLCArrangement( leaf_arrangement=[(10,10),(40,5),(10,10)] )
mlc_23EX   = MLCArrangement( leaf_arrangement=[(40,10)] )

#-------------------------------------------------------------------------

def ErrorBox():
	messagebox.showinfo(message="ERROR",title="Mensaje")


def QA_PicketFence(fields, entries,filename):
	entries['Equipo (mayus)'] = str(entries['Equipo (mayus)'].get())
	entries['1er Nivel de Acción (mm)'] = float(entries['1er Nivel de Acción (mm)'].get())
	entries['2do Nivel de Acción (mm)'] = float(entries['2do Nivel de Acción (mm)'].get())
	entries['Fecha (dd-mm-aa)'] = str(entries['Fecha (dd-mm-aa)'].get())
	entries['Físico'] = str(entries['Físico'].get())

	if (entries['Equipo (mayus)'] == '23EX'):
		mlc = mlc_23EX
	elif(entries['Equipo (mayus)'] == 'UNIQUE'):
		mlc = mlc_UNIQUE
	else:
		ErrorBox()

	pf = PicketFence( filename , mlc=mlc )

	pf.analyze(action_tolerance=entries['1er Nivel de Acción (mm)'], tolerance=entries['2do Nivel de Acción (mm)'])
	pf.plot_analyzed_image()

	pf.save_analyzed_image('mypf'+entries['Fecha (dd-mm-aa)']+'.png')
	pf.publish_pdf(filename='mypf.pdf', open_file=True, 
					metadata={'Físico': entries['Físico'],
							'Equipo': entries['Equipo (mayus)'], 
							'Fecha medición': entries['Fecha (dd-mm-aa)']})

def Programa_PicketFence():
	root = tk.Tk()
	root.wm_withdraw()

	filename =  filedialog.askopenfilename(
				title='Seleccionar PicketFence medido',
				filetypes=(('Archivo DICOM','*.dcm'),
					('Todos los archivos','*.')))

	fields = ('Equipo (mayus)', '1er Nivel de Acción (mm)', '2do Nivel de Acción (mm)', 'Fecha (dd-mm-aa)', 'Físico')
	entries = {}

	root2 = tk.Toplevel()

	for field in fields:
		row = tk.Frame(root2)
		lab = tk.Label(row, width=22, text=field+": ", anchor='w')
		ent = tk.Entry(row)
		row.pack(side=tk.TOP, 
				 fill=tk.X, 
				 padx=5, 
				 pady=5)
		lab.pack(side=tk.LEFT)
		ent.pack(side=tk.RIGHT, 
				 expand=tk.YES, 
				 fill=tk.X)
		entries[field] = ent

	b1 = tk.Button(root2, text='Analizar',
		   command=(lambda e=entries: QA_PicketFence(fields,e,filename) ) )
	b1.pack(side=tk.LEFT, padx=5, pady=5)

	b2 = tk.Button(root2, text='Quit', command=root2.quit)
	b2.pack(side=tk.LEFT, padx=5, pady=5)

	root2.mainloop()

# -------------------------------------------------------------------------

Programa_PicketFence()