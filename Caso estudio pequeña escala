import pandas as pd
# Variables de entrada
data = {
    'Columna1': ['Demanda de energía nueva', 'Precio de energía nueva', 'Demanda de potencia nueva',
                 'Precio de potencia nueva', 'Peaje TRANSMISIÓN', 'Peaje DISTRIBUCIÓN',
                 'Costo energía no suministrado', 'Energia no suministrada', 'Tiempo (en hrs al año)',
                 '% que la demanda crezca', '% que la demanda no crezca', 'Costo de inversión en GD:',
                 'longitud de alimentador en km', 'Tasa de retorno', 'Depreciación', 'Inflación'],
    'Columna2': ['DEN', 'PEN', 'DPN', 'PPN', 'PTX', 'PDX', 'CENS', 'ENS', 't', 'Ps', 'Pm', 'CIGD', 'l', 'r', 'α',
                 'β'],
    'Columna3': [2, 40, 2, 10, 48, 1_000, 325, 2, 1_920, 0.5, 0.5, 700_000, 10, 0.05, 0.95, 0.07],
    'Columna4': ['MWh', 'USD/MWh', 'MW', 'USD/MW', 'USD/MWaño', 'USD/MWaño', 'USD/MWh', 'MWh', 'hrs-año', '-', '-',
                 'USD/MW', 'km', '-', '-', '-']
}

# Formato ##############################################################################################################
df = pd.DataFrame(data)

df.index = range(1, len(df) + 1)

def format_number(x):
    if isinstance(x, (int, float)):
        return f"{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    return x

df_formatted = df.applymap(format_number)
########################################################################################################################

print("Variables de entrada:")
print(df_formatted)

def parse_value(value):
    return float(str(value).replace(',', ''))

DEN = parse_value(df.loc[df['Columna2'] == 'DEN', 'Columna3'].values[0])
PEN = parse_value(df.loc[df['Columna2'] == 'PEN', 'Columna3'].values[0])
t = parse_value(df.loc[df['Columna2'] == 't', 'Columna3'].values[0])
DPN = parse_value(df.loc[df['Columna2'] == 'DPN', 'Columna3'].values[0])
PPN = parse_value(df.loc[df['Columna2'] == 'PPN', 'Columna3'].values[0])
PTX = parse_value(df.loc[df['Columna2'] == 'PTX', 'Columna3'].values[0])
PDX = parse_value(df.loc[df['Columna2'] == 'PDX', 'Columna3'].values[0])
l = parse_value(df.loc[df['Columna2'] == 'l', 'Columna3'].values[0])
CIGD = parse_value(df.loc[df['Columna2'] == 'CIGD', 'Columna3'].values[0])
CENS = parse_value(df.loc[df['Columna2'] == 'CENS', 'Columna3'].values[0])
ENS = parse_value(df.loc[df['Columna2'] == 'ENS', 'Columna3'].values[0])
Ps = parse_value(df.loc[df['Columna2'] == 'Ps', 'Columna3'].values[0])
Pm = parse_value(df.loc[df['Columna2'] == 'Pm', 'Columna3'].values[0])
r = parse_value(df.loc[df['Columna2'] == 'r', 'Columna3'].values[0])
α = parse_value(df.loc[df['Columna2'] == 'α', 'Columna3'].values[0])
β = parse_value(df.loc[df['Columna2'] == 'β', 'Columna3'].values[0])

# Escenarios y su ecuación
E1 = (DEN * PEN * t) + (DPN * PPN) + (PTX * DPN) + (PDX * DPN * l)
E2 = CIGD * DPN
E3 = CENS * ENS * t
E4 = (DPN * PPN) + (PTX * DPN) + (PDX * DPN * l)
E5 = CIGD * DPN
E6 = 0

# Costo de decisión
CD1 = Ps * E1 + Pm * E4
CD2 = Ps * E2 + Pm * E5
CD3 = Ps * E3 + Pm * E6

# Opciones de inversión
Inversion_Alimentadores = (CENS * ENS * t) + (PDX * DPN * l) / (1 + r)
Inversion_GD = (CENS * ENS * t) + (CIGD * DPN) / (1 + r)

# Costos de decisión para las opciones de inversión
CD_opc_Al = CD3 + Inversion_Alimentadores
CD_opc_GD = CD3 + Inversion_GD

# Calcular la opción de abandono
Abandono = (CIGD * DPN) - (CIGD * DPN * α) / (1 + r)

# Costo de decisión para la opción de abandono
CD_opc_Ab = CD2 - Abandono

results = {
    'Escenario': ['E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
    'Valor': [E1, E2, E3, E4, E5, E6]
}

df_results = pd.DataFrame(results)

df_results.index = range(1, len(df_results) + 1)

df_results['Valor'] = df_results['Valor'].apply(lambda x: format_number(x))

print("\nValor de cada uno de los escenarios generados:")
print(df_results)

decision_costs = {
    'Costo de Decisión': ['CD1', 'CD2', 'CD3'],
    'Valor': [CD1, CD2, CD3]
}

df_decision_costs = pd.DataFrame(decision_costs)

df_decision_costs['Valor'] = df_decision_costs['Valor'].apply(lambda x: format_number(x))

print("\nTabla de Costos de Decisión:")
print(df_decision_costs)

inversion_options = {
    'Nuevas Decisiones': ['Nueva Inv Al', 'Nueva Inv GD'],
    'Valor': [Inversion_Alimentadores, Inversion_GD]
}

df_inversion_options = pd.DataFrame(inversion_options)

df_inversion_options['Valor'] = df_inversion_options['Valor'].apply(lambda x: format_number(x))

print("\nSi la DEMANDA CRECE y NO INV. se tienen dos opciones para t2:")
print(df_inversion_options)

decision_costs_options = {
    'Costo de Decisión': ['CD Nueva Inv Al', 'CD Nueva Inv GD'],
    'Valor': [CD_opc_Al, CD_opc_GD]
}

df_decision_costs_options = pd.DataFrame(decision_costs_options)

df_decision_costs_options['Valor'] = df_decision_costs_options['Valor'].apply(lambda x: format_number(x))

# Mostrar la tabla de costos de decisión de las opciones de inversión

print(
    "\nSi la DEMANDA CRECE y NO INV. se tienen dos opciones para t2: (Costos de Decisión de Opciones de Inversión AL y GD en t2):")
print(df_decision_costs_options)

abandon_options = {
    'Decision': ['Abandonar'],
    'Valor': [Abandono]
}

df_abandon_options = pd.DataFrame(abandon_options)

df_abandon_options['Valor'] = df_abandon_options['Valor'].apply(lambda x: format_number(x))

print("\nSi la DEMANDA NO CRECE e INVIERTO EN GD, para t2 puedo abandonar:")
print(df_abandon_options)

decision_costs_abandon = {
    'Costo de Decisión': ['CD Abandonar'],
    'Valor': [CD_opc_Ab]
}

df_decision_costs_abandon = pd.DataFrame(decision_costs_abandon)

df_decision_costs_abandon['Valor'] = df_decision_costs_abandon['Valor'].apply(lambda x: format_number(x))

print("\nSi la DEMANDA NO CRECE e INVIERTO EN GD, para t2 puedo abandonar: (Costos de Decisión de Abandonar):")
print(df_decision_costs_abandon)

# Variable extra: CENS_15
def calculate_cens_15(CENS, ENS, t, r):
    cens_15 = sum((ENS * CENS * t) / (1 + r) ** j for j in range(1, 16))
    return cens_15

# Variable extra: CIALs
def calculate_cials(DEN, PEN, DPN, PPN, PTX, PDX, l, r, t):
    cials = sum((DEN * PEN * t + DPN * PPN + PTX * DPN + PDX * DPN * l) / (1 + r) ** j for j in range(1, 16))
    return cials

CENS_15 = calculate_cens_15(CENS, ENS, t, r)
CIALs = calculate_cials(DEN, PEN, DPN, PPN, PTX, PDX, l, r, t)

extra_results = {
    'Cálculo': ['CENS_{15}', 'CIALs'],
    'Valor': [CENS_15, CIALs]
}

df_extra_results = pd.DataFrame(extra_results)

df_extra_results['Valor'] = df_extra_results['Valor'].apply(lambda x: format_number(x))

print("\nTabla Extra de Resultados:")
print(df_extra_results)

# Función para formatear números sin notación científica y con separador de miles
def format_number(x):
    if isinstance(x, (int, float)):
        return f"{x:,.2f}"  # Formato con separador de miles y dos decimales
    return x

# Aqui se realiza la comparacion de la flexibilidad que aporta la GD v/s el enfoque tradicional dado por invertir en alimentadores

# Analisis de las Opciones GD
Opc_Aban = -(max(CIGD * DPN * α - CENS_15, 0) * Ps + max(CIGD * DPN * α, 0) * Pm)
Opc_AL = CIALs
Opc_Aban_AL = Opc_Aban + Opc_AL
Opciones_GD = Ps * min(Opc_Aban, Opc_AL, Opc_Aban_AL) + Pm * min(Opc_Aban, Opc_AL, Opc_Aban_AL)

# Opción de Diferir (FGD(t1,GD))

CENS1 = CENS * ENS * t * (1 + β)
Opc_Diferir = Ps * min(CIGD * DPN + CENS1, CENS_15) + Pm * 0

# Flexibilidad que aporta la GD (FGD(t0,CIGD))

FGD_t0_CIGD = min(CIGD * DPN + Opciones_GD / (1 + r), Opc_Diferir / (1 + r))

# Tabla para los resultados de Flexibilidad que aporta la GD

opciones_gd = {
    'Cálculo': ['Opc. Aban.', 'Opc. AL.', 'Opc. Aban. + Opc. AL.'],
    'Valor': [Opc_Aban, Opc_AL, Opc_Aban_AL]
}

fgd_t1_gd = {
    'Cálculo': ['Opc. Diferir'],
    'Valor': [Opc_Diferir]
}

flexibilidad_gd = {
    'Cálculo': ['FGD(t0,CIGD)'],
    'Valor': [FGD_t0_CIGD]
}

df_opciones_gd = pd.DataFrame(opciones_gd)
df_fgd_t1_gd = pd.DataFrame(fgd_t1_gd)
df_flexibilidad_gd = pd.DataFrame(flexibilidad_gd)

df_opciones_gd['Valor'] = df_opciones_gd['Valor'].apply(format_number)
df_fgd_t1_gd['Valor'] = df_fgd_t1_gd['Valor'].apply(format_number)
df_flexibilidad_gd['Valor'] = df_flexibilidad_gd['Valor'].apply(format_number)

print("\nOpciones GD")
print(df_opciones_gd)

print("\nFGD(t1,GD)")
print(df_fgd_t1_gd)

print("\nFlexibilidad que aporta la GD")
print(df_flexibilidad_gd)

# Variables extra: CIALm
def calculate_cialm(PDX, DPN, l, PTX, PPN, r):
    numerador = (PDX * DPN * l) + (PTX * DPN) + (DPN * PPN)
    cialm = sum(numerador / (1 + r) ** j for j in range(1, 16))
    return cialm

CIALm = calculate_cialm(PDX, DPN, l, PTX, PPN, r)

# ENFOQUE TRADICIONAL
def calculate_enfoque_tradicional(Ps, CIALs, Pm, CIALm):
    return Ps * CIALs + Pm * CIALm

ENFOQUE_TRADICIONAL = calculate_enfoque_tradicional(Ps, CIALs, Pm, CIALm)

cialm_results = {
    'Cálculo': ['CIALm'],
    'Valor': [CIALm]
}

df_cialm = pd.DataFrame(cialm_results)
df_cialm['Valor'] = df_cialm['Valor'].apply(
    lambda x: f"{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))

enfoque_tradicional_results = {
    'Cálculo': ['ENFOQUE TRADICIONAL'],
    'Valor': [ENFOQUE_TRADICIONAL]
}

df_enfoque_tradicional = pd.DataFrame(enfoque_tradicional_results)
df_enfoque_tradicional['Valor'] = df_enfoque_tradicional['Valor'].apply(
    lambda x: f"{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))

print("\nTabla Extra de Resutados N°2 de:")
print(df_cialm)

print("\nTabla del Efoque tradicional dado por Inversion en Alimentadores:")
print(df_enfoque_tradicional)
