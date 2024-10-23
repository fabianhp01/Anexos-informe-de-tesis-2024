import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def calc_valor_presente_neto(lista_valores, r, valor_2025):
    valor_presente_neto = valor_2025
    value = 0
    for i in range(1, len(lista_valores)):
        value = lista_valores[i] / (1 + r) ** i
        valor_presente_neto += value
    return valor_presente_neto


def valor_de_opcion(precios_estocastico):
    # Variables de entrada
    data = {
        'Columna1': ['Demanda de energía nueva', 'Precio de energía nueva', 'Demanda de potencia nueva',
                     'Precio de potencia nueva', 'Peaje TRANSMISIÓN', 'Peaje DISTRIBUCIÓN',
                     'Costo energía no suministrado', 'Energia no suministrada', 'Tiempo (en hrs al año)',
                     '% que la demanda crezca', '% que la demanda no crezca', 'Costo de inversión en GD:',
                     'longitud de alimentador en km', 'Tasa de retorno', 'Depreciación', 'Inflación'],
        'Columna2': ['DEN', 'PEN', 'DPN', 'PPN', 'PTX', 'PDX', 'CENS', 'ENS', 't', 'Ps', 'Pm', 'CIGD', 'l', 'r', 'α',
                     'β'],
        'Columna3': [2, 40, 2, 10, 48, 1_000, 325, 2, 1_920, 0.1, 0.9, 1_400_000, 10, 0.05, 0.95, 0.07],
        'Columna4': ['MWh', 'USD/MWh', 'MW', 'USD/MW', 'USD/MWaño', 'USD/MWaño', 'USD/MWh', 'MWh', 'hrs-año', '-', '-',
                     'USD/MW', 'km', '-', '-', '-']
    }

    # Formato ##########################################################################################################
    df = pd.DataFrame(data)

    df.index = range(1, len(df) + 1)

    def format_number(x):
        if isinstance(x, (int, float)):
            return f"{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        return x

    df_formatted = df.applymap(format_number)
    ####################################################################################################################

    #print("Variables de entrada:")
    #print(df_formatted)

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

    #print("\nTabla Extra de Resultados:")
    #print(df_extra_results)

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

    #print("\nOpciones GD")
    #print(df_opciones_gd)

    #print("\nFGD(t1,GD)")
    #print(df_fgd_t1_gd)

    #print("\nFlexibilidad que aporta la GD")
    #print(df_flexibilidad_gd)

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

    #print("\nTabla Extra de Resutados N°2 de:")
    #print(df_cialm)

    #print("\nTabla del Efoque tradicional dado por Inversion en Alimentadores:")
    #print(df_enfoque_tradicional)

    # Hasta aqui llega el CASO DE ESTUDIO A PEQUEÑA ESCALA realizado para la mesa 3 #
    #################################################################################
    # Mejorando CASO DE ESTUDIO A PEQUEÑA ESCALA para mesa 4:
    # 1) Recopilacion de datos de entrada final (dimensionando planta FV).
    # 2) Hiptesis de estudio (mediante proyeccion del PMM).
    # 3) Implementacion y validacion del modelo (Se realizo agragando 2 nuevas opciones).

    # *Se realiza proyeccion de el precio de energia considerando el precio medio mercado (PMM desde el 2006 hasta el 2024)
    # calculando una variacion promedio para proyectar un precio de proyeccion hasta el 2039

    # Datos obtenidos de excel de CNE sobre PMM historico
    years = list(range(2006, 2040))
    precios_chile = [
        27.7836, 40.9173333333333, 53.2598888888889, 49.2798888888889, 54.1761111111111,
        55.9303333333333, 54.5758888888889, 52.6415555555556, 58.4066666666667, 61.8434444444444,
        62.8471690066959, 63.6845555555555, 62.8106666666667, 68.7557777777778, 74.161,
        73.2941111111111, 88.8085555555556, 103.320111111111, 103.4245
    ]

    # Valores calculados de 2025 a 2039
    precios_futuros = [
        112.0062262, 121.3000277, 131.3649894, 142.2650989, 154.0696532,
        166.8536994, 180.6985115, 195.6921074, 211.9298082, 229.5148446,
        248.5590127, 269.1833851, 291.5190804, 315.7080969, 341.9042154
    ]

    precios_chile += precios_futuros

    variacion = [None] + [round((precios_chile[i] - precios_chile[i - 1]) / precios_chile[i - 1] * 100) for i in
                          range(1, len(precios_chile))]

    # Conversión a USD/MWh 1000 pesos = 1 USD, 1 kWh = 0.001 MWh
    precios_usd = [round(precio * 1.0, 0) for precio in precios_chile]

    data = {
        "Año": years,
        "PMM($/kWh)": precios_chile,
        "Variación (%)": variacion,
        "PMM(USD/MWh)": precios_usd
    }

    df = pd.DataFrame(data)

    df.name = "Proyección de precios de energía para los próximos 15 años"

    #print("\nProyección de precios de energía para los próximos 15 años:")
    #print(df)

    ####################################################################################################################
    # *En esta seccion se calcula como seria la venta de excedentes de energia. Cabe destacar que los ingresos los consideramos negativos

    years = list(range(2025, 2040))

    DPN = 2
    t = 1920

    venta_exc = []
    valor_nuevo = []
    acumulado = 0

    for precio in precios_estocastico:
        valor = -DPN * t * precio
        venta_exc.append(valor)

        valor_nuevo.append(valor)

    venta_exc_data = {
        'Año': years,
        'Venta exc (USD)': venta_exc
    }

    df_venta_exc = pd.DataFrame(venta_exc_data)

    df_venta_exc['Venta exc (USD)'] = df_venta_exc['Venta exc (USD)'].apply(
        lambda x: f"{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))

    #print("\nVenta de Excedentes para los años 2025 a 2039:")
    #print(df_venta_exc)

    # Valor extra: CENS_15
    def calculate_cens_15(CENS, ENS, t, r):
        cens_15 = sum((ENS * CENS * t) / (1 + r) ** j for j in range(1, 16))
        return cens_15

    # Valor extra: CIALs
    def calculate_cials(DEN, PEN, DPN, PPN, PTX, PDX, l, r, t):
        cials = sum((DEN * PEN * t + DPN * PPN + PTX * DPN + PDX * DPN * l) / (1 + r) ** j for j in range(1, 16))
        return cials

    CENS_15 = calculate_cens_15(CENS, ENS, t, r)
    CIALs = calculate_cials(DEN, PEN, DPN, PPN, PTX, PDX, l, r, t)

    ####################################################################################################################
    # Nueva Opcion 2AL + venta exc.
    # Para obtener el valor de esta nueva opcion se analiza nuevamente la Flexibilidad que aporta la GD

    Opc_Aban = -(max(CIGD * DPN * α - CENS_15, 0) * Ps + max(CIGD * DPN * α, 0) * Pm)
    Opc_AL = CIALs
    Opc_Aban_AL = Opc_Aban + Opc_AL
    Opciones_GD = Ps * min(Opc_Aban, Opc_AL, Opc_Aban_AL) + Pm * min(Opc_Aban, Opc_AL, Opc_Aban_AL)

    venta_exc_dict = dict(zip(years, venta_exc))

    # Analisis para las Opciones GD
    opciones_gd = {
        'Cálculo': ['Opc. Aban.', 'Opc. AL.', 'Opc. Aban. + Opc. AL.']
    }

    valores_opciones_gd = [Opc_Aban, Opc_AL, Opc_Aban_AL]

    for year in years:
        if year == 2025:
            opciones_gd['Cálculo'].append(f'Opc. 2AL + venta exc {year}')
            valores_opciones_gd.append(2 * CIALs + venta_exc_dict[year])
        else:
            opciones_gd['Cálculo'].append(f'Opc. 2AL + venta exc {year}')
            valores_opciones_gd.append(venta_exc_dict[year])

    opciones_gd['Valor'] = valores_opciones_gd

    df_opciones_gd = pd.DataFrame(opciones_gd)

    df_opciones_gd['Valor'] = df_opciones_gd['Valor'].apply(format_number)

    print("\nOpciones GD")
    print(df_opciones_gd)

    # Opción de Diferir (FGD(t1,GD))
    CENS1 = CENS * ENS * t * (1 + β)
    Opc_Diferir = Ps * min(CIGD * DPN + CENS1, CENS_15) + Pm * 0

    # Valo presente neto 2AL+venta exc
    VPN_de_opcion_2AL_mas_ventaexc = calc_valor_presente_neto(lista_valores=valor_nuevo, r=r,
                                                              valor_2025=valores_opciones_gd[3])
    valores_opciones_gd.append(VPN_de_opcion_2AL_mas_ventaexc)

    # Flexibilidad que aporta la GD (FGD(t0,CIGD))
    opcdiff = Opc_Diferir / (1 + r)
    opciones_gd_porVTD = (min(valores_opciones_gd) / (1 + r))
    result = (CIGD * DPN) + opciones_gd_porVTD
    arreglocomparar = [result, opcdiff]
    flex_gd = min(arreglocomparar)
    print("\nFlexibilidad que aporta la GD: {:,}".format(int(flex_gd)))

    # Variable extra: CIALm
    def calculate_cialm(PDX, DPN, l, PTX, PPN, r):
        numerador = (PDX * DPN * l) + (PTX * DPN) + (DPN * PPN)
        cialm = sum(numerador / (1 + r) ** j for j in range(1, 16))
        return cialm

    CIALm = calculate_cialm(PDX, DPN, l, PTX, PPN, r)

    # ENFOQUE TRADICIONAL
    def calculate_enfoque_tradicional(Ps, CIALs, Pm, CIALm):
        return Ps * CIALs + Pm * CIALm

    ENFOQUE_TRADICIONAL = calculate_enfoque_tradicional(Ps, CIALs, Pm, CIALm)

    df_cialm = pd.DataFrame(cialm_results)
    df_cialm['Valor'] = df_cialm['Valor'].apply(
        lambda x: f"{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))

    enfoque_tradicional_results = {
        'Cálculo': ['ENFOQUE TRADICIONAL'],
        'Valor': [ENFOQUE_TRADICIONAL]
    }

    df_enfoque_tradicional = pd.DataFrame(enfoque_tradicional_results)
    df_enfoque_tradicional['Valor'] = df_enfoque_tradicional['Valor'].apply(format_number)

    print("\nENFOQUE TRADICIONAL:")
    print(df_enfoque_tradicional)

    # Se evaluara la inversion en GD + venta de excedentes
    # Cuando la demanda no crece y se invirtio en +E y P, por ende, inversion en alimentador

    def calcular_venta_exc(DPN, t, precios_estocastico):
        return -(DPN * t * precios_estocastico)

    # Año 2025
    def calcular_inversion_gd_venta_2025(CIALs, CIGD, r, venta_exc):
        return CIALs + (CIGD * DPN / (1 + r)) + venta_exc

    # Para los años desde 2026 en adelante
    def calcular_inversion_gd_venta_exc_anual(venta_exc):
        return venta_exc

    resultados_gd_venta_exc = []

    # Valores para cada año desde 2025 hasta 2039
    print("\nGD + venta de excedentes")
    for year in range(2025, 2040):
        if year == 2025:
            # Obtener el valor de precios_usd_mwh para 2025
            precio_2025_GD = precios_estocastico[year - 2025]
            venta_exc_2025 = calcular_venta_exc(DPN, t, precio_2025_GD)
            resultado_GD_2025 = calcular_inversion_gd_venta_2025(CIALs, CIGD, r, venta_exc_2025)
            resultados_gd_venta_exc.append({'Año': year, 'Inversión GD + Venta Exc': resultado_GD_2025})

        else:
            # Obtener el valor de precios_usd_mwh para el año correspondiente (2026-2039)
            precio_anual = precios_estocastico[year - 2025]
            venta_exc_anual = calcular_venta_exc(DPN, t, precio_anual)
            resultado_anual = calcular_inversion_gd_venta_exc_anual(venta_exc_anual)
            resultados_gd_venta_exc.append({'Año': year, 'Inversión GD + Venta Exc': resultado_anual})

    valor_acumulado = resultado_GD_2025
    value = 0
    # Valor presente neto de GD+venta exc
    VPN_GD_venta_exc = calc_valor_presente_neto(lista_valores=valor_nuevo, r=r, valor_2025=resultado_GD_2025)
    resultados_gd_venta_exc.append({'Año': 'VPN', 'Inversión GD + Venta Exc': VPN_GD_venta_exc})

    for resultado in resultados_gd_venta_exc:
        if resultado['Año'] == 'VPN':
            print(f"VPN, Inversión GD + Venta Exc: {resultado['Inversión GD + Venta Exc']:,.2f}")
        else:
            print(f"Año: {resultado['Año']}, Inversión GD + Venta Exc: {resultado['Inversión GD + Venta Exc']:,.2f}")

    # Se evaluara la inversion en AL + venta de excedentes
    # Cuando la demanda no crece y se invirtio GD

    def calcular_venta_exc(DPN, t, precios_estocastico):
        return -(DPN * t * precios_estocastico)

    # año 2025
    def calcular_inversion_gd_venta_2025(CIALs, CIGD, r, venta_exc):
        return CIGD * DPN + (CIALs / (1 + r)) + venta_exc

    # Para los años desde 2026 en adelante
    def calcular_inversion_gd_venta_exc_anual(venta_exc):
        return venta_exc

    resultados_AL_venta_exc = []

    # Valores para cada año desde 2025 hasta 2039
    print("\nAL + venta de excedentes")
    for year in range(2025, 2040):
        if year == 2025:
            # valor de precios_usd_mwh para 2025
            precio_2025_AL = precios_estocastico[year - 2025]
            venta_exc_2025 = calcular_venta_exc(DPN, t, precio_2025_AL)
            resultado_AL_2025 = calcular_inversion_gd_venta_2025(CIALs, CIGD, r, venta_exc_2025)
            resultados_AL_venta_exc.append({'Año': year, 'Inversión AL + Venta Exc': resultado_AL_2025})
        else:
            # valor de precios_usd_mwh para el año 2026-2039
            precio_anual = precios_estocastico[year - 2025]
            venta_exc_anual = calcular_venta_exc(DPN, t, precio_anual)
            resultado_anual = calcular_inversion_gd_venta_exc_anual(venta_exc_anual)
            resultados_AL_venta_exc.append({'Año': year, 'Inversión AL + Venta Exc': resultado_anual})

    # Valor presente neto de AL+venta exc
    VPN_AL_venta_exc = calc_valor_presente_neto(lista_valores=valor_nuevo, r=r, valor_2025=resultado_AL_2025)
    resultados_AL_venta_exc.append({'Año': 'VPN', 'Inversión AL + Venta Exc': VPN_AL_venta_exc})

    for resultado in resultados_AL_venta_exc:
        if resultado['Año'] == 'VPN':
            print(f"VPN, Inversión AL + Venta Exc: {resultado['Inversión AL + Venta Exc']:,.2f}")
        else:
            print(f"Año: {resultado['Año']}, Inversión AL + Venta Exc: {resultado['Inversión AL + Venta Exc']:,.2f}")

    # costos de decesion totales
    costo_decision_total_gd = Ps * flex_gd + Pm * VPN_AL_venta_exc
    print(f"\nCosto de decisión total (para GD): {costo_decision_total_gd:,.2f}")

    costo_decision_total_al = Ps * ENFOQUE_TRADICIONAL + Pm * ((VPN_GD_venta_exc) / (1 + r))
    print(f"\nCosto de decisión total (para AL): {costo_decision_total_al:,.2f}")

    return costo_decision_total_gd, costo_decision_total_al

def gbm_simulation(T, mu, sigma, S0, N):
    dt = T / N
    t = np.linspace(0, T, N)
    W = np.random.normal(0, np.sqrt(dt), N)
    W = np.cumsum(W)  # Movimiento Browniano estándar
    S = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)

    return t, S

def precio(NR):
    # Parámetros
    T = 1.0
    mu = 0.073022
    sigma = 0.1162675 #inicialmente esto es 0.1
    S0 = 150 #inicialmente esto es 103, pero si este valor aumenta CDtotalGD y AL aumentan
    N = 15

    # Lista parra almacenar los 1000 resultados
    precios_energia_aleatorios = []

    # Grafico
    for i in range (NR):
        t, S = gbm_simulation(T, mu, sigma, S0, N)
        precios_energia_aleatorios.append(S)
        plt.plot(t, S,label=f'Simulación {i+1}')

    precios_energia = np.array(precios_energia_aleatorios)

    # Archivo CSV para precios energia aleatorios
    np.savetxt("precios_energia_aleatorios.csv", precios_energia_aleatorios, delimiter=",")

    plt.title('Simulación de Movimiento Browniano Geométrico')
    plt.xlabel('Tiempo')
    plt.ylabel('Precio')
    plt.show()

    return precios_energia

#precios_usd_mwh = [
    #112, 121, 131, 142, 154,
    #167, 181, 196, 212, 230,
    #249, 269, 292, 316, 342
#]

#precios_energia_aleatorios = pd.read_csv("precios_energia_aleatorios.csv", header=None)
#precios_usd_mwh = precios_energia_aleatorios.values.flatten().tolist()

NR =1000
precio_est=precio(NR)
x = np.zeros(NR)
y = np.zeros(NR)
for nr in range(0, NR):

    precio_nr=precio_est[nr]
    x[nr],y[nr]= valor_de_opcion(precio_nr)

np.savetxt("x[nr].csv", x, delimiter=",")
np.savetxt("y[nr].csv", y, delimiter=",")

xmean = np.mean(x)
ymean = np.mean(y)

print("")
print(f"CDtotalGD medio posterior a las mil simulaciones: {xmean:,.2f}")
print(f"CDtotalAL medio posterior a las mil simulaciones: {ymean:,.2f}")


# Datos
valor_opcion_GD = x
valor_opcion_AL = y

# densidad utilizando gaussian_kde
kde_GD = gaussian_kde(valor_opcion_GD)
kde_AL = gaussian_kde(valor_opcion_AL)

x_densidad = np.linspace(min(valor_opcion_GD), max(valor_opcion_GD*2), 1000)  # Rango de precios
densidad = kde_GD(x_densidad)

y_densidad = np.linspace(min(valor_opcion_AL), max(valor_opcion_AL*2), 1000)  # Rango de precios
densidad_AL = kde_AL(y_densidad)

plt.figure()
plt.plot(x_densidad, densidad, color='blue', label='PDF GD')
plt.plot(y_densidad, densidad_AL, color='orange', label='PDF AL')

plt.title('Función de Densidad de Probabilidad de Precios de Energía')
plt.xlabel('Valor de opciones')
plt.ylabel('Densidad')
plt.legend()
plt.grid()
plt.show()

# Ordenar los datos
x_sorted = np.sort(x)
y_sorted = np.sort(y)

# CDF empírica
cdf_GD = np.arange(1, len(x_sorted)+1) / len(x_sorted)
cdf_AL = np.arange(1, len(y_sorted)+1) / len(y_sorted)

# Grafica
plt.figure()
plt.plot(x_sorted, cdf_GD, label='GD', color='blue')
plt.plot(y_sorted, cdf_AL, label='AL', color='orange')

plt.title('Función de Distribución Acumulada (CDF) Empírica')
plt.xlabel('Valor de opciones')
plt.ylabel('Probabilidad acumulada')
plt.legend()
plt.show()
