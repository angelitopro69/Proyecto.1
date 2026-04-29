import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, t
from data_download import descargar_datos

# Inciso a 
st.set_page_config(page_title="Proyecto Finanzas AAPL", layout="wide")

st.title("Análisis del activo financiero AAPL")

st.markdown("Datos históricos de Apple Inc. desde Yahoo Finance")

ticker = st.text_input("Activo", "AAPL")

inicio = st.date_input("Fecha inicio", pd.to_datetime("2010-01-01"))
fin = st.date_input("Fecha fin", pd.to_datetime("2025-01-01"))

if st.button("Descargar datos"):
    try:
        data = descargar_datos(ticker, inicio, fin)

        st.success("Datos descargados correctamente")

        st.subheader("Vista previa de datos")
        st.dataframe(data.head())

        # Inciso b 
        st.subheader("Estadísticas de los rendimientos")

        returns = data['Returns']

        mean = returns.mean()
        skew = returns.skew()
        kurt = returns.kurt()

        st.write(f"Media: {mean:.6f}")
        st.write(f"Sesgo: {skew:.6f}")
        st.write(f"Exceso de curtosis: {kurt:.6f}")

        niveles = [0.95, 0.975, 0.99]

        # Inciso c historico
        st.subheader("Value at Risk y Expected Shortfall histórico")

        resultados_hist = []

        for alpha in niveles:
            var = np.percentile(returns, (1 - alpha) * 100)
            es = returns[returns <= var].mean()

            resultados_hist.append({
                "Nivel de confianza": alpha,
                "VaR": var,
                "ES": es
            })

        tabla_hist = pd.DataFrame(resultados_hist)
        st.table(tabla_hist)

        # Inciso c parametrico
        st.subheader("Value at Risk y Expected Shortfall paramétrico")

        mu = returns.mean()
        sigma = returns.std()

        resultados_param = []

        df = 5  # grados de libertad

        for alpha in niveles:
            z = norm.ppf(1 - alpha)

            
            var_normal = mu + sigma * z
            es_normal = mu - sigma * (norm.pdf(z) / (1 - alpha))

    
            t_quantile = t.ppf(1 - alpha, df)
            var_t = mu + sigma * t_quantile
            es_t = mu - sigma * ((t.pdf(t_quantile, df) / (1 - alpha)) * ((df + t_quantile**2) / (df - 1)))

            resultados_param.append({
                "Nivel": alpha,
                "VaR Normal": var_normal,
                "ES Normal": es_normal,
                "VaR t-Student": var_t,
                "ES t-Student": es_t
            })

        tabla_param = pd.DataFrame(resultados_param)
        st.table(tabla_param)

        # Inciso c Monte Carlo
        st.subheader("Value at Risk y Expected Shortfall Monte Carlo")

        np.random.seed(42)
        simulaciones = 10000

        resultados_mc = []

        for alpha in niveles:
            simulados = np.random.normal(mu, sigma, simulaciones)

            var_mc = np.percentile(simulados, (1 - alpha) * 100)
            es_mc = simulados[simulados <= var_mc].mean()

            resultados_mc.append({
                "Nivel": alpha,
                "VaR Monte Carlo": var_mc,
                "ES Monte Carlo": es_mc
            })

        tabla_mc = pd.DataFrame(resultados_mc)
        st.table(tabla_mc)
       # INCISO d
        # =========================
        
        st.subheader("VaR y ES con Rolling Window (252 días)")
        
        window = 252
        niveles = [0.95, 0.99]
        
        # Listas para guardar resultados
        resultados = {
            "Returns": [],
            "VaR_hist_95": [],
            "ES_hist_95": [],
            "VaR_hist_99": [],
            "ES_hist_99": [],
            "VaR_param_95": [],
            "ES_param_95": [],
            "VaR_param_99": [],
            "ES_param_99": []
        }
        
        for t in range(window, len(returns)):
            data_window = returns[t-window:t]
        
            mu = np.mean(data_window)
            sigma = np.std(data_window)
        
            # Retorno real (el que se quiere predecir)
            resultados["Returns"].append(returns.iloc[t])
        
            for alpha in niveles:
                # HISTÓRICO
                var_hist = np.percentile(data_window, (1 - alpha) * 100)
                es_hist = data_window[data_window <= var_hist].mean()
        
                # PARAMÉTRICO (normal)
                z = norm.ppf(1 - alpha)
                var_param = mu + sigma * z
                es_param = mu - sigma * (norm.pdf(z) / (1 - alpha))
        
                if alpha == 0.95:
                    resultados["VaR_hist_95"].append(var_hist)
                    resultados["ES_hist_95"].append(es_hist)
                    resultados["VaR_param_95"].append(var_param)
                    resultados["ES_param_95"].append(es_param)
        
                elif alpha == 0.99:
                    resultados["VaR_hist_99"].append(var_hist)
                    resultados["ES_hist_99"].append(es_hist)
                    resultados["VaR_param_99"].append(var_param)
                    resultados["ES_param_99"].append(es_param)
        
        # Convertir a DataFrame
        df_rolling = pd.DataFrame(resultados)
        
        # recuperar fechas
        df_rolling.index = returns.index[window:]
        
        # =========================
        # GRÁFICA PRINCIPAL
        # =========================
        
        st.subheader("Serie de tiempo: Returns, VaR y ES")
        
        st.line_chart(df_rolling)
        
        # =========================
        # Grafica mas clara donde muestra Var vs Returns 
        # =========================
        
        st.subheader("VaR vs Returns (más claro)")
        
        st.line_chart(df_rolling[[
            "Returns",
            "VaR_hist_95", "VaR_hist_99",
            "VaR_param_95", "VaR_param_99"
        ]])
        
        st.write("Últimos valores:")
        st.dataframe(df_rolling.tail())
        

        #inciso E
        st.subheader("Análisis de Eficiencia (Violaciones)")

def calcular_violaciones(df_rolling, returns_col="Returns"):
    
    # Se agrega la lista con las metricas que se van a utilizar
    metricas = [
        "VaR_hist_95", "ES_hist_95", "VaR_hist_99", "ES_hist_99",
        "VaR_param_95", "ES_param_95", "VaR_param_99", "ES_param_99"
    ]
    
    # Se agrega una libreria con los nombres comletos de las metricas
    nombres = {
        "VaR_hist_95": "Histórico VaR 95%",
        "ES_hist_95": "Histórico ES 95%",
        "VaR_hist_99": "Histórico VaR 99%",
        "ES_hist_99": "Histórico ES 99%",
        "VaR_param_95": "Paramétrico VaR 95%",
        "ES_param_95": "Paramétrico ES 95%",
        "VaR_param_99": "Paramétrico VaR 99%",
        "ES_param_99": "Paramétrico ES 99%",
    }
    
    total_obs = len(df_rolling)
    resultados = []

    for metrica in metricas:
        violaciones = (df_rolling[returns_col] < df_rolling[metrica]).sum()
        porcentaje = (violaciones / total_obs) * 100
        
        resultados.append({
            "Medida de Riesgo": nombres[metrica],
            "Violaciones": int(violaciones),
            "Porcentaje (%)": f"{porcentaje:.2f}"
        })

    return pd.DataFrame(resultados)



tabla_eficiencia = calcular_violaciones(df_rolling)

st.table(tabla_eficiencia)


st.subheader("Porcentaje de violaciones")


tabla_plot = tabla_eficiencia.copy()
tabla_plot["Porcentaje (%)"] = tabla_plot["Porcentaje (%)"].astype(float)

st.bar_chart(tabla_plot.set_index("Medida de Riesgo")["Porcentaje (%)"])


        # Inciso f VaR con volatilidad movil
        st.subheader("VaR con volatilidad móvil")

        sigma_t = returns.rolling(252).std().shift(1)

        df_f = pd.DataFrame(index=returns.index)
        df_f["Returns"] = returns

        for alpha in [0.05, 0.01]:
            q = norm.ppf(alpha)
            var = q * sigma_t

            if alpha == 0.05:
                df_f["VaR_95"] = var
            else:
                df_f["VaR_99"] = var

        df_f = df_f.dropna()

        st.line_chart(df_f)
        st.line_chart(df_f[["Returns", "VaR_95", "VaR_99"]])


        
        st.subheader("Serie de precios")
        st.line_chart(data['Precio'])

        st.subheader("Rendimientos logarítmicos")
        st.line_chart(data['Returns'])

        st.subheader("Distribución de rendimientos")
        st.bar_chart(data['Returns'])

    except Exception as e:
        st.error(f"Error: {e}")
