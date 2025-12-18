import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
data=pd.read_excel(r"https://d1b4gd4m8561gs.cloudfront.net/sites/default/files/caracteristicas-financieras-titulos-vigentes.xlsx",skiprows=3)
suma=data[data["Categoría del Título"]=="TIDIS"]["Saldo en circulación"].sum()
st.write(f"Suma TIDIS: ${suma:,.2f}")
data["Mes de vencimiento"]=data["Fecha de vencimiento"]
plot=data[data["Categoría del Título"]=="TIDIS"][["Mes de vencimiento","Saldo en circulación"]].groupby(pd.Grouper(key="Mes de vencimiento", freq='M')).sum().plot.bar()
st.pyplot(plot.figure)
