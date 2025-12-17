import pandas as pd
import streamlit as st
data=pd.read_excel(r"https://d1b4gd4m8561gs.cloudfront.net/sites/default/files/caracteristicas-financieras-titulos-vigentes.xlsx",skiprows=3)
suma=data[data["Categoría del Título"]=="TIDIS"]["Saldo en circulación"].sum()
st.write(f"Suma TIDIS: ${suma:,.2f}")
