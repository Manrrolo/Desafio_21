import pandas as pd

# Definimos los nombres de las columnas
column_names = ['Time', 'Document', 'Question', 'Response']

# Leemos el CSV indicando los nombres de las columnas
df = pd.read_csv('log.csv', header=None, names=column_names)

# Convertimos la columna 'Time' a datetime
df['Time'] = pd.to_datetime(df['Time'])

# Crea una nueva columna 'Hour' que contiene la hora del día
df['Hour'] = df['Time'].dt.hour

# Agrupa los datos por 'Document' y 'Hour'
grouped = df.groupby(['Document', 'Hour'])

# Imprime cada grupo
for name, group in grouped:
    print(f"Document: {name[0]}, Hour: {name[1]}")
    print(group)
    print("\n---\n")
