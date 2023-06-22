import pandas as pd

# Definimos los nombres de las columnas
column_names = ['Time', 'Document', 'Question', 'Response']

# Leemos el CSV indicando los nombres de las columnas
df = pd.read_csv('log.csv', header=None, names=column_names)

# Convertimos la columna 'Time' a datetime
df['Time'] = pd.to_datetime(df['Time'])

# Agrupa los datos por 'Document'
grouped = df.groupby('Document')

# Imprime cada grupo
for name, group in grouped:
    print(f"Document: {name}")
    print(group)
    print("\n---\n")
