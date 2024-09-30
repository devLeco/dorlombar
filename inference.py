"""ESTAMOS EM PRODUCAO
1.CARREGAR O MODELO TREINADO
2. CARREGAR O DATASET A SER INFERIDO
3. EXECUTAR A INFERENCIA
4. APRESENTAR OS RESULTADOS
"""

import joblib
model = joblib.load(r'I:\Drive\MBA PUC\API\dorlombar\model\spine_model.pkl')

import pandas as pd
data = pd.read_csv(r'I:\Drive\MBA PUC\API\dorlombar\data\Dataset_spine_unknown.csv')

inferences = model.predict(data)

print(inferences)
data['previsoes'] = inferences
data.to_csv(r'I:\Drive\MBA PUC\API\dorlombar\model\inferences.csv',index=False)