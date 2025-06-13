import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# ---------- Paso 1: Cargar los datos ----------
train = pd.read_csv('KDDTrain+.csv', header=None)
test = pd.read_csv('KDDTest+.csv', header=None)

# Agregamos nombres de columnas
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "attack_type"
]

train.columns = columns
test.columns = columns

# ---------- Paso 2: Preprocesar ----------
# Clasificamos si la conexión es normal o ataque (binario)
train['label'] = train['label'].apply(lambda x: 0 if x == 'normal' else 1)
test['label'] = test['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Concatenamos train y test para codificar variables categóricas
combined = pd.concat([train, test], axis=0)

# Codificamos las columnas categóricas
for col in ['protocol_type', 'service', 'flag']:
    encoder = LabelEncoder()
    combined[col] = encoder.fit_transform(combined[col])

# Separamos nuevamente
train = combined[:len(train)]
test = combined[len(train):]

# Separar variables predictoras y etiquetas
X_train = train.drop(['label'], axis=1)
y_train = train['label']
X_test = test.drop(['label'], axis=1)
y_test = test['label']

# Escalamos los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- Paso 3: Crear el modelo ----------
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Clasificación binaria

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ---------- Paso 4: Entrenar ----------
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# ---------- Paso 5: Evaluar ----------
y_pred = (model.predict(X_test) > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {acc:.4f}')
print(f'F1-score: {f1:.4f}')
