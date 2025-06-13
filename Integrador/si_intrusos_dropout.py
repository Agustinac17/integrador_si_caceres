import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

train = pd.read_csv("KDDTrain+.csv", header=None)
test = pd.read_csv("KDDTest+.csv", header=None)

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

train['attack_type'] = train['label'].apply(lambda x: 0 if x == 'normal' else 1)
test['attack_type'] = test['label'].apply(lambda x: 0 if x == 'normal' else 1)

#Eliminar columnas no numéricas y que no aportan
train.drop(["protocol_type", "service", "flag", "label"], axis=1, inplace=True)
test.drop(["protocol_type", "service", "flag", "label"], axis=1, inplace=True)

X_train = train.drop("attack_type", axis=1)
y_train = train["attack_type"]
X_test = test.drop("attack_type", axis=1)
y_test = test["attack_type"]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Modelo con Dropout
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

#Compilar
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Entrenamiento
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

#Evaluar el modelo
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)


acc = accuracy_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)

print("Accuracy con Dropout:", round(acc, 4))
print("F1-score con Dropout:", round(f1, 4))
