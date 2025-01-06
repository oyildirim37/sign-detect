import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Lade die Daten
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

# Setze die maximale Länge für Trunkierung
max_length = 42  # Ändere dies nach Bedarf

# Trunkieren der Daten
data_truncated = [d[:max_length] for d in data]

# Konvertiere die trunkierte Liste in ein NumPy Array
data_truncated = np.array(data_truncated)

# Konvertiere Labels in ein NumPy Array
labels = np.array(labels)

# Aufteilen in Trainings- und Testdaten
x_train, x_test, y_train, y_test = train_test_split(data_truncated, labels, test_size=0.2, shuffle=True, stratify=labels)

# Modell initialisieren und trainieren
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Vorhersagen machen
y_predict = model.predict(x_test)

# Genauigkeit berechnen
score = accuracy_score(y_test, y_predict)

print('{}% der Proben wurden korrekt klassifiziert!'.format(score * 100))

# Modell speichern
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
