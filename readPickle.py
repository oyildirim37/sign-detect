import pickle

# Pickle-Datei öffnen und lesen
with open("data.pickle", "rb") as file:  # "rb" steht für "read binary"
    data = pickle.load(file)

# Eingelesene Daten anzeigen
print(data)
