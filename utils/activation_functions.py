import pandas as pd
from utils.fonctions_utils import echantillons

CONFIG = pd.read_csv('config/micros.csv', sep = ";").set_index('faisceau')
mic0 = bin(CONFIG.iloc[0, 0])
mic1 = bin(CONFIG.iloc[1, 0])
mic2 = bin(CONFIG.iloc[2, 0])
mic3 = bin(CONFIG.iloc[3, 0])

def configuration(dev):
    dev.set_configuration()

# va initialiser les param sur le boitier pour l'acquisition
def initialisation(dev):
    dev.ctrl_transfer(0x40, 0xC0, 0, 0, None)
    dev.ctrl_transfer(0x40, 0xC4, 0, 0, None)
    dev.ctrl_transfer(0x40, 0xB0, 0, 0, [0x00])
    dev.ctrl_transfer(0x40, 0xC4, 0, 0, None)
    dev.ctrl_transfer(0x40, 0xC0, 0, 0, None)


def activation(micros, dev, srinv, length):
    l0, l1, l2, l3 = echantillons(length) # va creer plusieurs taille d'echantillons
    #mic0, mic1, mic2, mic3 = make_micros()
    if micros is not None: # va mettre les micros si changement de l'interface
        mic0 = bin(int(micros[0], 2))
        mic1 = bin(int(micros[1], 2))
        mic2 = bin(int(micros[2], 2))
        mic3 = bin(int(micros[3], 2))
    dev.ctrl_transfer(0x40, 0xB1, 0, 0, [0x01, srinv])
    # longueur
    dev.ctrl_transfer(0x40, 0xB4, 0, 0, [0x04, l3, l2, l1, l0]) # envoie les echantillons
    # activation des micros
    dev.ctrl_transfer(0x40, 0xB3, 0, 0, [0x05, 0x00, 0x00, 0xFF]) # set les 4 faisceau
    dev.ctrl_transfer(0x40, 0xB3, 0, 0, [0x05, 0x00, 0x01, 0xFF])
    dev.ctrl_transfer(0x40, 0xB3, 0, 0, [0x05, 0x00, 0x02, 0xFF])
    dev.ctrl_transfer(0x40, 0xB3, 0, 0, [0x05, 0x00, 0x03, 0xFF])
    # résultat en int32
    dev.ctrl_transfer(0x40, 0xB1, 0, 0, [0x09, 0x00]) #[0x09, 0x00] for int32 and [0x09, 0x00] for float32 n'a pas fonctionne

def deactivation(dev):
    dev.ctrl_transfer(0x40, 0xB1, 0, 0, [0x03])
