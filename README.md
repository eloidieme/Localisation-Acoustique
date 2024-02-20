# Localisation Acoustique
# Pôle Projet Data Science 2023

## AVANT D'UTILISER LE CODE

Exécuter les commandes suivantes:
conda create --name loca-env --file requirements_conda.txt  
conda activate loca-env
pip install -r requirements_pip.txt 

Pour libusb, effectuer:
conda install -c conda-forge libusb
conda install -c conda-forge/label/cf202003 libusb

Ouvrir le dossier "C:\ProgramData\" et faire clique droit sur "\Anaconda3"
Cliquer sur security
Cocher toutes les boxes

S'assurer que le fichier .dll est dans le path de l'environnement

Avant de lancer le code après redémarrage, activer l'environnement avec:
conda activate loca-env
(Sélectionner également cet environnement dans les notebooks)

### Note
Pour rechercher les ports de la machine:
import usb.core
import usb.util
import usb.backend.libusb1
backend = usb.backend.libusb1.get_backend()
dev = usb.core.find(backend = backend, find_all = True)

## Pour la localisation en temps réel (2023)

Pour tester le temps réel sans avoir accés aux micros, lancer le fichier realtime_acoular.py
Ce fichier utilise Acoular pour générer des signaux et les localise en temps réel à l'aide de notre algorithme.

Si accès aux micros, d'abord utiliser les fonctions dans utils/fonctions_utils (verification_position) pour mapper les positions des micros par rapport aux colonnes dans la matrice d'enregistrement. 
Protocole préconisé: 
+ modifier la fonction coordonnees_micros dans utils/geometrie.py par rapport à la géométrie des microphones ou modifier directement le fichier xml dans utils/xml (ou en ajouter un) pour spécifier une nouvelle géométrie grâce à la fonction coordonnees_micros_xml
+ se placer dans positionnement.ipynb (pour des raisons de simplicité)
+ placer une source sonore (téléphone par ex.) près d'un micro en essayant de l'isoler au maximum
+ lancer la fonction verification_position (troisième cellule du notebook)
+ vérifier que le micro affiché est bien celui qui recevait du son
+ si ce n'est pas le cas, il faut changer la fonction ordre_micros dans utils/geometrie.py (les index donnés par verification_position aident à effectuer le mapping: index actuel à droite, index qu'on veut à gauche)

Une fois tout cela fait, lancer le fichier main.py. L'interface permet de spécifier plusieurs paramètres. Par défaut(avec le signal "Sinus"), un signal sinusoïdal est généré par Acoular et mis à jour en temps réel avec d'être détecté par la méthode choisie. Pour récupérer un signal des micros, il faut choisir le signal "Enregistrement". Pour l'instant, on ne fait pas usage de threading pour découpler l'interface et le calcul, d'où l'absence de réponse des boutons lors du calcul. C'est une piste d'amélioration, mais peut introduire des restrictions au niveau du calcul (concernant la possibilité d'arrêter la localisation et de la reprendre dans une même instance du programme).

## Autres

Le fichier signal_gen_acoular.py permet de générer des signaux à l'aide d'Acoular pour tester la localisation sans les micros.
Les positions de micros considérés par Acoular sont données dans le fichier utils/xml/mu32.xml
C'est un fichier XML et l'ordre d'entrée des micros donne l'ordre dans lequel Acoular renvoie les signaux de chaque micro. 
Dans l'idéal, cet ordre devrait coïncider avec l'ordre dans lequel le boîtier renvoie les signaux des micros, afin de pouvoir tester correctement la fonction ordonnant les micros.
Par ailleurs, le "bon" ordre est simplement celui du vecteur coordonnees_micros. Il faut s'y référer pour savoir comment ré-ordonner les micros. C'est important lors du calcul des distances aux sources de chaque micro.

signal_test.py permet de produire des signaux à partir des haut-parleurs de l'ordinateur. Audacity fait pareil (Générer -> Tonalité -> Sinus ou Générer -> Bruit -> Bruit blanc).

