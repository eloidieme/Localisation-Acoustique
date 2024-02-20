import numpy as np
from scipy.optimize import minimize
import scipy
import matplotlib.pyplot as plt
from scipy.signal import stft

def initialize_Lambda():
    # Initialize Lambda as an array of identity matrices 
    Lambda =  np.array([np.eye(M) for _ in range(N)])
    return Lambda

# Define the MCLP filter matrix G(ω)
def define_mclp_filter_matrix(Kl, M):
    # Kl: Prediction order
    # Initialize gm to [1, 1, ..., 1]
    gm = np.ones(Kl*M)
    G = np.array([gm for _ in range(M)])
    G=np.transpose(G)
    return G

# Estimate dereverberated speech components ˆy(n, ω)
def calculate_dereverberated_speech(x, G):
    y_hat = np.zeros((N,M), dtype=np.complex128)

    for n in range(N):
        x_tilde = np.array([x[max(n - d - 1, 0)].T for d in range(Delta, Delta + Kl)]).flatten().T
        G_H = np.conj(G).T  # Hermitian (conjugate transpose) of G
        y_hat[n] = x[n] - np.dot(G_H, x_tilde)
    return y_hat

# Calculate log-likelihood function L(Θω)
def log_likelihood(y_hat, Lambda):
    Loglik=0
    for i in range(N):
        term1 = -np.log(np.linalg.det(Lambda[i]))  # Log determinant of Lambda[i]
        term2 = -np.dot(y_hat[i].conj().T, np.dot(np.linalg.inv(Lambda[i]), y_hat[i]))  # y_hat Hermitian * Lambda_i_inv * y_hat
    Loglik += (term1 + term2).real
    return Loglik
    
def objective_function_G(G_flattened, *args):
    x, Lambda = args
    G = G_flattened.reshape((Kl*M, M))  # Reshape G back to its original shape
    y_hat = calculate_dereverberated_speech(x, G)
    loglik = 0
    for n in range(N):
        term1 = np.log(np.linalg.det(Lambda[n]))  # Log determinant of Lambda_n
        term2 = np.dot(y_hat[n].conj().T, np.dot(np.linalg.inv(Lambda[n]), y_hat[n]))
        loglik += term1 + term2
    return loglik.real
 # Negative log-likelihood as we minimize in scipy.optimize

def optimize_G(x, G, Lambda):
    G_flattened = G.flatten()  # Flatten G to a 1D array
    args = (x, Lambda)
    result = minimize(objective_function_G, G_flattened, args=args, method='L-BFGS-B', options={'maxiter': 100})
    optimized_G = result.x.reshape(G.shape)  # Reshape the result back to the original shape of G
    return optimized_G

def objective_function_Lambda_i(Lambda_flattened_i, *args):
    x, G,i= args
    Lambda_i = Lambda_flattened_i.reshape((M, M))  # Reshape Lambda back to its original shape
    y_hat = calculate_dereverberated_speech(x, G)
    loglik = 0
    term1 = np.log(np.linalg.det(Lambda_i))  # Log determinant of Lambda_n
    term2 = np.dot(y_hat[i].conj().T, np.dot(np.linalg.inv(Lambda_i), y_hat[i]))
    loglik += term1 + term2
    return loglik.real

def contrainte_det_positif(Lambda_flattened_i):
    Lambda_i = Lambda_flattened_i.reshape((M, M))
    return np.linalg.det(Lambda_i)-0.001

def optimize_Lambda_i(x, G, Lambda, i):
    Lambda_flattened_i = Lambda[i].flatten()
    args = (x, G,i)

    det_positive_constraint = {'type': 'ineq', 'fun': contrainte_det_positif}
    
    result = minimize(objective_function_Lambda_i, Lambda_flattened_i, args=args, method='SLSQP', constraints=[det_positive_constraint], options={'maxiter': 100})
    optimized_Lambda_i = result.x.reshape(Lambda[0].shape)
    return optimized_Lambda_i

def iterative_maximization(x, initial_G, initial_Lambda, iterations):
    loglikes=[]
    G = initial_G
    Lambda = initial_Lambda

    for i in range(iterations):
        print(f"{i}th iteration")
        y_hat = calculate_dereverberated_speech(x, G)
        print(f"Current G : {G}, current Lambda : {Lambda}")
        print( f"current log-likelihood : {log_likelihood(y_hat, Lambda)}")
        
        # Step 1: Maximization with respect to G^H
        G = optimize_G(x, G, Lambda)
        y_hat = calculate_dereverberated_speech(x, G)
        print( f"current log-likelihood_after_G : {log_likelihood(y_hat, Lambda)}")
        loglikes.append(log_likelihood(y_hat, Lambda))
        
         # Step 2: Maximization with respect to Λn
        for i in range(N):
            Lambda[i]= optimize_Lambda_i(x, G, Lambda,i)
            ###print(f" det lambda[{i}] : {np.linalg.det(Lambda[i])}")

    return G, Lambda, loglikes

frequence_cible = 5.0
frequence_echantillonnage = 50000 
largeur_fenetre = int(300*frequence_echantillonnage/1000)

data=np.array(scipy.io.loadmat(r'C:\Users\admin\OneDrive - CentraleSupelec\Documents\Perso\Scolaire\CS 2A\Projet\1_2m_x_0_y_-10_1000hz_15s.mat')['mat'])[:3,:900*int(frequence_echantillonnage/1000)]
print(f"shape data : {data.shape}")

def calculer_stft(matrice_signaux, frequence_cible, frequence_echantillonnage, largeur_fenetre):
    _, n = matrice_signaux.shape

    # Utiliser la largeur de la fenêtre pour définir les paramètres de la STFT
    _, _, Zxx = stft(matrice_signaux, fs=frequence_echantillonnage, nperseg=largeur_fenetre)

    # Extraire la transformée de Fourier pour la fréquence cible
    indice_frequence_cible = int(frequence_cible * largeur_fenetre / frequence_echantillonnage)
    transformees_fourier = Zxx[indice_frequence_cible, :]

    return transformees_fourier
 
# Calcul des transformées de Fourier pour des plages successives de n
transformees_fourier = calculer_stft(data, frequence_cible, frequence_echantillonnage, largeur_fenetre)

print(f"shape transformée de fourier : {transformees_fourier.shape}")


N,M = transformees_fourier.shape
Kl = 3 # Define prediction order
G_initial = define_mclp_filter_matrix(Kl, M)
iterations = 20 
w = 2*np.pi*frequence_cible  
x_test = np.random.random((N,M)) + 1j * np.random.random((N,M))
Lambda_initial = initialize_Lambda()
Delta = 10

G_final, Lambda_final, loglikes = iterative_maximization(transformees_fourier, G_initial, Lambda_initial, iterations)
y_final= calculate_dereverberated_speech(transformees_fourier, G_final)
###print(f"G_final : {G_final}, Lambda final {Lambda_final}, y_final : {y_final}")

X=[j for j in range(20)]
plt.plot(X,loglikes)
plt.show()

