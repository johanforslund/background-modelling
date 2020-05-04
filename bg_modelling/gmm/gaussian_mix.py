import numpy as np
import PIL.Image
from matplotlib import pyplot as plt

def gaussian_mix():
    K = 3
    
    my = 0.5 * np.ones((288, 384, K, 3))
    #my = my / np.linalg.norm(my)

    sigma_squared = 0.05 * np.ones((288, 384, K, 3))
    #sigma_squared = sigma_squared / np.linalg.norm(sigma_squared[0, 0, :, :])
    sigma_init_squared = np.array((0.05, 0.05, 0.05))
    
    w = np.ones((288, 384, K))
    w = w / K
    w_init = 1.0/K
    
    lamb = 3.0
    T = 0.8
    alpha = 1.0/600.0

    B_hat = np.zeros((288, 384))
    B = np.zeros((288, 384))
        
    for f in range(20, 50):
        print(f)
        file_name = '../../frames2/Walk1{:03d}.jpg'.format(f)
        frame = np.asarray(PIL.Image.open(file_name))
        frame = frame / 255.0
        
        c = np.zeros(K)
        p = np.zeros((288, 384, K))

        for x in range(frame.shape[0]):
            for y in range(frame.shape[1]):
                match = False
                for k in range(K):
                    dk_squared = np.sum(np.power(frame[x, y, :] - my[x, y, k, :], 2) / sigma_squared[x, y, k, :])

                    if np.sqrt(dk_squared) < lamb:                    
                        if not match:
                            m = k
                        elif w[x, y, k]/np.sqrt(np.linalg.norm(sigma_squared[x, y, k])) > (w[x, y, m]/np.sqrt(np.linalg.norm(sigma_squared[x, y, m]))):
                            m = k
                                            
                        match = True
                
                if not match:
                    m = K - 1
                    w[x, y, m] = w_init
                    my[x, y, m] = frame[x,y]
                    sigma_squared[x, y, m] = sigma_init_squared
                else:
                    w[x, y, m] = (1 - alpha)*w[x, y, m] + alpha
                    p[x, y, m] = alpha / w[x, y, m]
                    my[x, y, m] = (1-p[x, y, m])*my[x, y, m] + p[x, y, m]*frame[x,y]
                    sigma_squared[x, y, m] = (1-p[x, y, m])*sigma_squared[x, y, m] + np.multiply((p[x, y, m]*(frame[x,y]-my[x, y, m])), (frame[x,y] - my[x, y, m]))

                w[x, y, :] = w[x, y, :] / np.sum(w[x, y, :])

                for k in range(K):
                    c[k] = w[x, y, k]/np.sqrt(np.linalg.norm(sigma_squared[x, y, k]))

                if match:
                    #print(c)
                    indices = np.argsort(-c)
                    #print(indices)
                    w[x, y] = w[x, y, indices]
                    my[x, y] = my[x, y, indices]
                    sigma_squared[x, y] = sigma_squared[x, y, indices]
                
                i = 0

                for i in range(K):
                    if np.sum(w[x, y, 0:i]) > T:
                        #print("BREAK")
                        break
                
                B[x, y] = i #???
                #print(i)

        for x in range(frame.shape[0]):
            for y in range(frame.shape[1]):
                B_hat[x, y] = 0
                for k in range(int(B[x, y])):
                    dk_squared = np.sum(np.power(frame[x, y, :] - my[x, y, k, :], 2) / sigma_squared[x, y, k, :])
                    
                    if np.sqrt(dk_squared) < lamb:
                        B_hat[x,y] = 1
        plt.figure(f)
        plt.imshow(B_hat, cmap="gray")

    plt.show()

    return B_hat

gaussian_mix()