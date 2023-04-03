# Tyler Boudreau
# 04/02/2023
# Assignment 3:
# COT 4500
import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

def Runge_Kutta(f, t, y, h, n):
	for i in range(n):
		func_value = f(t, y) # Sets func_value equal to function
		t += h       
		next_value = float(f(t, y + h*func_value))
		y = float(y + h/2*(func_value + next_value))
	print(f'{y:.5f}'"\n") # Prints result to 5 decimal places

def eulers_estimate(f, t, y, h, n):
	for i in range(n):
		y = y + h*f(t, y)
		t += h
	print(f'{y:.5f}'"\n") # Prints result to 5 decimal places

def gauss_jordan(A, b):
    n = len(b)
    # Combine A and b into augmented matrix
    Ab = np.concatenate((A, b.reshape(n,1)), axis=1)
    # Perform elimination
    for i in range(n):
        # Find pivot row
        max_row = i
        for j in range(i+1, n):
            if abs(Ab[j,i]) > abs(Ab[max_row,i]):
                max_row = j
        # Swap rows to bring pivot element to diagonal
        Ab[[i,max_row], :] = Ab[[max_row,i], :] # operation 1 of row operations
        # Divide pivot row by pivot element
        pivot = Ab[i,i]
        Ab[i,:] = Ab[i,:] / pivot
        # Eliminate entries below pivot
        for j in range(i+1, n):
            factor = Ab[j,i]
            Ab[j,:] -= factor * Ab[i,:] # operation 2 of row operations
    # Perform back-substitution
    for i in range(n-1, -1, -1):
        for j in range(i-1, -1, -1):
            factor = Ab[j,i]
            Ab[j,:] -= factor * Ab[i,:]
    # Extract solution vector x
    x = Ab[:,n]
    return x

def determiniate(A):
    # Computes Determinate of matrix A
    det = np.linalg.det(A)
    return det

def luDecomposition(A, n):
 
    lower = [[0 for x in range(n)]
             for y in range(n)]
    upper = [[0 for x in range(n)]
             for y in range(n)]
 
    # Decomposing matrix into Upper and Lower Matrixes
    for i in range(n):
        # Upper Matrix
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (lower[i][j] * upper[j][k])
            upper[i][k] = A[i][k] - sum

        # Lower Matrix
        for k in range(i, n):
            if (i == k):
                lower[i][i] = 1  # Diagonal as 1
            else:
                sum = 0
                for j in range(i):
                    sum += (lower[k][j] * upper[j][i])
                lower[k][i] = int((A[k][i] - sum) /upper[i][i])
                
    # Displays Lower Matrix
    print("[", end='')
    print (*lower, sep="\n", end="]\n\n")
    # Displays Upper Matrix
    print("[", end='')
    print (*upper, sep="\n", end="]\n\n")

def dd(A):
    D = np.diag(np.abs(A)) # Find diagonal coefficients
    S = np.sum(np.abs(A), axis=1) - D # Find sum of rows without diagonal
    
    # Returns True if (D > S) and matrix is diagonally dominate
    if np.all(D > S):
        return True
    else:
        return False
    # Returns False if matrix is not diagonally dominate

def det_subMatrixes1():
    # First Determinate
    A = np.array([2])
    det1 = 2 # Is Equal to 2 which is positive
    # Second Determinate
    A = np.array([[2,2],[2,3]])
    det2 = determiniate(A)
    # Third Determinate
    A = np.array([[2, 2, 1],[2, 3, 0],[1, 0, 2]])
    det3 = determiniate(A)

    # Determine if each matrix determinate is positive
    # If all determinates are positive matrix is positive definite; return True
    if (det1>0):
        if(det2>0):
            if(det3>0):
                return True
            else:
                return False
        else:
            return False
    else:
        return False
    # Return False if not all matrixes are positive


def main():
    # Problem 1:
	# Assign f to Function
	f = lambda t, y: (t-y**2)
        
    # Initial x point
	t0 = 0
        
    #Inintial y point
	y0 = 1
        
    # Value to estimate
	p = 2
    
    # Number of Iterations
	n = 10
        
    # Value of h
	h = .2    
	eulers_estimate(f, t0, y0, h, n)
	
	# Problem 2:
	Runge_Kutta(f, t0, y0, h, n)
	
    # Problem 3:
	A = np.array([[2,-1,1], [1,3,1], [-1,5,4]], dtype=np.double)
	b = np.array([6,0,-3], dtype=np.double)
	x = gauss_jordan(A, b)
	print(x,"\n")
        
    # Problem 4:
	A = np.array([[1, 1, 0, 3],
       [2, 1, -1, 1],
       [3, -1, -1, 2],
       [-1, 2, 3, -1]], dtype=np.double)
	A = A.astype(int)
	print("%.5f" %round(determiniate(A)),"\n")
        
    # Problems 5 and 6:
	luDecomposition(A, 4)
        
    # Problem 7:
	A=np.array([[ 9, 0,   5, 2, 1],
               [3, 9, 1, 2, 1],
               [0, 1, 7, 2, 3],
               [4, 2, 3, 12, 2],
               [3, 2, 4, 0, 8]])
	print(dd(A),"\n")
    
	# Problem 8:
	print(det_subMatrixes1(),"\n")


if __name__ == "__main__":
    main()

