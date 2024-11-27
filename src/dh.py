import sympy as sp

# Define symbolic variables for joint angles (theta) and constants (a, alpha, d)
theta1, theta2, theta3, theta4, theta5 = sp.symbols('theta1 theta2 theta3 theta4 theta5')
a0, a1, a2, a3, a4 = 0, 0, 0, 0, 0  # All link lengths are zero as per the model
d0, d1, d2, d3, d4 = 0.09, 0.045, 0.11, 0.115, 0.12  # Offsets from the XML
alpha0, alpha1, alpha2, alpha3, alpha4 = 0, -sp.pi/2, -sp.pi/2, -sp.pi/2, 0

# Helper function to construct the DH transformation matrix
def dh_matrix(a, alpha, d, theta):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta), sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0, sp.sin(alpha), sp.cos(alpha), d],
        [0, 0, 0, 1]
    ])

# Define each transformation matrix
T0_1 = dh_matrix(a0, alpha0, d0, theta1)
T1_2 = dh_matrix(a1, alpha1, d1, theta2)
T2_3 = dh_matrix(a2, alpha2, d2, theta3)
T3_4 = dh_matrix(a3, alpha3, d3, theta4)
T4_5 = dh_matrix(a4, alpha4, d4, theta5)

# Compute the overall transformation matrix
T0_5 = T0_1 * T1_2 * T2_3 * T3_4 * T4_5

print(T0_5)
