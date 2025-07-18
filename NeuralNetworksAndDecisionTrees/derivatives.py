# Informal definition of derivatives
#
# The formal definition of derivatives can be a bit daunting with limits and values 'going to zero'.
# The idea is really much simpler.
#
# The derivative of a function describes how the output of a function changes when there is a small change in an input variable.
#
# Let's use the cost function J(w) as an example.
# The cost J is the output and w is the input variable.
# Let's give a 'small change' a name: epsilon or ε.
# We use these Greek letters because it is traditional in mathematics to use epsilon (ε) or delta (Δ)
# to represent a small value. You can think of it as representing 0.001 or some other small value.
#
# If increasing w by ε causes J(w) to increase by k × ε, then:
#
#      dJ(w)
#      ---- = k
#       dw
#
# This just says if you change the input to the function J(w) by a little bit and the output changes by k times that little bit,
# then the derivative of J(w) is equal to k.
#
# Let's try this out. Let's look at the derivative of the function J(w) = w^2 at the point w = 3 and ε = 0.001

from sympy import symbols, diff

j = (3)**2  # J(w) = w^2 at w = 3
j_epsilon = (3 + 0.001)**2  # J(w + ε) = (w + ε)^2 at w = 3 and ε = 0.001
print(f"J(3) = {j}, J(3 + 0.001) = {j_epsilon}")

# We have increased the input value a little bit (0.001), causing the output to change from 9 to 9.006001,
# an increase of 6 times the input increase.
#
# Referencing (1) above, this says that k = 6, so:
#     dJ(w)
#     ----  ≈ 6
#      dw
#
# If you are familiar with calculus, you know, written symbolically:
#     dJ(w)
#     ----  = 2w
#      dw
# With w = 3, this is 6.
#
# Our calculation above is not exactly 6 because to be exactly correct, epsilon (ε) would need to be infinitesimally small,
# or really, really small. That is why we use the symbols ≈ or ~= rather than =.
# Let's see what happens if we make ε smaller.

j = (3)**2  # J(w) = w^2 at w = 3
j_epsilon = (3 + 0.000001)**2  # J(w + ε) = (w + ε)^2 at w = 3 and ε = 0.000001
print(f"J(3) = {j}, J(3 + 0.000001) = {j_epsilon}")

# The value gets close to exactly 6 as we reduce the size of ε.
# Feel free to try reducing the value further.
#
# Finding symbolic derivatives
#
# In backpropagation, it is useful to know the derivative of simple functions at any input value.
# Put another way, we would like to know the 'symbolic' derivative rather than the 'arithmetic' derivative.
# An example of a symbolic derivative is:
#     dJ(w)
#     ----  = 2w
#      dw
# The derivative of J(w) = w^2 above.
# With the symbolic derivative, you can find the value of the derivative at any input value w.
#
# If you have taken a calculus course, you are familiar with the many differentiation rules that mathematicians have developed
# to solve for a derivative given an expression.
# Well, it turns out this process has been automated with symbolic differentiation programs.
# An example of this in Python is the SymPy library.
# Let's take a look at how to use this.
#
#     J = w^2
#
# Define the Python variables and their symbolic names.

J, w = symbols('J w')
J = w**2    # J(w) = w^2
# Now we can find the symbolic derivative of J with respect to w.
dJ_dw = diff(J, w)  # This is the symbolic derivative of J with respect to w    
print(f"The symbolic derivative of J(w) = w^2 is: dJ/dw = {dJ_dw}")

dJ_dw.subs([(w, 2)])  # Substitute w = 2 into the symbolic derivative
print(f"The value of the symbolic derivative at w = 2 is: {dJ_dw.subs([(w, 2)])}")
print(f"The value of the symbolic derivative at w = 3 is: {dJ_dw.subs([(w, 3)])}")
print(f"The value of the symbolic derivative at w = -3 is: {dJ_dw.subs([(w, -3)])}")

# j = 2w
j, w = symbols('J w')
j = 2 * w  # This is the symbolic derivative of J(w) = w^2
print(f"The symbolic derivative of J(w) = w^2 is: j = {j}")
dJ_dw = diff(j, w)  # Recalculate to ensure consistency
print(f"Recalculated symbolic derivative: dJ/dw = {dJ_dw}")
dJ_dw.subs([(w, -3)])  # Substitute w = -3 into the symbolic derivative
print(f"The value of the symbolic derivative at w = -3 is: {dJ_dw.subs([(w, -3)])}")
j = 2*3
j_epsilon = 2 * (3 + 0.001)  # Calculate the derivative at w = 3 with ε = 0.001
print(f"j(3 + 0.001) = {j_epsilon}, j(3) = {j}")    
k = (j_epsilon - j) / 0.001  # Calculate the change in output
print(f"The value of k (the change in output per change in input) is: {k}")

#j = w**3
j, w = symbols('J w')
j = w**3  # J(w) = w^3
dJ_dw = diff(j, w)  # This is the symbolic derivative of J(w) = w^3
print(f"The symbolic derivative of J(w) = w^3 is: dJ/dw = {dJ_dw}")
dJ_dw.subs([(w, 2)])  # Substitute w = 2 into the symbolic derivative
print(f"The value of the symbolic derivative at w = 2 is: {dJ_dw.subs([(w, 2)])}")
print(f"The value of the symbolic derivative at w = 3 is: {dJ_dw.subs([(w, 3)])}")
print(f"The value of the symbolic derivative at w = -3 is: {dJ_dw.subs([(w, -3)])}")    

J = (2)**3
J_epsilon = (2+0.001)**3
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")