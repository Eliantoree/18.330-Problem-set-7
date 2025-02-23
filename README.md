# 18.330-Problem-set-7
18.330 Problem set 7

**Download Link:https://programming.engineering/product/18-330-problem-set-7/**

Description
5/5 – (2 votes)
Exercise 1: LU factorization

Suppos you are given an upper-triangular × matrix U, i.e. all elements below the main diagonal are known to be 0. We wish to solve Ux = b for the vector x.

Find the analytical solution for the components of x in terms of the , and using backsubstitution. (Hint: Where should you start?)

Write a function backsubstitution(U, b) that implements this.

What is the approximate operation count to perform backsubstitution?

Repeat [1]–[3] for forward substitution to solve Lx = b for a lower-triangular matrix L.

Write down what happens at a general step in the Gaussian elimination process for a matrix , in terms of the old coefficients , and the new coefficients ′ , .

Implement Gaussian elimination for an × square matrix to calculate the factorization without pivoting. Your function

LU_factorization should return L and U.

What is the operation count of this algorithm?

Write a function solve_linear_system(A, b) that uses the functions you have written to solve the linear system Ax = b.

Consider the matrix A built by the following Julia code. Please use exactly this code so that the results are consistent:

using Random

rng = MersenneTwister(1234)

A = randn(rng, Float64, (10, 10)) b = randn(rng, Float64, 10)

Solve Ax = b for this A and b.

Print out the full x vector using jl show(stdout, “text/plain”, x)

(This prevents the ellipses, …, that can occur in compressed output.)

Compare this to the result from Julia’s \ (backslash) operator and com-ment.

Find the size of the residual Ax − b using the maximum norm (i.e. find the largest entry of the vector) for the x you found using your solve_linear_system(A, b) function.


Plot the residual as a function of for matrices of sizes between = 10 and = 1000 (or however high your computer can manage) on suitable axes.

How does the residual grow as the size grows?

Note that Gaussian elimination is exact in exact arithmetic, so the residual must be due to round-off error.

Exercise 2: Exploiting structure in matrices 1 : Banded LU

In this exercise we will exploit structure in a matrix, i.e. the location of zero elements. If we know in advance that there are zeros in a certain part of the matrix (“structural zeros”), we can exploit that information to define more effi-cient operations on matrices of that type.

The simplest example is diagonal matrices, where we know that all off-diagonal elements are 0; thus calculations that would use those off-diagonal elements do not need to be carried out. (Note that there may also be zeros on the diagonal, but we don’t know that in advance and can’t use that information.)

We define a matrix to be ( , )-banded if its nonzero elements lie within columns to the left of the main diagonal and columns to the right:

≠0 − ≤≤+

For example, a diagonal matrix is (0,0)-banded (only the main diagonal is present) and a tridiagonal matrix is (1,1)-banded.

As an example, think of a system of particles in a line, each of which interacts only with its nearest neighbour on each side. This could lead to an interaction matrix that is known in advance to be tridiagonal, hence leading to the possibility of more efficiently solving the dynamics. Finite difference matrices are also banded. The matrices without structure that we discussed in the lectures on LU and QR are called dense matrices.

We will define a new type Banded as follows:

struct Banded{T} <: AbstractMatrix{T}

bands::Matrix{T}

p::Int

q::Int

end

Here, bands is a matrix that stores only the bands (diagonals) of the original matrix, numbered (− , −( − 1), … ,0, 1, … ), where 0 corresponds to the main diagonal.


Note that there are more terms in bands than we need to store. We will store the lower bands justified to the bottom and the upper bands justified to the top. For example, consider the general (3, 2) banded matrix, i.e. with 3 diagonals below and 2 diagonals above the main diagonal.

11

12

13

24

21

22

23

35

31

32

33

34

=

53 54 55 56

63 64 65 66414243444546

This would be stored as follows in bands:

41

31

21

11

0

0

52

42

32

22

12

0

63

53

43

33

23

13

=

64

54

44

34

24

0

65

55

45

35

0

0

0

0

0

66

56

46

The columns of the new matrix correspond to the diagonals of the old matrix.

Although this seems not to be more efficient in terms of storage, it will be for a larger matrix.

Note that when stored in this way the ( , )th element in B is in band ( − ) and the element is the th element of that band. This means that B[i, j] == B.bands[j, p + 1 + j – i] or is 0 if the term is not in bands. You can use the code in banded.jl on the course website as the implementation for the Banded type. This should work just like an Array in Julia. You are encouraged to read and understand the code. Similar code would allow you to implement your own matrix types. This may be useful for your final projects!

Write a function mymul(B::Banded, v::Vector) that returns the matrix– vector product B*v. The function should be optimized in the sense that terms we know are 0 are not calculated. Compare the result to the built-in dense matrix multiply using the code Array(B) * v. (Array(B) produces a dense matrix with the same bands as B).

Consider the LU factorization. Show, using the formula for matrix multipli-cation, that if is ( , 0)-banded and is (0, )-banded then A = LU is ( , )-banded.

Now show, by thinking about how the elimination algorithm works, that if A is ( , )-banded then the factor L is ( , 0)-banded and U is (0, )-banded.

We have seen that the structure of bandedness is maintained during an LU factorization. We saw in Exercise 1 that the operation count for a


dense LU factorization is ( 3 ). What happens to the operation count when we exploit the banded structure of a ( , )– banded matrix of size × ? Find the approximate operation count in terms of , and .

Implement LU factorization for a banded matrix, LU_factorization(B::Banded), making sure to operate only on non-zero entries.

Given the new operation count, is the naive back/forward substitution or the LU factorization the bottleneck when solving a banded system? If LU is no longer the bottleneck, add methods for your substitution functions so that they are specialized for banded matrices.

(Hint for parts 5 & 6: if you fully understand what is happening here, mod-ifying your LU / substitution codes from Exercise 1 should be simple for banded matrices, changing at most 1 or two lines of code.)

As an example of how necessary it is to exploit structure for efficiency, create a random tridiagonal matrix A of size × and a random vector b of length , taking = 10, 000 (or whatever your computer can safely handle. How many times faster does your structured LU code run than if you were to use dense (standard) linear algebra (after you convert your banded matrix to a dense matrix)? (Remember to run your code once before timing it.)

Check numerically the dependence on that you found in question [4] by solving random tridiagonal systems of sizes × for between 10 and 10,000 and plotting the time taken as a function of . (If the times are too fast, use larger values of !)

[You can use @elapsed (or @belapsed from BenchmarkTools.jl) to cap-ture the time taken by a Julia operation, so that you can automate this in a loop.]

Exercise 3: Exploiting structure in matrices 2 : Tridiagonal QR

In this exercise we will see how to exploit structure (zeros again) to make a more efficient QR algorithm. (However, we do not ask you to code the algorithm this time.)

Consider a general tridiagonal matrix (i.e. a (1,1)-banded matrix):

1

1

2

2

2

3

3

3

⋱

=

⋱

⋱

−1

−1

−1


Since there is only a single subdiagonal, it should be easier to make the matrix upper-triangular by operating on it.

1. Consider the following rotation matrix:

( ) = [

sin

−

sin

]

cos

cos

How should you choose so that

( ) [ ] = [0]

where 2 = 2 + 2 ?

2. Consider the matrix

0

= [

( )

0

]

0

−2

where is the × identity matrix, and where is chosen as above for the terms = 1 and = 2. What does the resulting matrix 1̃= 0 look like? Show that the first column is now upper-triangular.

To be more concrete consider the matrix

1

4

0

0

2

1

3

0

=

0

5

6

7

.

0

0

1

2

Show the resulting matrix when you multiply by 0 – is it what you ex-pected?

What matrix should you multiply 1̃by to make the second column upper-triangular? (Hint: it will have a similar structure to 0) Check that the first column is still upper triangular. Call this matrix 1. Multiply the result of 0 by 1 and show the result.

Generalize to the rotation matrix that makes the th column of a tridiag-onal matrix upper-triangular. Call this matrix −1. Show that −1 is orthogonal, i.e. ⊤ = .

We can now write a tridiagonal matrix in upper-triangular form by form-ing the product −2 −3 ⋯ 1 0 . From the above we see that the result of this will be an upper-triangular matrix .

Show that the product of orthogonal matrices is orthogonal, and hence that this procedure gives a QR decomposition of .

What is the approximate operation count for a full QR factorization on a dense matrix ?


What is the approximate operation count for this reduced tridiagonal QR?

Extra credit Implement this algorithm in an efficient way.

