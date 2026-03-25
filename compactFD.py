import numpy as np
from math import factorial

class CompactFD:
    
    def __init__(self, x=None, n=None, a=None, b=None, mesh="uniform",u1d=3,du1d=3,u2d=5,du2d=5):
        """
        Initialize Compact Finite Difference class.

        Parameters
        ----------
        x : ndarray, optional
            Grid points (if provided directly)
        n : int
            Number of grid points
        a, b : float
            Domain limits
        mesh : str
            Mesh type: 'uniform', 'cosine', 'tanh', 'exponential'
        """

        if x is not None:
            self.x = np.asarray(x)

        else:
            if n is None or a is None or b is None:
                raise ValueError("Either provide x or (n,a,b)")

            self.x = self.get_discretization(a, b, n, mesh)

        self.n = len(self.x)

        # detect uniform grid
        dx = np.diff(self.x)
        self.uniform = np.allclose(dx, dx[0])

        if self.uniform:
            self.dx_grid = dx[0]
        else:
            self.dx_grid = None

        # matrices (constructed later)

        self.u1d = u1d
        self.u2d = u2d
        self.du1d = du1d
        self.du2d = du2d


        self.Ax = None
        self.Bx = None
        self.Axx = None
        self.Bxx = None

    def build_matrix(self, fMat, nfMat, derivative):

        #print("fMat shape:", fMat.shape)
        #print("nfMat shape:", nfMat.shape)

        AnoCond = np.hstack((nfMat, fMat))
        m, n = AnoCond.shape
        # print("AnoCond shape:", AnoCond.shape)

        if m != n:
            raise ValueError("Resultant matrix ([fMat,nfMat]) is not square.")

        B = np.zeros(m)
        B[derivative] = 1

        A = AnoCond.copy()

        for ii in range(m):
            nz = np.abs(A[ii][A[ii] != 0])
            cond = nz.min()
            A[ii] /= cond
            B[ii] /= cond

        return A, B
    
    def get_discretization(self, a, b, n, method="uniform"):
        """
        Generate grid points in the interval [a,b]. 

        Parameters
        ----------
        a, b : float
            Domain limits
        n : int
            Number of grid points
        method : str
            Type of discretization

            "uniform"      -> equally spaced grid
            "cosine"       -> cosine clustering (Chebyshev-like)
            "tanh"         -> wall clustering using tanh stretching
            "exponential"  -> exponential stretching

        Returns
        -------
        x : ndarray
            Grid points
        """

        if b <= a:
            raise ValueError("b must be greater than a")
        
        if n<=2:
            raise ValueError("n must be greater than 2")        


        if method == "uniform":

            x = np.linspace(a, b, n)

        elif method == "cosine":

            i = np.arange(n)
            x = 0.5*(a+b) + 0.5*(b-a)*np.cos(np.pi*(1 - i/(n-1)))

        elif method == "tanh":

            beta = 2.5
            eta = np.linspace(-1,1,n)
            x = 0.5*(a+b) + 0.5*(b-a)*np.tanh(beta*eta)/np.tanh(beta)

        elif method == "exponential":

            s = np.linspace(0,1,n)
            beta = 3
            x = a + (b-a)*(np.exp(beta*s)-1)/(np.exp(beta)-1)

        else:
            raise ValueError(f"Unknown discretization method: {method}")

        x = np.sort(x)
        return x
    
    def get_stencil(self, n, nder):
        """
        Construct compact finite difference stencils.

        Parameters
        ----------
        n : int
            Number of points in stencil for the function
        nder : int
            Number of points in stencil for the derivative

        Returns
        -------
        fStencil : ndarray
            Stencil offsets for the function
        nfStencil : ndarray
            Stencil offsets for the derivative
        """

        if n % 2 == 0:
            raise ValueError("n must be odd")

        if nder % 2 == 0:
            raise ValueError("nder must be odd")

        # stencil for the function
        center = n // 2
        fStencil = np.arange(n) - center

        # stencil for derivative
        center_der = nder // 2
        nfStencil = np.arange(nder) - center_der

        return fStencil, nfStencil
    

    def locate_stencil(self, node, fStencil, nfStencil):
        """
        Place the stencil on the mesh and remove nodes outside the domain.

        Parameters
        ----------
        node : int
            Node index (Python indexing)
        nx : int
            Number of grid points
        fStencil : ndarray
            Function stencil offsets
        nfStencil : ndarray
            Derivative stencil offsets

        Returns
        -------
        fStencilMesh : ndarray
        nfStencilMesh : ndarray
        """

        nx = self.n

        # shift stencil to node location
        fStencilMesh = fStencil + node
        nfStencilMesh = nfStencil + node

        # keep nodes inside the mesh
        fStencilMesh = fStencilMesh[(fStencilMesh >= 0) & (fStencilMesh < nx)]
        nfStencilMesh = nfStencilMesh[(nfStencilMesh >= 0) & (nfStencilMesh < nx)]

        return fStencilMesh, nfStencilMesh
    
    def matrix_cfd(self, fStencil, nfStencil, derivative, extract_diagonals=False):
        """
        Build compact finite difference matrices A and B.

        Parameters
        ----------
        fStencil : ndarray
            Function stencil
        nfStencil : ndarray
            Derivative stencil
        derivative : int
            Derivative order
        extract_diagonals : bool
            If True, return diagonal representation

        Returns
        -------
        A, B : ndarray
            Compact finite difference matrices

        Ad, Bd : ndarray (optional)
            Diagonal representation
        """

        nx = self.n
        x = self.x

        A = np.eye(nx)
        B = np.zeros((nx, nx))

        nf = len(fStencil)
        nfS = len(nfStencil)

        Ad = np.zeros((nx, nf))
        Bd = np.zeros((nx, nfS))

        for i in range(nx):

            # locate stencil inside mesh
            fStencilMesh, nfStencilMesh = self.locate_stencil(
                i, fStencil, nfStencil
            )

            # nodal distances
            fDistance, nfDistance0, s1, s2 = self.nodal_distance(
                fStencilMesh, nfStencilMesh, i
            )

            # pivot derivative stencil
            nfDistance = self.nf_stencil_pivot(nfDistance0)

            # Taylor expansion
            fMat, nfMat, lf, lnf = self.taylor_expansion(
                fDistance, nfDistance, derivative
            )

            # build local system
            C, d = self.build_matrix(fMat, nfMat, derivative)

            # solve system
            coef = np.linalg.solve(C, d)

            # coefficients for A
            coefA = coef[:lnf]

            for j in range(lnf):
                A[i, int(nfDistance[j, 1])] = coefA[j]

            # coefficients for B
            coefB = coef[lnf:lnf+lf]

            for j in range(lf):
                B[i, int(fDistance[j, 1])] = coefB[j]

        # optionally extract diagonals
        if extract_diagonals:

            for i in range(nf):
                ind = i - (nf // 2)

                if ind < 0:
                    Ad[:, i] = np.concatenate(
                        (np.zeros(abs(ind)), np.diag(A, ind))
                    )
                else:
                    Ad[:, i] = np.concatenate(
                        (np.diag(A, ind), np.zeros(abs(ind)))
                    )

            for i in range(nfS):
                ind = i - (nfS // 2)

                if ind < 0:
                    Bd[:, i] = np.concatenate(
                        (np.zeros(abs(ind)), np.diag(B, ind))
                    )
                else:
                    Bd[:, i] = np.concatenate(
                        (np.diag(B, ind), np.zeros(abs(ind)))
                    )

            return A, B, Ad, Bd

        return A, B
    

    def nf_stencil_pivot(self, nfDistance0):
        """
        Remove the row corresponding to the study node from the derivative stencil.

        Parameters
        ----------
        nfDistance0 : ndarray
            Matrix describing derivative stencil distances
        node : int
            Current node index

        Returns
        -------
        nfDistance : ndarray
            Pivoted stencil matrix
        """

        A = nfDistance0[:, 0] - nfDistance0[:, 1]

        mask = A != 0

        if not np.any(~mask):
            raise ValueError(
                "Study node not found in nfDistance matrix. Check stencil construction."
            )

        nfDistance = nfDistance0[mask]

        return nfDistance
    
    def nodal_distance(self, fStencilMesh, nfStencilMesh, px):

        x = self.x

        s1 = len(fStencilMesh)
        s2 = len(nfStencilMesh)

        f_hx = x[fStencilMesh] - x[px]
        fDistance = np.column_stack((np.full(s1, px), fStencilMesh, f_hx))

        nf_hx = x[nfStencilMesh] - x[px]
        nfDistance = np.column_stack((np.full(s2, px), nfStencilMesh, nf_hx))

        return fDistance, nfDistance, s1, s2
    

    def taylor_expansion(self, fDistance, nfDistance, derivative):
        """
        Build Taylor expansion matrices for compact finite differences.

        Parameters
        ----------
        fDistance : ndarray
            Distances for function stencil
        nfDistance : ndarray
            Distances for derivative stencil
        derivative : int
            Derivative order

        Returns
        -------
        fMat : ndarray
        nfMat : ndarray
        lf : int
        lnf : int
        """

        hX = fDistance[:, 2]
        hXX = nfDistance[:, 2]

        const = derivative + 1

        lf = len(hX)
        lnf = len(hXX)

        # initialize matrices
        fMat = np.zeros((lf + lnf, lf))
        nfMat = np.zeros((lf + lnf, lnf))

        # first row
        fMat[0, :] = 1

        # nfMat structure
        nfMat[derivative, :] = -1

        # Taylor expansion for function
        for k in range(1, lf + lnf):
            fMat[k, :] = hX**k / factorial(k)

        # Taylor expansion for derivative
        for k in range(1, lf + lnf - const + 1):
            nfMat[k + const - 1, :] = -hXX**k / factorial(k)

        return fMat, nfMat, lf, lnf
    
    def build_first_derivative(self):
        """
        Build compact operator for first derivative.
        """

        derivative = 1

        fStencil, nfStencil = self.get_stencil(self.u1d,self.du1d)

        A, B = self.matrix_cfd(
            fStencil,
            nfStencil,
            derivative
        )

        self.Ax = A
        self.Bx = B

    def build_second_derivative(self):
        """
        Build compact operator for second derivative.
        """

        derivative = 2

        fStencil, nfStencil = self.get_stencil(self.u2d,self.du2d)

        A, B = self.matrix_cfd(
            fStencil,
            nfStencil,
            derivative
        )

        self.Axx = A
        self.Bxx = B


    def dx(self, f):
        """
        Compute first derivative.
        """
        return np.linalg.solve(self.Ax, self.Bx @ f)

    def dxx(self, f):
        """
        Compute second derivative.
        """
        return np.linalg.solve(self.Axx, self.Bxx @ f)

    def solve_dirichlet(self, f, CC1, CC2):
        """
        Solve Au'' = Bu with Dirichlet boundary conditions.

        Parameters
        ----------
        f : ndarray
            RHS vector
        CC1 : float
            Boundary condition at x[0]
        CC2 : float
            Boundary condition at x[-1]

        Returns
        -------
        u : ndarray
            Numerical solution
        """

        nx = self.n

        # solution vector with boundary conditions
        u = np.zeros(nx)
        u[0] = CC1
        u[-1] = CC2

        # modified RHS
        g = -(self.Axx @ f + self.Bxx @ u)

        # solve interior system
        u[1:-1] = np.linalg.solve(
            self.Bxx[1:-1, 1:-1],
            g[1:-1]
        )

        return u
