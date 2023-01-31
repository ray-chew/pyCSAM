import numpy as np

class f_trans(object):
    def __init__(self, nhar_i, nhar_j):
        self.nhar_i = nhar_i
        self.nhar_j = nhar_j

    def __get_IJ(self, cell):
        lon, lat = cell.lon, cell.lat
        lon_m, lat_m = cell.lon_m, cell.lat_m

        # now define appropriate indices for the points withing the triangle
        # by shifting the origin to the minimum lat and lon
        lat_res = lat[1] - lat[0]
        lon_res = lon[1] - lon[0]
        self.J = np.ceil((lat_m - lat_m.min())/lat_res).astype(int)
        self.I = np.ceil((lon_m - lon_m.min())/lon_res).astype(int)


    def __prepare_terms(self, cell):
        lon_m, lat_m = cell.lon_m, cell.lat_m

        self.Ni, self.Nj = np.unique(lat_m).size, np.unique(lon_m).size

        self.m_i = np.arange(0,self.nhar_i)
        if self.nhar_j == 2:
            self.m_j = np.arange(-self.nhar_j/2+1,self.nhar_j/2+1)
        elif self.nhar_j % 2 == 0:
            self.m_j = np.arange(-self.nhar_j/2+1,self.nhar_j/2+1)
        else:
            self.m_j = np.arange(-(self.nhar_j-1)/2,(self.nhar_j+1)/2)

        self.term1 = self.m_i.reshape(1,-1) * self.I.reshape(-1,1) / self.Ni
        self.term2 = self.m_j.reshape(1,-1) * self.J.reshape(-1,1) / self.Nj

    def full_f_coeffs(self, cell):
        self.typ = 'full'
        self.__get_IJ(cell)
        self.__prepare_terms(cell)
        
        self.term1 = np.expand_dims(self.term1,-1)
        self.term1 = np.repeat(self.term1,self.nhar_j,-1)
        self.term2 = np.expand_dims(self.term2,1)
        self.term2 = np.repeat(self.term2,self.nhar_i,1)

        tt_sum = self.term1 + self.term2
        tt_sum = tt_sum.reshape(tt_sum.shape[0],-1)

        bcos = 2.0 * np.cos(2.0 * np.pi * (tt_sum))
        bsin = 2.0 * np.sin(2.0 * np.pi * (tt_sum))

        if ((self.nhar_i == 2) and (self.nhar_j == 2)):
            Ncos = bcos[:,:]
            Nsin = bsin[:,1:]

        else:
            if (self.nhar_j % 2 == 0):
                Ncos = bcos[:,int(self.nhar_j/2-1):]
                Nsin = bsin[:,int(self.nhar_j/2):]
            else:
                Ncos = bcos[:,int(self.nhar_j/2-1):]
                Nsin = bsin[:,int(self.nhar_j/2):]
        
        self.bf_cos = Ncos
        self.bf_sin = Nsin


    def axial_f_coeffs(self, cell, alpha = 0.0):
        self.typ = 'axial'
        self.__get_IJ(cell)
        self.__prepare_terms(cell)
        
        alpha = alpha / 180.0 * np.pi
        
        ktil = self.m_i * np.cos(alpha)
        ltil = self.m_i * np.sin(alpha)
        
        self.term1 = ktil.reshape(1,-1) * self.I.reshape(-1,1) / self.Ni + ltil.reshape(1,-1) * self.J.reshape(-1,1) / self.Nj
        
        khat = self.m_j * np.cos(alpha + np.pi/2.0)
        lhat = self.m_j * np.sin(alpha + np.pi/2.0)
        
        self.term2 = khat.reshape(1,-1) * self.I.reshape(-1,1) / self.Ni + lhat.reshape(1,-1) * self.J.reshape(-1,1) / self.Nj
        
        bcos = 2.0 * np.cos(2.0 * np.pi * np.hstack([self.term1, self.term2[:,int(self.nhar_j/2):]]))
        bsin = 2.0 * np.sin(2.0 * np.pi * np.hstack([self.term1[:,1:], self.term2[:,int(self.nhar_j/2):]]))
        
        self.bf_cos = bcos
        self.bf_sin = bsin


    def get_freq_grid(self, a_m):
        nhar_i, nhar_j = self.nhar_i, self.nhar_j

        fourier_coeff = np.zeros((nhar_i, nhar_j))
        nc = self.bf_cos.shape[1]

        zrs = np.zeros((int(self.nhar_j/2)-1))

        if self.typ == 'full':
            cos_terms = a_m[:nc]
            sin_terms = a_m[nc:]
            
            if ((nhar_i == 2) and (nhar_j == 2)):
                sin_terms = np.concatenate(([0.0],sin_terms))
            else:
                cos_terms = np.concatenate((zrs,cos_terms))
                sin_terms = np.concatenate((zrs,[0.0],sin_terms))
            
            fourier_coeff = (cos_terms + 1.0j * sin_terms) / 2.0
            fourier_coeff = fourier_coeff.reshape(nhar_i,nhar_j).swapaxes(1,0)

            
        if self.typ == 'axial':
            f00 = a_m[0]
            cos_terms = a_m[:nc]
            sin_terms = a_m[nc:]
            sin_terms = np.concatenate(([0.0], sin_terms))

            if (nhar_j %2 == 0):
                k_terms = (cos_terms[:nhar_i] + 1.0j * sin_terms[:nhar_i]) / 2.0
                l_terms = (cos_terms[nhar_i:] + 1.0j * sin_terms[nhar_i:]) / 2.0

                l_blk = np.zeros(( int(nhar_j/2-1), int(nhar_i) ))
                u_blk = np.zeros(( int(nhar_j/2), int(nhar_i-1) ))
                
                u_blk = np.hstack((l_terms.reshape(-1,1), u_blk))
                
                fourier_coeff = np.vstack((l_blk, k_terms, u_blk))
                
            else:
                y_axs = (cos_terms[:int((nhar_j+1)/2+1)] + 1.0j * sin_terms[:int((nhar_j+1)/2+1)]) / 2.0
                x_axs = (cos_terms[int((nhar_j-1)/2):] + 1.0j * sin_terms[int((nhar_j-1)/2):]) / 2.0
                x_axs = x_axs.reshape(-1,1)
                l_blk = np.zeros(( int(nhar_i-1), int((nhar_j-1)/2-1) ))
                u_blk = np.zeros(( int(nhar_i-1), int((nhar_j-1)/2) ))
            
                r1 = np.hstack(([0]*int(nhar_j/2),[f00],y_axs)).reshape(1,-1)
                r2 = np.hstack((u_blk,x_axs,l_blk))
                fourier_coeff = np.vstack((r1,r2))
                fourier_coeff = fourier_coeff.T