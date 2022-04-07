import numpy as np

class SIFT:
    def __init__(self, gs = 6, ps = 31, gaussian_thres = 7., \
                 gaussian_sigma = 4, sift_threshold = .3, num_angles = 12, \
                 bins = 5, alpha = 9.0):
        self.num_angles = num_angles
        self.bins = bins
        self.alpha = alpha
        self.angle_list = np.array(range(num_angles))*2.0*np.pi/num_angles
        self.gs = gs
        self.ps = ps
        self.gaussian_thres = gaussian_thres
        self.gaussian_sigma = gaussian_sigma
        self.sift_threshold = sift_threshold
        self.weights = self.weights(bins)
    
    # Private Methods
    
    def gaussian_filter(self, sigma):
        coeff = int(2*np.ceil(sigma))
        gaussian_range = np.array(range(-coeff, coeff+1))**2
        gaussian_range = gaussian_range[:, np.newaxis] + gaussian_range
        value = np.exp(- gaussian_range / (2.0 * sigma**2))
        value /= np.sum(value)
        height, width = np.gradient(gaussian_range)
        height *= 2.0/np.sum(np.abs(height))
        width  *= 2.0/np.sum(np.abs(width))
        return height, width
    
    def convolution_image_gaussian(self, image, gaussian):
        imR, imC = image.shape
        kR, kC = gaussian.shape
        y = np.zeros((imR,imC))
        center_kX, center_kY = kC//2, kR//2
        for i in range(imR):
            for j in range(imC):
                for m in range(kR):
                    m_ = kR - 1 - m
                    for n in range(kC):
                        n_ = kC - 1 - n
                        i_ = i + (m - center_kY)
                        j_ = j + (n - center_kX)
                        if i_ >= 0 and i_ < imR and j_ >= 0 and j_ < imC :
                            y[i][j] += image[i_][j_] * gaussian[m_][n_]
        return y
    
    def normalize_features(self, features):
        features_len = np.sqrt(np.sum(features**2, axis=1))
        hcontrast = (features_len >= self.gaussian_thres)
        features_len[features_len < self.gaussian_thres] = self.gaussian_thres
        features /= features_len.reshape((features_len.size, 1))
        features[features>self.sift_threshold] = self.sift_threshold
        features[hcontrast] /= np.sqrt(np.sum(features[hcontrast]**2, axis=1)).\
                reshape((features[hcontrast].shape[0], 1))
        return features
    
    def grid(self, image, grid_H, grid_W):
        H, W = image.shape
        n_patches = grid_H.size
        features = np.zeros((n_patches, self.bins * self.bins * self.num_angles))
        height, width = self.gaussian_filter(self.gaussian_sigma)
        IH, IW = self.convolution_image_gaussian(image, height), self.convolution_image_gaussian(image, width)
        img = np.sqrt(IH**2 + IW**2)
        theta = np.arctan2(IH,IW)
        orient = np.zeros((self.num_angles, H, W))
        for i in range(self.num_angles):
            orient[i] = img * np.maximum(np.cos(theta - self.angle_list[i])**self.alpha, 0)
        for i in range(n_patches):
            feat = np.zeros((self.num_angles, self.bins**2))
            for j in range(self.num_angles):
                feat[j] = np.dot(self.weights,\
                        orient[j,grid_H[i]:grid_H[i]+self.ps, grid_W[i]:grid_W[i]+self.ps].flatten())
            features[i] = feat.flatten()
        return features

    def weights(self, bins):
        size_unit = np.array(range(self.ps))
        sph, spw = np.meshgrid(size_unit, size_unit)
        sph.resize(sph.size)
        spw.resize(spw.size)
        bincenter = np.array(range(1, bins*2, 2)) / 2.0 / bins * self.ps - 0.5
        bincenter_h, bincenter_w = np.meshgrid(bincenter, bincenter)
        bincenter_h.resize((bincenter_h.size, 1))
        bincenter_w.resize((bincenter_w.size, 1))
        dist_ph = abs(sph - bincenter_h)
        dist_pw = abs(spw - bincenter_w)
        weights_h = dist_ph / (self.ps / np.double(bins))
        weights_w = dist_pw / (self.ps / np.double(bins))
        weights_h = (1-weights_h) * (weights_h <= 1)
        weights_w = (1-weights_w) * (weights_w <= 1)
        return weights_h * weights_w
    
    # Public Methods
    
    def get_features(self, image):
        image = image.astype(np.double)
        if image.ndim == 3:
            image = np.mean(image, axis=2)
        H, W = image.shape
        gS = self.gs
        pS = self.ps
        remH = np.mod(H-pS, gS)
        remW = np.mod(W-pS, gS)
        offsetH = remH//2
        offsetW = remW//2
        gridH, gridW = np.meshgrid(range(offsetH, H-pS+1, gS), range(offsetW, W-pS+1, gS))
        gridH = gridH.flatten()
        gridW = gridW.flatten()
        features = self.grid(image, gridH, gridW)
        features = self.normalize_features(features)
        pos = np.vstack((gridH / np.double(H), gridW / np.double(W)))
        return features, pos
    
    def get_features_from_data(self, data):
        out = []
        for idx, dt in enumerate(data):
            out.append(self.get_features(np.mean(np.double(dt), axis=2))[0][0])
        return np.array(out)