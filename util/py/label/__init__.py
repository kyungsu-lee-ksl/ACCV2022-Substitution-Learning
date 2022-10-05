import numpy as np


def genOneHotLabelfromTDNandRCT(tdn_mask, rct_mask):

    assert tdn_mask.shape == rct_mask.shape

    h, w = tdn_mask.shape[:2]

    oneHot = np.zeros(shape=(h, w, 3), dtype=np.float32)
    oneHot[: ,:, 0] = np.logical_not(np.logical_or(tdn_mask > 0, rct_mask > 0)).astype(np.float32)
    oneHot[:, :, 1] = (tdn_mask > 0).astype(np.float32)
    oneHot[:, :, 2] = (rct_mask > 0).astype(np.float32)

    return oneHot
