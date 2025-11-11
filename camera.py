"""
Shutter Class

"""

import torch
import torch.nn.functional as F

try:
    from . import global_vars as gv
except:
    import global_vars as gv


class Shutter:
    """
    Simulates a rolling shutter exposure pattern over time for a video sensor.

    Conventions follow: https://arxiv.org/pdf/1905.13221.pdf

    Attributes:
        T_l (float): Time between the start of exposure of successive rows.
        T_e (float): Exposure duration for each row.
        delta (float): Pixel size (not actively used in shutter computation).
        w (int): Image width.
        h (int): Image height (also determines the number of time steps).
        sf_num (int): Number of shutter frames (used for downsampling).
        mode (str): Shutter mode: "dual" or "single".
        sf (torch.Tensor): Binary spatiotemporal shutter mask of shape (T, H, W).
        act_l (float): Line activation ratio (T_l / T_e).
    """

    def __init__(
        self,
        T_l: float,
        T_e: float,
        delta: float,
        w: int,
        h: int,
        sf_num: int = 1,
        mode: str = "single",
    ):
        """
        Initialize the Shutter instance.

        Args:
            T_l (float): Time delay between row exposures.
            T_e (float): Exposure duration for each row.
            delta (float): Pixel size.
            w (int): Width of the frame in pixels.
            h (int): Height of the frame in pixels.
            sf_num (int, optional): Number of shutter frames for debugging or downsampling. Default is 1.
            mode (str, optional): Shutter type, "dual" or "single". Default is "dual".
        """
        self.T_l = T_l
        self.T_e = T_e
        self.delta = delta
        self.w = w
        self.h = h
        self.sf_num = sf_num
        self.mode = mode
        self.sf = self.shutter_frames()
        self.act_l = self.T_l / self.T_e  # number of rows active at a time

    def shutter_function(self, t: float, row: int) -> int:
        """
        Legacy function to check if the shutter is open for a given row at time t.

        This method is deprecated and not used in the current pipeline.

        Args:
            t (float): Time instant to query.
            row (int): Pixel row index.

        Returns:
            int: 1 if the row is exposed at time t, else 0.
        """
        offset = abs(t - self.T_e / 2.0 - row * self.T_l)
        return int(offset <= self.T_e / 2.0)

    def shutter_frames(self):
        """
        Computes and returns K binary frames representing shutter behavior.
        Now returns: torch.Tensor of shape (T, H, W)
        """

        sf = None
        act_l = round(self.T_e / self.T_l)  # Number of active lines
        H = self.h
        W = self.w

        if self.mode == "dual":
            print("Dual shutter")
            T = H // 2 + act_l - 1
            sf = torch.zeros((T, H, W), dtype=torch.float32)

            for t in range(1, H // 2 + act_l):
                rstart = max(0, t - act_l)
                rend = min(t, H // 2)

                rstart_b = max(min(H, H - t), H // 2)
                rend_b = min(H - t + act_l, H)

                sf[t - 1, rstart:rend, :] = 1
                sf[t - 1, rstart_b:rend_b, :] = 1

        elif self.mode == "single":
            print("Single shutter")
            # T = H + act_l - 1  # number of frames
            T = self.sf_num
            print("t, h, w = ", T, H, W)

            # # DEBUGGING
            # sf = torch.ones((T, H, W, 3), dtype=torch.float32)

            if gv.rgb:
                sf = torch.zeros((T, H, W, 3), dtype=torch.float32)
            else:
                sf = torch.zeros((T, H, W), dtype=torch.float32)

            act_l_ds = int(act_l / gv.downsampling_factor)
            for t in range(1, T+1):
                rstart = max(0, t - act_l_ds)
                rend = min(t, H)
                sf[t - 1, rstart:rend, :] = 1
            
            # for t in range(1, H + act_l):
            #     rstart = max(0, t - act_l)
            #     rend = min(t, H)
            #     sf[t - 1, rstart:rend, :] = 1
            

        """
        You can either downsample 
        1. the inputs to the shutter object or,
        2. send the full size to the object and downsample here

        It turns out that in Option 2, the above line
        >> sf = torch.zeros((T, H, W), dtype=torch.float32)
        runs out of memory. 
        Hence we're using Option 1. 
        """
        # # Save pre-downsampling version for debugging
        # from scipy.io import savemat

        # savemat("before_ds_sf.mat", {"data": sf.cpu().numpy()})

        # # Downsample in time (temporal downsampling)
        # ds_factor = max(1, gv.downsampling_factor)

        # # Add dummy batch/channel dim to interpolate
        # sf = sf.unsqueeze(0).unsqueeze(0)  # shape (1, 1, T, H, W)
        # target_size = (
        #     int(sf.shape[2] / ds_factor),
        #     int(gv.space_downsampling_multiplier * H / ds_factor),
        # )

        # sf_ds = F.interpolate(
        #     sf.squeeze(1),  # shape (1, T, H, W)
        #     size=(target_size[0], H),  # NOTE: only temporal downsampling here
        #     mode="bilinear",
        #     align_corners=True,
        #     antialias=True,
        # )

        # print("sfds shape: ", sf_ds.shape)
        # sf_ds = sf_ds.squeeze(0)  # shape (T_ds, H)
        # sf_ds = sf_ds.unsqueeze(-1).repeat(1, 1, W)  # expand width

        # self.sf_num = sf_ds.shape[0]
        # return sf_ds  # (T_ds, H, W)

        return sf

    def shutter_output(self, v_tld, partition):
        """
        Returns output of sensor after shutter behavior.

        :param v_tld: time varying optical intensity on sensor
        :type v_tld: 4D tensor (T, H, W, C)
        :param partition: subset of T in which we perform shutter functions
        :type: list of times [t1, t2, ..., tn]

        :return: Exposure on each point of the sensor.
        :rtype: 3D tensor (H, W, C)
        """
        T, H, W, C = v_tld.shape
        L = torch.zeros(H, W, C, dtype=torch.float, device=v_tld.device)

        # Get appropriate subset of shutter frames
        if partition is None:
            stf = self.sf  # shape: (T_sf, H, W, C)
        else:
            time_start = partition[0]
            time_end = partition[-1]
            stf = self.sf[time_start : time_end + 1, :, :]  # (T_sub, H, W)

        # EDIT: stf already accounts for colour channels
        # # Repeat the shutter mask across color channels
        # stf_expanded = stf.unsqueeze(-1).repeat(1, 1, 1, C)  # (T_sub, H, W, C)

        # Apply shutter effect
        L = (stf * v_tld[: stf.shape[0]]).sum(dim=0)

        return L

    def shutter_duplicate_frames(self, frame):
        """
        Duplicates a single frame across time for use in adjoint ops.

        :param frame: Tensor of shape (H, W, C)
        :return: Tensor of shape (T, H, W, C)
        """
        assert frame.ndim == 3, f"Expected (H, W, C), got {frame.shape}"
        T = self.sf_num
        return frame.unsqueeze(0).repeat(T, 1, 1, 1)

    def random_shutter(self):
        """
        Overwrites the shutter function with a randomly generated binary shutter mask.
        Result shape: (T, H, W)
        """
        T = self.sf_num
        H = self.h
        W = self.w

        # Randomly choose 0 or 1 with equal probability for each pixel at each time
        shutter_tensor = torch.bernoulli(
            0.5 * torch.ones((T, H, W), dtype=torch.float32)
        )

        self.sf = shutter_tensor.to(gv.device)
