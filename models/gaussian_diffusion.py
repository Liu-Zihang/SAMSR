import enum
import math
from torchvision.utils import save_image
import numpy as np
import torch as th
import torch.nn.functional as F

from .basic_ops import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from script_segment import generate_masks_from_batch,load_sam_model, generate_weighted_noise, count_masks_per_pixel,  merge_all_mask_to_one_RoPE_batch, normalize_noise_to_unit_variance

from ldm.models.autoencoder import AutoencoderKLTorch

def normalize_mask_counts(mask_count: th.Tensor) -> th.Tensor:
    """
    Normalize the mask count for each pixel to the range [0, 1], 
    based on the maximum mask count in each image.

    Args:
        mask_count: A tensor of shape (B, H, W), where each value 
                    represents the number of times that pixel is covered.

    Returns:
        normalized_weights: A tensor of shape (B, H, W) containing 
                            the normalized weights.
    """
    # Compute the maximum mask count for each image in the batch
    max_masks_per_image = (
        mask_count
        .view(mask_count.size(0), -1)
        .max(dim=1)[0]
        .view(-1, 1, 1)  # reshape to (B, 1, 1)
    )

    # Prevent division by zero by ensuring a minimum of 1.0
    max_masks_per_image = th.clamp(max_masks_per_image, min=1.0)

    # Normalize the mask counts by the maximum per image
    normalized_weights = mask_count / max_masks_per_image  # shape: (B, H, W)

    # Clamp the results to ensure they lie within [0, 1]
    normalized_weights = th.clamp(normalized_weights, 0.0, 1.0)

    return normalized_weights


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_start, beta_end):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        return np.linspace(
            beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64
        )**2
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=1.0,
        kwargs=None):
    """
    Get a pre-defined eta schedule for the given name.

    The eta schedule library consists of eta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    """
    if schedule_name == 'exponential':
        # ponential = kwargs.get('ponential', None)
        # start = math.exp(math.log(min_noise_level / kappa) / ponential)
        # end = math.exp(math.log(etas_end) / (2*ponential))
        # xx = np.linspace(start, end, num_diffusion_timesteps, endpoint=True, dtype=np.float64)
        # sqrt_etas = xx**ponential
        power = kwargs.get('power', None)
        etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        increaser = math.exp(1/(num_diffusion_timesteps-1)*math.log(etas_end/etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True)**power
        power_timestep *= (num_diffusion_timesteps-1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
    elif schedule_name == 'ldm':
        import scipy.io as sio
        mat_path = kwargs.get('mat_path', None)
        sqrt_etas = sio.loadmat(mat_path)['sqrt_etas'].reshape(-1)
    else:
        raise ValueError(f"Unknow schedule_name {schedule_name}")

    return sqrt_etas

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon
    PREVIOUS_X = enum.auto()  # the model predicts epsilon
    RESIDUAL = enum.auto()  # the model predicts epsilon
    EPSILON_SCALE = enum.auto()  # the model predicts epsilon

class LossType(enum.Enum):
    MSE = enum.auto()           # simplied MSE
    WEIGHTED_MSE = enum.auto()  # weighted mse derived from KL

class ModelVarTypeDDPM(enum.Enum):
    """
    What is used as the model's output variance.
    """

    LEARNED = enum.auto()
    LEARNED_RANGE = enum.auto()
    FIXED_LARGE = enum.auto()
    FIXED_SMALL = enum.auto()


def _extract_from_tensor(tensor, timesteps, broadcast_shape):
    """

    :param tensor: the input tensor.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = tensor[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def _extract_from_array(arr, timesteps, broadcast_shape):
    """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    :param sqrt_etas: a 1-D numpy array of etas for each diffusion timestep,
                starting at T and going to 1.
    :param kappa: a scaler controling the variance of the diffusion kernel
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param loss_type: a LossType determining the loss function to use.
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    :param scale_factor: a scaler to scale the latent code
    :param sf: super resolution factor
    """

    def __init__(
        self,
        *,
        sqrt_etas,
        kappa,
        model_mean_type,
        loss_type,
        sf=4,
        scale_factor=None,
        normalize_input=True,
        latent_flag=True,
    ):
        self.kappa = kappa
        self.model_mean_type = model_mean_type
        self.loss_type = loss_type
        self.scale_factor = scale_factor
        self.normalize_input = normalize_input
        self.latent_flag = latent_flag
        self.sf = sf

        # Use float64 for accuracy.
        self.sqrt_etas = sqrt_etas
        self.etas = sqrt_etas**2
        assert len(self.etas.shape) == 1, "etas must be 1-D"
        assert (self.etas > 0).all() and (self.etas <= 1).all()

        self.num_timesteps = int(self.etas.shape[0])
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = kappa**2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
                self.posterior_variance[1], self.posterior_variance[1:]
                )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas
        
        # coefficient for DDIM inference
        self.etas_prev_clipped = np.append(
                self.etas_prev[1], self.etas_prev[1:]
                )
        self.ddim_coef1 = self.etas_prev * self.etas
        self.ddim_coef2 = self.etas_prev / self.etas

        # weight for the mse loss
        if model_mean_type in [ModelMeanType.START_X, ModelMeanType.RESIDUAL]:
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (self.alpha / self.etas)**2
        elif model_mean_type in [ModelMeanType.EPSILON, ModelMeanType.EPSILON_SCALE]  :
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (
                    kappa * self.alpha / ((1-self.etas) * self.sqrt_etas)
                    )**2
        else:
            raise NotImplementedError(model_mean_type)

        # self.weight_loss_mse = np.append(weight_loss_mse[1],  weight_loss_mse[1:])
        self.weight_loss_mse = weight_loss_mse

    def reset_weight(self, masks_tensor, t, y, device):
        """
        Dynamically adjust diffusion parameters based on masks, 
        while preserving their 1D structure.
        """
        # Ensure original parameters are tensors (but keep 1D)
        if not isinstance(self.kappa, th.Tensor):
            if isinstance(self.kappa, (float, int)):
                kappa = th.full((self.num_timesteps,), float(self.kappa), device=device)
            elif isinstance(self.kappa, np.ndarray):
                kappa = th.from_numpy(self.kappa).to(device=device).float()
            else:
                raise TypeError(f"self.kappa must be a PyTorch tensor, but got {type(self.kappa)}")
                    
        if not isinstance(self.sqrt_etas, th.Tensor):
            if isinstance(self.sqrt_etas, (float, int)):
                sqrt_etas = th.full((self.num_timesteps,), float(self.sqrt_etas), device=device)
            elif isinstance(self.sqrt_etas, np.ndarray):
                sqrt_etas = th.from_numpy(self.sqrt_etas).to(device=device).float()
            else:
                raise TypeError(f"self.sqrt_etas must be a PyTorch tensor, but got {type(self.sqrt_etas)}")

        # Compute mask counts and normalize to get weights
        mask_count = count_masks_per_pixel(masks_tensor)      # (B, H, W)
        W = normalize_mask_counts(mask_count)                 # (B, H, W)
        W_expanded = W.unsqueeze(1)                           # (B, 1, H, W)
        W_expanded = W_expanded / 5.0

        # Extract parameters at the current timestep and apply spatial weights
        kappa_t = _extract_from_tensor(kappa, t, y.shape)         # (B, C, H, W)
        sqrt_etas_t = _extract_from_tensor(sqrt_etas, t, y.shape) # (B, C, H, W)

        # Apply spatially-varying weights
        self.kappa_modified = (1.0 - W_expanded) * kappa_t        # (B, C, H, W)
        self.sqrt_etas_modified = (1.0 + W_expanded) * sqrt_etas_t# (B, C, H, W)
        
        # Compute modified etas
        self.etas_modified = self.sqrt_etas_modified ** 2

        # Prepare for computing etas_prev_modified, handling boundary cases
        batch_size, channels, height, width = self.etas_modified.shape

        # Extract parameters for the previous timestep (t-1), clamping at 0
        t_prev = th.clamp(t - 1, min=0)
        sqrt_etas_prev_t = _extract_from_tensor(sqrt_etas, t_prev, y.shape)
        sqrt_etas_prev_modified = (1.0 + W_expanded) * sqrt_etas_prev_t
        self.etas_prev_modified = sqrt_etas_prev_modified ** 2

        # For t = 0, ensure etas_prev is zero
        t_zero_mask = (t == 0).float().view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        self.etas_prev_modified = self.etas_prev_modified * (1 - t_zero_mask)

        # Compute other related parameters
        self.alpha_modified = self.etas_modified - self.etas_prev_modified

        # Compute modified posterior variance
        self.posterior_variance_modified = (
            self.kappa_modified ** 2 
            * self.etas_prev_modified 
            / self.etas_modified 
            * self.alpha_modified
        )

        # Clip posterior variance to avoid zeros (for log stability)
        self.posterior_variance_clipped_modified = th.where(
            self.posterior_variance_modified > 0,
            self.posterior_variance_modified,
            th.ones_like(self.posterior_variance_modified) * 1e-8
        )
        self.posterior_log_variance_clipped_modified = th.log(self.posterior_variance_clipped_modified)

        # Compute posterior mean coefficients
        self.posterior_mean_coef1_modified = self.etas_prev_modified / self.etas_modified
        self.posterior_mean_coef2_modified = self.alpha_modified / self.etas_modified

        # Compute DDIM-related coefficients
        self.ddim_coef1_modified = self.etas_prev_modified * self.etas_modified
        self.ddim_coef2_modified = self.etas_prev_modified / self.etas_modified

        # Compute modified MSE weight if using certain model mean types
        if self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.RESIDUAL]:
            self.weight_loss_mse_modified = (
                0.5 
                / self.posterior_variance_clipped_modified 
                * (self.alpha_modified / self.etas_modified) ** 2
            )
        elif self.model_mean_type in [ModelMeanType.EPSILON, ModelMeanType.EPSILON_SCALE]:
            self.weight_loss_mse_modified = (
                0.5 
                / self.posterior_variance_clipped_modified 
                * (
                    self.kappa_modified 
                    * self.alpha_modified 
                    / ((1 - self.etas_modified) * self.sqrt_etas_modified)
                ) ** 2
            )

    def q_mean_variance(self, x_start, y, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
        variance = _extract_into_tensor(self.etas, t, x_start.shape) * self.kappa**2
        log_variance = variance.log()
        return mean, variance, log_variance
    
    def q_mean_variance_modified(self, x_start, y, t):
        """
        Get the distribution q(x_t | x_0).
        """
        if hasattr(self, 'etas_modified') and hasattr(self, 'kappa_modified'):
            mean = self.etas_modified * (y - x_start) + x_start
            variance = self.etas_modified * self.kappa_modified**2
            log_variance = variance.log()
            return mean, variance, log_variance
        else:
            return self.q_mean_variance(x_start, y, t)
        

    def q_sample(self, x_start, y, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
            + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape) * noise
        )

    def q_sample_modified(self, x_start, y, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        if hasattr(self, 'etas_modified') and hasattr(self, 'sqrt_etas_modified') and hasattr(self, 'kappa_modified'):
            return (
                self.etas_modified * (y - x_start) + x_start
                + self.sqrt_etas_modified * self.kappa_modified * noise
            )
        else:
            return self.q_sample(x_start, y, t, noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_posterior_mean_variance_modified(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        
        if (hasattr(self, 'posterior_mean_coef1_modified') and 
            hasattr(self, 'posterior_mean_coef2_modified') and 
            hasattr(self, 'posterior_variance_modified') and
            hasattr(self, 'posterior_log_variance_clipped_modified')):
            
            posterior_mean = (
                self.posterior_mean_coef1_modified * x_t
                + self.posterior_mean_coef2_modified * x_start
            )
            posterior_variance = self.posterior_variance_modified
            posterior_log_variance_clipped = self.posterior_log_variance_clipped_modified
            
            return posterior_mean, posterior_variance, posterior_log_variance_clipped
        else:
            return self.q_posterior_mean_variance(x_start, x_t, t)

    def p_mean_variance(
        self, model, x_t, y, t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x_t: the [N x C x ...] tensor at time t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        model_output = model(self._scale_input(x_t, t), t, **model_kwargs)

        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        # ddim coef
        ddim_coef1 = _extract_into_tensor(self.ddim_coef1, t, x_t.shape) # etas_pre*etas
        ddim_coef2 = _extract_into_tensor(self.ddim_coef2, t, x_t.shape) # etas_pre/etas
        etas = _extract_into_tensor(self.etas, t, x_t.shape)
        etas_prev = _extract_into_tensor(self.etas_prev, t, x_t.shape)
        k = (1-etas_prev+th.sqrt(ddim_coef1)-th.sqrt(ddim_coef2))
        m = th.sqrt(ddim_coef2)
        j = (etas_prev - th.sqrt(ddim_coef1))
        
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
            pred_xstart = process_xstart(model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:      # predict x_0
            pred_xstart = process_xstart(
                self._predict_xstart_from_residual(y=y, residual=model_output)
                )
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x_t, y=y, t=t, eps=model_output)
            )                                                  #  predict \eps
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps_scale(x_t=x_t, y=y, t=t, eps=model_output)
            )                                                  #  predict \eps
        else:
            raise ValueError(f'Unknown Mean type: {self.model_mean_type}')

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            # used for ddim
            "ddim_k": k,
            "ddim_m": m,
            "ddim_j": j,
            # "etas": etas,
            # "etas_prev": etas_prev
        }

    def p_mean_variance_modified(
        self, model, x_t, y, t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x_t: the [N x C x ...] tensor at time t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        model_output = model(self._scale_input(x_t, t), t, **model_kwargs)

        if (hasattr(self, 'posterior_variance_modified') and 
            hasattr(self, 'posterior_log_variance_clipped_modified') and
            hasattr(self, 'ddim_coef1_modified') and
            hasattr(self, 'ddim_coef2_modified') and
            hasattr(self, 'etas_modified') and
            hasattr(self, 'etas_prev_modified')):
            
            model_variance = self.posterior_variance_modified
            model_log_variance = self.posterior_log_variance_clipped_modified

            # ddim coef
            ddim_coef1 = self.ddim_coef1_modified  # etas_pre*etas
            ddim_coef2 = self.ddim_coef2_modified  # etas_pre/etas
            etas = self.etas_modified
            etas_prev = self.etas_prev_modified
            
        else:
            model_variance = _extract_from_tensor(self.posterior_variance, t, x_t.shape)
            model_log_variance = _extract_from_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

            ddim_coef1 = _extract_from_tensor(self.ddim_coef1, t, x_t.shape) 
            ddim_coef2 = _extract_from_tensor(self.ddim_coef2, t, x_t.shape) 
            etas = _extract_from_tensor(self.etas, t, x_t.shape)
            etas_prev = _extract_from_tensor(self.etas_prev, t, x_t.shape)

        k = (1-etas_prev+th.sqrt(ddim_coef1)-th.sqrt(ddim_coef2))
        m = th.sqrt(ddim_coef2)
        j = (etas_prev - th.sqrt(ddim_coef1))
        
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
            pred_xstart = process_xstart(model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:      # predict x_0
            pred_xstart = process_xstart(
                self._predict_xstart_from_residual(y=y, residual=model_output)
                )
        elif self.model_mean_type == ModelMeanType.EPSILON:
            if hasattr(self, 'etas_modified'):
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps_modified(x_t=x_t, y=y, t=t, eps=model_output)
                )
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x_t, y=y, t=t, eps=model_output)
                )
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            if hasattr(self, 'etas_modified'):
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps_scale_modified(x_t=x_t, y=y, t=t, eps=model_output)
                )
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps_scale(x_t=x_t, y=y, t=t, eps=model_output)
                )
        else:
            raise ValueError(f'Unknown Mean type: {self.model_mean_type}')

        # 计算模型均值
        if hasattr(self, 'posterior_mean_coef1_modified'):
            model_mean, _, _ = self.q_posterior_mean_variance_modified(
                x_start=pred_xstart, x_t=x_t, t=t
            )
        else:
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x_t, t=t
            )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            # used for ddim
            "ddim_k": k,
            "ddim_m": m,
            "ddim_j": j,
        }

    def _predict_xstart_from_eps(self, x_t, y, t, eps):
        assert x_t.shape == eps.shape
        return  (
            x_t - _extract_into_tensor(self.sqrt_etas, t, x_t.shape) * self.kappa * eps
                - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_eps_scale(self, x_t, y, t, eps):
        assert x_t.shape == eps.shape
        return  (
            x_t - eps - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_residual(self, y, residual):
        assert y.shape == residual.shape
        return (y - residual)

    def _predict_eps_from_xstart(self, x_t, y, t, pred_xstart):
        return (
            x_t - _extract_into_tensor(1 - self.etas, t, x_t.shape) * pred_xstart
                - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(self.kappa * self.sqrt_etas, t, x_t.shape)
    
    def q_posterior_mean_variance_modified(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        
        if (hasattr(self, 'posterior_mean_coef1_modified') and 
            hasattr(self, 'posterior_mean_coef2_modified') and 
            hasattr(self, 'posterior_variance_modified') and
            hasattr(self, 'posterior_log_variance_clipped_modified')):
            
            posterior_mean = (
                self.posterior_mean_coef1_modified * x_t
                + self.posterior_mean_coef2_modified * x_start
            )
            posterior_variance = self.posterior_variance_modified
            posterior_log_variance_clipped = self.posterior_log_variance_clipped_modified
            
            return posterior_mean, posterior_variance, posterior_log_variance_clipped
        else:
            return self.q_posterior_mean_variance(x_start, x_t, t)

    def _predict_xstart_from_eps_modified(self, x_t, y, t, eps):
        assert x_t.shape == eps.shape
        
        if (hasattr(self, 'sqrt_etas_modified') and 
            hasattr(self, 'kappa_modified') and 
            hasattr(self, 'etas_modified')):
            
            return (
                x_t - self.sqrt_etas_modified * self.kappa_modified * eps
                - self.etas_modified * y
            ) / (1 - self.etas_modified)
        else:
            return self._predict_xstart_from_eps(x_t, y, t, eps)

    def _predict_xstart_from_eps_scale_modified(self, x_t, y, t, eps):
        assert x_t.shape == eps.shape
        
        if hasattr(self, 'etas_modified'):
            return (
                x_t - eps - self.etas_modified * y
            ) / (1 - self.etas_modified)
        else:
            return self._predict_xstart_from_eps_scale(x_t, y, t, eps)

    def _predict_eps_from_xstart_modified(self, x_t, y, t, pred_xstart):
        if (hasattr(self, 'etas_modified') and 
            hasattr(self, 'kappa_modified') and 
            hasattr(self, 'sqrt_etas_modified')):
            
            return (
                x_t - (1 - self.etas_modified) * pred_xstart
                - self.etas_modified * y
            ) / (self.kappa_modified * self.sqrt_etas_modified)
        else:
            return self._predict_eps_from_xstart(x_t, y, t, pred_xstart)
        
    def ddim_inverse(self, model, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, noise_repeat=False):
        """
        Sample x_{t} from the model at the given timestep (x_{t-1}).

        :param model: the model to sample from.
        :param x: the current tensor at x_t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            y,
            t,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=model_kwargs,
        )
        pred_xstart = out["pred_xstart"]
        sample = (x - pred_xstart * out["ddim_k"] - out["ddim_j"] * y) / out["ddim_m"]
        return {"sample": sample, "pred_xstart": pred_xstart}


    def p_sample(self, model, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, noise_repeat=False):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            y,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        if noise_repeat:
            noise = noise[0,].repeat(x.shape[0], 1, 1, 1)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean":out["mean"]}

    def p_sample_modified(self, model, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, noise_repeat=False):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance_modified(
            model,
            x,
            y,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        if noise_repeat:
            noise = noise[0,].repeat(x.shape[0], 1, 1, 1)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean":out["mean"]}

    def ddim_inverse_loop(
        self,
        x,
        y,
        model,
        first_stage_model=None,
        noise=None,
        noise_repeat=False,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Find the latent code zT that generate the high quality image $x$ conditioned on the low quaility one $y$
        
        :param x: the [N x C x ...] tensor of high-quality inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param model: the model module.
        :param first_stage_model: the autoencoder model
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of the the latent features of the input high-quality samples.
        """
        final = None
        for sample in self.ddim_inverse_loop_progressive(
            x,
            y,
            model,
            first_stage_model=first_stage_model,
            noise=noise,
            noise_repeat=noise_repeat,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
        ):
            final = sample["sample"]
        return final
    
    def inverse_reflow(
        self,
        x,
        y,
        model,
        first_stage_model=None,
        noise=None,
        noise_repeat=False,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        hyper=1
    ):
        """
        Find the latent code zT that generate the high quality image $x$ conditioned on the low quaility one $y$
        
        :param x: the [N x C x ...] tensor of high-quality inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param model: the model module.
        :param first_stage_model: the autoencoder model
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of the the latent features of the input high-quality samples.
        """
        if device is None:
            device = next(model.parameters()).device
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True) 
        z_x = self.encode_first_stage(x, first_stage_model, up_sample=False)         
    
        t = th.tensor([0] * y.shape[0], device=device)
        out = self.p_mean_variance(
            model,
            z_x,
            z_y,
            t,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=model_kwargs,
        )
        pred_xstart = out["pred_xstart"]
        
        final = (z_x - pred_xstart) / hyper
        return final


    def p_sample_loop(
        self,
        y,
        model,
        first_stage_model=None,
        noise=None,
        noise_repeat=False,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        one_step=False,
        apply_decoder=True
    ):
        """
        Generate samples from the model.

        :param y: the [N x C x ...] tensor of degraded inputs.
        :param model: the model module.
        :param first_stage_model: the autoencoder model
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            y,
            model,
            first_stage_model=first_stage_model,
            noise=noise,
            noise_repeat=noise_repeat,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            one_step=one_step
        ):
            final = sample
        if apply_decoder:
            return self.decode_first_stage(final["sample"], first_stage_model)
        return final

    def ddim_inverse_loop_progressive(
            self, x, y, model,
            first_stage_model=None,
            noise=None,
            noise_repeat=False,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

            x: the high-quality image

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True) 
        z_x = self.encode_first_stage(x, first_stage_model, up_sample=False) 
    
        indices = list(range(1, self.num_timesteps))
        z_sample = z_x
        
        for i in indices:
            t = th.tensor([i] * y.shape[0], device=device)
            with th.no_grad():
                out = self.ddim_inverse(
                    model,
                    z_sample,
                    z_y,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    noise_repeat=noise_repeat,
                )
                yield out
                z_sample = out["sample"]

    def p_sample_loop_progressive(
            self, y, model,
            first_stage_model=None,
            noise=None,
            noise_repeat=False,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            one_step=False
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)

        if noise is None:
            sam_model = load_sam_model()            
            with th.no_grad():
                masks_tensor = generate_masks_from_batch(
                    sam_model, 
                    z_y,
                    upscale=False,
                    scale_factor=2,
                    interp_mode='bilinear',
                    apply_activation=False,
                    )
                noise_weighted = generate_weighted_noise(masks_tensor)
                noise = normalize_noise_to_unit_variance(noise_weighted)

        if noise_repeat:
            noise = noise[0,].repeat(z_y.shape[0], 1, 1, 1)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * y.shape[0], device=device)

            self.reset_weight(masks_tensor, t, y, device)
            z_sample = self.prior_sample_modified(z_y, masks_tensor, noise)

            with th.no_grad():
                out = self.p_sample_modified(
                    model,
                    z_sample,
                    z_y,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    noise_repeat=noise_repeat,
                )
                if one_step:
                    out["sample"]=out["pred_xstart"]
                    yield out
                    break
                yield out
                z_sample = out["sample"]

    def decode_first_stage(self, z_sample, first_stage_model=None, no_grad=True):
        ori_dtype = z_sample.dtype
        if first_stage_model is None:
            return z_sample
        else:
            if no_grad:
                with th.no_grad():
                    z_sample = 1 / self.scale_factor * z_sample
                    z_sample = z_sample.type(next(first_stage_model.parameters()).dtype)
                    out = first_stage_model.decode(z_sample)
            else:
                z_sample = 1 / self.scale_factor * z_sample
                z_sample = z_sample.type(next(first_stage_model.parameters()).dtype)
                out = first_stage_model.decode(z_sample, grad_forward=True)
            return out.type(ori_dtype)
        
    def encode_first_stage(self, y, first_stage_model, up_sample=False):
        ori_dtype = y.dtype
        if up_sample:
            y = F.interpolate(y, scale_factor=self.sf, mode='bicubic')
        if first_stage_model is None:
            return y
        else:
            with th.no_grad():
                y = y.type(dtype=next(first_stage_model.parameters()).dtype)
                z_y = first_stage_model.encode(y)
                out = z_y * self.scale_factor
                return out.type(ori_dtype)
            

    def prior_sample_train(self, y, masks_tensor=None, noise=None):
        """
        Generate a sample from the prior distribution, i.e., q(x_T | x_0) ~= N(x_T | y, ~)

        Args:
            y: A degraded input tensor of shape [N x C x H x W].
            masks_tensor: A mask tensor of shape [N x M x H x W]. If None, it is generated automatically.
            noise: A noise tensor of shape [N x C x H x W]. If None, it is generated automatically.

        Returns:
            A sampled tensor of shape [N x C x H x W].
        """
        # Ensure kappa is a tensor
        kappa = self.kappa
        if not isinstance(kappa, th.Tensor):
            if isinstance(kappa, (float, int)):
                kappa = th.full((self.num_timesteps,), float(kappa), device=y.device)
            elif isinstance(kappa, np.ndarray):
                kappa = th.from_numpy(kappa).to(device=y.device).float()
            else:
                raise TypeError(f"self.kappa must be a PyTorch tensor, but got {type(kappa)}")
        
        # Ensure sqrt_etas is a tensor
        sqrt_etas = self.sqrt_etas
        if not isinstance(sqrt_etas, th.Tensor):
            if isinstance(sqrt_etas, (float, int)):
                sqrt_etas = th.full((self.num_timesteps,), float(sqrt_etas), device=y.device)
            elif isinstance(sqrt_etas, np.ndarray):
                sqrt_etas = th.from_numpy(sqrt_etas).to(device=y.device).float()
            else:
                raise TypeError(f"self.sqrt_etas must be a PyTorch tensor, but got {type(sqrt_etas)}")

        # Compute the mask count per pixel
        mask_count = count_masks_per_pixel(masks_tensor)  # (B, H, W)

        # Normalize the mask counts
        W = normalize_mask_counts(mask_count)  # (B, H, W)

        # Extract the current timestep
        t = th.full((y.shape[0],), self.num_timesteps - 1, device=y.device, dtype=th.long)  # (B,)

        # Extract kappa and sqrt_etas for the current timestep
        kappa_extracted = _extract_from_tensor(kappa, t, y.shape)  # (B, C, H, W)
        sqrt_etas_extracted = _extract_from_tensor(sqrt_etas, t, y.shape)  # (B, C, H, W)

        # Normalize weights W, expand to shape (B, 1, H, W)
        W_expanded = W.unsqueeze(1)  # (B, 1, H, W)
        W_expanded = W_expanded / 5

        kappa_new = (1.0 - W_expanded) * kappa_extracted  # (B, C, H, W)
        sqrt_etas_new = (1.0 + W_expanded) * sqrt_etas_extracted  # (B, C, H, W)
        sampled = y + (kappa_new * sqrt_etas_new) * noise  # (B, C, H, W)

        return sampled

    
    def prior_sample_modified(self, y, noise=None):
        """
        Generate a prior sample using the modified parameters.
        """
        # If no external noise is provided, sample Gaussian noise
        if noise is None:
            noise = th.randn_like(y)
        
        # Check if modified parameters are available
        if hasattr(self, 'kappa_modified') and hasattr(self, 'sqrt_etas_modified'):
            print("Using modified paramter")
            # Use the modified parameters at the final timestep
            # Build a tensor of the last timestep indices for the batch
            device = y.device
            t_last = th.tensor([self.num_timesteps - 1] * y.shape[0], device=device).long()
            
            # If modified parameters are already 4D, use them directly;
            # otherwise, extract the last-timestep slice from the 1D tensors
            if self.kappa_modified.dim() == 4:
                kappa_last = self.kappa_modified
                sqrt_etas_last = self.sqrt_etas_modified
            else:
                kappa_last = _extract_from_tensor(self.kappa_modified, t_last, y.shape)
                sqrt_etas_last = _extract_from_tensor(self.sqrt_etas_modified, t_last, y.shape)
            
            # Return the prior sample with spatially-varying noise scaling
            return y + kappa_last * sqrt_etas_last * noise
        else:
            # Fallback to the original method if modified parameters are unavailable
            device = y.device
            t = th.tensor([self.num_timesteps - 1] * y.shape[0], device=device).long()
            return y + _extract_from_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    def prior_sample(self, y, noise=None):
        """
        Generate samples from the prior distribution, i.e., q(x_T|x_0) ~= N(x_T|y, ~)

        :param y: the [N x C x ...] tensor of degraded inputs.
        :param noise: the [N x C x ...] tensor of degraded inputs.
        """
        if noise is None:
            noise = th.randn_like(y)

        t = th.tensor([self.num_timesteps-1,] * y.shape[0], device=y.device).long()

        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    def training_losses_distill(
            self, sam_model, model, teacher_model, x_start, y, t,
            first_stage_model=None,
            model_kwargs=None,
            noise=None, distill_ddpm=False, uncertainty_hyper=False, uncertainty_num_aux=2, learn_xT=False, finetune_use_gt=False, xT_cov_loss=False, reformulated_reflow=False, loss_in_image_space=False
            ):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param first_stage_model: autoencoder model
        :param x_start: the [N x C x ...] tensor of inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
                 
        :finetune_use_gt: do not use teacher model, instead only use the groud-truth and its inverse
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True) # TODO can be eliminated to speed up, since z_y is already obtained in self.ddim_sample_loop/p_sample_loop
        if noise is None:
            with th.no_grad():
                masks_tensor = generate_masks_from_batch(
                    sam_model, 
                    z_y,
                    upscale=False,
                    scale_factor=2,
                    interp_mode='bilinear',
                    apply_activation=False,
                    )
                noise_before = generate_weighted_noise(masks_tensor)
                noise = normalize_noise_to_unit_variance(noise_before)
                device = noise.device
                self.reset_weight(masks_tensor, t, z_y, device)
        #masks_tensor = generate_masks_from_batch(sam_model, z_y)
        terms = {}
        loss_type = "mse" # "mse"
        assert loss_type in ["mse", "mae"]
        terms["loss"] = 0
        z_t = self.prior_sample_train(z_y, masks_tensor, noise)
        # if not finetune_use_gt:
        if True:
            # obtain *z_start_teacher*, i.e., x_0 predicted from x_T
            if distill_ddpm:
                z_start_teacher = self.p_sample_loop(y, teacher_model, first_stage_model, noise, clip_denoised=True if first_stage_model is None else False, apply_decoder=False, model_kwargs=model_kwargs)["sample"]
            else:
                z_start_teacher = self.ddim_sample_loop(y, teacher_model, noise, first_stage_model, clip_denoised=True if first_stage_model is None else False, apply_decoder=False, model_kwargs=model_kwargs)["sample"]

            # z_t = self.q_sample(z_start_teacher, z_y, t, noise=noise)
            if self.loss_type == LossType.MSE or self.loss_type == LossType.WEIGHTED_MSE:
                model_output = model(self._scale_input(z_t, t), t, **model_kwargs)
                if uncertainty_hyper:
                    with th.no_grad():
                        # model.eval()
                        # first_stage_model.eval()
                        model_output_aux_list = []
                        for _ in range(uncertainty_num_aux):
                            z_t_aux = self.q_sample(z_start_teacher, z_y, t, noise=noise)
                            model_output_aux_list.append(model(self._scale_input(z_t_aux, t), t, **model_kwargs))
                        model_output_aux = th.stack(model_output_aux_list, dim=0)
                        uncertainty = (model_output_aux.max(dim=0)[0]-model_output_aux.min(dim=0)[0]) # B*C*H*W
                        uncertainty = uncertainty.max(dim=1, keepdim=True)[0]
                        z_start_gt = self.encode_first_stage(x_start, first_stage_model, up_sample=False) 
                        
                        uncertainty = (uncertainty*uncertainty_hyper).clip(0,1)
                        z_start = z_start_teacher * uncertainty + z_start_gt * (1-uncertainty) 
                else:
                    z_start = z_start_teacher
                    
                target = {
                    ModelMeanType.START_X: z_start,
                    ModelMeanType.RESIDUAL: z_y - z_start,
                    ModelMeanType.EPSILON: noise,
                    ModelMeanType.EPSILON_SCALE: noise*self.kappa*_extract_into_tensor(self.sqrt_etas, t, noise.shape),
                }[self.model_mean_type]
                assert model_output.shape == target.shape   

                if loss_in_image_space:
                     assert self.model_mean_type == ModelMeanType.START_X
                     model_output_rgb = self.decode_first_stage(model_output, first_stage_model, no_grad=False)
                     target = self.decode_first_stage(z_start, first_stage_model)
                     terms[loss_type] = mean_flat((target - model_output_rgb) ** 2 if loss_type=="mse" else (target - model_output_rgb).abs()) 
                else:
                    terms[loss_type] = mean_flat((target - model_output) ** 2 if loss_type=="mse" else (target - model_output).abs())            
                if self.model_mean_type == ModelMeanType.EPSILON_SCALE:
                    terms[loss_type] /= (self.kappa**2 * _extract_into_tensor(self.etas, t, t.shape))
                if self.loss_type == LossType.WEIGHTED_MSE:
                    weights = _extract_into_tensor(self.weight_loss_mse, t, t.shape)
                else:
                    weights = 1
                terms["loss"] += terms[loss_type] * weights
                
                if learn_xT:
                    predicted_xT = model(self._scale_input(z_start_teacher, t), t*0, **model_kwargs) # TODO scale_input有必要吗？
                    terms[loss_type+"_xT"] = mean_flat((z_t - predicted_xT) ** 2 if loss_type=="mse" else (z_t - predicted_xT).abs())
                    terms["loss"] += terms[loss_type+"_xT"]   
                     
            else:
                raise NotImplementedError(self.loss_type)
            if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
                pred_zstart = model_output.detach()
            elif self.model_mean_type == ModelMeanType.EPSILON:
                pred_zstart = self._predict_xstart_from_eps(x_t=z_t, y=z_y, t=t, eps=model_output.detach())
            elif self.model_mean_type == ModelMeanType.RESIDUAL:
                pred_zstart = self._predict_xstart_from_residual(y=z_y, residual=model_output.detach())
            elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
                pred_zstart = self._predict_xstart_from_eps_scale(x_t=z_t, y=z_y, t=t, eps=model_output.detach())
            else:
                raise NotImplementedError(self.model_mean_type)     
               
        if finetune_use_gt:
            z_start_gt=self.encode_first_stage(x_start, first_stage_model, up_sample=False)

            if not xT_cov_loss:
                with th.no_grad():
                    predicted_xT_from_gt = model(self._scale_input(z_start_gt, t), t*0, **model_kwargs)
                    
                    xT_std_align = False
                    
                    if xT_std_align:
                        noise_gt_pred = predicted_xT_from_gt-z_y
                        sampled_noise = z_t-z_y
                        noise_gt_new=noise_gt_pred/noise_gt_pred.std()*sampled_noise.std()
                        predicted_xT_from_gt=z_y+noise_gt_new
            else:
                predicted_xT_from_gt = model(self._scale_input(z_start_gt, t), t*0, **model_kwargs)
                noise_gt_pred = predicted_xT_from_gt-z_y.detach()
                terms["mse_xT_cov"] = self.cov_loss(noise_gt_pred)
                terms["loss"] += (terms["mse_xT_cov"]*xT_cov_loss)
            
            model_output_pedict_gt = model(self._scale_input(predicted_xT_from_gt.detach(), t), t, **model_kwargs)

            if not loss_in_image_space: 
                terms[loss_type+"_gt"] = mean_flat((z_start_gt - model_output_pedict_gt) ** 2 if loss_type=="mse" else (z_start_gt - model_output_pedict_gt).abs())
                terms["loss"] += (terms[loss_type+"_gt"]*finetune_use_gt)

            else:
                model_output_pedict_gt_rgb = self.decode_first_stage(model_output_pedict_gt, first_stage_model, no_grad=False) # after decode range from -1 to 1
                x_start_tensor = generate_masks_from_batch(
                    sam_model, 
                    x_start,
                    upscale=False,
                    scale_factor=2,
                    interp_mode='bilinear',
                    apply_activation=False,
                    )
                x_start_once = count_masks_per_pixel(x_start_tensor)
                model_output_tensor = generate_masks_from_batch(
                    sam_model, 
                    model_output_pedict_gt_rgb,
                    upscale=False,
                    scale_factor=2,
                    interp_mode='bilinear',
                    apply_activation=False,
                    )
                model_output_once = count_masks_per_pixel(model_output_tensor)
                
                terms[loss_type+"_gt_rgb"] = mean_flat((x_start - model_output_pedict_gt_rgb) ** 2 if loss_type=="mse" else (x_start - model_output_pedict_gt_rgb).abs())
                terms["loss"] += (terms[loss_type+"_gt_rgb"]*finetune_use_gt)

                terms[loss_type+"sam"] = mean_flat((x_start_once - model_output_once) ** 2 if loss_type=="mse" else (x_start_once - model_output_once).abs())
                terms["loss"] += (terms[loss_type+"sam"]*finetune_use_gt)
            
            if pred_zstart is None: pred_zstart=model_output_pedict_gt
        return terms, z_t, pred_zstart
        
        
    def cov_loss(self, noise):
        feat = noise
        kernel_size=8
        b, c, h, w = feat.shape
        feat = feat.view(b*c, 1, h, w)

        feat_unfold = F.unfold(feat, kernel_size=kernel_size, stride=1)
        
        # n_patch = feat_unfold.shape[-1]
        # ratio = 0.1
        # feat_unfold = feat_unfold[..., th.randperm(n_patch)[...,:int(n_patch*ratio)]]
        
        feat_flatten = feat_unfold.permute(0,2,1).contiguous()
        def batch_cov(points):
            B, N, D = points.size()
            mean = points.mean(dim=1).unsqueeze(1)
            diffs = (points - mean).reshape(B * N, D)
            prods = th.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
            bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
            return bcov  # (B, D, D)
        cov = batch_cov(feat_flatten)
        target_cov =  th.eye(cov.shape[1]).repeat([cov.shape[0],1,1]).to(cov.device) * ((self.kappa * self.sqrt_etas)[-1])**2
        loss_cov = mean_flat((target_cov - cov)**2).view(b, c).sum(dim=1)
        return loss_cov

    def training_losses(
            self, model, x_start, y, t,
            first_stage_model=None,
            model_kwargs=None,
            noise=None,
            ):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param first_stage_model: autoencoder model
        :param x_start: the [N x C x ...] tensor of inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}

        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
        z_start = self.encode_first_stage(x_start, first_stage_model, up_sample=False)

        if noise is None:
            noise = th.randn_like(z_start)

        z_t = self.q_sample(z_start, z_y, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.MSE or self.loss_type == LossType.WEIGHTED_MSE:
            model_output = model(self._scale_input(z_t, t), t, **model_kwargs)
            target = {
                ModelMeanType.START_X: z_start,
                ModelMeanType.RESIDUAL: z_y - z_start,
                ModelMeanType.EPSILON: noise,
                ModelMeanType.EPSILON_SCALE: noise*self.kappa*_extract_into_tensor(self.sqrt_etas, t, noise.shape),
            }[self.model_mean_type]
            assert model_output.shape == target.shape == z_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if self.model_mean_type == ModelMeanType.EPSILON_SCALE:
                terms["mse"] /= (self.kappa**2 * _extract_into_tensor(self.etas, t, t.shape))
            if self.loss_type == LossType.WEIGHTED_MSE:
                weights = _extract_into_tensor(self.weight_loss_mse, t, t.shape)
            else:
                weights = 1
            terms["loss"] = terms["mse"] * weights
        else:
            raise NotImplementedError(self.loss_type)

        if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
            pred_zstart = model_output.detach()
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_zstart = self._predict_xstart_from_eps(x_t=z_t, y=z_y, t=t, eps=model_output.detach())
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_zstart = self._predict_xstart_from_residual(y=z_y, residual=model_output.detach())
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_zstart = self._predict_xstart_from_eps_scale(x_t=z_t, y=z_y, t=t, eps=model_output.detach())
        else:
            raise NotImplementedError(self.model_mean_type)

        return terms, z_t, pred_zstart

    def _scale_input(self, inputs, t):
        if self.normalize_input:
            if self.latent_flag:
                # the variance of latent code is around 1.0
                std = th.sqrt(_extract_into_tensor(self.etas, t, inputs.shape) * self.kappa**2 + 1)
                inputs_norm = inputs / std
            else:
                inputs_max = _extract_into_tensor(self.sqrt_etas, t, inputs.shape) * self.kappa * 3 + 1
                inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs
        return inputs_norm

    def ddim_sample(
        self,
        model,
        x,
        y,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        ddim_eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model=model,
            x_t=x,
            y=y,
            t=t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        pred_xstart = out["pred_xstart"]
        
        # residual = y - pred_xstart
        # eps = self._predict_eps_from_xstart(x, y, t, pred_xstart)
        # etas = _extract_into_tensor(self.etas, t, x.shape)
        # etas_prev = _extract_into_tensor(self.etas_prev, t, x.shape)
        # alpha = _extract_into_tensor(self.alpha, t, x.shape)
        # sigma = ddim_eta * self.kappa * th.sqrt(etas_prev / etas) * th.sqrt(alpha)
        # noise = th.randn_like(x)
        
        
        # mean_pred = (
        #     pred_xstart + etas_prev * residual
        #     + th.sqrt(etas_prev*self.kappa**2 - sigma**2) * eps
        # )
        # nonzero_mask = (
        #     (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        # )  # no noise when t == 0
        # sample = mean_pred + nonzero_mask * sigma * noise

        sample = \
            pred_xstart*out["ddim_k"] \
            + out["ddim_m"] * x \
            + out["ddim_j"] * y 
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        y,
        model,
        noise=None,
        first_stage_model=None,
        start_timesteps=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        ddim_eta=0.0,
        zT=None,
        apply_decoder=True,
        one_step=False
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            y=y,
            model=model,
            noise=noise,
            first_stage_model=first_stage_model,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            ddim_eta=ddim_eta,
            zT=zT,
            one_step=one_step
        ):
            final = sample
        if apply_decoder:
            return self.decode_first_stage(final["sample"], first_stage_model)
        return final

    def ddim_sample_loop_progressive(
        self,
        y,
        model,
        noise=None,
        first_stage_model=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        ddim_eta=0.0,
        zT=None,
        one_step=False
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        
        if device is None:
            device = next(model.parameters()).device
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
    
        if zT is None:
            z_sample = self.prior_sample(z_y, noise)
        else:
            z_sample = zT

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * z_y.shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model=model,
                    x=z_sample,
                    y=z_y,
                    t=t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    ddim_eta=ddim_eta,
                )
                if one_step:
                    out["sample"]=out["pred_xstart"]
                    yield out
                    break
                yield out
                z_sample = out["sample"]

        
            
