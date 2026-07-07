import argparse
import os
import time
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from networks.hessian_vector_products import *
from tqdm import trange
from utils.data_generators import generate_test_data, generate_train_data
from utils.eval_functions import setup_eval_function
from utils.training_utils import *
from utils.visualizer import show_solution

# CUDA_VISIBLE_DEVICES should be set in the launch command, e.g. CUDA_VISIBLE_DEVICES=0.
# Do not hard-code it here, otherwise multi-GPU runs may be confusing.


# ============================================================
# Pure Frequency-Aware Fourier-CoPINN for Helmholtz3D
# ------------------------------------------------------------
# This file is self-contained at the training-script level:
# you do NOT need to modify networks/physics_informed_neural_networks.py
# or utils/training_utils.py to use Fourier features for Helmholtz3D.
#
# Default Fourier mode is deterministic harmonic encoding:
#   [x, sin(pi*x), cos(pi*x), ..., sin(K*pi*x), cos(K*pi*x)]
# This is especially suitable for Helmholtz3D, whose exact solution is
#   sin(a1*pi*x) sin(a2*pi*y) sin(a3*pi*z)
# with default a1=4, a2=4, a3=3.
# ============================================================


class AxisFourierEmbedding(nn.Module):
    """Per-axis deterministic Fourier embedding.

    For a scalar coordinate X in shape (N, 1), return
        [X, sin(pi*f_1*X), cos(pi*f_1*X), ..., sin(pi*f_K*X), cos(pi*f_K*X)]
    if include_raw=True; otherwise only sin/cos features.

    Deterministic harmonics are used instead of random B so that the default
    frequencies directly cover the Helmholtz target modes a1/a2/a3.
    """
    fourier_order: int = 8
    scale: float = 1.0
    include_raw: bool = True

    @nn.compact
    def __call__(self, X):
        if self.fourier_order <= 0:
            return X
        freqs = jnp.arange(1, self.fourier_order + 1, dtype=X.dtype).reshape(1, -1)
        proj = jnp.pi * self.scale * (X @ freqs)
        emb = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
        if self.include_raw:
            emb = jnp.concatenate([X, emb], axis=-1)
        return emb


class FourierSPINN3d(nn.Module):
    """SPINN/CoPINN-compatible separable 3D network with per-axis Fourier input.

    It keeps the original separable tensor contraction:
        u(x,y,z) = sum_r f_x^r(x) f_y^r(y) f_z^r(z)
    but replaces each scalar axis input with a Fourier embedding.
    """
    features: Sequence[int]
    r: int
    out_dim: int
    mlp: str = 'modified_mlp'
    fourier_order: int = 8
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0
    include_raw: bool = True

    def _axis_forward(self, X, scale, init):
        X = AxisFourierEmbedding(
            fourier_order=self.fourier_order,
            scale=scale,
            include_raw=self.include_raw,
        )(X)

        if self.mlp == 'mlp':
            for fs in self.features[:-1]:
                X = nn.Dense(fs, kernel_init=init)(X)
                X = nn.activation.tanh(X)
            X = nn.Dense(self.r * self.out_dim, kernel_init=init)(X)
            return X

        if self.mlp == 'modified_mlp':
            U = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
            V = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
            H = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
            for fs in self.features[:-1]:
                Z = nn.Dense(fs, kernel_init=init)(H)
                Z = nn.activation.tanh(Z)
                H = (jnp.ones_like(Z) - Z) * U + Z * V
            H = nn.Dense(self.r * self.out_dim, kernel_init=init)(H)
            return H

        raise NotImplementedError(f'Unknown mlp type: {self.mlp}')

    @nn.compact
    def __call__(self, x, y, z):
        init = nn.initializers.glorot_normal()
        inputs = [x, y, z]
        scales = [self.scale_x, self.scale_y, self.scale_z]
        outputs, xy, pred = [], [], []

        for X, scale in zip(inputs, scales):
            H = self._axis_forward(X, scale, init)
            outputs.append(jnp.transpose(H, (1, 0)))

        for i in range(self.out_dim):
            xy_i = jnp.einsum(
                'fx,fy->fxy',
                outputs[0][self.r * i:self.r * (i + 1)],
                outputs[1][self.r * i:self.r * (i + 1)],
            )
            pred_i = jnp.einsum(
                'fxy,fz->xyz',
                xy_i,
                outputs[2][self.r * i:self.r * (i + 1)],
            )
            pred.append(pred_i)

        if len(pred) == 1:
            return pred[0]
        return pred


def setup_helmholtz_network(args, key):
    """Use FourierSPINN3d for Helmholtz when --use_fourier=1; otherwise original setup."""
    if args.model in ['copinn', 'spinn'] and args.equation == 'helmholtz3d' and args.use_fourier:
        feat_sizes = tuple([args.features for _ in range(args.n_layers)])
        model = FourierSPINN3d(
            features=feat_sizes,
            r=args.r,
            out_dim=args.out_dim,
            mlp=args.mlp,
            fourier_order=args.fourier_order,
            scale_x=args.scale_x,
            scale_y=args.scale_y,
            scale_z=args.scale_z,
            include_raw=bool(args.include_raw),
        )
        params = model.init(
            key,
            jnp.ones((args.nc, 1)),
            jnp.ones((args.nc, 1)),
            jnp.ones((args.nc, 1)),
        )
        return jax.jit(model.apply), params

    return setup_networks(args, key)


def get_fourier_jac_norm_1d(K, scale=1.0, include_raw=1, use_pi=True):
    """Compute ||J_Phi||_2 for one deterministic Fourier axis.

    If Phi(x) = [x, sin(w_1 x), cos(w_1 x), ..., sin(w_K x), cos(w_K x)],
    then for each sin/cos pair, the squared derivatives add up to w_k^2.
    This scalar is used as the frequency-sensitivity coefficient in the
    frequency-aware sample difficulty.
    """
    base = 1.0 if include_raw else 0.0
    factor = jnp.pi if use_pi else 1.0
    freq_sq = 0.0
    for kk in range(1, int(K) + 1):
        omega = kk * scale * factor
        freq_sq += omega ** 2
    return jnp.sqrt(base + freq_sq)


def get_fourier_jac_norm_3d(Kx, Ky, Kz, sx=1.0, sy=1.0, sz=1.0, include_raw=1, use_pi=True):
    """Compute the Frobenius norm of the 3D Fourier feature Jacobian.

    This corresponds to ||J_Phi(x,y,z)||_F for the concatenated per-axis
    Fourier mapping used by Helmholtz3D.
    """
    jx = get_fourier_jac_norm_1d(Kx, sx, include_raw, use_pi)
    jy = get_fourier_jac_norm_1d(Ky, sy, include_raw, use_pi)
    jz = get_fourier_jac_norm_1d(Kz, sz, include_raw, use_pi)
    return jnp.sqrt(jx ** 2 + jy ** 2 + jz ** 2)


@partial(jax.jit, static_argnums=(2, 4))
def apply_model_fourier_copinn(epoch, num_epochs, apply_fn, params, stop_spl_grad, freq_jac_norm, *train_data):
    def residual_loss(epoch, num_epochs, params, x, y, z, source_term, lda=1.0):
        def compute_loss(params, x, y, z, source_term, lda=1.0, mean=True):
            u = apply_fn(params, x, y, z)
            v_x = jnp.ones(x.shape)
            v_y = jnp.ones(y.shape)
            v_z = jnp.ones(z.shape)
            uxx = hvp_fwdfwd(lambda x_: apply_fn(params, x_, y, z), (x,), (v_x,))
            uyy = hvp_fwdfwd(lambda y_: apply_fn(params, x, y_, z), (y,), (v_y,))
            uzz = hvp_fwdfwd(lambda z_: apply_fn(params, x, y, z_), (z,), (v_z,))
            loss = ((uxx + uyy + uzz + lda * u) - source_term) ** 2
            if mean:
                return jnp.mean(loss)
            return loss

        loss = compute_loss(params, x, y, z, source_term, lda=1.0, mean=False)
        residual_abs = jnp.sqrt(loss + 1e-12)

        # Pure frequency-aware difficulty no longer uses the original
        # CoPINN residual-gradient difficulty term. This makes the difficulty
        # assessment 100% Fourier-frequency-aware:
        #     D_FA = |R_theta| * ||J_Phi||_F
        V = get_SPL_V(
            residual_abs,
            epoch,
            num_epochs,
            freq_jac_norm,
        )
        if stop_spl_grad:
            V = jax.lax.stop_gradient(V)

        return jnp.mean(loss.flatten() * V)

    def boundary_loss(params, x, y, z):
        loss = 0.0
        for i in range(6):
            loss += jnp.mean(apply_fn(params, x[i], y[i], z[i]) ** 2)
        return loss

    def get_SPL_V(residual_abs, epoch, num_epochs, freq_jac_norm):
        # Pure Frequency-Aware difficulty:
        #   D_FA = |R_theta| * ||J_Phi||_F
        #
        # Compared with the mixed version
        #   D = alpha * ||grad L_pde||_2 + (1-alpha) * |R_theta| * ||J_Phi||_F,
        # this file removes the original residual-gradient term completely.
        # Therefore the Fourier-frequency-aware term accounts for 100% of the
        # difficulty assessment.
        difficulty = residual_abs.flatten() * freq_jac_norm

        d_min = difficulty.min()
        d_max = difficulty.max()
        norm_d = (difficulty - d_min) / (d_max - d_min + 1e-8)

        beta = 0.001
        ve = 1.0 - (epoch - 1.0) / num_epochs
        vh = (epoch - 1.0) / num_epochs
        D_tilde = norm_d
        delta = (ve - vh) * D_tilde
        V = ve - beta * delta
        return V

    xc, yc, zc, uc, xb, yb, zb = train_data
    loss_fn = lambda params: residual_loss(epoch, num_epochs, params, xc, yc, zc, uc) + boundary_loss(params, xb, yb, zb)
    loss, gradient = jax.value_and_grad(loss_fn)(params)
    return loss, gradient


def write_best_csv(result_dir, best_error, best_rmse, best_epoch, last_error, last_rmse, runtime):
    with open(os.path.join(result_dir, 'best_error.csv'), 'w') as f:
        f.write('best_error,best_rmse,best_epoch,last_error,last_rmse,total_runtime_sec\n')
        f.write(f'{best_error},{best_rmse},{best_epoch},{last_error},{last_rmse},{runtime}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pure Frequency-Aware Fourier-CoPINN for Helmholtz3D')

    parser.add_argument('--model', type=str, default='copinn', help='model name')
    parser.add_argument('--equation', type=str, default='helmholtz3d', help='equation to solve')
    parser.add_argument('--nc', type=int, default=64, help='number of training points for each axis')
    parser.add_argument('--nc_test', type=int, default=100, help='number of test points for each axis')

    parser.add_argument('--seed', type=int, default=113, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50000, help='training epochs')

    parser.add_argument('--mlp', type=str, default='modified_mlp', choices=['mlp', 'modified_mlp'])
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--r', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=1)
    parser.add_argument('--pos_enc', type=int, default=0)

    parser.add_argument('--a1', type=int, default=4, help='sin(a1*pi*x)sin(a2*pi*y)sin(a3*pi*z)')
    parser.add_argument('--a2', type=int, default=4, help='sin(a1*pi*x)sin(a2*pi*y)sin(a3*pi*z)')
    parser.add_argument('--a3', type=int, default=3, help='sin(a1*pi*x)sin(a2*pi*y)sin(a3*pi*z)')

    # Fourier controls.
    parser.add_argument('--use_fourier', type=int, default=1, help='1: use FourierSPINN3d, 0: original setup_networks')
    parser.add_argument('--fourier_order', type=int, default=8, help='highest harmonic K in sin/cos(k*pi*x)')
    parser.add_argument('--scale_x', type=float, default=1.0)
    parser.add_argument('--scale_y', type=float, default=1.0)
    parser.add_argument('--scale_z', type=float, default=1.0)
    parser.add_argument('--include_raw', type=int, default=1)
    parser.add_argument('--stop_spl_grad', type=int, default=1, help='stop-gradient on SPL weights for stability')
    parser.add_argument('--alpha_freq', type=float, default=0.0, help='kept for backward compatibility; ignored in this pure 100% frequency-aware version')

    parser.add_argument('--log_iter', type=int, default=100)
    parser.add_argument('--plot_iter', type=int, default=100)
    parser.add_argument('--eval_iter', type=int, default=1)
    parser.add_argument('--resample_iter', type=int, default=100)

    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key, 2)
    apply_fn, params = setup_helmholtz_network(args, subkey)
    args.total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    freq_jac_norm = get_fourier_jac_norm_3d(
        Kx=args.fourier_order,
        Ky=args.fourier_order,
        Kz=args.fourier_order,
        sx=args.scale_x,
        sy=args.scale_y,
        sz=args.scale_z,
        include_raw=args.include_raw,
        use_pi=True,
    )
    args.freq_jac_norm = float(freq_jac_norm)

    name = name_model(args)
    if args.use_fourier:
        name += f'_FourierK{args.fourier_order}_sx{args.scale_x}_sy{args.scale_y}_sz{args.scale_z}_raw{args.include_raw}_sg{args.stop_spl_grad}_PureFA100'
    else:
        name += f'_NoFourier_sg{args.stop_spl_grad}'

    root_dir = os.path.join(os.getcwd(), 'results', args.equation, args.model)
    result_dir = os.path.join(root_dir, name)
    os.makedirs(result_dir, exist_ok=True)
    print('Result dir:', result_dir)

    optim = optax.adam(learning_rate=args.lr)
    state = optim.init(params)

    key, subkey = jax.random.split(key, 2)
    train_data = generate_train_data(args, subkey, result_dir=result_dir)
    test_data = generate_test_data(args, result_dir)
    eval_fn = setup_eval_function(args.model, args.equation)

    save_config(args, result_dir)
    for fn in ['log (loss, error).csv', 'best_error.csv']:
        path = os.path.join(result_dir, fn)
        if os.path.exists(path):
            os.remove(path)

    best_error = 1e9
    best_rmse = 1e9
    best_epoch = 0
    latest_error = 1e9
    latest_rmse = 1e9
    start = None

    for e in trange(1, args.epochs + 1):
        if e == 2:
            start = time.time()

        if e % args.resample_iter == 0:
            key, subkey = jax.random.split(key, 2)
            train_data = generate_train_data(args, subkey)

        epoch = jnp.array(e)
        num_epochs = jnp.array(args.epochs)
        loss, gradient = apply_model_fourier_copinn(
            epoch,
            num_epochs,
            apply_fn,
            params,
            bool(args.stop_spl_grad),
            jnp.array(freq_jac_norm),
            *train_data,
        )
        params, state = update_model(optim, gradient, params, state)

        if e % args.eval_iter == 0 or e == args.epochs:
            latest_error, latest_rmse = eval_fn(apply_fn, params, *test_data)
            if latest_error < best_error:
                best_error = latest_error
                best_rmse = latest_rmse
                best_epoch = e

            runtime_so_far = 0.0 if start is None else time.time() - start
            write_best_csv(result_dir, best_error, best_rmse, best_epoch, latest_error, latest_rmse, runtime_so_far)

        if e % args.log_iter == 0:
            print(f'Epoch: {e}/{args.epochs} --> total loss: {loss:.8f}, error: {latest_error:.8f}, '
                  f'best error {best_error:.8f}, best epoch {best_epoch}, rmse: {latest_rmse:.8f}')
            log_path = os.path.join(result_dir, 'log (loss, error).csv')
            with open(log_path, 'a') as f:
                if e == args.log_iter:
                    f.write('epoch,loss,error,rmse,best_error,best_rmse,best_epoch\n')
                f.write(f'{e},{loss},{latest_error},{latest_rmse},{best_error},{best_rmse},{best_epoch}\n')

        if args.plot_iter > 0 and e % args.plot_iter == 0:
            show_solution(args, apply_fn, params, test_data, result_dir, e, resol=50)

    runtime = 0.0 if start is None else time.time() - start
    print(f'Runtime --> total: {runtime:.2f}sec ({(runtime / max(1, args.epochs - 1) * 1000):.2f}ms/iter.)')
    print(f'Best --> error: {best_error:.8f}, rmse: {best_rmse:.8f}, epoch: {best_epoch}')

    jnp.save(os.path.join(result_dir, 'params.npy'), params)
    np.savetxt(os.path.join(result_dir, 'total runtime (sec).csv'), np.array([runtime]), delimiter=',')
    write_best_csv(result_dir, best_error, best_rmse, best_epoch, latest_error, latest_rmse, runtime)
