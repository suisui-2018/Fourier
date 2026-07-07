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
# PureFA100 Fourier-CoPINN for (2+1)-d Klein-Gordon
# ------------------------------------------------------------
# This file is self-contained at the training-script level:
# you do NOT need to modify networks/physics_informed_neural_networks.py
# or utils/training_utils.py to use Fourier features for KG3D.
#
# The exact solution used by the original dataset has temporal oscillation
#     u(t,x,y) = (x+y) cos(k t) + (xy) sin(k t)
# so Fourier features are most useful along the t-axis.  By default this file
# applies deterministic sin/cos harmonics on t and keeps x,y mostly raw.
# You can enable spatial Fourier with --fourier_order_x/--fourier_order_y.
# ============================================================


class AxisFourierEmbedding(nn.Module):
    """Per-axis deterministic Fourier embedding.

    For a scalar coordinate X in shape (N, 1), return
        [X, sin(scale * 1 * X), cos(scale * 1 * X), ...,
         sin(scale * K * X), cos(scale * K * X)]
    if include_raw=True; otherwise only sin/cos features.

    For Klein-Gordon3D, the temporal target contains sin(k t), cos(k t),
    so the angular-frequency form scale * m * t is preferable to pi*m*t.
    """
    fourier_order: int = 4
    scale: float = 1.0
    include_raw: bool = True

    @nn.compact
    def __call__(self, X):
        if self.fourier_order <= 0:
            return X
        freqs = jnp.arange(1, self.fourier_order + 1, dtype=X.dtype).reshape(1, -1)
        proj = self.scale * (X @ freqs)
        emb = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
        if self.include_raw:
            emb = jnp.concatenate([X, emb], axis=-1)
        return emb


class FourierSPINN3d(nn.Module):
    """SPINN/CoPINN-compatible separable 3D network with Fourier input.

    It keeps the original separable tensor contraction:
        u(t,x,y) = sum_r f_t^r(t) f_x^r(x) f_y^r(y)
    but optionally replaces each scalar axis input with Fourier features.
    """
    features: Sequence[int]
    r: int
    out_dim: int
    mlp: str = 'modified_mlp'
    fourier_order_t: int = 4
    fourier_order_x: int = 0
    fourier_order_y: int = 0
    scale_t: float = 1.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    include_raw: bool = True

    def _axis_forward(self, X, order, scale, init):
        X = AxisFourierEmbedding(
            fourier_order=order,
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
    def __call__(self, t, x, y):
        init = nn.initializers.glorot_normal()
        inputs = [t, x, y]
        orders = [self.fourier_order_t, self.fourier_order_x, self.fourier_order_y]
        scales = [self.scale_t, self.scale_x, self.scale_y]
        outputs, tx, pred = [], [], []

        for X, order, scale in zip(inputs, orders, scales):
            H = self._axis_forward(X, order, scale, init)
            outputs.append(jnp.transpose(H, (1, 0)))

        for i in range(self.out_dim):
            tx_i = jnp.einsum(
                'ft,fx->ftx',
                outputs[0][self.r * i:self.r * (i + 1)],
                outputs[1][self.r * i:self.r * (i + 1)],
            )
            pred_i = jnp.einsum(
                'ftx,fy->txy',
                tx_i,
                outputs[2][self.r * i:self.r * (i + 1)],
            )
            pred.append(pred_i)

        if len(pred) == 1:
            return pred[0]
        return pred


def setup_kg_network(args, key):
    """Use FourierSPINN3d for KG3D when --use_fourier=1; otherwise original setup."""
    if args.model in ['copinn', 'spinn'] and args.equation == 'klein_gordon3d' and args.use_fourier:
        feat_sizes = tuple([args.features for _ in range(args.n_layers)])
        model = FourierSPINN3d(
            features=feat_sizes,
            r=args.r,
            out_dim=args.out_dim,
            mlp=args.mlp,
            fourier_order_t=args.fourier_order_t,
            fourier_order_x=args.fourier_order_x,
            fourier_order_y=args.fourier_order_y,
            scale_t=args.scale_t,
            scale_x=args.scale_x,
            scale_y=args.scale_y,
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



def get_fourier_jac_norm_1d(K, scale=1.0, include_raw=1, use_pi=False):
    """Compute the feature-space Jacobian norm for one Fourier axis.

    For KG3D, temporal Fourier features use angular frequencies
        sin(scale * m * t), cos(scale * m * t)
    rather than pi*m*t, so use_pi=False by default.

    If K <= 0, this axis has no Fourier features and contributes 0 to the
    frequency-aware term.  This keeps KG3D focused on the temporal axis when
    fourier_order_x = fourier_order_y = 0.
    """
    if int(K) <= 0:
        return jnp.array(0.0)

    base = 1.0 if include_raw else 0.0
    factor = jnp.pi if use_pi else 1.0
    freq_sq = 0.0
    for kk in range(1, int(K) + 1):
        omega = kk * scale * factor
        freq_sq += omega ** 2
    return jnp.sqrt(base + freq_sq)


def get_fourier_jac_norm_3d(Kt, Kx, Ky, st=1.0, sx=1.0, sy=1.0, include_raw=1, use_pi=False):
    """Compute ||J_Phi||_F for the KG3D Fourier feature mapping.

    By default KG3D uses Fourier features only on t, so Kx=Ky=0 and only the
    temporal Fourier sensitivity contributes to the frequency-aware difficulty.
    """
    jt = get_fourier_jac_norm_1d(Kt, st, include_raw, use_pi)
    jx = get_fourier_jac_norm_1d(Kx, sx, include_raw, use_pi)
    jy = get_fourier_jac_norm_1d(Ky, sy, include_raw, use_pi)
    return jnp.sqrt(jt ** 2 + jx ** 2 + jy ** 2)


@partial(jax.jit, static_argnums=(2, 4))
def apply_model_fourier_copinn(epoch, num_epochs, apply_fn, params, stop_spl_grad, beta_spl, freq_jac_norm, *train_data):
    def residual_loss(epoch, num_epochs, params, t, x, y, source_term):
        def compute_loss(params, t, x, y, source_term, mean=True):
            u = apply_fn(params, t, x, y)
            v_t = jnp.ones(t.shape)
            v_x = jnp.ones(x.shape)
            v_y = jnp.ones(y.shape)
            utt = hvp_fwdfwd(lambda t_: apply_fn(params, t_, x, y), (t,), (v_t,))
            uxx = hvp_fwdfwd(lambda x_: apply_fn(params, t, x_, y), (x,), (v_x,))
            uyy = hvp_fwdfwd(lambda y_: apply_fn(params, t, x, y_), (y,), (v_y,))
            loss = (utt - uxx - uyy + u ** 2 - source_term) ** 2
            if mean:
                return jnp.mean(loss)
            return loss

        loss = compute_loss(params, t, x, y, source_term, mean=False)
        residual_abs = jnp.sqrt(loss + 1e-12)

        # Pure frequency-aware SPL weight:
        # only use |R_theta| * ||J_Phi||_F as the sample difficulty.
        V = get_SPL_V(residual_abs, epoch, num_epochs, beta_spl, freq_jac_norm)
        if stop_spl_grad:
            V = jax.lax.stop_gradient(V)

        return jnp.mean(loss.flatten() * V)

    def get_SPL_V(residual_abs, epoch, num_epochs, beta, freq_jac_norm):
        # 100% Fourier frequency-aware difficulty:
        #   D_FA = |R_theta| * ||J_Phi||_F
        #
        # This removes the original residual-gradient difficulty term and uses
        # the Fourier feature sensitivity term with full weight.
        difficulty = residual_abs.flatten() * freq_jac_norm

        d_min = difficulty.min()
        d_max = difficulty.max()
        norm_d = (difficulty - d_min) / (d_max - d_min + 1e-8)

        ve = 1.0 - (epoch - 1.0) / num_epochs
        vh = (epoch - 1.0) / num_epochs
        D_tilde = norm_d
        delta = (ve - vh) * D_tilde
        V = ve - beta * delta
        return V

    def initial_loss(params, t, x, y, u):
        return jnp.mean((apply_fn(params, t, x, y) - u) ** 2)

    def boundary_loss(params, t, x, y, u):
        loss = 0.0
        for i in range(4):
            loss += jnp.mean((apply_fn(params, t[i], x[i], y[i]) - u[i]) ** 2)
        return loss

    tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub = train_data
    loss_fn = lambda params: (
        residual_loss(epoch, num_epochs, params, tc, xc, yc, uc)
        + initial_loss(params, ti, xi, yi, ui)
        + boundary_loss(params, tb, xb, yb, ub)
    )
    loss, gradient = jax.value_and_grad(loss_fn)(params)
    return loss, gradient


def write_best_csv(result_dir, best_error, best_rmse, best_epoch, last_error, last_rmse, runtime):
    with open(os.path.join(result_dir, 'best_error.csv'), 'w') as f:
        f.write('best_error,best_rmse,best_epoch,last_error,last_rmse,total_runtime_sec\n')
        f.write(f'{best_error},{best_rmse},{best_epoch},{last_error},{last_rmse},{runtime}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fourier-CoPINN for (2+1)-d Klein-Gordon')

    parser.add_argument('--model', type=str, default='copinn', help='model name')
    parser.add_argument('--equation', type=str, default='klein_gordon3d', help='equation to solve')
    parser.add_argument('--nc', type=int, default=32, help='number of training points for each axis')
    parser.add_argument('--nc_test', type=int, default=100, help='number of test points for each axis')

    parser.add_argument('--seed', type=int, default=113, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50000, help='training epochs')

    parser.add_argument('--mlp', type=str, default='modified_mlp', choices=['mlp', 'modified_mlp'])
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--features', type=int, default=64)
    parser.add_argument('--r', type=int, default=32)
    parser.add_argument('--out_dim', type=int, default=1)
    parser.add_argument('--pos_enc', type=int, default=0)

    parser.add_argument('--k', type=int, default=2, help='temporal frequency of the solution')

    # Fourier controls.  By default only t receives Fourier features because
    # KG3D's target oscillates in time and is polynomial in x,y.
    parser.add_argument('--use_fourier', type=int, default=1, help='1: use FourierSPINN3d, 0: original setup_networks')
    parser.add_argument('--fourier_order', type=int, default=4,
                        help='shortcut for temporal Fourier order if --fourier_order_t is not set')
    parser.add_argument('--fourier_order_t', type=int, default=-1)
    parser.add_argument('--fourier_order_x', type=int, default=0)
    parser.add_argument('--fourier_order_y', type=int, default=0)
    parser.add_argument('--scale_t', type=float, default=1.0,
                        help='temporal angular-frequency scale; scale_t=1 covers sin(m*t)')
    parser.add_argument('--scale_x', type=float, default=1.0)
    parser.add_argument('--scale_y', type=float, default=1.0)
    parser.add_argument('--include_raw', type=int, default=1)
    parser.add_argument('--stop_spl_grad', type=int, default=1, help='stop-gradient on SPL weights for stability')
    parser.add_argument('--beta_spl', type=float, default=0.01,
                        help='same default beta as the original KG3D CoPINN code')
    parser.add_argument('--alpha_freq', type=float, default=0.0,
                        help='kept for compatibility; this PureFA100 version always uses 100% Fourier frequency-aware difficulty')

    parser.add_argument('--log_iter', type=int, default=100)
    parser.add_argument('--plot_iter', type=int, default=5000)
    parser.add_argument('--eval_iter', type=int, default=1)
    parser.add_argument('--resample_iter', type=int, default=100)

    args = parser.parse_args()
    if args.fourier_order_t < 0:
        args.fourier_order_t = args.fourier_order

    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key, 2)
    apply_fn, params = setup_kg_network(args, subkey)
    args.total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    freq_jac_norm = get_fourier_jac_norm_3d(
        Kt=args.fourier_order_t,
        Kx=args.fourier_order_x,
        Ky=args.fourier_order_y,
        st=args.scale_t,
        sx=args.scale_x,
        sy=args.scale_y,
        include_raw=args.include_raw,
        use_pi=False,
    )
    args.freq_jac_norm = float(freq_jac_norm)

    name = name_model(args)
    if args.use_fourier:
        name += (
            f'_FourierT{args.fourier_order_t}_X{args.fourier_order_x}_Y{args.fourier_order_y}'
            f'_st{args.scale_t}_sx{args.scale_x}_sy{args.scale_y}'
            f'_raw{args.include_raw}_sg{args.stop_spl_grad}_PureFA100'
        )
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
            jnp.array(args.beta_spl),
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
