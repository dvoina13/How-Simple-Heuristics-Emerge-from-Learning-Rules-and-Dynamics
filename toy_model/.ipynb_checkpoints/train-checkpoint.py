import numpy as np
import torch
import torch.nn as nn

root = "/home/dvoina/myproj1/fast_and_slow_p1/different_learning_rules/toy_data/"
WEIGHTS_DIR = root + "results/Oja_new_model/"
GD_WEIGHTS_DIR = root + "results/GD_new_model/"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _shuffle(features, y):
    """Shuffle samples once so the online rule sees them in random order.

    Returns features as a float32 tensor and labels as a flat numpy array.
    """
    features = torch.as_tensor(features).float()
    labels = torch.as_tensor(y).detach().cpu().numpy().flatten()
    perm = np.random.permutation(features.shape[0])
    return features[torch.from_numpy(perm)], labels[perm]


def _init_weights(student_w_init, n_features):
    """Copy of the initial weights (the caller's tensor is never modified)."""
    if student_w_init is None:
        print("take weights from here...:", student_w_init)
        return torch.randn(1, n_features) * 0.01
    return torch.as_tensor(student_w_init).float().clone()


def _load_or_init_weights(phase, parameter_list, student_w_init, n_features):
    """Phases 2/3 load the previous phase's saved weights; phase 1 initialises."""
    if phase in ("2", "3"):
        path = WEIGHTS_DIR + _weights_filename(parameter_list, phase)
        return torch.as_tensor(torch.load(path)).float().squeeze()
    return _init_weights(student_w_init, n_features)


def _output(w, b, x, gain):
    """Student output, with the bias scaled by `gain`."""
    return torch.dot(w, x).item() + gain * b


def _bias_step(w, b, x, bias_lr):
    """Push the bias toward a target output of 0, using the current weights.

    Returns the new bias and the output (without gain) it was computed from.
    """
    y_bar = torch.dot(w, x).item() + b
    return b - bias_lr * y_bar, y_bar


def _hebbian_update(w, x, y_out, eta, target_norm=None):
    """In-place Hebbian update  dw = eta * y * x.

    If `target_norm` is given, w is rescaled to that norm after the update
    (explicit normalisation; Oja's rule is its first-order approximation).
    """
    w += eta * y_out * x
    if target_norm is not None:
        w *= target_norm / w.norm()


def _oja_update(w, x, y_out, eta, modulator=1.0):
    """In-place Oja update  dw = eta * M * (y * x - y^2 * w)."""
    w += eta * modulator * (y_out * x - y_out ** 2 * w)


def _oja_bias_step(w, b, x, gain, eta, bias_lr):
    """One online step: Oja update of w (in place), then a bias update.

    The Oja output includes the gain-scaled bias; the bias is then pushed
    toward a target output of 0 using the post-update output (without gain).
    Returns the new bias and that post-update output.
    """
    y_out = _output(w, b, x, gain)
    _oja_update(w, x, y_out, eta)
    return _bias_step(w, b, x, bias_lr)


def _targets(labels, flip):
    """Labels {0, 1} -> targets {-1, +1}, matching the sign used by _predict."""
    t = 2.0 * labels - 1.0
    return -t if flip else t


def _predict(outputs, flip):
    """Binary prediction in {0, 1} from the sign of the output."""
    outputs = np.asarray(outputs)
    return (outputs < 0 if flip else outputs > 0).astype(int)


def _window_accuracy(outputs, labels, end, flip, window=100):
    """Accuracy over the `window` samples just before index `end`."""
    preds = _predict(outputs[end - window:end], flip)
    return (preds == labels[end - window:end]).mean()


def _centering_bias(w, features):
    """Heuristic bias that centres the student's output over the dataset."""
    return -torch.mean(features @ w.reshape(-1))


def _weights_filename(params, phase, prefix="weights"):
    """Weights saved by the previous phase (phase 2 loads phase 1, 3 loads 2).

    Also checks that `phase` agrees with params["phase"] (if present), since
    the file name and the gain must refer to the same phase.
    """
    stored_phase = str(params.get("phase", phase))
    if stored_phase != phase:
        raise ValueError(f"phase={phase} but parameter_list['phase']={stored_phase}")
    p_over = 0.8 if phase == "2" else 0.5
    noise = "_".join(str(n) for n in params["noise_gp_co_ind"][:3])
    return (
        f"{prefix}_crit_{params['criterion']}_p_over_{p_over}_{params['W_init']}"
        f"_noise_gp_co_ind_{noise}_seed_{params['seed']}"
        f"_phase{int(phase) - 1}{params['centering']}.pt"
    )

def _wta_step(u1, u2, I1, I2, w_inh, dt_tau):
    """One Euler step of two rectified units with mutual inhibition:
        tau du1/dt = -u1 + [I1 - w_inh*u2]_+
        tau du2/dt = -u2 + [I2 - w_inh*u1]_+
    """
    du1 = -u1 + max(I1 - w_inh * u2, 0.0)
    du2 = -u2 + max(I2 - w_inh * u1, 0.0)
    return u1 + dt_tau * du1, u2 + dt_tau * du2
# --------------------------------------------------------------------------- #
# Plain Hebbian
# --------------------------------------------------------------------------- #

def train_hebbian_unsupervised(features, y, student_w_init=None, add_bias=False,
                               phase="1", parameter_list=None,
                               n_epochs=300, eta=1e-6, bias_lr_scale=1000,
                               normalize=True, window=100, verbose=True):
    """Train a single linear student with the plain Hebbian rule dw = eta * y * x.

    Same setup as `train_oja_unsupervised` (shuffling, phases, gain, learned
    bias, accuracy windows). Pure Hebbian learning grows |w| without bound;
    with `normalize=True` w is rescaled to its initial norm after every step,
    which is the explicit normalisation that Oja's rule approximates.

    Returns (student_w, student_b, eta, acc_arr).
    """
    phase = str(phase)
    features, labels = _shuffle(features, y)
    gain = 1 if phase == "2" else 0

    w = _load_or_init_weights(phase, parameter_list, student_w_init, features.shape[1])
    w_shape = w.shape
    w = w.reshape(-1)
    target_norm = w.norm().item() if normalize else None
    b = 0.0
    bias_lr = bias_lr_scale * eta
    acc_arr = []

    if verbose:
        print("phase", phase)
        print("Student learning via Hebbian rule...")
    with torch.no_grad():
        for _ in range(n_epochs):
            outputs = []
            for i, x in enumerate(features):
                y_out = _output(w, b, x, gain)
                _hebbian_update(w, x, y_out, eta, target_norm=target_norm)
                b, y_bar = _bias_step(w, b, x, bias_lr)
                outputs.append(y_bar)

                if i > 0 and i % window == 0:
                    acc_arr.append(_window_accuracy(outputs, labels, i, flip=False, window=window))

    w = w.reshape(w_shape)
    if not add_bias:
        b = _centering_bias(w, features)
    return w, b, eta, acc_arr


# --------------------------------------------------------------------------- #
# Oja with a fixed gain on the bias
# --------------------------------------------------------------------------- #

def train_oja_unsupervised(features, y, student_w_init=None, add_bias=False,
                           phase="1", parameter_list=None,
                           n_epochs=300, eta=1e-8, bias_lr_scale=1000,
                           window=100, verbose=True):
    """Train a single linear student with Oja's rule plus a learned bias.

    Phase 1 and 3 use gain 0 (bias not fed into the Oja output); phase 2 uses
    gain 1. Phases 2 and 3 start from the weights saved by the previous phase
    (found via `parameter_list`); phase 1 starts from `student_w_init` or a
    small random vector.

    Returns (student_w, student_b, eta, acc_arr). If `add_bias` is True,
    student_b is the learned bias; otherwise it is the centering heuristic.
    """
    phase = str(phase)
    features, labels = _shuffle(features, y)

    gain = 1 if phase == "2" else 0
    if verbose:
        print("phase", phase)

    w = _load_or_init_weights(phase, parameter_list, student_w_init, features.shape[1])
    w_shape = w.shape
    w = w.reshape(-1)   # work on a flat view; same storage as the returned tensor
    b = 0.0
    bias_lr = bias_lr_scale * eta
    acc_arr = []

    if verbose:
        print("Student learning via Oja's Rule...")
    with torch.no_grad():
        for _ in range(n_epochs):
            outputs = []
            for i, x in enumerate(features):
                b, y_bar = _oja_bias_step(w, b, x, gain, eta, bias_lr)
                outputs.append(y_bar)

                if i > 0 and i % window == 0:
                    acc_arr.append(_window_accuracy(outputs, labels, i, flip=False, window=window))

    w = w.reshape(w_shape)
    if not add_bias:
        b = _centering_bias(w, features)
    return w, b, eta, acc_arr


# --------------------------------------------------------------------------- #
# Three-factor Oja (error-modulated)
# --------------------------------------------------------------------------- #

def _sample_error(y_out, label, target, error_type, flip):
    """Per-sample error: 0/1 mistake, or squared distance to the ±1 target."""
    if error_type == "binary":
        return float(_predict(y_out, flip) != label)
    if error_type == "squared":
        return (target - y_out) ** 2
    raise ValueError(f"error_type must be 'binary' or 'squared', got {error_type!r}")


def train_3factor_oja(features, y, student_w_init=None, add_bias=False,
                      phase="1", parameter_list=None, error_type="binary",
                      n_epochs=1000, eta=1e-8, bias_lr_scale=1000, alpha=0.1,
                      flip=False, window=100, verbose=True):
    """Oja's rule gated by a third factor: dw = eta * M * (y * x - y^2 * w).

    The modulator compares the current sample's error with a running baseline
    (an exponential moving average with rate `alpha`):
        M = -(err - err_bar)
    so lower-than-usual error gives an Oja step and higher-than-usual error an
    anti-Oja step. Subtracting the baseline keeps the mean update near zero,
    as in reward-modulated Hebbian learning. No explicit normalisation: the
    Oja decay term keeps |w| near 1 on average (note that when M < 0 the decay
    term also flips sign, so |w| is only held on average, not at every step).

    error_type:
        "squared" - err = (t - y)^2 with targets t in {-1, +1}
        "binary"  - err = 1 if the sign prediction is wrong, else 0
                    (no squared error; same 0/1 error as the Adler variant)

    `flip` sets the sign convention (False: output > 0 predicts label 1, as in
    train_oja_unsupervised; True: output < 0 predicts 1, as in the Adler variant).
    Weights and bias are otherwise handled as in `train_oja_unsupervised`.

    Returns (student_w, student_b, eta, acc_arr, err_arr), where err_arr is the
    running error baseline recorded at each accuracy window.
    """
    phase = str(phase)
    features, labels = _shuffle(features, y)
    targets = _targets(labels, flip)
    gain = 1 if phase == "2" else 0

    w = _load_or_init_weights(phase, parameter_list, student_w_init, features.shape[1])
    
    w_shape = w.shape
    w = w.reshape(-1)
    b = 0.0
    bias_lr = bias_lr_scale * eta
    err_bar = None   # initialised from the first sample
    acc_arr, err_arr = [], []

    if verbose:
        print("phase", phase, "error_type", error_type)
        print("Student learning via three-factor Oja rule...")
    with torch.no_grad():
        for _ in range(n_epochs):
            outputs = []
            for i, x in enumerate(features):
                y_out = _output(w, b, x, gain)
                err = _sample_error(y_out, labels[i], targets[i], error_type, flip)
                if err_bar is None:
                    err_bar = err
                modulator = err_bar #-(err - err_bar)
                err_bar = (1 - alpha) * err_bar + alpha * err

                _oja_update(w, x, y_out, eta, modulator)
                b, y_bar = _bias_step(w, b, x, bias_lr)
                outputs.append(y_bar)

                if i > 0 and i % window == 0:
                    acc_arr.append(_window_accuracy(outputs, labels, i, flip=flip, window=window))
                    err_arr.append(err_bar)

    w = w.reshape(w_shape)
    if not add_bias:
        b = _centering_bias(w, features)
    return w, b, eta, acc_arr, err_arr


# --------------------------------------------------------------------------- #
# Oja with an error-driven gain (latent phase oscillator)
# --------------------------------------------------------------------------- #

def _gain_from_latent(theta):
    """Map the latent phase to a gain in [0, 1]: 0 at theta=0, 1 at theta=pi."""
    return (1 - np.cos(theta)) / 2

def train_oja_unsupervised_adler(features, y, student_w_init=None, add_bias=False,
                                 phase="1", n_epochs=40, b_init=None,
                                 eta_rel=0.5, bias_lr=None, alpha=0.02,
                                 beta=0.01, gamma=10.0, e_switch=0.08,
                                 n_hold=1000, verbose=False):
    """Adler switch with rescaled parameters and a commitment period (n_hold).

    eta = eta_rel / E|x|^2.  A = 2*gamma*e_switch (switch when smoothed error >
    e_switch; ~0.08 at noise 0.1, ~0.25 at noise 1.0).  bias_lr=None means 0.5
    in phases 1-2 and 0.01 in phase 3: the slow bias in phase 3 gives the bad
    centering that sends the gate back to summing.  Inputs must not be
    pre-centered (load_data(centering=False)), or w can diverge to NaN.

    Returns (student_w, student_b, eta, acc_arr, gain_arr).
    """
    phase = str(phase)
    features = torch.as_tensor(features).float()
    labels = torch.as_tensor(y).detach().cpu().numpy().flatten()
    perm = np.random.permutation(features.shape[0])
    features, labels = features[torch.from_numpy(perm)], labels[perm]
    n_samples, n_total = features.shape

    if student_w_init is None:
        w = -torch.randn(n_total).abs() * 0.01
    else:
        w = torch.as_tensor(student_w_init).float().clone().reshape(-1)

    if bias_lr is None:
        bias_lr = 0.01 if phase == "3" else 0.5
    b = float(b_init) if b_init is not None else -(features[:50] @ w).mean().item()
    if features.mean(0).abs().max() < 1e-3 * features.abs().max():
        warnings.warn("features look pre-centered; use load_data(centering=False), "
                      "centered inputs can make the weights diverge.")

    eta = eta_rel / (features ** 2).sum(1).mean().item()
    A = 2 * gamma * e_switch
    theta = np.pi if phase == "3" else 0.0
    gain = (1 - np.cos(theta)) / 2
    error, hold = 0.0, 0

    acc_arr, gain_arr = [], []
    with torch.no_grad():
        for _ in range(n_epochs):
            y_arr = []
            for i in range(n_samples):
                x = features[i]
                y_s = torch.dot(w, x).item() + gain * b
                w += eta * (y_s * x - y_s ** 2 * w)
                y_new = torch.dot(w, x).item() + b
                b -= bias_lr * y_new
                y_arr.append(y_new)

                pred = (1 - np.sign(y_new)) // 2              # output < 0 predicts 1
                error = (1 - alpha) * error + alpha * float(pred != labels[i])
                gain_signal = 1 - 2 * abs(error - 0.5)
                drive = 0.0 if hold > 0 else gamma * gain_signal
                hold = max(hold - 1, 0)

                theta = (theta + beta * (drive - A * np.sin(2 * theta))) % (2 * np.pi)
                new_gain = (1 - np.cos(theta)) / 2
                if (new_gain - 0.5) * (gain - 0.5) < 0:       # switched: commit
                    hold = n_hold
                gain = new_gain

                if i > 0 and i % 100 == 0:
                    p = (1 - np.sign(np.array(y_arr[i - 100:i]))) // 2
                    acc_arr.append((p == labels[i - 100:i]).mean())
                    gain_arr.append(gain)
                    if verbose:
                        print(f"i={i:4d} acc={acc_arr[-1]:.2f} error={error:.3f} "
                              f"theta={theta:.2f} gain={gain:.2f} hold={hold}")

    w = w.reshape(1, -1)
    if not add_bias:
        b = -torch.mean(w @ features.T).item()
    return w, b, eta, acc_arr, gain_arr


###############################
# Oja with switch, 2 pathways #
###############################

def train_oja_switch(features, y, student_w_init=None, add_bias=False,
                     init_state=None, n_epochs=40, eta_rel=0.5, bias_lr=0.5,
                     alpha_err=0.05, alpha_pol=0.05, e_tol=0.2,
                     w_inh=1.2, cost2=0.2, tau=20.0, u_init=(0.0, 0.0), eps=1e-6,
                     window=100, verbose=True):
    """Two Oja readouts learned in parallel, with a gate that switches between them.
 
    Readouts (both unsupervised, same eta, same input):
      y1 = w1.x + b1   uncentered Oja (gain=0). b1 tracks -mean(y1) but only
                       sets the threshold; it never enters learning.
                       -> summing heuristic under asymmetric representations.
      y2 = w2.x + b2   centered Oja (gain=1): learns on y - y_mean.
                       -> balanced heuristic.
 
    Each readout's sign from Oja is arbitrary, so each has a polarity s_k in
    {-1, +1}: the sign of an EMA (rate `alpha_pol`) of y_k * (2*label - 1).
    Each readout makes its own prediction, and its running error e_k is an
    EMA (rate `alpha_err`) of its own mistakes.
 
    Gate: winner-take-all between two units with mutual inhibition,
        tau du1/dt = -u1 + [I1 - w_inh*u2]_+
        tau du2/dt = -u2 + [I2 - w_inh*u1]_+
    (Euler steps of one trial; `tau` is in trials). Inputs are capped
    above-chance accuracy,
        I_k = clip(1 - 2*e_k, 0, 1 - 2*e_tol),
    so I_k = 0 at chance and saturates once the error is below `e_tol`.
    Gate and output:
        g = u2 / (u1 + u2 + eps),   y = (1 - g) * s1*y1 + g * s2*y2.
    - "Good enough": once both readouts are under e_tol their inputs are
      equal, so the unit that is already active stays active (satisficing).
    - Penalty on change: with w_inh > 1 the winner sits at u_k = I_k and the
      loser is silent until I_j > w_inh * I_k. w_inh is the switching cost:
      the challenger needs w_inh times the incumbent's evidence.
    - Cost: y2's input is reduced by `cost2`. While both readouts are good
      enough, y1 (summing) has the larger input, so it wins in phase 1 and takes
      back over in phase 3. Windows (cap = 1 - 2*e_tol):
        phase 2 switch to y2:  cost2 < cap - w_inh * I1   (I1 = summing's input there)
        phase 3 return to y1:  cost2 > cap * (1 - 1/w_inh)
      With w_inh = 1.2, cap = 0.6: roughly 0.1 < cost2 < 0.3. cost2 = 0 gives the
      plain gate (no return to summing in phase 3; phase 1 decided by a race).
 
    Timescales: y2 only behaves as centered Oja if b2 tracks the mean much
    faster than w2 changes (bias_lr >> eta * lambda1). With bias_lr = 1000*eta
    on the toy data the mean leaked into w2 and it never converged; 1e4*eta
    worked. Re-check this ratio on your own data scale.
 
    To carry state across phases, pass the returned `state` dict as
    `init_state` of the next call (weights, biases, polarities, errors, u1, u2).
 
    Returns (w1, b1, w2, b2, eta, acc_arr, gate_arr, err_arr, state), where
    err_arr holds (e1, e2) per window. If `add_bias` is False, b1 and b2 are
    replaced by the centering heuristic over the dataset.
    """
    features, labels = _shuffle(features, y)
    n_features = features.shape[1]
    st = dict(init_state or {})
 
    w1 = _init_weights(st.get("w1", student_w_init), n_features).reshape(-1)
    w2 = _init_weights(st.get("w2", w1.clone()), n_features).reshape(-1)
    b1, b2 = st.get("b1", 0.0), st.get("b2", 0.0)
    c1, c2 = st.get("c1", 0.0), st.get("c2", 0.0)      # polarity EMAs
    e1, e2 = st.get("e1", 0.5), st.get("e2", 0.5)      # error EMAs, start at chance
    u1, u2 = st.get("u1", u_init[0]), st.get("u2", u_init[1])
    g = u2 / (u1 + u2 + eps)
    I_cap = 1 - 2 * e_tol
    dt_tau = 1.0 / tau
    eta = eta_rel / (features ** 2).sum(1).mean().item()   # scale-free learning rate
 
    acc_arr, gate_arr, err_arr = [], [], []
    with torch.no_grad():
        for _ in range(n_epochs):
            correct = []
            for i, x in enumerate(features):
                lab = labels[i]
                b1, y1 = _oja_bias_step(w1, b1, x, 0, eta, bias_lr)
                b2, y2 = _oja_bias_step(w2, b2, x, 1, eta, bias_lr)
 
                # Polarity of each readout (which side is class 1)
                t = 2 * lab - 1
                c1 = (1 - alpha_pol) * c1 + alpha_pol * y1 * t
                c2 = (1 - alpha_pol) * c2 + alpha_pol * y2 * t
                s1, s2 = (1.0 if c1 >= 0 else -1.0), (1.0 if c2 >= 0 else -1.0)
 
                # Each readout's own error
                e1 = (1 - alpha_err) * e1 + alpha_err * float((s1 * y1 > 0) != lab)
                e2 = (1 - alpha_err) * e2 + alpha_err * float((s2 * y2 > 0) != lab)
 
                # Combined output and its accuracy
                y_out = (1 - g) * s1 * y1 + g * s2 * y2
                correct.append(float((y_out > 0) == lab))
 
                # Winner-take-all gate on capped above-chance accuracy
                I1 = min(max(1 - 2 * e1, 0.0), I_cap)
                I2 = min(max(1 - 2 * e2, 0.0), I_cap) - cost2   # cost of the balanced pathway
                u1, u2 = _wta_step(u1, u2, I1, I2, w_inh, dt_tau)
                g = u2 / (u1 + u2 + eps)
 
                if i > 0 and i % window == 0:
                    acc = np.mean(correct[i - window:i])
                    acc_arr.append(acc)
                    gate_arr.append(g)
                    err_arr.append((e1, e2))
                    if verbose:
                        print(f"i={i:5d}  acc={acc:.2f}  e1={e1:.3f}  e2={e2:.3f}  "
                              f"u1={u1:.3f}  u2={u2:.3f}  g={g:.3f}")
 
    state = dict(w1=w1.clone(), w2=w2.clone(), b1=b1, b2=b2, c1=c1, c2=c2,
                 e1=e1, e2=e2, u1=u1, u2=u2)
    if not add_bias:
        b1, b2 = _centering_bias(w1, features), _centering_bias(w2, features)
    return w1, b1, w2, b2, eta, acc_arr, gate_arr, err_arr, state
    
# --------------------------------------------------------------------------- #
# Gradient-descent baseline (trains an nn.Linear-style network)
# --------------------------------------------------------------------------- #

def _gd_correct(out, y, criterion):
    """Boolean mask of correct predictions for MSE (0.5 threshold) or BCE-with-logits (0 threshold)."""
    out, y = out.reshape(-1), y.reshape(-1)
    if isinstance(criterion, nn.MSELoss):
        threshold = 0.5
    elif isinstance(criterion, nn.BCEWithLogitsLoss):
        threshold = 0.0
    else:
        raise ValueError(f"Unsupported criterion: {criterion}")
    return ((out < threshold) & (y == 0)) | ((out > threshold) & (y == 1))


def _gd_accuracy(out, y, criterion):
    return _gd_correct(out, y, criterion).float().mean()


def _predict_dataset(network, dataloader, device):
    """Run the network over every batch; return (outputs, X, y) concatenated."""
    outs, xs, ys = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device)
            outs.append(network(X).reshape(-1))
            xs.append(X)
            ys.append(y.reshape(-1))
    return torch.cat(outs), torch.cat(xs), torch.cat(ys)


def train_model(network, train_dataloader, test_dataloader, optimizer, criterion,
                n_epochs, parameter_list, phase, device, log_every=5, verbose=True):
    """Train `network` with gradient descent, then evaluate on the test set.

    Phases 2 and 3 start from the weights saved by the previous phase. If
    parameter_list["W_init"] != "no_initW", 1 is then added to every weight
    (as in the original, this happens in every phase, including after loading).

    Returns (network, training_acc, valid_acc): training_acc has one entry per
    batch; valid_acc is computed over the whole test set.
    """
    phase = str(phase)
    network.float()

    if phase in ("2", "3"):
        path = GD_WEIGHTS_DIR + _weights_filename(parameter_list, phase, prefix="weights_GD")
        with torch.no_grad():
            network.weight.copy_(torch.load(path))

    if parameter_list["W_init"] != "no_initW":
        if verbose:
            print("init W", parameter_list["W_init"])
        with torch.no_grad():
            network.weight.add_(1)

    network.train()
    training_acc = []
    for _ in range(n_epochs):
        for bid, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device).float(), y.to(device).float().reshape(-1)

            optimizer.zero_grad()
            logits = network(X).reshape(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            acc = _gd_accuracy(logits.detach(), y, criterion).item()
            training_acc.append(acc)
            if verbose and bid % log_every == 0:
                print(f"batch {bid:4d}  loss {loss.item():.4f}  accuracy {acc:.3f}")

    network.eval()
    out, _, y_test = _predict_dataset(network, test_dataloader, device)
    valid_acc = _gd_accuracy(out, y_test, criterion)
    if verbose:
        print("testing accuracy teacher:", valid_acc.item())
    return network, training_acc, valid_acc


def test_model_simple(network, test_dataloader, criterion, device, verbose=True):
    """Evaluate `network` on the whole test set.

    Returns (valid_acc, valid_loss, X, y), with X and y concatenated over all
    test batches.
    """
    network.eval()
    network.float()
    out, X, y = _predict_dataset(network, test_dataloader, device)
    valid_loss = criterion(out, y.float())
    valid_acc = _gd_accuracy(out, y, criterion)
    if verbose:
        print(f"VALID Loss {valid_loss:.4f}, Accuracy {valid_acc:.4f}")
    return valid_acc, valid_loss, X, y