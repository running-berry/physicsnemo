from matplotlib import pyplot as plt
import numpy as np


def validation_plot(generated, truth, variable, experiment_name=None, step=None):
    """Produce validation plot with shared color scale for generated vs truth,
    annotated with experiment name and step.
    """
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.4)  # extra spacing between panels

    # Find global min/max across both fields
    vmin = min(np.min(generated), np.min(truth))
    vmax = max(np.max(generated), np.max(truth))

    # Titles
    title_generated = f"generated, {variable}.png"
    title_truth = "truth"

    if experiment_name is not None or step is not None:
        exp_info = []
        if experiment_name is not None:
            exp_info.append(f"Exp: {experiment_name}")
        if step is not None:
            exp_info.append(f"Step: {step}")
        exp_info_str = " | ".join(exp_info)

        # Add info to both titles
        title_generated += f"\n{exp_info_str}"
        title_truth += f"\n{exp_info_str}"

    im = a.imshow(generated, vmin=vmin, vmax=vmax, origin="lower")
    a.set_title(title_generated)
    plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    im = b.imshow(truth, vmin=vmin, vmax=vmax, origin="lower")
    b.set_title(title_truth)
    plt.colorbar(im, ax=b, fraction=0.046, pad=0.04)

    return fig


color_limits = {
    "u10m": (-5, 5),
    "v10": (-5, 5),
    "t2m": (260, 310),
    "tcwv": (0, 60),
    "msl": (0.1, 0.3),
    "refc": (-10, 30),
}


def inference_plot(
    background,
    state_pred,
    state_true,
    plot_var_background,
    plot_var_state,
    initial_time,
    lead_time,
):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    state_error = state_pred - state_true

    if plot_var_state in color_limits:
        im = ax[0].imshow(
            state_pred,
            origin="lower",  # fix orientation
            cmap="magma",
            clim=color_limits[plot_var_state],
        )
    else:
        im = ax[0].imshow(state_pred, origin="lower", cmap="magma")

    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].set_title(
        "Predicted, {}, \n initial time {} \n lead_time {} hours".format(
            plot_var_state, initial_time, lead_time
        )
    )
    if plot_var_state in color_limits:
        im = ax[1].imshow(
            state_true,
            origin="lower",  # fix orientation
            cmap="magma",
            clim=color_limits[plot_var_state],
        )
    else:
        im = ax[1].imshow(state_true, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].set_title("Actual, {}".format(plot_var_state))

    if plot_var_background in color_limits:
        im = ax[2].imshow(
            background,
            origin="lower",  # fix orientation
            cmap="magma",
            clim=color_limits[plot_var_background],
        )
    else:
        im = ax[2].imshow(background, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    ax[2].set_title("Background, {}".format(plot_var_background))

    maxerror = np.max(np.abs(state_error))
    im = ax[3].imshow(
        state_error,
        origin="lower",  # fix orientation
        cmap="RdBu_r",
        vmax=maxerror,
        vmin=-maxerror,
    )
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
    ax[3].set_title("Error, {}".format(plot_var_state))

    return fig