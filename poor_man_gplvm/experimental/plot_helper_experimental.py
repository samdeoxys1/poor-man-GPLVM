"""
Experimental plotting helpers (cluster-notebook friendly).
"""

import base64
import io
import pathlib
import uuid

import IPython.display as ipd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def tuning_browser_prev_next_jshtml(
    res,
    neuron_ids=None,
    init_key="initial",
    final_key="final",
    tuning_key="tuning_grid_xr",
    figsize=(10, 4),
    cmap="viridis",
    interpolation="none",
    vmin=None,
    vmax=None,
    origin="lower",
    aspect="auto",
    image_dpi=120,
    fig=None,
    axs=None,
    add_colorbar=False,
    constrained_layout=True,
    display_inline=True,
    save_html_path=None,
    panel_title_init="initial",
    panel_title_final="final",
):
    """
    Build a Prev/Next HTML browser for tuning maps across neurons (no ipywidgets).

    Cluster Jupyter example
    ```python
    import poor_man_gplvm.experimental.plot_helper_experimental as phe

    out = phe.tuning_browser_prev_next_jshtml(
        res,
        neuron_ids=np.array([0, 1, 2, 13, 20]),
        cmap='magma',
        figsize=(11, 4),
        save_html_path='/mnt/home/szheng/ceph/ad/poor_gplvm/figure/tuning_prev_next.html',
    )
    ```
    """
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["image.interpolation"] = "none"

    tuning_init = res[init_key][tuning_key]
    tuning_final = res[final_key][tuning_key]

    if neuron_ids is None:
        neuron_ids = np.asarray(tuning_init.coords["neuron"].values)
    else:
        neuron_ids = np.asarray(neuron_ids)

    if vmin is None:
        vmin = float(
            np.nanmin(
                [
                    np.asarray(tuning_init.sel(neuron=neuron_ids).values),
                    np.asarray(tuning_final.sel(neuron=neuron_ids).values),
                ]
            )
        )
    if vmax is None:
        vmax = float(
            np.nanmax(
                [
                    np.asarray(tuning_init.sel(neuron=neuron_ids).values),
                    np.asarray(tuning_final.sel(neuron=neuron_ids).values),
                ]
            )
        )

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=constrained_layout)
    axs = np.asarray(axs).ravel()

    if axs.size < 2:
        raise ValueError(f"Need 2 axes, got {axs.size}")

    data_urls = []
    cbar = None
    for neuron_id in neuron_ids:
        da_init = tuning_init.sel(neuron=neuron_id)
        da_final = tuning_final.sel(neuron=neuron_id)

        arr_init = np.asarray(da_init.values)
        arr_final = np.asarray(da_final.values)

        axs[0].clear()
        axs[1].clear()

        im0 = axs[0].imshow(
            arr_init.T,
            cmap=cmap,
            interpolation=interpolation,
            vmin=vmin,
            vmax=vmax,
            origin=origin,
            aspect=aspect,
        )
        axs[1].imshow(
            arr_final.T,
            cmap=cmap,
            interpolation=interpolation,
            vmin=vmin,
            vmax=vmax,
            origin=origin,
            aspect=aspect,
        )

        dim_init = list(da_init.dims)
        if len(dim_init) >= 2:
            axs[0].set_xlabel(str(dim_init[0]))
            axs[0].set_ylabel(str(dim_init[1]))
            axs[1].set_xlabel(str(dim_init[0]))
            axs[1].set_ylabel(str(dim_init[1]))

        axs[0].set_title(f"{panel_title_init} | neuron={int(neuron_id)}")
        axs[1].set_title(f"{panel_title_final} | neuron={int(neuron_id)}")

        if add_colorbar and (cbar is None):
            cbar = fig.colorbar(im0, ax=axs.tolist(), shrink=0.9)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=image_dpi, bbox_inches=None)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        data_urls.append(f"data:image/png;base64,{encoded}")
        buf.close()

    container_id = f"tuning_browser_{uuid.uuid4().hex[:8]}"
    neuron_ids_json = np.asarray(neuron_ids).astype(int).tolist()
    data_urls_json = data_urls

    html = f"""
<div id="{container_id}" style="font-family: sans-serif; max-width: 1200px;">
  <div style="margin-bottom: 8px; display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
    <button id="{container_id}_prev" style="padding: 4px 10px;">Prev</button>
    <button id="{container_id}_next" style="padding: 4px 10px;">Next</button>
    <label style="margin-left: 8px;">index</label>
    <input id="{container_id}_idx" type="number" min="0" max="{len(neuron_ids_json)-1}" value="0" style="width: 70px;" />
    <label>neuron</label>
    <span id="{container_id}_neuron" style="min-width: 50px; display: inline-block;"></span>
    <input id="{container_id}_slider" type="range" min="0" max="{len(neuron_ids_json)-1}" value="0" step="1" style="width: 360px;" />
  </div>
  <img id="{container_id}_img" src="" style="max-width: 100%; border: 1px solid #ddd;" />
</div>
<script>
(function() {{
  const neuronIds = {neuron_ids_json};
  const dataUrls = {data_urls_json};
  const img = document.getElementById("{container_id}_img");
  const idxBox = document.getElementById("{container_id}_idx");
  const neuronText = document.getElementById("{container_id}_neuron");
  const slider = document.getElementById("{container_id}_slider");
  const prevBtn = document.getElementById("{container_id}_prev");
  const nextBtn = document.getElementById("{container_id}_next");
  let idx = 0;

  function clamp(v) {{
    return Math.max(0, Math.min(dataUrls.length - 1, v));
  }}

  function render(i) {{
    idx = clamp(i);
    img.src = dataUrls[idx];
    idxBox.value = idx;
    slider.value = idx;
    neuronText.textContent = neuronIds[idx];
  }}

  prevBtn.onclick = function() {{
    render(idx - 1);
  }};
  nextBtn.onclick = function() {{
    render(idx + 1);
  }};
  idxBox.onchange = function() {{
    render(parseInt(idxBox.value || "0", 10));
  }};
  slider.oninput = function() {{
    render(parseInt(slider.value, 10));
  }};

  render(0);
}})();
</script>
"""

    if display_inline:
        ipd.display(ipd.HTML(html))

    if save_html_path is not None:
        save_path = pathlib.Path(save_html_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(html)
        print(f"saved html to: {save_path}")

    out = {
        "html": html,
        "n_neuron": int(len(neuron_ids_json)),
        "neuron_ids": np.asarray(neuron_ids_json),
        "vmin": float(vmin),
        "vmax": float(vmax),
    }
    return out
