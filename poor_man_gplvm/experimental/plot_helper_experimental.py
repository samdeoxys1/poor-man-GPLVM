"""
Experimental plotting helpers (cluster-notebook friendly).
"""

import pathlib
import uuid

import IPython.display as ipd
import matplotlib
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
    render_mode="client_canvas",
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

    if render_mode != "client_canvas":
        raise ValueError(f"Unsupported render_mode={render_mode}, use 'client_canvas'")

    init_arr = np.asarray(tuning_init.sel(neuron=neuron_ids).values, dtype=np.float32)
    final_arr = np.asarray(tuning_final.sel(neuron=neuron_ids).values, dtype=np.float32)

    # Keep same orientation as previous imshow(arr.T, ...)
    init_arr = np.transpose(init_arr, (0, 2, 1))
    final_arr = np.transpose(final_arr, (0, 2, 1))

    h_px = int(init_arr.shape[1])
    w_px = int(init_arr.shape[2])

    dim_init = list(tuning_init.dims)
    if len(dim_init) >= 3:
        xlabel = str(dim_init[1])
        ylabel = str(dim_init[2])
    else:
        xlabel = "x"
        ylabel = "y"

    cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    lut = np.asarray(cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255, dtype=np.uint8).tolist()

    container_id = f"tuning_browser_{uuid.uuid4().hex[:8]}"
    neuron_ids_json = np.asarray(neuron_ids).astype(int).tolist()
    init_data_json = init_arr.tolist()
    final_data_json = final_arr.tolist()
    lut_json = lut

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
  <div style="margin-bottom: 4px; font-size: 12px;">
    <span id="{container_id}_title_init" style="display:inline-block; width: 49%;"></span>
    <span id="{container_id}_title_final" style="display:inline-block; width: 49%;"></span>
  </div>
  <div style="display: flex; gap: 8px; align-items: flex-start; flex-wrap: wrap;">
    <div>
      <canvas id="{container_id}_canvas_init" width="{w_px}" height="{h_px}" style="border:1px solid #ddd; image-rendering: pixelated;"></canvas>
      <div style="font-size: 11px; margin-top: 2px;">{ylabel} vs {xlabel}</div>
    </div>
    <div>
      <canvas id="{container_id}_canvas_final" width="{w_px}" height="{h_px}" style="border:1px solid #ddd; image-rendering: pixelated;"></canvas>
      <div style="font-size: 11px; margin-top: 2px;">{ylabel} vs {xlabel}</div>
    </div>
  </div>
  {"<div style='margin-top:6px; font-size:11px;'>color range: " + str(round(vmin, 4)) + " to " + str(round(vmax, 4)) + "</div>" if add_colorbar else ""}
</div>
<script>
(function() {{
  const neuronIds = {neuron_ids_json};
  const dataInit = {init_data_json};
  const dataFinal = {final_data_json};
  const lut = {lut_json};
  const vmin = {float(vmin)};
  const vmax = {float(vmax)};
  const titleInitBase = {repr(panel_title_init)};
  const titleFinalBase = {repr(panel_title_final)};
  const idxBox = document.getElementById("{container_id}_idx");
  const neuronText = document.getElementById("{container_id}_neuron");
  const slider = document.getElementById("{container_id}_slider");
  const prevBtn = document.getElementById("{container_id}_prev");
  const nextBtn = document.getElementById("{container_id}_next");
  const titleInit = document.getElementById("{container_id}_title_init");
  const titleFinal = document.getElementById("{container_id}_title_final");
  const canvasInit = document.getElementById("{container_id}_canvas_init");
  const canvasFinal = document.getElementById("{container_id}_canvas_final");
  const ctxInit = canvasInit.getContext("2d");
  const ctxFinal = canvasFinal.getContext("2d");
  let idx = 0;

  function clamp(v) {{
    return Math.max(0, Math.min(dataInit.length - 1, v));
  }}

  function drawHeatmap(ctx, arr2d) {{
    const h = arr2d.length;
    const w = arr2d[0].length;
    const img = ctx.createImageData(w, h);
    const out = img.data;
    const den = (vmax - vmin) > 1e-12 ? (vmax - vmin) : 1.0;
    let p = 0;
    for (let y = 0; y < h; y++) {{
      for (let x = 0; x < w; x++) {{
        const v = arr2d[y][x];
        let t = (v - vmin) / den;
        if (!Number.isFinite(t)) {{
          t = 0.0;
        }}
        t = Math.max(0.0, Math.min(1.0, t));
        const k = Math.min(255, Math.max(0, Math.floor(t * 255)));
        out[p] = lut[k][0];
        out[p + 1] = lut[k][1];
        out[p + 2] = lut[k][2];
        out[p + 3] = 255;
        p += 4;
      }}
    }}
    ctx.putImageData(img, 0, 0);
  }}

  function render(i) {{
    idx = clamp(i);
    drawHeatmap(ctxInit, dataInit[idx]);
    drawHeatmap(ctxFinal, dataFinal[idx]);
    idxBox.value = idx;
    slider.value = idx;
    neuronText.textContent = neuronIds[idx];
    titleInit.textContent = titleInitBase + " | neuron=" + neuronIds[idx];
    titleFinal.textContent = titleFinalBase + " | neuron=" + neuronIds[idx];
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
        "shape_init": np.asarray(init_arr.shape),
        "shape_final": np.asarray(final_arr.shape),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "render_mode": render_mode,
    }
    print(f"rendered tuning browser ({render_mode}) for n_neuron={len(neuron_ids_json)}")
    return out
