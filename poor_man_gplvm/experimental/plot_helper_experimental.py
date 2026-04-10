"""
Experimental plotting helpers (cluster-notebook friendly).
"""

import pathlib
import uuid
import json

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
    color_range_mode="per_image",
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

    global_vmin = float(
        np.nanmin(
            [
                np.asarray(tuning_init.sel(neuron=neuron_ids).values),
                np.asarray(tuning_final.sel(neuron=neuron_ids).values),
            ]
        )
    )
    global_vmax = float(
        np.nanmax(
            [
                np.asarray(tuning_init.sel(neuron=neuron_ids).values),
                np.asarray(tuning_final.sel(neuron=neuron_ids).values),
            ]
        )
    )
    if vmin is None:
        vmin = global_vmin
    if vmax is None:
        vmax = global_vmax

    if render_mode != "client_canvas":
        raise ValueError(f"Unsupported render_mode={render_mode}, use 'client_canvas'")

    # Robust to neuron axis position and xarray dim order:
    # always build [n_neuron, height, width] with same orientation as imshow(arr.T)
    init_arr = np.asarray(
        [np.asarray(tuning_init.sel(neuron=nid).values, dtype=np.float32).T for nid in neuron_ids],
        dtype=np.float32,
    )
    final_arr = np.asarray(
        [np.asarray(tuning_final.sel(neuron=nid).values, dtype=np.float32).T for nid in neuron_ids],
        dtype=np.float32,
    )

    h_px = int(init_arr.shape[1])
    w_px = int(init_arr.shape[2])

    da0 = tuning_init.sel(neuron=neuron_ids[0])
    dim_init = list(da0.dims)
    if len(dim_init) >= 2:
        # Because we plot arr.T, axes labels are swapped from raw da order
        xlabel = str(dim_init[0])
        ylabel = str(dim_init[1])
    else:
        xlabel = "x"
        ylabel = "y"

    cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    lut = np.asarray(cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255, dtype=np.uint8).tolist()

    container_id = f"tuning_browser_{uuid.uuid4().hex[:8]}"
    neuron_ids_json = json.dumps(np.asarray(neuron_ids).astype(int).tolist())
    # Use JSON so NaN becomes JS NaN (valid in script), not python "nan" (invalid JS)
    init_data_json = json.dumps(np.asarray(init_arr, dtype=np.float32).tolist(), allow_nan=True)
    final_data_json = json.dumps(np.asarray(final_arr, dtype=np.float32).tolist(), allow_nan=True)
    lut_json = json.dumps(lut)
    panel_title_init_json = json.dumps(str(panel_title_init))
    panel_title_final_json = json.dumps(str(panel_title_final))
    color_range_mode_json = json.dumps(str(color_range_mode))

    html = f"""
<div id="{container_id}" style="font-family: sans-serif; max-width: 1200px;">
  <div style="margin-bottom: 8px; display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
    <button id="{container_id}_prev" style="padding: 4px 10px;">Prev</button>
    <button id="{container_id}_next" style="padding: 4px 10px;">Next</button>
    <label style="margin-left: 8px;">index</label>
    <input id="{container_id}_idx" type="number" min="0" max="{len(neuron_ids)-1}" value="0" style="width: 70px;" />
    <label>neuron</label>
    <span id="{container_id}_neuron" style="min-width: 50px; display: inline-block;"></span>
    <input id="{container_id}_slider" type="range" min="0" max="{len(neuron_ids)-1}" value="0" step="1" style="width: 360px;" />
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
  const vminGlobal = {float(vmin)};
  const vmaxGlobal = {float(vmax)};
  const colorRangeMode = {color_range_mode_json};
  const titleInitBase = {panel_title_init_json};
  const titleFinalBase = {panel_title_final_json};
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

  function getArrayMinMax(arr2d) {{
    let vmin = Infinity;
    let vmax = -Infinity;
    for (let y = 0; y < arr2d.length; y++) {{
      const row = arr2d[y];
      for (let x = 0; x < row.length; x++) {{
        const v = row[x];
        if (!Number.isFinite(v)) {{
          continue;
        }}
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }}
    }}
    if (!Number.isFinite(vmin) || !Number.isFinite(vmax)) {{
      return [0.0, 1.0];
    }}
    if (vmax <= vmin) {{
      return [vmin, vmin + 1e-12];
    }}
    return [vmin, vmax];
  }}

  function drawHeatmap(ctx, arr2d, vmin, vmax) {{
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
    const arrInit = dataInit[idx];
    const arrFinal = dataFinal[idx];
    let vminInit = vminGlobal;
    let vmaxInit = vmaxGlobal;
    let vminFinal = vminGlobal;
    let vmaxFinal = vmaxGlobal;

    if (colorRangeMode === "per_image") {{
      [vminInit, vmaxInit] = getArrayMinMax(arrInit);
      [vminFinal, vmaxFinal] = getArrayMinMax(arrFinal);
    }}

    drawHeatmap(ctxInit, arrInit, vminInit, vmaxInit);
    drawHeatmap(ctxFinal, arrFinal, vminFinal, vmaxFinal);
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
        "color_range_mode": color_range_mode,
    }
    print(f"rendered tuning browser ({render_mode}) for n_neuron={len(neuron_ids_json)}")
    return out
