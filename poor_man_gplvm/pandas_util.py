import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Union

Spec = Union[List[Any], Tuple[Any, ...], Dict[str, Any]]

def _btick(col: str) -> str:
    return f"`{col}`" if re.search(r"\W", col) else col

def _new_var(env: Dict[str, Any], var_id: List[int], v: Any) -> str:
    name = f"v{var_id[0]}"
    var_id[0] += 1
    env[name] = v
    return name

def _ensure_listlike(x):
    if isinstance(x, (list, tuple, set, pd.Index, np.ndarray)):
        return list(x)
    return [x]

def _is_logic_list(node: Any) -> bool:
    return isinstance(node, (list, tuple)) and node and isinstance(node[0], str) and node[0].lower() in {"all","any","not"}

def _is_logic_dict(node: Any) -> bool:
    return isinstance(node, dict) and any(k in node for k in ("all","any","not"))

def _is_leaf_list(node: Any) -> bool:
    return (
        isinstance(node, (list, tuple))
        and len(node) >= 2
        and isinstance(node[0], str)
        and node[0].lower() not in {"all","any","not"}
    )

def _compile_leaf_list(df: pd.DataFrame, leaf: Union[List[Any], Tuple[Any, ...]], env: Dict[str, Any], var_id: List[int]):
    """Compile ["col","op", value?, opts?] into (mask, query_snippet)."""
    col = leaf[0]
    op  = str(leaf[1]).lower()
    val = leaf[2] if len(leaf) >= 3 else None
    opts = leaf[3] if len(leaf) >= 4 and isinstance(leaf[3], dict) else {}
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not in DataFrame.")
    s = df[col]
    col_bt = _btick(col)

    # Comparators
    if op in {"==","!=","<",">","<=",">="}:
        var = _new_var(env, var_id, val)
        mask = getattr(s, {"==":"eq","!=":"ne","<":"lt",">":"gt","<=":"le",">=":"ge"}[op])(env[var])
        return mask, f"{col_bt} {op} @{var}"

    # Membership
    if op in {"in","not in"}:
        vals = _ensure_listlike(val)
        var = _new_var(env, var_id, vals)
        mask = s.isin(env[var])
        q = f"{col_bt} in @{var}"
        if op == "not in":
            mask = ~mask
            q = f"{col_bt} not in @{var}"
        return mask, q

    # Between
    if op == "between":
        if not (isinstance(val, (list, tuple)) and len(val) == 2):
            raise ValueError("between expects value=(low, high).")
        low, high = val
        inclusive = opts.get("inclusive", "both")  # 'both'|'neither'|'left'|'right'
        mask = s.between(low, high, inclusive=inclusive)
        vL, vH = _new_var(env, var_id, low), _new_var(env, var_id, high)
        if inclusive in ("both", True):
            q = f"(@{vL} <= {col_bt}) and ({col_bt} <= @{vH})"
        elif inclusive in ("neither", False):
            q = f"(@{vL} < {col_bt}) and ({col_bt} < @{vH})"
        elif inclusive == "left":
            q = f"(@{vL} <= {col_bt}) and ({col_bt} < @{vH})"
        elif inclusive == "right":
            q = f"(@{vL} < {col_bt}) and ({col_bt} <= @{vH})"
        else:
            q = f"(@{vL} <= {col_bt}) and ({col_bt} <= @{vH})"
        return mask, q

    # Null checks
    if op in {"isna","isnull"}:
        return s.isna(), f"{col_bt}.isnull()"
    if op in {"notna","notnull"} or op == "notna":  # allow "notna" explicitly
        return s.notna(), f"{col_bt}.notnull()"

    # String ops
    if op in {"contains","startswith","endswith","regex"}:
        case = bool(opts.get("case", True))
        na = opts.get("na", False)
        strobj = s.astype("string")
        if op == "contains":
            pat = str(val)
            regex = bool(opts.get("regex", True))
            mask = strobj.str.contains(pat, case=case, na=na, regex=regex)
            var = _new_var(env, var_id, pat)
            return mask, f"{col_bt}.str.contains(@{var}, case={case}, na={na}, regex={regex})"
        if op == "regex":
            pat = str(val)
            mask = strobj.str.contains(pat, case=case, na=na, regex=True)
            var = _new_var(env, var_id, pat)
            return mask, f"{col_bt}.str.contains(@{var}, case={case}, na={na}, regex=True)"
        if op == "startswith":
            pat = str(val)
            mask = strobj.str.startswith(pat, na=na)
            var = _new_var(env, var_id, pat)
            return mask, f"{col_bt}.str.startswith(@{var}, na={na})"
        if op == "endswith":
            pat = str(val)
            mask = strobj.str.endswith(pat, na=na)
            var = _new_var(env, var_id, pat)
            return mask, f"{col_bt}.str.endswith(@{var}, na={na})"

    raise ValueError(f"Unsupported op: {op}")

def _compile_spec(df: pd.DataFrame, spec: Spec, env: Dict[str, Any], var_id: List[int]) -> Tuple[pd.Series, str]:
    """
    Accepts BOTH:
      - dict logic nodes: {'all': [ ... ]} | {'any': [ ... ]} | {'not': spec}
      - list logic nodes: ['all', ...]     | ['any', ...]     | ['not', spec]
      - leaf lists:       ['col','op', val?, opts?]
      - implicit-AND list: [node, node, ...]
    """
    # dict logic nodes
    if _is_logic_dict(spec):
        if "not" in spec:
            m, q = _compile_spec(df, spec["not"], env, var_id)
            return (~m), f"not ({q})"
        if "all" in spec:
            items = spec["all"]
            if not isinstance(items, (list, tuple)):
                items = [items]
            parts = [ _compile_spec(df, it, env, var_id) for it in items ]
            mask = pd.Series(True, index=df.index)
            qs = []
            for m, q in parts:
                mask &= m
                qs.append(f"({q})")
            return mask, "(" + " and ".join(qs) + ")" if qs else ""
        if "any" in spec:
            items = spec["any"]
            if not isinstance(items, (list, tuple)):
                items = [items]
            parts = [ _compile_spec(df, it, env, var_id) for it in items ]
            mask = pd.Series(False, index=df.index)
            qs = []
            for m, q in parts:
                mask |= m
                qs.append(f"({q})")
            return mask, "(" + " or ".join(qs) + ")" if qs else ""

    # list logic nodes
    if _is_logic_list(spec):
        tag = spec[0].lower()
        if tag == "not":
            if len(spec) != 2:
                raise ValueError("['not', spec] expects exactly one child.")
            m, q = _compile_spec(df, spec[1], env, var_id)
            return (~m), f"not ({q})"
        parts = [ _compile_spec(df, s, env, var_id) for s in spec[1:] ]
        if tag == "all":
            mask = pd.Series(True, index=df.index)
            qs = []
            for m, q in parts:
                mask &= m
                qs.append(f"({q})")
            return mask, "(" + " and ".join(qs) + ")" if qs else ""
        if tag == "any":
            mask = pd.Series(False, index=df.index)
            qs = []
            for m, q in parts:
                mask |= m
                qs.append(f"({q})")
            return mask, "(" + " or ".join(qs) + ")" if qs else ""
        raise ValueError(f"Unknown logic tag: {tag}")

    # leaf list
    if _is_leaf_list(spec):
        return _compile_leaf_list(df, spec, env, var_id)

    # implicit-AND list of nodes (each node can be a leaf list or a logic subtree)
    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ValueError("Empty spec list.")
        mask = pd.Series(True, index=df.index)
        qs = []
        for node in spec:
            m, q = _compile_spec(df, node, env, var_id)
            mask &= m
            qs.append(f"({q})")
        return mask, " and ".join(qs)

    raise ValueError("Invalid spec structure.")

def filter_df_with_spec(df: pd.DataFrame, spec: Spec, *, return_query: bool = True) -> Dict[str, Any]:
    env: Dict[str, Any] = {}
    var_id = [0]
    mask, q = _compile_spec(df, spec, env, var_id)
    out = {"df": df[mask], "mask": mask}
    if return_query:
        out["query"] = q
        out["env"] = env
    return out
