import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Union

Spec = Union[Dict[str, Any], List[Any]]  # nested {"all": [...]}, {"any": [...]}, {"not": {...}}, or {col:(op,val[,opts]), ...}

def _btick(col: str) -> str:
    return f"`{col}`" if re.search(r"\W", col) else col

def _ensure_listlike(x):
    if isinstance(x, (list, tuple, set, pd.Index, np.ndarray)):
        return list(x)
    return [x]

def _new_var(env: Dict[str, Any], var_id: List[int], v: Any) -> str:
    name = f"v{var_id[0]}"
    var_id[0] += 1
    env[name] = v
    return name

def _compile_condition(df: pd.DataFrame, col: str, op_tuple: Tuple, env: Dict[str, Any], var_id: List[int]):
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not in DataFrame.")
    if not isinstance(op_tuple, (list, tuple)) or len(op_tuple) < 1:
        raise ValueError(f"Condition for '{col}' must be a tuple like (op, value[, options]).")

    op = str(op_tuple[0]).lower()
    val = None if len(op_tuple) < 2 else op_tuple[1]
    opts = {} if len(op_tuple) < 3 or not isinstance(op_tuple[2], dict) else op_tuple[2]
    s = df[col]
    col_bt = _btick(col)

    # Comparators
    if op in ("==", "!=", ">", ">=", "<", "<="):
        var = _new_var(env, var_id, val)
        mask = getattr(s, {"==":"eq","!=":"ne",">":"gt",">=":"ge","<":"lt","<=":"le"}[op])(env[var])
        return mask, f"{col_bt} {op} @{var}"

    # Membership
    if op in ("in", "not in"):
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
        varL, varH = _new_var(env, var_id, low), _new_var(env, var_id, high)
        if inclusive in ("both", True):
            q = f"(@{varL} <= {col_bt}) and ({col_bt} <= @{varH})"
        elif inclusive in ("neither", False):
            q = f"(@{varL} < {col_bt}) and ({col_bt} < @{varH})"
        elif inclusive == "left":
            q = f"(@{varL} <= {col_bt}) and ({col_bt} < @{varH})"
        elif inclusive == "right":
            q = f"(@{varL} < {col_bt}) and ({col_bt} <= @{varH})"
        else:
            q = f"(@{varL} <= {col_bt}) and ({col_bt} <= @{varH})"
        return mask, q

    # Null checks
    if op in ("isna", "isnull"):
        return s.isna(), f"{col_bt}.isnull()"
    if op in ("notna", "notnull"):
        return s.notna(), f"{col_bt}.notnull()"

    # String ops
    if op in ("contains", "startswith", "endswith", "regex"):
        case = bool(opts.get("case", True))
        na = opts.get("na", False)
        strobj = s.astype("string")
        if op == "contains":
            pat = str(val)
            mask = strobj.str.contains(pat, case=case, na=na, regex=opts.get("regex", True))
            var = _new_var(env, var_id, pat)
            q = f"{col_bt}.str.contains(@{var}, case={case}, na={na}, regex={opts.get('regex', True)})"
            return mask, q
        if op == "regex":
            pat = str(val)
            mask = strobj.str.contains(pat, case=case, na=na, regex=True)
            var = _new_var(env, var_id, pat)
            q = f"{col_bt}.str.contains(@{var}, case={case}, na={na}, regex=True)"
            return mask, q
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

def _is_logic_spec(spec: Dict[str, Any]) -> bool:
    return any(k in spec for k in ("all", "any", "not"))

def _compile_spec(df: pd.DataFrame, spec: Spec, env: Dict[str, Any], var_id: List[int]):
    """
    Returns (mask, query_string). Supports:
      - {"all":[spec, ...]}  -> AND
      - {"any":[spec, ...]}  -> OR
      - {"not": spec}        -> NOT
      - {col:(op,val[,opts]), col2:(op,val[,opts]), ...} -> implicit AND over columns
    """
    # Logical forms
    if isinstance(spec, dict) and "all" in spec:
        parts = [ _compile_spec(df, s, env, var_id) for s in spec["all"] ]
        mask = pd.Series(True, index=df.index)
        qs = []
        for m, q in parts:
            mask &= m
            qs.append(f"({q})")
        return mask, "(" + " and ".join(qs) + ")" if qs else ""
    if isinstance(spec, dict) and "any" in spec:
        parts = [ _compile_spec(df, s, env, var_id) for s in spec["any"] ]
        mask = pd.Series(False, index=df.index)
        qs = []
        for m, q in parts:
            mask |= m
            qs.append(f"({q})")
        return mask, "(" + " or ".join(qs) + ")" if qs else ""
    if isinstance(spec, dict) and "not" in spec:
        m, q = _compile_spec(df, spec["not"], env, var_id)
        return (~m), f"not ({q})"

    # Leaf or implicit-AND dict: {col:(op,val[,opts]), ...}
    if isinstance(spec, dict):
        mask = pd.Series(True, index=df.index)
        qs = []
        for col, op_tuple in spec.items():
            m, q = _compile_condition(df, col, op_tuple, env, var_id)
            mask &= m
            qs.append(f"({q})")
        return mask, " and ".join(qs)

    raise ValueError("Invalid spec structure.")

def filter_df_with_spec(df: pd.DataFrame, spec: Spec, *, return_query: bool = True) -> Dict[str, Any]:
    """
    Apply simplified spec to `df`.
    Returns:
      - df:    filtered DataFrame
      - mask:  boolean mask used
      - query: generated df.query string (use engine='python' if string ops present)
      - env:   local_dict for df.query
    """
    env: Dict[str, Any] = {}
    var_id = [0]
    mask, q = _compile_spec(df, spec, env, var_id)
    out = {"df": df[mask], "mask": mask}
    if return_query:
        out["query"] = q
        out["env"] = env
    return out
