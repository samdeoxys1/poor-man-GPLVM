'''
random pandas helper functions
'''

import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Union

Condition = Dict[str, Any]
Spec = Union[Condition, Dict[str, List["Spec"]]]  # recursive: {"all":[...]} | {"any":[...]} | {"not": {...}} | condition dict

def _btick(col: str) -> str:
    """Backtick a column if necessary for df.query."""
    return f"`{col}`" if re.search(r"\W", col) else col

def _ensure_listlike(x):
    if isinstance(x, (list, tuple, set, pd.Index, np.ndarray)):
        return list(x)
    return [x]

def _compile_condition(df: pd.DataFrame, cond: Condition, env: Dict[str, Any], var_id: List[int]) -> Tuple[pd.Series, str]:
    """
    Compile a single condition into a boolean mask and a query snippet.
    Supported ops: ==, !=, >, >=, <, <=, in, not in, between, isna, notna,
                   contains, startswith, endswith, regex
    """
    col = cond["col"]
    op  = cond["op"].lower()
    val = cond.get("value", None)
    col_bt = _btick(col)

    if col not in df.columns:
        raise KeyError(f"Column '{col}' not in DataFrame.")

    s = df[col]

    # Pick a fresh @var name for query env when needed
    def new_var(v):
        name = f"v{var_id[0]}"
        var_id[0] += 1
        env[name] = v
        return name

    # Numeric and categorical comparisons
    if op in ("==", "!=", ">", ">=", "<", "<="):
        var = new_var(val)
        mask = getattr(s, {"==":"eq","!=":"ne",">":"gt",">=":"ge","<":"lt","<=":"le"}[op])(env[var])
        q = f"{col_bt} {op} @{var}"
        return mask, q

    # Membership
    if op in ("in", "not in"):
        vals = _ensure_listlike(val)
        var = new_var(vals)
        mask = s.isin(env[var])
        if op == "not in":
            mask = ~mask
            q = f"{col_bt} not in @{var}"
        else:
            q = f"{col_bt} in @{var}"
        return mask, q

    # Between
    if op == "between":
        if not (isinstance(val, (list, tuple)) and len(val) == 2):
            raise ValueError("between expects value=[low, high].")
        low, high = val
        inclusive = cond.get("inclusive", "both")  # 'both'|'neither'|'left'|'right'
        mask = s.between(low, high, inclusive=inclusive)
        varL, varH = new_var(low), new_var(high)
        # df.query has no native 'between', so expand:
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
        mask = s.isna()
        q = f"{col_bt}.isnull()"
        return mask, q
    if op in ("notna", "notnull"):
        mask = s.notna()
        q = f"{col_bt}.notnull()"
        return mask, q

    # String ops: contains/startswith/endswith/regex
    # For query, we emit a .str.* expression (requires engine='python' in df.query).
    if op in ("contains", "startswith", "endswith", "regex"):
        case = bool(cond.get("case", True))
        na = cond.get("na", False)
        if op == "contains":
            pat = str(val)
            mask = s.astype("string").str.contains(pat, case=case, na=na, regex=True)
            var = new_var(pat)
            q = f"{col_bt}.str.contains(@{var}, case={case}, na={na}, regex=True)"
            return mask, q
        if op == "startswith":
            pat = str(val)
            mask = s.astype("string").str.startswith(pat, na=na)
            var = new_var(pat)
            q = f"{col_bt}.str.startswith(@{var}, na={na})"
            return mask, q
        if op == "endswith":
            pat = str(val)
            mask = s.astype("string").str.endswith(pat, na=na)
            var = new_var(pat)
            q = f"{col_bt}.str.endswith(@{var}, na={na})"
            return mask, q
        if op == "regex":
            pat = str(val)
            mask = s.astype("string").str.contains(pat, regex=True, case=case, na=na)
            var = new_var(pat)
            q = f"{col_bt}.str.contains(@{var}, case={case}, na={na}, regex=True)"
            return mask, q

    raise ValueError(f"Unsupported op: {op}")

def _compile_spec(df: pd.DataFrame, spec: Spec, env: Dict[str, Any], var_id: List[int]) -> Tuple[pd.Series, str]:
    """
    Recursively compile a spec into (mask, query_string).
    Spec forms:
      - {"all": [spec, spec, ...]}
      - {"any": [spec, spec, ...]}
      - {"not": spec}
      - {"col": ..., "op": ..., "value": ...}  (a single condition)
    """
    if isinstance(spec, dict) and "all" in spec:
        parts = [ _compile_spec(df, s, env, var_id) for s in spec["all"] ]
        masks, qs = zip(*parts) if parts else ([], [])
        mask = pd.Series(True, index=df.index) if not parts else parts[0][0]
        for m,_ in parts[1:]:
            mask = mask & m
        q = "(" + " and ".join([f"({qq})" for qq in qs]) + ")" if qs else ""
        return mask, q

    if isinstance(spec, dict) and "any" in spec:
        parts = [ _compile_spec(df, s, env, var_id) for s in spec["any"] ]
        masks, qs = zip(*parts) if parts else ([], [])
        mask = pd.Series(False, index=df.index) if not parts else parts[0][0]
        for m,_ in parts[1:]:
            mask = mask | m
        q = "(" + " or ".join([f"({qq})" for qq in qs]) + ")" if qs else ""
        return mask, q

    if isinstance(spec, dict) and "not" in spec:
        m, q = _compile_spec(df, spec["not"], env, var_id)
        return (~m), f"not ({q})"

    # leaf condition
    return _compile_condition(df, spec, env, var_id)

def filter_df_with_spec(
    df: pd.DataFrame,
    spec: Spec,
    *,
    return_query: bool = True
) -> Dict[str, Any]:
    """
    Apply a nested filter spec to `df`.

    Returns a dict with:
      - df:        filtered DataFrame
      - mask:      boolean mask used
      - query:     generated df.query string (works best with engine='python' if string ops present)
      - env:       local_dict for df.query
    """
    env: Dict[str, Any] = {}
    var_id = [0]
    mask, q = _compile_spec(df, spec, env, var_id)
    out = {
        "df": df[mask],
        "mask": mask,
    }
    if return_query:
        out["query"] = q
        out["env"] = env
    return out

# Example call:
# spec = {
#   "all": [
#     {"col": "age", "op": ">=", "value": 18},
#     {"col": "status", "op": "in", "value": ["active", "pending"]},
#     {"col": "email", "op": "contains", "value": "@nyu.edu", "case": False, "na": False},
#     {"col": "last_login", "op": "notna"},
#     {"col": "score", "op": "between", "value": [0.7, 0.95], "inclusive": "right"},
#   ],
#   "any": [
#     {"col": "role", "op": "==", "value": "admin"},
#     {"col": "dept", "op": "startswith", "value": "neuro"}
#   ]
# }
# 
# out = filter_df_with_spec(df, spec)
# df_filtered = out["df"]
# mask = out["mask"]
# qstr = out["query"]
# env = out["env"]
# 
# # If you really want to use df.query (string ops require engine='python'):
# df_q = df.query(qstr, local_dict=env, engine="python")
# assert df_q.equals(df_filtered)
