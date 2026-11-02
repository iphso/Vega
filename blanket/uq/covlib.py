"""Parse NJOY ERRORR output tapes (ASCII, irelco=1) into relative covariance blocks."""
import numpy as np


def _f(s):
    s = s.strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:  # ENDF '1.234567-1' / '1.234567+1'
        for i in range(len(s) - 1, 0, -1):
            if s[i] in "+-" and s[i - 1] not in "eE":
                return float(s[:i] + "e" + s[i:])
        raise


def _records(path):
    for line in open(path):
        if len(line) < 75:
            continue
        mat, mf, mt = int(line[66:70] or 0), int(line[70:72] or 0), int(line[72:75] or 0)
        yield line, mat, mf, mt


def read_errorr(path):
    """Returns (bounds[ng+1] ascending eV, xs{mt: [ng]}, cov{(mt, mt1): [ng, ng]} relative)."""
    lines = list(_records(path))
    bounds, xs, cov = None, {}, {}
    i = 0
    while i < len(lines):
        line, mat, mf, mt = lines[i]
        if mf == 1 and mt == 451 and bounds is None:
            nw = int(lines[i + 1][0][44:55])
            vals = []
            j = i + 2
            while len(vals) < nw:
                l = lines[j][0]; vals += [_f(l[k:k + 11]) for k in range(0, 66, 11) if len(vals) < nw and True]
                j += 1
            bounds = np.array(vals[:nw]); i = j; continue
        if mf == 3 and mt > 0:
            ng = int(line[44:55]); vals = []; j = i + 1
            while len(vals) < ng:
                l = lines[j][0]; vals += [_f(l[k:k + 11]) for k in range(0, 66, 11)]; j += 1
            xs[mt] = np.array(vals[:ng]); i = j
            while lines[i][3] == mt and lines[i][2] == 3:
                i += 1
            continue
        if mf == 33 and mt > 0:
            nl = int(line[55:66]); j = i + 1
            for _ in range(nl):
                c = lines[j][0]; mt1 = int(c[33:44]); ng = int(c[55:66]); j += 1
                M = np.zeros((ng, ng))
                while True:
                    c = lines[j][0]
                    ng2, ig2lo, nw, ig = int(c[22:33]), int(c[33:44]), int(c[44:55]), int(c[55:66])
                    j += 1; vals = []
                    while len(vals) < nw:
                        l = lines[j][0]; vals += [_f(l[k:k + 11]) for k in range(0, 66, 11)]; j += 1
                    M[ig - 1, ig2lo - 1: ig2lo - 1 + ng2] = vals[:ng2]
                    if ig == ng:
                        break
                cov[(mt, mt1)] = M
            i = j
            while i < len(lines) and lines[i][3] == mt and lines[i][2] == 33:
                i += 1
            continue
        i += 1
    return bounds, xs, cov


def lumped_map(endf_path):
    """{constituent MT: lumped MT} from MF33 section headers (L2 = MTL)."""
    out = {}
    for line, mat, mf, mt in _records(endf_path):
        pass
    seen = set()
    for line, mat, mf, mt in _records(endf_path):
        if mf == 33 and mt > 0 and mt not in seen:
            seen.add(mt)
            mtl = int(line[33:44])
            if mtl:
                out[mt] = mtl
    return out
