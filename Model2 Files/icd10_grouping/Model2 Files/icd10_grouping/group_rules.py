def categorize(code: str) -> str:
    # Normalize and guard against blanks
    if code is None:
        return "Other"
    code = str(code).strip().upper()
    if not code:
        return "Other"

    ch = code[0]

    # helper: numeric part right after the first letter (for D/E splits)
    def num_after_letter(s: str):
        i, n = 1, ""
        while i < len(s) and s[i].isdigit():
            n += s[i]; i += 1
        return int(n) if n else None

    # Core buckets (chapters)
    if ch in {"A", "B"}:               # Certain infectious and parasitic diseases
        return "Infectious Diseases"
    if ch == "U":                       # Special purposes (e.g., U07.* = COVID-19)
        return "Infectious Diseases" if code.startswith("U07") else "Special Codes"

    if ch == "C":                       # Neoplasms
        return "Neoplasms"
    if ch == "D":
        n = num_after_letter(code)
        if n is not None and n <= 49:   # D00–D49 Neoplasms
            return "Neoplasms"
        return "Blood & Immune"         # D50–D89

    if ch == "E":
        n = num_after_letter(code)
        if n is not None and 10 <= n <= 14:   # E10–E14
            return "Diabetes"
        return "Endocrine/Metabolic (other)"

    if ch == "F":  return "Mental Health"
    if ch == "G":  return "Neurologic"
    if ch == "H":  return "Eye & Ear"
    if ch == "I":  return "Heart / Circulatory"
    if ch == "J":  return "Respiratory"
    if ch == "K":  return "Digestive"
    if ch == "L":  return "Skin & Subcutaneous"
    if ch == "M":  return "Musculoskeletal"
    if ch == "N":  return "Genitourinary"
    if ch == "O":  return "Pregnancy/Childbirth"
    if ch == "P":  return "Perinatal"
    if ch == "Q":  return "Congenital"
    if ch == "R":  return "Symptoms & Abnormal Findings"
    if ch == "S" or ch == "T": return "Injury / Physical harm"
    if ch in {"V","W","X","Y"}: return "Injury / External causes"
    if ch == "Z":  return "Encounters & Health Services"

    return "Other"
