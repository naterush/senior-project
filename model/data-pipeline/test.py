import tarfile

with tarfile.open('./downloaded_sat_data/LE07_L1TP_016036_19990719_20161003_01_T1.tar.gz', "r:gz") as mytar:
    for m in mytar.getnames():
        if not m.endswith(".TIF"):
            mytar.extract(m, path="downloaded_sat_data/LE07_L1TP_016036_19990719_20161003_01_T1")