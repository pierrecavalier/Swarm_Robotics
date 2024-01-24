import re


def tex_table(array, title, columns, rows, save_to, size="1.5"):
    assert len(array) == len(rows)
    assert len(array[0]) == len(columns)

    with open("tex_table.txt", "r") as f:
        notice = f.read()

    col_header = ("m{" + size + "cm} | ") * len(columns)

    tab = "& " + " & ".join(columns) + " \\\\\\ \n \\hline \n"
    for i in range(len(array)):
        tab += rows[i] + " & " + " & ".join(array[i]) + " \\\\\\ \n \\hline \n"

    notice = re.sub(r"TITLE", title, notice)
    notice = re.sub(r"HEAD", col_header, notice)
    notice = re.sub(r"CORE", tab, notice)

    with open(save_to, "w") as f:
        f.write(notice)

