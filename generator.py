from models import StrictCage

def get_restrict_rules():
    rules = []
    blocks = [StrictCage(45) for _ in range(9)]
    for i in range(9):
        c = StrictCage(45)
        d = StrictCage(45)
        for j in range(9):
            c.append(i, j)
            d.append(j, i)
            blocks[i // 3 * 3 + j // 3].append(i, j)
        rules.append(c)
        rules.append(d)
    return rules + blocks