
def print_values(V, g):
    for i in range(g.width):
        print('-'*28)
        row = ''
        for j in range(g.height):
            v = V.get((i, j), 0)
            if v >= 0:
                row += ' {:0.2f} |'.format(v)
            else:
                row += '{:0.2f} |'.format(v)
        print(row)
        
def print_policy(P, g):
    for i in range(g.width):
        print('-'*28)
        row = ''
        for j in range(g.height):
            a = P.get((i, j), ' ')
            row += '  {}   |'.format(a)
        print(row)