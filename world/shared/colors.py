rgb_colors = {
    'white': [1, 1, 1],
    'red': [1, 0, 0],
    'green': [0, 1, 0],
    'blue': [0, 0, 1],
    'yellow': [1, 1, 0],
    'magenta': [1, 0, 1],
    'cyan': [0, 1, 1]
}

colors_names = list(rgb_colors.keys())


def color_map(c, s=0.8):
    if c in rgb_colors:
        color = rgb_colors[c]
        return f'{color[0] * s} {color[1] * s} {color[2] * s}'
    return c
