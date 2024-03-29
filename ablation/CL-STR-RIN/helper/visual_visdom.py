import numpy as np
from visdom import Visdom

_WINDOW_CASH = {}

def _vis(env='main'):
    return Visdom(env=env)


def visualize_scalars(scalars, names, title, iteration, env='main', ylable=None):
    '''Continually update line-plot with numbers arriving in [scalars].'''

    assert len(scalars) == len(names)

    # Convert scalar tensors to numpy arrays.
    scalars, names = list(scalars), list(names)
    scalars = [s.cpu().numpy() if (hasattr(s, 'cpu') and hasattr(s.cpu(), 'numpy')) else np.array([s]) for s in scalars]
    num = len(scalars)
    X = np.column_stack(np.array([iteration]*num)) if (num>1) else np.array([iteration]*num)
    Y = np.column_stack(scalars) if (num>1) else scalars[0]

    # title if (ylable is None) else ylable
    # Plotting options
    options = dict(
        fillares=False, legend=names, width=440, height=400,
        xlabel='Iterations', ylable=title if (ylable is None) else ylable, title=title,
        marginleft=70, marginright=30, marginbottom=80, margintop=30,
    )

    # Updata plot (or start new one if not yet present)
    if title in _WINDOW_CASH:
        #_vis(env).updateTrace(X=X, Y=Y, win=_WINDOW_CASH[title], opts=options)          # for older versions of visdom
        _vis(env).line(X=X, Y=Y, win=_WINDOW_CASH[title], opts=options, update='append') # for newer versions of visdom
    else:
        _WINDOW_CASH[title] = _vis(env).line(X=X, Y=Y, opts=options)
