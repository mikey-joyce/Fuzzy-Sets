import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plot

b  = np.arange(0, 5, 1)
b_lo = fuzz.trimf(b, [1, 2, 2])
a = fuzz.trimf(b, [2, 3, 4])

fig, (b_plot) = plot.subplots(nrows=1, figsize=(7, 10))

#below is the plot for the first aggregation
b_plot.plot(b, b_lo, linewidth=0.5, linestyle='--', )
b_plot.plot(b, a, linewidth=0.5, linestyle='--', )
b_plot.set_title('plot')


fig.tight_layout()
plot.show()