import numpy as np
from datetime import datetime
import torch
import torch.nn as nn

spans = "1,2,3"
if len(spans) > 0:
    span_list = [int(i) for i in spans.split(",")]
else:
    span_list = []



import numpy as np
import matplotlib
import matplotlib.pyplot as plt
harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


fig, ax = plt.subplots()
im = ax.imshow(harvest)
fig.show()


a = torch.randn([3, 5], dtype=torch.float32)
b = a.unsqueeze(1)
res = b.expand(-1, 2, -1)
print(res.size())


l, d, t, k = 10, 200, 50, 20
U = torch.randn([l, d, t], dtype=torch.float32)
V = torch.randn([d, k, t], dtype=torch.float32)
A = torch.randn([d, d], dtype=torch.float32)
F_sim = torch.einsum('ldt,dd,dkt->lkt', [U, A, V])

now = datetime.now()  # current date and time

date_time = now.strftime("%Y%m%d_%H%M%S")


avgDists = np.array([[1, 8, 6, 9, 4], [6, 5, 4, 3, 2]])
ids = avgDists.argsort(axis=1)
pass