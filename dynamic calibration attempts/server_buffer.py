# let's figure out how the buffer works

import pyigtl
import matplotlib.pyplot as plt

client = pyigtl.OpenIGTLinkClient("127.0.0.1", 18944)

positions = []
number_of_points = 100

while True: 
    message = client.wait_for_message("SensorTipToFG", timeout=5)
    positions.append(message.matrix.T[3][:3])
    if len(positions) == number_of_points:
        break
    
plt.plot(range(number_of_points), [el[0] for el in positions])