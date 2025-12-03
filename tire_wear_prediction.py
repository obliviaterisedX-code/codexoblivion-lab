# Challenge: Tire Grip Predictor

# Steps
# 1. Import scikit-learn and pandas.
# 2. Create a dataset with lap, track temp, and grip.
# 3. Train a LinearRegression model.
# 4. Predict grip at Lap 20, Track Temp 35°C.
# 5. Add a pit warning if grip < threshold.

#Try it yourself before peeking at the full code below.
#---------------------------------------------------------


from sklearn.linear_model import LinearRegression
import pandas as pd

# Step 1: Create dataset [Lap, TrackTemp, Grip%]
data = [
    [1, 25, 100],
    [5, 27, 95],
    [10, 30, 85],
    [15, 32, 70],
    [20, 35, 60]
]

# Step 2: Convert to DataFrame for clarity
df = pd.DataFrame(data, columns=["Lap", "TrackTemp", "Grip"])

# Step 3: Define features (X) and target (y)
X = df[["Lap", "TrackTemp"]]
y = df["Grip"]

# Step 4: Train regression model
model = LinearRegression()
model.fit(X, y)

# Step 5: Predict grip at Lap 20, Track Temp 35°C
predicted = model.predict([[20, 35]])
print(f"Lap 20 grip: {predicted[0]:.1f}%")

# Step 6: Pit warning if grip below threshold
threshold = 70
if predicted[0] < threshold:
    print("Pit in 3 laps!")
else:
    print("Tires are still good")


# --------------------------------
# Next Steps for Challenge:
# a) Replace sample data with FastF1 lap + weather data
# b) Add features: compound, driver style, surface abrasion
# c) Try non-linear models (Polynomial, Random Forest, etc.)


# --------------------------------
# Example: Tire Grip Curve (Polynomial Regression)
# --------------------------------

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Dataset
data = [
    [1, 25, 100],
    [5, 27, 95],
    [10, 30, 85],
    [15, 32, 70],
    [20, 35, 60]
]
df = pd.DataFrame(data, columns=["Lap", "TrackTemp", "Grip"])

X = df[["Lap", "TrackTemp"]]
y = df["Grip"]

# Step 2: Polynomial transform
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Step 3: Train model
model = LinearRegression()
model.fit(X_poly, y)

# Step 4: Predict grip at Lap 25, Track Temp 38°C
pred = model.predict(poly.transform([[25, 38]]))
print(f"Lap 25 grip: {pred[0]:.1f}%")

# Step 5: Pit warning (compound-aware)
compound = "Soft"
thresholds = {"Soft": 75, "Medium": 65, "Hard": 55}
if pred[0] < thresholds[compound]:
    print(f"Pit soon! {compound} tire below threshold")

# Step 6: Plot grip curve
laps = np.arange(1, 30)
temps = np.linspace(25, 38, 29)
X_curve = poly.transform(np.column_stack((laps, temps)))
y_curve = model.predict(X_curve)

plt.plot(laps, y_curve, label="Grip Curve")
plt.xlabel("Lap")
plt.ylabel("Grip %")
plt.title("Tire Grip Degradation Curve")
plt.legend()
plt.show()


# --------------------------------
# Example: FastF1 Grip Predictor
# --------------------------------

import fastf1
import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 1: Enable FastF1 cache (so data loads faster next time)
fastf1.Cache.enable_cache('cache')  

# Step 2: Load a race session (example: 2025 British GP, Race)
session = fastf1.get_session(2025, 'Silverstone', 'R')
session.load()

# Step 3: Pick a driver (example: VER for Verstappen)
laps = session.laps.pick_driver('VER')

# Step 4: Build dataset
# Use lap number, track temp, and lap time delta as grip proxy
data = []
best_lap = laps['LapTime'].min()

for _, lap in laps.iterrows():
    lap_num = lap['LapNumber']
    track_temp = lap['TrackTemp']
    # Grip proxy: slower lap time = less grip
    grip = 100 - ((lap['LapTime'] - best_lap).total_seconds() * 5)  
    data.append([lap_num, track_temp, grip])

df = pd.DataFrame(data, columns=["Lap", "TrackTemp", "Grip"])

# Step 5: Train regression model
X = df[["Lap", "TrackTemp"]]
y = df["Grip"]

model = LinearRegression()
model.fit(X, y)

# Step 6: Predict grip at Lap 20, Track Temp 35°C
predicted = model.predict([[20, 35]])
print(f"Lap 20 grip: {predicted[0]:.1f}%")

# Step 7: Pit warning if grip below threshold
threshold = 70
if predicted[0] < threshold:
    print("Pit in 3 laps!")
else:
    print("Tires are still good")



