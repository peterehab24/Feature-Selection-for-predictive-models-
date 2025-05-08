# Feature-Selection-for-predictive-models-
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
random_state = 42

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
num_features = X.shape[1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split
(X, y, test_size=0.2, random_state=random_state)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================================
# 1. Feature Selection BEFORE Optimization (Random Forest Importance)
# ==============================================
model = RandomForestClassifier(n_estimators=2, random_state=random_state)
model.fit(X_train, y_train)
importances = model.feature_importances_

threshold = np.mean(importances)
selected_features = np.where(importances >= threshold)[0]

X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

model_selected = RandomForestClassifier(n_estimators=2, 
                                random_state=random_state)
model_selected.fit(X_train_selected, y_train)
y_pred_selected = model_selected.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test, y_pred_selected)

print(f"\nAccuracy before optimization: {accuracy_selected:.4f}")
print("Selected features before optimization:")
for idx in selected_features:
    print(f" - {feature_names[idx]}")

# ==============================================
# 2. Feature Selection AFTER Optimization using PSO
# ==============================================

# Define the fitness function
def fitness_fun(x):
    result = sum(x**2)  # Sum of squared values
    return result

# PSO algorithm
def pso(cost_func, dim, num_particles, max_iter=10, w=0.5, 
        c1=2, c2=2, random_state=None):
    np.random.seed(random_state)
    particles = np.random.randint(0, 2, (num_particles, dim)).astype(float)
    velocities = np.random.rand(num_particles, dim)

    best_positions = particles.copy()
    best_fitness = np.array([cost_func(p) for p in particles])
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)

    for i in range(max_iter):
        r1 = np.random.uniform(0, 1, (num_particles, dim))
        r2 = np.random.uniform(0, 1, (num_particles, dim))
        velocities = (
            w * velocities 
            + c1 * r1 * (best_positions - particles) 
            + c2 * r2 * (swarm_best_position - particles)
        )

        particles += velocities
        particles = np.clip(particles, 0, 1)

        fitness_values = np.array([cost_func(p) for p in particles])
        improved_indices = np.where(fitness_values < best_fitness)

        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]

        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)

    return swarm_best_position, swarm_best_fitness

# PSO Configuration
dim = 30
num_particles = 10
max_iter = 100

# Run PSO
best_position, best_fitness = pso(fitness_fun, dim=dim, 
num_particles=num_particles, max_iter=max_iter, random_state=random_state)

# Ensure best_position contains only 0 and 1
best_position = np.round(best_position).astype(int)
best_position = best_position[:X_train.shape[1]]

# If no features are selected, choose one randomly
if np.sum(best_position) == 0:
    print("No features selected! Selecting one feature randomly.")
    best_position[np.random.randint(0, dim)] = 1

# Filter selected features
X_train_pso = X_train[:, best_position == 1]
X_test_pso = X_test[:, best_position == 1]

if X_train_pso.shape[1] == 0:
    raise ValueError
    ("No features selected! Make sure PSO selected at least one feature.")

# Train model using PSO-selected features
model_pso = RandomForestClassifier(n_estimators=10, random_state=random_state)
model_pso.fit(X_train_pso, y_train)
y_pred_pso = model_pso.predict(X_test_pso)
accuracy_pso = accuracy_score(y_test, y_pred_pso)

print(f"\nAccuracy after PSO optimization: {accuracy_pso:.4f}")
print("Selected features after optimization:")
for idx in np.where(best_position == 1)[0]:
    print(f" - {feature_names[idx]}")

