from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch

from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Make the random starting weights repeatable.
torch.manual_seed(42)

# Use one CPU thread for this small experiment.
torch.set_num_threads(1)

# Find the project folder.
project_folder = Path(__file__).resolve().parent

# Keep neural-network outputs separate from our first experiment.
output_folder = project_folder / "results_neural"
output_folder.mkdir(exist_ok=True)

# Read the original features and split boundaries.
settings = json.loads(
    (
        project_folder / "results" / "training_settings.json"
    ).read_text(encoding="utf-8")
)

# Read the prepared market dataset.
dataset = pd.read_csv(
    project_folder / "data" / "model_dataset.csv",
    index_col="Date",
    parse_dates=True,
).sort_index()

# Reuse exactly the same five input clues.
features = settings["features"]

# Recover the original training and validation periods.
train_end = settings["train_end"]
validation_end = settings["validation_end"]

train_data = dataset.iloc[:train_end - 1].copy()
validation_data = dataset.iloc[
    train_end:validation_end - 1
].copy()

# Learn how to scale clues using TRAINING rows only.
scaler = StandardScaler()

X_train = scaler.fit_transform(train_data[features])

# Apply those same scaling rules to validation rows.
X_validation = scaler.transform(validation_data[features])

# Convert the training clues into PyTorch's number format.
train_inputs = torch.tensor(
    X_train,
    dtype=torch.float32,
)

# Convert training answers into a single-column tensor.
train_answers = torch.tensor(
    train_data["target"].to_numpy(),
    dtype=torch.float32,
).reshape(-1, 1)

# Convert validation clues into a tensor.
validation_inputs = torch.tensor(
    X_validation,
    dtype=torch.float32,
)

# Build a small network: five inputs, two hidden layers, one output.
model = nn.Sequential(
    nn.Linear(len(features), 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

# Measure binary prediction mistakes using raw output scores.
loss_function = nn.BCEWithLogitsLoss()

# Choose how the network adjusts its weights.
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

# Fix the number of learning rounds before evaluating validation.
epochs = 100

# Keep a record of training errors.
loss_history = []

# Enable learning mode.
model.train()

# Review the training examples 100 times.
for epoch in range(1, epochs + 1):

    # Clear the previous round's weight-adjustment calculations.
    optimizer.zero_grad()

    # Calculate raw prediction scores from the training clues.
    logits = model(train_inputs)

    # Compare those scores with the correct answers.
    loss = loss_function(logits, train_answers)

    # Calculate how the weights should change.
    loss.backward()

    # Apply one small adjustment to the weights.
    optimizer.step()

    # Save this round's training error.
    loss_history.append({
        "epoch": epoch,
        "training_loss": loss.item(),
    })

    # Display progress every 25 rounds.
    if epoch % 25 == 0:
        print(
            f"Round {epoch}/{epochs} "
            f"| Training loss: {loss.item():.4f}"
        )

# Switch to prediction mode.
model.eval()

# Predict without calculating learning adjustments.
with torch.no_grad():

    # Produce raw scores for the validation examples.
    validation_logits = model(validation_inputs)

    # Convert raw scores into estimated up-day probabilities.
    probability_up = torch.sigmoid(
        validation_logits
    ).numpy().ravel()

# Turn estimated probabilities into up/down answers.
predictions = (probability_up >= 0.50).astype(int)

# Read the correct validation answers.
answers = validation_data["target"]

# Calculate the neural network's practice-exam scores.
neural_scores = {
    "model": "pytorch_neural_network",
    "accuracy": accuracy_score(answers, predictions),
    "balanced_accuracy": balanced_accuracy_score(
        answers,
        predictions,
    ),
}

# Load the original models' validation scores.
previous_scores = pd.read_csv(
    project_folder / "results" / "validation_scores.csv"
)

# Add the neural network as another competitor.
comparison = pd.concat(
    [previous_scores, pd.DataFrame([neural_scores])],
    ignore_index=True,
)

# Save the comparison without changing the original selected model.
comparison.to_csv(
    output_folder / "validation_comparison.csv",
    index=False,
)

# Save the neural network's individual validation predictions.
validation_output = validation_data.copy()
validation_output["probability_up"] = probability_up
validation_output["prediction"] = predictions

validation_output.to_csv(
    output_folder / "validation_predictions.csv"
)

# Save training progress for inspection.
pd.DataFrame(loss_history).to_csv(
    output_folder / "training_loss.csv",
    index=False,
)

# Save the learned weights.
torch.save(
    model.state_dict(),
    output_folder / "neural_weights.pt",
)

# Save the scaling rules needed for future predictions.
joblib.dump(
    scaler,
    output_folder / "scaler.joblib",
)

# Record the architecture and experiment settings.
neural_settings = {
    "features": features,
    "hidden_layers": [16, 8],
    "epochs": epochs,
    "learning_rate": 0.001,
    "random_seed": 42,
    "train_end": train_end,
    "validation_end": validation_end,
    "torch_version": str(torch.__version__),
    "status": "Exploratory extension; original test previously viewed",
}

(output_folder / "neural_settings.json").write_text(
    json.dumps(neural_settings, indent=2),
    encoding="utf-8",
)

# Show the measured comparison.
print("\nVALIDATION COMPARISON:")
print(comparison.to_string(index=False))

print("\nSaved to:", output_folder)
print("Original experiment files were not changed.")
print("No neural-network test evaluation was performed.")