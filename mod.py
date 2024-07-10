from keras.models import Sequential

# Example model (use your actual model)
model = Sequential()
# Add layers to the model
# model.add(...)

# Save the model
model.save('model.keras')
print("Model saved to model.keras")