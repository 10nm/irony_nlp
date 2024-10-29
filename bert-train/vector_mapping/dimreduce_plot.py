from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import numpy as np

# Path to the .pt file
path_irony = 'IronyVectors.npy'
path_not_irony = 'NotIronyVectors.npy'

# # Function to process data in batches
# def process_in_batches(path, batch_size):
#     data = np.load(path, mmap_mode='r')  # Memory-map the file to avoid loading it all into memory at once
#     num_batches = data.shape[0] // batch_size + (1 if data.shape[0] % batch_size != 0 else 0)
#     for i in range(num_batches):
#         batch = data[i * batch_size:(i + 1) * batch_size]
#         yield batch

# # Parameters
# batch_size = 100  # Adjust this value based on your available memory

# # Process irony vectors in batches
# irony_batches = list(process_in_batches(path_irony, batch_size))
# not_irony_batches = list(process_in_batches(path_not_irony, batch_size))

# # Combine all batches into a single numpy array
# irony_vectors_np = np.concatenate(irony_batches, axis=0)
# not_irony_vectors_np = np.concatenate(not_irony_batches, axis=0)

# Load the vectors
irony_vectors_np = np.load(path_irony)
not_irony_vectors_np = np.load(path_not_irony)


# Show the shapes of the arrays
print(f'Irony vectors shape: {irony_vectors_np.shape}')
print(f'Not Irony vectors shape: {not_irony_vectors_np.shape}')

# Ensure the shapes match for concatenation
if irony_vectors_np.shape[1] != not_irony_vectors_np.shape[1]:
    raise ValueError(f"Shape mismatch: Irony vectors have shape {irony_vectors_np.shape}, "
                    f"but Not Irony vectors have shape {not_irony_vectors_np.shape}")

print('Combining vectors and creating labels...')
# Combine the vectors and create labels
all_vectors = np.concatenate((irony_vectors_np, not_irony_vectors_np), axis=0)
labels = np.array([0] * irony_vectors_np.shape[0] + [1] * not_irony_vectors_np.shape[0])

# Apply t-SNE to reduce dimensions to 2
tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(all_vectors)
print(reduced_vectors)

# Plot the results
plt.figure(figsize=(10, 7))
plt.scatter(reduced_vectors[labels == 0, 0], reduced_vectors[labels == 0, 1], label='Irony', alpha=0.5)
plt.scatter(reduced_vectors[labels == 1, 0], reduced_vectors[labels == 1, 1], label='Not Irony', alpha=0.5)
plt.legend()
plt.title('t-SNE of Irony and Not Irony Vectors')
plt.xticks([])  # Hide x-axis ticks
plt.yticks([])  # Hide y-axis ticks
plt.savefig('tsne_plot.png')
plt.show()