import jax
import jax.numpy as jnp
import equinox as eqx


class DSDM(eqx.Module):
    # Memory structure: addresses (A) and contents (C) as per the paper's core components
    addresses: jnp.ndarray  # High-dimensional points representing memory locations
    contents: jnp.ndarray   # Associated data (e.g., labels or continuous values)
    rt: float               # Recursive Temperature for dynamic thresholding
    beta: float             # Temperature parameter for softmin weighting
    lambda_rt: float        # Learning rate for RT update
    lambda_update: float    # Learning rate for memory updates
    capacity: int           # Maximum number of memory units

    def __init__(self, input_dim, output_dim, capacity, beta=1.0, lambda_rt=0.9, lambda_update=0.01):
        """
        Initialize DSDM with empty memory and hyperparameters.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output (e.g., number of classes)
            capacity: Maximum number of memory units
            beta: Controls the softness of weight distribution
            lambda_rt: Controls RT adaptation speed
            lambda_update: Controls memory update speed
        """
        # Initial empty memory as per Step 1 in thinking trace
        self.addresses = jnp.empty((0, input_dim))
        self.contents = jnp.empty((0, output_dim))
        self.rt = 0.0
        self.beta = beta
        self.lambda_rt = lambda_rt
        self.lambda_update = lambda_update
        self.capacity = capacity

    def find_bmu(self, x):
        """
        Find the Best Matching Unit (BMU) based on Euclidean distance.
        Logic from Section 2.1: Compute distances to all addresses and find the closest.
        """
        distances = jnp.linalg.norm(self.addresses - x, axis=1)
        bmu_idx = jnp.argmin(distances)
        return bmu_idx, distances[bmu_idx]

    def train_step(self, x, y):
        """
        Train DSDM with a single sample (x, y).
        Logic from Section 2.2: Add new unit if distance > RT, else update existing units.
        """
        # Handle first sample (initial memory population)
        if self.addresses.shape[0] == 0:
            new_addresses = jnp.vstack([self.addresses, x])
            new_contents = jnp.vstack([self.contents, y])
            return eqx.tree_at(lambda m: (m.addresses, m.contents), self, (new_addresses, new_contents))

        # Find BMU and distance (Step 2.1 in thinking trace)
        bmu_idx, d_bmu = self.find_bmu(x)

        if d_bmu > self.rt:
            # Add new memory unit if distance exceeds RT (dynamic growth)
            new_addresses = jnp.vstack([self.addresses, x])
            new_contents = jnp.vstack([self.contents, y])
            new_self = eqx.tree_at(lambda m: (m.addresses, m.contents), self, (new_addresses, new_contents))
        else:
            # Update existing memory units (weighted update based on proximity)
            distances = jnp.linalg.norm(self.addresses - x, axis=1)
            weights = jax.nn.softmax(-distances / self.beta)

            updated_addresses = self.addresses + self.lambda_update * weights[:, None] * (x - self.addresses)
            updated_contents = self.contents + self.lambda_update * weights[:, None] * (y - self.contents)

            new_self = eqx.tree_at(lambda m: (m.addresses, m.contents), self, (updated_addresses, updated_contents))

        # Update Recursive Temperature (RT) adaptively
        updated_rt = self.lambda_rt * self.rt + (1 - self.lambda_rt) * d_bmu
        new_self = eqx.tree_at(lambda m: m.rt, new_self, updated_rt)

        # Prune if memory exceeds capacity (Step 2.3 in thinking trace)
        if new_self.addresses.shape[0] > self.capacity:
            new_self = new_self.prune()

        return new_self

    def prune(self):
        """
        Simple pruning by removing the memory unit with the smallest average distance to others.
        Logic from Section 2.3: Ideally uses LOF, but simplified here for demonstration.
        """
        # Compute average distance to other points for each address
        distances = jax.vmap(lambda a: jnp.mean(jnp.linalg.norm(self.addresses - a, axis=1)))(self.addresses)
        idx_to_remove = jnp.argmin(distances)  # Remove the most "dense" point
        new_addresses = jnp.delete(self.addresses, idx_to_remove, axis=0)
        new_contents = jnp.delete(self.contents, idx_to_remove, axis=0)
        return eqx.tree_at(lambda m: (m.addresses, m.contents), self, (new_addresses, new_contents))

    def predict(self, x):
        """
        Predict output for query x using weighted sum of contents.
        Logic from Section 3: Inference phase computes a softmin-weighted average.
        """
        distances = jnp.linalg.norm(self.addresses - x, axis=1)
        weights = jax.nn.softmax(-distances / self.beta)
        y_pred = jnp.sum(weights[:, None] * self.contents, axis=0)
        return y_pred


# Example usage
def main():
    # Hyperparameters from paper (Step 8 in thinking trace)
    input_dim = 10    # Example feature dimension
    output_dim = 3    # Example number of classes
    capacity = 1000   # Memory capacity
    beta = 1.0        # Softmin temperature
    lambda_rt = 0.9   # RT update rate
    lambda_update = 0.01  # Memory update rate

    # Initialize DSDM
    key = jax.random.PRNGKey(0)
    dsdm = DSDM(input_dim, output_dim, capacity, beta, lambda_rt, lambda_update)

    # Dummy data
    features = jax.random.normal(key, (100, input_dim))  # 100 samples
    labels = jax.random.randint(key, (100, output_dim), 0, 2)  # Binary labels

    # Train online (Step 6 in thinking trace)
    for x, y in zip(features, labels):
        dsdm = dsdm.train_step(x, y)

    # Test prediction
    test_x = jax.random.normal(key, (input_dim,))
    pred = dsdm.predict(test_x)
    print(f"Prediction: {pred}")


if __name__ == "__main__":
    main()
