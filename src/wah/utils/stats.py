import torch


class MeanSTD:
    def __init__(self):
        self.total_samples = 0
        self.cumulative_mean = 0
        self.cumulative_m2 = 0

    def update(self, batch):
        # Get the batch size and the number of elements in each sample
        batch_size = batch.size(0)
        num_elements_per_sample = batch.numel() / batch_size

        # Flatten the batch to treat all elements equally
        batch = batch.view(batch_size, -1)

        # Compute mean and variance across all elements in the batch
        batch_mean = torch.mean(batch)
        batch_var = torch.var(batch, unbiased=False)

        self.total_samples += batch_size * num_elements_per_sample

        delta = batch_mean - self.cumulative_mean
        self.cumulative_mean += delta * (
            batch_size * num_elements_per_sample / self.total_samples
        )

        self.cumulative_m2 += (
            batch_size
            * num_elements_per_sample
            * (
                batch_var
                + delta**2
                * (self.total_samples - batch_size * num_elements_per_sample)
                / self.total_samples
            )
        )

    def finalize(self):
        if self.total_samples == 0:
            raise ValueError("No data has been added.")

        overall_mean = self.cumulative_mean
        overall_var = self.cumulative_m2 / self.total_samples
        overall_std = torch.sqrt(overall_var)

        return overall_mean.item(), overall_std.item()


# Example usage:
# Assuming `data_loader` is a PyTorch DataLoader that yields mini-batches of data
stats = IncrementalStats()

for batch in data_loader:
    batch_data = batch[0]  # DataLoader returns a tuple, we need the first element
    stats.update(batch_data)

mean, std = stats.finalize()
print("Overall Mean:", mean)
print("Overall Standard Deviation:", std)
