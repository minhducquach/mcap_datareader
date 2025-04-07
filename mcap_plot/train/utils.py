import itertools
import torch
from torch import nn

def alternate_training(loaders, model, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        loader_iters = {file: iter(loader) for file, loader in loaders.items()}
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        loss_acc = 0
        num_batch = 0
        
        # Use zip_longest to handle different dataset sizes
        for batches in itertools.zip_longest(*loader_iters.values(), fillvalue=None):
            for file, batch in zip(loaders.keys(), batches):
                if batch is not None:
                    optimizer.zero_grad()
                    inputs, labels = batch
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels.squeeze())
                    loss.backward()
                    optimizer.step()
                    loss_acc += loss.item()
                    num_batch += 1
        print(f"Epoch Loss: {loss_acc / num_batch}")

    return model
        
def evaluate_model(model, test_loaders):
    model.eval()

    criterion = nn.L1Loss()  # Adjust if using another loss function

    results = {}
    
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for file, loader in test_loaders.items():
            total_loss = 0.0
            num_batches = 0
            
            for inputs, labels in loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            results[file] = avg_loss
            print(f"Test set [{file}] - Average Loss: {avg_loss:.4f}")

    return results  # Returns a dictionary with loss values per dataset
