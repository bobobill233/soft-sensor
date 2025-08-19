import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time  # 导入 time 模块
from dataloader import *

def train_model(model, train_loader, valid_loader, optimizer, criterion, scheduler, num_epochs, device, save_path,
                 scaler, target, log_file):

    best_val_loss = float('inf')
    best_val_accuracy = 0.0  # Initialize the best accuracy
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    train_accuracies = []  # Track training accuracies
    val_accuracies = []  # Track validation accuracies

    with open(log_file, 'w') as f:  # Open the log_7g file for writing
        for epoch in range(num_epochs):
            # 记录epoch开始时间
            epoch_start_time = time.time()

            model.train()
            running_loss = 0.0
            total_train_mae = 0.0
            total_train = 0
            train_batch_mae_list = []
            all_train_labels = []
            all_train_predictions = []

            for i, (gasf_tensor, mtf_tensor, labels) in enumerate(train_loader):
                gasf_tensor, mtf_tensor, labels = gasf_tensor.to(device), mtf_tensor.to(device), labels.to(device)

                optimizer.zero_grad()

                # 如果使用了 Swin 模型，则进行特征提取
                outputs = model(gasf_tensor, mtf_tensor)

                # Forward pass
                loss = criterion(outputs, labels.view(-1, 1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * gasf_tensor.size(0)

                # Calculate MAE for this batch
                preds = outputs.view(-1)
                batch_mae = torch.mean(torch.abs(preds - labels))
                total_train_mae += batch_mae.item() * gasf_tensor.size(0)
                total_train += labels.size(0)

                # Print batch loss and MAE
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, MAE: {batch_mae.item():.4f}')

                # Store MAE for the batch
                train_batch_mae_list.append(batch_mae.item())

                # Collect labels and predictions for accuracy calculation
                all_train_labels.extend(labels.cpu().numpy())
                all_train_predictions.extend(preds.detach().cpu().numpy())

            train_loss = running_loss / len(train_loader.dataset)
            train_mae = total_train_mae / total_train
            train_losses.append(train_loss)
            train_maes.append(train_mae)

            # Calculate and mark anomalous batches for training
            train_mae_array = np.array(train_batch_mae_list)
            train_mean_mae = np.mean(train_mae_array)
            train_std_mae = np.std(train_mae_array)
            train_threshold = train_mean_mae + 3 * train_std_mae
            train_anomalous_batches = [i for i, mae in enumerate(train_batch_mae_list) if mae > train_threshold]

            # Calculate training accuracy (within threshold of 10)
            all_train_labels = np.array(all_train_labels).reshape(-1, 1)
            all_train_predictions = np.array(all_train_predictions).reshape(-1, 1)

            # Inverse transform the normalized values
            all_train_labels = scaler.inverse_transform(all_train_labels)
            all_train_predictions = scaler.inverse_transform(all_train_predictions)

            # Calculate accuracy (within threshold of 10)
            within_train_threshold = np.abs(all_train_predictions - all_train_labels) <= target
            train_accuracy = np.mean(within_train_threshold)
            train_accuracies.append(train_accuracy)

            print(f'Training std: {train_std_mae:.4f}, threshold: {train_threshold:.4f}')
            print(f'Anomalous training batches in epoch {epoch + 1}: {train_anomalous_batches}')

            # Validate model
            model.eval()
            val_loss = 0.0
            total_val_mae = 0.0
            total_val = 0
            val_batch_mae_list = []
            all_labels = []
            all_predictions = []
            with torch.no_grad():
                for i, (gasf_tensor, mtf_tensor, labels) in enumerate(valid_loader):
                    gasf_tensor, mtf_tensor, labels = gasf_tensor.to(device), mtf_tensor.to(device), labels.to(device)

                    outputs = model(gasf_tensor, mtf_tensor)
                    loss = criterion(outputs, labels.view(-1, 1))
                    val_loss += loss.item() * gasf_tensor.size(0)

                    # Calculate MAE for this batch
                    preds = outputs.view(-1)
                    batch_mae = torch.mean(torch.abs(preds - labels))
                    total_val_mae += batch_mae.item() * gasf_tensor.size(0)
                    total_val += labels.size(0)

                    # Collect labels and predictions
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(preds.cpu().numpy())

                    # Print batch validation loss and MAE
                    print(
                        f'Validation Batch [{i + 1}/{len(valid_loader)}], Loss: {loss.item():.4f}, MAE: {batch_mae.item():.4f}')

                    # Store MAE for the batch
                    val_batch_mae_list.append(batch_mae.item())

            val_loss /= len(valid_loader.dataset)
            val_mae = total_val_mae / total_val
            val_losses.append(val_loss)
            val_maes.append(val_mae)

            # Calculate and mark anomalous batches for validation
            val_mae_array = np.array(val_batch_mae_list)
            val_mean_mae = np.mean(val_mae_array)
            val_std_mae = np.std(val_mae_array)
            val_threshold = val_mean_mae + 2.5 * val_std_mae
            val_anomalous_batches = [i for i, mae in enumerate(val_batch_mae_list) if mae > val_threshold]

            # Calculate accuracy (within threshold of 10)
            all_labels = np.array(all_labels).reshape(-1, 1)
            all_predictions = np.array(all_predictions).reshape(-1, 1)

            all_labels = scaler.inverse_transform(all_labels)
            all_predictions = scaler.inverse_transform(all_predictions)

            within_threshold = np.abs(all_predictions - all_labels) <= target
            accuracy = np.mean(within_threshold)
            val_accuracies.append(accuracy)

            print(
                f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train MAE: {train_mae:.4f}, Validation MAE: {val_mae:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {accuracy:.4f}')
            f.write(
                f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train MAE: {train_mae:.4f}, Validation MAE: {val_mae:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {accuracy:.4f}.\n')
            print(f'Validation std: {val_std_mae:.4f}, threshold: {val_threshold:.4f}')
            print(f'Anomalous validation batches in epoch {epoch + 1}: {val_anomalous_batches}')

            if accuracy > best_val_accuracy:
                best_val_accuracy = accuracy
                torch.save(model.state_dict(), save_path)

            # Plot predictions vs true values and their differences
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(all_labels, label='True Values', color='blue', marker='o')
            plt.plot(all_predictions, label='Predictions', linestyle='--', color='red', marker='x')
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.legend()
            plt.title(f'Epoch {epoch + 1}: Predictions vs True Values')

            differences = np.abs(np.array(all_predictions) - np.array(all_labels))
            plt.subplot(2, 1, 2)
            plt.plot(differences, label='Difference', color='green', marker='.')
            plt.xlabel('Sample Index')
            plt.ylabel('Difference')
            plt.legend()
            plt.title(f'Epoch {epoch + 1}: Prediction Differences')
            plt.text(0.5, 0.9, f'Validation MAE: {val_mae:.4f}', horizontalalignment='center',
                     verticalalignment='center', transform=plt.gca().transAxes)

            plt.tight_layout()
            plt.show()

            # 记录epoch结束时间并计算耗时
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f'Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds.')
            f.write(f'Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds.\n')

            # Step the scheduler at the end of each epoch
            scheduler.step()

        f.write("Finished Training.\n")
    print('Finished Training')

    # Plotting the loss and accuracy curves
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    axs[0].plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].legend()

    axs[1].plot(range(1, num_epochs + 1), train_maes, label='Train MAE')
    axs[1].plot(range(1, num_epochs + 1), val_maes, label='Validation MAE')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('MAE')
    axs[1].set_title('Training and Validation MAE')
    axs[1].legend()

    axs[2].plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Accuracy')
    axs[2].set_title('Validation Accuracy')
    axs[2].legend()

    plt.tight_layout()
    plt.show()
