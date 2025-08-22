import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time  # 导入 time 模块

# 计算模型参数存储的内存大小
def model_memory_size(model):
    total_memory = 0
    for param in model.parameters():
        total_memory += param.numel() * param.element_size()
    total_memory_MB = total_memory / (1024 ** 2)  # 转换为MB
    return total_memory_MB

# 计算评估指标
def calculate_metrics(true_values, predictions, naive_forecast):
    # MAE
    mae = np.mean(np.abs(true_values - predictions))

    # RMSE
    rmse = np.sqrt(np.mean((true_values - predictions) ** 2))

    # SMAPE
    smape = 100 * np.mean(np.abs(true_values - predictions) / ((np.abs(true_values) + np.abs(predictions)) / 2))

    # MAPE
    mape = 100 * np.mean(np.abs(true_values - predictions) / np.abs(true_values))

    # MASE (Assume naive forecast is available for scale)
    naive_mae = np.mean(np.abs(true_values - naive_forecast))
    mase = mae / naive_mae if naive_mae != 0 else float('inf')  # Avoid division by zero

    # OWA (based on SMAPE and MASE)
    naive_smape = 100 * np.mean(
        np.abs(true_values - naive_forecast) / ((np.abs(true_values) + np.abs(naive_forecast)) / 2))
    owa = (smape / naive_smape + mase) / 2

    return mae, rmse, smape, mape, mase, owa

def test_model(model, best_path, test_loader, device, scaler, target, log_file, I, naive_forecast, csv_file):
    # 记录开始时间
    start_time = time.time()

    # Load the trained model weights
    model.load_state_dict(torch.load(best_path))
    model.eval()  # Set the model to evaluation mode

    # 计算并记录模型参数的内存大小
    memory_size = model_memory_size(model)
    print(f'Model memory size: {memory_size:.2f} MB')

    with open(log_file, 'a') as log:
        log.write(f'Model memory size: {memory_size:.2f} MB\n')  # 写入日志文件

    all_predictions = []
    all_labels = []

    total_test_mae = 0.0
    total_test = 0
    test_batch_mae_list = []

    with open(log_file, 'a') as log:  # Open the log file for writing
        with torch.no_grad():  # Disable gradient calculation
            for i, (gasf_tensor, mtf_tensor, labels) in enumerate(test_loader):
                gasf_tensor, mtf_tensor, labels = gasf_tensor.to(device), mtf_tensor.to(device), labels.to(device)

                # Forward pass: Get model predictions
                outputs = model(gasf_tensor, mtf_tensor)
                preds = outputs.view(-1)

                # Calculate Mean Absolute Error (MAE) for this batch
                batch_mae = torch.mean(torch.abs(preds - labels))
                total_test_mae += batch_mae.item() * gasf_tensor.size(0)
                total_test += labels.size(0)

                # Store predictions and labels for later analysis
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                test_batch_mae_list.append(batch_mae.item())

                # Print and log the batch MAE
                log_message = f'Test Batch [{i + 1}/{len(test_loader)}], MAE: {batch_mae.item():.4f}'
                print(log_message)
                log.write(log_message + '\n')  # Write the message to the log file

        # Inverse transform the normalized values
        all_labels = np.array(all_labels).reshape(-1, 1)
        all_predictions = np.array(all_predictions).reshape(-1, 1)

        all_labels = scaler.inverse_transform(all_labels)
        all_predictions = scaler.inverse_transform(all_predictions)

        # Create a DataFrame and add the Iteration column
        results_df = pd.DataFrame({
            'Iteration': i,  # Add the iteration number
            'True Values': all_labels.flatten(),  # Flatten arrays
            'Predictions': all_predictions.flatten()
        })

        # Save to CSV (append mode)
        results_df.to_csv(csv_file, mode='a', header=not pd.io.common.file_exists(csv_file), index=False)
        print(f"Iteration {I} predictions and true values saved to {csv_file}")

        # Calculate accuracy (within threshold of 10)
        within_threshold = np.abs(all_predictions - all_labels) <= target
        test_accuracy = np.mean(within_threshold)
        log_message = f'Test Accuracy (within threshold of {target}): {test_accuracy:.4f}'
        print(log_message)
        log.write(log_message + '\n')

        # 计算评估指标
        mae, rmse, smape, mape, mase, owa = calculate_metrics(all_labels, all_predictions, naive_forecast)

        # 记录评估指标
        log_message = (f'Overall Test MAE: {mae:.4f}\n'
                       f'Overall Test RMSE: {rmse:.4f}\n'
                       f'Overall Test SMAPE: {smape:.4f}\n'
                       f'Overall Test MAPE: {mape:.4f}\n'
                       f'Overall Test MASE: {mase:.4f}\n'
                       f'Overall Test OWA: {owa:.4f}')
        print(log_message)
        log.write(log_message + '\n')

        # 记录结束时间并计算总耗时
        end_time = time.time()
        elapsed_time = end_time - start_time
        log_message = f'Total test time: {elapsed_time:.2f} seconds'
        print(log_message)
        log.write(log_message + '\n')

        # Plot predictions vs true values
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(all_labels, label='True Values', color='blue', marker='o')
        plt.plot(all_predictions, label='Predictions', linestyle='--', color='red', marker='x')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.title(f'Test Predictions vs True Values')

        # Plot the differences between predictions and true values
        differences = np.abs(all_predictions - all_labels)
        plt.subplot(2, 1, 2)
        plt.plot(differences, label='Difference', color='green', marker='.')
        plt.xlabel('Sample Index')
        plt.ylabel('Difference')
        plt.legend()
        plt.title(f'Test Prediction Differences')
        plt.tight_layout()
        plt.show()

# Example usage:
# test_model(model, 'best_model.pth', test_loader, device, scaler, target, log_file="test_log.txt", naive_forecast=naive_forecast)
