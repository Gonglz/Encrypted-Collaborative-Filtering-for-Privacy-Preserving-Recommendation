
# Encrypted Collaborative Filtering for Privacy-Preserving Recommendation 

- This project is made for final-project of SIT 24F CPE602-Applied Discrete Math.
- This project implements a privacy-preserving collaborative filtering recommendation system using encryption techniques, secret sharing, and zero-knowledge proof validation. The core objective is to ensure the security and privacy of user and item data while maintaining high recommendation accuracy.

## Project Overview
- **Framework:** PyTorch
- **Data Source:** MovieLens dataset (ml-latest-small)  
  - https://grouplens.org/datasets/movielens/
- **Core Techniques:**
  - AES encryption for privacy-preserving computation
  - Zero-Knowledge Proof (ZKP) for data integrity
  - Enhanced matrix factorization for recommendation

## Features
1. **Privacy-Preserving Computation:**
   - User and item IDs are encrypted using AES encryption.
   - Encrypted data ensures confidentiality during computations.

2. **Zero-Knowledge Proof:**
   - Validates the integrity of secret-shared data.

3. **Enhanced Matrix Factorization:**
   - Incorporates user and item embeddings with a deep neural network for accurate predictions.

4. **Performance Metrics:**
   - Tracks training loss, testing loss, RMSE, and MAE during model training.
   - Visualizes performance metrics over training epochs.

## Dependencies
Ensure the following dependencies are installed:
```plaintext
Pandas==1.3.5
Scikit-learn==1.0.2
PyTorch==1.7.1
Matplotlib==3.5.3
```

## Sample Results
```plaintext
Epoch 1: Train Loss=1.3476, Test Loss=0.1851, RMSE=0.4302, MAE=0.3394
...
Epoch 30: Train Loss=0.0432, Test Loss=0.0356, RMSE=0.1886, MAE=0.1471

Sample Predictions:
Actual: [0.6 0.7 0.8 0.8 0.7], Predicted: [0.5351176  0.7222369  0.8366588  0.7901543  0.64153785]
```

## Challenges and Decisions
During development, we considered using frameworks like **PySyft**, **CrypTen**, and **TF-Encrypted** for privacy-preserving computations. However, we encountered the following challenges:

1. **Hardware and Environment Requirements:**
   - These frameworks require specific hardware configurations and software environments, such as particular versions of Python, PyTorch, or TensorFlow. Incompatibilities with our current setup made their usage impractical.

2. **Learning Curve and Performance Limitations:**
   - **PySyft** and **CrypTen** abstract the complexities of multi-party computation and encryption but involve a steep learning curve and debugging challenges.
   - **TF-Encrypted** has slower updates and limited efficiency, making it unsuitable for the performance demands of this project.

### Why Manual Implementation?
Instead of relying on these frameworks, we manually implemented AES encryption and zero-knowledge proof mechanisms. While not a fully secure encryption system, this approach:
- Simplified hardware and environment dependencies, ensuring ease of execution.
- Provided deeper insights into encryption principles and multi-party computation.
- Enhanced flexibility for experimenting with different configurations and scenarios.

## File Structure
- `main.py`
- `Dataset/`
- `README.md`
- `LICENSE`

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For any questions or contributions, feel free to reach out!

