# Fake_News_Detection
Fake news detection using **ML, NLP and Blockchain**.

**Project Details:**
In this project, we propose a Fake News Detection System that leverages deep learning models and blockchain technology to ensure both accurate classification and secure authentication. The system is designed to automatically classify news articles as real or fake using natural language processing techniques and deep learning, while securing the user authentication process with blockchain technology.
**Key Components:**
1.	**Data Preprocessing with NLP:** The preprocessing phase involves cleaning and transforming the raw news articles into a suitable format for feature extraction and model training. This includes:
o	**Tokenization:** Splitting the text into individual words or tokens.
o **Stopword Removal:** Filtering out common words that do not contribute significantly to the meaning of the text (e.g., 'the', 'is', 'and').
o	**Stemming and Lemmatization:** Reducing words to their root forms, helping to standardize the dataset and reduce vocabulary size.
o	**Vectorization:** Utilizing FastText embeddings to represent words in a dense, semantic vector space, capturing both word meaning and context.
2.	**Feature Extraction with FastText:** FastText is employed for feature extraction, as it provides rich word embeddings that consider subword information, improving the system's ability to generalize across different languages and misspelled words. FastText captures the relationships between words and their surrounding context, generating embeddings that serve as input to the CNN classifier.
3.	**Fake News Classification with CNN:** The extracted FastText embeddings are fed into a Convolutional Neural Network (CNN), which is responsible for learning spatial patterns in the text data. The CNN architecture consists of:
o	**Convolutional Layers:** These layers detect relevant features and patterns in the text embeddings.
o	**Pooling Layers:** These layers reduce the dimensionality of the features, focusing on the most important patterns.
o	**Fully Connected Layers:** These layers classify the news articles as 'real' or 'fake' based on the extracted features.
4.	**Blockchain-based JWT Authentication:** To ensure data integrity, transparency, and security in the authentication process, blockchain technology is integrated with JWT (JSON Web Token)-based authentication using MetaMask. This approach leverages blockchain to store and validate authentication tokens in a decentralized, tamper-proof ledger, providing several benefits:
o	**Decentralized Authentication:** MetaMask interacts with the blockchain to verify user credentials and securely store authentication tokens.
o	**Tamper-proof Records:** The use of blockchain ensures that the authentication tokens cannot be altered, providing an additional layer of security.
o	**Transparency:** All authentication requests and token validations are recorded on the blockchain, ensuring an audit trail for verification purposes.


**Working ScreenShots:**

**signin:**
![Screenshot 2024-10-18 235236](https://github.com/user-attachments/assets/dbee23e7-ba3d-4865-93ee-3591514c7cd3)


**Metamask Page**
![Screenshot 2024-10-18 233355](https://github.com/user-attachments/assets/75660367-c15a-4240-81cc-a44750dec361)

**Prediction Interface and example predictions**
![Screenshot 2024-10-18 233457](https://github.com/user-attachments/assets/9741abcb-b917-4d43-98bf-338073e25655)
![Screenshot 2024-10-18 233553](https://github.com/user-attachments/assets/42753f7e-9b63-4dfa-b7a4-4d083291e40b)
![Screenshot 2024-10-18 233632](https://github.com/user-attachments/assets/f8f80902-aabd-4473-80be-c2f38c85662f)
