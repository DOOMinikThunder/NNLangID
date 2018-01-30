# NNLangID :earth_americas::speech_balloon:
Project for neural network-based tweet language identification.

## Getting Started
* Fetch tweet data from twitter via the `TweetRetriever.py` and place it into `data/input_data/original` (already done for the [Twitter blog post](https://blog.twitter.com/engineering/en_us/a/2015/evaluating-language-identification-performance.html) data this project is based on).
* Run `Main.py` for the main procedure. It reads in one of the two YAML settings files, which contain all user parameters.
	* Set `create_splitted_data_files = True` to split a file from `data/input_data/original` into separate training, validation and test set files. The data is then fetched from those files, preprocessed and transformed to be readily used by the subsequent embedding and RNN.
	* Set `train_embed = True` to train the embedding and get the embedding weights. The embedding is implemented as a Skip-Gram with Negative Sampling. While training, the best embedding model checkpoint and extracted embedding weights are automatically saved to file (`/data/save/embed_model_checkpoint.pth` and `/data/save/embed_weights.txt`).
	* Set `train_rnn = True` to use the embedding weights to embed the characters of a tweet and feed them into the RNN, which is implemented as a (mono- or bidirectional) GRU. While training, the best RNN model checkpoint is automatically saved to file (`/data/save/rnn_model_checkpoint.pth`).
	* Set `eval_test_set = True` to evaluate a trained RNN model checkpoint on the test set, to get further metrics on the performance, which are then stored back to the checkpoint file.
	* Set `run_terminal = True` to run the terminal for interactive evaluation of a trained RNN model checkpoint with arbitrary input text or live tweets fetched directly from Twitter.
	* Set `print_embed_testing = True` to print the embedding test to the console.
	* Set `print_model_checkpoint_embed_weights` and `print_rnn_model_checkpoint` or `print_embed_model_checkpoint` to the respective file paths to print the stored model checkpoint data to the console.

### Prerequisites
* Python v2.7
* PyTorch v0.2.0_4
* CUDA is used if available.

## Authors
Project developed by Alexander Heilig, Dominik Sauter and Tabea Kiupel in the context of the Neural Networks practical course at the Karlsruhe Institute of Technology (KIT), Germany.

## License
Licensed unter the MIT license (see [LICENSE](LICENSE) file for more details).
- Enjoy!