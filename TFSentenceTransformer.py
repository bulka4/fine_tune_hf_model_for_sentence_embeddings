"""
This is a class for creating a Tensorflow model using a Hugging Face model.
"""

from transformers import TFAutoModel
import tensorflow as tf

class TFSentenceTransformer(tf.keras.layers.Layer):
    def __init__(self, model_name_or_path, **kwargs):
        super(TFSentenceTransformer, self).__init__()
        # loads transformers model from Hugging Face using its name / URL
        try:
            self.model = TFAutoModel.from_pretrained(model_name_or_path, **kwargs)
            # self.model = TFAutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        except:
            self.model = TFAutoModel.from_pretrained(model_name_or_path, from_pt=True, **kwargs)
            # self.model = TFAutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
            
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            
    def call(self, inputs: list[str], normalize=True):
        "This function creates embeddings for a given list with input strings."
        # tokenize input string
        # inputs = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='tf')
        # runs model on inputs
        model_output = self.model(inputs)
        
        # Perform pooling. In this case, mean pooling.
        embeddings = self.mean_pooling(model_output, inputs["attention_mask"])
        # normalizes the embeddings if wanted
        if normalize:
            embeddings = self.normalize(embeddings)
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = tf.cast(
            tf.broadcast_to(tf.expand_dims(attention_mask, -1), tf.shape(token_embeddings)),
            tf.float32
        )
        return tf.math.reduce_sum(token_embeddings * input_mask_expanded, axis=1) / tf.clip_by_value(tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max)

    def normalize(self, embeddings):
        embeddings, _ = tf.linalg.normalize(embeddings, 2, axis=1)
        return embeddings
