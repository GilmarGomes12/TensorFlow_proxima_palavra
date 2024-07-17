import streamlit as st
import gdown
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def carregar_modelo():
    url = 'https://colab.research.google.com/drive/1d5gLEan4PgLO0ycCRGZlpn9bJLMlp0L1?usp=drive_link'
    gdown.download(url, 'modelo_vidente.h5', quiet=False)
    loaded_model = tf.keras.models.load_model('modelo_vidente.keras')
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
        
    return loaded_model, vectorizer


def predict_next_words(model, vectorizer, text_sequence, num_words=3):
    """Prevê as próximas palavras mais prováveis em uma sequência de texto.

    Args:
        model: O modelo Keras treinado para prever a próxima palavra.
        vectorizer: O objeto vectorizer usado para transformar o texto em sequências numéricas.
        text_sequence: A sequência de texto para a qual prever as próximas palavras.
        num_words (opcional): O número de palavras a serem previstas (padrão: 3).

    Returns:
        Uma lista das próximas palavras mais prováveis.
    """
    #tokenizar o texto de entrada
    token_text = np.squeeze(tokenized_text)
    
    #adicionar padding à esquerda
    padded_text = pad_sequences([token_text], maxlen=max_sequence_len - 1, padding='pre')
    
    #fazer a previsão
    predicted_probs = model.predict(padded_text, verbose=0)[0]
    
    #Obter os índices dos top_k com as maiores probabilidades
    top_k_indices = np.argsort(predicted_probs)[-top_k:][::-1]
    
    #converter ow tokens previstos de volta para palavras
    predicted_words = [vectorizer.get_vocabulary()[index] for index in top_k_indices]
    
    return predicted_words

def main():

    max_sequence_len = 50

    #carregar_modelo
    loaded_model, vectorizer = carregar_modelo()

    st.title('Previsão de Próximas Palavras')

    input_text = st.text_input('Digite uma sequência de texto:')
    
    if st.button("Prever"):
        if input_text:
            try:
                predicted_words = predict_next_words(loaded_model, vectorzer,input_text, max_sequence_len)
                st.info('Palavras mais prováveis')
                for word in predicted_words:
                    st.success(word)
            except:
                st.error('Erro na previsão {e}')        
        else:
            st.warning('Por favor, insira algum texto')                    

if __name__ == "__main__":
    main()
    