import pickle
import numpy as np

class ModelInputGenerator:
    
    def __init__(self) -> None:
        with open('./ingredients.pkl', 'rb') as file:
            self.ingredients = pickle.load(file)
        self.base_dict = {element: 0 for element in self.ingredients if element not in ['product_name', '_id', 'nova_group']}

    def gen_input_vector(self, lista_strings):
        base_dict_copy = self.base_dict.copy()
        # Iterar sobre los strings en la lista
        for string in lista_strings:
            # Si el string está en el diccionario, actualiza su valor a 1
            if string in base_dict_copy:
                base_dict_copy[string] = base_dict_copy[string] + 1
                
        return np.expand_dims(np.array(list(base_dict_copy.values())), axis=0)

    