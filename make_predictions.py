import joblib
from train_model import prepare_data
import pandas as pd


def main():
    # Cargar el modelo entrenado
    model_loaded = joblib.load("./trained_model/logistics_model.pkl")

    # Cargar los nuevos datos
    new_data = pd.read_excel("./data_to_predict/Human_Resources.xlsx")

    # Preprocesar los nuevos datos
    Processed_data = prepare_data(new_data)

    # Hacer predicciones sobre los nuevos datos
    predictions = model_loaded.predict(Processed_data)

    # Imprimir las predicciones
    df_y = pd.DataFrame(predictions, columns=['Prediction'])
    df_y.to_excel("./predictions/prediction.xlsx", index=False)


if __name__ == "__main__":
    main()
