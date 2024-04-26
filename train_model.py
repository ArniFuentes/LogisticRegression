from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def prepare_data(employee_df):
    # Si se va a entrenar debería venir Attrition
    if "Attrition" in employee_df.columns:
        employee_df['Attrition'] = employee_df['Attrition'].apply(
            lambda x: 1 if x == 'Yes' else 0)
        employee_df['OverTime'] = employee_df['OverTime'].apply(
            lambda x: 1 if x == 'Yes' else 0)

        # Filtrar empleados que hayan dejado la empresa y submuestrear empleados que no
        employee_df_yes = employee_df[employee_df["Attrition"] == 1]
        employee_df_no = employee_df[employee_df["Attrition"] == 0]

        # Ver la desproporción entre los Yes y No
        count = employee_df["Attrition"].value_counts()
        sample = count[1]
        employee_df_no_reduced = employee_df_no.sample(
            sample, random_state=103)
        df_concat = pd.concat(
            [employee_df_yes, employee_df_no_reduced], axis=0)
        employee_df = df_concat

    else:
        employee_df['OverTime'] = employee_df['OverTime'].apply(
            lambda x: 1 if x == 'Yes' else 0)

    # Eliminar columnas irrelevantes
    employee_df.drop(["EmployeeCount", "StandardHours", "Over18",
                      "EmployeeNumber"], axis=1, inplace=True)

    # Variables categóricas de interés
    X_cat = employee_df[['BusinessTravel', 'Department',
                         'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]

    # Obtener las columnas que tienen en común employee_df y X_cat
    columnas_comunes = employee_df.columns.intersection(X_cat.columns)
    # Eliminar las columnas presentes en X_cat del DataFrame original para obtener todas las numéricas
    X_numerical = employee_df.drop(columns=columnas_comunes)

    if "Attrition" in employee_df.columns:
        # quitar también la columna target, 'Attrition'
        X_numerical.drop(columns="Attrition", inplace=True)
        
    X_numerical.reset_index(inplace=True)

    onehotencoder = OneHotEncoder()

    # pasar estas variables de categóricas a numéricas
    X_cat = onehotencoder.fit_transform(X_cat).toarray()
    X_cat = pd.DataFrame(X_cat)

    X_all = pd.concat([X_cat, X_numerical], axis=1).values

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X_all)

    if "Attrition" in employee_df.columns:
        y = employee_df['Attrition']
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25)

        return X_train, X_test, y_train, y_test

    return X_scaled


def evaluate_model(modelo, X_test, y_test):
    """Evalúa el rendimiento del modelo utilizando el conjunto de prueba."""
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, cm, report


def main():
    data = pd.read_excel("./data/Human_Resources.xlsx")
    X_train, X_test, y_train, y_test = prepare_data(data)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluar el modelo
    accuracy, cm, report = evaluate_model(model, X_test, y_test)
    print("Accuracy: {} %".format(100 * accuracy))
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    # Guarda el modelo
    joblib.dump(model, "./trained_model/logistics_model.pkl")


if __name__ == "__main__":
    main()
