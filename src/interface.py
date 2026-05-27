import streamlit as st
import requests

st.set_page_config(page_title="Прогноз проката велосипедов", page_icon="🚲", layout="centered")

st.title("🚲 Прогнозирование спроса на велопрокат")
st.write("Укажите погодные условия и временные факторы для расчета ожидаемого количества аренд.")

FASTAPI_URL = "http://ml-pipeline:8000/predict"

# Создаем форму для ввода данных
with st.form("prediction_form"):
    st.subheader(" Временные параметры")
    col1, col2, col3 = st.columns(3)
    with col1:
        hour = st.slider("Час суток", 0, 23, 12)
    with col2:
        day_of_week = st.selectbox("День недели",
                                   options=[0, 1, 2, 3, 4, 5, 6],
                                   format_func=lambda x: ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"][x])
    with col3:
        month = st.selectbox("Месяц",
                             options=list(range(1, 13)),
                             format_func=lambda x:
                             ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"][
                                 x - 1])

    st.subheader(" Метеорологические условия")
    col4, col5, col6 = st.columns(3)
    with col4:
        temp = st.number_input("Температура (°C)", value=20.0, step=0.5)
    with col5:
        precip = st.number_input("Осадки (мм)", value=0.0, step=0.1, min_value=0.0)
    with col6:
        wind = st.number_input("Скорость ветра (км/ч)", value=10.0, step=0.5, min_value=0.0)

    st.subheader(" Авторегрессионный маркер")

    submit_button = st.form_submit_with_confidence = st.form_submit_button(label="Рассчитать спрос")

if submit_button:
    # Собираем JSON
    payload = {
        "temp": temp,
        "precip": precip,
        "wind": wind,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month
    }

    try:
        with st.spinner("Запрос к модели..."):
            response = requests.post(FASTAPI_URL, json=payload, timeout=5)

        if response.status_code == 200:
            result = response.json()
            predicted_cnt = result["prediction"]

            st.success("Прогноз успешно сформирован!")
            st.metric(label="Ожидаемое количество аренд в этот час:", value=f"{predicted_cnt} велосипедов")
        else:
            st.error(f" Ошибка сервера FastAPI: {response.status_code}. {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Не удалось соединиться с FastAPI. Проверьте, запущен ли контейнер бэкенда.")